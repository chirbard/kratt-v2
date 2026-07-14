// The daemon's night cycle: a dream window during quiet hours, not a general
// autonomous mode. Inside the window, after an idle poll, the daemon may run
// one scheduler cycle inline on the poll thread — a message arriving mid-dream
// waits in Telegram's queue and is answered when the pass finishes. That is
// acceptable at 3-6am by construction; that is what the window is for.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::api::{EmbeddingClient, LlmClient};
use crate::memory::scheduler::{self, SchedulerAction};
use crate::memory::{self, DreamEvent, ExploreEvent, IngestEvent};
use crate::runlog;
use crate::timeutil;
use crate::web::SearchClient;

// replay-tunable.
const NIGHT_ACTION_COOLDOWN: Duration = Duration::from_secs(20 * 60);
// replay-tunable.
const MAX_NIGHT_ACTIONS: usize = 3;
const DEFAULT_DREAM_HOURS: &str = "3-6";

// Hour range [start, end) in UTC (timeutil::current_hour is epoch-derived,
// like every date in the app), end exclusive. A wrapping range like `23-5`
// means 23,0,1,2,3,4 are inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DreamWindow {
    start: u32,
    end: u32,
}

impl DreamWindow {
    fn parse(raw: &str) -> Result<Self, String> {
        let (start_raw, end_raw) = match raw.split_once('-') {
            Some(parts) => parts,
            None => return Err(format!("KRATT_DREAM_HOURS must be 'start-end', got '{raw}'")),
        };
        let start = parse_hour(start_raw.trim())?;
        let end = parse_hour(end_raw.trim())?;
        if start == end {
            return Err(format!(
                "KRATT_DREAM_HOURS start and end must differ, got '{raw}'"
            ));
        }
        Ok(Self { start, end })
    }

    fn contains(&self, hour: u32) -> bool {
        if self.start < self.end {
            hour >= self.start && hour < self.end
        } else {
            hour >= self.start || hour < self.end
        }
    }
}

fn parse_hour(raw: &str) -> Result<u32, String> {
    let hour: u32 = match raw.parse() {
        Ok(value) => value,
        Err(_) => return Err(format!("KRATT_DREAM_HOURS hour '{raw}' is not a number")),
    };
    if hour > 23 {
        return Err(format!("KRATT_DREAM_HOURS hour '{raw}' must be 0-23"));
    }
    Ok(hour)
}

// Pure window/cooldown/cap gate: no clients, no IO — tests inject the hour,
// the day, and the current instant.
struct NightGate {
    window: DreamWindow,
    cooldown: Duration,
    max_actions: usize,
    last_action: Option<Instant>,
    count: usize,
    night_day: Option<String>,
    // Invalidation flag instead of periodic re-evaluation: the daemon is the
    // only writer, so scheduler inputs can only change when it handles a turn,
    // runs an action itself, or the day rolls over — all of which set this.
    // A Rest with a clean flag would be provably identical, so it is skipped.
    // If a second write path is ever added, its author must set the flag (or
    // revert to periodic checks) — correctness depends on single-writer.
    dirty: bool,
    dreamed_tonight: bool,
    studied_inbox_tonight: bool,
    last_curiosity_had_yield: Option<bool>,
}

impl NightGate {
    fn new(window: DreamWindow, cooldown: Duration, max_actions: usize) -> Self {
        Self {
            window,
            cooldown,
            max_actions,
            last_action: None,
            count: 0,
            night_day: None,
            dirty: true,
            dreamed_tonight: false,
            studied_inbox_tonight: false,
            last_curiosity_had_yield: None,
        }
    }

    // May a scheduler evaluation run right now? Re-entering the window on a
    // new day resets the per-night action count and the nightly-dream flag,
    // and marks state dirty (the day-derived inputs just changed).
    fn should_evaluate(&mut self, hour: u32, today: &str, now: Instant) -> bool {
        let inside = self.window.contains(hour);
        if inside && self.night_day.as_deref() != Some(today) {
            self.night_day = Some(today.to_string());
            self.count = 0;
            self.dreamed_tonight = false;
            self.studied_inbox_tonight = false;
            self.last_curiosity_had_yield = None;
            self.dirty = true;
        }
        inside && self.dirty && self.cooldown_elapsed(now) && self.count < self.max_actions
    }

    fn cooldown_elapsed(&self, now: Instant) -> bool {
        match self.last_action {
            Some(last) => now.duration_since(last) >= self.cooldown,
            None => true,
        }
    }

    // A dream/explore counts against both the cooldown and the per-night cap.
    fn record_action(&mut self, now: Instant) {
        self.count += 1;
        self.last_action = Some(now);
        self.last_curiosity_had_yield = None;
    }

    fn record_curiosity(&mut self, now: Instant, had_yield: bool) {
        self.count += 1;
        self.last_action = Some(now);
        self.last_curiosity_had_yield = Some(had_yield);
        self.dirty = had_yield;
    }

    // A skipped action (ingest, non-headless explore) consumes the cooldown so
    // the daemon does not re-decide it every idle tick, but not the cap.
    fn record_skipped(&mut self, now: Instant) {
        self.last_action = Some(now);
    }

    // Rest with unchanged inputs would repeat forever — stop evaluating until
    // something actually changes (turn handled, action ran, day rolled over).
    fn record_rest(&mut self) {
        self.dirty = false;
    }
}

pub struct NightCycle {
    gate: NightGate,
    memory_llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
}

impl NightCycle {
    pub fn from_env(
        memory_llm: Arc<dyn LlmClient>,
        embed: Arc<dyn EmbeddingClient>,
        search: Option<Arc<dyn SearchClient>>,
    ) -> Result<Self, Box<dyn Error>> {
        let raw = std::env::var("KRATT_DREAM_HOURS").unwrap_or_else(|_| DEFAULT_DREAM_HOURS.into());
        let window = DreamWindow::parse(&raw)?;
        runlog::log("tg", &format!("dream window {}-{}", window.start, window.end));
        let gate = NightGate::new(window, NIGHT_ACTION_COOLDOWN, MAX_NIGHT_ACTIONS);
        Ok(Self {
            gate,
            memory_llm,
            embed,
            search,
        })
    }

    // Called after each idle poll. Runs at most one scheduler cycle inline.
    pub fn maybe_run(&mut self, shutdown: &AtomicBool) {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        let hour = timeutil::current_hour();
        let today = timeutil::today_iso();
        let now = Instant::now();
        if !self.gate.should_evaluate(hour, &today, now) {
            return;
        }
        if self.try_guaranteed_dream(now) {
            return;
        }
        self.run_scheduler_cycle(now);
    }

    // A turn was handled (or state otherwise changed): scheduler inputs may
    // differ now, so the next idle poll inside the window re-evaluates.
    pub fn mark_dirty(&mut self) {
        self.gate.dirty = true;
        self.gate.last_curiosity_had_yield = None;
    }

    // The nightly-dream guarantee: a brain dreams every night that anything
    // happened, regardless of debt thresholds. Once per night, if any
    // experience postdates the last dream, dream — bypassing decide().
    fn try_guaranteed_dream(&mut self, now: Instant) -> bool {
        if self.gate.dreamed_tonight {
            return false;
        }
        if !scheduler::has_unconsolidated_experience() {
            return false;
        }
        let reason = "nightly guarantee: unconsolidated experience";
        scheduler::log_scheduler_decision(&SchedulerAction::Dream {
            reason: reason.to_string(),
        });
        runlog::log("tg", &format!("night action: dream ({reason})"));
        self.run_dream();
        self.gate.dreamed_tonight = true;
        self.gate.record_action(now);
        true
    }

    fn run_scheduler_cycle(&mut self, now: Instant) {
        let mut inputs = scheduler::gather_inputs();
        if self.gate.studied_inbox_tonight {
            inputs.queued_inbox = 0;
        }
        inputs.last_curiosity_had_yield = self.gate.last_curiosity_had_yield;
        let action = scheduler::decide(&inputs);
        scheduler::log_scheduler_decision(&action);
        if !matches!(action, SchedulerAction::Rest { .. }) {
            scheduler::log_drift_info(&inputs.actor_positions);
        }
        self.dispatch(action, now);
    }

    fn dispatch(&mut self, action: SchedulerAction, now: Instant) {
        match action {
            SchedulerAction::Dream { reason } => {
                runlog::log("tg", &format!("night action: dream ({reason})"));
                self.run_dream();
                self.gate.dreamed_tonight = true;
                self.gate.record_action(now);
            }
            SchedulerAction::Explore { reason } => {
                runlog::log("tg", &format!("night action: explore ({reason})"));
                self.run_explore();
                self.gate.record_action(now);
            }
            SchedulerAction::Curiosity { reason } => {
                runlog::log("tg", &format!("night action: curiosity ({reason})"));
                let had_yield = self.run_explore();
                self.gate.record_curiosity(now, had_yield);
            }
            SchedulerAction::InboxStudy { reason } => {
                runlog::log("tg", &format!("night action: inbox study ({reason})"));
                self.run_inbox_study();
                self.gate.studied_inbox_tonight = true;
                self.gate.record_action(now);
            }
            SchedulerAction::Ingest { path, reason } => {
                runlog::log(
                    "tg",
                    &format!("night action skipped: ingest deferred ({reason}, {})", path.display()),
                );
                self.gate.record_skipped(now);
            }
            SchedulerAction::Rest { reason } => {
                runlog::log("tg", &format!("night rest: {reason}"));
                self.gate.record_rest();
            }
        }
    }

    // Inline drain, same pattern as run_memory_to_completion: the pass drops
    // its sender on return, so draining the receiver runs to completion.
    fn run_dream(&self) {
        let (sender, receiver) = std::sync::mpsc::channel::<DreamEvent>();
        let llm = Arc::clone(&self.memory_llm);
        let embed = Arc::clone(&self.embed);
        if let Err(err) = memory::run_dream_pass(llm, embed, sender) {
            runlog::log("tg", &format!("night dream failed: {err}"));
        }
        for _event in receiver {}
    }

    fn run_explore(&self) -> bool {
        let (sender, receiver) = std::sync::mpsc::channel::<ExploreEvent>();
        let llm = Arc::clone(&self.memory_llm);
        let embed = Arc::clone(&self.embed);
        let search = self.search.clone();
        if let Err(err) = memory::run_exploration_pass(llm, embed, search, sender) {
            runlog::log("tg", &format!("night explore failed: {err}"));
        }
        let mut had_yield = false;
        for event in receiver {
            if let ExploreEvent::Done { events_recorded } = event {
                had_yield = events_recorded > 0;
            }
        }
        had_yield
    }

    fn run_inbox_study(&self) {
        let Ok(conn) = memory::open_connection() else { return };
        let Ok(Some(item)) = memory::oldest_queued_inbox(&conn) else { return };
        let result = self.study_inbox_item(&item);
        match result {
            Ok(session) => {
                let _ = memory::resolve_inbox_item(&conn, item.item_id, "done", &session);
                runlog::log("inbox", &format!("studied: {} (session {session})", item.source));
            }
            Err(reason) => {
                let _ = memory::resolve_inbox_item(&conn, item.item_id, "failed", &reason);
                runlog::log("inbox", &format!("failed: {} ({reason})", item.source));
            }
        }
    }

    fn study_inbox_item(&self, item: &memory::InboxItem) -> Result<String, String> {
        let path = if item.source.starts_with("http://") || item.source.starts_with("https://") {
            let search = self.search.as_ref().ok_or("web extraction is unavailable")?;
            let text = search.extract_url(&item.source)?;
            let dir = std::path::PathBuf::from(crate::paths::data_file("inbox"));
            if let Err(err) = std::fs::create_dir_all(&dir) { return Err(err.to_string()) }
            let path = dir.join(format!("{}.txt", item.item_id));
            if let Err(err) = std::fs::write(&path, text) { return Err(err.to_string()) }
            path
        } else {
            std::path::PathBuf::from(&item.source)
        };
        let (sender, receiver) = std::sync::mpsc::channel::<IngestEvent>();
        let result = match memory::run_focused_ingestion_pass(
            Arc::clone(&self.memory_llm), Arc::clone(&self.embed), self.search.clone(),
            (&path, &item.instruction), sender,
        ) { Ok(session) => Ok(session), Err(err) => Err(err.to_string()) };
        for _event in receiver {}
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(raw: &str) -> DreamWindow {
        match DreamWindow::parse(raw) {
            Ok(win) => win,
            Err(err) => panic!("valid window {raw}: {err}"),
        }
    }

    fn gate(raw: &str) -> NightGate {
        NightGate::new(window(raw), Duration::from_secs(1200), 3)
    }

    #[test]
    fn window_parses_and_bounds_are_end_exclusive() {
        let win = window("3-6");
        assert!(win.contains(3));
        assert!(win.contains(4));
        assert!(win.contains(5));
        assert!(!win.contains(2));
        assert!(!win.contains(6));
    }

    #[test]
    fn window_wraps_across_midnight() {
        let win = window("23-5");
        for hour in [23, 0, 1, 2, 3, 4] {
            assert!(win.contains(hour), "hour {hour} should be inside");
        }
        assert!(!win.contains(5));
        assert!(!win.contains(22));
    }

    #[test]
    fn malformed_window_is_an_error_not_a_default() {
        assert!(DreamWindow::parse("banana").is_err());
        assert!(DreamWindow::parse("3").is_err());
        assert!(DreamWindow::parse("3-99").is_err());
        assert!(DreamWindow::parse("4-4").is_err());
    }

    #[test]
    fn dispatches_inside_window_when_cooldown_elapsed_and_under_cap() {
        let mut gate = gate("3-6");
        assert!(gate.should_evaluate(4, "2026-07-10", Instant::now()));
    }

    #[test]
    fn never_dispatches_outside_window() {
        let mut gate = gate("3-6");
        assert!(!gate.should_evaluate(2, "2026-07-10", Instant::now()));
        assert!(!gate.should_evaluate(6, "2026-07-10", Instant::now()));
    }

    #[test]
    fn does_not_dispatch_before_cooldown_elapses() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_action(base);
        let soon = base + Duration::from_secs(60);
        assert!(!gate.should_evaluate(4, "2026-07-10", soon));
        let later = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn does_not_dispatch_at_cap() {
        let mut gate = gate("3-6");
        let mut now = Instant::now();
        for _ in 0..3 {
            assert!(gate.should_evaluate(4, "2026-07-10", now));
            gate.record_action(now);
            now += Duration::from_secs(1200);
        }
        assert!(!gate.should_evaluate(4, "2026-07-10", now));
    }

    #[test]
    fn cap_resets_on_a_new_day() {
        let mut gate = gate("3-6");
        let mut now = Instant::now();
        for _ in 0..3 {
            assert!(gate.should_evaluate(4, "2026-07-10", now));
            gate.record_action(now);
            now += Duration::from_secs(1200);
        }
        assert!(!gate.should_evaluate(4, "2026-07-10", now));
        // Re-entering the window on a new day clears the counter.
        assert!(gate.should_evaluate(4, "2026-07-11", now));
    }

    #[test]
    fn rest_clears_dirty_and_evaluation_stops_until_marked_dirty() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_rest();
        // Same night, cooldown irrelevant: inputs unchanged, never re-evaluate.
        let later = base + Duration::from_secs(3600);
        assert!(!gate.should_evaluate(4, "2026-07-10", later));
        // A state change (turn handled) re-enables evaluation.
        gate.dirty = true;
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn a_new_day_sets_dirty_and_resets_the_nightly_dream_flag() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        // Establish the night first — entering the window on a new day is
        // itself a dirtying event, so the rest must come after.
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.dreamed_tonight = true;
        gate.record_rest();
        assert!(!gate.should_evaluate(4, "2026-07-10", base));
        // Window re-entry on the next day: dirty again, dream flag cleared.
        assert!(gate.should_evaluate(4, "2026-07-11", base));
        assert!(!gate.dreamed_tonight);
    }

    #[test]
    fn actions_leave_dirty_set_so_the_follow_up_evaluation_happens() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_action(base);
        // Cooldown gates the timing, but the state is still dirty.
        let later = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn empty_curiosity_consumes_a_slot_and_settles() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_curiosity(base, false);
        assert_eq!(gate.count, 1);
        let later = base + Duration::from_secs(1200);
        assert!(!gate.should_evaluate(4, "2026-07-10", later));
        assert_eq!(gate.last_curiosity_had_yield, Some(false));
    }

    #[test]
    fn skipped_action_consumes_cooldown_but_not_cap() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_skipped(base);
        assert_eq!(gate.count, 0, "cap not consumed");
        let soon = base + Duration::from_secs(60);
        assert!(!gate.should_evaluate(4, "2026-07-10", soon), "cooldown consumed");
    }
}
