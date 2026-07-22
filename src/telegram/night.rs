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
const MAX_DREAM_ATTEMPTS: usize = 2; // replay-tunable.
const DEFAULT_DREAM_HOURS: &str = "3-6";

struct PassExecution {
    outcome: &'static str,
    reason_class: &'static str,
    items_added: usize,
}

impl PassExecution {
    fn ok(items_added: usize) -> Self {
        Self {
            outcome: "ok",
            reason_class: "none",
            items_added,
        }
    }

    fn failed(reason_class: &'static str) -> Self {
        Self {
            outcome: "failed",
            reason_class,
            items_added: 0,
        }
    }

    fn skipped(reason_class: &'static str) -> Self {
        Self {
            outcome: "skipped",
            reason_class,
            items_added: 0,
        }
    }

    fn succeeded(&self) -> bool {
        self.outcome == "ok"
    }

    fn completed(&self) -> bool {
        self.outcome != "failed"
    }
}

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
            None => {
                return Err(format!(
                    "KRATT_DREAM_HOURS must be 'start-end', got '{raw}'"
                ));
            }
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
    dream_attempts: usize,
    dream_skip_recorded: bool,
    studied_inbox_tonight: bool,
    scanned_news_today: bool,
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
            dream_attempts: 0,
            dream_skip_recorded: false,
            studied_inbox_tonight: false,
            scanned_news_today: false,
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
            self.dream_attempts = 0;
            self.dream_skip_recorded = false;
            self.studied_inbox_tonight = false;
            self.scanned_news_today = false;
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

    fn can_attempt_dream(&self) -> bool {
        !self.dreamed_tonight && self.dream_attempts < MAX_DREAM_ATTEMPTS
    }

    fn record_dream(&mut self, now: Instant, succeeded: bool) {
        self.dream_attempts += 1;
        self.dreamed_tonight = succeeded || self.dream_attempts >= MAX_DREAM_ATTEMPTS;
        self.record_action(now);
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
        runlog::log(
            "tg",
            &format!("dream window {}-{}", window.start, window.end),
        );
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
        if !self.gate.can_attempt_dream() {
            return false;
        }
        if !scheduler::has_unconsolidated_experience() {
            if !self.gate.dream_skip_recorded {
                self.record_skipped_pass("dream", "no_material");
                self.gate.dream_skip_recorded = true;
            }
            return false;
        }
        let reason = "nightly guarantee: unconsolidated experience";
        scheduler::log_scheduler_decision(&SchedulerAction::Dream {
            reason: reason.to_string(),
        });
        runlog::log("tg", &format!("night action: dream ({reason})"));
        let result = self.run_dream();
        self.record_dream_result(now, &result);
        true
    }

    fn run_scheduler_cycle(&mut self, now: Instant) {
        let mut inputs = scheduler::gather_inputs();
        if self.gate.studied_inbox_tonight {
            inputs.queued_inbox = 0;
        }
        if self.gate.scanned_news_today {
            inputs.news_scan_due = false;
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
                self.dispatch_dream(&reason, now);
            }
            SchedulerAction::Explore { reason } => {
                runlog::log("tg", &format!("night action: explore ({reason})"));
                let _ = self.run_explore();
                self.gate.record_action(now);
            }
            SchedulerAction::Curiosity { reason } => {
                runlog::log("tg", &format!("night action: curiosity ({reason})"));
                let had_yield = self.run_explore().items_added > 0;
                self.gate.record_curiosity(now, had_yield);
            }
            SchedulerAction::NewsScan { reason } => self.dispatch_news_scan(&reason, now),
            SchedulerAction::InboxStudy { reason } => {
                runlog::log("tg", &format!("night action: inbox study ({reason})"));
                let result = self.run_inbox_study();
                self.gate.studied_inbox_tonight = result.completed();
                self.gate.record_action(now);
            }
            SchedulerAction::Ingest { path, reason } => {
                runlog::log(
                    "tg",
                    &format!(
                        "night action skipped: ingest deferred ({reason}, {})",
                        path.display()
                    ),
                );
                self.record_skipped_pass("ingestion", "not_eligible");
                self.gate.record_skipped(now);
            }
            SchedulerAction::Rest { reason } => {
                runlog::log("tg", &format!("night rest: {reason}"));
                self.gate.record_rest();
            }
        }
    }

    fn dispatch_dream(&mut self, reason: &str, now: Instant) {
        runlog::log("tg", &format!("night action: dream ({reason})"));
        if !self.gate.can_attempt_dream() {
            self.record_skipped_pass("dream", "not_eligible");
            self.gate.record_skipped(now);
            return;
        }
        let result = self.run_dream();
        self.record_dream_result(now, &result);
    }

    fn dispatch_news_scan(&mut self, reason: &str, now: Instant) {
        runlog::log("tg", &format!("night action: news scan ({reason})"));
        let result = self.run_news_scan();
        self.gate.scanned_news_today = result.completed();
        self.gate.record_action(now);
    }

    // Inline drain, same pattern as run_memory_to_completion: the pass drops
    // its sender on return, so draining the receiver runs to completion.
    fn run_dream(&self) -> PassExecution {
        let run_id = self.start_pass("dream");
        let (sender, receiver) = std::sync::mpsc::channel::<DreamEvent>();
        let llm = Arc::clone(&self.memory_llm);
        let embed = Arc::clone(&self.embed);
        let result = memory::run_dream_pass(llm, embed, sender);
        for _event in receiver {}
        let execution = match result {
            Ok(edges_added) => PassExecution::ok(edges_added),
            Err(err) => {
                runlog::log("tg", &format!("night dream failed: {err}"));
                PassExecution::failed(classify_pass_error(&err.to_string()))
            }
        };
        self.finish_pass(run_id, "dream", &execution);
        execution
    }

    fn run_explore(&self) -> PassExecution {
        let run_id = self.start_pass("exploration");
        let (sender, receiver) = std::sync::mpsc::channel::<ExploreEvent>();
        let llm = Arc::clone(&self.memory_llm);
        let embed = Arc::clone(&self.embed);
        let search = self.search.clone();
        let result = memory::run_exploration_pass(llm, embed, search, sender);
        for _event in receiver {}
        let execution = match result {
            Ok(events_recorded) => PassExecution::ok(events_recorded),
            Err(err) => {
                runlog::log("tg", &format!("night explore failed: {err}"));
                PassExecution::failed(classify_pass_error(&err.to_string()))
            }
        };
        self.finish_pass(run_id, "exploration", &execution);
        execution
    }

    fn run_news_scan(&self) -> PassExecution {
        let run_id = self.start_pass("news");
        let execution = match memory::run_news_scan(&*self.embed, self.search.as_deref()) {
            Ok(summary) => {
                runlog::log(
                    "news",
                    &format!(
                        "[autonomy] news scan — {} interests, {} novel findings",
                        summary.interests, summary.findings
                    ),
                );
                if summary.interests == 0 {
                    PassExecution::skipped("no_material")
                } else {
                    PassExecution::ok(summary.findings)
                }
            }
            Err(err) => {
                runlog::log("news", &format!("news scan failed: {err}"));
                PassExecution::failed(classify_pass_error(&err.to_string()))
            }
        };
        self.finish_pass(run_id, "news", &execution);
        execution
    }

    fn run_inbox_study(&self) -> PassExecution {
        let run_id = self.start_pass("ingestion");
        let Ok(conn) = memory::open_connection() else {
            let execution = PassExecution::failed("database_error");
            self.finish_pass(run_id, "ingestion", &execution);
            return execution;
        };
        let item = match memory::oldest_queued_inbox(&conn) {
            Ok(Some(item)) => item,
            Ok(None) => {
                let execution = PassExecution::skipped("no_material");
                self.finish_pass(run_id, "ingestion", &execution);
                return execution;
            }
            Err(err) => {
                runlog::log("inbox", &format!("queue read failed: {err}"));
                let execution = PassExecution::failed("database_error");
                self.finish_pass(run_id, "ingestion", &execution);
                return execution;
            }
        };
        let result = self.study_inbox_item(&item);
        let execution = match result {
            Ok(session) => {
                let _ = memory::resolve_inbox_item(&conn, item.item_id, "done", &session);
                runlog::log(
                    "inbox",
                    &format!("studied: {} (session {session})", item.source),
                );
                PassExecution::ok(1)
            }
            Err(reason) => {
                let _ = memory::resolve_inbox_item(&conn, item.item_id, "failed", &reason);
                runlog::log("inbox", &format!("failed: {} ({reason})", item.source));
                PassExecution::failed(classify_pass_error(&reason))
            }
        };
        self.finish_pass(run_id, "ingestion", &execution);
        execution
    }

    fn record_dream_result(&mut self, now: Instant, result: &PassExecution) {
        self.gate.record_dream(now, result.succeeded());
        if result.succeeded() {
            return;
        }
        if self.gate.can_attempt_dream() {
            runlog::log(
                "tg",
                "night dream retry permitted after the normal action cooldown",
            );
        } else {
            runlog::log("tg", "night dream retry cap reached");
        }
    }

    fn start_pass(&self, pass_type: &str) -> Option<i64> {
        match memory::start_pass_run(pass_type) {
            Ok(run_id) => Some(run_id),
            Err(err) => {
                runlog::log(
                    "tg",
                    &format!("{pass_type} pass outcome start failed: {err}"),
                );
                None
            }
        }
    }

    fn finish_pass(&self, run_id: Option<i64>, pass_type: &str, result: &PassExecution) {
        let Some(run_id) = run_id else {
            return;
        };
        if let Err(err) = memory::finish_pass_run(
            run_id,
            result.outcome,
            result.reason_class,
            result.items_added,
        ) {
            runlog::log(
                "tg",
                &format!("{pass_type} pass outcome finish failed: {err}"),
            );
            return;
        }
        memory::sync_push_best_effort("pass outcome");
    }

    fn record_skipped_pass(&self, pass_type: &str, reason_class: &'static str) {
        let run_id = self.start_pass(pass_type);
        self.finish_pass(run_id, pass_type, &PassExecution::skipped(reason_class));
    }

    fn study_inbox_item(&self, item: &memory::InboxItem) -> Result<String, String> {
        let path = if item.source.starts_with("http://") || item.source.starts_with("https://") {
            let search = self
                .search
                .as_ref()
                .ok_or("web extraction is unavailable")?;
            let text = search.extract_url(&item.source)?;
            let dir = std::path::PathBuf::from(crate::paths::data_file("inbox"));
            if let Err(err) = std::fs::create_dir_all(&dir) {
                return Err(err.to_string());
            }
            let path = dir.join(format!("{}.txt", item.item_id));
            if let Err(err) = std::fs::write(&path, text) {
                return Err(err.to_string());
            }
            path
        } else {
            std::path::PathBuf::from(&item.source)
        };
        let (sender, receiver) = std::sync::mpsc::channel::<IngestEvent>();
        let result = match memory::run_focused_ingestion_pass(
            Arc::clone(&self.memory_llm),
            Arc::clone(&self.embed),
            self.search.clone(),
            (&path, &item.instruction),
            sender,
        ) {
            Ok(session) => Ok(session),
            Err(err) => Err(err.to_string()),
        };
        for _event in receiver {}
        result
    }
}

fn classify_pass_error(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("context length")
        || lower.contains("maximum context")
        || lower.contains("too many tokens")
    {
        return "context_length";
    }
    if lower.contains("database") || lower.contains("sqlite") || lower.contains("turso") {
        return "database_error";
    }
    if lower.contains("unavailable") || lower.contains("not configured") {
        return "unavailable";
    }
    "provider_error"
}

#[cfg(test)]
#[path = "night_tests.rs"]
mod tests;
