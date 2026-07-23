// The daemon's night cycle runs autonomous maintenance during quiet hours.
// After an idle poll, it may run one scheduler cycle inline on the poll thread.

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::api::{EmbeddingClient, LlmClient};
use crate::memory::scheduler::{self, SchedulerAction};
use crate::memory::{self, IngestEvent};
use crate::runlog;
use crate::timeutil;
use crate::web::SearchClient;

// replay-tunable.
const NIGHT_ACTION_COOLDOWN: Duration = Duration::from_secs(20 * 60);
// replay-tunable.
const MAX_NIGHT_ACTIONS: usize = 3;
const DEFAULT_NIGHT_HOURS: &str = "3-6";

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

    fn completed(&self) -> bool {
        self.outcome != "failed"
    }
}

// Hour range [start, end) in UTC (timeutil::current_hour is epoch-derived,
// like every date in the app), end exclusive. A wrapping range like `23-5`
// means 23,0,1,2,3,4 are inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightWindow {
    start: u32,
    end: u32,
}

impl NightWindow {
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
    window: NightWindow,
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
    studied_inbox_tonight: bool,
}

impl NightGate {
    fn new(window: NightWindow, cooldown: Duration, max_actions: usize) -> Self {
        Self {
            window,
            cooldown,
            max_actions,
            last_action: None,
            count: 0,
            night_day: None,
            dirty: true,
            studied_inbox_tonight: false,
        }
    }

    // May a scheduler evaluation run right now? Re-entering the window on a
    // new day resets per-night state and marks inputs dirty.
    fn should_evaluate(&mut self, hour: u32, today: &str, now: Instant) -> bool {
        let inside = self.window.contains(hour);
        if inside && self.night_day.as_deref() != Some(today) {
            self.night_day = Some(today.to_string());
            self.count = 0;
            self.studied_inbox_tonight = false;
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

    // A completed action counts against both the cooldown and the per-night cap.
    fn record_action(&mut self, now: Instant) {
        self.count += 1;
        self.last_action = Some(now);
    }

    // A skipped action consumes the cooldown so
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
        let raw = std::env::var("KRATT_DREAM_HOURS").unwrap_or_else(|_| DEFAULT_NIGHT_HOURS.into());
        let window = NightWindow::parse(&raw)?;
        runlog::log(
            "tg",
            &format!("night window {}-{}", window.start, window.end),
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
        self.run_scheduler_cycle(now);
    }

    // A turn was handled (or state otherwise changed): scheduler inputs may
    // differ now, so the next idle poll inside the window re-evaluates.
    pub fn mark_dirty(&mut self) {
        self.gate.dirty = true;
    }

    fn run_scheduler_cycle(&mut self, now: Instant) {
        let mut inputs = scheduler::gather_inputs();
        if self.gate.studied_inbox_tonight {
            inputs.queued_inbox = 0;
        }
        let action = scheduler::decide(&inputs);
        scheduler::log_scheduler_decision(&action);
        if !matches!(action, SchedulerAction::Rest { .. }) {
            scheduler::log_drift_info(&inputs.actor_positions);
        }
        self.dispatch(action, now);
    }

    fn dispatch(&mut self, action: SchedulerAction, now: Instant) {
        match action {
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
