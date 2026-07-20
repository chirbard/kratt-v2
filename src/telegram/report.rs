// The daily 1:1 in its report form: each morning, on the idle-poll path, the
// daemon sends one Telegram message articulating what it learned and
// consolidated in the last day, then treats a bare-number reply as a Phase 23
// outcome over the whole day of learning. Report + verdict only — the teaching
// form comes later.

use std::error::Error;
use std::sync::Arc;

use crate::api::LlmClient;
use crate::memory::{self, DayDigest};
use crate::prompts;
use crate::runlog;
use crate::session;
use crate::timeutil;

use super::bot::BotTransport;

const DEFAULT_REPORT_HOUR: u32 = 7;
// Appended in code, never by the LLM: a fixed, unambiguous scoring convention.
const VERDICT_ASK: &str = "Reply with a single number from -2 (wasted day) to 2 (great day) to score it, or just keep chatting.";

// Pure day-marker gate: fires at the report hour at most once per civil day.
// In-memory only — a daemon restart mid-morning may re-send once (same
// tradeoff as NightGate::night_day); acceptable and cheap.
struct ReportGate {
    hour: u32,
    report_day: Option<String>,
}

impl ReportGate {
    fn new(hour: u32) -> Self {
        Self {
            hour,
            report_day: None,
        }
    }

    // True at most once per day, when the local hour equals the report hour.
    // The marker is set on the first firing so a quiet day is not re-checked
    // every idle tick.
    fn should_fire(&mut self, hour: u32, today: &str) -> bool {
        if hour != self.hour {
            return false;
        }
        if self.report_day.as_deref() == Some(today) {
            return false;
        }
        self.report_day = Some(today.to_string());
        true
    }
}

pub struct ReportCycle {
    gate: ReportGate,
    memory_llm: Arc<dyn LlmClient>,
    // Private-chat invariant: chat id equals the allowed user id.
    chat_id: i64,
    // The yesterday-date a report was sent for and is awaiting a verdict on;
    // cleared by a verdict or replaced by the next report day.
    awaiting_verdict: Option<String>,
}

impl ReportCycle {
    pub fn from_env(memory_llm: Arc<dyn LlmClient>, chat_id: i64) -> Result<Self, Box<dyn Error>> {
        let hour = report_hour_from_env()?;
        runlog::log("tg", &format!("daily report hour {hour}"));
        Ok(Self {
            gate: ReportGate::new(hour),
            memory_llm,
            chat_id,
            awaiting_verdict: None,
        })
    }

    // Called after each idle poll. Sends at most one report per day.
    // The night slice this report covers: yesterday's report hour → today's.
    // Dreams run after midnight, so a civil-day slice for dream events would
    // systematically miss the night that consolidated yesterday (live evidence
    // 2026-07-12: dream at 00:00:47, report said "no dream events overnight").
    fn night_window(&self, yesterday: &str) -> memory::NightWindow {
        let hour = self.gate.hour;
        memory::NightWindow {
            from: format!("{yesterday}T{hour:02}"),
            until: format!("{}T{hour:02}", timeutil::today_iso()),
        }
    }

    pub fn maybe_send(&mut self, transport: &dyn BotTransport) {
        let hour = timeutil::current_hour();
        let today = timeutil::today_iso();
        if !self.gate.should_fire(hour, &today) {
            return;
        }
        // A new report day supersedes any still-pending verdict.
        self.awaiting_verdict = None;
        let yesterday = timeutil::yesterday_iso();
        let digest = memory::day_digest(&yesterday, &self.night_window(&yesterday));
        if digest.is_quiet() {
            runlog::log("tg", "report skipped: quiet day");
            return;
        }
        let message = self.compose_message(&digest);
        super::send_answer(transport, self.chat_id, &message);
        // Into today's session file too: the user sees the report on the same
        // Telegram page as the chat, so the model must see it in its recent
        // context as well — otherwise it has no idea what it reported.
        session::append_entry(&session_path_for_today(&today), "assistant", &message);
        runlog::log("tg", "daily report sent");
        self.awaiting_verdict = Some(yesterday);
    }

    // The LLM phrases the digest; on any failure the digest itself is sent so
    // the report never silently vanishes. The verdict ask is appended in code.
    fn compose_message(&self, digest: &DayDigest) -> String {
        let digest_text = render_digest(digest);
        let body = match self
            .memory_llm
            .collect(&prompts::build_daily_report_prompt(&digest_text))
        {
            Ok(text) if !text.trim().is_empty() => text.trim().to_string(),
            Ok(_) | Err(_) => {
                runlog::log("tg", "daily report: LLM phrasing failed, sending digest");
                digest_text
            }
        };
        format!("{body}\n\n{VERDICT_ASK}")
    }

    // Verdict interception: while awaiting a verdict, a message that trims to a
    // bare integer in -2..=2 is a control message, not a chat turn. Returns
    // true (with a confirmation already sent) when it consumed the message.
    pub fn try_handle_verdict(&mut self, transport: &dyn BotTransport, text: &str) -> bool {
        let Some((yesterday, score)) = self.take_verdict(text) else {
            return false;
        };
        record_day_verdict(&yesterday, score);
        let confirmation = format!("scored yesterday {score:+.1} — thanks");
        super::send_answer(transport, self.chat_id, &confirmation);
        // The verdict exchange belongs to the visible conversation too.
        let session = session_path_for_today(&timeutil::today_iso());
        session::append_entry(&session, "user", text);
        session::append_entry(&session, "assistant", &confirmation);
        runlog::log("tg", &format!("daily verdict recorded: {score:+.1}"));
        true
    }

    // Pure classification: when armed and `text` is a bare verdict, disarm and
    // return (yesterday, score); otherwise leave the flag untouched. A verdict
    // is only intercepted while a report is awaiting one.
    fn take_verdict(&mut self, text: &str) -> Option<(String, f64)> {
        let yesterday = self.awaiting_verdict.clone()?;
        let value = parse_verdict(text)?;
        self.awaiting_verdict = None;
        Some((yesterday, value as f64 / 2.0))
    }
}

#[cfg(test)]
impl ReportCycle {
    // Unarmed cycle for gateway tests that exercise the poll path without a
    // pending verdict — verdict interception is a no-op here.
    pub(super) fn for_test(memory_llm: Arc<dyn LlmClient>, chat_id: i64) -> Self {
        Self {
            gate: ReportGate::new(DEFAULT_REPORT_HOUR),
            memory_llm,
            chat_id,
            awaiting_verdict: None,
        }
    }
}

fn report_hour_from_env() -> Result<u32, Box<dyn Error>> {
    let raw = match std::env::var("KRATT_REPORT_HOUR") {
        Ok(raw) => raw,
        Err(_) => return Ok(DEFAULT_REPORT_HOUR),
    };
    let hour: u32 = match raw.trim().parse() {
        Ok(hour) => hour,
        Err(_) => return Err(format!("KRATT_REPORT_HOUR '{raw}' is not a number").into()),
    };
    if hour > 23 {
        return Err(format!("KRATT_REPORT_HOUR '{raw}' must be 0-23").into());
    }
    Ok(hour)
}

// A verdict scores yesterday's tg day-session and, if a dream ran, the newest
// dream session too — the day and its overnight consolidation are judged
// together. The payload identifies the source so replay can tell 1:1 outcomes
// from game outcomes.
fn record_day_verdict(yesterday: &str, score: f64) {
    let payload = serde_json::json!({"kind": "daily-1:1"}).to_string();
    for session_id in verdict_session_ids(yesterday, memory::newest_dream_session_id()) {
        if let Err(err) = memory::record_outcome(&session_id, score, &payload) {
            runlog::log(
                "tg",
                &format!("verdict outcome failed ({session_id}): {err}"),
            );
        }
    }
}

// The sessions a verdict scores: yesterday's tg day-session always, plus the
// newest dream session when one exists (the night's consolidation is part of
// the day being judged).
fn verdict_session_ids(yesterday: &str, newest_dream: Option<String>) -> Vec<String> {
    let mut ids = vec![format!("tg-{yesterday}")];
    if let Some(dream) = newest_dream {
        ids.push(dream);
    }
    ids
}

// A bare integer in -2..=2 with nothing else. "2 but great" and "yesterday was
// good" parse to None — the convention is deliberately strict for zero false
// positives on real chat.
fn parse_verdict(text: &str) -> Option<i64> {
    let value: i64 = text.trim().parse().ok()?;
    if (-2..=2).contains(&value) {
        Some(value)
    } else {
        None
    }
}

// The deterministic half of the report: a compact text rendering of the digest
// that both feeds the phrasing call and stands alone as the fallback body.
fn session_path_for_today(today: &str) -> std::path::PathBuf {
    super::session_path_for_id(&format!("tg-{today}"))
}

fn render_digest(digest: &DayDigest) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Day: {}", digest.day));
    push_conversation_lines(&mut lines, digest);
    lines.push(format!(
        "Edges created {}, updated {}, pruned {}",
        digest.created, digest.updated, digest.pruned
    ));
    push_dream_lines(&mut lines, digest);
    if let Some(line) = digest.autonomy.line() {
        lines.push(line);
    }
    push_ref_line(&mut lines, "Gists consolidated", &digest.gists);
    push_ref_line(&mut lines, "Due loops", &digest.due_loops);
    push_ref_line(&mut lines, "Inbox", &digest.inbox_outcomes);
    push_ref_line(&mut lines, "Held for you", &digest.held_findings);
    push_ref_line(&mut lines, "Most-used beliefs", &digest.top_injected);
    push_ref_line(&mut lines, "Example new edges", &digest.new_edges);
    lines.join("\n")
}

fn push_conversation_lines(lines: &mut Vec<String>, digest: &DayDigest) {
    if digest.conversation_user_messages == 0 {
        lines.push("Conversation: none today.".to_string());
        return;
    }
    if digest.turn_groups.is_empty() {
        lines.push(format!(
            "Conversation: {} messages, but no durable learning was recorded.",
            digest.conversation_user_messages
        ));
        return;
    }
    let mut index = 0usize;
    while index < digest.turn_groups.len() {
        lines.push(format!(
            "Thread {}: {}",
            index + 1,
            digest.turn_groups[index]
        ));
        index += 1;
    }
}

fn push_dream_lines(lines: &mut Vec<String>, digest: &DayDigest) {
    let Some(run) = digest.dream_runs.last() else {
        lines.push("Dream: did not run (no recorded outcome).".to_string());
        return;
    };
    match run.outcome.as_str() {
        "ok" => {
            let attempts = attempt_suffix(digest.dream_runs.len());
            lines.push(format!(
                "Dream: ok — {} edges written{attempts}.",
                run.items_added
            ));
            lines.push(format!("Overnight dream events: {}", digest.dream_events));
        }
        "failed" => lines.push(failed_dream_line(digest, &run.reason_class)),
        "skipped" if run.reason_class == "no_material" => {
            lines.push("Dream: did not run (no unconsolidated experience).".to_string());
        }
        "skipped" => lines.push(format!("Dream: skipped ({}).", run.reason_class)),
        _ => lines.push("Dream: did not run (invalid recorded outcome).".to_string()),
    }
}

fn failed_dream_line(digest: &DayDigest, reason_class: &str) -> String {
    let mut line = format!("Dream: failed ({reason_class})");
    if digest.dream_consecutive_failures > 1 {
        line.push_str(&format!(
            ", {} consecutive night",
            ordinal(digest.dream_consecutive_failures)
        ));
    }
    if let Some(timestamp) = &digest.dream_last_successful_at {
        line.push_str(&format!(", last success {}", date_prefix(timestamp)));
    }
    line.push('.');
    line
}

fn attempt_suffix(attempts: usize) -> String {
    if attempts > 1 {
        return format!(" after {attempts} attempts");
    }
    String::new()
}

fn ordinal(value: usize) -> String {
    let suffix = match value % 100 {
        11..=13 => "th",
        _ => match value % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{value}{suffix}")
}

fn date_prefix(timestamp: &str) -> &str {
    timestamp.get(..10).unwrap_or(timestamp)
}

fn push_ref_line(lines: &mut Vec<String>, label: &str, refs: &[String]) {
    if refs.is_empty() {
        return;
    }
    lines.push(format!("{label}: {}", refs.join("; ")));
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::sync::Arc;
    use std::sync::mpsc::Sender;

    use crate::api::{Chunk, LlmClient, TurnOutcome};
    use crate::memory::PassRun;

    use super::{DayDigest, ReportGate, parse_verdict, render_digest, verdict_session_ids};

    // The classification tests never phrase a report, so the LLM is never
    // called — this stub exists only to construct a ReportCycle.
    struct StubLlm;

    impl LlmClient for StubLlm {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unreachable!("classification tests never call the LLM")
        }
        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {}
        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unreachable!("classification tests never call the LLM")
        }
    }

    // A ReportCycle with a stub LLM, armed on `yesterday`, for verdict
    // classification tests that must not touch the network or the database.
    fn armed_cycle(awaiting_verdict: Option<String>) -> super::ReportCycle {
        super::ReportCycle {
            gate: ReportGate::new(7),
            memory_llm: Arc::new(StubLlm),
            chat_id: 42,
            awaiting_verdict,
        }
    }

    #[test]
    fn a_verdict_is_taken_only_while_armed_and_then_disarms() {
        let mut cycle = armed_cycle(Some("2026-07-10".to_string()));
        let taken = cycle.take_verdict("2");
        assert_eq!(taken, Some(("2026-07-10".to_string(), 1.0)));
        // The flag is cleared, so the same reply is no longer a verdict.
        assert_eq!(cycle.take_verdict("2"), None);
    }

    #[test]
    fn a_bare_number_is_not_a_verdict_when_unarmed() {
        let mut cycle = armed_cycle(None);
        assert_eq!(cycle.take_verdict("2"), None);
    }

    #[test]
    fn a_non_verdict_reply_leaves_the_flag_armed() {
        let mut cycle = armed_cycle(Some("2026-07-10".to_string()));
        assert_eq!(cycle.take_verdict("2 but great"), None);
        // Still armed: the user can chat first and score later that day.
        assert_eq!(
            cycle.take_verdict("1"),
            Some(("2026-07-10".to_string(), 0.5))
        );
    }

    #[test]
    fn a_verdict_scores_the_day_session_and_the_dream_when_present() {
        assert_eq!(
            verdict_session_ids("2026-07-10", None),
            vec!["tg-2026-07-10".to_string()]
        );
        assert_eq!(
            verdict_session_ids("2026-07-10", Some("dream-123".to_string())),
            vec!["tg-2026-07-10".to_string(), "dream-123".to_string()]
        );
    }

    #[test]
    fn gate_fires_once_at_the_report_hour() {
        let mut gate = ReportGate::new(7);
        assert!(!gate.should_fire(6, "2026-07-11"), "before the hour");
        assert!(gate.should_fire(7, "2026-07-11"), "at the hour");
        assert!(!gate.should_fire(7, "2026-07-11"), "not twice the same day");
    }

    #[test]
    fn gate_fires_again_on_the_next_day() {
        let mut gate = ReportGate::new(7);
        assert!(gate.should_fire(7, "2026-07-11"));
        assert!(
            gate.should_fire(7, "2026-07-12"),
            "a new day re-arms the gate"
        );
    }

    #[test]
    fn verdict_parsing_accepts_only_bare_integers_in_range() {
        assert_eq!(parse_verdict("2"), Some(2));
        assert_eq!(parse_verdict("-1"), Some(-1));
        assert_eq!(parse_verdict("0"), Some(0));
        assert_eq!(parse_verdict(" 2 "), Some(2));
        assert_eq!(parse_verdict("3"), None, "out of range");
        assert_eq!(parse_verdict("2 but great"), None);
        assert_eq!(parse_verdict("yesterday was good"), None);
        assert_eq!(parse_verdict(""), None);
    }

    #[test]
    fn verdict_maps_to_the_reinforcement_range() {
        for (raw, expected) in [("2", 1.0), ("-1", -0.5), ("0", 0.0)] {
            let Some(value) = parse_verdict(raw) else {
                unreachable!("valid verdict")
            };
            assert!((value as f64 / 2.0 - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn a_quiet_digest_is_recognised() {
        assert!(DayDigest::default().is_quiet());
        let busy = DayDigest {
            turns: 3,
            ..DayDigest::default()
        };
        assert!(!busy.is_quiet());
        let due_only = DayDigest {
            due_loops: vec!["ask about the file (due 2026-07-10)".to_string()],
            ..DayDigest::default()
        };
        assert!(!due_only.is_quiet());
    }

    #[test]
    fn render_omits_empty_ref_lines_but_keeps_counts() {
        let digest = DayDigest {
            day: "2026-07-10".to_string(),
            turns: 4,
            conversation_user_messages: 4,
            created: 2,
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains("Conversation: 4 messages, but no durable learning was recorded."));
        assert!(text.contains("Edges created 2"));
        assert!(!text.contains("Gists consolidated"));
        assert!(!text.contains("Due loops"));
    }

    fn pass_run(outcome: &str, reason_class: &str, items_added: usize) -> PassRun {
        PassRun {
            run_id: 1,
            pass_type: "dream".to_string(),
            started_at: String::new(),
            finished_at: String::new(),
            outcome: outcome.to_string(),
            reason_class: reason_class.to_string(),
            items_added,
        }
    }

    #[test]
    fn render_asserts_all_three_conversation_states() {
        let none = render_digest(&DayDigest::default());
        assert!(none.contains("Conversation: none today."));
        assert!(!none.contains("Thread 1:"));

        let messages_only = DayDigest {
            conversation_user_messages: 2,
            ..DayDigest::default()
        };
        let messages_text = render_digest(&messages_only);
        assert!(
            messages_text
                .contains("Conversation: 2 messages, but no durable learning was recorded.")
        );

        let learned = DayDigest {
            conversation_user_messages: 2,
            turn_groups: vec!["a → b".to_string()],
            ..DayDigest::default()
        };
        let learned_text = render_digest(&learned);
        assert!(learned_text.contains("Thread 1: a → b"));
        assert!(!learned_text.contains("no durable learning"));
        assert!(!learned_text.contains("none today"));
    }

    #[test]
    fn failed_dream_renders_outcome_and_omits_event_count() {
        let last_success = crate::timeutil::yesterday_iso();
        let digest = DayDigest {
            dream_events: 451,
            dream_runs: vec![pass_run("failed", "context_length", 0)],
            dream_consecutive_failures: 3,
            dream_last_successful_at: Some(format!("{last_success}T04:00:00Z")),
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains(&format!(
            "Dream: failed (context_length), 3rd consecutive night, last success {last_success}."
        )));
        assert!(!text.contains("Overnight dream events"));
    }

    #[test]
    fn successful_dream_renders_written_edges_and_event_count() {
        let digest = DayDigest {
            dream_events: 12,
            dream_runs: vec![pass_run("ok", "none", 3)],
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains("Dream: ok — 3 edges written."));
        assert!(text.contains("Overnight dream events: 12"));
    }

    #[test]
    fn render_includes_due_loops() {
        let digest = DayDigest {
            due_loops: vec![
                "ask about the file (due 2026-07-10)".to_string(),
                "check the service (due 2026-07-11)".to_string(),
            ],
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains(
            "Due loops: ask about the file (due 2026-07-10); check the service (due 2026-07-11)"
        ));
    }

    #[test]
    fn render_includes_autonomy_only_when_non_empty() {
        let empty = render_digest(&DayDigest::default());
        assert!(!empty.contains("Autonomy:"));
        let mut digest = DayDigest::default();
        digest.autonomy.actions = 2;
        digest.autonomy.yielded = 1;
        assert!(render_digest(&digest).contains("Autonomy: Ran 2 self-directed passes"));
    }
}
