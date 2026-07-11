// Offline replay harness (`--seat-replay`). Reads the last N logged turn pairs
// and compares the resolved candidate seat (CLASSIFIER_MODEL/SCRIBE_MODEL)
// against the baseline (MEMORY_MODEL) on the actual classify/scribe tasks.
// Strictly read-only: it collects, parses, counts and prints — it never opens
// the graph DB nor applies a single tool call.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;

use crate::api::LlmClient;
use crate::memory::{self, MemorySeats};
use crate::prompts::{build_classify_prompt, build_scribe_prompt};

const DEFAULT_TURNS: usize = 30;
const SAMPLE_TURNS: usize = 3;

#[derive(Deserialize)]
struct ReplayEntry {
    role: String,
    content: String,
}

pub fn run(seats: &MemorySeats) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CLASSIFIER_MODEL").is_err() && std::env::var("SCRIBE_MODEL").is_err() {
        return Err(
            "--seat-replay needs CLASSIFIER_MODEL or SCRIBE_MODEL set (nothing to compare)".into(),
        );
    }
    let turns = requested_turns();
    let pairs = load_recent_pairs(turns);
    if pairs.is_empty() {
        println!("seat-replay: no logged turn pairs found");
        return Ok(());
    }
    println!("seat-replay: comparing on {} turn pairs\n", pairs.len());
    let classifier = run_classifier_replay(&seats.heavy, &seats.classifier, &pairs);
    println!("{}", format_classifier_report(&classifier));
    let scribe = run_scribe_replay(&seats.heavy, &seats.scribe, &pairs);
    println!("{}", format_scribe_report(&scribe));
    Ok(())
}

fn requested_turns() -> usize {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--seat-replay"
            && let Some(value) = args.next()
            && let Ok(parsed) = value.parse::<usize>()
        {
            return parsed;
        }
    }
    DEFAULT_TURNS
}

// ---- turn-pair extraction -------------------------------------------------

fn load_recent_pairs(last_n: usize) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    for file in session_files() {
        load_entries_into(&file, &mut entries);
    }
    extract_turn_pairs(&entries, last_n)
}

// Consecutive (user, assistant) entries form one pair; anything else (a system
// note, a trailing unpaired user turn) is skipped. Returns the last `last_n`
// pairs in chronological order.
fn extract_turn_pairs(entries: &[ReplayEntry], last_n: usize) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut index = 0;
    while index + 1 < entries.len() {
        if entries[index].role == "user" && entries[index + 1].role == "assistant" {
            pairs.push((
                entries[index].content.clone(),
                entries[index + 1].content.clone(),
            ));
            index += 2;
        } else {
            index += 1;
        }
    }
    if pairs.len() > last_n {
        return pairs[pairs.len() - last_n..].to_vec();
    }
    pairs
}

fn load_entries_into(path: &PathBuf, entries: &mut Vec<ReplayEntry>) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return,
    };
    for line in text.lines() {
        if let Ok(entry) = serde_json::from_str::<ReplayEntry>(line) {
            entries.push(entry);
        }
    }
}

// Session filenames are timestamps, so a lexical sort is chronological.
fn session_files() -> Vec<PathBuf> {
    let read_dir = match std::fs::read_dir(crate::session::sessions_dir()) {
        Ok(read_dir) => read_dir,
        Err(_) => return Vec::new(),
    };
    let mut paths = Vec::new();
    for dir_entry in read_dir.flatten() {
        let path = dir_entry.path();
        if path.extension().is_some_and(|ext| ext == "jsonl") {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

// ---- shared timing --------------------------------------------------------

// Collects one response and its wall time; an error yields an empty reply so
// downstream parsing counts it as a malformed/no-call response rather than
// aborting the whole run.
fn timed_collect(llm: &Arc<dyn LlmClient>, prompt: &str) -> (String, u128) {
    let start = Instant::now();
    let reply = llm.collect(prompt).unwrap_or_default();
    (reply, start.elapsed().as_millis())
}

fn mean_ms(total_ms: u128, calls: usize) -> f64 {
    if calls == 0 {
        return 0.0;
    }
    total_ms as f64 / calls as f64
}

// ---- classifier replay ----------------------------------------------------

struct Disagreement {
    text: String,
    baseline: String,
    candidate: String,
}

struct ClassifierResult {
    total: usize,
    agreements: usize,
    disagreements: Vec<Disagreement>,
    baseline_ms: u128,
    candidate_ms: u128,
}

fn run_classifier_replay(
    baseline: &Arc<dyn LlmClient>,
    candidate: &Arc<dyn LlmClient>,
    pairs: &[(String, String)],
) -> ClassifierResult {
    let mut result = ClassifierResult {
        total: pairs.len(),
        agreements: 0,
        disagreements: Vec::new(),
        baseline_ms: 0,
        candidate_ms: 0,
    };
    for (user_msg, assistant_msg) in pairs {
        let prompt = build_classify_prompt(user_msg, assistant_msg, false);
        let (base_reply, base_ms) = timed_collect(baseline, &prompt);
        let (cand_reply, cand_ms) = timed_collect(candidate, &prompt);
        result.baseline_ms += base_ms;
        result.candidate_ms += cand_ms;
        let base_label = memory::classify_reply_label(&base_reply).to_string();
        let cand_label = memory::classify_reply_label(&cand_reply).to_string();
        if base_label == cand_label {
            result.agreements += 1;
        } else {
            result.disagreements.push(Disagreement {
                text: truncate(user_msg, 60),
                baseline: base_label,
                candidate: cand_label,
            });
        }
    }
    result
}

fn format_classifier_report(result: &ClassifierResult) -> String {
    let mut out = String::from("== classifier ==\n");
    let percent = percentage(result.agreements, result.total);
    out.push_str(&format!(
        "classifier agreement: {}/{} ({percent}%)\n",
        result.agreements, result.total
    ));
    out.push_str(&format!(
        "classifier latency: baseline {:.0}ms  candidate {:.0}ms (mean/call)\n",
        mean_ms(result.baseline_ms, result.total),
        mean_ms(result.candidate_ms, result.total),
    ));
    for disagreement in &result.disagreements {
        out.push_str(&format!(
            "  disagree [base={} cand={}]: {}\n",
            disagreement.baseline, disagreement.candidate, disagreement.text
        ));
    }
    out
}

// ---- scribe replay --------------------------------------------------------

struct ScribeSeatStats {
    parsed: usize,
    total_calls: usize,
    total_ms: u128,
}

struct ScribeResult {
    total: usize,
    baseline: ScribeSeatStats,
    candidate: ScribeSeatStats,
    samples: Vec<Vec<String>>,
}

fn run_scribe_replay(
    baseline: &Arc<dyn LlmClient>,
    candidate: &Arc<dyn LlmClient>,
    pairs: &[(String, String)],
) -> ScribeResult {
    let mut result = ScribeResult {
        total: pairs.len(),
        baseline: new_seat_stats(),
        candidate: new_seat_stats(),
        samples: Vec::new(),
    };
    for (index, (user_msg, assistant_msg)) in pairs.iter().enumerate() {
        let prompt = build_scribe_prompt(user_msg, assistant_msg);
        let (base_reply, base_ms) = timed_collect(baseline, &prompt);
        let (cand_reply, cand_ms) = timed_collect(candidate, &prompt);
        accumulate_seat(&mut result.baseline, &base_reply, base_ms);
        let cand_calls = accumulate_seat(&mut result.candidate, &cand_reply, cand_ms);
        if index < SAMPLE_TURNS {
            result.samples.push(cand_calls);
        }
    }
    result
}

fn new_seat_stats() -> ScribeSeatStats {
    ScribeSeatStats {
        parsed: 0,
        total_calls: 0,
        total_ms: 0,
    }
}

// Parses one scribe response, folds its counts into the seat stats, and returns
// the human-readable descriptions of its calls (for the sample section).
fn accumulate_seat(stats: &mut ScribeSeatStats, reply: &str, elapsed_ms: u128) -> Vec<String> {
    stats.total_ms += elapsed_ms;
    let calls = memory::parse_tool_calls(reply);
    if !calls.is_empty() {
        stats.parsed += 1;
        stats.total_calls += calls.len();
    }
    let mut described = Vec::new();
    for call in &calls {
        described.push(memory::describe_tool_call(call));
    }
    described
}

fn format_scribe_report(result: &ScribeResult) -> String {
    let mut out = String::from("== scribe ==\n");
    out.push_str(&format_scribe_seat("baseline", &result.baseline, result.total));
    out.push_str(&format_scribe_seat(
        "candidate",
        &result.candidate,
        result.total,
    ));
    for (index, calls) in result.samples.iter().enumerate() {
        out.push_str(&format!("  candidate sample turn {}: ", index + 1));
        if calls.is_empty() {
            out.push_str("(no calls)\n");
        } else {
            out.push_str(&calls.join(", "));
            out.push('\n');
        }
    }
    out
}

fn format_scribe_seat(label: &str, stats: &ScribeSeatStats, total: usize) -> String {
    let malformed = total - stats.parsed;
    format!(
        "scribe {label}: parse {}/{} ({}%)  mean-calls {:.2}  malformed {malformed}  latency {:.0}ms\n",
        stats.parsed,
        total,
        percentage(stats.parsed, total),
        mean_ms(stats.total_calls as u128, total),
        mean_ms(stats.total_ms, total),
    )
}

// ---- small helpers --------------------------------------------------------

fn percentage(part: usize, whole: usize) -> usize {
    if whole == 0 {
        return 0;
    }
    part * 100 / whole
}

fn truncate(text: &str, max_chars: usize) -> String {
    let cleaned = text.replace('\n', " ");
    if cleaned.chars().count() <= max_chars {
        return cleaned;
    }
    let mut out = String::new();
    for (index, character) in cleaned.chars().enumerate() {
        if index >= max_chars {
            break;
        }
        out.push(character);
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::{
        ReplayEntry, extract_turn_pairs, format_classifier_report, format_scribe_report,
        run_classifier_replay, run_scribe_replay,
    };
    use crate::api::{Chunk, LlmClient, TurnOutcome};
    use std::error::Error;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::mpsc::Sender;

    fn entry(role: &str, content: &str) -> ReplayEntry {
        ReplayEntry {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn pairs_are_ordered_and_last_n_respected() {
        let entries = vec![
            entry("user", "u1"),
            entry("assistant", "a1"),
            entry("user", "u2"),
            entry("assistant", "a2"),
            entry("user", "u3"),
            entry("assistant", "a3"),
        ];
        let pairs = extract_turn_pairs(&entries, 2);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("u2".to_string(), "a2".to_string()));
        assert_eq!(pairs[1], ("u3".to_string(), "a3".to_string()));
    }

    #[test]
    fn unpaired_trailing_user_entry_is_skipped() {
        let entries = vec![
            entry("user", "u1"),
            entry("assistant", "a1"),
            entry("user", "dangling"),
        ];
        let pairs = extract_turn_pairs(&entries, 30);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("u1".to_string(), "a1".to_string()));
    }

    // Hands out one canned reply per collect() call, in order.
    struct ScriptedClient {
        replies: Mutex<Vec<String>>,
    }

    impl ScriptedClient {
        fn scripted(replies: &[&str]) -> Arc<dyn LlmClient> {
            let mut queued = Vec::new();
            for reply in replies.iter().rev() {
                queued.push((*reply).to_string());
            }
            Arc::new(ScriptedClient {
                replies: Mutex::new(queued),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            let Ok(mut queued) = self.replies.lock() else {
                return Ok(String::new());
            };
            Ok(queued.pop().unwrap_or_default())
        }

        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {
            unimplemented!("scripted client collects only")
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client collects only")
        }
    }

    fn two_pairs() -> Vec<(String, String)> {
        vec![
            ("u1".to_string(), "a1".to_string()),
            ("u2".to_string(), "a2".to_string()),
        ]
    }

    #[test]
    fn classifier_agreement_line_counts_matches() {
        let baseline = ScriptedClient::scripted(&["TASK", "CHAT"]);
        let candidate = ScriptedClient::scripted(&["TASK", "TASK"]);
        let result = run_classifier_replay(&baseline, &candidate, &two_pairs());
        let report = format_classifier_report(&result);
        assert!(report.contains("classifier agreement: 1/2 (50%)"), "{report}");
        assert!(report.contains("disagree [base=chat cand=task]"), "{report}");
    }

    const ONE_CALL: &str =
        r#"{"tools":[{"tool":"upsert_edge","args":{"source":"game","target":"move-e4","content":"played e4","stability":0.3,"confidence":1.0,"importance":1.0,"subject":false,"date":"2026-07-11","memory_type":"episodic"}}]}"#;

    #[test]
    fn scribe_parse_rate_line_counts_valid_responses() {
        let baseline = ScriptedClient::scripted(&[ONE_CALL, ONE_CALL]);
        let candidate = ScriptedClient::scripted(&[ONE_CALL, "not json at all"]);
        let result = run_scribe_replay(&baseline, &candidate, &two_pairs());
        let report = format_scribe_report(&result);
        assert!(report.contains("scribe baseline: parse 2/2 (100%)"), "{report}");
        assert!(
            report.contains("scribe candidate: parse 1/2 (50%)"),
            "{report}"
        );
        assert!(report.contains("malformed 1"), "{report}");
    }
}
