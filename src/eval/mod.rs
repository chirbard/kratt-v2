use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use serde_json::Value;

use crate::api::{self, EmbeddingClient, LlmClient};
use crate::memory::{self, Edge, Node};
use crate::trace;

#[derive(Clone, Debug, PartialEq)]
struct Commit {
    hash: String,
    timestamp: String,
    subject: String,
    body: String,
    files: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct Score {
    precision: f64,
    recall: f64,
    f1_score: f64,
}

struct RunCtx<'ctx> {
    llm: &'ctx dyn LlmClient,
    embed: Option<&'ctx dyn EmbeddingClient>,
    output: &'ctx Path,
    checkpoint: usize,
    mode: &'ctx str,
}

struct DumpRow<'row> {
    commit: &'row Commit,
    task: &'row str,
    claims: &'row str,
    prediction: &'row Prediction,
    score: &'row Score,
}

#[derive(Clone, Debug, PartialEq)]
struct Prediction {
    paths: Vec<String>,
}

const EXTRACT_PROMPT: &str = "You are given a development task and the repository-relative files the change actually touched. For each given file, write one short claim stating its role in this kind of change. Return JSON only as {\"claims\":[{\"file\":\"repo-relative path\",\"why\":\"short reason\"}]}. Use only the given paths.\n\n";
const PREDICT_PROMPT: &str = "Predict the repository-relative files this task will touch. Return JSON only as {\"paths\":[\"path\"]}. Use exact paths when a claim supports them; do not explain.\n\nTASK:\n";

type EvalResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub fn run(repo: &str) -> EvalResult<()> {
    let train_n = cli_usize("--train", env_usize("PREDICT_TRAIN", 20));
    let holdout_n = cli_usize("--holdout", env_usize("PREDICT_HOLDOUT", 10));
    let skip_n = cli_usize("--skip", env_usize("PREDICT_SKIP", 0));
    let commits = extract_commits(Path::new(repo))?;
    let split = chronological_split(&commits, train_n, holdout_n, skip_n);
    if split.1.is_empty() {
        return Err("not enough qualifying commits for holdout split".into());
    }
    memory::startup_database()?;
    memory::db_init()?;
    let (llm, embed) = build_clients();
    let checkpoints = checkpoints(split.0.len());
    let output = output_dir();
    std::fs::create_dir_all(&output)?;
    let mut csv = String::from("training,mode,precision,recall,f1\n");
    let mut consumed = 0usize;
    for checkpoint in checkpoints {
        while consumed < checkpoint {
            train_commit(&split.0[consumed], &*llm)?;
            consumed += 1;
        }
        memory::run_embed_pass(Arc::clone(&embed));
        let cold_ctx = RunCtx { llm: &*llm, embed: None, output: &output, checkpoint, mode: "cold" };
        let cold = score_holdout(&split.1, &cold_ctx)?;
        let memory_ctx = RunCtx { llm: &*llm, embed: Some(&*embed), output: &output, checkpoint, mode: "memory" };
        let memory = score_holdout(&split.1, &memory_ctx)?;
        append_csv(&mut csv, checkpoint, "cold", &cold);
        append_csv(&mut csv, checkpoint, "memory", &memory);
        println!("{checkpoint:>8}  cold F1 {:>6.3}  memory F1 {:>6.3}", cold.f1_score, memory.f1_score);
    }
    std::fs::write(output.join("curve.csv"), csv)?;
    Ok(())
}

fn build_clients() -> (Arc<dyn LlmClient>, Arc<dyn EmbeddingClient>) {
    let key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let model = std::env::var("MEMORY_MODEL").unwrap_or_else(|_| "openai/gpt-oss-20b".to_string());
    let llm: Arc<dyn LlmClient> = trace::TracingClient::wrap(
        api::resolve_model_client(&model, &key),
        trace::TraceContext::new("predict-eval"),
    );
    let embed_model = std::env::var("EMBED_MODEL")
        .unwrap_or_else(|_| "openai/text-embedding-3-small".to_string());
    let embed: Arc<dyn EmbeddingClient> = Arc::new(api::new_embed_client(key, embed_model));
    (llm, embed)
}

fn env_usize(name: &str, default: usize) -> usize {
    let Ok(value) = std::env::var(name) else { return default; };
    match value.parse() { Ok(parsed) => parsed, Err(_) => default }
}

fn cli_usize(name: &str, default: usize) -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == name
            && let Some(value) = args.next()
            && let Ok(parsed) = value.parse()
        {
            return parsed;
        }
    }
    default
}

fn output_dir() -> PathBuf {
    PathBuf::from(crate::paths::data_file("predict-eval"))
}

fn checkpoints(total: usize) -> Vec<usize> {
    let mut values = BTreeSet::new();
    values.insert(0);
    for value in [10, 20, 40, total] {
        if value <= total { values.insert(value); }
    }
    values.into_iter().collect()
}

/// Takes the most RECENT `train + holdout` qualifying commits, chronologically
/// ordered, and holds out the latest `holdout` of them. Recent history reflects
/// the codebase that exists now; the oldest commits describe one that is gone.
///
/// `skip` drops that many of the NEWEST qualifying commits before windowing —
/// use it to step over an unrepresentative recent stretch (e.g. a demolition
/// campaign) and land the holdout on real feature work.
fn chronological_split(commits: &[Commit], train: usize, holdout: usize, skip: usize) -> (Vec<Commit>, Vec<Commit>) {
    let kept = commits.len().saturating_sub(skip);
    let commits = &commits[..kept];
    let window = train.saturating_add(holdout);
    let start = commits.len().saturating_sub(window);
    let recent = &commits[start..];
    let holdout_len = holdout.min(recent.len());
    let split_at = recent.len() - holdout_len;
    (recent[..split_at].to_vec(), recent[split_at..].to_vec())
}

fn extract_commits(repo: &Path) -> EvalResult<Vec<Commit>> {
    let output = git(repo, &["log", "--no-merges", "--reverse", "--name-only", "--format=%x01%H%x00%aI%x00%s%x00%b%x00"])?;
    let mut commits = Vec::new();
    for record in output.split('\x01') {
        if record.trim().is_empty() { continue; }
        let fields: Vec<&str> = record.split('\0').collect();
        if fields.len() < 5 || fields[0].trim().is_empty() { continue; }
        let mut files = Vec::new();
        for line in fields[4].lines() {
            let file = line.trim();
            if !file.is_empty() { files.push(file.to_string()); }
        }
        let files = filter_files(files);
        if files.len() >= 3 {
            commits.push(Commit { hash: fields[0].trim().to_string(), timestamp: fields[1].to_string(), subject: fields[2].to_string(), body: fields[3].trim().to_string(), files });
        }
    }
    Ok(commits)
}

fn git(repo: &Path, args: &[&str]) -> EvalResult<String> {
    let output = Command::new("git").args(args).current_dir(repo).output()?;
    if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).to_string().into()); }
    Ok(String::from_utf8(output.stdout)?)
}

fn filter_files(files: Vec<String>) -> Vec<String> {
    let mut unique = BTreeSet::new();
    for file in files {
        let lower = file.to_ascii_lowercase();
        if lower.ends_with(".lock") || lower.starts_with("vendor/") || lower.starts_with("target/") || lower.contains("/generated/") { continue; }
        unique.insert(file);
    }
    unique.into_iter().collect()
}

fn sanitize_task(commit: &Commit) -> String {
    let mut kept = Vec::new();
    let text = format!("{} {}", commit.subject, commit.body);
    for token in text.split_whitespace() {
        let bare = token.trim_matches(|character: char| ",.;:()[]{}<>\"'".contains(character));
        if bare.is_empty() || is_path_token(bare) { continue; }
        kept.push(bare.to_string());
    }
    kept.join(" ")
}

fn is_path_token(token: &str) -> bool {
    let value = token.trim_matches('`');
    if value.contains('/') || value.contains('\\') || value.starts_with("./") { return true; }
    let Some((_, suffix)) = value.rsplit_once('.') else { return false; };
    if suffix.is_empty() || suffix.len() > 8 { return false; }
    for character in suffix.chars() {
        if !character.is_ascii_alphanumeric() { return false; }
    }
    true
}

fn train_commit(commit: &Commit, llm: &dyn LlmClient) -> EvalResult<()> {
    let task = sanitize_task(commit);
    let prompt = format!("{EXTRACT_PROMPT}TASK:\n{task}\n\nFILES TOUCHED:\n{}\n", commit.files.join("\n"));
    let response = llm.collect(&prompt)?;
    let claims = parse_claims(&response, &commit.files);
    let conn = memory::open_connection()?;
    let task_id = format!("predict-task-{}", commit.hash);
    memory::write_node(&conn, &task_id, &Node { node_type: "task".to_string(), body: task, date: commit.timestamp.clone() })?;
    for (file, why) in claims {
        let file_id = format!("predict-file-{file}");
        memory::write_node(&conn, &file_id, &Node { node_type: "file".to_string(), body: file.clone(), date: commit.timestamp.clone() })?;
        let edge = Edge { source: task_id.clone(), target: file_id, content: why, stability: 0.3, confidence: 0.5, importance: 0.5, date: commit.timestamp.clone(), subject: true, occurred_at: None, expired_at: None, memory_type: "semantic".to_string(), origin_session: "predict-eval".to_string(), provenance: commit.hash.clone(), ceiling: None, last_decayed_at: String::new() };
        memory::write_edge(&conn, &edge, &commit.hash)?;
    }
    Ok(())
}

fn parse_claims(response: &str, actual: &[String]) -> Vec<(String, String)> {
    let value = parse_json(response);
    let mut claims = Vec::new();
    let Some(claims_value) = value.get("claims") else { return claims; };
    let Some(items) = claims_value.as_array() else { return claims; };
    {
        for item in items {
            let Some(file_value) = item.get("file") else { continue; };
            let Some(file) = file_value.as_str() else { continue; };
            let mut is_actual = false;
            for path in actual { if path == file { is_actual = true; break; } }
            if is_actual {
                let why = match item.get("why") { Some(value) => value.as_str().unwrap_or_default(), None => "" };
                claims.push((file.to_string(), why.to_string()));
            }
        }
    }
    claims.sort(); claims.dedup(); claims
}

fn parse_json(response: &str) -> Value {
    let trimmed = response.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    serde_json::from_str(trimmed).unwrap_or(Value::Null)
}

fn score_holdout(commits: &[Commit], ctx: &RunCtx) -> EvalResult<Score> {
    let mut total = Score { precision: 0.0, recall: 0.0, f1_score: 0.0 };
    for commit in commits {
        let task = sanitize_task(commit);
        let claims = match ctx.embed { Some(client) => memory::edges_to_toon(&memory::select_semantic_edges(client, &task)), None => String::new() };
        let prompt = format!("{PREDICT_PROMPT}{task}\n\nRETRIEVED CLAIMS:\n{claims}");
        let prediction = parse_prediction(&ctx.llm.collect(&prompt)?);
        let score = score_paths(&prediction.paths, &commit.files);
        total.precision += score.precision; total.recall += score.recall; total.f1_score += score.f1_score;
        write_dump(ctx, &DumpRow { commit, task: &task, claims: &claims, prediction: &prediction, score: &score })?;
    }
    let count = commits.len() as f64;
    Ok(Score { precision: total.precision / count, recall: total.recall / count, f1_score: total.f1_score / count })
}

fn parse_prediction(response: &str) -> Prediction {
    let value = parse_json(response);
    let mut paths = Vec::new();
    let Some(paths_value) = value.get("paths") else { return Prediction { paths }; };
    let Some(items) = paths_value.as_array() else { return Prediction { paths }; };
    for item in items {
        if let Some(path) = item.as_str() && valid_prediction_path(path) { paths.push(path.to_string()); }
    }
    paths.sort(); paths.dedup(); Prediction { paths }
}

fn valid_prediction_path(path: &str) -> bool {
    if path.is_empty() || Path::new(path).is_absolute() { return false; }
    for part in path.split('/') { if part == ".." { return false; } }
    true
}

fn score_paths(predicted: &[String], actual: &[String]) -> Score {
    let predicted: BTreeSet<&String> = predicted.iter().collect();
    let actual: BTreeSet<&String> = actual.iter().collect();
    let hits = predicted.intersection(&actual).count() as f64;
    let precision = if predicted.is_empty() { 0.0 } else { hits / predicted.len() as f64 };
    let recall = if actual.is_empty() { 0.0 } else { hits / actual.len() as f64 };
    let f1_score = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    Score { precision, recall, f1_score }
}

fn write_dump(ctx: &RunCtx, row: &DumpRow) -> EvalResult<()> {
    let commit = row.commit;
    let prediction = row.prediction;
    let (task, claims, score) = (row.task, row.claims, row.score);
    let mode = ctx.mode;
    let name = format!("{}-{mode}-{}.md", ctx.checkpoint, &commit.hash[..commit.hash.len().min(12)]);
    let mut hits = Vec::new();
    for path in &prediction.paths { if commit.files.contains(path) { hits.push(path); } }
    let mut misses = Vec::new();
    for path in &commit.files { if !prediction.paths.contains(path) { misses.push(path); } }
    let text = format!("# {} ({mode})\n\nTask: {task}\n\nRetrieved claims:\n```toon\n{claims}\n```\n\nPredicted: {:?}\n\nActual: {:?}\n\nHits: {:?}\n\nMisses: {:?}\n\nPrecision: {:.3}\nRecall: {:.3}\nF1: {:.3}\n", commit.hash, prediction.paths, commit.files, hits, misses, score.precision, score.recall, score.f1_score);
    std::fs::write(ctx.output.join(name), text)?;
    Ok(())
}

fn append_csv(csv: &mut String, training: usize, mode: &str, score: &Score) {
    csv.push_str(&format!("{training},{mode},{:.6},{:.6},{:.6}\n", score.precision, score.recall, score.f1_score));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_removes_paths_and_filenames() {
        let commit = Commit { hash: "h".to_string(), timestamp: String::new(), subject: "fix retrieval in `embed.rs`".to_string(), body: "touch src/memory/db.rs and Cargo.toml".to_string(), files: Vec::new() };
        assert_eq!(sanitize_task(&commit), "fix retrieval in touch and");
    }

    #[test]
    fn filter_drops_lockfiles_vendor_and_generated() {
        let mut files = Vec::new();
        for file in ["a.rs", "b.rs", "c.rs", "Cargo.lock", "vendor/x.rs", "src/generated/x.rs"] { files.push(file.to_string()); }
        assert_eq!(filter_files(files), vec!["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn split_is_chronological_and_reproducible() {
        let mut commits = Vec::new();
        for index in 0..5 { commits.push(Commit { hash: index.to_string(), timestamp: String::new(), subject: String::new(), body: String::new(), files: Vec::new() }); }
        let first = chronological_split(&commits, 2, 2, 0);
        assert_eq!(first.0[0].hash, "1");
        assert_eq!(first.0[1].hash, "2");
        assert_eq!(first.1[1].hash, "4");
        assert_eq!(first, chronological_split(&commits, 2, 2, 0));
    }

    #[test]
    fn skip_drops_the_newest_commits_before_windowing() {
        let mut commits = Vec::new();
        for index in 0..5 { commits.push(Commit { hash: index.to_string(), timestamp: String::new(), subject: String::new(), body: String::new(), files: Vec::new() }); }
        let skipped = chronological_split(&commits, 2, 2, 1);
        assert_eq!(skipped.0[0].hash, "0");
        assert_eq!(skipped.1[1].hash, "3");
    }

    #[test]
    fn scorer_handles_empty_prediction() {
        assert_eq!(score_paths(&[], &["a".to_string()]).f1_score, 0.0);
        let score = score_paths(&["a".to_string()], &["a".to_string(), "b".to_string()]);
        assert!((score.precision - 1.0).abs() < f64::EPSILON);
        assert!((score.recall - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn extracts_a_fixture_git_history() {
        let root = std::env::temp_dir().join(format!("kratt-predict-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let Ok(()) = std::fs::create_dir_all(&root) else { panic!("fixture directory") };
        run_fixture_git(&root, &["init", "-q"]);
        run_fixture_git(&root, &["config", "user.email", "fixture@example.invalid"]);
        run_fixture_git(&root, &["config", "user.name", "fixture"]);
        let fixture_files = ["a.rs", "b.rs", "c.rs"];
        for file in fixture_files {
            let Ok(()) = std::fs::write(root.join(file), "one") else { panic!("fixture file") };
        }
        run_fixture_git(&root, &["add", "."]);
        run_fixture_git(&root, &["commit", "-qm", "add first files"]);
        let Ok(()) = std::fs::write(root.join("a.rs"), "two") else { panic!("fixture update") };
        run_fixture_git(&root, &["add", "a.rs"]);
        run_fixture_git(&root, &["commit", "-qm", "one file only"]);
        let Ok(commits) = extract_commits(&root) else { panic!("extract fixture") };
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].files, vec!["a.rs", "b.rs", "c.rs"]);
        let _ = std::fs::remove_dir_all(root);
    }

    fn run_fixture_git(root: &Path, arguments: &[&str]) {
        let Ok(status) = Command::new("git").args(arguments).current_dir(root).status() else { panic!("git fixture") };
        assert!(status.success());
    }
}
