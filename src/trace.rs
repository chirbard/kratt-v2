//! Opt-in forensic LLM transcripts.
//!
//! Traces contain raw private content: user messages, documents, prompts and
//! model output. Only exact configured secrets are scrubbed. Protection comes
//! from the trace directory and file permissions, not broad redaction.

use std::collections::HashMap;
use std::error::Error;
use std::fs::{self as filesystem, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{Chunk, LlmClient, TurnOutcome};
use crate::{paths, runlog, timeutil};

const SCHEMA_VERSION: u32 = 1;
const TRACE_FILE_CAP: usize = 100; // replay-tunable
const TRACE_BYTE_CAP: u64 = 512 * 1024 * 1024; // replay-tunable

#[derive(Clone)]
pub(crate) struct TraceContext {
    pub(crate) trace_id: String,
    pub(crate) parent_span: String,
    pub(crate) activity: String,
    pub(crate) seat: Option<String>,
    last_span: Arc<Mutex<String>>,
}

impl TraceContext {
    pub(crate) fn new(activity: &str) -> Self {
        if !enabled() {
            return Self {
                trace_id: String::new(),
                parent_span: String::new(),
                activity: String::new(),
                seat: None,
                last_span: Arc::new(Mutex::new(String::new())),
            };
        }
        let trace_id = new_id("trace");
        Self {
            parent_span: trace_id.clone(),
            trace_id,
            activity: activity.to_string(),
            seat: None,
            last_span: Arc::new(Mutex::new(String::new())),
        }
    }

    pub(crate) fn with_seat(&self, seat: &str) -> Self {
        let mut context = self.clone();
        context.seat = Some(seat.to_string());
        context
    }

    fn llm_span(&self) -> TraceSpan {
        let span = TraceSpan {
            context: self.clone(),
            span_id: new_id("span"),
        };
        if let Ok(mut last_span) = self.last_span.lock() {
            *last_span = span.span_id.clone();
        }
        span
    }

    fn child_span(&self) -> TraceSpan {
        let mut context = self.clone();
        if let Ok(last_span) = self.last_span.lock()
            && !last_span.is_empty()
        {
            context.parent_span = last_span.clone();
        }
        TraceSpan {
            context,
            span_id: new_id("span"),
        }
    }
}

#[derive(Clone)]
pub(crate) struct TraceSpan {
    context: TraceContext,
    span_id: String,
}

pub(crate) struct TracingClient {
    inner: Arc<dyn LlmClient>,
    context: TraceContext,
}

impl TracingClient {
    pub(crate) fn wrap(inner: Arc<dyn LlmClient>, context: TraceContext) -> Arc<dyn LlmClient> {
        if context.trace_id.is_empty() {
            return inner;
        }
        Arc::new(Self { inner, context })
    }
}

impl LlmClient for TracingClient {
    fn collect(&self, prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let (sender, receiver) = channel();
        self.stream(prompt.to_string(), None, sender);
        collect_chunks(receiver)
    }

    fn stream(&self, prompt: String, system: Option<String>, sender: Sender<Chunk>) {
        let span = self.context.llm_span();
        self.inner.stream_traced(prompt, system, sender, &span);
    }

    fn stream_turn(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        sender: &Sender<Chunk>,
    ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
        let span = self.context.llm_span();
        self.inner
            .stream_turn_traced(messages, tools, sender, &span)
    }
}

fn collect_chunks(
    receiver: std::sync::mpsc::Receiver<Chunk>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut content = String::new();
    let mut reasoning = String::new();
    for chunk in receiver {
        match chunk {
            Chunk::Content(text) => content.push_str(&text),
            Chunk::Reasoning(text) => reasoning.push_str(&text),
            Chunk::Err(err) => return Err(err.into()),
            Chunk::Done(_) => break,
            Chunk::ToolCall(_) | Chunk::MemoryLog { .. } | Chunk::Attribution(_) => {}
        }
    }
    if content.is_empty() {
        Ok(reasoning)
    } else {
        Ok(content)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct TraceRecord {
    schema_version: u32,
    #[serde(rename = "ts")]
    timestamp: String,
    trace_id: String,
    span_id: String,
    parent: String,
    sequence: u64,
    activity: String,
    provider: Option<String>,
    model: Option<String>,
    seat: Option<String>,
    kind: TraceKind,
    status: Option<String>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    duration_ms: Option<u128>,
    tool_call_id: Option<String>,
    content: Option<Value>,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TraceKind {
    Request,
    Response,
    ToolCall,
    ToolResult,
    #[allow(dead_code)]
    Compacted,
    Error,
    #[allow(dead_code)]
    PartialResponse,
}

impl TraceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Response => "response",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Compacted => "compacted",
            Self::Error => "error",
            Self::PartialResponse => "partial_response",
        }
    }
}

impl TraceRecord {
    fn new(span: &TraceSpan, kind: TraceKind) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            timestamp: timeutil::timestamp_now(),
            trace_id: span.context.trace_id.clone(),
            span_id: span.span_id.clone(),
            parent: span.context.parent_span.clone(),
            sequence: 0,
            activity: span.context.activity.clone(),
            provider: None,
            model: None,
            seat: span.context.seat.clone(),
            kind,
            status: None,
            finish_reason: None,
            usage: None,
            duration_ms: None,
            tool_call_id: None,
            content: None,
            error: None,
        }
    }
}

pub(crate) fn request(span: &TraceSpan, provider: &str, model: &str, body: Value) {
    let mut record = TraceRecord::new(span, TraceKind::Request);
    record.provider = Some(provider.to_string());
    record.model = Some(model.to_string());
    record.status = Some("started".to_string());
    record.content = Some(body);
    emit(record);
}

pub(crate) struct ResponseRecord {
    pub(crate) content: Value,
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: Value,
    pub(crate) duration_ms: u128,
}

pub(crate) fn response(span: &TraceSpan, provider: &str, model: &str, data: ResponseRecord) {
    let mut record = TraceRecord::new(span, TraceKind::Response);
    record.provider = Some(provider.to_string());
    record.model = Some(model.to_string());
    record.status = Some("ok".to_string());
    record.finish_reason = data.finish_reason;
    record.usage = Some(data.usage);
    record.duration_ms = Some(data.duration_ms);
    record.content = Some(data.content);
    emit(record);
}

pub(crate) fn error(span: &TraceSpan, provider: &str, model: &str, err: &str, duration_ms: u128) {
    let mut record = TraceRecord::new(span, TraceKind::Error);
    record.provider = Some(provider.to_string());
    record.model = Some(model.to_string());
    record.status = Some("error".to_string());
    record.duration_ms = Some(duration_ms);
    record.error = Some(err.to_string());
    emit(record);
}

pub(crate) fn tool(context: &TraceContext, kind: &str, call_id: &str, content: Value) {
    if context.trace_id.is_empty() {
        return;
    }
    let span = context.child_span();
    let record_kind = if kind == "tool_call" {
        TraceKind::ToolCall
    } else {
        TraceKind::ToolResult
    };
    let mut record = TraceRecord::new(&span, record_kind);
    record.tool_call_id = Some(call_id.to_string());
    record.content = Some(content);
    emit(record);
}

struct WriteRequest {
    record: TraceRecord,
    done: Sender<()>,
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KRATT_TRACE").is_ok_and(|value| value == "1"))
}

fn emit(record: TraceRecord) {
    if !enabled() {
        return;
    }
    let Some(sender) = writer_sender() else {
        return;
    };
    let (done, receiver) = channel();
    if sender.send(WriteRequest { record, done }).is_err() {
        log_writer_error_once("trace writer stopped");
        return;
    }
    // Forensic mode may block on disk so a completed model call has a durable transcript.
    let _ = receiver.recv();
}

fn writer_sender() -> Option<&'static Sender<WriteRequest>> {
    static WRITER: OnceLock<Sender<WriteRequest>> = OnceLock::new();
    if let Some(sender) = WRITER.get() {
        return Some(sender);
    }
    let (sender, receiver) = channel();
    std::thread::spawn(move || writer_loop(receiver));
    let _ = WRITER.set(sender);
    WRITER.get()
}

fn writer_loop(receiver: std::sync::mpsc::Receiver<WriteRequest>) {
    let mut sequences: HashMap<String, u64> = HashMap::new();
    for mut request in receiver {
        assign_sequence(&mut request.record, &mut sequences);
        if let Err(err) = write_record(&request.record) {
            log_writer_error_once(&err.to_string());
        }
        let _ = request.done.send(());
    }
}

fn assign_sequence(record: &mut TraceRecord, sequences: &mut HashMap<String, u64>) {
    let sequence = sequences.entry(record.trace_id.clone()).or_insert(0);
    *sequence += 1;
    record.sequence = *sequence;
}

fn write_record(record: &TraceRecord) -> Result<(), Box<dyn Error + Send + Sync>> {
    let dir = trace_dir();
    write_record_to(record, &dir)
}

fn write_record_to(record: &TraceRecord, dir: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    filesystem::create_dir_all(dir)?;
    set_dir_permissions(dir)?;
    let path = dir.join(format!("{}.jsonl", safe_id(&record.trace_id)));
    let is_new = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    set_file_permissions(&path)?;
    let mut scrubbed = record.clone();
    redact_record(&mut scrubbed);
    serde_json::to_writer(&mut file, &scrubbed)?;
    file.write_all(b"\n")?;
    file.flush()?;
    if is_new {
        prune_traces(dir)?;
    }
    Ok(())
}

fn trace_dir() -> PathBuf {
    PathBuf::from(paths::data_file("traces"))
}

fn safe_id(trace_id: &str) -> String {
    let mut safe = String::new();
    for character in trace_id.chars() {
        if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
            safe.push(character);
        }
    }
    safe
}

fn new_id(prefix: &str) -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let nanos = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(value) => value.as_nanos(),
        Err(_) => 0,
    };
    let serial = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{nanos}-{serial}", std::process::id())
}

fn configured_secrets() -> Vec<String> {
    let mut secrets = Vec::new();
    for name in [
        "OPENROUTER_API_KEY",
        "TAVILY_API_KEY",
        "TURSO_AUTH_TOKEN",
        "TELEGRAM_BOT_TOKEN",
    ] {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            secrets.push(value);
        }
    }
    secrets
}

fn redact_record(record: &mut TraceRecord) {
    let secrets = configured_secrets();
    if let Some(content) = &mut record.content {
        redact_value(content, &secrets);
    }
    if let Some(error) = &mut record.error {
        *error = redact_text(error, &secrets);
    }
}

fn redact_value(value: &mut Value, secrets: &[String]) {
    match value {
        Value::String(text) => *text = redact_text(text, secrets),
        Value::Array(items) => {
            for item in items {
                redact_value(item, secrets);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                redact_value(item, secrets);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_text(text: &str, secrets: &[String]) -> String {
    let mut scrubbed = text.to_string();
    for secret in secrets {
        scrubbed = scrubbed.replace(secret, "[REDACTED]");
    }
    scrubbed
}

fn prune_traces(dir: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut files = Vec::new();
    for entry in filesystem::read_dir(dir)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|ext| ext == "jsonl") {
            let metadata = entry.metadata()?;
            files.push((metadata.modified()?, metadata.len(), entry.path()));
        }
    }
    files.sort_by_key(|(modified, _, _)| *modified);
    let mut bytes = 0u64;
    for (_, size, _) in &files {
        bytes += size;
    }
    while files.len() > TRACE_FILE_CAP || bytes > TRACE_BYTE_CAP {
        let (_, size, path) = files.remove(0);
        bytes = bytes.saturating_sub(size);
        filesystem::remove_file(path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_dir_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    filesystem::set_permissions(path, filesystem::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn set_dir_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_file_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    filesystem::set_permissions(path, filesystem::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_file_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn log_writer_error_once(message: &str) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        runlog::log("trace", &format!("writer failed: {message}"));
    }
}

pub(crate) fn view(trace_id: &str) -> Result<(), Box<dyn Error>> {
    let path = trace_dir().join(format!("{}.jsonl", safe_id(trace_id)));
    let file = OpenOptions::new().read(true).open(path)?;
    let mut records = Vec::new();
    for line in BufReader::new(file).lines() {
        records.push(serde_json::from_str::<TraceRecord>(&line?)?);
    }
    records.sort_by_key(|record| record.sequence);
    print!("{}", render_tree(&records));
    Ok(())
}

fn render_tree(records: &[TraceRecord]) -> String {
    let mut output = String::new();
    let mut depths: HashMap<String, usize> = HashMap::new();
    if let Some(first) = records.first() {
        output.push_str(&format!("{} {}\n", first.activity, first.trace_id));
        depths.insert(first.trace_id.clone(), 0);
    }
    for record in records {
        let parent_depth = match depths.get(&record.parent) {
            Some(depth) => *depth,
            None => 0,
        };
        let depth = parent_depth + 1;
        depths.insert(record.span_id.clone(), depth);
        output.push_str(&"  ".repeat(depth));
        output.push_str(&format!(
            "{:04} {} {}",
            record.sequence,
            record.kind.as_str(),
            record.span_id
        ));
        if let Some(call_id) = &record.tool_call_id {
            output.push_str(&format!(" tool={call_id}"));
        }
        if let Some(status) = &record.status {
            output.push_str(&format!(" [{status}]"));
        }
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};

    use super::{
        ResponseRecord, TraceContext, TraceKind, TraceRecord, TracingClient, assign_sequence,
        enabled, new_id, redact_value, render_tree, request, response, trace_dir, write_record_to,
    };

    struct PassClient;

    impl LlmClient for PassClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            Ok("ok".to_string())
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            let _ = sender.send(Chunk::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }));
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            Ok(TurnOutcome::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }))
        }
    }

    fn record(context: &TraceContext, kind: TraceKind) -> TraceRecord {
        TraceRecord::new(&context.llm_span(), kind)
    }

    fn test_context(activity: &str) -> TraceContext {
        let trace_id = new_id("trace");
        TraceContext {
            parent_span: trace_id.clone(),
            trace_id,
            activity: activity.to_string(),
            seat: None,
            last_span: Arc::new(Mutex::new(String::new())),
        }
    }

    #[test]
    fn writer_sequence_is_global_within_each_trace() {
        let context = test_context("dream");
        let mut first = record(&context, TraceKind::Request);
        let mut second = record(&context, TraceKind::Request);
        let mut third = record(&context, TraceKind::Response);
        let mut sequences = HashMap::new();
        assign_sequence(&mut first, &mut sequences);
        assign_sequence(&mut second, &mut sequences);
        assign_sequence(&mut third, &mut sequences);
        assert_eq!([first.sequence, second.sequence, third.sequence], [1, 2, 3]);
    }

    #[test]
    fn tool_spans_attach_to_the_preceding_model_call() {
        let context = test_context("chat");
        let model_span = context.llm_span();
        let tool_span = context.child_span();
        assert_eq!(tool_span.context.parent_span, model_span.span_id);
    }

    #[test]
    fn tracing_off_returns_the_original_client_and_keeps_calls_unchanged() {
        if std::env::var("KRATT_TRACE").is_ok_and(|value| value == "1") {
            return;
        }
        let inner: Arc<dyn LlmClient> = Arc::new(PassClient);
        let wrapped = TracingClient::wrap(Arc::clone(&inner), TraceContext::new("test"));
        assert!(Arc::ptr_eq(&inner, &wrapped));
        assert_eq!(wrapped.collect("prompt").unwrap_or_default(), "ok");
    }

    #[test]
    fn enabled_writer_persists_request_and_response_shape() {
        if !enabled() {
            return;
        }
        let context = TraceContext::new("test");
        let span = context.llm_span();
        request(
            &span,
            "provider",
            "model",
            json!({"messages": [], "tools": [{"name": "search"}]}),
        );
        response(
            &span,
            "provider",
            "model",
            ResponseRecord {
                content: json!({"content": "answer", "reasoning": "", "tool_calls": []}),
                finish_reason: Some("stop".to_string()),
                usage: json!({"prompt_tokens": 3, "completion_tokens": 1}),
                duration_ms: 4,
            },
        );
        let path = trace_dir().join(format!("{}.jsonl", context.trace_id));
        let raw = std::fs::read_to_string(&path).unwrap_or_default();
        let mut lines = raw.lines();
        let request_line = lines.next().unwrap_or_default();
        let response_line = lines.next().unwrap_or_default();
        assert!(request_line.contains("\"kind\":\"request\""));
        assert!(request_line.contains("\"tools\""));
        assert!(response_line.contains("\"kind\":\"response\""));
        assert!(response_line.contains("\"finish_reason\":\"stop\""));
        assert!(response_line.contains("\"prompt_tokens\":3"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_failure_is_isolated_from_a_model_call() {
        let context = test_context("test");
        let record = record(&context, TraceKind::Request);
        let blocker = std::env::temp_dir().join(new_id("trace-blocker"));
        let created = std::fs::File::create(&blocker);
        assert!(created.is_ok());
        assert!(write_record_to(&record, &blocker).is_err());
        let client = PassClient;
        assert_eq!(client.collect("still runs").unwrap_or_default(), "ok");
        let _ = std::fs::remove_file(blocker);
    }

    #[test]
    fn redaction_scrubs_only_exact_secrets() {
        let mut value = json!({
            "prompt": "secret-value https://example.com sk-ordinary-looking",
            "nested": ["secret-value-suffix"]
        });
        redact_value(&mut value, &["secret-value".to_string()]);
        assert_eq!(
            value["prompt"],
            "[REDACTED] https://example.com sk-ordinary-looking"
        );
        assert_eq!(value["nested"][0], "[REDACTED]-suffix");
    }

    #[test]
    fn viewer_orders_and_indents_a_small_trace_tree() {
        let context = test_context("chat");
        let mut response = record(&context, TraceKind::Response);
        response.sequence = 2;
        let mut request = record(&context, TraceKind::Request);
        request.sequence = 1;
        response.parent = request.span_id.clone();
        let rendered = render_tree(&[request, response]);
        let lines: Vec<&str> = rendered.lines().collect();
        assert!(lines[0].starts_with("chat trace-"));
        assert!(lines[1].starts_with("  0001 request"));
        assert!(lines[2].starts_with("    0002 response"));
    }
}
