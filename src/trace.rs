//! Opt-in forensic LLM transcripts.
//!
//! Traces contain raw private content: user messages, documents, prompts and
//! model output. Only exact configured secrets are scrubbed. Protection comes
//! from the trace directory and file permissions, not broad redaction.

use std::collections::HashMap;
use std::error::Error;
use std::fs::{self as filesystem, OpenOptions};
use std::io::Write;
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
const TRACE_DAY_CAP: usize = 100; // replay-tunable
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

#[derive(Default)]
struct WriterState {
    sequences: HashMap<String, u64>,
    flows: HashMap<String, FlowLayout>,
}

struct FlowLayout {
    day: String,
    folder: String,
    start_time: String,
    spans: HashMap<String, SpanLayout>,
    next_span: usize,
}

#[derive(Clone)]
struct SpanLayout {
    index: usize,
    agent: String,
    start_time: String,
    file_stem: String,
}

struct RecordPaths {
    day_dir: PathBuf,
    flow_dir: PathBuf,
    jsonl: PathBuf,
    human: PathBuf,
    flow_start: String,
    span: SpanLayout,
}

struct TraceTime {
    day: String,
    compact: String,
    clock: String,
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
    let mut state = WriterState::default();
    for mut request in receiver {
        assign_sequence(&mut request.record, &mut state.sequences);
        if let Err(err) = write_record(&request.record, &mut state) {
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

fn write_record(
    record: &TraceRecord,
    state: &mut WriterState,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let dir = trace_dir();
    write_record_to(record, &dir, state)
}

fn write_record_to(
    record: &TraceRecord,
    dir: &Path,
    state: &mut WriterState,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut scrubbed = record.clone();
    redact_record(&mut scrubbed);
    let paths = record_paths(&scrubbed, dir, state);
    prepare_trace_dirs(dir, &paths)?;
    append_jsonl(&paths.jsonl, &scrubbed)?;
    append_human(&paths, &scrubbed)?;
    prune_traces(dir)?;
    Ok(())
}

fn prepare_trace_dirs(root: &Path, paths: &RecordPaths) -> std::io::Result<()> {
    filesystem::create_dir_all(&paths.flow_dir)?;
    set_dir_permissions(root)?;
    set_dir_permissions(&paths.day_dir)?;
    set_dir_permissions(&paths.flow_dir)
}

fn append_jsonl(path: &Path, record: &TraceRecord) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    set_file_permissions(path)?;
    serde_json::to_writer(&mut file, record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn append_human(paths: &RecordPaths, record: &TraceRecord) -> std::io::Result<()> {
    let is_new = !paths.human.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.human)?;
    set_file_permissions(&paths.human)?;
    if is_new {
        file.write_all(render_file_header(paths, record).as_bytes())?;
    }
    file.write_all(render_human_record(record).as_bytes())?;
    file.flush()
}

fn trace_dir() -> PathBuf {
    PathBuf::from(paths::data_file("traces"))
}

fn record_paths(record: &TraceRecord, root: &Path, state: &mut WriterState) -> RecordPaths {
    let flow = state
        .flows
        .entry(record.trace_id.clone())
        .or_insert_with(|| FlowLayout::new(record));
    let span_key = routing_span_id(record).to_string();
    let span = match flow.spans.entry(span_key) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            flow.next_span += 1;
            entry.insert(SpanLayout::new(record, flow.next_span)).clone()
        }
    };
    let day_dir = root.join(&flow.day);
    let flow_dir = day_dir.join(&flow.folder);
    RecordPaths {
        jsonl: flow_dir.join(format!("{}.jsonl", span.file_stem)),
        human: flow_dir.join(format!("{}.txt", span.file_stem)),
        day_dir,
        flow_dir,
        flow_start: flow.start_time.clone(),
        span,
    }
}

impl FlowLayout {
    fn new(record: &TraceRecord) -> Self {
        let time = TraceTime::from_timestamp(&record.timestamp);
        let activity = safe_component(&record.activity, "trace");
        let folder = format!(
            "{}-{}-{}",
            time.compact,
            activity,
            short_id(&record.trace_id)
        );
        Self {
            day: time.day,
            folder,
            start_time: time.compact,
            spans: HashMap::new(),
            next_span: 0,
        }
    }
}

impl SpanLayout {
    fn new(record: &TraceRecord, index: usize) -> Self {
        let time = TraceTime::from_timestamp(&record.timestamp);
        let agent = trace_agent(record);
        Self {
            index,
            file_stem: format!("{index:02}-{agent}-{}", time.compact),
            agent,
            start_time: time.compact,
        }
    }
}

impl TraceTime {
    fn from_timestamp(timestamp: &str) -> Self {
        let day = timestamp.get(0..10).unwrap_or("unknown-day").to_string();
        let hour = timestamp.get(11..13).unwrap_or("00");
        let minute = timestamp.get(14..16).unwrap_or("00");
        let second = timestamp.get(17..19).unwrap_or("00");
        Self {
            day,
            compact: format!("{hour}{minute}{second}"),
            clock: format!("{hour}:{minute}:{second}"),
        }
    }
}

fn routing_span_id(record: &TraceRecord) -> &str {
    if matches!(record.kind, TraceKind::ToolCall | TraceKind::ToolResult)
        && !record.parent.is_empty()
    {
        &record.parent
    } else {
        &record.span_id
    }
}

fn trace_agent(record: &TraceRecord) -> String {
    let agent = match record.seat.as_deref() {
        Some(seat) => seat,
        None if record.activity == "chat" => "answering",
        None => &record.activity,
    };
    safe_component(agent, "agent")
}

fn safe_component(value: &str, fallback: &str) -> String {
    let mut safe = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
            safe.push(character.to_ascii_lowercase());
        } else if !safe.ends_with('-') {
            safe.push('-');
        }
    }
    let trimmed = safe.trim_matches('-');
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

fn short_id(trace_id: &str) -> String {
    let safe = safe_id(trace_id);
    let mut reversed: String = safe.chars().rev().take(6).collect();
    if reversed.is_empty() {
        return "trace".to_string();
    }
    reversed = reversed.chars().rev().collect();
    reversed.to_ascii_lowercase()
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

fn render_file_header(paths: &RecordPaths, record: &TraceRecord) -> String {
    let mut identity = format!("{} · seat={}", record.activity, paths.span.agent);
    if let Some(model) = &record.model {
        if model.contains('/') {
            identity.push_str(&format!(" · {model}"));
        } else if let Some(provider) = &record.provider {
            identity.push_str(&format!(" · {provider}/{model}"));
        } else {
            identity.push_str(&format!(" · {model}"));
        }
    }
    format!(
        "════════════════════════════════════════\n{identity}\nflow {}  ·  span {}  ·  round {:02}\n════════════════════════════════════════\n\n",
        paths.flow_start, paths.span.start_time, paths.span.index
    )
}

fn render_human_record(record: &TraceRecord) -> String {
    match record.kind {
        TraceKind::Request => render_request(record),
        TraceKind::Response => render_response(record),
        TraceKind::ToolCall => render_tool_call(record),
        TraceKind::ToolResult => render_tool_result(record),
        TraceKind::Error => render_error(record),
        TraceKind::Compacted | TraceKind::PartialResponse => render_generic(record),
    }
}

fn render_request(record: &TraceRecord) -> String {
    let mut output = record_header("REQUEST", record, "");
    let Some(content) = &record.content else {
        output.push_str("(no request body)\n\n");
        return output;
    };
    if let Some(messages) = array_field(content, "messages") {
        render_messages(&mut output, messages);
    } else {
        output.push_str(&readable_value(content));
        output.push_str("\n\n");
    }
    render_tools_summary(&mut output, content.get("tools"));
    output
}

fn render_messages(output: &mut String, messages: &[Value]) {
    for message in messages {
        let role = string_field(message, "role").unwrap_or("message");
        output.push_str(&format!("[{role}]\n"));
        match message.get("content") {
            Some(content) => output.push_str(&readable_value(content)),
            None => output.push_str("(no content)"),
        }
        if let Some(calls) = array_field(message, "tool_calls") {
            output.push_str("\n[tool_calls]\n");
            render_calls(output, calls);
        }
        output.push_str("\n\n");
    }
}

fn render_tools_summary(output: &mut String, tools: Option<&Value>) {
    let Some(Value::Array(tools)) = tools else {
        return;
    };
    if tools.is_empty() {
        return;
    }
    let mut names = Vec::new();
    for tool in tools.iter().take(6) {
        names.push(tool_name(tool).unwrap_or("unnamed"));
    }
    output.push_str("tools: ");
    output.push_str(&names.join(", "));
    output.push_str(&format!("   ({} total)\n\n", tools.len()));
}

fn render_response(record: &TraceRecord) -> String {
    let details = response_details(record);
    let mut output = record_header("RESPONSE", record, &details);
    let Some(content) = &record.content else {
        output.push_str("(empty response)\n\n");
        return output;
    };
    render_response_content(&mut output, content);
    output.push('\n');
    output
}

fn render_response_content(output: &mut String, content: &Value) {
    let Some(object) = content.as_object() else {
        output.push_str(&readable_value(content));
        output.push('\n');
        return;
    };
    if let Some(value) = object.get("content")
        && !value.is_null()
    {
        output.push_str(&readable_value(value));
        output.push('\n');
    }
    if let Some(reasoning) = object.get("reasoning")
        && !reasoning.as_str().unwrap_or("").is_empty()
    {
        output.push_str("\n[reasoning]\n");
        output.push_str(&readable_value(reasoning));
        output.push('\n');
    }
    if let Some(calls) = array_field(content, "tool_calls")
        && !calls.is_empty()
    {
        output.push_str("\n[tool_calls]\n");
        render_calls(output, calls);
    }
}

fn response_details(record: &TraceRecord) -> String {
    let mut details = Vec::new();
    if let Some(reason) = &record.finish_reason {
        details.push(format!("finish={reason}"));
    }
    if let Some(tokens) = usage_tokens(record.usage.as_ref()) {
        details.push(format!("{tokens} tok"));
    }
    if let Some(duration) = record.duration_ms {
        details.push(format_duration(duration));
    }
    details.join(", ")
}

fn usage_tokens(usage: Option<&Value>) -> Option<u64> {
    let usage = usage?;
    let prompt = u64_field(usage, "prompt_tokens").unwrap_or(0);
    let completion = u64_field(usage, "completion_tokens").unwrap_or(0);
    if prompt == 0 && completion == 0 {
        None
    } else {
        Some(prompt + completion)
    }
}

fn format_duration(duration_ms: u128) -> String {
    if duration_ms < 1_000 {
        format!("{duration_ms}ms")
    } else {
        format!("{:.1}s", duration_ms as f64 / 1_000.0)
    }
}

fn render_tool_call(record: &TraceRecord) -> String {
    let mut output = record_header("TOOL CALL", record, tool_id(record));
    if let Some(content) = &record.content {
        output.push_str(&render_call(content));
    } else {
        output.push_str("(no tool call body)");
    }
    output.push_str("\n\n");
    output
}

fn render_tool_result(record: &TraceRecord) -> String {
    let mut output = record_header("TOOL RESULT", record, tool_id(record));
    let Some(content) = &record.content else {
        output.push_str("(no tool result)\n\n");
        return output;
    };
    if let Some(name) = string_field(content, "name") {
        output.push_str(&format!("{name}\n"));
    }
    match content.get("result") {
        Some(result) => output.push_str(&readable_value(result)),
        None => output.push_str(&readable_value(content)),
    }
    output.push_str("\n\n");
    output
}

fn render_error(record: &TraceRecord) -> String {
    let details = match record.duration_ms {
        Some(duration) => format_duration(duration),
        None => String::new(),
    };
    let mut output = record_header("ERROR", record, &details);
    output.push_str(record.error.as_deref().unwrap_or("unknown error"));
    output.push_str("\n\n");
    output
}

fn render_generic(record: &TraceRecord) -> String {
    let mut output = record_header(record.kind.as_str(), record, "");
    if let Some(content) = &record.content {
        output.push_str(&readable_value(content));
    }
    output.push_str("\n\n");
    output
}

fn record_header(label: &str, record: &TraceRecord, details: &str) -> String {
    let time = TraceTime::from_timestamp(&record.timestamp).clock;
    let label = label.to_ascii_uppercase();
    if details.is_empty() {
        format!("── {label}  {time} ────────────────────\n")
    } else {
        format!("── {label}  {time}  ({details}) ──\n")
    }
}

fn render_calls(output: &mut String, calls: &[Value]) {
    for call in calls {
        output.push_str(&render_call(call));
        output.push('\n');
    }
}

fn render_call(call: &Value) -> String {
    let name = tool_name(call).unwrap_or("tool");
    let arguments = tool_arguments(call);
    format!("{name}({})", readable_value(arguments.unwrap_or(&Value::Null)))
}

fn tool_name(value: &Value) -> Option<&str> {
    if let Some(name) = string_field(value, "name") {
        return Some(name);
    }
    let function = value.get("function")?;
    string_field(function, "name")
}

fn tool_arguments(value: &Value) -> Option<&Value> {
    if let Some(arguments) = value.get("arguments") {
        return Some(arguments);
    }
    if let Some(arguments) = value.get("args") {
        return Some(arguments);
    }
    let function = value.get("function")?;
    function.get("arguments")
}

fn tool_id(record: &TraceRecord) -> &str {
    record.tool_call_id.as_deref().unwrap_or("")
}

fn string_field<'value>(value: &'value Value, key: &str) -> Option<&'value str> {
    match value.get(key) {
        Some(field) => field.as_str(),
        None => None,
    }
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    match value.get(key) {
        Some(field) => field.as_u64(),
        None => None,
    }
}

fn array_field<'value>(value: &'value Value, key: &str) -> Option<&'value [Value]> {
    match value.get(key) {
        Some(Value::Array(items)) => Some(items),
        _ => None,
    }
}

fn readable_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        _ => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
    }
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
    let mut days = Vec::new();
    for entry in filesystem::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && is_day_folder(&entry.file_name()) {
            let size = directory_size(&entry.path())?;
            days.push((entry.file_name(), size, entry.path()));
        }
    }
    days.sort_by(|left, right| left.0.cmp(&right.0));
    let mut bytes = 0u64;
    for (_, size, _) in &days {
        bytes += size;
    }
    while days.len() > TRACE_DAY_CAP || bytes > TRACE_BYTE_CAP {
        let (_, size, path) = days.remove(0);
        bytes = bytes.saturating_sub(size);
        filesystem::remove_dir_all(path)?;
    }
    Ok(())
}

fn is_day_folder(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy();
    if name.len() != 10 || name.as_bytes()[4] != b'-' || name.as_bytes()[7] != b'-' {
        return false;
    }
    for (index, character) in name.chars().enumerate() {
        if index != 4 && index != 7 && !character.is_ascii_digit() {
            return false;
        }
    }
    true
}

fn directory_size(dir: &Path) -> std::io::Result<u64> {
    let mut size = 0u64;
    for entry in filesystem::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            size += directory_size(&entry.path())?;
        } else {
            size += entry.metadata()?.len();
        }
    }
    Ok(size)
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

pub(crate) fn view(target: &str) -> Result<(), Box<dyn Error>> {
    let path = resolve_view_path(target)?;
    if flow_text_files(&path)?.is_empty() {
        list_day_flows(&path)?;
    } else {
        print_flow(&path)?;
    }
    Ok(())
}

fn resolve_view_path(target: &str) -> Result<PathBuf, Box<dyn Error>> {
    let direct = PathBuf::from(target);
    if direct.is_dir() {
        return Ok(direct);
    }
    let root = trace_dir();
    let nested = root.join(target);
    if nested.is_dir() {
        return Ok(nested);
    }
    for entry in filesystem::read_dir(&root)? {
        let candidate = entry?.path().join(target);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("trace day or flow not found: {target}"),
    )
    .into())
}

fn flow_text_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in filesystem::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "txt") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn list_day_flows(day: &Path) -> std::io::Result<()> {
    let mut flows = Vec::new();
    for entry in filesystem::read_dir(day)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            flows.push(entry.file_name());
        }
    }
    flows.sort();
    for flow in flows {
        println!("{}", flow.to_string_lossy());
    }
    Ok(())
}

fn print_flow(flow: &Path) -> std::io::Result<()> {
    for path in flow_text_files(flow)? {
        let content = filesystem::read_to_string(path)?;
        print!("{content}");
        if !content.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
