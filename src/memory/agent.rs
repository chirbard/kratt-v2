use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient, ToolCallRequest};
use crate::prompts::{CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, build_agent_prompt};
use crate::runlog;
use turso::Connection;

use super::db;
use super::decay;
use super::graph::{Edge, Node, graph_to_json, normalize_id};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

pub enum MemoryEvent {
    Log { phase: &'static str, text: String },
    Append { phase: &'static str, text: String },
    Done,
    MaintenanceDone,
}

struct AgentRun<'run> {
    llm: Arc<dyn LlmClient>,
    user_msg: &'run str,
    assistant_msg: &'run str,
    sender: &'run std::sync::mpsc::Sender<MemoryEvent>,
    user_message_id: &'run str,
}

pub(crate) enum ToolCall {
    UpsertNode {
        name: String,
        node_type: String,
        body: String,
        date: String,
    },
    UpsertEdge {
        source: String,
        target: String,
        content: String,
        stability: f64,
        confidence: f64,
        importance: f64,
        subject: bool,
        date: String,
    },
    DeleteNode {
        name: String,
    },
    DeleteEdge {
        source: String,
        target: String,
    },
    ExpireEdge {
        source: String,
        target: String,
        rewritten_content: String,
    },
}

pub fn run_memory_agent(
    llm: Arc<dyn LlmClient>,
    user_msg: &str,
    assistant_msg: &str,
    user_message_id: &str,
    sender: std::sync::mpsc::Sender<MemoryEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let agent_run = AgentRun {
        llm,
        user_msg,
        assistant_msg,
        sender: &sender,
        user_message_id,
    };
    let result = agent_run.run();
    db::sync_push_best_effort("memory agent");
    let _ = sender.send(MemoryEvent::Done);
    result
}

impl<'run> AgentRun<'run> {
    fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _write_guard = db::acquire_write_lock();
        let conn = db::open_connection()?;
        let mut graph = db::load_graph();
        let decay_ran = decay::apply_decay_if_needed(&mut graph);
        if decay_ran && let Err(err) = db::save_graph(&graph) {
            self.send_log("memory", format!("error: {err}"));
            return Err(err);
        }

        let pre_extract_graph = graph.clone();
        let pre_extract_toon = toon::encode(&graph_to_json(&pre_extract_graph), None);
        let extract_prompt = build_agent_prompt(
            EXTRACT_EXPLANATION,
            &pre_extract_toon,
            self.user_msg,
            self.assistant_msg,
            None,
        );
        self.run_pass(&conn, "extract", extract_prompt)?;

        let post_extract_graph = db::load_graph();
        let diff = db::graph_diff_toon(&pre_extract_graph, &post_extract_graph);
        let consolidate_prompt = build_agent_prompt(
            CONSOLIDATE_EXPLANATION,
            &pre_extract_toon,
            self.user_msg,
            self.assistant_msg,
            if diff.is_empty() { None } else { Some(&diff) },
        );
        self.run_pass(&conn, "consolidate", consolidate_prompt)?;

        let final_graph = db::load_graph();
        self.send_log(
            "memory",
            format!(
                "saved — {} nodes, {} edges",
                final_graph.nodes.len(),
                final_graph.edges.len()
            ),
        );
        Ok(())
    }

    fn run_pass(
        &self,
        conn: &Connection,
        phase: &'static str,
        prompt: String,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.send_log(phase, "calling LLM…".to_string());
        let call_text = match stream_collect(
            Arc::clone(&self.llm),
            prompt,
            |text| self.send_append("think", text),
            |text| self.send_append("debug", text),
        ) {
            Ok(call_text) => call_text,
            Err(err) => {
                self.send_log(phase, format!("error: {err}"));
                return Err(err);
            }
        };
        let tool_calls = parse_tool_calls(&call_text);
        if tool_calls.is_empty() {
            self.send_log(phase, "no tool calls".to_string());
        } else {
            self.send_log(phase, format!("{} tool calls", tool_calls.len()));
        }
        if let Err(err) = apply_tool_calls(self, conn, phase, tool_calls) {
            self.send_log(phase, format!("error: {err}"));
            return Err(err);
        }
        self.send_log(phase, "done".to_string());
        Ok(())
    }

    fn send_log(&self, phase: &'static str, text: String) {
        runlog::log(phase, &text);
        let _ = self.sender.send(MemoryEvent::Log { phase, text });
    }

    fn send_append(&self, phase: &'static str, text: String) {
        let _ = self.sender.send(MemoryEvent::Append { phase, text });
    }
}

pub(super) fn stream_collect(
    llm: Arc<dyn LlmClient>,
    prompt: String,
    on_reasoning: impl Fn(String),
    on_content: impl Fn(String),
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let (chunk_tx, chunk_rx) = std::sync::mpsc::channel::<Chunk>();
    std::thread::spawn(move || llm.stream(prompt, None, chunk_tx));
    let mut reasoning_full = String::new();
    let mut content = String::new();
    let mut err: Option<Box<dyn Error + Send + Sync>> = None;
    for chunk in chunk_rx {
        match chunk {
            Chunk::Reasoning(text) => {
                reasoning_full.push_str(&text);
                on_reasoning(text);
            }
            Chunk::Content(text) => {
                content.push_str(&text);
                on_content(text);
            }
            Chunk::Done(_) => break,
            Chunk::Err(error) => {
                err = Some(error.into());
                break;
            }
            Chunk::ToolCall(_) | Chunk::MemoryLog { .. } | Chunk::Attribution(_) => {}
        }
    }
    if let Some(error) = err {
        return Err(error);
    }
    if !content.is_empty() {
        return Ok(content);
    }
    Ok(reasoning_full)
}

pub(crate) fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let stripped = strip_markdown_fences(text);
    let json_value = match serde_json::from_str::<serde_json::Value>(stripped) {
        Ok(json_value) => json_value,
        // Models sometimes wrap the JSON in prose or channel markers — retry
        // on the outermost brace span. If that also fails the response is
        // likely truncated mid-array: salvage the complete tool objects
        // rather than discarding the whole batch.
        Err(_) => match parse_embedded_json(stripped) {
            Some(json_value) => json_value,
            None => return salvage_tool_calls(stripped),
        },
    };
    if let Some(tools) = tool_array(&json_value) {
        let mut calls = Vec::new();
        for tool in tools {
            if let Some(call) = parse_single_tool(tool) {
                calls.push(call);
            }
        }
        return calls;
    }
    match parse_single_tool(&json_value) {
        Some(call) => vec![call],
        None => Vec::new(),
    }
}

// Walks the "tools" array of a truncated response and parses each complete
// {...} object individually; an object cut off by truncation is dropped.
fn salvage_tool_calls(text: &str) -> Vec<ToolCall> {
    let Some(tools_key) = text.find("\"tools\"") else {
        return Vec::new();
    };
    let after_key = &text[tools_key..];
    let Some(array_offset) = after_key.find('[') else {
        return Vec::new();
    };
    let array_body = &after_key[array_offset + 1..];
    let mut calls = Vec::new();
    for span in complete_object_spans(array_body) {
        let Ok(json_value) = serde_json::from_str::<serde_json::Value>(span) else {
            continue;
        };
        if let Some(call) = parse_single_tool(&json_value) {
            calls.push(call);
        }
    }
    calls
}

// Returns the byte spans of balanced top-level {...} objects, string-aware.
fn complete_object_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut strings = StringScanner::default();
    for (index, byte) in text.bytes().enumerate() {
        if strings.consumes(byte) {
            continue;
        }
        match byte {
            b'{' => {
                if depth == 0 {
                    start = index;
                }
                depth += 1;
            }
            b'}' => {
                if depth == 1 {
                    spans.push(&text[start..=index]);
                }
                depth = depth.saturating_sub(1);
            }
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    spans
}

// Tracks whether the scan is inside a JSON string literal (escape-aware).
#[derive(Default)]
struct StringScanner {
    in_string: bool,
    escaped: bool,
}

impl StringScanner {
    fn consumes(&mut self, byte: u8) -> bool {
        if self.in_string {
            if self.escaped {
                self.escaped = false;
            } else if byte == b'\\' {
                self.escaped = true;
            } else if byte == b'"' {
                self.in_string = false;
            }
            return true;
        }
        if byte == b'"' {
            self.in_string = true;
            return true;
        }
        false
    }
}

fn parse_embedded_json(text: &str) -> Option<serde_json::Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(&text[start..=end]).ok()
}

fn strip_markdown_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(fenced) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let Some((_, body)) = fenced.split_once('\n') else {
        return trimmed;
    };
    let Some(body) = body.strip_suffix("```") else {
        return trimmed;
    };
    body.trim()
}

fn parse_single_tool(obj: &serde_json::Value) -> Option<ToolCall> {
    let tool_name = obj.get("tool")?.as_str()?;
    let args = match obj.get("args") {
        Some(val) => val,
        None => obj.get("arguments")?,
    };
    match tool_name {
        "upsert_node" => parse_upsert_node(args),
        "upsert_edge" => parse_upsert_edge(args),
        "delete_node" => parse_delete_node(args),
        "delete_edge" => parse_delete_edge(args),
        "expire_edge" => parse_expire_edge(args),
        _ => None,
    }
}

// gpt-oss models leak harmony channel markers into the tool name
// ("upsert_node<|channel|>json") — strip everything from the first "<|" so
// those calls are not silently dropped.
pub(crate) fn sanitized_tool_name(name: &str) -> &str {
    match name.find("<|") {
        Some(index) => &name[..index],
        None => name,
    }
}

pub(crate) fn parse_tool_call_request(call: &ToolCallRequest) -> Option<ToolCall> {
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.args) else {
        return None;
    };
    match sanitized_tool_name(&call.name) {
        "upsert_node" => parse_upsert_node(&args),
        "upsert_edge" => parse_upsert_edge(&args),
        "delete_node" => parse_delete_node(&args),
        "delete_edge" => parse_delete_edge(&args),
        "expire_edge" => parse_expire_edge(&args),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn memory_mutation_tools() -> serde_json::Value {
    serde_json::Value::Array(vec![
        memory_tool(
            "upsert_node",
            "Create or update a graph node.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["entity", "fact", "note"]},
                    "body": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["name", "type", "body", "date"]
            }),
        ),
        memory_tool(
            "upsert_edge",
            "Create or update a directed graph edge.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                    "stability": {"type": "number"},
                    "confidence": {"type": "number"},
                    "importance": {"type": "number"},
                    "subject": {"type": "boolean"},
                    "date": {"type": "string"}
                },
                "required": ["source", "target", "content", "stability", "confidence", "importance", "subject", "date"]
            }),
        ),
        memory_tool(
            "delete_node",
            "Delete a node and its incident edges.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }),
        ),
        memory_tool(
            "delete_edge",
            "Delete a directed edge.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["source", "target"]
            }),
        ),
        memory_tool(
            "expire_edge",
            "Mark a belief historical instead of deleting it.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "rewritten_content": {"type": "string"}
                },
                "required": ["source", "target", "rewritten_content"]
            }),
        ),
    ])
}

fn memory_tool(name: &str, description: &str, parameters: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    })
}

fn tool_array(obj: &serde_json::Value) -> Option<&[serde_json::Value]> {
    let tools = obj.get("tools")?;
    match tools.as_array() {
        Some(array) => Some(array.as_slice()),
        None => None,
    }
}

fn parse_upsert_node(args: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(args, "name");
    let node_type = json_str(args, "type");
    if name.is_empty() || node_type.is_empty() {
        return None;
    }
    Some(ToolCall::UpsertNode {
        name,
        node_type,
        body: json_str(args, "body"),
        date: json_str(args, "date"),
    })
}

fn parse_upsert_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    let date = {
        let date_val = json_str(args, "date");
        if date_val.is_empty() {
            json_str(args, "reinforced_at")
        } else {
            date_val
        }
    };
    Some(ToolCall::UpsertEdge {
        source,
        target,
        content: json_str(args, "content"),
        stability: json_f64(args, "stability"),
        confidence: json_f64(args, "confidence"),
        importance: json_f64(args, "importance"),
        subject: json_bool(args, "subject"),
        date,
    })
}

fn parse_delete_node(args: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(args, "name");
    if name.is_empty() {
        return None;
    }
    Some(ToolCall::DeleteNode { name })
}

fn parse_delete_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some(ToolCall::DeleteEdge { source, target })
}

fn parse_expire_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    let rewritten_content = json_str(args, "rewritten_content");
    if source.is_empty() || target.is_empty() || rewritten_content.is_empty() {
        return None;
    }
    Some(ToolCall::ExpireEdge {
        source,
        target,
        rewritten_content,
    })
}

fn json_str(obj: &serde_json::Value, key: &str) -> String {
    let Some(value) = obj.get(key) else {
        return String::new();
    };
    match value.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

fn json_bool(obj: &serde_json::Value, key: &str) -> bool {
    let Some(value) = obj.get(key) else {
        return false;
    };
    value.as_bool().unwrap_or(false)
}

fn json_f64(obj: &serde_json::Value, key: &str) -> f64 {
    let Some(value) = obj.get(key) else {
        return 0.0;
    };
    value.as_f64().unwrap_or(0.0)
}

pub(crate) struct ApplyToolOptions<'scope> {
    pub actor: Actor,
    pub context: EventContext,
    pub source_message_id: &'scope str,
    pub event_payload: &'scope str,
    pub confidence_cap: Option<f64>,
}

fn apply_tool_calls<'agent_run>(
    agent_run: &AgentRun<'agent_run>,
    conn: &Connection,
    phase: &'static str,
    tool_calls: Vec<ToolCall>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut events: Vec<Event> = Vec::new();
    let result = (|| {
        let options = ApplyToolOptions {
            actor: Actor::new("system", phase),
            context: mutation_context(agent_run.user_message_id),
            source_message_id: agent_run.user_message_id,
            event_payload: "",
            confidence_cap: None,
        };
        for call in tool_calls {
            let description = apply_tool_call(call, conn, &options, &mut events)?;
            agent_run.send_log(phase, description);
        }
        db::block_on(ledger::record_events_in_tx(conn, &events))
    })();
    if let Err(err) = result {
        if let Err(rollback_err) = db::block_on(async { conn.execute("ROLLBACK", ()).await }) {
            runlog::log("ledger", &format!("rollback failed: {rollback_err}"));
        }
        return Err(err);
    }
    db::commit_transaction(conn)?;
    Ok(())
}

#[rustfmt::skip]
pub(crate) fn apply_tool_call<'scope>(
    call: ToolCall,
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    match call {
        ToolCall::UpsertNode { name, node_type, body, date } =>
            apply_upsert_node(conn, name, node_type, body, date),
        ToolCall::UpsertEdge { source, target, content, stability, confidence, importance, subject, date } =>
            apply_upsert_edge_call(conn, options, events, UpsertEdgeCall { source, target, content, stability, confidence, importance, subject, date }),
        ToolCall::DeleteNode { name } =>
            apply_delete_node(conn, name, options, events),
        ToolCall::DeleteEdge { source, target } =>
            apply_delete_edge(conn, EdgeRef::new(&source, &target), options, events),
        ToolCall::ExpireEdge { source, target, rewritten_content } =>
            apply_expire_edge(conn, ExpireEdgeCall { source, target, rewritten_content }, options, events),
    }
}

struct ExpireEdgeCall {
    source: String,
    target: String,
    rewritten_content: String,
}

fn apply_expire_edge<'scope>(
    conn: &Connection,
    call: ExpireEdgeCall,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let ExpireEdgeCall {
        source,
        target,
        rewritten_content,
    } = call;
    let normalized_source = normalize_id(&source);
    let normalized_target = normalize_id(&target);
    let Some(mut edge) = db::load_edge(conn, &normalized_source, &normalized_target)? else {
        return Ok(format!(
            "expire_edge skipped: {source} → {target} (not found)"
        ));
    };
    edge.content = rewritten_content;
    edge.expired_at = Some(crate::timeutil::timestamp_now());
    db::write_edge(conn, &edge, options.source_message_id)?;
    let payload = if options.event_payload.is_empty() {
        serde_json::json!({"expired": true}).to_string()
    } else {
        options.event_payload.to_string()
    };
    events.push(Event::updated(
        options.actor.clone(),
        EdgeRef::new(&edge.source, &edge.target),
        options.context.clone(),
        payload,
    ));
    Ok(format!("expire_edge: {source} → {target}"))
}

struct UpsertEdgeCall {
    source: String,
    target: String,
    content: String,
    stability: f64,
    confidence: f64,
    importance: f64,
    subject: bool,
    date: String,
}

fn apply_upsert_edge_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    mut call: UpsertEdgeCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(confidence_cap) = options.confidence_cap {
        call.confidence = call.confidence.min(confidence_cap);
    }
    let edge = Edge {
        source: normalize_id(&call.source),
        target: normalize_id(&call.target),
        content: call.content,
        stability: call.stability,
        confidence: call.confidence,
        importance: call.importance,
        subject: call.subject,
        date: call.date,
        occurred_at: None,
        expired_at: None,
    };
    apply_upsert_edge(conn, edge, options, events)
}

fn apply_upsert_node(
    conn: &Connection,
    name: String,
    node_type: String,
    body: String,
    date: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let node = Node {
        node_type: node_type.clone(),
        body,
        date,
    };
    db::write_node(conn, &normalized, &node)?;
    Ok(format!("upsert_node: {name} ({node_type})"))
}

fn apply_upsert_edge<'scope>(
    conn: &Connection,
    edge: Edge,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let existing = db::load_edge(conn, &edge.source, &edge.target)?;
    db::write_edge(conn, &edge, options.source_message_id)?;
    let event = if options.event_payload.is_empty() {
        Event::upsert(
            options.actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            options.context.clone(),
            existing.is_some(),
        )
    } else {
        Event::upsert_with_payload(
            options.actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            options.context.clone(),
            existing.is_some(),
            options.event_payload.to_string(),
        )
    };
    events.push(event);
    Ok(format!("upsert_edge: {} → {}", edge.source, edge.target))
}

fn apply_delete_node<'scope>(
    conn: &Connection,
    name: String,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let deleted_edges = db::load_edges_for_node(conn, &normalized)?;
    db::erase_node(conn, &normalized)?;
    let pruned = ledger::pruned_events(
        options.actor.clone(),
        options.context.clone(),
        &deleted_edges,
    )?;
    events.extend(pruned);
    Ok(format!("delete_node: {name}"))
}

fn apply_delete_edge<'scope>(
    conn: &Connection,
    edge_ref: EdgeRef,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&edge_ref.source);
    let normalized_target = normalize_id(&edge_ref.target);
    let deleted_edge = db::load_edge(conn, &normalized_source, &normalized_target)?;
    db::erase_edge(conn, &normalized_source, &normalized_target)?;
    if let Some(edge) = deleted_edge {
        let pruned =
            ledger::pruned_events(options.actor.clone(), options.context.clone(), &[edge])?;
        events.extend(pruned);
    }
    Ok(format!(
        "delete_edge: {} → {}",
        edge_ref.source, edge_ref.target
    ))
}

fn mutation_context(source_message_id: &str) -> EventContext {
    let session_id = db::source_session_id(source_message_id).unwrap_or_default();
    EventContext::new(session_id.as_str(), source_message_id)
}

#[cfg(test)]
mod tests {
    use super::{ToolCall, parse_tool_calls};

    #[test]
    fn truncated_tool_array_salvages_complete_objects() {
        let truncated = r#"{"tools":[{"tool":"upsert_node","args":{"name":"a","type":"entity","body":"","date":""}},{"tool":"upsert_edge","args":{"source":"a","target":"b","content":"a relates to b","stability":0.5,"confidence":0.5,"importance":0.5,"subject":false,"date":"2026-07-06"}},{"tool":"upsert_node","args":{"name":"trunc"#;
        let calls = parse_tool_calls(truncated);
        assert_eq!(calls.len(), 2);
        assert!(matches!(calls[0], ToolCall::UpsertNode { .. }));
        assert!(matches!(calls[1], ToolCall::UpsertEdge { .. }));
    }

    #[test]
    fn prose_wrapped_json_is_recovered() {
        let wrapped = r#"Here are the extracted facts: {"tools":[{"tool":"delete_edge","args":{"source":"a","target":"b"}}]} done."#;
        let calls = parse_tool_calls(wrapped);
        assert_eq!(calls.len(), 1);
    }
}
