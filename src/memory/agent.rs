use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient};
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

pub(super) enum ToolCall {
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

pub(super) fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let stripped = strip_markdown_fences(text);
    let Ok(json_value) = serde_json::from_str::<serde_json::Value>(stripped) else {
        return Vec::new();
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

fn apply_tool_calls<'agent_run>(
    agent_run: &AgentRun<'agent_run>,
    conn: &Connection,
    phase: &'static str,
    tool_calls: Vec<ToolCall>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut events: Vec<Event> = Vec::new();
    let result = (|| {
        for call in tool_calls {
            let description =
                apply_tool(call, conn, agent_run.user_message_id, phase, &mut events)?;
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
fn apply_tool(call: ToolCall, conn: &Connection, source_message_id: &str, phase: &str, events: &mut Vec<Event>) -> Result<String, Box<dyn Error + Send + Sync>> {
    match call {
        ToolCall::UpsertNode { name, node_type, body, date } =>
            apply_upsert_node(conn, name, node_type, body, date),
        ToolCall::UpsertEdge { source, target, content, stability, confidence, importance, subject, date } =>
            apply_upsert_edge_call(conn, source_message_id, phase, events, UpsertEdgeCall { source, target, content, stability, confidence, importance, subject, date }),
        ToolCall::DeleteNode { name } =>
            apply_delete_node(conn, name, source_message_id, phase, events),
        ToolCall::DeleteEdge { source, target } =>
            apply_delete_edge(conn, EdgeRef::new(&source, &target), source_message_id, phase, events),
        ToolCall::ExpireEdge { source, target, rewritten_content } =>
            apply_expire_edge(conn, ExpireEdgeCall { source, target, rewritten_content }, source_message_id, phase, events),
    }
}

struct ExpireEdgeCall {
    source: String,
    target: String,
    rewritten_content: String,
}

fn apply_expire_edge(
    conn: &Connection,
    call: ExpireEdgeCall,
    source_message_id: &str,
    phase: &str,
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
        return Ok(format!("expire_edge skipped: {source} → {target} (not found)"));
    };
    edge.content = rewritten_content;
    edge.expired_at = Some(crate::timeutil::timestamp_now());
    db::write_edge(conn, &edge, source_message_id)?;
    let payload = serde_json::json!({"expired": true}).to_string();
    events.push(Event::updated(
        Actor::new("system", phase),
        EdgeRef::new(&edge.source, &edge.target),
        mutation_context(source_message_id),
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

fn apply_upsert_edge_call(
    conn: &Connection,
    source_message_id: &str,
    phase: &str,
    events: &mut Vec<Event>,
    call: UpsertEdgeCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
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
    apply_upsert_edge(conn, edge, source_message_id, phase, events)
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

fn apply_upsert_edge(
    conn: &Connection,
    edge: Edge,
    source_message_id: &str,
    phase: &str,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let existing = db::load_edge(conn, &edge.source, &edge.target)?;
    db::write_edge(conn, &edge, source_message_id)?;
    events.push(Event::upsert(
        Actor::new("system", phase),
        EdgeRef::new(&edge.source, &edge.target),
        mutation_context(source_message_id),
        existing.is_some(),
    ));
    Ok(format!("upsert_edge: {} → {}", edge.source, edge.target))
}

fn apply_delete_node(
    conn: &Connection,
    name: String,
    source_message_id: &str,
    phase: &str,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let deleted_edges = db::load_edges_for_node(conn, &normalized)?;
    db::erase_node(conn, &normalized)?;
    let pruned = ledger::pruned_events(
        Actor::new("system", phase),
        mutation_context(source_message_id),
        &deleted_edges,
    )?;
    events.extend(pruned);
    Ok(format!("delete_node: {name}"))
}

fn apply_delete_edge(
    conn: &Connection,
    edge_ref: EdgeRef,
    source_message_id: &str,
    phase: &str,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&edge_ref.source);
    let normalized_target = normalize_id(&edge_ref.target);
    let deleted_edge = db::load_edge(conn, &normalized_source, &normalized_target)?;
    db::erase_edge(conn, &normalized_source, &normalized_target)?;
    if let Some(edge) = deleted_edge {
        let pruned = ledger::pruned_events(
            Actor::new("system", phase),
            mutation_context(source_message_id),
            &[edge],
        )?;
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
