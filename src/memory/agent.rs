use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient};
use crate::prompts::{CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, build_agent_prompt};
use turso::Connection;

use super::db;
use super::decay;
use super::graph::{Edge, Node, graph_to_json, normalize_id};

pub enum MemoryEvent {
    Log { phase: &'static str, text: String },
    Append { phase: &'static str, text: String },
    Done,
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
            Chunk::ToolCall(_) => {}
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
    for call in tool_calls {
        let description = apply_tool(call, conn, agent_run.user_message_id)?;
        agent_run.send_log(phase, description);
    }
    db::commit_transaction(conn)?;
    Ok(())
}

fn apply_tool(
    call: ToolCall,
    conn: &Connection,
    source_message_id: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    match call {
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => apply_upsert_node(conn, name, node_type, body, date),
        ToolCall::UpsertEdge {
            source,
            target,
            content,
            stability,
            confidence,
            importance,
            subject,
            date,
        } => {
            let edge = Edge {
                source: normalize_id(&source),
                target: normalize_id(&target),
                content,
                stability,
                confidence,
                importance,
                subject,
                date,
            };
            apply_upsert_edge(conn, source, target, edge, source_message_id)
        }
        ToolCall::DeleteNode { name } => apply_delete_node(conn, name),
        ToolCall::DeleteEdge { source, target } => apply_delete_edge(conn, source, target),
    }
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
    source: String,
    target: String,
    edge: Edge,
    source_message_id: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    db::write_edge(conn, &edge, source_message_id)?;
    Ok(format!("upsert_edge: {source} → {target}"))
}

fn apply_delete_node(
    conn: &Connection,
    name: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    db::erase_node(conn, &normalized)?;
    Ok(format!("delete_node: {name}"))
}

fn apply_delete_edge(
    conn: &Connection,
    source: String,
    target: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&source);
    let normalized_target = normalize_id(&target);
    db::erase_edge(conn, &normalized_source, &normalized_target)?;
    Ok(format!("delete_edge: {source} → {target}"))
}
