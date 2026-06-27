use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient};
use crate::prompts::{CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, build_agent_prompt};
use super::decay;
use super::git;
use super::graph::{
    Edge, Graph, Node, delete_node, graph_to_json, insert_edge, load_graph, normalize_id,
    remove_edges, save_graph,
};

pub enum MemoryEvent {
    Log { phase: &'static str, text: String },
    Append { phase: &'static str, text: String },
    Done,
}

struct AgentRun<'run> {
    llm: Arc<dyn LlmClient>,
    user_msg: &'run str,
    assistant_msg: &'run str,
    session_label: &'run str,
    sender: &'run std::sync::mpsc::Sender<MemoryEvent>,
}

enum ToolCall {
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
    session_label: String,
    sender: std::sync::mpsc::Sender<MemoryEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let agent_run = AgentRun {
        llm,
        user_msg,
        assistant_msg,
        session_label: &session_label,
        sender: &sender,
    };
    let result = agent_run.run();
    let _ = sender.send(MemoryEvent::Done);
    result
}

impl<'run> AgentRun<'run> {
    fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut graph = load_graph();
        decay::apply_decay_if_needed(&mut graph);

        let pre_extract_toon = toon::encode(&graph_to_json(&graph), None);
        let extract_prompt = build_agent_prompt(
            EXTRACT_EXPLANATION,
            &pre_extract_toon,
            self.user_msg,
            self.assistant_msg,
            None,
        );
        self.run_pass(&mut graph, "extract", extract_prompt)?;

        if let Err(err) = save_graph(&graph) {
            self.send_log("memory", format!("error: {err}"));
            return Err(err);
        }
        let extract_committed = self.git_commit_logged(&format!("{} [extract]", self.session_label));

        let diff = if extract_committed { git::git_diff_last() } else { String::new() };
        let consolidate_prompt = build_agent_prompt(
            CONSOLIDATE_EXPLANATION,
            &pre_extract_toon,
            self.user_msg,
            self.assistant_msg,
            if diff.is_empty() { None } else { Some(&diff) },
        );
        self.run_pass(&mut graph, "consolidate", consolidate_prompt)?;

        if let Err(err) = save_graph(&graph) {
            self.send_log("memory", format!("error: {err}"));
            return Err(err);
        }
        self.git_commit_logged(self.session_label);
        self.send_log(
            "memory",
            format!("saved — {} nodes, {} edges", graph.nodes.len(), graph.edges.len()),
        );
        Ok(())
    }

    fn git_commit_logged(&self, label: &str) -> bool {
        match git::git_commit(label) {
            Ok(committed) => {
                if committed {
                    self.send_log("debug", format!("git: committed {label}"));
                }
                committed
            }
            Err(err) => {
                self.send_log("memory", format!("git: {err}"));
                false
            }
        }
    }

    fn run_pass(
        &self,
        graph: &mut Graph,
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
        for call in tool_calls {
            let description = apply_tool(call, graph);
            self.send_log(phase, description);
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

fn stream_collect(
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

fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
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

fn tool_array(obj: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    let tools = obj.get("tools")?;
    tools.as_array()
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

fn apply_tool(call: ToolCall, graph: &mut Graph) -> String {
    match call {
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => apply_upsert_node(graph, name, node_type, body, date),
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
            apply_upsert_edge(graph, source, target, edge)
        }
        ToolCall::DeleteNode { name } => apply_delete_node(graph, name),
        ToolCall::DeleteEdge { source, target } => apply_delete_edge(graph, source, target),
    }
}

fn apply_upsert_node(
    graph: &mut Graph,
    name: String,
    node_type: String,
    body: String,
    date: String,
) -> String {
    graph.nodes.insert(
        normalize_id(&name),
        Node {
            node_type: node_type.clone(),
            body,
            date,
        },
    );
    format!("upsert_node: {name} ({node_type})")
}

fn apply_upsert_edge(graph: &mut Graph, source: String, target: String, edge: Edge) -> String {
    insert_edge(graph, edge);
    format!("upsert_edge: {source} → {target}")
}

fn apply_delete_node(graph: &mut Graph, name: String) -> String {
    delete_node(graph, &normalize_id(&name));
    format!("delete_node: {name}")
}

fn apply_delete_edge(graph: &mut Graph, source: String, target: String) -> String {
    remove_edges(
        &mut graph.edges,
        &normalize_id(&source),
        &normalize_id(&target),
    );
    format!("delete_edge: {source} → {target}")
}
