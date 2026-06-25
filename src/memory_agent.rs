use std::error::Error;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

use crate::graph::{Edge, Graph, Node, delete_node, graph_to_json, insert_edge, load_graph, normalize_id, remove_edges, save_graph};
use crate::memory_prompt::{CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, build_agent_prompt};

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
    },
    DeleteNode {
        name: String,
    },
    DeleteEdge {
        source: String,
        target: String,
    },
}

#[derive(Deserialize)]
struct LlmResponse {
    choices: Vec<LlmChoice>,
}

#[derive(Deserialize)]
struct LlmChoice {
    message: LlmMessage,
}

#[derive(Deserialize)]
struct LlmMessage {
    content: String,
}

pub fn run_memory_agent(
    api_key: &str,
    model: &str,
    user_msg: &str,
    assistant_msg: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut graph = load_graph();
    let mut toon_text = toon::encode(&graph_to_json(&graph), None);

    let extract_prompt =
        build_agent_prompt(EXTRACT_EXPLANATION, &toon_text, user_msg, assistant_msg);
    let extract_text = call_llm(api_key, model, &extract_prompt)?;
    for call in parse_tool_calls(&extract_text) {
        apply_tool(call, &mut graph);
    }

    toon_text = toon::encode(&graph_to_json(&graph), None);
    let consolidate_prompt =
        build_agent_prompt(CONSOLIDATE_EXPLANATION, &toon_text, user_msg, assistant_msg);
    let consolidate_text = call_llm(api_key, model, &consolidate_prompt)?;
    for call in parse_tool_calls(&consolidate_text) {
        apply_tool(call, &mut graph);
    }

    save_graph(&graph)?;
    Ok(())
}

fn call_llm(
    api_key: &str,
    model: &str,
    prompt: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
    });
    let response = Client::new()
        .post(OPENROUTER_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()?;
    if !response.status().is_success() {
        let status = response.status();
        let body_text = response.text().unwrap_or_default();
        return Err(format!("HTTP {status}: {body_text}").into());
    }
    let parsed: LlmResponse = response.json()?;
    let content = match parsed.choices.first() {
        Some(choice) => choice.message.content.clone(),
        None => String::new(),
    };
    Ok(content)
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
    let arguments = obj.get("arguments")?;
    match tool_name {
        "upsert_node" => parse_upsert_node(arguments),
        "upsert_edge" => parse_upsert_edge(arguments),
        "delete_node" => parse_delete_node(arguments),
        "delete_edge" => parse_delete_edge(arguments),
        _ => None,
    }
}

fn tool_array(obj: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    let tools = obj.get("tools")?;
    tools.as_array()
}

fn parse_upsert_node(arguments: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(arguments, "name");
    let node_type = json_str(arguments, "type");
    if name.is_empty() || node_type.is_empty() {
        return None;
    }
    Some(ToolCall::UpsertNode {
        name,
        node_type,
        body: json_str(arguments, "body"),
        date: json_str(arguments, "date"),
    })
}

fn parse_upsert_edge(arguments: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(arguments, "source");
    let target = json_str(arguments, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some(ToolCall::UpsertEdge {
        source,
        target,
        content: json_str(arguments, "content"),
        stability: json_f64(arguments, "stability"),
        confidence: json_f64(arguments, "confidence"),
        importance: json_f64(arguments, "importance"),
        subject: json_bool(arguments, "subject"),
    })
}

fn parse_delete_node(arguments: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(arguments, "name");
    if name.is_empty() {
        return None;
    }
    Some(ToolCall::DeleteNode { name })
}

fn parse_delete_edge(arguments: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(arguments, "source");
    let target = json_str(arguments, "target");
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

fn apply_tool(call: ToolCall, graph: &mut Graph) {
    match call {
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => {
            graph.nodes.insert(normalize_id(&name), Node { node_type, body, date });
        }
        ToolCall::UpsertEdge {
            source,
            target,
            content,
            stability,
            confidence,
            importance,
            subject,
        } => {
            insert_edge(graph, Edge {
                source: normalize_id(&source),
                target: normalize_id(&target),
                content,
                stability,
                confidence,
                importance,
                subject,
            });
        }
        ToolCall::DeleteNode { name } => {
            delete_node(graph, &normalize_id(&name));
        }
        ToolCall::DeleteEdge { source, target } => {
            remove_edges(&mut graph.edges, &normalize_id(&source), &normalize_id(&target));
        }
    }
}
