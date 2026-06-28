use std::io::{BufRead, BufReader};
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Chunk, LlmClient, Stats, ToolCallRequest, TurnOutcome};

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

#[derive(Deserialize)]
struct ToolCallFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
#[allow(clippy::min_ident_chars)]
struct ToolCallDelta {
    index: Option<u64>,
    id: Option<String>,
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct Choice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    cost: Option<f64>,
}

#[derive(Deserialize)]
struct SseMessage {
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
}

#[allow(clippy::min_ident_chars)]
struct ToolCallAcc {
    id: String,
    name: String,
    args: String,
}

struct StreamState {
    start: Instant,
    ttft: Option<u128>,
    tokens_in: u32,
    tokens_out: u32,
    cost_usd: Option<f64>,
    tool_calls: Vec<ToolCallAcc>,
    finish_reason: Option<String>,
}

pub(super) struct OpenRouterClient {
    api_key: String,
    model: String,
}

impl OpenRouterClient {
    pub(super) fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

impl LlmClient for OpenRouterClient {
    #[allow(dead_code)]
    fn collect(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let (sender, receiver) = channel::<Chunk>();
        self.stream(prompt.to_string(), None, sender);
        let mut content = String::new();
        let mut reasoning = String::new();
        loop {
            match receiver.recv() {
                Ok(Chunk::Content(text)) => content.push_str(&text),
                Ok(Chunk::Reasoning(text)) => reasoning.push_str(&text),
                Ok(Chunk::ToolCall(_)) => {}
                Ok(Chunk::Done(_)) => break,
                Ok(Chunk::Err(err)) => return Err(err.into()),
                Err(err) => return Err(err.into()),
            }
        }
        if !content.is_empty() {
            return Ok(content);
        }
        Ok(reasoning)
    }

    fn stream(&self, prompt: String, system: Option<String>, sender: Sender<Chunk>) {
        let messages = build_stream_messages(prompt, system);
        match self.stream_turn(&messages, None, &sender) {
            Ok(TurnOutcome::Done(stats)) => {
                let _ = sender.send(Chunk::Done(stats));
            }
            Ok(TurnOutcome::ToolCalls(_)) => {
                let _ = sender.send(Chunk::Err("unexpected tool call".to_string()));
            }
            Err(err) => {
                let _ = sender.send(Chunk::Err(err.to_string()));
            }
        }
    }

    fn stream_turn(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        sender: &Sender<Chunk>,
    ) -> Result<TurnOutcome, Box<dyn std::error::Error + Send + Sync>> {
        let resp = make_request_messages(&self.api_key, &self.model, messages, tools)?;
        stream_response(resp, sender)
    }
}

fn build_stream_messages(prompt: String, system: Option<String>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = system {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

fn make_request_messages(
    api_key: &str,
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
) -> Result<reqwest::blocking::Response, String> {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(tool_defs) = tools {
        insert_tools(&mut body, tool_defs);
    }
    let resp = match Client::new()
        .post(OPENROUTER_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
    {
        Ok(resp) => resp,
        Err(err) => return Err(err.to_string()),
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().unwrap_or_default();
        return Err(format!("HTTP {status}: {body_text}"));
    }
    Ok(resp)
}

fn insert_tools(body: &mut Value, tool_defs: &Value) {
    if let Some(map) = body.as_object_mut() {
        map.insert("tools".to_string(), tool_defs.clone());
    }
}

fn stream_response(
    resp: reqwest::blocking::Response,
    sender: &Sender<Chunk>,
) -> Result<TurnOutcome, Box<dyn std::error::Error + Send + Sync>> {
    let mut state = StreamState {
        start: Instant::now(),
        ttft: None,
        tokens_in: 0,
        tokens_out: 0,
        cost_usd: None,
        tool_calls: Vec::new(),
        finish_reason: None,
    };
    let reader = BufReader::new(resp);
    for line_result in reader.lines() {
        let line = line_result?;
        let data = match line.strip_prefix("data: ") {
            Some(data) => data,
            None => continue,
        };
        if process_line(data, &mut state, sender)? {
            break;
        }
    }
    Ok(build_turn_outcome(state))
}

fn process_line(
    data: &str,
    state: &mut StreamState,
    sender: &Sender<Chunk>,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    if data == "[DONE]" {
        return Ok(true);
    }
    let msg: SseMessage = match serde_json::from_str(data) {
        Ok(msg) => msg,
        Err(_) => return Ok(false),
    };
    apply_sse_message(msg, state, sender)?;
    Ok(false)
}

fn apply_sse_message(
    msg: SseMessage,
    state: &mut StreamState,
    sender: &Sender<Chunk>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(usage) = msg.usage {
        apply_usage(usage, state);
    }
    if let Some(choices) = msg.choices {
        for choice in choices {
            apply_choice(choice, state, sender)?;
        }
    }
    Ok(())
}

fn apply_choice(
    choice: Choice,
    state: &mut StreamState,
    sender: &Sender<Chunk>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(reason) = choice.finish_reason {
        state.finish_reason = Some(reason);
    }
    apply_delta(choice.delta, state, sender)
}

fn apply_delta(
    delta: Delta,
    state: &mut StreamState,
    sender: &Sender<Chunk>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    send_reasoning_delta(delta.reasoning, state, sender);
    send_content_delta(delta.content, state, sender);
    if let Some(tool_calls) = delta.tool_calls {
        for tool_call in tool_calls {
            apply_tool_call_delta(tool_call, state)?;
        }
    }
    Ok(())
}

fn send_reasoning_delta(reasoning: Option<String>, state: &mut StreamState, sender: &Sender<Chunk>) {
    if let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        record_ttft(state);
        let _ = sender.send(Chunk::Reasoning(reasoning));
    }
}

fn send_content_delta(content: Option<String>, state: &mut StreamState, sender: &Sender<Chunk>) {
    if let Some(content) = content
        && !content.is_empty()
    {
        record_ttft(state);
        let _ = sender.send(Chunk::Content(content));
    }
}

fn apply_tool_call_delta(
    delta: ToolCallDelta,
    state: &mut StreamState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let index_value = match delta.index {
        Some(index_value) => index_value,
        None => return Ok(()),
    };
    let index = ensure_tool_call_slot(state, index_value)?;
    let call = &mut state.tool_calls[index];
    if let Some(id_value) = delta.id {
        call.id = id_value;
    }
    if let Some(function) = delta.function {
        apply_tool_call_function(function, call);
    }
    Ok(())
}

fn ensure_tool_call_slot(
    state: &mut StreamState,
    index_value: u64,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let index = usize::try_from(index_value)?;
    while state.tool_calls.len() <= index {
        state.tool_calls.push(ToolCallAcc {
            id: String::new(),
            name: String::new(),
            args: String::new(),
        });
    }
    Ok(index)
}

fn apply_tool_call_function(function: ToolCallFunctionDelta, call: &mut ToolCallAcc) {
    if let Some(name) = function.name {
        call.name.push_str(&name);
    }
    if let Some(arguments) = function.arguments {
        call.args.push_str(&arguments);
    }
}

fn record_ttft(state: &mut StreamState) {
    if state.ttft.is_none() {
        state.ttft = Some(state.start.elapsed().as_millis());
    }
}

fn apply_usage(usage: Usage, state: &mut StreamState) {
    if let Some(tok_in) = usage.prompt_tokens {
        state.tokens_in = tok_in;
    }
    if let Some(tok_out) = usage.completion_tokens {
        state.tokens_out = tok_out;
    }
    if let Some(cost) = usage.cost {
        state.cost_usd = Some(cost);
    }
}

fn build_turn_outcome(state: StreamState) -> TurnOutcome {
    if let Some(reason) = &state.finish_reason
        && reason == "tool_calls"
    {
        return TurnOutcome::ToolCalls(build_tool_call_requests(&state.tool_calls));
    }
    TurnOutcome::Done(Stats {
        tokens_in: state.tokens_in,
        tokens_out: state.tokens_out,
        cost_usd: state.cost_usd,
        ttft_ms: state.ttft.unwrap_or(0),
    })
}

fn build_tool_call_requests(tool_calls: &[ToolCallAcc]) -> Vec<ToolCallRequest> {
    let mut calls = Vec::new();
    for tool_call in tool_calls {
        calls.push(ToolCallRequest {
            id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            args: tool_call.args.clone(),
        });
    }
    calls
}
