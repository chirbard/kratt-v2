use std::io::{BufRead, BufReader};
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Chunk, LlmClient, Stats, ToolCallRequest, TurnOutcome};
use crate::trace::{self, ResponseRecord, TraceSpan};

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
struct PromptTokensDetails {
    cached_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    cost: Option<f64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
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
    cached_tokens: Option<u32>,
    saw_prompt_tokens: bool,
    tool_calls: Vec<ToolCallAcc>,
    finish_reason: Option<String>,
    content: String,
    reasoning: String,
}

struct ApiTrace<'scope> {
    span: &'scope TraceSpan,
    provider: &'scope str,
    model: &'scope str,
}

pub(super) struct OpenRouterClient {
    api_key: String,
    model: String,
    // Full chat-completions URL — OpenRouter's endpoint by default, or a local
    // Ollama server's OpenAI-compatible endpoint when the model spec asked for
    // one. The wire format is identical either way.
    url: String,
}

impl OpenRouterClient {
    pub(super) fn new(api_key: String, model: String, url: String) -> Self {
        Self {
            api_key,
            model,
            url,
        }
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
                Ok(Chunk::MemoryLog { .. }) | Ok(Chunk::Attribution(_)) => {}
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
        self.run_stream_turn(messages, tools, sender, None)
    }

    fn stream_traced(
        &self,
        prompt: String,
        system: Option<String>,
        sender: Sender<Chunk>,
        span: &TraceSpan,
    ) {
        let messages = build_stream_messages(prompt, system);
        match self.run_stream_turn(&messages, None, &sender, Some(span)) {
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

    fn stream_turn_traced(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        sender: &Sender<Chunk>,
        span: &TraceSpan,
    ) -> Result<TurnOutcome, Box<dyn std::error::Error + Send + Sync>> {
        self.run_stream_turn(messages, tools, sender, Some(span))
    }
}

impl OpenRouterClient {
    fn run_stream_turn(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        sender: &Sender<Chunk>,
        span: Option<&TraceSpan>,
    ) -> Result<TurnOutcome, Box<dyn std::error::Error + Send + Sync>> {
        let body = request_body(&self.model, messages, tools);
        let provider = provider_name(&self.url);
        if let Some(span) = span {
            trace::request(span, provider, &self.model, body.clone());
        }
        let started = Instant::now();
        let resp = send_traced_request(self, &body, provider, span, started)?;
        #[allow(clippy::manual_map)]
        let api_trace = match span {
            Some(span) => Some(ApiTrace {
                span,
                provider,
                model: &self.model,
            }),
            None => None,
        };
        let result = stream_response(resp, sender, request_area(messages), api_trace);
        if let Err(err) = &result
            && let Some(span) = span
        {
            trace::error(
                span,
                provider,
                &self.model,
                &err.to_string(),
                started.elapsed().as_millis(),
            );
        }
        result
    }
}

fn send_traced_request(
    client: &OpenRouterClient,
    body: &Value,
    provider: &str,
    span: Option<&TraceSpan>,
    started: Instant,
) -> Result<reqwest::blocking::Response, Box<dyn std::error::Error + Send + Sync>> {
    match send_request(&client.api_key, &client.url, body) {
        Ok(response) => Ok(response),
        Err(err) => {
            if let Some(span) = span {
                trace::error(
                    span,
                    provider,
                    &client.model,
                    &err,
                    started.elapsed().as_millis(),
                );
            }
            Err(err.into())
        }
    }
}

fn request_area(messages: &[Value]) -> &'static str {
    for message in messages {
        let role = match message.get("role") {
            Some(value) => value.as_str(),
            None => None,
        };
        if role == Some("system") {
            return "chat";
        }
    }
    for message in messages {
        let content = match message.get("content") {
            Some(value) => value.as_str().unwrap_or(""),
            None => "",
        };
        if content.contains("## Current memory graph") {
            return "dream";
        }
    }
    "mem"
}

fn build_stream_messages(prompt: String, system: Option<String>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = system {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

fn request_body(model: &str, messages: &[Value], tools: Option<&Value>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(tool_defs) = tools {
        insert_tools(&mut body, tool_defs);
    }
    body
}

fn send_request(
    api_key: &str,
    url: &str,
    body: &Value,
) -> Result<reqwest::blocking::Response, String> {
    let resp = match Client::new()
        .post(url)
        .bearer_auth(api_key)
        .json(body)
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

fn provider_name(url: &str) -> &'static str {
    if url.contains("openrouter.ai") {
        "openrouter"
    } else {
        "openai-compatible"
    }
}

fn insert_tools(body: &mut Value, tool_defs: &Value) {
    if let Some(map) = body.as_object_mut() {
        map.insert("tools".to_string(), tool_defs.clone());
    }
}

fn stream_response<'scope>(
    resp: reqwest::blocking::Response,
    sender: &Sender<Chunk>,
    area: &'static str,
    trace_info: Option<ApiTrace<'scope>>,
) -> Result<TurnOutcome, Box<dyn std::error::Error + Send + Sync>> {
    let mut state = StreamState {
        start: Instant::now(),
        ttft: None,
        tokens_in: 0,
        tokens_out: 0,
        cost_usd: None,
        cached_tokens: None,
        saw_prompt_tokens: false,
        tool_calls: Vec::new(),
        finish_reason: None,
        content: String::new(),
        reasoning: String::new(),
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
    log_cached_usage(area, &state);
    if let Some(trace_info) = trace_info {
        trace_response(&state, trace_info);
    }
    Ok(build_turn_outcome(state))
}

fn trace_response<'scope>(state: &StreamState, trace_info: ApiTrace<'scope>) {
    let content = json!({
        "content": state.content,
        "reasoning": state.reasoning,
        "tool_calls": tool_calls_json(&state.tool_calls),
    });
    let usage = json!({
        "prompt_tokens": state.tokens_in,
        "completion_tokens": state.tokens_out,
        "cost_usd": state.cost_usd,
        "cached_tokens": state.cached_tokens,
    });
    trace::response(
        trace_info.span,
        trace_info.provider,
        trace_info.model,
        ResponseRecord {
            content,
            finish_reason: state.finish_reason.clone(),
            usage,
            duration_ms: state.start.elapsed().as_millis(),
        },
    );
}

fn tool_calls_json(calls: &[ToolCallAcc]) -> Value {
    let mut values = Vec::new();
    for call in calls {
        values.push(json!({"id": call.id, "name": call.name, "args": call.args}));
    }
    Value::Array(values)
}

fn log_cached_usage(area: &str, state: &StreamState) {
    let Some(line) = cached_usage_line(state) else {
        return;
    };
    crate::runlog::log(area, &line);
}

fn cached_usage_line(state: &StreamState) -> Option<String> {
    let cached = state.cached_tokens?;
    if !state.saw_prompt_tokens {
        return None;
    }
    Some(format!("cached {cached}/{} prompt tokens", state.tokens_in))
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

fn send_reasoning_delta(
    reasoning: Option<String>,
    state: &mut StreamState,
    sender: &Sender<Chunk>,
) {
    if let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        record_ttft(state);
        state.reasoning.push_str(&reasoning);
        let _ = sender.send(Chunk::Reasoning(reasoning));
    }
}

fn send_content_delta(content: Option<String>, state: &mut StreamState, sender: &Sender<Chunk>) {
    if let Some(content) = content
        && !content.is_empty()
    {
        record_ttft(state);
        state.content.push_str(&content);
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
        state.saw_prompt_tokens = true;
    }
    if let Some(tok_out) = usage.completion_tokens {
        state.tokens_out = tok_out;
    }
    if let Some(cost) = usage.cost {
        state.cost_usd = Some(cost);
    }
    if let Some(details) = usage.prompt_tokens_details
        && let Some(cached) = details.cached_tokens
    {
        state.cached_tokens = Some(cached);
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

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
