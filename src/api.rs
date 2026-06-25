use std::error::Error;
use std::io::{BufRead, BufReader};
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

pub struct Stats {
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub cost_usd: Option<f64>,
    pub ttft_ms: u128,
}

pub enum Chunk {
    Reasoning(String),
    Content(String),
    Done(Stats),
    Err(String),
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning: Option<String>,
}

#[derive(Deserialize)]
struct Choice {
    delta: Delta,
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

struct StreamState {
    start: Instant,
    ttft: Option<u128>,
    tokens_in: u32,
    tokens_out: u32,
    cost_usd: Option<f64>,
}

fn make_request(
    api_key: &str,
    model: &str,
    prompt: &str,
    system: Option<&str>,
) -> Result<reqwest::blocking::Response, String> {
    let messages = match system {
        Some(system_prompt) => json!([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]),
        None => json!([
            {"role": "user", "content": prompt},
        ]),
    };
    let body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
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

fn record_ttft(state: &mut StreamState) {
    if state.ttft.is_none() {
        state.ttft = Some(state.start.elapsed().as_millis());
    }
}

fn send_delta(delta: Delta, state: &mut StreamState, sender: &Sender<Chunk>) {
    if let Some(reasoning) = delta.reasoning
        && !reasoning.is_empty()
    {
        record_ttft(state);
        let _ = sender.send(Chunk::Reasoning(reasoning));
    }
    if let Some(content) = delta.content
        && !content.is_empty()
    {
        record_ttft(state);
        let _ = sender.send(Chunk::Content(content));
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

fn process_line(data: &str, state: &mut StreamState, sender: &Sender<Chunk>) -> bool {
    if data == "[DONE]" {
        return true;
    }
    let msg: SseMessage = match serde_json::from_str(data) {
        Ok(msg) => msg,
        Err(_) => return false,
    };
    if let Some(usage) = msg.usage {
        apply_usage(usage, state);
    }
    if let Some(choices) = msg.choices {
        for choice in choices {
            send_delta(choice.delta, state, sender);
        }
    }
    false
}

fn send_done(state: StreamState, sender: &Sender<Chunk>) {
    let _ = sender.send(Chunk::Done(Stats {
        tokens_in: state.tokens_in,
        tokens_out: state.tokens_out,
        cost_usd: state.cost_usd,
        ttft_ms: state.ttft.unwrap_or(0),
    }));
}

pub fn collect(
    api_key: &str,
    model: &str,
    prompt: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let (sender, receiver) = channel::<Chunk>();
    stream(api_key.to_string(), model.to_string(), prompt.to_string(), None, sender);
    let mut content = String::new();
    let mut reasoning = String::new();
    loop {
        match receiver.recv() {
            Ok(Chunk::Content(text)) => content.push_str(&text),
            Ok(Chunk::Reasoning(text)) => reasoning.push_str(&text),
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

pub fn stream(
    api_key: String,
    model: String,
    prompt: String,
    system: Option<String>,
    sender: Sender<Chunk>,
) {
    let resp = match make_request(&api_key, &model, &prompt, system.as_deref()) {
        Ok(resp) => resp,
        Err(err) => {
            let _ = sender.send(Chunk::Err(err));
            return;
        }
    };
    let mut state = StreamState {
        start: Instant::now(),
        ttft: None,
        tokens_in: 0,
        tokens_out: 0,
        cost_usd: None,
    };
    let reader = BufReader::new(resp);
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                let _ = sender.send(Chunk::Err(err.to_string()));
                return;
            }
        };
        let data = match line.strip_prefix("data: ") {
            Some(data) => data,
            None => continue,
        };
        if process_line(data, &mut state, &sender) {
            break;
        }
    }
    send_done(state, &sender);
}
