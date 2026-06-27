use std::error::Error;

use crate::api;
use crate::memory_prompt::today_iso;

const TAVILY_URL: &str = "https://api.tavily.com/search";

const WEB_CHECK_EXPLANATION: &str = r#"You are a web fact-check decision agent. Before the user receives an answer, decide if a web search is needed to verify a factual claim.

Search when the user message:
- Mentions a specific software version, release, or changelog (e.g. "Python 3.14", "Rust 2.0")
- Asserts a date, news item, or world event that might post-date your training knowledge
- Contains a claim that contradicts what is already in the memory graph
- Asks about the current state of something that changes over time

Skip when the message:
- Is conversational, personal, or opinion-based
- Is about something already verified in memory or timeless knowledge
- Contains no verifiable factual claims

Output ONLY one of:
1. The single word: skip — if no web search is needed
2. Raw JSON with 2–3 targeted queries (no prose, no markdown): {"tool": "search", "arguments": {"queries": ["exact version or claim", "latest stable release", "release candidate or upcoming"]}}

Always provide separate queries for: the exact claim the user made, the current stable release, and any RC or upcoming release. This gives a complete picture and catches cases where the claimed version does not exist yet."#;

fn build_web_check_prompt(prompt: &str, memory_toon: &str) -> String {
    let mut text = String::from(WEB_CHECK_EXPLANATION);
    text.push_str("\nToday's date is ");
    text.push_str(&today_iso());
    text.push('.');
    if !memory_toon.trim().is_empty() {
        text.push_str("\n\n<memory>\n");
        text.push_str(memory_toon);
        text.push_str("\n</memory>");
    }
    text.push_str("\n\nUser message: ");
    text.push_str(prompt);
    text
}

pub fn decide_queries(api_key: &str, model: &str, prompt: &str, memory_toon: &str) -> Option<Vec<String>> {
    let check_prompt = build_web_check_prompt(prompt, memory_toon);
    let response = api::collect(api_key, model, &check_prompt).ok()?;
    parse_search_queries(&response)
}

fn parse_search_queries(response: &str) -> Option<Vec<String>> {
    let trimmed = response.trim();
    if trimmed == "skip" {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let tool = value.get("tool")?.as_str()?;
    if tool != "search" {
        return None;
    }
    let args = value.get("arguments")?;
    let arr = args.get("queries")?.as_array()?;
    let mut queries = Vec::new();
    for item in arr {
        let Some(query) = item.as_str() else { continue; };
        let query = query.trim();
        if !query.is_empty() {
            queries.push(query.to_string());
        }
    }
    if queries.is_empty() { None } else { Some(queries) }
}

pub fn execute_searches(tavily_key: &str, queries: &[String]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for query in queries {
        let Ok(value) = tavily_post(tavily_key, query) else { continue; };
        let section = format_search_section(query, &value);
        if !section.is_empty() {
            parts.push(section);
        }
    }
    if parts.is_empty() { None } else { Some(parts.join("\n\n")) }
}

fn format_search_section(query: &str, value: &serde_json::Value) -> String {
    let results = format_results(value);
    if results.is_empty() {
        return String::new();
    }
    format!("[Search: \"{query}\"]\n{results}")
}

fn tavily_post(
    tavily_key: &str,
    query: &str,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let body = serde_json::json!({
        "api_key": tavily_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
    });
    let resp = reqwest::blocking::Client::new()
        .post(TAVILY_URL)
        .json(&body)
        .send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        return Err(format!("HTTP {status}: {text}").into());
    }
    match resp.json::<serde_json::Value>() {
        Ok(value) => Ok(value),
        Err(err) => Err(err.into()),
    }
}

fn get_str_field<'rf>(obj: &'rf serde_json::Value, key: &str) -> &'rf str {
    let Some(val) = obj.get(key) else {
        return "";
    };
    val.as_str().unwrap_or("")
}

fn format_results(value: &serde_json::Value) -> String {
    let results_val = match value.get("results") {
        Some(result_value) => result_value,
        None => return String::new(),
    };
    let results = match results_val.as_array() {
        Some(arr) => arr,
        None => return String::new(),
    };
    let mut out = String::new();
    for (idx, result) in results.iter().enumerate() {
        let title = get_str_field(result, "title");
        let url = get_str_field(result, "url");
        let content = get_str_field(result, "content");
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("[{}] {}\n{}\n{}", idx + 1, title, url, content));
    }
    out
}
