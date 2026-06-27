use std::error::Error;

use crate::api;
use crate::prompts;

const TAVILY_URL: &str = "https://api.tavily.com/search";

pub fn decide_queries(api_key: &str, model: &str, prompt: &str, memory_toon: &str) -> Option<Vec<String>> {
    let check_prompt = prompts::build_web_check_prompt(prompt, memory_toon);
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
