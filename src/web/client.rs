use std::error::Error;

use super::SearchClient;

const TAVILY_URL: &str = "https://api.tavily.com/search";

pub(super) struct TavilyClient {
    api_key: String,
}

impl TavilyClient {
    pub(super) fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

impl SearchClient for TavilyClient {
    fn search_many(&self, queries: &[String]) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        for query in queries {
            let Ok(value) = tavily_post(&self.api_key, query) else {
                continue;
            };
            let section = format_search_section(query, &value);
            if !section.is_empty() {
                parts.push(section);
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

fn format_search_section(query: &str, value: &serde_json::Value) -> String {
    let results = format_results(value);
    if results.is_empty() {
        return String::new();
    }
    format!("[Search: \"{query}\"]\n{results}")
}

fn tavily_post(
    api_key: &str,
    query: &str,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let body = serde_json::json!({
        "api_key": api_key,
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
