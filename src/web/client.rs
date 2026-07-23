use std::error::Error;

use super::{NewsItem, SearchClient};

const TAVILY_URL: &str = "https://api.tavily.com/search";
const TAVILY_EXTRACT_URL: &str = "https://api.tavily.com/extract";

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

    fn search_news(&self, topic: &str, days: usize) -> Result<Vec<NewsItem>, String> {
        let value = match tavily_news_post(&self.api_key, topic, days) {
            Ok(response) => response,
            Err(err) => return Err(err.to_string()),
        };
        Ok(parse_news_results(&value))
    }

    fn extract_url(&self, url: &str) -> Result<String, String> {
        match tavily_extract(&self.api_key, url) {
            Ok(text) => Ok(text),
            Err(err) => Err(err.to_string()),
        }
    }
}

#[allow(dead_code)]
fn tavily_news_post(
    api_key: &str,
    topic: &str,
    days: usize,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let body = serde_json::json!({
        "api_key": api_key,
        "query": topic,
        "topic": "news",
        "days": days,
        "search_depth": "basic",
        "max_results": 3,
    });
    tavily_request(&body)
}

fn tavily_extract(api_key: &str, url: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let body = serde_json::json!({"api_key": api_key, "urls": [url]});
    let response = reqwest::blocking::Client::new()
        .post(TAVILY_EXTRACT_URL)
        .json(&body)
        .send()?;
    if !response.status().is_success() {
        return Err(format!(
            "HTTP {}: {}",
            response.status(),
            response.text().unwrap_or_default()
        )
        .into());
    }
    let value = response.json::<serde_json::Value>()?;
    let content = match value.get("results") {
        Some(results) => match results.as_array() {
            Some(results) => match results.first() {
                Some(result) => get_str_field(result, "raw_content"),
                None => "",
            },
            None => "",
        },
        None => "",
    };
    if content.is_empty() {
        return Err("Tavily extract returned no text".into());
    }
    Ok(content.to_string())
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
    tavily_request(&body)
}

fn tavily_request(
    body: &serde_json::Value,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let resp = reqwest::blocking::Client::new()
        .post(TAVILY_URL)
        .json(body)
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

#[allow(dead_code)]
fn parse_news_results(value: &serde_json::Value) -> Vec<NewsItem> {
    let Some(results_value) = value.get("results") else {
        return Vec::new();
    };
    let Some(results) = results_value.as_array() else {
        return Vec::new();
    };
    let mut items = Vec::new();
    for result in results {
        let title = get_str_field(result, "title").trim();
        let url = get_str_field(result, "url").trim();
        let snippet = get_str_field(result, "content").trim();
        if title.is_empty() || url.is_empty() {
            continue;
        }
        items.push(NewsItem {
            title: title.to_string(),
            url: url.to_string(),
            snippet: snippet.to_string(),
        });
    }
    items
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
