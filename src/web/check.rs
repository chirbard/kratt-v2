use crate::api::LlmClient;
use crate::prompts;

pub fn decide_queries(llm: &dyn LlmClient, prompt: &str, memory_toon: &str) -> Option<Vec<String>> {
    let check_prompt = prompts::build_web_check_prompt(prompt, memory_toon);
    let response = llm.collect(&check_prompt).ok()?;
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
    let args = match value.get("args") {
        Some(val) => val,
        None => value.get("arguments")?,
    };
    let arr = args.get("queries")?.as_array()?;
    let mut queries = Vec::new();
    for item in arr {
        let Some(query) = item.as_str() else {
            continue;
        };
        let query = query.trim();
        if !query.is_empty() {
            queries.push(query.to_string());
        }
    }
    if queries.is_empty() {
        None
    } else {
        Some(queries)
    }
}
