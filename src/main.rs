#![deny(clippy::disallowed_methods)]
#![warn(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::min_ident_chars
)]

mod api;
mod memory;
mod prompts;
mod session;
mod tui;
mod web;

use std::sync::Arc;

const DEFAULT_MODEL: &str = "deepseek/deepseek-v4-flash";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let tavily_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    let llm: Arc<dyn api::LlmClient> = Arc::new(api::new_client(api_key, DEFAULT_MODEL.to_string()));
    let search: Option<Arc<dyn web::SearchClient>> = if tavily_key.is_empty() {
        None
    } else {
        Some(Arc::new(web::new_search_client(tavily_key)))
    };
    let graph = memory::load_graph();
    let memory_toon = toon::encode(&memory::graph_to_json(&graph), None);
    let session_path = session::new_session_path();
    if let Err(err) = memory::git_init_if_needed() {
        eprintln!("warning: git init failed: {err}");
    }
    tui::run(llm, search, memory_toon, session_path)
}
