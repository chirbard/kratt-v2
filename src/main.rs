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
mod runlog;
mod session;
mod timeutil;
mod tui;
mod web;

use std::sync::Arc;

const DEFAULT_CHAT_MODEL: &str = "deepseek/deepseek-v4-flash";
const DEFAULT_MEMORY_MODEL: &str = "openai/gpt-oss-20b:nitro";
const DEFAULT_EMBED_MODEL: &str = "perplexity/pplx-embed-v1-0.6b";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let tavily_key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
    let chat_model = match std::env::var("CHAT_MODEL") {
        Ok(model) => model,
        Err(_) => DEFAULT_CHAT_MODEL.to_string(),
    };
    let memory_model = match std::env::var("MEMORY_MODEL") {
        Ok(model) => model,
        Err(_) => DEFAULT_MEMORY_MODEL.to_string(),
    };
    let embed_model = DEFAULT_EMBED_MODEL.to_string();
    let chat_llm: Arc<dyn api::LlmClient> = Arc::new(api::new_client(api_key.clone(), chat_model));
    let memory_llm: Arc<dyn api::LlmClient> =
        Arc::new(api::new_client(api_key.clone(), memory_model));
    let embed: Arc<dyn api::EmbeddingClient> =
        Arc::new(api::new_embed_client(api_key.clone(), embed_model));
    let search: Option<Arc<dyn web::SearchClient>> = if tavily_key.is_empty() {
        None
    } else {
        Some(Arc::new(web::new_search_client(tavily_key)))
    };
    memory::sync_pull_best_effort();
    if let Err(err) = memory::db_init() {
        eprintln!("warning: db init failed: {err}");
    }
    let (embed_tx, _embed_rx) = std::sync::mpsc::channel::<memory::MemoryEvent>();
    memory::run_embed_pass(Arc::clone(&embed), &embed_tx);
    let session_path = session::new_session_path();
    let result = tui::run(chat_llm, memory_llm, embed, search, session_path);
    memory::sync_push_best_effort("app exit");
    result
}
