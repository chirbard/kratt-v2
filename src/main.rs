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
mod paths;
mod prompts;
mod runlog;
mod session;
mod telegram;
mod timeutil;
mod trace;
mod tui;
mod turn;
mod web;

use std::sync::Arc;

const DEFAULT_CHAT_MODEL: &str = "deepseek/deepseek-v4-flash";
const DEFAULT_MEMORY_MODEL: &str = "openai/gpt-oss-20b:nitro";
const DEFAULT_EMBED_MODEL: &str = "perplexity/pplx-embed-v1-0.6b";

type MainResult<Output> = Result<Output, Box<dyn std::error::Error>>;

// Panics land in the runlog before the default hook prints them — under a
// daemon (systemd, detached terminal) the runlog is the only place they are
// seen at all. Observed live: turso pager panics inside tokio tasks survived
// the process but aborted DB operations invisibly (backlog: turso pager panic).
fn install_panic_runlog() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        runlog::log("panic", &info.to_string());
        default_hook(info);
    }));
}

fn main() -> MainResult<()> {
    let trace_view = trace_view_target()?;
    // All data paths (kratt-data/, logs, .env) are relative — pin the working
    // directory to the crate root so `cargo run` from a subdirectory doesn't
    // scatter kratt-data/ copies around the repo.
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;
    install_panic_runlog();
    if let Some(target) = trace_view {
        return trace::view(&target);
    }
    run_application()
}

fn trace_view_target() -> MainResult<Option<String>> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--trace-view" {
            return match args.next() {
                Some(target) => Ok(Some(target)),
                None => Err("--trace-view requires a flow folder or day".into()),
            };
        }
    }
    Ok(None)
}

fn run_application() -> MainResult<()> {
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
    let embed: Arc<dyn api::EmbeddingClient> =
        Arc::new(api::new_embed_client(api_key.clone(), embed_model));
    // The three memory seats. Heavy is always MEMORY_MODEL; classifier and
    // scribe default to it unless CLASSIFIER_MODEL/SCRIBE_MODEL override them
    // (which may be an "ollama:" local model). Unset → all three identical.
    let seats = build_memory_seats(&api_key, &memory_model, Arc::clone(&embed));
    let search: Option<Arc<dyn web::SearchClient>> = if tavily_key.is_empty() {
        None
    } else {
        Some(Arc::new(web::new_search_client(tavily_key)))
    };
    initialize_database()?;
    let (embed_tx, _embed_rx) = std::sync::mpsc::channel::<memory::MemoryEvent>();
    memory::run_embed_pass(Arc::clone(&embed), &embed_tx);
    // Same binary, two faces: --telegram runs the headless gateway daemon, no
    // args runs the interactive TUI. One process owns kratt-data/.
    let result = if telegram_mode() {
        telegram::run(chat_llm, seats, embed, search)
    } else {
        let session_path = session::new_session_path();
        tui::run(chat_llm, seats, embed, search, session_path)
    };
    memory::sync_push_best_effort("app exit");
    result
}

fn initialize_database() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(err) = memory::startup_database() {
        return Err(format!("startup database recovery failed: {err}").into());
    }
    if let Err(err) = memory::db_init() {
        return Err(format!("db init failed: {err}").into());
    }
    Ok(())
}

fn telegram_mode() -> bool {
    for arg in std::env::args() {
        if arg == "--telegram" {
            return true;
        }
    }
    false
}

// Heavy is always MEMORY_MODEL. classifier/scribe follow their own env spec
// (which may be an "ollama:" local model); unset falls back to MEMORY_MODEL so
// default configuration is byte-identical to the pre-split single client.
fn build_memory_seats(
    api_key: &str,
    memory_model: &str,
    embed: Arc<dyn api::EmbeddingClient>,
) -> memory::MemorySeats {
    let heavy = api::resolve_model_client(memory_model, api_key);
    let classifier = match std::env::var("CLASSIFIER_MODEL") {
        Ok(spec) => api::resolve_model_client(&spec, api_key),
        Err(_) => Arc::clone(&heavy),
    };
    let scribe = match std::env::var("SCRIBE_MODEL") {
        Ok(spec) => api::resolve_model_client(&spec, api_key),
        Err(_) => Arc::clone(&heavy),
    };
    memory::MemorySeats {
        classifier,
        scribe,
        heavy,
        embed,
    }
}
