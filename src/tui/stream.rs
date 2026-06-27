use std::path::Path;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient};
use crate::prompts;
use crate::session;
use crate::web::{self, SearchClient};

use super::app::{App, Message, Role, State, RECENT_TURNS};

pub(crate) struct StreamJob {
    pub(crate) llm: Arc<dyn LlmClient>,
    pub(crate) search: Option<Arc<dyn SearchClient>>,
    pub(crate) prompt: String,
    pub(crate) session_path: std::path::PathBuf,
    pub(crate) memory_toon: String,
    pub(crate) system_opt: Option<String>,
    pub(crate) sender: std::sync::mpsc::Sender<Chunk>,
}

pub(crate) fn start_stream(app: &mut App) {
    let prompt = std::mem::take(&mut app.input);
    let recent_entries = session::load_recent_entries(&app.session_path, RECENT_TURNS);
    session::append_entry(&app.session_path, "user", &prompt);
    app.history.push(Message {
        role: Role::User,
        reasoning: String::new(),
        content: prompt.clone(),
        tool_call: String::new(),
    });
    app.history.push(Message {
        role: Role::Assistant,
        reasoning: String::new(),
        content: String::new(),
        tool_call: String::new(),
    });
    app.stats = None;
    app.auto_scroll = true;
    app.state = State::Streaming;
    let (sender, recvr) = std::sync::mpsc::channel();
    app.stream_rx = Some(recvr);
    let system = prompts::build_chat_system_prompt(&app.memory_toon, &recent_entries);
    let system_opt = if system.is_empty() { None } else { Some(system) };
    let job = StreamJob {
        llm: Arc::clone(&app.chat_llm),
        search: match &app.search {
            Some(arc) => Some(Arc::clone(arc)),
            None => None,
        },
        prompt,
        session_path: app.session_path.clone(),
        memory_toon: app.memory_toon.clone(),
        system_opt,
        sender,
    };
    std::thread::spawn(move || run_stream_worker(job));
}

fn run_stream_worker(job: StreamJob) {
    let mut system_opt = job.system_opt;
    let (stream_prompt, tool_desc) =
        maybe_augment_prompt_with_history(job.llm.as_ref(), &job.prompt, &job.session_path);
    if let Some(desc) = tool_desc {
        let _ = job.sender.send(Chunk::ToolCall(desc));
    }
    if let Some(queries) = web::decide_queries(job.llm.as_ref(), &stream_prompt, &job.memory_toon) {
        let display = queries.join(" | ");
        let _ = job.sender.send(Chunk::ToolCall(format!("checking web: {display}…")));
        if let Some(ref search) = job.search
            && let Some(context) = search.search_many(&queries)
        {
            if let Some(ref mut sys) = system_opt {
                sys.push_str("\n\n<web_context>\n");
                sys.push_str(&context);
                sys.push_str("\n</web_context>");
            }
            let _ = job.sender.send(Chunk::ToolCall(format!("web: {display}")));
        }
    }
    job.llm.stream(stream_prompt, system_opt, job.sender);
}

enum HistoryResult {
    Found(String, String),
    NotFound(String, String),
    Skip,
}

fn maybe_augment_prompt_with_history(
    llm: &dyn LlmClient,
    prompt: &str,
    session_path: &Path,
) -> (String, Option<String>) {
    if !should_lookup_history(prompt) {
        return (prompt.to_string(), None);
    }
    let lookup_prompt = prompts::build_history_lookup_prompt(prompt);
    let lookup_response = match llm.collect(&lookup_prompt) {
        Ok(lookup_response) => lookup_response,
        Err(_) => return (prompt.to_string(), None),
    };
    match parse_history_response(&lookup_response, session_path) {
        HistoryResult::Skip => (prompt.to_string(), None),
        HistoryResult::Found(text, ui_desc) => {
            let mut augmented = prompt.to_string();
            augmented.push_str(prompts::HISTORY_FOUND_PREFIX);
            augmented.push_str(&text);
            (augmented, Some(ui_desc))
        }
        HistoryResult::NotFound(keywords, ui_desc) => {
            let mut augmented = prompt.to_string();
            augmented.push_str(prompts::HISTORY_NOT_FOUND_PREFIX);
            augmented.push_str(&keywords);
            augmented.push_str(prompts::HISTORY_NOT_FOUND_SUFFIX);
            (augmented, Some(ui_desc))
        }
    }
}

fn should_lookup_history(prompt: &str) -> bool {
    let prompt_lower = prompt.to_ascii_lowercase();
    for keyword in prompts::HISTORY_LOOKUP_KEYWORDS {
        if prompt_lower.contains(keyword) {
            return true;
        }
    }
    false
}

fn parse_history_response(response: &str, session_path: &Path) -> HistoryResult {
    let trimmed = response.trim();
    if trimmed == "skip" {
        return HistoryResult::Skip;
    }
    let call_value = match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(call_value) => call_value,
        Err(_) => return HistoryResult::Skip,
    };
    let tool_entry = match call_value.get("tool") {
        Some(entry) => entry,
        None => return HistoryResult::Skip,
    };
    let tool_name = match tool_entry.as_str() {
        Some(name) => name,
        None => return HistoryResult::Skip,
    };
    let arguments = match call_value.get("args") {
        Some(val) => val,
        None => match call_value.get("arguments") {
            Some(val) => val,
            None => return HistoryResult::Skip,
        },
    };
    match tool_name {
        "read_history" => execute_read_history(arguments, session_path),
        "search_history" => execute_search_history(arguments),
        _ => HistoryResult::Skip,
    }
}

fn execute_read_history(arguments: &serde_json::Value, session_path: &Path) -> HistoryResult {
    let offset_entry = match arguments.get("offset") {
        Some(entry) => entry,
        None => return HistoryResult::Skip,
    };
    let offset_value = match offset_entry.as_u64() {
        Some(val) => val,
        None => return HistoryResult::Skip,
    };
    let offset = match usize::try_from(offset_value) {
        Ok(offset) => offset,
        Err(_) => return HistoryResult::Skip,
    };
    let text = session::read_history(session_path, offset);
    if text.is_empty() {
        HistoryResult::NotFound(
            format!("turn offset {offset}"),
            format!("read_history offset={offset} → no results"),
        )
    } else {
        HistoryResult::Found(text, format!("read_history offset={offset} → found"))
    }
}

fn execute_search_history(arguments: &serde_json::Value) -> HistoryResult {
    let queries_entry = match arguments.get("queries") {
        Some(entry) => entry,
        None => return HistoryResult::Skip,
    };
    let queries_array = match queries_entry.as_array() {
        Some(arr) => arr,
        None => return HistoryResult::Skip,
    };
    let mut queries = Vec::new();
    for item in queries_array {
        if let Some(query) = item.as_str() {
            let trimmed = query.trim();
            if !trimmed.is_empty() {
                queries.push(trimmed.to_string());
            }
        }
    }
    if queries.is_empty() {
        return HistoryResult::Skip;
    }
    let keywords_display = queries.join(", ");
    let text = session::search_history(&queries);
    if text.is_empty() {
        HistoryResult::NotFound(
            keywords_display.clone(),
            format!("search_history [{keywords_display}] → no results"),
        )
    } else {
        HistoryResult::Found(text, format!("search_history [{keywords_display}] → found"))
    }
}
