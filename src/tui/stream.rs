use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::api::{Chunk, EmbeddingClient, LlmClient, Stats, ToolCallRequest, TurnOutcome};
use crate::memory;
use crate::prompts;
use crate::session;
use crate::web::SearchClient;

use super::app::{App, Message, Role, State};

pub(crate) struct StreamJob {
    pub(crate) llm: Arc<dyn LlmClient>,
    pub(crate) embed: Arc<dyn EmbeddingClient>,
    pub(crate) search: Option<Arc<dyn SearchClient>>,
    pub(crate) prompt: String,
    pub(crate) session_path: std::path::PathBuf,
    pub(crate) sender: std::sync::mpsc::Sender<Chunk>,
}

struct StatsAgg {
    tokens_in: u32,
    tokens_out: u32,
    cost_usd: f64,
    ttft_ms: u128,
}

pub(crate) fn start_stream(app: &mut App) {
    let prompt = std::mem::take(&mut app.input);
    app.last_user_message_id = session::append_entry(&app.session_path, "user", &prompt);
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
    let job = StreamJob {
        llm: Arc::clone(&app.chat_llm),
        embed: Arc::clone(&app.embed),
        search: clone_search_client(app),
        prompt,
        session_path: app.session_path.clone(),
        sender,
    };
    std::thread::spawn(move || run_stream_worker(job));
}

fn clone_search_client(app: &App) -> Option<Arc<dyn SearchClient>> {
    match &app.search {
        Some(arc) => Some(Arc::clone(arc)),
        None => None,
    }
}

fn run_stream_worker(job: StreamJob) {
    let recent_entries = session::load_recent_entries(&job.session_path, super::app::RECENT_TURNS);
    let memory_block = build_memory_block(&*job.embed, &job.prompt);
    let system = prompts::build_chat_system_prompt(&memory_block, &recent_entries);
    let system_opt = if system.is_empty() {
        None
    } else {
        Some(system)
    };
    let mut messages = build_initial_messages(&job.prompt, system_opt.as_deref());
    let tools = build_tools(job.search.is_some());
    let mut stats_agg = StatsAgg {
        tokens_in: 0,
        tokens_out: 0,
        cost_usd: 0.0,
        ttft_ms: 0,
    };
    let mut first_turn = true;
    loop {
        match job.llm.stream_turn(&messages, Some(&tools), &job.sender) {
            Ok(TurnOutcome::Done(stats)) => {
                send_final_stats(&job.sender, stats, &mut stats_agg, first_turn);
                return;
            }
            Ok(TurnOutcome::ToolCalls(calls)) => {
                execute_tool_turn(&job, &calls, &mut messages);
            }
            Err(err) => {
                let _ = job.sender.send(Chunk::Err(err.to_string()));
                return;
            }
        }
        first_turn = false;
    }
}

fn build_memory_block(embed: &dyn EmbeddingClient, query: &str) -> String {
    let edges = memory::select_semantic_edges(embed, query);
    if edges.is_empty() {
        return String::new();
    }
    memory::edges_to_toon(&edges)
}

fn build_initial_messages(prompt: &str, system_opt: Option<&str>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system) = system_opt {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

fn send_final_stats(
    sender: &std::sync::mpsc::Sender<Chunk>,
    stats: Stats,
    stats_agg: &mut StatsAgg,
    first_turn: bool,
) {
    stats_agg.tokens_in += stats.tokens_in;
    stats_agg.tokens_out += stats.tokens_out;
    stats_agg.cost_usd += stats.cost_usd.unwrap_or(0.0);
    if first_turn {
        stats_agg.ttft_ms = stats.ttft_ms;
    }
    let cost_usd = if stats_agg.cost_usd > 0.0 {
        Some(stats_agg.cost_usd)
    } else {
        None
    };
    let _ = sender.send(Chunk::Done(Stats {
        tokens_in: stats_agg.tokens_in,
        tokens_out: stats_agg.tokens_out,
        cost_usd,
        ttft_ms: stats_agg.ttft_ms,
    }));
}

fn execute_tool_turn(job: &StreamJob, calls: &[ToolCallRequest], messages: &mut Vec<Value>) {
    messages.push(build_tool_call_message(calls));
    for call in calls {
        let _ = job
            .sender
            .send(Chunk::ToolCall(format!("{}: running…", call.name)));
        let result = execute_tool(call, job.search.as_deref(), &job.session_path);
        let _ = job
            .sender
            .send(Chunk::ToolCall(format!("{}: done", call.name)));
        messages.push(json!({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result
        }));
    }
}

fn build_tool_call_message(calls: &[ToolCallRequest]) -> Value {
    let mut tool_calls_json = Vec::new();
    for call in calls {
        tool_calls_json.push(json!({
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.args}
        }));
    }
    json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": tool_calls_json
    })
}

fn build_tools(has_search: bool) -> Value {
    let mut tools = Vec::new();
    if has_search {
        tools.push(search_web_tool());
    }
    tools.push(search_history_tool());
    tools.push(read_history_tool());
    tools.push(read_source_tool());
    Value::Array(tools)
}

fn search_web_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information: version numbers, release dates, recent events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-3 targeted queries"
                    }
                },
                "required": ["queries"]
            }
        }
    })
}

fn search_history_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search past conversation sessions by keyword. Use 3-5 short keywords with different phrasings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["queries"]
            }
        }
    })
}

fn read_history_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "read_history",
            "description": "Read a specific past turn from the current session. offset=0 is most recent, 1 is one before that.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {"type": "integer"}
                },
                "required": ["offset"]
            }
        }
    })
}

fn read_source_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "Read the original conversation exchange that produced a specific memory edge. Use this to verify where a belief came from. Pass the edge's source and target node names exactly as they appear in your memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Edge source node name"},
                    "target": {"type": "string", "description": "Edge target node name"}
                },
                "required": ["source", "target"]
            }
        }
    })
}

fn execute_tool(
    call: &ToolCallRequest,
    search: Option<&dyn SearchClient>,
    session_path: &Path,
) -> String {
    match call.name.as_str() {
        "search_web" => execute_search_web(&call.args, search),
        "search_history" => execute_search_history(&call.args),
        "read_history" => execute_read_history(&call.args, session_path),
        "read_source" => execute_read_source(&call.args),
        _ => "unknown tool".to_string(),
    }
}

fn execute_search_web(args: &str, search: Option<&dyn SearchClient>) -> String {
    let queries = parse_queries(args);
    if queries.is_empty() {
        return "no results".to_string();
    }
    let Some(search_client) = search else {
        return "no results".to_string();
    };
    match search_client.search_many(&queries) {
        Some(result) if !result.is_empty() => result,
        _ => "no results".to_string(),
    }
}

fn execute_search_history(args: &str) -> String {
    let queries = parse_queries(args);
    if queries.is_empty() {
        return "no results".to_string();
    }
    let result = session::search_history(&queries);
    if result.is_empty() {
        return "no results".to_string();
    }
    result
}

fn execute_read_history(args: &str, session_path: &Path) -> String {
    let offset = match parse_offset(args) {
        Some(offset) => offset,
        None => return "no results".to_string(),
    };
    let result = session::read_history(session_path, offset);
    if result.is_empty() {
        return "no results".to_string();
    }
    result
}

fn execute_read_source(args: &str) -> String {
    let root = match serde_json::from_str::<Value>(args) {
        Ok(root) => root,
        Err(_) => return "no results".to_string(),
    };
    let source = match parse_string_field(&root, "source") {
        Some(source) => source,
        None => return "no results".to_string(),
    };
    let target = match parse_string_field(&root, "target") {
        Some(target) => target,
        None => return "no results".to_string(),
    };
    let result = memory::read_message_context(&source, &target);
    if result.is_empty() {
        return "no results".to_string();
    }
    result
}

fn parse_queries(args: &str) -> Vec<String> {
    let root = match serde_json::from_str::<Value>(args) {
        Ok(root) => root,
        Err(_) => return Vec::new(),
    };
    let queries_value = match root.get("queries") {
        Some(queries_value) => queries_value,
        None => return Vec::new(),
    };
    let queries_array = match queries_value.as_array() {
        Some(queries_array) => queries_array,
        None => return Vec::new(),
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
    queries
}

fn parse_offset(args: &str) -> Option<usize> {
    let root = match serde_json::from_str::<Value>(args) {
        Ok(root) => root,
        Err(_) => return None,
    };
    let offset_value = root.get("offset")?;
    let offset = offset_value.as_u64()?;
    usize::try_from(offset).ok()
}

fn parse_string_field(root: &Value, key: &str) -> Option<String> {
    let value = root.get(key)?;
    let text = value.as_str()?;
    Some(text.to_string())
}
