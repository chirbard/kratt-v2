use std::error::Error;
use std::path::Path;
use std::sync::mpsc::{Sender, channel};
use std::time::Instant;

use serde_json::{Value, json};

use crate::api::{Chunk, EmbeddingClient, LlmClient, ToolCallRequest, TurnOutcome};
use crate::memory;
use crate::prompts;
use crate::runlog;
use crate::session;
use crate::web::SearchClient;

// How many prior user/assistant turns to replay into the system prompt.
pub(crate) const RECENT_TURNS: usize = 2;

// Front-end-agnostic turn pipeline: memory-block assembly, initial messages,
// the tool loop, and routing/attribution helpers. Both the TUI streamer and
// the Telegram gateway drive the same logic through here — the TUI streams the
// chunks to a panel, the gateway collects them into one reply.
pub(crate) struct MemoryBuild {
    pub block: String,
    pub debug_log: String,
    pub injected_edges: usize,
    pub elapsed_ms: u128,
}

pub(crate) struct TurnContext<'ctx> {
    pub embed: &'ctx dyn EmbeddingClient,
    pub search: Option<&'ctx dyn SearchClient>,
    pub prompt: &'ctx str,
    pub session_path: &'ctx Path,
    pub turn_id: &'ctx str,
    // Front-end formatting constraints appended to the system prompt (e.g.
    // Telegram: no tables, phone-width answers). None for the TUI.
    pub style_hint: Option<&'ctx str>,
}

pub(crate) struct TurnClients<'cl> {
    pub llm: &'cl dyn LlmClient,
    pub embed: &'cl dyn EmbeddingClient,
    pub search: Option<&'cl dyn SearchClient>,
    pub style_hint: Option<&'cl str>,
}

pub(crate) fn prepare_stream_context(
    ctx: &TurnContext,
) -> Result<(Vec<Value>, Value, MemoryBuild), Box<dyn Error + Send + Sync>> {
    let recent_entries = session::load_recent_entries(ctx.session_path, RECENT_TURNS);
    let session_id = session::session_id_from_path(ctx.session_path);
    let memory = build_memory_block(ctx.embed, ctx.prompt, &session_id, ctx.turn_id);
    let mut system = prompts::build_chat_system_prompt(&memory.block, &recent_entries);
    if let Some(hint) = ctx.style_hint {
        system.push_str("\n\n");
        system.push_str(hint);
    }
    let system_opt = if system.is_empty() {
        None
    } else {
        Some(system)
    };
    Ok((
        build_initial_messages(ctx.prompt, system_opt.as_deref()),
        build_tools(ctx.search.is_some()),
        memory,
    ))
}

// Headless turn: run the same pipeline as the TUI but collect the streamed
// content into one answer string instead of rendering it live. The channel is
// unbounded, so draining it after the loop finishes cannot deadlock.
pub(crate) fn run_turn_collecting(
    clients: &TurnClients,
    prompt: &str,
    session_path: &Path,
    turn_id: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let ctx = TurnContext {
        embed: clients.embed,
        search: clients.search,
        prompt,
        session_path,
        turn_id,
        style_hint: clients.style_hint,
    };
    let (mut messages, tools, memory_build) = prepare_stream_context(&ctx)?;
    log_memory_context(&memory_build);
    let (sender, receiver) = channel::<Chunk>();
    loop {
        match clients.llm.stream_turn(&messages, Some(&tools), &sender)? {
            TurnOutcome::Done(_) => break,
            TurnOutcome::ToolCalls(calls) => {
                execute_tool_turn(clients.search, session_path, &sender, &calls, &mut messages);
            }
        }
    }
    drop(sender);
    Ok(collect_content(&receiver))
}

fn collect_content(receiver: &std::sync::mpsc::Receiver<Chunk>) -> String {
    let mut content = String::new();
    for chunk in receiver {
        if let Chunk::Content(text) = chunk {
            content.push_str(&text);
        }
    }
    content
}

pub(crate) fn log_memory_context(memory_build: &MemoryBuild) {
    runlog::log("routing", &memory_build.debug_log);
    runlog::log(
        "chat",
        &format!("injected edges: {}", memory_build.injected_edges),
    );
    runlog::log(
        "chat",
        &format!("memory context built in {}ms", memory_build.elapsed_ms),
    );
}

pub(crate) fn build_memory_block(
    embed: &dyn EmbeddingClient,
    query: &str,
    session_id: &str,
    turn_id: &str,
) -> MemoryBuild {
    let started = Instant::now();
    let selection = memory::select_semantic_edges_routed(embed, query, session_id);
    let edges = selection.edges;
    let mut events = Vec::new();
    events.push(memory::Event::routed(
        memory::EventContext::new(session_id, turn_id),
        routed_payload(&selection.routing.expert, selection.routing.expert_score),
    ));
    for edge in &edges {
        let edge_ref = memory::EdgeRef::new(&edge.source, &edge.target);
        push_injected(&mut events, (session_id, turn_id), "local", &edge_ref);
    }
    // The current game's move chain: complete and ordered, outside TOP_K. Its
    // edges get `injected` events too, so attribution and the spacing-effect
    // clock refresh apply exactly like any other injection. Actor id "game"
    // (vs "local" for ranked beliefs) marks them as part of the game's play —
    // the Phase 23 outcome slice reinforces game-tagged injections only,
    // otherwise a win would credit all 60 ranked beliefs of every turn.
    let chain = memory::load_task_chain(session_id);
    for edge_ref in &chain.refs {
        push_injected(&mut events, (session_id, turn_id), "game", edge_ref);
    }
    // The game's rules and derived strategies: deterministic, outside TOP_K.
    // A strategy used in a win must be attributable — same "game" tag.
    let knowledge = memory::load_game_knowledge(session_id);
    for edge_ref in &knowledge.refs {
        push_injected(&mut events, (session_id, turn_id), "game", edge_ref);
    }
    // Best-effort bookkeeping: a failed ledger write must never abort the turn.
    if let Err(err) = memory::record_events(&events) {
        runlog::log("ledger", &format!("write failed: {err}"));
    }
    let block = assemble_memory_block(&edges, &knowledge, &chain);
    MemoryBuild {
        block,
        debug_log: routing_debug_line(&selection.routing),
        injected_edges: edges.len() + chain.refs.len() + knowledge.refs.len(),
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn push_injected(
    events: &mut Vec<memory::Event>,
    context: (&str, &str),
    actor_id: &str,
    edge_ref: &memory::EdgeRef,
) {
    let (session_id, turn_id) = context;
    events.push(memory::Event::injected(
        memory::Actor::new("user", actor_id),
        edge_ref.clone(),
        memory::EventContext::new(session_id, turn_id),
    ));
}

// Belief section first (semantic-ranked TOON), then the game block (rules +
// strategies), then the ordered task chain. Any section may be empty.
fn assemble_memory_block(
    edges: &[memory::SemanticEdge],
    knowledge: &memory::GameKnowledge,
    chain: &memory::TaskChain,
) -> String {
    let mut block = if edges.is_empty() {
        String::new()
    } else {
        memory::edges_to_toon(edges)
    };
    append_section(&mut block, &knowledge.block);
    append_section(&mut block, &chain.block);
    block
}

fn append_section(block: &mut String, section: &str) {
    if section.is_empty() {
        return;
    }
    if !block.is_empty() {
        block.push_str("\n\n");
    }
    block.push_str(section);
}

fn routed_payload(expert: &Option<String>, score: f64) -> String {
    json!({
        "expert": expert,
        "score": score,
    })
    .to_string()
}

fn routing_debug_line(routing: &memory::Routing) -> String {
    let Some(expert) = &routing.expert else {
        return "routing: none (no positions)".to_string();
    };
    let suffix = if routing.expert_with_direction {
        " (pos+dir)"
    } else {
        " (pos)"
    };
    format!("routing: {expert} {:.2}{suffix}", routing.expert_score)
}

pub(crate) fn build_initial_messages(prompt: &str, system_opt: Option<&str>) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system) = system_opt {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    messages
}

pub(crate) fn format_attribution_line(attribution: &memory::Attribution) -> String {
    if attribution.injected_edges == 0 {
        return "memory: none — answered from model knowledge".to_string();
    }
    let expert = match &attribution.expert {
        Some(expert) => expert.clone(),
        None => "none".to_string(),
    };
    let mut line = format!(
        "memory: {} beliefs consulted · lens: {} ({:.2})",
        attribution.injected_edges, expert, attribution.expert_score
    );
    if !attribution.top_edges.is_empty() {
        line.push_str(" · top: ");
        line.push_str(&format_top_edges(&attribution.top_edges));
    }
    line
}

fn format_top_edges(edges: &[String]) -> String {
    let mut parts = Vec::new();
    for edge in edges {
        let short = shorten_belief(edge, 72);
        parts.push(format!("\"{short}\""));
    }
    parts.join(", ")
}

fn shorten_belief(text: &str, max_chars: usize) -> String {
    let mut output = String::new();
    let chars: Vec<char> = text.trim().chars().collect();
    let mut index = 0usize;
    while index < chars.len() {
        if index >= max_chars {
            output.push('…');
            break;
        }
        output.push(chars[index]);
        index += 1;
    }
    output
}

pub(crate) fn execute_tool_turn(
    search: Option<&dyn SearchClient>,
    session_path: &Path,
    sender: &Sender<Chunk>,
    calls: &[ToolCallRequest],
    messages: &mut Vec<Value>,
) {
    messages.push(build_tool_call_message(calls));
    for call in calls {
        let _ = sender.send(Chunk::ToolCall(format!("{}: running…", call.name)));
        let result = execute_tool(call, search, session_path);
        let _ = sender.send(Chunk::ToolCall(format!("{}: done", call.name)));
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

pub(crate) fn build_tools(has_search: bool) -> Value {
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

#[cfg(test)]
mod tests {
    use super::assemble_memory_block;
    use crate::memory::{GameKnowledge, TaskChain};

    fn knowledge(block: &str) -> GameKnowledge {
        GameKnowledge {
            block: block.to_string(),
            refs: Vec::new(),
        }
    }

    fn chain(block: &str) -> TaskChain {
        TaskChain {
            block: block.to_string(),
            refs: Vec::new(),
        }
    }

    #[test]
    fn game_block_and_chain_block_coexist_when_both_present() {
        let block = assemble_memory_block(
            &[],
            &knowledge("## Game: chess\nthe rules\n"),
            &chain("## Current task (in order)\nply 1 (assistant): e4\n"),
        );
        assert_eq!(
            block,
            "## Game: chess\nthe rules\n\n\n## Current task (in order)\nply 1 (assistant): e4\n"
        );
    }

    #[test]
    fn no_game_link_adds_no_game_block() {
        let block = assemble_memory_block(&[], &knowledge(""), &chain("## Current task (in order)\n"));
        assert_eq!(block, "## Current task (in order)\n");
    }
}
