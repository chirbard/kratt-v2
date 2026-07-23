use std::collections::BTreeSet;
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
use crate::trace::TraceContext;
use crate::web::SearchClient;

// How many prior user/assistant turns to replay into the system prompt.
pub(crate) const RECENT_TURNS: usize = 2;
pub(crate) const MAX_TOOL_ROUNDS: usize = 5;
const RECALL_K: usize = 7; // replay-tunable

// Front-end-agnostic turn pipeline: memory-block assembly, initial messages,
// the tool loop, and retrieval/attribution helpers. Both the TUI streamer and
// the Telegram gateway drive the same logic through here — the TUI streams the
// chunks to a panel, the gateway collects them into one reply.
pub(crate) struct MemoryBuild {
    pub block: String,
    pub debug_log: String,
    pub injected_edges: usize,
    pub elapsed_ms: u128,
}

struct MemorySections {
    loops: memory::OpenLoops,
    knowledge: memory::GameKnowledge,
    chain: memory::TaskChain,
}

impl MemorySections {
    fn injected_edges(&self) -> usize {
        self.loops.refs.len() + self.knowledge.refs.len() + self.chain.refs.len()
    }
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
    // Yesterday's session file, when the front-end shows one continuous
    // conversation across the session boundary (Telegram): a fresh day
    // session tops up its recent context from this tail, so the model wakes
    // up seeing the same last messages the user's screen shows.
    pub prior_session_path: Option<&'ctx Path>,
}

pub(crate) struct TurnClients<'cl> {
    pub llm: &'cl dyn LlmClient,
    pub embed: &'cl dyn EmbeddingClient,
    pub search: Option<&'cl dyn SearchClient>,
    pub style_hint: Option<&'cl str>,
    pub trace: &'cl TraceContext,
}

pub(crate) fn prepare_stream_context(
    ctx: &TurnContext,
) -> Result<(Vec<Value>, Value, MemoryBuild), Box<dyn Error + Send + Sync>> {
    let recent_entries = load_recent_with_prior(ctx);
    let session_id = session::session_id_from_path(ctx.session_path);
    let sequence = memory::sequence_line(&session_id, ctx.turn_id);
    if let Some(line) = &sequence {
        crate::runlog::log("chat", &format!("sequence: {line}"));
    }
    let memory = build_memory_block(ctx.embed, ctx.prompt, &session_id, ctx.turn_id);
    let pressure = memory::pressure_line();
    if let Some(line) = &pressure {
        crate::runlog::log("chat", &format!("pressure: {line}"));
    }
    let system = prompts::build_chat_system_prompt(
        &memory.block,
        &recent_entries,
        pressure.as_deref(),
        sequence.as_deref(),
        ctx.style_hint,
    );
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
    paths: (&Path, Option<&Path>),
    turn_id: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let _turn_timer = runlog::time_step("chat", "turn total");
    let (session_path, prior_session_path) = paths;
    let ctx = TurnContext {
        embed: clients.embed,
        search: clients.search,
        prompt,
        session_path,
        turn_id,
        style_hint: clients.style_hint,
        prior_session_path,
    };
    let (mut messages, tools, memory_build) = prepare_stream_context(&ctx)?;
    log_memory_context(&memory_build);
    let (sender, receiver) = channel::<Chunk>();
    let context = ToolExecutionContext {
        embed: clients.embed,
        search: clients.search,
        session_path,
        sender: &sender,
    };
    run_collecting_tool_loop(clients, &tools, &mut messages, &context)?;
    drop(sender);
    Ok(collect_content(&receiver))
}

fn run_collecting_tool_loop<'context>(
    clients: &TurnClients,
    tools: &Value,
    messages: &mut Vec<Value>,
    context: &ToolExecutionContext<'context>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut tool_rounds = 0usize;
    loop {
        let available_tools = tools_for_round(tools, tool_rounds);
        let outcome = {
            let _stream_timer = runlog::time_step("chat", "LLM stream turn");
            clients
                .llm
                .stream_turn(messages, available_tools, context.sender)?
        };
        match outcome {
            TurnOutcome::Done(_) => break,
            TurnOutcome::ToolCalls(calls) => {
                if available_tools.is_none() {
                    runlog::log(
                        "chat",
                        "tool round cap: final generation returned tool calls",
                    );
                    break;
                }
                execute_tool_turn(context, &calls, messages, clients.trace);
                tool_rounds += 1;
            }
        }
    }
    Ok(())
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
    runlog::log("retrieval", &memory_build.debug_log);
    runlog::log(
        "chat",
        &format!("injected edges: {}", memory_build.injected_edges),
    );
    runlog::log(
        "chat",
        &format!("memory context built in {}ms", memory_build.elapsed_ms),
    );
}

// Today's recent turns, topped up from yesterday's session when today is
// still fresh — the Telegram page shows one continuous conversation, so the
// model's context should too. The elapsed-time note in the system prompt
// tells it how long ago those messages were.
fn load_recent_with_prior(ctx: &TurnContext) -> Vec<session::RecentEntry> {
    let recent = session::load_recent_entries(ctx.session_path, RECENT_TURNS);
    let Some(prior) = ctx.prior_session_path else {
        return recent;
    };
    let mut pairs = 0usize;
    for entry in &recent {
        if entry.role == "user" {
            pairs += 1;
        }
    }
    if pairs >= RECENT_TURNS {
        return recent;
    }
    let mut merged = session::load_recent_entries(prior, RECENT_TURNS - pairs);
    merged.extend(recent);
    merged
}

pub(crate) fn build_memory_block(
    embed: &dyn EmbeddingClient,
    query: &str,
    session_id: &str,
    turn_id: &str,
) -> MemoryBuild {
    let started = Instant::now();
    let selection = memory::select_semantic_edges_for_chat(embed, query, session_id);
    let edges = selection.edges;
    let sections = load_memory_sections(&edges, session_id, &selection.query_embedding);
    let events = build_memory_events((session_id, turn_id), &edges, &sections);
    // Best-effort bookkeeping: a failed ledger write must never abort the turn.
    if let Err(err) = memory::record_events(&events) {
        runlog::log("ledger", &format!("write failed: {err}"));
    }
    let block = assemble_memory_block(
        &edges,
        &sections.loops,
        &sections.knowledge,
        &sections.chain,
    );
    MemoryBuild {
        block,
        debug_log: "retrieval: plain semantic ranking".to_string(),
        injected_edges: edges.len() + sections.injected_edges(),
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn load_memory_sections(
    edges: &[memory::SemanticEdge],
    session_id: &str,
    query_vec: &[f64],
) -> MemorySections {
    let anchor_ids = collect_anchor_ids(edges);
    let today = crate::timeutil::today_iso();
    MemorySections {
        loops: memory::load_open_loops(&anchor_ids, &today, query_vec),
        knowledge: memory::load_game_knowledge(session_id),
        chain: memory::load_task_chain(session_id),
    }
}

fn collect_anchor_ids(edges: &[memory::SemanticEdge]) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for edge in edges {
        ids.insert(edge.source.clone());
        ids.insert(edge.target.clone());
    }
    ids.into_iter().collect()
}

fn build_memory_events(
    context: (&str, &str),
    edges: &[memory::SemanticEdge],
    sections: &MemorySections,
) -> Vec<memory::Event> {
    let mut events = Vec::new();
    push_ranked_injected(&mut events, context, edges);
    push_injected_refs(&mut events, context, "local", &sections.loops.refs);
    push_injected_refs(&mut events, context, "game", &sections.chain.refs);
    push_injected_refs(&mut events, context, "game", &sections.knowledge.refs);
    events
}

fn push_ranked_injected(
    events: &mut Vec<memory::Event>,
    context: (&str, &str),
    edges: &[memory::SemanticEdge],
) {
    for edge in edges {
        let edge_ref = memory::EdgeRef::new(&edge.source, &edge.target);
        push_injected(events, context, "local", &edge_ref);
    }
}

fn push_injected_refs(
    events: &mut Vec<memory::Event>,
    context: (&str, &str),
    actor_id: &str,
    refs: &[memory::EdgeRef],
) {
    for edge_ref in refs {
        push_injected(events, context, actor_id, edge_ref);
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

// Beliefs first, then deterministic open loops, game knowledge, and task chain.
// Any section may be empty.
fn assemble_memory_block(
    edges: &[memory::SemanticEdge],
    loops: &memory::OpenLoops,
    knowledge: &memory::GameKnowledge,
    chain: &memory::TaskChain,
) -> String {
    let mut block = if edges.is_empty() {
        String::new()
    } else {
        memory::edges_to_toon(edges)
    };
    append_section(&mut block, &loops.block);
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

pub(crate) struct ToolExecutionContext<'ctx> {
    pub(crate) embed: &'ctx dyn EmbeddingClient,
    pub(crate) search: Option<&'ctx dyn SearchClient>,
    pub(crate) session_path: &'ctx Path,
    pub(crate) sender: &'ctx Sender<Chunk>,
}

pub(crate) fn execute_tool_turn<'context>(
    context: &ToolExecutionContext<'context>,
    calls: &[ToolCallRequest],
    messages: &mut Vec<Value>,
    trace_context: &TraceContext,
) {
    messages.push(build_tool_call_message(calls));
    for call in calls {
        crate::trace::tool(
            trace_context,
            "tool_call",
            &call.id,
            json!({"name": call.name, "arguments": call.args}),
        );
        let _ = context
            .sender
            .send(Chunk::ToolCall(format!("{}: running…", call.name)));
        let result = execute_tool(call, context);
        crate::trace::tool(
            trace_context,
            "tool_result",
            &call.id,
            json!({"name": call.name, "result": &result}),
        );
        let _ = context
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

pub(crate) fn build_tools(has_search: bool) -> Value {
    let mut tools = Vec::new();
    if has_search {
        tools.push(search_web_tool());
    }
    tools.push(search_history_tool());
    tools.push(read_history_tool());
    tools.push(read_source_tool());
    tools.push(retrieve_memory_tool());
    tools.push(simple_recall_tool(
        "recall_curiosities",
        "Recall up to five real open questions you are curious about, with their age.",
    ));
    tools.push(simple_recall_tool(
        "recall_interests",
        "Recall up to five durable or self-derived topics you are interested in.",
    ));
    tools.push(simple_recall_tool(
        "recall_ping_policy",
        "Recall whether proactive pings are muted and which topics are held for the digest.",
    ));
    tools.push(simple_recall_tool(
        "recall_intentions",
        "Recall up to five real open or due intentions and what each is waiting on.",
    ));
    tools.push(recall_recent_learning_tool());
    tools.push(simple_recall_tool(
        "recall_queue",
        "Recall up to five queued and recently completed or failed inbox studies.",
    ));
    tools.push(simple_recall_tool(
        "catch_up",
        "Summarize real graph, study, and due-loop activity since the previous conversation.",
    ));
    Value::Array(tools)
}

pub(crate) fn tools_for_round(tools: &Value, tool_rounds: usize) -> Option<&Value> {
    if tool_rounds < MAX_TOOL_ROUNDS {
        return Some(tools);
    }
    runlog::log(
        "chat",
        &format!("tool round cap hit ({MAX_TOOL_ROUNDS}); forcing final answer"),
    );
    None
}

fn simple_recall_tool(name: &str, description: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}}
        }
    })
}

fn recall_recent_learning_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "recall_recent_learning",
            "description": "Recall up to five beliefs actually created or strengthened recently. Defaults to today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "since_days": {
                        "type": "integer", "minimum": 0, "maximum": 7,
                        "description": "Days to look back; 0 means today and values above 7 are clamped"
                    }
                }
            }
        }
    })
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

fn retrieve_memory_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "Retrieve memory edges for a focused query about the user, their preferences, decisions, gear, or projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    })
}

fn execute_tool<'context>(
    call: &ToolCallRequest,
    context: &ToolExecutionContext<'context>,
) -> String {
    let _timer = runlog::time_step("tool", call.name.clone());
    match call.name.as_str() {
        "search_web" => execute_search_web(&call.args, context.search),
        "search_history" => execute_search_history(&call.args),
        "read_history" => execute_read_history(&call.args, context.session_path),
        "read_source" => execute_read_source(&call.args),
        "retrieve_memory" => execute_retrieve_memory(&call.args, context.embed),
        "recall_curiosities" => execute_recall_curiosities(),
        "recall_interests" => execute_recall_interests(),
        "recall_ping_policy" => execute_recall_ping_policy(),
        "recall_intentions" => execute_recall_intentions(),
        "recall_recent_learning" => execute_recall_recent_learning(&call.args),
        "recall_queue" => execute_recall_queue(),
        "catch_up" => execute_catch_up(),
        _ => "unknown tool".to_string(),
    }
}

fn execute_retrieve_memory(args: &str, embed: &dyn EmbeddingClient) -> String {
    let Ok(root) = serde_json::from_str::<Value>(args) else {
        return "no edges found".to_string();
    };
    let Some(query) = parse_string_field(&root, "query") else {
        return "no edges found".to_string();
    };
    if query.trim().is_empty() {
        return "no edges found".to_string();
    }
    let mut edges = memory::select_semantic_edges(embed, &query);
    if edges.len() > RECALL_K {
        edges.truncate(RECALL_K);
    }
    if edges.is_empty() {
        return "no edges found".to_string();
    }
    memory::edges_to_toon(&edges)
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

const INTROSPECTION_LIMIT: usize = 5;

fn execute_recall_curiosities() -> String {
    let Ok(conn) = memory::open_connection() else {
        return "No open curiosities right now.".to_string();
    };
    execute_recall_curiosities_with(&conn, &crate::timeutil::today_iso())
}

fn execute_recall_curiosities_with(conn: &turso::Connection, today: &str) -> String {
    let notes = memory::block_on(memory::load_open_curiosity_notes_async(
        conn,
        INTROSPECTION_LIMIT,
    ))
    .unwrap_or_default();
    if notes.is_empty() {
        return "No open curiosities right now.".to_string();
    }
    let mut lines = vec!["Open curiosities:".to_string()];
    for (_, question, target, opened) in notes {
        lines.push(format!(
            "- {question} ({}; about {target})",
            curiosity_age(&opened, today)
        ));
    }
    lines.join("\n")
}

fn execute_recall_interests() -> String {
    render_interests(memory::top_interests(INTROSPECTION_LIMIT))
}

#[cfg(test)]
fn execute_recall_interests_with(conn: &turso::Connection) -> String {
    let interests =
        memory::block_on(memory::top_interests_with(conn, INTROSPECTION_LIMIT)).unwrap_or_default();
    render_interests(interests)
}

fn render_interests(interests: Vec<memory::Interest>) -> String {
    if interests.is_empty() {
        return "No interests have emerged yet.".to_string();
    }
    let mut lines = vec!["Interests:".to_string()];
    for interest in interests {
        lines.push(format!("- {} ({})", interest.topic, interest.origin));
    }
    lines.join("\n")
}

fn execute_recall_ping_policy() -> String {
    let Ok(conn) = memory::open_connection() else {
        return "Ping policy is unavailable.".to_string();
    };
    execute_recall_ping_policy_with(&conn, &crate::timeutil::timestamp_now())
}

fn execute_recall_ping_policy_with(conn: &turso::Connection, now: &str) -> String {
    let policy = match memory::load_ping_policy(conn, now) {
        Ok(policy) => policy,
        Err(_) => return "Ping policy is unavailable.".to_string(),
    };
    render_ping_policy(&policy, now)
}

fn render_ping_policy(policy: &memory::PingPolicy, now: &str) -> String {
    let mut lines = Vec::new();
    if policy.muted_until.as_str() > now {
        lines.push(format!("Pings are muted until {}.", policy.muted_until));
    }
    for (topic, until) in &policy.diverts {
        lines.push(format!("{topic} is held for the digest until {until}."));
    }
    if lines.is_empty() {
        return "Pinging is on; nothing is suppressed.".to_string();
    }
    lines.join("\n")
}

fn curiosity_age(opened: &str, today: &str) -> String {
    let opened_ts = format!("{}T00:00:00", opened.trim());
    let today_ts = format!("{}T00:00:00", today.trim());
    let Some(opened_secs) = crate::timeutil::timestamp_to_epoch_seconds(&opened_ts) else {
        return "age unknown".to_string();
    };
    let Some(today_secs) = crate::timeutil::timestamp_to_epoch_seconds(&today_ts) else {
        return "age unknown".to_string();
    };
    let days = (today_secs - opened_secs).max(0) / 86_400;
    match days {
        0 => "opened today".to_string(),
        1 => "1 day old".to_string(),
        _ => format!("{days} days old"),
    }
}

fn execute_recall_intentions() -> String {
    let Ok(conn) = memory::open_connection() else {
        return "No open loops.".to_string();
    };
    execute_recall_intentions_with(&conn, &crate::timeutil::today_iso())
}

fn execute_recall_intentions_with(conn: &turso::Connection, today: &str) -> String {
    let loops = memory::block_on(memory::load_open_intentions_async(
        conn,
        today,
        INTROSPECTION_LIMIT,
    ))
    .unwrap_or_default();
    if loops.is_empty() {
        return "No open loops.".to_string();
    }
    let mut lines = vec!["Open loops:".to_string()];
    for open_loop in loops {
        let due = if open_loop.due.is_empty() {
            String::new()
        } else {
            format!("; due {}", open_loop.due)
        };
        lines.push(format!(
            "- Waiting on {}: {}{due}",
            open_loop.target, open_loop.intention
        ));
    }
    lines.join("\n")
}

fn execute_recall_recent_learning(args: &str) -> String {
    let Ok(conn) = memory::open_connection() else {
        return "Nothing new consolidated recently.".to_string();
    };
    execute_recall_recent_learning_with(&conn, parse_since_days(args))
}

fn execute_recall_recent_learning_with(conn: &turso::Connection, since_days: usize) -> String {
    let learning = memory::block_on(memory::load_recent_learning_async(
        conn,
        since_days,
        INTROSPECTION_LIMIT,
    ))
    .unwrap_or_default();
    render_recent_learning(&learning)
}

fn render_recent_learning(learning: &[memory::RecentLearning]) -> String {
    if learning.is_empty() {
        return "Nothing new consolidated recently.".to_string();
    }
    let mut lines = vec!["Recent learning:".to_string()];
    for item in learning {
        let verb = if item.event_type == "created" {
            "New"
        } else {
            "Strengthened"
        };
        lines.push(format!(
            "- {verb} ({}): {} ({} → {})",
            item.provenance, item.content, item.source, item.target
        ));
    }
    lines.join("\n")
}

fn parse_since_days(args: &str) -> usize {
    let Ok(root) = serde_json::from_str::<Value>(args) else {
        return 0;
    };
    let Some(value) = root.get("since_days") else {
        return 0;
    };
    value.as_u64().unwrap_or(0).min(7) as usize
}

fn execute_recall_queue() -> String {
    let Ok(conn) = memory::open_connection() else {
        return "Inbox is empty.".to_string();
    };
    execute_recall_queue_with(&conn)
}

fn execute_recall_queue_with(conn: &turso::Connection) -> String {
    let items = memory::block_on(memory::load_inbox_snapshot_async(conn, INTROSPECTION_LIMIT))
        .unwrap_or_default();
    if items.is_empty() {
        return "Inbox is empty.".to_string();
    }
    let mut lines = vec!["Inbox:".to_string()];
    for item in items {
        lines.push(format!("- {item}"));
    }
    lines.join("\n")
}

fn execute_catch_up() -> String {
    let Ok(conn) = memory::open_connection() else {
        return "Nothing has happened since we last talked.".to_string();
    };
    let since = catch_up_since(&conn);
    let until = crate::timeutil::timestamp_now();
    execute_catch_up_with(&conn, &since, &until, &crate::timeutil::today_iso())
}

// An "away" gap: below this, consecutive messages are the same sitting and
// there is nothing to catch up on; above it, the user stepped away and the
// night/idle passes may have run.
const CATCH_UP_GAP_SECONDS: i64 = 90 * 60; // replay-tunable.
const CATCH_UP_SCAN_LIMIT: usize = 50;

// Window start for "catch me up": the last message before the user stepped
// away (cross-session, from the ledger — TUI writes one file per turn, so the
// session file alone cannot see the gap). No away-gap → collapse to the last
// message so the digest is honestly empty.
fn catch_up_since(conn: &turso::Connection) -> String {
    let boundary = memory::block_on(memory::conversation_boundary_async(
        conn,
        CATCH_UP_GAP_SECONDS,
        CATCH_UP_SCAN_LIMIT,
    ));
    if let Ok(Some(since)) = boundary {
        return since;
    }
    let prior = memory::block_on(memory::latest_assistant_message_timestamp_async(conn))
        .unwrap_or_default();
    prior.unwrap_or_else(|| format!("{}T", crate::timeutil::today_iso()))
}

fn execute_catch_up_with(
    conn: &turso::Connection,
    since: &str,
    until: &str,
    today: &str,
) -> String {
    let window = memory::NightWindow {
        from: since.to_string(),
        until: until.to_string(),
    };
    let digest =
        memory::block_on(memory::catch_up_digest_async(conn, &window, today)).unwrap_or_default();
    render_catch_up(&digest)
}

fn render_catch_up(digest: &memory::DayDigest) -> String {
    if digest.is_quiet() {
        return "Nothing has happened since we last talked.".to_string();
    }
    let mut lines = vec![format!(
        "Since we last talked: {} beliefs added, {} updated, {} pruned.",
        digest.created, digest.updated, digest.pruned
    )];
    if let Some(line) = digest.autonomy.line() {
        lines.push(line);
    }
    if !digest.inbox_outcomes.is_empty() {
        lines.push(format!("Inbox: {}.", join_capped(&digest.inbox_outcomes)));
    }
    if !digest.due_loops.is_empty() {
        lines.push(format!("Due now: {}.", join_capped(&digest.due_loops)));
    }
    if !digest.held_findings.is_empty() {
        lines.push(format!(
            "Held for you: {}.",
            join_capped(&digest.held_findings)
        ));
    }
    lines.join("\n")
}

fn join_capped(items: &[String]) -> String {
    let mut capped = Vec::new();
    let mut index = 0usize;
    while index < items.len() && index < INTROSPECTION_LIMIT {
        capped.push(items[index].as_str());
        index += 1;
    }
    capped.join("; ")
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
#[path = "turn_tests.rs"]
mod tests;
