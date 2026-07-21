use std::error::Error;
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use turso::Connection;

use crate::api::{Chunk, EmbeddingClient, LlmClient, ToolCallRequest, TurnOutcome};
use crate::prompts::{
    EXPLORE_BIOLOGIST, EXPLORE_ECONOMIST, EXPLORE_HISTORIAN, build_exploration_prompt,
};
use crate::runlog;
use crate::web::SearchClient;

use super::agent::{self, ApplyToolOptions, ToolCall};
use super::db;
use super::dream::{self, DOMAIN_AGENT_NAMES};
use super::embed::{self, SemanticEdge};
use super::graph::{Edge, PROVENANCE_WEB_SURFACED, edge_ceiling, normalize_id};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

const EXPLORATION_TURNS: usize = 4; // replay-tunable.
const EXPLORE_SEARCHES_PER_PASS: usize = 2; // replay-tunable.
const RETRIEVE_LIMIT: usize = 15; // replay-tunable.
const TRAVERSE_LIMIT: usize = 20; // replay-tunable.
const MAX_TOOL_ROUNDS: usize = 6; // per turn; runaway guard for the autonomous loop.
const INCUBATION_NEIGHBOUR_DISTANCE: f64 = 0.45; // replay-tunable.
const RIPE_NEIGHBOUR_THRESHOLD: usize = 3; // replay-tunable.
const INCUBATION_CHECK_LIMIT: usize = 10; // replay-tunable.

pub enum ExploreEvent {
    Log { phase: &'static str, text: String },
    Done { events_recorded: usize },
}

struct TurnOutcomeInfo {
    used_tools: bool,
}

struct ToolContext<'context> {
    embed: &'context dyn EmbeddingClient,
    conn: &'context Connection,
    search: Option<&'context dyn SearchClient>,
    agent_name: &'context str,
    session_id: &'context str,
    turn_id: &'context str,
    events: &'context mut Vec<Event>,
    active_query: &'context mut Option<String>,
    remaining_searches: &'context mut usize,
}

struct OpenCuriosityNote {
    note_id: String,
    question: String,
    topic_id: String,
}

pub fn run_exploration_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    sender: Sender<ExploreEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let _timer = runlog::time_step("explore", "pass");
    let trace_context = crate::trace::TraceContext::new("explore");
    let llm = crate::trace::TracingClient::wrap(llm, trace_context);
    match execute_exploration_pass(Arc::clone(&llm), Arc::clone(&embed), search, &sender) {
        Ok(events_recorded) => {
            db::sync_push_best_effort("explore");
            let _ = sender.send(ExploreEvent::Done { events_recorded });
            Ok(events_recorded)
        }
        Err(err) => {
            send_log(
                &sender,
                "explore",
                format!("pass failed — writes may be partial: {err}"),
            );
            let _ = sender.send(ExploreEvent::Done { events_recorded: 0 });
            Err(err)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn execute_exploration_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    sender: &Sender<ExploreEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let _write_guard = db::acquire_write_lock();
    let session_id = exploration_session_id();
    let created_at = crate::timeutil::timestamp_now();
    db::register_exploration_session(&session_id, &created_at);
    let open_note = load_open_note(&*embed, sender);
    send_log(
        sender,
        "explore",
        format!("pass start — session {session_id}"),
    );
    let mut total_events = 0usize;
    let mut search_budget = EXPLORE_SEARCHES_PER_PASS;
    for agent_name in DOMAIN_AGENT_NAMES {
        total_events += run_agent_loop(
            Arc::clone(&llm),
            Arc::clone(&embed),
            search.as_deref(),
            &session_id,
            agent_name,
            if agent_name == DOMAIN_AGENT_NAMES[0] {
                open_note.as_ref()
            } else {
                None
            },
            &mut search_budget,
            sender,
        )?;
    }
    send_log(
        sender,
        "explore",
        format!("pass complete — {total_events} events recorded"),
    );
    send_log(
        sender,
        "explore",
        "pass complete — consider a dream pass to consolidate".to_string(),
    );
    Ok(total_events)
}

fn load_open_note(
    embed: &dyn EmbeddingClient,
    sender: &Sender<ExploreEvent>,
) -> Option<OpenCuriosityNote> {
    let mut rows = db::load_open_curiosity_notes(INCUBATION_CHECK_LIMIT);
    if rows.is_empty() {
        return None;
    }
    let mut questions = Vec::new();
    for row in &rows {
        questions.push(row.1.clone());
    }
    let Ok(vectors) = embed.embed(&questions) else {
        return Some(open_note_from_row(rows.remove(0)));
    };
    if vectors.len() != rows.len() {
        return Some(open_note_from_row(rows.remove(0)));
    }
    let counts = incubation_counts(&rows, &vectors);
    let selected = choose_note_index(&counts);
    if counts[selected] >= RIPE_NEIGHBOUR_THRESHOLD {
        let prefix = question_prefix(&rows[selected].1);
        send_log(
            sender,
            "explore",
            format!(
                "incubation: note '{prefix}' ripe ({} new neighbours)",
                counts[selected]
            ),
        );
    }
    Some(open_note_from_row(rows.remove(selected)))
}

fn incubation_counts(rows: &[db::OpenCuriosityNoteRow], vectors: &[Vec<f64>]) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut index = 0usize;
    while index < rows.len() {
        let count = db::count_new_embedding_neighbours(
            &vectors[index],
            &rows[index].3,
            INCUBATION_NEIGHBOUR_DISTANCE,
        )
        .unwrap_or_default();
        counts.push(count);
        index += 1;
    }
    counts
}

fn choose_note_index(counts: &[usize]) -> usize {
    let mut index = 0usize;
    while index < counts.len() {
        if counts[index] >= RIPE_NEIGHBOUR_THRESHOLD {
            return index;
        }
        index += 1;
    }
    0
}

fn open_note_from_row(row: db::OpenCuriosityNoteRow) -> OpenCuriosityNote {
    OpenCuriosityNote {
        note_id: row.0,
        question: row.1,
        topic_id: row.2,
    }
}

fn question_prefix(question: &str) -> String {
    let mut prefix = String::new();
    let mut characters = question.chars();
    let mut remaining = 48usize;
    while remaining > 0 {
        let Some(character) = characters.next() else {
            break;
        };
        prefix.push(character);
        remaining -= 1;
    }
    prefix
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_agent_loop(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<&dyn SearchClient>,
    session_id: &str,
    agent_name: &'static str,
    open_note: Option<&OpenCuriosityNote>,
    remaining_searches: &mut usize,
    sender: &Sender<ExploreEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let conn = db::open_connection()?;
    let seed_edges = seed_edges_for_agent(&*embed, agent_name, session_id);
    let mut messages = initial_messages(agent_name, &seed_edges, open_note);
    let tools = exploration_tools();
    let mut total_events = 0usize;
    let mut turns_used = 0usize;
    send_log(sender, "explore", format!("{agent_name}: loop start"));
    for turn_index in 0..EXPLORATION_TURNS {
        let turn_id = format!("{agent_name}-turn-{}", turn_index + 1);
        let outcome = run_agent_turn(
            &*llm,
            &*embed,
            &conn,
            search,
            agent_name,
            session_id,
            &turn_id,
            turn_index,
            &seed_edges,
            open_note,
            remaining_searches,
            &mut messages,
            &tools,
            sender,
        )?;
        total_events += outcome.0;
        turns_used = turn_index + 1;
        if !outcome.1.used_tools {
            break;
        }
        messages.push(json!({
            "role": "user",
            "content": continuation_prompt(turn_index + 1)
        }));
    }
    send_log(
        sender,
        "explore",
        format!("{agent_name}: loop end — {turns_used} turns, {total_events} events"),
    );
    Ok(total_events)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn run_agent_turn(
    llm: &dyn LlmClient,
    embed: &dyn EmbeddingClient,
    conn: &Connection,
    search: Option<&dyn SearchClient>,
    agent_name: &'static str,
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    seed_edges: &[SemanticEdge],
    open_note: Option<&OpenCuriosityNote>,
    remaining_searches: &mut usize,
    messages: &mut Vec<Value>,
    tools: &Value,
    sender: &Sender<ExploreEvent>,
) -> Result<(usize, TurnOutcomeInfo), Box<dyn Error + Send + Sync>> {
    let _timer = runlog::time_step("explore", format!("{agent_name} turn {}", turn_index + 1));
    send_log(
        sender,
        "explore",
        format!("{agent_name}: turn {} start", turn_index + 1),
    );
    db::begin_transaction(conn)?;
    let mut events = seed_events(
        agent_name, session_id, turn_id, turn_index, seed_edges, open_note,
    );
    let result = run_single_turn(
        llm,
        embed,
        conn,
        search,
        agent_name,
        session_id,
        turn_id,
        remaining_searches,
        messages,
        tools,
        &mut events,
        sender,
    );
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(err) => {
            let _ = db::block_on(async { conn.execute("ROLLBACK", ()).await });
            return Err(err);
        }
    };
    let recorded = match record_turn_events(conn, &events) {
        Ok(recorded) => recorded,
        Err(err) => {
            let _ = db::block_on(async { conn.execute("ROLLBACK", ()).await });
            return Err(err);
        }
    };
    send_log(
        sender,
        "explore",
        format!(
            "{agent_name}: turn {} done — {} events",
            turn_index + 1,
            recorded
        ),
    );
    Ok((recorded, outcome))
}

#[allow(clippy::too_many_arguments)]
fn run_single_turn(
    llm: &dyn LlmClient,
    embed: &dyn EmbeddingClient,
    conn: &Connection,
    search: Option<&dyn SearchClient>,
    agent_name: &str,
    session_id: &str,
    turn_id: &str,
    remaining_searches: &mut usize,
    messages: &mut Vec<Value>,
    tools: &Value,
    events: &mut Vec<Event>,
    sender: &Sender<ExploreEvent>,
) -> Result<TurnOutcomeInfo, Box<dyn Error + Send + Sync>> {
    let mut used_tools = false;
    let mut rounds = 0usize;
    let mut active_query: Option<String> = None;
    loop {
        // Autonomous loop: cap tool rounds per turn so a model that never
        // stops calling tools cannot run away unattended.
        if rounds >= MAX_TOOL_ROUNDS {
            return Ok(TurnOutcomeInfo { used_tools });
        }
        rounds += 1;
        let turn_outcome = stream_turn_noop_chunks(llm, messages, tools)?;
        match turn_outcome {
            TurnOutcome::Done(_) => return Ok(TurnOutcomeInfo { used_tools }),
            TurnOutcome::ToolCalls(calls) => {
                used_tools = true;
                messages.push(build_tool_call_message(&calls));
                for call in calls {
                    let mut context = ToolContext {
                        embed,
                        conn,
                        search,
                        agent_name,
                        session_id,
                        turn_id,
                        events,
                        active_query: &mut active_query,
                        remaining_searches,
                    };
                    let result = handle_tool_call(&call, &mut context, sender)?;
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    }));
                }
            }
        }
    }
}

fn stream_turn_noop_chunks(
    llm: &dyn LlmClient,
    messages: &[Value],
    tools: &Value,
) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
    let (chunk_sender, _chunk_receiver) = channel::<Chunk>();
    llm.stream_turn(messages, Some(tools), &chunk_sender)
}

fn handle_tool_call(
    call: &ToolCallRequest,
    context: &mut ToolContext,
    sender: &Sender<ExploreEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    send_log(
        sender,
        "explore",
        format!(
            "{}:{}: tool {} args={}",
            context.agent_name, context.turn_id, call.name, call.args
        ),
    );
    let result = execute_tool_call(call, context)?;
    send_log(
        sender,
        "explore",
        format!(
            "{}:{}: result {}",
            context.agent_name,
            context.turn_id,
            summarize_result(&call.name, &result)
        ),
    );
    Ok(result)
}

fn execute_tool_call(
    call: &ToolCallRequest,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(tool_call) = agent::parse_tool_call_request(call) {
        return apply_web_mutation(tool_call, context);
    }
    match agent::sanitized_tool_name(&call.name) {
        "retrieve_memory" => retrieve_memory(&call.args, context),
        "traverse_graph" => traverse_graph(&call.args, context),
        "mark_interesting" => mark_interesting(&call.args, context),
        "search_web" => search_web(&call.args, context),
        _ => Ok("unknown tool".to_string()),
    }
}

fn retrieve_memory(
    args: &str,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let query = parse_string_arg(args, "query");
    if query.trim().is_empty() {
        return Ok("no edges found".to_string());
    }
    let mut edges = embed::select_semantic_edges(context.embed, &query);
    if edges.len() > RETRIEVE_LIMIT {
        edges.truncate(RETRIEVE_LIMIT);
    }
    append_retrieved_events(context, &edges);
    if edges.is_empty() {
        return Ok("no edges found".to_string());
    }
    Ok(embed::edges_to_toon(&edges))
}

fn traverse_graph(
    args: &str,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let node = parse_string_arg(args, "node");
    let _ = parse_usize_arg(args, "depth");
    if node.trim().is_empty() {
        return Ok("no edges found".to_string());
    }
    let normalized = normalize_id(&node);
    let mut edges = live_edges(db::load_edges_for_node(context.conn, &normalized)?);
    if edges.len() > TRAVERSE_LIMIT {
        edges.truncate(TRAVERSE_LIMIT);
    }
    if edges.is_empty() {
        return Ok("no edges found".to_string());
    }
    let semantic_edges = semantic_edges_from_graph(&edges);
    append_retrieved_graph_events(context, &edges);
    Ok(embed::edges_to_toon(&semantic_edges))
}

fn mark_interesting(
    args: &str,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let source = normalize_id(&parse_string_arg(args, "source"));
    let target = normalize_id(&parse_string_arg(args, "target"));
    let reason = parse_string_arg(args, "reason");
    if source.is_empty() || target.is_empty() {
        return Ok("edge not found".to_string());
    }
    let edge = db::load_edge(context.conn, &source, &target)?;
    if !is_live_edge(edge.as_ref()) {
        return Ok("edge not found".to_string());
    }
    context.events.push(Event::marked(
        Actor::new("agent", context.agent_name),
        EdgeRef::new(&source, &target),
        EventContext::new(context.session_id, context.turn_id),
        &reason,
    ));
    Ok("marked".to_string())
}

fn search_web(
    args: &str,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let Ok(root) = serde_json::from_str::<Value>(args) else {
        return Ok("search denied: invalid arguments".to_string());
    };
    let queries = parse_queries(&root);
    if queries.is_empty() {
        return Ok("search denied: provide queries".to_string());
    }
    let reason = parse_string_field(&root, "reason");
    if reason != "curiosity" {
        return Ok("search denied: exploration only allows reason=curiosity".to_string());
    }
    if *context.remaining_searches == 0 {
        runlog::log("explore", "search denied — curiosity budget exhausted");
        return Ok("search denied: curiosity budget exhausted".to_string());
    }
    let Some(search_client) = context.search else {
        runlog::log("explore", "search unavailable");
        return Ok("search unavailable: no search client configured".to_string());
    };
    *context.remaining_searches -= 1;
    let joined = queries.join(" | ");
    runlog::log("explore", &format!("search allowed — {joined}"));
    *context.active_query = Some(joined);
    Ok(match search_client.search_many(&queries) {
        Some(result) if !result.trim().is_empty() => result,
        _ => "no results".to_string(),
    })
}

fn apply_web_mutation(
    tool_call: ToolCall,
    context: &mut ToolContext,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let source_message_id = format!("{}:{}", context.session_id, context.turn_id);
    // The payload distinguishes searched findings from unsourced exploration,
    // while slice-1 provenance classifies the whole pass as web-surfaced.
    let payload = match &context.active_query {
        Some(query) => json!({"via": "web_search", "query": query}).to_string(),
        None => json!({"via": "exploration"}).to_string(),
    };
    let options = ApplyToolOptions {
        actor: Actor::new("agent", context.agent_name),
        context: EventContext::new(context.session_id, context.turn_id),
        source_message_id: &source_message_id,
        event_payload: &payload,
        confidence_cap: Some(edge_ceiling(PROVENANCE_WEB_SURFACED, None)),
        provenance: PROVENANCE_WEB_SURFACED,
        // Exploration is lived experience, not durable belief by default —
        // it becomes knowledge only if dreaming promotes it later.
        default_memory_type: "episodic",
    };
    agent::apply_tool_call(tool_call, context.conn, &options, context.events)
}

fn append_retrieved_events(context: &mut ToolContext, edges: &[SemanticEdge]) {
    for edge in edges {
        context.events.push(Event::retrieved(
            Actor::new("agent", context.agent_name),
            EdgeRef::new(&edge.source, &edge.target),
            EventContext::new(context.session_id, context.turn_id),
        ));
    }
}

fn append_retrieved_graph_events(context: &mut ToolContext, edges: &[Edge]) {
    for edge in edges {
        context.events.push(Event::retrieved(
            Actor::new("agent", context.agent_name),
            EdgeRef::new(&edge.source, &edge.target),
            EventContext::new(context.session_id, context.turn_id),
        ));
    }
}

fn semantic_edges_from_graph(edges: &[Edge]) -> Vec<SemanticEdge> {
    let mut semantic_edges = Vec::new();
    for edge in edges {
        semantic_edges.push(SemanticEdge {
            source: edge.source.clone(),
            target: edge.target.clone(),
            content: edge.content.clone(),
            confidence: edge.confidence,
            stability: edge.stability,
            importance: edge.importance,
            subject: edge.subject,
            date: edge.date.clone(),
            occurred_at: edge.occurred_at.clone(),
            expired_at: edge.expired_at.clone(),
            memory_type: edge.memory_type.clone(),
            origin_session: edge.origin_session.clone(),
            provenance: edge.provenance.clone(),
            embedding: None,
        });
    }
    semantic_edges
}

fn live_edges(edges: Vec<Edge>) -> Vec<Edge> {
    let mut live = Vec::new();
    for edge in edges {
        if is_live_edge(Some(&edge)) {
            live.push(edge);
        }
    }
    live
}

fn is_live_edge(edge: Option<&Edge>) -> bool {
    let Some(edge) = edge else {
        return false;
    };
    edge.expired_at.is_none()
}

fn record_turn_events(
    conn: &Connection,
    events: &[Event],
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    if events.is_empty() {
        db::commit_transaction(conn)?;
        return Ok(0);
    }
    db::block_on(ledger::record_events_in_tx(conn, events))?;
    db::commit_transaction(conn)?;
    Ok(events.len())
}

fn seed_edges_for_agent(
    embed: &dyn EmbeddingClient,
    agent_name: &str,
    session_id: &str,
) -> Vec<SemanticEdge> {
    let anchor = dream::domain_anchor(agent_name).unwrap_or_default();
    embed::select_exploration_edges(embed, anchor, agent_name, session_id)
}

fn initial_messages(
    agent_name: &str,
    seed_edges: &[SemanticEdge],
    open_note: Option<&OpenCuriosityNote>,
) -> Vec<Value> {
    let explanation = exploration_explanation(agent_name);
    let seed_toon = embed::edges_to_toon(seed_edges);
    let prompt = build_exploration_prompt(explanation, &seed_toon, 0);
    let opening = match open_note {
        Some(note) if agent_name == DOMAIN_AGENT_NAMES[0] => note.question.clone(),
        _ => "Start exploration. Retrieve memory, traverse nodes, and mark specific edges as significant. Use search_web only for bounded curiosity, then extract any durable web findings with memory mutation tools.".to_string(),
    };
    vec![
        json!({"role": "system", "content": prompt}),
        json!({"role": "user", "content": opening}),
    ]
}

#[allow(clippy::too_many_arguments)]
fn seed_events(
    agent_name: &str,
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    seed_edges: &[SemanticEdge],
    open_note: Option<&OpenCuriosityNote>,
) -> Vec<Event> {
    if turn_index != 0 {
        return Vec::new();
    }
    let mut events = Vec::new();
    for edge in seed_edges {
        events.push(Event::injected(
            Actor::new("agent", agent_name),
            EdgeRef::new(&edge.source, &edge.target),
            EventContext::new(session_id, turn_id),
        ));
    }
    if agent_name == DOMAIN_AGENT_NAMES[0]
        && let Some(note) = open_note
    {
        events.push(Event::retrieved(
            Actor::new("agent", agent_name),
            EdgeRef::new(&note.note_id, &note.topic_id),
            EventContext::new(session_id, turn_id),
        ));
    }
    events
}

fn continuation_prompt(turn_index: usize) -> String {
    format!(
        "Continue exploration turn {}. Stop without tool calls if no high-value lead remains.",
        turn_index + 1
    )
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

fn summarize_result(tool_name: &str, result: &str) -> String {
    format!("{tool_name}: {}", shorten_text(result, 120))
}

fn shorten_text(text: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for (index, character) in text.trim().chars().enumerate() {
        if index >= max_chars {
            output.push('…');
            break;
        }
        output.push(character);
    }
    output
}

fn parse_string_arg(args: &str, key: &str) -> String {
    let Ok(root) = serde_json::from_str::<Value>(args) else {
        return String::new();
    };
    let Some(value) = root.get(key) else {
        return String::new();
    };
    match value.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

fn parse_usize_arg(args: &str, key: &str) -> Option<usize> {
    let Ok(root) = serde_json::from_str::<Value>(args) else {
        return None;
    };
    let value = root.get(key)?;
    let number = value.as_u64()?;
    usize::try_from(number).ok()
}

fn parse_queries(root: &Value) -> Vec<String> {
    let Some(queries_value) = root.get("queries") else {
        return Vec::new();
    };
    let Some(items) = queries_value.as_array() else {
        return Vec::new();
    };
    let mut queries = Vec::new();
    for item in items {
        let Some(query) = item.as_str() else {
            continue;
        };
        let trimmed = query.trim();
        if !trimmed.is_empty() {
            queries.push(trimmed.to_string());
        }
    }
    queries
}

fn parse_string_field(root: &Value, key: &str) -> String {
    let Some(value) = root.get(key) else {
        return String::new();
    };
    match value.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

fn exploration_explanation(agent_name: &str) -> &'static str {
    match agent_name {
        "biologist" => EXPLORE_BIOLOGIST,
        "historian" => EXPLORE_HISTORIAN,
        "economist" => EXPLORE_ECONOMIST,
        _ => EXPLORE_BIOLOGIST,
    }
}

fn exploration_tools() -> Value {
    let mut tools = match agent::memory_mutation_tools() {
        Value::Array(array) => array,
        _ => Vec::new(),
    };
    tools.push(retrieve_memory_tool());
    tools.push(traverse_graph_tool());
    tools.push(mark_interesting_tool());
    tools.push(search_web_tool());
    Value::Array(tools)
}

fn retrieve_memory_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "Retrieve memory edges for a query.",
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

fn traverse_graph_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "traverse_graph",
            "description": "Load graph edges incident to a node (depth is fixed to 1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "depth": {"type": "integer", "enum": [1]}
                },
                "required": ["node", "depth"]
            }
        }
    })
}

fn mark_interesting_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "mark_interesting",
            "description": "Mark an existing edge as significant, with a specific reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["source", "target", "reason"]
            }
        }
    })
}

fn search_web_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for bounded curiosity during exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "enum": ["curiosity"]}
                },
                "required": ["queries", "reason"]
            }
        }
    })
}

fn exploration_session_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("explore-{seconds}")
}

fn send_log(sender: &Sender<ExploreEvent>, phase: &'static str, text: String) {
    runlog::log(phase, &text);
    let _ = sender.send(ExploreEvent::Log { phase, text });
}

#[cfg(test)]
mod tests {
    use super::choose_note_index;

    #[test]
    fn ripe_note_precedes_older_unripe_note() {
        assert_eq!(choose_note_index(&[0, 3]), 1);
    }

    #[test]
    fn oldest_of_two_ripe_notes_is_selected() {
        assert_eq!(choose_note_index(&[3, 4]), 0);
    }

    #[test]
    fn oldest_note_remains_first_when_none_are_ripe() {
        assert_eq!(choose_note_index(&[0, 2]), 0);
    }
}
