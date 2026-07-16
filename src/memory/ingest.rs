use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use turso::Connection;

use crate::api::{Chunk, EmbeddingClient, LlmClient, ToolCallRequest, TurnOutcome};
use crate::prompts::{INGEST_EXPLANATION, build_ingestion_prompt};
use crate::runlog;
use crate::timeutil;
use crate::web::SearchClient;

use super::agent::{self, ApplyToolOptions, ToolCall};
use super::db;
use super::embed::{self, SemanticEdge};
use super::graph::{Edge, Node, PROVENANCE_WEB_SURFACED, edge_ceiling, normalize_id};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

const INGEST_CHUNK_CHARS: usize = 4000; // replay-tunable.
const INGEST_MAX_FULL_CHUNKS: usize = 25; // replay-tunable.
const INGEST_SEARCH_BUDGET_DIVISOR: usize = 4; // replay-tunable.
const INGEST_MAX_TOOL_ROUNDS: usize = 8; // replay-tunable.
// Graph context per chunk: the full retrieval pool (up to 150 edges) drowns
// the chunk text for the small extraction model — keep the ranked head only.
const INGEST_CONTEXT_EDGES: usize = 15; // replay-tunable.

pub enum IngestEvent {
    Log { phase: &'static str, text: String },
    Done { chunks_processed: usize },
}

enum MutationProvenance {
    Document,
    Web { query: String },
}

struct SearchBudget {
    curiosity_remaining: usize,
    contradiction_used: usize,
    contradiction_cap: usize,
}

struct ChunkStudyContext<'scope> {
    conn: &'scope Connection,
    search: Option<&'scope dyn SearchClient>,
    session_id: &'scope str,
    turn_id: &'scope str,
    source_message_id: &'scope str,
    doc_path: &'scope str,
    chunk_index: usize,
    chunk_count: usize,
    events: &'scope mut Vec<Event>,
    provenance: MutationProvenance,
    // Rounds elapsed since the last search: web provenance covers the search
    // round and the one after it (where extraction from results happens),
    // then mutations are document-sourced again.
    rounds_since_search: usize,
    search_budget: &'scope mut SearchBudget,
}

pub fn run_ingestion_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    path: PathBuf,
    sender: Sender<IngestEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let result = execute_ingestion_pass(llm, embed, search, (&path, ""), &sender);
    db::sync_push_best_effort("ingest");
    result
}

pub fn run_focused_ingestion_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    request: (&Path, &str),
    sender: Sender<IngestEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let result = execute_ingestion_pass(llm, embed, search, request, &sender);
    db::sync_push_best_effort("ingest");
    result
}

#[allow(clippy::too_many_lines)]
fn execute_ingestion_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    request: (&Path, &str),
    sender: &Sender<IngestEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let _timer = runlog::time_step("ingest", "pass");
    let (path, focus) = request;
    validate_ingest_path(path)?;
    let _write_guard = db::acquire_write_lock();
    let conn = db::open_connection()?;
    let session_id = ingestion_session_id();
    let created_at = timeutil::timestamp_now();
    db::register_ingestion_session(&session_id, &created_at);
    let doc_path = path.to_string_lossy().to_string();
    send_log(
        sender,
        "ingest",
        format!("pass start — session {session_id}"),
    );
    let document = std::fs::read_to_string(path)?;
    let chunks = chunk_document(&document, INGEST_CHUNK_CHARS);
    send_log(
        sender,
        "ingest",
        format!("chunked {} into {} chunks", doc_path, chunks.len()),
    );
    let summaries = skim_chunks(&*llm, &chunks, focus, sender)?;
    let document_map = build_document_map(&summaries);
    send_log(sender, "ingest", "skim map built".to_string());
    let full_mask = choose_full_chunks(&*llm, &summaries);
    let doc_node_id = normalize_id(&doc_path);
    let doc_node = Node {
        node_type: "entity".to_string(),
        body: String::new(),
        date: String::new(),
    };
    db::write_node(&conn, &doc_node_id, &doc_node)?;
    let mut previous_summary = String::new();
    let mut search_budget = SearchBudget {
        curiosity_remaining: curiosity_budget(chunks.len()),
        contradiction_used: 0,
        contradiction_cap: chunks.len().max(1),
    };
    for chunk_index in 0..chunks.len() {
        let chunk_number = chunk_index + 1;
        let _chunk_timer =
            runlog::time_step("ingest", format!("chunk {chunk_number}/{}", chunks.len()));
        let turn_id = format!("{session_id}:chunk-{chunk_number}");
        let source_message_id = db::insert_message(
            &session_id,
            "ingestor",
            &chunks[chunk_index],
            &timeutil::timestamp_now(),
        );
        send_log(
            sender,
            "ingest",
            format!("chunk {chunk_number}/{} start", chunks.len()),
        );
        if !full_mask[chunk_index] {
            record_skipped_chunk(
                &conn,
                &doc_node_id,
                &doc_path,
                chunk_index,
                &summaries[chunk_index],
                &turn_id,
                &source_message_id,
            )?;
            send_log(
                sender,
                "ingest",
                format!("chunk {chunk_number}/{} skim-only", chunks.len()),
            );
            previous_summary = summaries[chunk_index].clone();
            continue;
        }
        let mut semantic_edges = embed::select_semantic_edges(&*embed, &summaries[chunk_index]);
        semantic_edges.truncate(INGEST_CONTEXT_EDGES);
        record_chunk_turn(
            &*llm,
            search.as_deref(),
            sender,
            ChunkTurnInput {
                conn: &conn,
                doc_path: &doc_path,
                chunk_index,
                chunk_count: chunks.len(),
                chunk_text: &chunks[chunk_index],
                chunk_summary: &summaries[chunk_index],
                previous_summary: &previous_summary,
                document_map: &document_map,
                semantic_edges: &semantic_edges,
                session_id: &session_id,
                turn_id: &turn_id,
                source_message_id: &source_message_id,
                search_budget: &mut search_budget,
            },
        )?;
        send_log(
            sender,
            "ingest",
            format!("chunk {chunk_number}/{} done", chunks.len()),
        );
        previous_summary = summaries[chunk_index].clone();
    }
    send_log(
        sender,
        "ingest",
        "pass complete — dream required".to_string(),
    );
    let _ = sender.send(IngestEvent::Done {
        chunks_processed: chunks.len(),
    });
    Ok(session_id)
}

struct ChunkTurnInput<'scope> {
    conn: &'scope Connection,
    doc_path: &'scope str,
    chunk_index: usize,
    chunk_count: usize,
    chunk_text: &'scope str,
    chunk_summary: &'scope str,
    previous_summary: &'scope str,
    document_map: &'scope str,
    semantic_edges: &'scope [SemanticEdge],
    session_id: &'scope str,
    turn_id: &'scope str,
    source_message_id: &'scope str,
    search_budget: &'scope mut SearchBudget,
}

#[allow(clippy::too_many_lines)]
fn record_chunk_turn<'scope>(
    llm: &dyn LlmClient,
    search: Option<&dyn SearchClient>,
    sender: &Sender<IngestEvent>,
    input: ChunkTurnInput<'scope>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    db::begin_transaction(input.conn)?;
    let mut events = injected_events(
        input.semantic_edges,
        input.session_id,
        input.turn_id,
        Actor::new("agent", "ingestor"),
    );
    let prompt = build_ingestion_prompt(
        INGEST_EXPLANATION,
        input.document_map,
        input.previous_summary,
        &embed::edges_to_toon(input.semantic_edges),
        input.chunk_text,
    );
    let mut messages = vec![
        json!({"role": "system", "content": prompt}),
        json!({"role": "user", "content": format!(
            "Study chunk {}/{} of {}. The chunk summary is: {}",
            input.chunk_index + 1,
            input.chunk_count,
            input.doc_path,
            input.chunk_summary
        )}),
    ];
    let tools = ingestion_tools();
    let result = (|| {
        let mut study = ChunkStudyContext {
            conn: input.conn,
            search,
            session_id: input.session_id,
            turn_id: input.turn_id,
            source_message_id: input.source_message_id,
            doc_path: input.doc_path,
            chunk_index: input.chunk_index,
            chunk_count: input.chunk_count,
            events: &mut events,
            provenance: MutationProvenance::Document,
            rounds_since_search: 0,
            search_budget: input.search_budget,
        };
        run_chunk_loop(llm, &tools, &mut messages, &mut study, sender)?;
        // The interactive loop handles fact-checking (search, notes, expiry)
        // but the small model extracts poorly one call at a time — a batch
        // text-protocol pass guarantees the chunk's facts land as edges.
        run_batch_extraction(
            llm,
            input.semantic_edges,
            input.chunk_text,
            &mut study,
            sender,
        )?;
        db::block_on(ledger::record_events_in_tx(input.conn, &events))
    })();
    if let Err(err) = result {
        if let Err(rollback_err) = db::block_on(async { input.conn.execute("ROLLBACK", ()).await })
        {
            runlog::log("ledger", &format!("rollback failed: {rollback_err}"));
        }
        return Err(err);
    }
    db::commit_transaction(input.conn)?;
    Ok(())
}

fn run_batch_extraction<'scope>(
    llm: &dyn LlmClient,
    semantic_edges: &[SemanticEdge],
    chunk_text: &str,
    study: &mut ChunkStudyContext<'scope>,
    sender: &Sender<IngestEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let prompt = crate::prompts::build_ingest_extract_prompt(
        &embed::edges_to_toon(semantic_edges),
        chunk_text,
    );
    let response = llm.collect(&prompt)?;
    let calls = agent::parse_tool_calls(&response);
    if calls.is_empty() {
        let head: String = response.chars().take(200).collect();
        runlog::log(
            "ingest",
            &format!("extraction parsed 0 calls; response head: {head}"),
        );
    }
    study.provenance = MutationProvenance::Document;
    let mut applied = 0usize;
    for call in calls {
        apply_ingest_mutation(call, study)?;
        applied += 1;
    }
    send_log(
        sender,
        "ingest",
        format!(
            "chunk {}/{} extraction: {applied} mutations",
            study.chunk_index + 1,
            study.chunk_count
        ),
    );
    Ok(())
}

fn run_chunk_loop<'scope>(
    llm: &dyn LlmClient,
    tools: &Value,
    messages: &mut Vec<Value>,
    study: &mut ChunkStudyContext<'scope>,
    sender: &Sender<IngestEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rounds = 0usize;
    loop {
        if rounds >= INGEST_MAX_TOOL_ROUNDS {
            return Ok(());
        }
        rounds += 1;
        match stream_turn_noop_chunks(llm, messages, tools)? {
            TurnOutcome::Done(_) => return Ok(()),
            TurnOutcome::ToolCalls(calls) => {
                messages.push(build_tool_call_message(&calls));
                for call in calls {
                    let result = execute_ingest_tool(&call, study, sender)?;
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    }));
                }
                expire_web_provenance(study);
            }
        }
    }
}

fn execute_ingest_tool<'scope>(
    call: &ToolCallRequest,
    study: &mut ChunkStudyContext<'scope>,
    sender: &Sender<IngestEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    send_log(
        sender,
        "ingest",
        format!(
            "chunk {}/{}: tool {}",
            study.chunk_index + 1,
            study.chunk_count,
            call.name
        ),
    );
    if let Some(tool_call) = agent::parse_tool_call_request(call) {
        return apply_ingest_mutation(tool_call, study);
    }
    match agent::sanitized_tool_name(&call.name) {
        "search_web" => execute_search_web(&call.args, study, sender),
        "note_curiosity" => execute_note_curiosity(&call.args, study),
        _ => Ok("unknown tool".to_string()),
    }
}

fn apply_ingest_mutation<'scope>(
    tool_call: ToolCall,
    study: &mut ChunkStudyContext<'scope>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let payload = study.provenance_payload();
    let options = ApplyToolOptions {
        actor: Actor::new("agent", "ingestor"),
        context: EventContext::new(study.session_id, study.turn_id),
        source_message_id: study.source_message_id,
        event_payload: &payload,
        confidence_cap: Some(edge_ceiling(PROVENANCE_WEB_SURFACED, None)),
        provenance: PROVENANCE_WEB_SURFACED,
        default_memory_type: "semantic",
    };
    agent::apply_tool_call(tool_call, study.conn, &options, study.events)
}

impl<'scope> ChunkStudyContext<'scope> {
    fn provenance_payload(&self) -> String {
        match &self.provenance {
            MutationProvenance::Document => json!({
                "source": self.doc_path,
                "chunk": self.chunk_index + 1,
                "via": "document"
            })
            .to_string(),
            MutationProvenance::Web { query } => json!({
                "source": self.doc_path,
                "chunk": self.chunk_index + 1,
                "via": "web_search",
                "query": query
            })
            .to_string(),
        }
    }
}

fn execute_search_web<'scope>(
    args: &str,
    study: &mut ChunkStudyContext<'scope>,
    sender: &Sender<IngestEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let root = parse_json(args)?;
    let queries = parse_queries(&root);
    let reason = parse_string_field(&root, "reason");
    let conflicting_belief = parse_string_field(&root, "conflicting_belief");
    if queries.is_empty() {
        log_search_decision(sender, study, "denied", "no queries");
        return Ok("search denied: provide 1-3 queries".to_string());
    }
    if reason == "contradiction" {
        if conflicting_belief.trim().is_empty() {
            log_search_decision(sender, study, "denied", "missing conflicting_belief");
            return Ok(
                "search denied: contradiction searches must name conflicting_belief".to_string(),
            );
        }
        if study.search_budget.contradiction_used >= study.search_budget.contradiction_cap {
            log_search_decision(sender, study, "denied", "contradiction hard cap");
            return Ok("search denied: contradiction hard cap reached".to_string());
        }
        study.search_budget.contradiction_used += 1;
        return perform_search(queries, study, sender, "allowed contradiction");
    }
    if reason != "curiosity" {
        log_search_decision(sender, study, "denied", "invalid reason");
        return Ok("search denied: reason must be contradiction or curiosity".to_string());
    }
    if study.search_budget.curiosity_remaining == 0 {
        log_search_decision(sender, study, "denied", "curiosity budget exhausted");
        return Ok(
            "search denied: curiosity budget exhausted; use note_curiosity instead".to_string(),
        );
    }
    study.search_budget.curiosity_remaining -= 1;
    perform_search(queries, study, sender, "allowed curiosity")
}

fn perform_search<'scope>(
    queries: Vec<String>,
    study: &mut ChunkStudyContext<'scope>,
    sender: &Sender<IngestEvent>,
    decision: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let joined = queries.join(" | ");
    log_search_decision(sender, study, decision, &joined);
    let Some(search_client) = study.search else {
        return Ok("search unavailable: no search client configured".to_string());
    };
    let result = match search_client.search_many(&queries) {
        Some(result) if !result.trim().is_empty() => result,
        _ => "no results".to_string(),
    };
    study.provenance = MutationProvenance::Web { query: joined };
    study.rounds_since_search = 0;
    Ok(result)
}

fn expire_web_provenance<'scope>(study: &mut ChunkStudyContext<'scope>) {
    if matches!(study.provenance, MutationProvenance::Document) {
        return;
    }
    study.rounds_since_search += 1;
    if study.rounds_since_search > 1 {
        study.provenance = MutationProvenance::Document;
    }
}

#[allow(clippy::too_many_lines)]
fn execute_note_curiosity<'scope>(
    args: &str,
    study: &mut ChunkStudyContext<'scope>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let root = parse_json(args)?;
    let question = parse_string_field(&root, "question");
    let topic = parse_string_field(&root, "topic");
    if question.trim().is_empty() || topic.trim().is_empty() {
        return Ok("note denied: question and topic are required".to_string());
    }
    let note_id = curiosity_note_id(study.doc_path, study.chunk_index, &question);
    let note_node = Node {
        node_type: "note".to_string(),
        body: question.clone(),
        date: timeutil::timestamp_now(),
    };
    let topic_id = normalize_id(&topic);
    let topic_node = Node {
        node_type: "entity".to_string(),
        body: String::new(),
        date: String::new(),
    };
    db::write_node(study.conn, &topic_id, &topic_node)?;
    db::write_node(study.conn, &note_id, &note_node)?;
    let edge = Edge {
        source: note_id.clone(),
        target: topic_id.clone(),
        content: format!(
            "curiosity while ingesting {}, chunk {}",
            study.doc_path,
            study.chunk_index + 1
        ),
        stability: 0.5,
        confidence: 0.5,
        importance: 0.5,
        subject: false,
        date: timeutil::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: PROVENANCE_WEB_SURFACED.to_string(),
        ceiling: None,
    };
    let existed = db::load_edge(study.conn, &edge.source, &edge.target)?.is_some();
    db::write_edge(study.conn, &edge, study.source_message_id)?;
    study.events.push(Event::upsert_with_payload(
        Actor::new("agent", "ingestor"),
        EdgeRef::new(&note_id, &topic_id),
        EventContext::new(study.session_id, study.turn_id),
        existed,
        json!({
            "source": study.doc_path,
            "chunk": study.chunk_index + 1,
            "via": "document"
        })
        .to_string(),
    ));
    Ok(format!("curiosity noted: {question}"))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn record_skipped_chunk(
    conn: &Connection,
    doc_node_id: &str,
    doc_path: &str,
    chunk_index: usize,
    summary: &str,
    turn_id: &str,
    source_message_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut events = Vec::new();
    let result = (|| {
        let fact_id = normalize_id(&format!("{doc_path}-skim-chunk-{}", chunk_index + 1));
        let fact = Node {
            node_type: "fact".to_string(),
            body: summary.to_string(),
            date: timeutil::today_iso(),
        };
        db::write_node(conn, &fact_id, &fact)?;
        // The summary must live on the edge: edge content is what retrieval
        // embeds and injects — a body-only node is invisible to recall.
        let edge = Edge {
            source: fact_id.clone(),
            target: doc_node_id.to_string(),
            content: summary.to_string(),
            stability: 0.7,
            confidence: 0.3,
            importance: 0.5,
            subject: true,
            date: timeutil::today_iso(),
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: PROVENANCE_WEB_SURFACED.to_string(),
            ceiling: None,
        };
        let existed = db::load_edge(conn, &edge.source, &edge.target)?.is_some();
        db::write_edge(conn, &edge, source_message_id)?;
        events.push(Event::upsert_with_payload(
            Actor::new("agent", "ingestor"),
            EdgeRef::new(&fact_id, doc_node_id),
            EventContext::new(
                db::source_session_id(source_message_id)
                    .as_deref()
                    .unwrap_or(""),
                turn_id,
            ),
            existed,
            json!({
                "source": doc_path,
                "chunk": chunk_index + 1,
                "via": "document"
            })
            .to_string(),
        ));
        db::block_on(ledger::record_events_in_tx(conn, &events))
    })();
    if let Err(err) = result {
        let _ = db::block_on(async { conn.execute("ROLLBACK", ()).await });
        return Err(err);
    }
    db::commit_transaction(conn)?;
    Ok(())
}

fn injected_events(
    edges: &[SemanticEdge],
    session_id: &str,
    turn_id: &str,
    actor: Actor,
) -> Vec<Event> {
    let mut events = Vec::new();
    for edge in edges {
        events.push(Event::injected(
            actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            EventContext::new(session_id, turn_id),
        ));
    }
    events
}

fn ingestion_tools() -> Value {
    let mut tools = match agent::memory_mutation_tools() {
        Value::Array(array) => array,
        _ => Vec::new(),
    };
    tools.push(search_web_tool());
    tools.push(note_curiosity_tool());
    Value::Array(tools)
}

fn search_web_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web to resolve a contradiction or satisfy bounded curiosity while studying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "enum": ["contradiction", "curiosity"]},
                    "conflicting_belief": {"type": "string"}
                },
                "required": ["queries", "reason"]
            }
        }
    })
}

fn note_curiosity_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "note_curiosity",
            "description": "Write down an interesting question instead of following a rabbit hole now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "topic": {"type": "string"}
                },
                "required": ["question", "topic"]
            }
        }
    })
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

fn stream_turn_noop_chunks(
    llm: &dyn LlmClient,
    messages: &[Value],
    tools: &Value,
) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
    let (chunk_sender, _chunk_receiver) = channel::<Chunk>();
    llm.stream_turn(messages, Some(tools), &chunk_sender)
}

fn skim_chunks(
    llm: &dyn LlmClient,
    chunks: &[String],
    focus: &str,
    sender: &Sender<IngestEvent>,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut summaries = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let _timer = runlog::time_step("ingest", format!("skim {}/{}", index + 1, chunks.len()));
        send_log(
            sender,
            "ingest",
            format!("skim {}/{}", index + 1, chunks.len()),
        );
        summaries.push(skim_chunk(llm, chunk, focus)?);
    }
    Ok(summaries)
}

fn skim_chunk(llm: &dyn LlmClient, chunk: &str, focus: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let focus_line = if focus.is_empty() {
        String::new()
    } else {
        format!("\nThe reader asked to focus on: {focus}")
    };
    let prompt = format!(
        "Summarize this document chunk in 1-2 sentences. Be concrete and terse.{focus_line}\n\n{chunk}"
    );
    Ok(llm.collect(&prompt)?.trim().to_string())
}

fn choose_full_chunks(llm: &dyn LlmClient, summaries: &[String]) -> Vec<bool> {
    if summaries.len() <= INGEST_MAX_FULL_CHUNKS {
        return vec![true; summaries.len()];
    }
    let prompt = format!(
        "A document has {} chunk summaries below. Pick the {} chunks that most deserve full sequential study. Respond with raw JSON: {{\"full_chunks\":[1,2,...]}} using 1-based indices.\n\n{}",
        summaries.len(),
        INGEST_MAX_FULL_CHUNKS,
        build_document_map(summaries)
    );
    let response = match llm.collect(&prompt) {
        Ok(response) => response,
        Err(_) => return fallback_full_mask(summaries.len()),
    };
    parse_full_chunk_mask(&response, summaries.len())
        .unwrap_or_else(|| fallback_full_mask(summaries.len()))
}

fn parse_full_chunk_mask(text: &str, count: usize) -> Option<Vec<bool>> {
    let root = serde_json::from_str::<Value>(text.trim()).ok()?;
    let values = root.get("full_chunks")?.as_array()?;
    let mut mask = vec![false; count];
    for value in values {
        let index = usize::try_from(value.as_u64()?).ok()?;
        if index == 0 || index > count {
            continue;
        }
        mask[index - 1] = true;
    }
    for selected in &mask {
        if *selected {
            return Some(mask);
        }
    }
    None
}

fn fallback_full_mask(count: usize) -> Vec<bool> {
    let mut mask = vec![false; count];
    let mut index = 0usize;
    while index < count && index < INGEST_MAX_FULL_CHUNKS {
        mask[index] = true;
        index += 1;
    }
    mask
}

fn build_document_map(summaries: &[String]) -> String {
    let mut output = String::new();
    for (index, summary) in summaries.iter().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        output.push_str(&format!("{}: {}", index + 1, summary));
    }
    output
}

fn chunk_document(document: &str, target_chars: usize) -> Vec<String> {
    let paragraphs = split_paragraphs(document);
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in paragraphs {
        let addition = if current.is_empty() {
            paragraph.len()
        } else {
            paragraph.len() + 2
        };
        if !current.is_empty() && current.len() + addition > target_chars {
            chunks.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(&paragraph);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_paragraphs(document: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = Vec::new();
    for line in document.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join("\n"));
                current.clear();
            }
            continue;
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        paragraphs.push(current.join("\n"));
    }
    if paragraphs.is_empty() && !document.trim().is_empty() {
        paragraphs.push(document.trim().to_string());
    }
    paragraphs
}

fn curiosity_budget(chunk_count: usize) -> usize {
    (chunk_count / INGEST_SEARCH_BUDGET_DIVISOR).clamp(1, 6)
}

fn log_search_decision<'scope>(
    sender: &Sender<IngestEvent>,
    study: &ChunkStudyContext<'scope>,
    status: &str,
    detail: &str,
) {
    send_log(
        sender,
        "ingest",
        format!(
            "search {status} — chunk {}/{} — {}",
            study.chunk_index + 1,
            study.chunk_count,
            detail
        ),
    );
}

fn validate_ingest_path(path: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    if !path.exists() {
        return Err(format!("ingest file not found: {}", path.display()).into());
    }
    if !path.is_file() {
        return Err(format!("ingest path is not a file: {}", path.display()).into());
    }
    let Some(extension_value) = path.extension() else {
        return Err("ingest file must be .md or .txt".into());
    };
    let Some(extension) = extension_value.to_str() else {
        return Err("ingest file must be .md or .txt".into());
    };
    if extension != "md" && extension != "txt" {
        return Err("ingest file must be .md or .txt".into());
    }
    Ok(())
}

fn parse_json(args: &str) -> Result<Value, Box<dyn Error + Send + Sync>> {
    Ok(serde_json::from_str::<Value>(args)?)
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

fn curiosity_note_id(doc_path: &str, chunk_index: usize, question: &str) -> String {
    normalize_id(&format!(
        "note {} {} {}",
        doc_path,
        chunk_index + 1,
        question
    ))
}

fn ingestion_session_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("ingest-{seconds}")
}

fn send_log(sender: &Sender<IngestEvent>, phase: &'static str, text: String) {
    runlog::log(phase, &text);
    let _ = sender.send(IngestEvent::Log { phase, text });
}

#[cfg(test)]
mod tests {
    use super::{INGEST_MAX_FULL_CHUNKS, chunk_document, curiosity_budget, parse_full_chunk_mask};

    #[test]
    fn chunking_packs_paragraphs_without_splitting() {
        let document = "a\n\na\n\na\n\na";
        let chunks = chunk_document(document, 4);
        assert_eq!(chunks, vec!["a\n\na".to_string(), "a\n\na".to_string()]);
    }

    #[test]
    fn parsed_full_chunk_mask_uses_one_based_indices() {
        let mask = parse_full_chunk_mask(r#"{"full_chunks":[2,4]}"#, 4);
        assert_eq!(mask, Some(vec![false, true, false, true]));
    }

    #[test]
    fn curiosity_budget_is_clamped() {
        assert_eq!(curiosity_budget(1), 1);
        assert_eq!(curiosity_budget(16), 4);
        assert_eq!(curiosity_budget(200), 6);
    }

    #[test]
    fn full_chunk_limit_constant_matches_expectation() {
        assert_eq!(INGEST_MAX_FULL_CHUNKS, 25);
    }
}
