use std::error::Error;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use turso::Connection;

use crate::api::{EmbeddingClient, LlmClient};
use crate::prompts::{
    self, DREAM_BIOLOGIST, DREAM_CHILD, DREAM_CONNECTOR, DREAM_CONSOLIDATION, DREAM_ECONOMIST,
    DREAM_HISTORIAN, DREAM_LATERAL, DREAM_SKEPTIC,
};

use super::agent::{ToolCall, parse_tool_calls, stream_collect};
use super::db;
use super::embed::{self, SemanticEdge};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};
use super::graph::{Edge, Node, graph_to_json, normalize_id};

const DREAM_CONFIDENCE_CAP: f64 = 0.6;
const DREAM_STABILITY: f64 = 0.3;
const BIOLOGIST_ANCHOR: &str =
    "biology ecosystem evolution adaptation living organisms life systems";
const HISTORIAN_ANCHOR: &str = "history timeline events temporal sequence causation past future";
const ECONOMIST_ANCHOR: &str = "economics incentives trade-offs optimization value cost benefit";

pub enum DreamEvent {
    Log { phase: &'static str, text: String },
    Append { phase: &'static str, text: String },
    Done { edges_added: usize },
}

pub fn run_dream_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    sender: std::sync::mpsc::Sender<DreamEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match execute_dream_pass(Arc::clone(&llm), Arc::clone(&embed), &sender) {
        Ok(edges_added) => {
            let _ = sender.send(DreamEvent::Done { edges_added });
            Ok(())
        }
        Err(err) => {
            send_log(
                &sender,
                "dream",
                format!("pass failed — graph unchanged: {err}"),
            );
            let _ = sender.send(DreamEvent::Done { edges_added: 0 });
            Err(err)
        }
    }
}

fn execute_dream_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let session_id = dream_session_id();
    let created_at = dream_timestamp();
    db::register_dream_session(&session_id, &created_at);
    let graph = db::load_graph();
    let global_toon = toon::encode(&graph_to_json(&graph), None);
    let domain = run_domain_agents(Arc::clone(&llm), embed, &session_id, &global_toon, sender)?;
    let abstract_outputs =
        run_abstract_agents(Arc::clone(&llm), &session_id, &global_toon, &domain, sender)?;
    let all_ctx = consolidation_context(&domain, &abstract_outputs);
    let consolidation = run_consolidation_agent(llm, &session_id, &global_toon, &all_ctx, sender)?;
    let tool_calls = parse_tool_calls(&consolidation);
    let source_id = format!("dream:{session_id}");
    let conn = db::open_connection()?;
    let edges_added = apply_dream_tool_calls(&conn, tool_calls, &source_id, sender)?;
    send_log(
        sender,
        "dream",
        format!("done — {edges_added} edges written"),
    );
    Ok(edges_added)
}

struct DomainOutputs {
    biologist: String,
    historian: String,
    economist: String,
    child: String,
}

struct DomainToons {
    biologist: String,
    historian: String,
    economist: String,
    child: String,
}

fn run_domain_agents(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    session_id: &str,
    global_toon: &str,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<DomainOutputs, Box<dyn Error + Send + Sync>> {
    type AgentHandle = std::thread::JoinHandle<Result<String, Box<dyn Error + Send + Sync>>>;
    let toons = build_domain_toons(&*embed, global_toon, session_id);
    let specs = [
        ("biologist", DREAM_BIOLOGIST, toons.biologist),
        ("historian", DREAM_HISTORIAN, toons.historian),
        ("economist", DREAM_ECONOMIST, toons.economist),
        ("child", DREAM_CHILD, toons.child),
    ];
    let mut handles: Vec<AgentHandle> = Vec::new();
    for (phase, explanation, toon) in specs {
        let llm_clone = Arc::clone(&llm);
        let session = session_id.to_string();
        let prompt = prompts::build_dream_prompt(explanation, &toon, None);
        let agent_tx = sender.clone();
        handles.push(std::thread::spawn(move || {
            run_dream_agent(llm_clone, phase, prompt, &session, &agent_tx)
        }));
    }
    let mut outputs: Vec<String> = Vec::new();
    for handle in handles {
        outputs.push(join_agent(handle)?);
    }
    let child_out = outputs.pop().unwrap_or_default();
    let economist = outputs.pop().unwrap_or_default();
    let historian = outputs.pop().unwrap_or_default();
    let biologist = outputs.pop().unwrap_or_default();
    Ok(DomainOutputs {
        biologist,
        historian,
        economist,
        child: child_out,
    })
}

fn build_domain_toons(
    embed: &dyn EmbeddingClient,
    global_toon: &str,
    session_id: &str,
) -> DomainToons {
    let biologist = exploration_subgraph(embed, BIOLOGIST_ANCHOR, "biologist");
    record_domain_views("biologist", &biologist, session_id);
    let historian = exploration_subgraph(embed, HISTORIAN_ANCHOR, "historian");
    record_domain_views("historian", &historian, session_id);
    let economist = exploration_subgraph(embed, ECONOMIST_ANCHOR, "economist");
    record_domain_views("economist", &economist, session_id);
    DomainToons {
        biologist: embed::edges_to_toon(&biologist),
        historian: embed::edges_to_toon(&historian),
        economist: embed::edges_to_toon(&economist),
        child: global_toon.to_string(),
    }
}

fn exploration_subgraph(
    embed: &dyn EmbeddingClient,
    anchor: &str,
    agent_name: &str,
) -> Vec<SemanticEdge> {
    embed::select_exploration_edges(embed, anchor, agent_name)
}

// Best-effort bookkeeping: a failed ledger write must never abort the dream.
fn record_domain_views(agent_name: &str, edges: &[SemanticEdge], session_id: &str) {
    let mut events = Vec::new();
    for edge in edges {
        events.push(Event::injected(
            Actor::new("agent", agent_name),
            EdgeRef::new(&edge.source, &edge.target),
            EventContext::new(session_id, ""),
        ));
    }
    if events.is_empty() {
        return;
    }
    let _ = ledger::record_events(&events);
}

fn join_agent(
    handle: std::thread::JoinHandle<Result<String, Box<dyn Error + Send + Sync>>>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err("agent thread panicked".into()),
    }
}

struct AbstractOutputs {
    connector: String,
    skeptic: String,
    lateral: String,
}

fn run_abstract_agents(
    llm: Arc<dyn LlmClient>,
    session_id: &str,
    toon: &str,
    domain: &DomainOutputs,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<AbstractOutputs, Box<dyn Error + Send + Sync>> {
    let domain_ctx = domain_context(domain);
    let connector = run_dream_agent(
        Arc::clone(&llm),
        "connector",
        prompts::build_dream_prompt(DREAM_CONNECTOR, toon, Some(&domain_ctx)),
        session_id,
        sender,
    )?;
    let skeptic_ctx = format_proposals(&[("connector", &connector)]);
    let skeptic = run_dream_agent(
        Arc::clone(&llm),
        "skeptic",
        prompts::build_dream_prompt(DREAM_SKEPTIC, toon, Some(&skeptic_ctx)),
        session_id,
        sender,
    )?;
    let lateral_ctx = format_proposals(&[("connector", &connector), ("skeptic", &skeptic)]);
    let lateral = run_dream_agent(
        llm,
        "lateral",
        prompts::build_dream_prompt(DREAM_LATERAL, toon, Some(&lateral_ctx)),
        session_id,
        sender,
    )?;
    Ok(AbstractOutputs {
        connector,
        skeptic,
        lateral,
    })
}

fn run_consolidation_agent(
    llm: Arc<dyn LlmClient>,
    session_id: &str,
    toon: &str,
    all_ctx: &str,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    run_dream_agent(
        llm,
        "consolidation",
        prompts::build_dream_prompt(DREAM_CONSOLIDATION, toon, Some(all_ctx)),
        session_id,
        sender,
    )
}

fn domain_context(domain: &DomainOutputs) -> String {
    format_proposals(&[
        ("biologist", &domain.biologist),
        ("historian", &domain.historian),
        ("economist", &domain.economist),
        ("child", &domain.child),
    ])
}

fn consolidation_context(domain: &DomainOutputs, abstract_outputs: &AbstractOutputs) -> String {
    format_proposals(&[
        ("biologist", &domain.biologist),
        ("historian", &domain.historian),
        ("economist", &domain.economist),
        ("child", &domain.child),
        ("connector", &abstract_outputs.connector),
        ("skeptic", &abstract_outputs.skeptic),
        ("lateral", &abstract_outputs.lateral),
    ])
}

fn run_dream_agent(
    llm: Arc<dyn LlmClient>,
    phase: &'static str,
    prompt: String,
    session_id: &str,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    send_log(sender, "dream", format!("{phase}: calling LLM…"));
    send_log(sender, phase, String::new());
    let output = match stream_collect(
        llm,
        prompt,
        |text| send_append(sender, "think", text),
        |text| send_append(sender, phase, text),
    ) {
        Ok(output) => output,
        Err(err) => {
            send_log(sender, phase, format!("error: {err}"));
            return Err(err);
        }
    };
    let created_at = dream_timestamp();
    let _ = db::insert_message(session_id, phase, &output, &created_at);
    send_log(sender, phase, "done".to_string());
    Ok(output)
}

fn apply_dream_tool_calls(
    conn: &Connection,
    tool_calls: Vec<ToolCall>,
    source_id: &str,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut edge_count = 0usize;
    let mut events: Vec<Event> = Vec::new();
    let result = (|| {
        for call in tool_calls {
            let (description, added_edge) = apply_dream_call(conn, call, source_id, &mut events)?;
            if !description.is_empty() {
                send_log(sender, "dream", description);
            }
            if added_edge {
                edge_count += 1;
            }
        }
        db::block_on(ledger::record_events_in_tx(conn, &events))
    })();
    if let Err(err) = result {
        let _ = db::block_on(async { conn.execute("ROLLBACK", ()).await });
        return Err(err);
    }
    db::commit_transaction(conn)?;
    Ok(edge_count)
}

fn apply_dream_call(
    conn: &Connection,
    call: ToolCall,
    source_id: &str,
    events: &mut Vec<Event>,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    match call {
        ToolCall::UpsertEdge {
            source,
            target,
            content,
            stability: _,
            confidence,
            importance,
            subject,
            date,
        } => apply_dream_upsert_edge(
            conn,
            DreamEdgeCall {
                source,
                target,
                content,
                confidence,
                importance,
                subject,
                date,
            },
            source_id,
            events,
        ),
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => apply_dream_upsert_node(conn, name, node_type, body, date),
        ToolCall::DeleteEdge { source, target } => {
            apply_dream_delete_edge(conn, source, target, source_id, events)
        }
        ToolCall::DeleteNode { name } => apply_dream_delete_node(conn, name, source_id, events),
    }
}

struct DreamEdgeCall {
    source: String,
    target: String,
    content: String,
    confidence: f64,
    importance: f64,
    subject: bool,
    date: String,
}

fn apply_dream_upsert_edge(
    conn: &Connection,
    edge_call: DreamEdgeCall,
    source_id: &str,
    events: &mut Vec<Event>,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let existing = db::load_edge(
        conn,
        &normalize_id(&edge_call.source),
        &normalize_id(&edge_call.target),
    )?;
    let edge = Edge {
        source: normalize_id(&edge_call.source),
        target: normalize_id(&edge_call.target),
        content: edge_call.content,
        stability: DREAM_STABILITY,
        confidence: edge_call.confidence.min(DREAM_CONFIDENCE_CAP),
        importance: edge_call.importance,
        subject: edge_call.subject,
        date: edge_call.date,
        occurred_at: None,
        expired_at: None,
    };
    db::write_edge(conn, &edge, source_id)?;
    events.push(Event::upsert(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&edge.source, &edge.target),
        EventContext::new(source_id, ""),
        existing.is_some(),
    ));
    Ok((
        format!("upsert_edge: {} → {}", edge_call.source, edge_call.target),
        true,
    ))
}

fn apply_dream_upsert_node(
    conn: &Connection,
    name: String,
    node_type: String,
    body: String,
    date: String,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let node = Node {
        node_type: node_type.clone(),
        body,
        date,
    };
    db::write_node(conn, &normalized, &node)?;
    Ok((format!("upsert_node: {name} ({node_type})"), false))
}

fn apply_dream_delete_edge(
    conn: &Connection,
    source: String,
    target: String,
    source_id: &str,
    events: &mut Vec<Event>,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&source);
    let normalized_target = normalize_id(&target);
    let deleted_edge = db::load_edge(conn, &normalized_source, &normalized_target)?;
    db::erase_edge(conn, &normalized_source, &normalized_target)?;
    if let Some(edge) = deleted_edge {
        let pruned = ledger::pruned_events(
            Actor::new("system", "consolidation"),
            EventContext::new(source_id, ""),
            &[edge],
        )?;
        events.extend(pruned);
    }
    Ok((format!("delete_edge: {source} → {target}"), false))
}

fn apply_dream_delete_node(
    conn: &Connection,
    name: String,
    source_id: &str,
    events: &mut Vec<Event>,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let deleted_edges = db::load_edges_for_node(conn, &normalized)?;
    db::erase_node(conn, &normalized)?;
    let pruned = ledger::pruned_events(
        Actor::new("system", "consolidation"),
        EventContext::new(source_id, ""),
        &deleted_edges,
    )?;
    events.extend(pruned);
    Ok((format!("delete_node: {name}"), false))
}

fn format_proposals(agents: &[(&str, &str)]) -> String {
    let mut output = String::new();
    let mut idx = 0usize;
    while idx < agents.len() {
        let (name, text) = agents[idx];
        if idx > 0 {
            output.push_str("\n\n");
        }
        output.push_str("### ");
        output.push_str(name);
        output.push('\n');
        output.push_str(text);
        idx += 1;
    }
    output
}

fn dream_session_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("dream-{seconds}")
}

fn dream_timestamp() -> String {
    crate::timeutil::timestamp_now()
}

fn send_log(sender: &std::sync::mpsc::Sender<DreamEvent>, phase: &'static str, text: String) {
    let _ = sender.send(DreamEvent::Log { phase, text });
}

fn send_append(sender: &std::sync::mpsc::Sender<DreamEvent>, phase: &'static str, text: String) {
    let _ = sender.send(DreamEvent::Append { phase, text });
}
