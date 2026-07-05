use std::error::Error;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use turso::Connection;

use crate::api::{EmbeddingClient, LlmClient};
use crate::prompts::{
    self, DREAM_BIOLOGIST, DREAM_CHILD, DREAM_CONNECTOR, DREAM_CONSOLIDATION, DREAM_ECONOMIST,
    DREAM_HISTORIAN, DREAM_LATERAL, DREAM_SKEPTIC,
};
use crate::runlog;

use super::agent::{ToolCall, parse_tool_calls, stream_collect};
use super::db;
use super::embed::{self, SemanticEdge};
use super::graph::{Edge, Node, graph_to_json, normalize_id};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};
use super::synthesis::{self, SynthesisPolicy};

const DREAM_CONFIDENCE_CAP: f64 = 0.6;
const DREAM_STABILITY: f64 = 0.3;
const MAX_PRUNES_PER_PASS: usize = 3; // replay-tunable.
const PRUNE_GUARD_THRESHOLD: f64 = 0.7; // replay-tunable.
pub const DOMAIN_AGENT_NAMES: [&str; 3] = ["biologist", "historian", "economist"];
const BIOLOGIST_ANCHOR: &str =
    "biology ecosystem evolution adaptation living organisms life systems";
const HISTORIAN_ANCHOR: &str = "history timeline events temporal sequence causation past future";
const ECONOMIST_ANCHOR: &str = "economics incentives trade-offs optimization value cost benefit";

pub fn domain_anchor(agent_name: &str) -> Option<&'static str> {
    match agent_name {
        "biologist" => Some(BIOLOGIST_ANCHOR),
        "historian" => Some(HISTORIAN_ANCHOR),
        "economist" => Some(ECONOMIST_ANCHOR),
        _ => None,
    }
}

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
    let _write_guard = db::acquire_write_lock();
    match execute_dream_pass(Arc::clone(&llm), Arc::clone(&embed), &sender) {
        Ok(edges_added) => {
            db::sync_push_best_effort("dream");
            let _ = sender.send(DreamEvent::Done { edges_added });
            Ok(())
        }
        Err(err) => {
            // Not necessarily "graph unchanged": turso has been observed to
            // dissolve a transaction mid-pass, auto-committing statements —
            // the final COMMIT then fails but writes are already durable.
            send_log(
                &sender,
                "dream",
                format!("pass failed — writes may be partial: {err}"),
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
    let conn = db::open_connection()?;
    let synthesis = synthesis::run_synthesis_stage(
        Arc::clone(&llm),
        &conn,
        |text| send_log(sender, "synthesis", text),
        |text| send_append(sender, "synthesis", text),
    )?;
    let all_ctx = consolidation_context(&domain, &abstract_outputs, &synthesis.proposal_text);
    let consolidation = run_consolidation_agent(llm, &session_id, &global_toon, &all_ctx, sender)?;
    let tool_calls = parse_tool_calls(&consolidation);
    let source_id = format!("dream:{session_id}");
    let edges_added = apply_dream_tool_calls(
        &conn,
        tool_calls,
        &source_id,
        &synthesis.policy,
        sender,
    )?;
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
        (DOMAIN_AGENT_NAMES[0], DREAM_BIOLOGIST, toons.biologist),
        (DOMAIN_AGENT_NAMES[1], DREAM_HISTORIAN, toons.historian),
        (DOMAIN_AGENT_NAMES[2], DREAM_ECONOMIST, toons.economist),
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
    let biologist = exploration_subgraph(embed, BIOLOGIST_ANCHOR, DOMAIN_AGENT_NAMES[0]);
    record_domain_views(DOMAIN_AGENT_NAMES[0], &biologist, session_id);
    let historian = exploration_subgraph(embed, HISTORIAN_ANCHOR, DOMAIN_AGENT_NAMES[1]);
    record_domain_views(DOMAIN_AGENT_NAMES[1], &historian, session_id);
    let economist = exploration_subgraph(embed, ECONOMIST_ANCHOR, DOMAIN_AGENT_NAMES[2]);
    record_domain_views(DOMAIN_AGENT_NAMES[2], &economist, session_id);
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
    if let Err(err) = ledger::record_events(&events) {
        runlog::log("ledger", &format!("write failed: {err}"));
    }
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

fn consolidation_context(
    domain: &DomainOutputs,
    abstract_outputs: &AbstractOutputs,
    synthesis: &str,
) -> String {
    let mut proposals: Vec<(&str, &str)> = vec![
        ("biologist", &domain.biologist),
        ("historian", &domain.historian),
        ("economist", &domain.economist),
        ("child", &domain.child),
        ("connector", &abstract_outputs.connector),
        ("skeptic", &abstract_outputs.skeptic),
        ("lateral", &abstract_outputs.lateral),
    ];
    if !synthesis.trim().is_empty() {
        proposals.push(("synthesis", synthesis));
    }
    format_proposals(&proposals)
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
    synthesis_policy: &SynthesisPolicy,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut edge_count = 0usize;
    let mut events: Vec<Event> = Vec::new();
    let mut apply_state = DreamApplyState::new(synthesis_policy);
    let result = (|| {
        let mut call_ctx = DreamCallContext {
            conn,
            source_id,
            events: &mut events,
            apply_state: &mut apply_state,
        };
        for call in tool_calls {
            let (description, added_edge) = apply_dream_call(&mut call_ctx, call)?;
            if !description.is_empty() {
                send_log(sender, "dream", description);
            }
            if added_edge {
                edge_count += 1;
            }
        }
        db::block_on(ledger::record_events_in_tx(conn, call_ctx.events))
    })();
    if let Err(err) = result {
        if let Err(rollback_err) = db::block_on(async { conn.execute("ROLLBACK", ()).await }) {
            runlog::log("ledger", &format!("rollback failed: {rollback_err}"));
        }
        return Err(err);
    }
    db::commit_transaction(conn)?;
    Ok(edge_count)
}

fn apply_dream_call<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    call: ToolCall,
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
            ctx,
            DreamEdgeCall {
                source,
                target,
                content,
                confidence,
                importance,
                subject,
                date,
            },
        ),
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => apply_dream_upsert_node(ctx, DreamNodeCall::new(name, node_type, body, date)),
        ToolCall::DeleteEdge { source, target } => {
            apply_dream_delete_edge(ctx, DreamDeleteEdgeCall::new(source, target))
        }
        ToolCall::DeleteNode { name } => apply_dream_delete_node(ctx, name),
        ToolCall::ExpireEdge {
            source,
            target,
            rewritten_content,
        } => apply_dream_expire_edge(ctx, DreamExpireEdgeCall::new(source, target, rewritten_content)),
    }
}

struct DreamApplyState<'policy> {
    prunes_used: usize,
    synthesis_policy: &'policy SynthesisPolicy,
}

impl<'policy> DreamApplyState<'policy> {
    fn new(synthesis_policy: &'policy SynthesisPolicy) -> Self {
        Self {
            prunes_used: 0,
            synthesis_policy,
        }
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

struct DreamNodeCall {
    name: String,
    node_type: String,
    body: String,
    date: String,
}

struct DreamDeleteEdgeCall {
    source: String,
    target: String,
}

struct DreamExpireEdgeCall {
    source: String,
    target: String,
    rewritten_content: String,
}

struct DreamCallContext<'scope, 'policy> {
    conn: &'scope Connection,
    source_id: &'scope str,
    events: &'scope mut Vec<Event>,
    apply_state: &'scope mut DreamApplyState<'policy>,
}

impl DreamNodeCall {
    fn new(name: String, node_type: String, body: String, date: String) -> Self {
        Self {
            name,
            node_type,
            body,
            date,
        }
    }
}

impl DreamDeleteEdgeCall {
    fn new(source: String, target: String) -> Self {
        Self { source, target }
    }
}

impl DreamExpireEdgeCall {
    fn new(source: String, target: String, rewritten_content: String) -> Self {
        Self {
            source,
            target,
            rewritten_content,
        }
    }
}

fn apply_dream_upsert_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    edge_call: DreamEdgeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let source = normalize_id(&edge_call.source);
    let target = normalize_id(&edge_call.target);
    let existing = db::load_edge(ctx.conn, &source, &target)?;
    let mut stability = DREAM_STABILITY;
    let mut confidence = edge_call.confidence.min(DREAM_CONFIDENCE_CAP);
    if let Some(override_values) = ctx
        .apply_state
        .synthesis_policy
        .edge_overrides
        .get(&(source.clone(), target.clone()))
    {
        stability = override_values.stability;
        confidence = override_values.confidence;
    }
    let edge = Edge {
        source: source.clone(),
        target: target.clone(),
        content: edge_call.content,
        stability,
        confidence,
        importance: edge_call.importance,
        subject: edge_call.subject,
        date: edge_call.date,
        occurred_at: None,
        expired_at: None,
    };
    db::write_edge(ctx.conn, &edge, ctx.source_id)?;
    ctx.events.push(Event::upsert(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&edge.source, &edge.target),
        EventContext::new(ctx.source_id, ""),
        existing.is_some(),
    ));
    Ok((
        format!("upsert_edge: {} → {}", edge_call.source, edge_call.target),
        true,
    ))
}

fn apply_dream_upsert_node<'scope, 'policy>(
    ctx: &DreamCallContext<'scope, 'policy>,
    node_call: DreamNodeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&node_call.name);
    let node_type = resolved_node_type(&normalized, &node_call.node_type, ctx.apply_state);
    let node = Node {
        node_type: node_type.clone(),
        body: node_call.body,
        date: node_call.date,
    };
    db::write_node(ctx.conn, &normalized, &node)?;
    Ok((format!("upsert_node: {} ({node_type})", node_call.name), false))
}

fn resolved_node_type<'policy>(
    node_id: &str,
    requested: &str,
    state: &DreamApplyState<'policy>,
) -> String {
    if state.synthesis_policy.synthesis_nodes.contains(node_id) {
        return "synthesis".to_string();
    }
    requested.to_string()
}

fn apply_dream_delete_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    edge_call: DreamDeleteEdgeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&edge_call.source);
    let normalized_target = normalize_id(&edge_call.target);
    let deleted_edge = db::load_edge(ctx.conn, &normalized_source, &normalized_target)?;
    match deleted_edge {
        Some(edge) => delete_existing_edge(ctx, edge_call, edge),
        None => Ok((
            format!(
                "delete_edge skipped: {} → {} (not found)",
                edge_call.source, edge_call.target
            ),
            false,
        )),
    }
}

fn delete_existing_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    edge_call: DreamDeleteEdgeCall,
    edge: Edge,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    if let Some(reason) = prune_block_reason(&edge, ctx.apply_state.prunes_used) {
        return Ok((
            format!(
                "delete_edge skipped: {} → {} ({reason})",
                edge_call.source, edge_call.target
            ),
            false,
        ));
    }
    let source = normalize_id(&edge_call.source);
    let target = normalize_id(&edge_call.target);
    db::erase_edge(ctx.conn, &source, &target)?;
    ctx.apply_state.prunes_used += 1;
    let pruned = ledger::pruned_events(
        Actor::new("system", "consolidation"),
        EventContext::new(ctx.source_id, ""),
        &[edge],
    )?;
    ctx.events.extend(pruned);
    Ok((
        format!("delete_edge: {} → {}", edge_call.source, edge_call.target),
        false,
    ))
}

fn apply_dream_delete_node<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    name: String,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let deleted_edges = db::load_edges_for_node(ctx.conn, &normalized)?;
    if deleted_edges.is_empty() {
        return Ok((format!("delete_node skipped: {name} (not found)"), false));
    }
    if let Some(reason) = delete_node_block_reason(&deleted_edges, ctx.apply_state.prunes_used) {
        return Ok((format!("delete_node skipped: {name} ({reason})"), false));
    }
    db::erase_node(ctx.conn, &normalized)?;
    ctx.apply_state.prunes_used += deleted_edges.len();
    let pruned = ledger::pruned_events(
        Actor::new("system", "consolidation"),
        EventContext::new(ctx.source_id, ""),
        &deleted_edges,
    )?;
    ctx.events.extend(pruned);
    Ok((format!("delete_node: {name}"), false))
}

fn delete_node_block_reason(edges: &[Edge], used_prunes: usize) -> Option<String> {
    let mut idx = 0usize;
    while idx < edges.len() {
        if let Some(reason) = prune_block_reason(&edges[idx], used_prunes + idx) {
            return Some(reason);
        }
        idx += 1;
    }
    if used_prunes + edges.len() > MAX_PRUNES_PER_PASS {
        return Some(format!("prune cap {}", MAX_PRUNES_PER_PASS));
    }
    None
}

fn apply_dream_expire_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    edge_call: DreamExpireEdgeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&edge_call.source);
    let normalized_target = normalize_id(&edge_call.target);
    let Some(mut edge) = db::load_edge(ctx.conn, &normalized_source, &normalized_target)? else {
        return Ok((
            format!(
                "expire_edge skipped: {} → {} (not found)",
                edge_call.source, edge_call.target
            ),
            false,
        ));
    };
    edge.content = edge_call.rewritten_content;
    edge.expired_at = Some(crate::timeutil::timestamp_now());
    db::write_edge(ctx.conn, &edge, ctx.source_id)?;
    let payload = serde_json::json!({"expired": true}).to_string();
    ctx.events.push(Event::updated(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&edge.source, &edge.target),
        EventContext::new(ctx.source_id, ""),
        payload,
    ));
    Ok((
        format!("expire_edge: {} → {}", edge_call.source, edge_call.target),
        false,
    ))
}

fn prune_block_reason(edge: &Edge, prunes_used: usize) -> Option<String> {
    if prunes_used >= MAX_PRUNES_PER_PASS {
        return Some(format!("prune cap {}", MAX_PRUNES_PER_PASS));
    }
    if edge.importance >= 1.0 {
        return Some("importance >= 1.0".to_string());
    }
    if edge.confidence >= PRUNE_GUARD_THRESHOLD {
        return Some(format!("confidence >= {}", PRUNE_GUARD_THRESHOLD));
    }
    if edge.stability >= PRUNE_GUARD_THRESHOLD {
        return Some(format!("stability >= {}", PRUNE_GUARD_THRESHOLD));
    }
    None
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
    runlog::log(phase, &text);
    let _ = sender.send(DreamEvent::Log { phase, text });
}

fn send_append(sender: &std::sync::mpsc::Sender<DreamEvent>, phase: &'static str, text: String) {
    let _ = sender.send(DreamEvent::Append { phase, text });
}
