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
use super::hygiene;
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};
use super::synthesis::{self, SynthesisPolicy};

const DREAM_CONFIDENCE_CAP: f64 = 0.6;
const DREAM_STABILITY: f64 = 0.3;
// A gist is an induced generalization — stronger than a raw dream hypothesis
// (0.6) is allowed to be, but only slightly.
const GIST_CONFIDENCE_CAP: f64 = 0.7; // replay-tunable.
const GIST_IMPORTANCE: f64 = 0.5; // replay-tunable.
const GIST_MIN_SUPPORTS: usize = 2; // replay-tunable.
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

// A clean graph logs nothing — hygiene should be invisible until it finds rot.
fn run_hygiene_pass(
    conn: &Connection,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let report = hygiene::run_hygiene(conn)?;
    if report.is_clean() {
        return Ok(());
    }
    send_log(sender, "hygiene", report.summary());
    Ok(())
}

fn execute_dream_pass(
    llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let session_id = dream_session_id();
    let created_at = dream_timestamp();
    db::register_dream_session(&session_id, &created_at);
    let conn = db::open_connection()?;
    // Repair before the agents read the graph, so they reason over clean state.
    run_hygiene_pass(&conn, sender)?;
    let graph = db::load_graph();
    let global_toon = toon::encode(&graph_to_json(&graph), None);
    let domain = run_domain_agents(Arc::clone(&llm), embed, &session_id, &global_toon, sender)?;
    let abstract_outputs =
        run_abstract_agents(Arc::clone(&llm), &session_id, &global_toon, &domain, sender)?;
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
    let edges_added =
        apply_dream_tool_calls(&conn, tool_calls, &source_id, &synthesis.policy, sender)?;
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
    let biologist =
        exploration_subgraph(embed, BIOLOGIST_ANCHOR, DOMAIN_AGENT_NAMES[0], session_id);
    record_domain_views(DOMAIN_AGENT_NAMES[0], &biologist, session_id);
    let historian =
        exploration_subgraph(embed, HISTORIAN_ANCHOR, DOMAIN_AGENT_NAMES[1], session_id);
    record_domain_views(DOMAIN_AGENT_NAMES[1], &historian, session_id);
    let economist =
        exploration_subgraph(embed, ECONOMIST_ANCHOR, DOMAIN_AGENT_NAMES[2], session_id);
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
    session_id: &str,
) -> Vec<SemanticEdge> {
    embed::select_exploration_edges(embed, anchor, agent_name, session_id)
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

#[allow(clippy::too_many_lines)]
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
            memory_type: _,
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
        } => apply_dream_expire_edge(
            ctx,
            DreamExpireEdgeCall::new(source, target, rewritten_content),
        ),
        ToolCall::PromoteEdge { source, target } => apply_dream_promote_edge(ctx, source, target),
        ToolCall::GistEdge {
            source,
            target,
            content,
            confidence,
            stability,
            supported_by,
        } => apply_dream_gist_edge(
            ctx,
            DreamGistCall {
                source,
                target,
                content,
                confidence,
                stability,
                supported_by,
            },
        ),
        ToolCall::RecordMove { .. } => Ok((
            "record_move denied: record_move is a scribe tool".to_string(),
            false,
        )),
    }
}

fn apply_dream_promote_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    source: String,
    target: String,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let description = super::agent::apply_promote_edge(
        ctx.conn,
        EdgeRef::new(&source, &target),
        Actor::new("system", "consolidation"),
        EventContext::new(ctx.source_id, ""),
        ctx.events,
    )?;
    Ok((description, false))
}

struct DreamGistCall {
    source: String,
    target: String,
    content: String,
    confidence: f64,
    stability: f64,
    supported_by: Vec<(String, String)>,
}

// Lossy consolidation: writes a new semantic edge for a cross-episode gist and
// records its provenance on the ledger. The supporting episodic edges are left
// untouched — they decay away on the fast episodic curve while the gist
// outlives them, which is the "lossy" half.
fn apply_dream_gist_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    call: DreamGistCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let supports = collect_gist_supports(ctx.conn, &call.supported_by)?;
    if supports.len() < GIST_MIN_SUPPORTS {
        return Ok((gist_denied_message(&call), false));
    }
    let source = normalize_id(&call.source);
    let target = normalize_id(&call.target);
    let edge = gist_edge(&source, &target, &call);
    db::write_edge(ctx.conn, &edge, ctx.source_id)?;
    ctx.events.push(Event::updated(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&source, &target),
        EventContext::new(ctx.source_id, ""),
        gist_payload(&supports),
    ));
    Ok((
        format!(
            "gist_edge: {source} → {target} (from {} episodes)",
            supports.len()
        ),
        true,
    ))
}

// Keeps only refs that resolve to an existing episodic edge; unknown or
// non-episodic refs are dropped with a log line (they may drop the count
// below the minimum, which the caller then denies).
fn collect_gist_supports(
    conn: &Connection,
    refs: &[(String, String)],
) -> Result<Vec<(String, String)>, Box<dyn Error + Send + Sync>> {
    let mut kept = Vec::new();
    for (source, target) in refs {
        match db::load_edge(conn, source, target)? {
            Some(edge) if edge.memory_type == "episodic" => {
                kept.push((source.clone(), target.clone()));
            }
            Some(_) => runlog::log(
                "dream",
                &format!("gist support dropped (not episodic): {source} → {target}"),
            ),
            None => runlog::log(
                "dream",
                &format!("gist support dropped (unknown): {source} → {target}"),
            ),
        }
    }
    Ok(kept)
}

fn gist_edge(source: &str, target: &str, call: &DreamGistCall) -> Edge {
    Edge {
        source: source.to_string(),
        target: target.to_string(),
        content: call.content.clone(),
        stability: call.stability.clamp(0.0, 1.0),
        confidence: call.confidence.clamp(0.0, GIST_CONFIDENCE_CAP),
        importance: GIST_IMPORTANCE,
        subject: false,
        date: crate::prompts::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
    }
}

fn gist_payload(supports: &[(String, String)]) -> String {
    let mut refs = Vec::new();
    for (source, target) in supports {
        refs.push(format!("{source} → {target}"));
    }
    serde_json::json!({ "gist_of": refs }).to_string()
}

fn gist_denied_message(call: &DreamGistCall) -> String {
    format!(
        "gist_edge denied: {} → {} needs {GIST_MIN_SUPPORTS}+ episodic supports — use promote_edge for one episode that is itself the lesson",
        call.source, call.target
    )
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
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
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
    Ok((
        format!("upsert_node: {} ({node_type})", node_call.name),
        false,
    ))
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

#[cfg(test)]
mod tests {
    use super::{
        DreamApplyState, DreamCallContext, DreamGistCall, Edge, Event, SynthesisPolicy,
        apply_dream_gist_edge,
    };
    use crate::memory::agent::{self, ToolCall};
    use crate::memory::db;
    use crate::memory::ledger::{Actor, EventContext};
    use std::collections::{HashMap, HashSet};
    use turso::{Builder, Connection};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";

    fn harness() -> Connection {
        db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            conn
        })
    }

    fn edge(source: &str, target: &str, memory_type: &str) -> Edge {
        Edge {
            source: source.to_string(),
            target: target.to_string(),
            content: "x".to_string(),
            stability: 0.2,
            confidence: 0.3,
            importance: 0.1,
            subject: false,
            date: "2020-01-01".to_string(),
            occurred_at: None,
            expired_at: None,
            memory_type: memory_type.to_string(),
            origin_session: "session-1".to_string(),
        }
    }

    fn insert(conn: &Connection, source: &str, target: &str, memory_type: &str) {
        let stored = edge(source, target, memory_type);
        let Ok(()) = db::write_edge(conn, &stored, "") else {
            unreachable!("edge written")
        };
    }

    fn gist_call(supports: &[(&str, &str)], confidence: f64) -> DreamGistCall {
        let mut supported_by = Vec::new();
        for (source, target) in supports {
            supported_by.push(((*source).to_string(), (*target).to_string()));
        }
        DreamGistCall {
            source: "queen-sorties".to_string(),
            target: "chess-openings".to_string(),
            content: "early queen sorties get punished".to_string(),
            confidence,
            stability: 0.5,
            supported_by,
        }
    }

    fn empty_policy() -> SynthesisPolicy {
        SynthesisPolicy {
            synthesis_nodes: HashSet::new(),
            edge_overrides: HashMap::new(),
        }
    }

    fn run_gist(
        conn: &Connection,
        call: DreamGistCall,
    ) -> ((String, bool), Vec<Event>) {
        let policy = empty_policy();
        let mut events: Vec<Event> = Vec::new();
        let mut apply_state = DreamApplyState::new(&policy);
        let mut ctx = DreamCallContext {
            conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let Ok(outcome) = apply_dream_gist_edge(&mut ctx, call) else {
            unreachable!("apply runs")
        };
        (outcome, events)
    }

    fn loaded(conn: &Connection, source: &str, target: &str) -> Option<Edge> {
        let Ok(edge) = db::load_edge(conn, source, target) else {
            unreachable!("load runs")
        };
        edge
    }

    fn must_load(conn: &Connection, source: &str, target: &str) -> Edge {
        let Some(edge) = loaded(conn, source, target) else {
            unreachable!("edge exists")
        };
        edge
    }

    // Two episodic supports, gisted at confidence 0.6. Returns the events.
    fn gist_two_supports(conn: &Connection) -> Vec<Event> {
        insert(conn, "game3-hub", "move-qh5", "episodic");
        insert(conn, "game4-hub", "move-qf3", "episodic");
        let ((description, added), events) = run_gist(
            conn,
            gist_call(&[("game3-hub", "move-qh5"), ("game4-hub", "move-qf3")], 0.6),
        );
        assert!(added, "a gist edge was written");
        assert!(description.starts_with("gist_edge:"));
        events
    }

    #[test]
    fn two_episodic_supports_write_a_semantic_gist() {
        let conn = harness();
        let _ = gist_two_supports(&conn);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.memory_type, "semantic");
        assert_eq!(written.date, crate::prompts::today_iso());
        assert!((written.confidence - 0.6).abs() < 1e-9);
    }

    #[test]
    fn a_gist_records_provenance_and_spares_its_supporters() {
        let conn = harness();
        let events = gist_two_supports(&conn);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "updated");
        assert!(events[0].payload.contains("gist_of"));
        assert!(events[0].payload.contains("game3-hub → move-qh5"));
        assert!(events[0].payload.contains("game4-hub → move-qf3"));

        // Supporters are left untouched — still episodic, not expired.
        let support = must_load(&conn, "game3-hub", "move-qh5");
        assert_eq!(support.memory_type, "episodic");
        assert!(support.expired_at.is_none());
    }

    #[test]
    fn confidence_is_clamped_to_the_gist_cap() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "game4-hub", "move-qf3", "episodic");
        let (_, _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("game4-hub", "move-qf3")], 0.95),
        );
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert!((written.confidence - 0.7).abs() < 1e-9);
    }

    #[test]
    fn a_single_support_is_denied_with_the_promote_edge_hint() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), events) =
            run_gist(&conn, gist_call(&[("game3-hub", "move-qh5")], 0.6));
        assert!(!added);
        assert!(description.contains("denied"));
        assert!(description.contains("promote_edge"));
        assert!(events.is_empty());
        assert!(loaded(&conn, "queen-sorties", "chess-openings").is_none());
    }

    #[test]
    fn an_unknown_ref_drops_the_count_below_the_minimum() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("ghost", "nowhere")], 0.6),
        );
        assert!(!added);
        assert!(description.contains("denied"));
    }

    #[test]
    fn a_semantic_support_is_dropped() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "old-belief", "chess", "semantic");
        let ((_, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("old-belief", "chess")], 0.6),
        );
        assert!(!added, "the semantic ref is dropped, leaving one support");
    }

    #[test]
    fn the_chat_apply_path_denies_gist_edge() {
        let conn = harness();
        let call = ToolCall::GistEdge {
            source: "a".to_string(),
            target: "b".to_string(),
            content: "c".to_string(),
            confidence: 0.6,
            stability: 0.5,
            supported_by: vec![("x".to_string(), "y".to_string())],
        };
        let options = agent::ApplyToolOptions {
            actor: Actor::new("system", "chat"),
            context: EventContext::new("session-1", "turn-1"),
            source_message_id: "msg-1",
            event_payload: "",
            confidence_cap: None,
            default_memory_type: "semantic",
        };
        let mut events: Vec<Event> = Vec::new();
        let Ok(description) = agent::apply_tool_call(call, &conn, &options, &mut events) else {
            unreachable!("apply runs")
        };
        assert!(description.contains("gist_edge is a dreaming tool"));
        assert!(events.is_empty());
    }
}
