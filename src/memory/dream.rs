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

use super::agent::{
    BatchPreflight, ToolCall, declared_node_denial, parse_tool_calls, preflight_tool_batch,
    stream_collect,
};
use super::db;
use super::embed::{self, SemanticEdge};
use super::graph::{
    Edge, Graph, Node, PROVENANCE_DREAMED, PROVENANCE_INFERRED, PROVENANCE_SYNTHESIZED,
    graph_to_json, inferred_ceiling, node_carries_value, normalize_id, stability_seed_max, synthesized_ceiling,
};
use super::hygiene;
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};
use super::synthesis::{self, SynthesisPolicy};

const GIST_IMPORTANCE: f64 = 0.5; // replay-tunable.
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
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let _timer = runlog::time_step("dream", "pass");
    let result = {
        let _write_guard = db::acquire_write_lock();
        execute_dream_pass(Arc::clone(&llm), Arc::clone(&embed), &sender)
    };
    match result {
        Ok(edges_added) => {
            db::sync_push_best_effort("dream");
            let _ = sender.send(DreamEvent::Done { edges_added });
            Ok(edges_added)
        }
        Err(err) => {
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

// A focused, game-scoped strategy pass per game — never the global TOON. For
// each game node that has at least one live played chain, assemble a compact
// context (definition, current rules/strategies, the ordered chains, outcomes)
// and run one LLM pass that writes portable strategy lessons. No games or no
// chains → silent no-op, same convention as hygiene.
fn run_strategy_stage(
    llm: &Arc<dyn LlmClient>,
    conn: &Connection,
    session_id: &str,
    sender: &std::sync::mpsc::Sender<DreamEvent>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let stage = StrategyStage {
        llm,
        conn,
        session_id,
        sender,
    };
    let games = db::load_game_nodes(conn)?;
    let mut total = 0usize;
    let mut processed = 0usize;
    for (game_id, node) in games {
        let hubs = db::load_game_hubs(conn, &game_id)?;
        if hubs.is_empty() {
            continue;
        }
        let ctx = assemble_strategy_context(conn, &game_id, &node, &hubs)?;
        let name = super::task_chain::game_display_name(&game_id);
        send_log(
            sender,
            "dream",
            format!(
                "strategy stage: {name} ({} chains, {} outcomes)",
                ctx.chains, ctx.outcomes
            ),
        );
        total += run_strategy_game(&stage, &name, &ctx.text)?;
        processed += 1;
    }
    if processed > 0 {
        send_log(
            sender,
            "dream",
            format!("strategy: {total} lessons written"),
        );
    }
    Ok(total)
}

// The shared handles a single game's strategy pass runs against — bundled so
// per-game helpers stay under the argument cap.
struct StrategyStage<'run> {
    llm: &'run Arc<dyn LlmClient>,
    conn: &'run Connection,
    session_id: &'run str,
    sender: &'run std::sync::mpsc::Sender<DreamEvent>,
}

struct GameStrategyContext {
    text: String,
    chains: usize,
    outcomes: usize,
}

// Pure assembly of one game's strategy context: definition body, current
// semantic rules/strategies, each played hub's ordered chain, and each
// session's recorded outcome. Small by construction — that is the point.
fn assemble_strategy_context(
    conn: &Connection,
    game_id: &str,
    node: &Node,
    hubs: &[String],
) -> Result<GameStrategyContext, Box<dyn Error + Send + Sync>> {
    let mut text = String::from("### Definition\n");
    text.push_str(&node.body);
    text.push('\n');
    append_current_edges(conn, game_id, &mut text)?;
    let mut chains = 0usize;
    let mut outcomes = 0usize;
    for hub in hubs {
        let session_id = hub.strip_prefix("task-").unwrap_or(hub);
        let chain = db::load_task_chain(conn, session_id, super::task_chain::MAX_CHAIN_INJECT)?;
        if chain.is_empty() {
            continue;
        }
        chains += 1;
        append_chain(&mut text, chains, session_id, &chain);
        if append_outcome(conn, session_id, &mut text)? {
            outcomes += 1;
        }
    }
    Ok(GameStrategyContext {
        text,
        chains,
        outcomes,
    })
}

fn append_current_edges(
    conn: &Connection,
    game_id: &str,
    text: &mut String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let edges = db::load_game_edges(conn, game_id, super::task_chain::MAX_GAME_KNOWLEDGE)?;
    if edges.is_empty() {
        return Ok(());
    }
    text.push_str("\n### Current rules & strategies\n");
    for edge in &edges {
        text.push_str(&edge.content);
        text.push('\n');
    }
    Ok(())
}

// Each line carries the chain edge's ref ("hub → move-node") before the
// content, because gist_edge's supported_by must cite these refs verbatim —
// the model can only copy what it can see.
fn append_chain(text: &mut String, index: usize, session_id: &str, chain: &[Edge]) {
    text.push_str(&format!("\n### Game {index} ({session_id})\n"));
    for edge in chain {
        text.push_str(&format!(
            "{} → {} | {}\n",
            edge.source, edge.target, edge.content
        ));
    }
}

// Appends the session's outcome line if one was recorded; returns whether it
// found one so the caller can count outcomes for the log line.
fn append_outcome(
    conn: &Connection,
    session_id: &str,
    text: &mut String,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some((score, comment)) = db::load_session_outcome(conn, session_id)? else {
        return Ok(false);
    };
    text.push_str(&format!("outcome: score {score:+.0}"));
    if !comment.is_empty() {
        text.push_str(&format!(" — {comment}"));
    }
    text.push('\n');
    Ok(true)
}

// One focused LLM pass for a single game, applied through the dream path
// (gist_edge/upsert/expire work; record_* stay denied). Returns lessons written.
fn run_strategy_game<'run>(
    stage: &StrategyStage<'run>,
    name: &str,
    context: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let prompt = prompts::build_strategy_prompt(name, context);
    let output = run_dream_agent(
        Arc::clone(stage.llm),
        "strategy",
        prompt,
        stage.session_id,
        stage.sender,
    )?;
    let tool_calls = parse_tool_calls(&output);
    let source_id = format!("dream:{}", stage.session_id);
    let policy = SynthesisPolicy::empty();
    apply_dream_tool_calls(stage.conn, tool_calls, &source_id, &policy, stage.sender)
}

// The graph context for the dream's global stages (child lens, abstract
// agents, consolidation), bounded by importance so the prompt plus the
// proposals context plus the reply always fit the model window. The full
// dump hit the 131k ceiling on 2026-07-17 (hard 400s; earlier dreams
// silently lost their output space and wrote zero edges). Work-unit dreams
// (graph-paging step 7) replace this; until then, highest-importance first.
const DREAM_GRAPH_BUDGET_TOKENS: usize = 60_000; // replay-tunable.

fn bounded_global_toon(graph: &Graph) -> String {
    let ranked = ranked_live_edges(graph);
    let (edges, node_ids) = pack_dream_edges(&ranked, &graph.nodes);
    let mut nodes = std::collections::HashMap::new();
    for node_id in &node_ids {
        if let Some(node) = graph.nodes.get(node_id) {
            nodes.insert(node_id.clone(), node.clone());
        }
    }
    let dropped = ranked.len().saturating_sub(edges.len());
    let bounded = Graph { nodes, edges };
    let toon_text = toon::encode(&graph_to_json(&bounded), None);
    runlog::log(
        "dream",
        &format!(
            "graph context: {} edges, {} dropped (~{} tokens)",
            bounded.edges.len(),
            dropped,
            super::heavy::estimate_tokens(&toon_text)
        ),
    );
    toon_text
}

fn ranked_live_edges(graph: &Graph) -> Vec<Edge> {
    let mut ranked = Vec::new();
    for edge in &graph.edges {
        if edge.expired_at.is_none() {
            ranked.push(edge.clone());
        }
    }
    ranked.sort_by(|left, right| {
        right
            .importance
            .total_cmp(&left.importance)
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| left.target.cmp(&right.target))
    });
    ranked
}

// Greedy pack under the budget: each edge costs its content plus any
// newly introduced value-bearing endpoint bodies (mute entities are free —
// the serializer omits them anyway).
fn pack_dream_edges(
    ranked: &[Edge],
    nodes: &std::collections::HashMap<String, Node>,
) -> (Vec<Edge>, Vec<String>) {
    let mut used = 500usize;
    let mut selected = Vec::new();
    let mut introduced: Vec<String> = Vec::new();
    for edge in ranked {
        let mut cost = super::heavy::estimate_tokens(&edge.content) + 45;
        let mut new_nodes = Vec::new();
        for endpoint in [&edge.source, &edge.target] {
            if introduced.contains(endpoint) {
                continue;
            }
            let Some(node) = nodes.get(endpoint) else {
                continue;
            };
            if node_carries_value(node) {
                cost += super::heavy::estimate_tokens(&node.body) + 25;
                new_nodes.push(endpoint.clone());
            }
        }
        if used + cost > DREAM_GRAPH_BUDGET_TOKENS {
            continue;
        }
        used += cost;
        selected.push(edge.clone());
        introduced.extend(new_nodes);
    }
    (selected, introduced)
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
    // Strategy consolidation runs on a game-scoped context of its own, before
    // the general agents — strategy work must not compete with global
    // consolidation for the same attention.
    run_strategy_stage(&llm, &conn, &session_id, sender)?;
    let graph = db::load_graph();
    let global_toon = bounded_global_toon(&graph);
    let domain = run_domain_agents(Arc::clone(&llm), embed, &session_id, &global_toon, sender)?;
    let abstract_outputs =
        run_abstract_agents(Arc::clone(&llm), &session_id, &global_toon, &domain, sender)?;
    let synthesis = {
        let _timer = runlog::time_step("synthesis", "stage");
        synthesis::run_synthesis_stage(
            Arc::clone(&llm),
            &conn,
            |text| send_log(sender, "synthesis", text),
            |text| send_append(sender, "synthesis", text),
        )?
    };
    let all_ctx = consolidation_context(&domain, &abstract_outputs, &synthesis.proposal_text);
    let edges_added = {
        let _timer = runlog::time_step("dream", "consolidation stage");
        let consolidation =
            run_consolidation_agent(llm, &session_id, &global_toon, &all_ctx, sender)?;
        let tool_calls = parse_tool_calls(&consolidation);
        if tool_calls.is_empty() {
            // Two consecutive post-reorder dreams wrote zero edges with no
            // visibility into why — a truncated raw reply is the diagnostic.
            let preview: String = consolidation.chars().take(600).collect();
            runlog::log("dream", &format!("consolidation parsed 0 tool calls; reply head: {preview}"));
        }
        let source_id = format!("dream:{session_id}");
        apply_dream_tool_calls(&conn, tool_calls, &source_id, &synthesis.policy, sender)?
    };
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
    let _timer = runlog::time_step("dream", format!("{phase} LLM"));
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
        let outcomes = apply_preflighted_dream_batch(&mut call_ctx, tool_calls)?;
        for (description, added_edge) in outcomes {
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

fn apply_preflighted_dream_batch<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    tool_calls: Vec<ToolCall>,
) -> Result<Vec<(String, bool)>, Box<dyn Error + Send + Sync>> {
    let preflight = preflight_tool_batch(
        ctx.conn,
        &tool_calls,
        &ctx.apply_state.synthesis_policy.synthesis_nodes,
    )?;
    let mut batch = DreamApplyBatch::new(tool_calls);
    apply_dream_nodes(ctx, &preflight, &mut batch)?;
    apply_dream_rest(ctx, &preflight, &mut batch)?;
    Ok(batch.finish())
}

struct DreamApplyBatch {
    pending: Vec<Option<ToolCall>>,
    outcomes: Vec<Option<(String, bool)>>,
}

impl DreamApplyBatch {
    fn new(tool_calls: Vec<ToolCall>) -> Self {
        let mut pending = Vec::new();
        for call in tool_calls {
            pending.push(Some(call));
        }
        let outcomes = vec![None; pending.len()];
        Self { pending, outcomes }
    }

    fn finish(self) -> Vec<(String, bool)> {
        let mut finished = Vec::new();
        for outcome in self.outcomes {
            finished.push(outcome.unwrap_or_default());
        }
        finished
    }
}

fn apply_dream_nodes<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    preflight: &BatchPreflight,
    batch: &mut DreamApplyBatch,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for idx in 0..batch.pending.len() {
        if preflight.denial(idx).is_some()
            || !matches!(batch.pending[idx], Some(ToolCall::UpsertNode { .. }))
        {
            continue;
        }
        let Some(call) = batch.pending[idx].take() else {
            continue;
        };
        batch.outcomes[idx] = Some(apply_dream_call(ctx, call)?);
    }
    Ok(())
}

fn apply_dream_rest<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    preflight: &BatchPreflight,
    batch: &mut DreamApplyBatch,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for idx in 0..batch.pending.len() {
        if let Some(denial) = preflight.denial(idx) {
            batch.outcomes[idx] = Some((denial.clone(), false));
            continue;
        }
        let Some(call) = batch.pending[idx].take() else {
            continue;
        };
        batch.outcomes[idx] = Some(apply_dream_call(ctx, call)?);
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn apply_dream_call<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    call: ToolCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    if !matches!(&call, ToolCall::UpsertNode { .. })
        && super::agent::tool_call_touches_loop(ctx.conn, &call)?
    {
        return Ok((
            "dream mutation denied: open loops are heavy-owned".to_string(),
            false,
        ));
    }
    match call {
        ToolCall::UpsertEdge {
            source,
            target,
            content,
            stability,
            confidence,
            importance,
            subject,
            date,
            memory_type: _,
            inferred_from,
        } => apply_dream_upsert_edge(
            ctx,
            DreamEdgeCall {
                source,
                target,
                content,
                stability,
                confidence,
                importance,
                subject,
                date,
                inferred_from,
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
        ToolCall::DefineTask { .. } => Ok((
            "define_task denied: define_task is a scribe tool".to_string(),
            false,
        )),
        ToolCall::RecordRule { .. } => Ok((
            "record_rule denied: record_rule is not a dreaming tool".to_string(),
            false,
        )),
        ToolCall::RecordOutcome { .. } => Ok((
            "record_outcome denied: record_outcome is a scribe tool".to_string(),
            false,
        )),
        ToolCall::RecordReaction { .. } => Ok((
            "record_reaction denied: record_reaction is not a dreaming tool".to_string(),
            false,
        )),
        ToolCall::OpenLoop { .. } => Ok((
            "open_loop denied: open_loop is a heavy tool".to_string(),
            false,
        )),
        ToolCall::CloseLoop { .. } => Ok((
            "close_loop denied: close_loop is a heavy tool".to_string(),
            false,
        )),
        ToolCall::InboxAdd { .. } => Ok((
            "inbox_add denied: inbox_add is a chat memory tool".to_string(),
            false,
        )),
        ToolCall::RegisterInterest { .. } => Ok((
            "register_interest denied: register_interest is not a dreaming tool".to_string(),
            false,
        )),
        ToolCall::SetPingPolicy { .. } => Ok((
            "set_ping_policy denied: set_ping_policy is not a dreaming tool".to_string(),
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
    let source = normalize_id(&call.source);
    let target = normalize_id(&call.target);
    let mut edge = gist_edge(&source, &target, &call, &supports);
    if let Some(existing) = db::load_edge(ctx.conn, &source, &target)? {
        edge.provenance = existing.provenance;
        edge.ceiling = existing.ceiling;
        edge.stability = existing.stability;
    } else if supports.is_empty() {
        runlog::log(
            "epistemics",
            &format!(
                "[epistemics] synthesized fallback: {source} → {target} has no resolvable supports"
            ),
        );
    }
    db::write_edge(ctx.conn, &edge, ctx.source_id)?;
    ctx.events.push(Event::updated(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&source, &target),
        EventContext::new(ctx.source_id, ""),
        gist_payload(&supports),
    ));
    Ok((
        format!(
            "gist_edge: {source} → {target} (from {} supports)",
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
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let mut kept = Vec::new();
    for (source, target) in refs {
        match db::load_edge(conn, source, target)? {
            Some(edge) if edge.expired_at.is_none() => kept.push(edge),
            None => runlog::log(
                "dream",
                &format!("gist support dropped (unknown): {source} → {target}"),
            ),
            Some(_) => runlog::log(
                "dream",
                &format!("gist support dropped (expired): {source} → {target}"),
            ),
        }
    }
    Ok(kept)
}

fn gist_edge(source: &str, target: &str, call: &DreamGistCall, supports: &[Edge]) -> Edge {
    let mut confidences = Vec::new();
    for edge in supports {
        confidences.push(edge.confidence);
    }
    let ceiling = synthesized_ceiling(&confidences);
    let provenance = if ceiling.is_some() {
        PROVENANCE_SYNTHESIZED
    } else {
        PROVENANCE_DREAMED
    };
    Edge {
        source: source.to_string(),
        target: target.to_string(),
        content: call.content.clone(),
        stability: call
            .stability
            .clamp(0.0, 1.0)
            .min(stability_seed_max(provenance)),
        confidence: call.confidence.clamp(0.0, 1.0),
        importance: GIST_IMPORTANCE,
        subject: false,
        date: crate::prompts::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: provenance.to_string(),
        ceiling,
        last_decayed_at: String::new(),
    }
}

fn gist_payload(supports: &[Edge]) -> String {
    let mut refs = Vec::new();
    for edge in supports {
        refs.push(format!("{} → {}", edge.source, edge.target));
    }
    serde_json::json!({ "gist_of": refs }).to_string()
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
    stability: f64,
    confidence: f64,
    importance: f64,
    subject: bool,
    date: String,
    inferred_from: Option<Vec<(String, String)>>,
}

struct DreamWeights {
    stability: f64,
    confidence: f64,
}

impl DreamWeights {
    fn from_call(call: &DreamEdgeCall) -> Self {
        Self {
            stability: call.stability,
            confidence: call.confidence,
        }
    }
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

// Weights and provenance are decided by the caller; this only assembles.
fn dream_edge(
    source: &str,
    target: &str,
    edge_call: DreamEdgeCall,
    weights: (f64, f64),
    epistemics: (String, Option<f64>),
) -> Edge {
    Edge {
        source: source.to_string(),
        target: target.to_string(),
        content: edge_call.content,
        stability: weights.0,
        confidence: weights.1,
        importance: edge_call.importance,
        subject: edge_call.subject,
        date: edge_call.date,
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: epistemics.0,
        ceiling: epistemics.1,
        last_decayed_at: String::new(),
    }
}

fn apply_dream_upsert_edge<'scope, 'policy>(
    ctx: &mut DreamCallContext<'scope, 'policy>,
    edge_call: DreamEdgeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let source = normalize_id(&edge_call.source);
    let target = normalize_id(&edge_call.target);
    let existing = db::load_edge(ctx.conn, &source, &target)?;
    let key = (source.clone(), target.clone());
    let inherited = dream_declared_ceiling(ctx.conn, &edge_call.inferred_from)?;
    let requested = DreamWeights::from_call(&edge_call);
    let (stability, confidence, provenance, ceiling) = dream_edge_values(
        ctx.apply_state.synthesis_policy,
        &key,
        existing.as_ref(),
        &requested,
        inherited,
    );
    let call_names = (edge_call.source.clone(), edge_call.target.clone());
    let edge = dream_edge(
        &source,
        &target,
        edge_call,
        (stability, confidence),
        (provenance, ceiling),
    );
    db::write_edge(ctx.conn, &edge, ctx.source_id)?;
    ctx.events.push(Event::upsert(
        Actor::new("system", "consolidation"),
        EdgeRef::new(&edge.source, &edge.target),
        EventContext::new(ctx.source_id, ""),
        existing.is_some(),
    ));
    Ok((
        format!("upsert_edge: {} → {}", call_names.0, call_names.1),
        true,
    ))
}

fn dream_edge_values(
    policy: &SynthesisPolicy,
    key: &(String, String),
    existing: Option<&Edge>,
    requested: &DreamWeights,
    inherited: Option<f64>,
) -> (f64, f64, String, Option<f64>) {
    let mut stability = requested.stability;
    let mut confidence = requested.confidence;
    if let Some(stored) = existing {
        return (
            stored.stability,
            confidence,
            stored.provenance.clone(),
            stored.ceiling,
        );
    }
    if let Some(override_values) = policy.edge_overrides.get(key) {
        stability = override_values.stability;
        confidence = override_values.confidence;
        return (
            stability.min(stability_seed_max(PROVENANCE_SYNTHESIZED)),
            confidence,
            PROVENANCE_SYNTHESIZED.to_string(),
            Some(override_values.ceiling),
        );
    }
    let values = dream_inferred_values(inherited, stability, confidence);
    (
        values.0.min(stability_seed_max(&values.2)),
        values.1,
        values.2,
        values.3,
    )
}

fn dream_declared_ceiling(
    conn: &Connection,
    inferred_from: &Option<Vec<(String, String)>>,
) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    let Some(refs) = inferred_from else {
        return Ok(None);
    };
    let confidences = super::agent::resolve_live_premise_confidences(conn, refs)?;
    let ceiling = inferred_ceiling(&confidences);
    if ceiling.is_none() {
        runlog::log(
            "epistemics",
            "[epistemics] dream inference fallback: no resolvable premises",
        );
    }
    Ok(ceiling)
}

fn dream_inferred_values(
    inherited: Option<f64>,
    stability: f64,
    confidence: f64,
) -> (f64, f64, String, Option<f64>) {
    let Some(ceiling) = inherited else {
        return (stability, confidence, PROVENANCE_DREAMED.to_string(), None);
    };
    (
        stability,
        confidence,
        PROVENANCE_INFERRED.to_string(),
        Some(ceiling),
    )
}

fn apply_dream_upsert_node<'scope, 'policy>(
    ctx: &DreamCallContext<'scope, 'policy>,
    node_call: DreamNodeCall,
) -> Result<(String, bool), Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&node_call.name);
    if let Some(denial) = declared_node_denial(
        ctx.conn,
        &normalized,
        &node_call.node_type,
        &ctx.apply_state.synthesis_policy.synthesis_nodes,
    )? {
        runlog::log("nodes", &format!("denied: {denial}"));
        return Ok((denial, false));
    }
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
    #[test]
    fn dream_graph_context_prefers_importance_and_respects_budget() {
        let mut graph = Graph::new();
        let mut edges = Vec::new();
        for index in 0..3000 {
            edges.push(Edge {
                source: format!("s{index:05}"),
                target: format!("t{index:05}"),
                content: "x".repeat(200),
                stability: 0.3,
                confidence: 0.5,
                importance: if index == 0 { 1.0 } else { 0.1 },
                date: String::new(),
                subject: false,
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
                ceiling: None,
                last_decayed_at: String::new(),
            });
        }
        graph.edges = edges;
        let ranked = super::ranked_live_edges(&graph);
        assert_eq!(ranked[0].source, "s00000");
        let (selected, _nodes) = super::pack_dream_edges(&ranked, &graph.nodes);
        assert!(!selected.is_empty());
        assert!(selected.len() < 3000);
        assert_eq!(selected[0].source, "s00000");
        let toon_a = super::bounded_global_toon(&graph);
        let toon_b = super::bounded_global_toon(&graph);
        assert_eq!(toon_a, toon_b);
        assert!(super::super::heavy::estimate_tokens(&toon_a) <= super::DREAM_GRAPH_BUDGET_TOKENS + 2000);
    }

    use super::{
        DreamApplyState, DreamCallContext, DreamGistCall, DreamNodeCall, Edge, Event, Graph,
        SynthesisPolicy, apply_dream_gist_edge, apply_dream_upsert_node, dream_edge_values,
    };
    use crate::memory::agent::{self, ToolCall};
    use crate::memory::db;
    use crate::memory::ledger::{Actor, EventContext};
    use std::collections::{HashMap, HashSet};
    use turso::{Builder, Connection};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";

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
            let _ = conn.execute(NODES_DDL, ()).await;
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
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
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

    fn run_gist(conn: &Connection, call: DreamGistCall) -> ((String, bool), Vec<Event>) {
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

    // Two 0.3-confidence supports, proposed at 0.6 and capped at mean × 0.9.
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
        assert!((written.confidence - 0.27).abs() < 1e-9);
        assert_eq!(written.provenance, "synthesized");
        assert_eq!(written.ceiling, Some(0.27));
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
    fn gist_confidence_is_clamped_to_the_inherited_ceiling() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "game4-hub", "move-qf3", "episodic");
        let (_, _) = run_gist(
            &conn,
            gist_call(
                &[("game3-hub", "move-qh5"), ("game4-hub", "move-qf3")],
                0.95,
            ),
        );
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert!((written.confidence - 0.27).abs() < 1e-9);
    }

    #[test]
    fn a_single_resolvable_support_can_synthesize() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), events) =
            run_gist(&conn, gist_call(&[("game3-hub", "move-qh5")], 0.6));
        assert!(added);
        assert!(description.contains("from 1 supports"));
        assert_eq!(events.len(), 1);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "synthesized");
        assert_eq!(written.ceiling, Some(0.27));
    }

    #[test]
    fn an_unknown_ref_is_skipped_when_another_support_resolves() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("ghost", "nowhere")], 0.6),
        );
        assert!(added);
        assert!(description.contains("from 1 supports"));
    }

    #[test]
    fn a_semantic_support_contributes_to_a_gist() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "old-belief", "chess", "semantic");
        let ((_, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("old-belief", "chess")], 0.6),
        );
        assert!(added);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "synthesized");
    }

    #[test]
    fn gist_ceiling_uses_the_mean_of_mixed_supporters() {
        let conn = harness();
        insert(&conn, "episode", "fact", "episodic");
        insert(&conn, "belief", "fact", "semantic");
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9 WHERE source = 'belief'",
            (),
        ));
        let _ = run_gist(
            &conn,
            gist_call(&[("episode", "fact"), ("belief", "fact")], 0.9),
        );
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.ceiling, Some(0.54));
        assert_eq!(written.confidence, 0.54);
    }

    #[test]
    fn zero_resolvable_gist_supports_fall_back_to_dreamed() {
        let conn = harness();
        let ((_, added), _) = run_gist(&conn, gist_call(&[("ghost", "nowhere")], 0.9));
        assert!(added);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "dreamed");
        assert_eq!(written.ceiling, None);
        assert_eq!(written.confidence, 0.25);
        assert_eq!(written.stability, 0.15);
    }

    #[test]
    fn dream_inference_inherits_the_dream_premise_floor() {
        let conn = harness();
        let mut first = edge("dream-a", "fact", "semantic");
        first.provenance = "dreamed".to_string();
        first.confidence = 0.25;
        let mut second = edge("dream-b", "fact", "semantic");
        second.provenance = "dreamed".to_string();
        second.confidence = 0.25;
        assert!(db::write_edge(&conn, &first, "").is_ok());
        assert!(db::write_edge(&conn, &second, "").is_ok());
        let refs = Some(vec![
            ("dream-a".to_string(), "fact".to_string()),
            ("dream-b".to_string(), "fact".to_string()),
        ]);
        let values = dream_edge_values(
            &empty_policy(),
            &("result".to_string(), "fact".to_string()),
            None,
            &super::DreamWeights {
                stability: 0.9,
                confidence: 0.9,
            },
            super::dream_declared_ceiling(&conn, &refs).unwrap_or_default(),
        );
        assert_eq!(values.2, "inferred");
        assert_eq!(values.3, Some(0.25));
        assert_eq!(values.0, 0.3);
    }

    #[test]
    fn dream_creation_clamps_model_stability_to_dream_seed() {
        let values = dream_edge_values(
            &empty_policy(),
            &("dream".to_string(), "fact".to_string()),
            None,
            &super::DreamWeights {
                stability: 0.9,
                confidence: 0.9,
            },
            None,
        );
        assert_eq!(values.0, 0.15);
        assert_eq!(values.2, "dreamed");
    }

    #[test]
    fn synthesis_policy_remains_the_only_non_fact_declaration_exception() {
        let conn = harness();
        let mut policy = empty_policy();
        policy.synthesis_nodes.insert("higher-order".to_string());
        let mut apply_state = DreamApplyState::new(&policy);
        let mut events = Vec::new();
        let ctx = DreamCallContext {
            conn: &conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let call = DreamNodeCall::new(
            "higher-order".to_string(),
            "synthesis".to_string(),
            "a supported synthesis".to_string(),
            String::new(),
        );
        let Ok((description, changed_edge)) = apply_dream_upsert_node(&ctx, call) else {
            unreachable!("synthesis applies")
        };
        assert!(description.contains("(synthesis)"), "{description}");
        assert!(!changed_edge);
        let Ok(Some(stored)) = db::load_node(&conn, "higher-order") else {
            unreachable!("synthesis exists")
        };
        assert_eq!(stored.node_type, "synthesis");
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
            provenance: "user-confirmed",
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

#[cfg(test)]
mod strategy_tests {
    use super::{SynthesisPolicy, assemble_strategy_context, run_strategy_stage};
    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};
    use crate::memory::db;
    use crate::memory::graph::{Edge, Node};
    use std::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use turso::{Builder, Connection};

    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";

    // Replays one canned response per stream() call and counts the calls so a
    // test can assert the stage never reached the LLM.
    struct ScriptedClient {
        responses: Mutex<Vec<String>>,
        calls: AtomicUsize,
    }

    impl ScriptedClient {
        fn new(responses: &[&str]) -> Arc<ScriptedClient> {
            let mut queued: Vec<String> = Vec::new();
            for text in responses.iter().rev() {
                queued.push((*text).to_string());
            }
            Arc::new(ScriptedClient {
                responses: Mutex::new(queued),
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let text = match self.responses.lock() {
                Ok(mut queued) => queued.pop().unwrap_or_default(),
                Err(_) => String::new(),
            };
            let _ = sender.send(Chunk::Content(text));
            let _ = sender.send(Chunk::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }));
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }
    }

    fn harness() -> Connection {
        db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            conn
        })
    }

    fn write_node(conn: &Connection, node_id: &str, node_type: &str, body: &str) {
        let node = Node {
            node_type: node_type.to_string(),
            body: body.to_string(),
            date: String::new(),
        };
        let Ok(()) = db::write_node(conn, node_id, &node) else {
            unreachable!("node written")
        };
    }

    fn write_edge(conn: &Connection, edge: &Edge) {
        let Ok(()) = db::write_edge(conn, edge, "") else {
            unreachable!("edge written")
        };
    }

    fn load_hubs(conn: &Connection, game_id: &str) -> Vec<String> {
        let Ok(hubs) = db::load_game_hubs(conn, game_id) else {
            unreachable!("hubs load")
        };
        hubs
    }

    fn assemble(conn: &Connection, game_id: &str, node: &Node) -> super::GameStrategyContext {
        let hubs = load_hubs(conn, game_id);
        let Ok(ctx) = assemble_strategy_context(conn, game_id, node, &hubs) else {
            unreachable!("context assembles")
        };
        ctx
    }

    fn must_edge(conn: &Connection, source: &str, target: &str) -> Edge {
        let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
            unreachable!("edge exists")
        };
        edge
    }

    fn index_of(text: &str, needle: &str) -> usize {
        match text.find(needle) {
            Some(index) => index,
            None => unreachable!("substring present"),
        }
    }

    fn link_edge(source: &str, target: &str, memory_type: &str, expired: bool) -> Edge {
        Edge {
            source: source.to_string(),
            target: target.to_string(),
            content: format!("{source} → {target}"),
            stability: 0.3,
            confidence: 1.0,
            importance: 1.0,
            subject: false,
            date: "2020-01-01".to_string(),
            occurred_at: None,
            expired_at: if expired {
                Some("2020-01-02".to_string())
            } else {
                None
            },
            memory_type: memory_type.to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    // A hub→game define_task link plus a two-ply move chain for one session.
    fn play_game(conn: &Connection, session: &str, game_id: &str, moves: &[(&str, &str)]) {
        let hub = format!("task-{session}");
        write_node(conn, &hub, "task", "");
        write_edge(conn, &link_edge(&hub, game_id, "episodic", false));
        for (idx, (actor, text)) in moves.iter().enumerate() {
            let ply = idx + 1;
            let target = format!("{hub}-ply-{ply:03}");
            let mut edge = link_edge(&hub, &target, "episodic", false);
            edge.content = format!("ply {ply} ({actor}): {text}");
            write_edge(conn, &edge);
        }
    }

    fn record_outcome(conn: &Connection, session: &str, score: f64, comment: &str) {
        let payload = serde_json::json!({ "comment": comment }).to_string();
        db::block_on(async {
            let _ = conn
                .execute(
                    "INSERT INTO events (event_type, session_id, weight, payload) VALUES ('outcome', ?1, ?2, ?3)",
                    turso::params![session, score, payload.as_str()],
                )
                .await;
        });
    }

    fn channel() -> (Sender<super::DreamEvent>, Receiver<super::DreamEvent>) {
        std::sync::mpsc::channel()
    }

    #[test]
    fn context_holds_definition_both_chains_in_order_and_the_outcome() {
        let conn = harness();
        write_node(
            &conn,
            "game-chess",
            "game",
            "user plays White, reply move only",
        );
        play_game(
            &conn,
            "s1",
            "game-chess",
            &[("assistant", "e4"), ("user", "e5")],
        );
        play_game(
            &conn,
            "s2",
            "game-chess",
            &[("assistant", "d4"), ("user", "d5")],
        );
        record_outcome(&conn, "s1", -1.0, "resigned");
        let node = game_node("user plays White, reply move only");
        let ctx = assemble(&conn, "game-chess", &node);
        // Definition body, both chains in ply order, and the outcome score.
        assert_all_present(
            &ctx.text,
            &[
                "user plays White, reply move only",
                "ply 1 (assistant): e4",
                "ply 2 (user): e5",
                "ply 1 (assistant): d4",
                "outcome: score -1",
                "resigned",
            ],
        );
        assert!(
            index_of(&ctx.text, "e4") < index_of(&ctx.text, "e5"),
            "ply order"
        );
        assert_eq!((ctx.chains, ctx.outcomes), (2, 1));
    }

    fn assert_all_present(text: &str, needles: &[&str]) {
        for needle in needles {
            assert!(text.contains(needle), "missing: {needle}");
        }
    }

    fn game_node(body: &str) -> Node {
        Node {
            node_type: "game".to_string(),
            body: body.to_string(),
            date: String::new(),
        }
    }

    #[test]
    fn a_game_with_no_chains_skips_the_stage_without_an_llm_call() {
        let conn = harness();
        // A defined game but no hub ever linked/played it.
        write_node(&conn, "game-go", "game", "19x19 board");
        let llm: Arc<ScriptedClient> = ScriptedClient::new(&["{}"]);
        let dyn_llm: Arc<dyn LlmClient> = llm.clone();
        let (sender, receiver) = channel();
        let Ok(total) = run_strategy_stage(&dyn_llm, &conn, "dream-1", &sender) else {
            unreachable!("stage runs")
        };
        drop(receiver);
        assert_eq!(total, 0);
        assert_eq!(
            llm.calls.load(Ordering::SeqCst),
            0,
            "no game reached the LLM"
        );
    }

    #[test]
    fn hub_discovery_excludes_expired_links_and_other_games() {
        let conn = harness();
        write_node(&conn, "game-chess", "game", "");
        write_node(&conn, "game-go", "game", "");
        write_node(&conn, "task-live", "task", "");
        write_node(&conn, "task-dead", "task", "");
        write_node(&conn, "task-other", "task", "");
        write_edge(
            &conn,
            &link_edge("task-live", "game-chess", "episodic", false),
        );
        write_edge(
            &conn,
            &link_edge("task-dead", "game-chess", "episodic", true),
        );
        write_edge(
            &conn,
            &link_edge("task-other", "game-go", "episodic", false),
        );
        assert_eq!(
            load_hubs(&conn, "game-chess"),
            vec!["task-live".to_string()]
        );
    }

    // Drives the strategy response through the dream apply path one call at a
    // time (as the existing gist tests do), so the ledger's position machinery
    // is out of scope — this test is about which calls the dream path honors.
    fn apply_all(conn: &Connection, response: &str) -> Vec<(String, bool)> {
        let policy = SynthesisPolicy::empty();
        let mut events: Vec<super::Event> = Vec::new();
        let mut apply_state = super::DreamApplyState::new(&policy);
        let mut ctx = super::DreamCallContext {
            conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let mut outcomes = Vec::new();
        for call in super::parse_tool_calls(response) {
            let Ok(outcome) = super::apply_dream_call(&mut ctx, call) else {
                unreachable!("apply runs")
            };
            outcomes.push(outcome);
        }
        outcomes
    }

    #[test]
    fn a_strategy_pass_applies_gist_upsert_expire_and_denies_record_move() {
        let conn = harness();
        write_node(&conn, "game-chess", "game", "rules");
        // Two episodic plies for a gist to cite.
        play_game(
            &conn,
            "s1",
            "game-chess",
            &[("assistant", "Qh5"), ("user", "Nc6")],
        );
        // A pre-existing strategy the pass expires on contradicting evidence.
        let mut stale = link_edge("game-chess", "castle-early", "semantic", false);
        stale.content = "castle early".to_string();
        write_edge(&conn, &stale);
        let response = r#"{"tools":[
            {"tool":"gist_edge","args":{"source":"game-chess","target":"queen-sorties","content":"early queen sorties get punished","confidence":0.6,"stability":0.4,"supported_by":["task-s1 → task-s1-ply-001","task-s1 → task-s1-ply-002"]}},
            {"tool":"upsert_edge","args":{"source":"game-chess","target":"develop-knights","content":"develop knights before bishops","stability":0.4,"confidence":0.5,"importance":0.5,"subject":false,"date":"2026-07-11"}},
            {"tool":"expire_edge","args":{"source":"game-chess","target":"castle-early","rewritten_content":"castling early is not always best"}},
            {"tool":"record_move","args":{"move":"e4","actor":"assistant"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        // gist + upsert report an added edge; expire and the denied move do not.
        let mut added: Vec<bool> = Vec::new();
        for outcome in &outcomes {
            added.push(outcome.1);
        }
        assert_eq!(added, vec![true, true, false, false]);
        assert!(outcomes[3].0.contains("denied"), "record_move denied");
        assert_dream_edges_written(&conn);
    }

    #[test]
    fn dream_path_denies_open_and_close_loop() {
        let conn = harness();
        let response = r#"{"tools":[
            {"tool":"open_loop","args":{"intention":"ask later","anchor":"topic","due":""}},
            {"tool":"close_loop","args":{"loop":"ask-later","resolution":"done","note":"asked"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        assert_eq!(outcomes.len(), 2);
        assert!(!outcomes[0].1 && outcomes[0].0.contains("open_loop denied"));
        assert!(!outcomes[1].1 && outcomes[1].0.contains("close_loop denied"));
    }

    #[test]
    fn dream_generic_mutations_cannot_rewrite_or_remove_loops() {
        let conn = harness();
        write_node(&conn, "loop-reminder", "loop", "ask about the service");
        write_node(&conn, "topic", "entity", "");
        let mut anchor = link_edge("loop-reminder", "topic", "semantic", false);
        anchor.content = "open loop: ask about the service".to_string();
        write_edge(&conn, &anchor);
        let response = r#"{"tools":[
            {"tool":"upsert_node","args":{"name":"loop-reminder","type":"fact","body":"rewritten","date":""}},
            {"tool":"upsert_edge","args":{"source":"loop-reminder","target":"topic","content":"rewritten","stability":0.2,"confidence":0.2,"importance":0.2,"subject":false,"date":"2026-07-12"}},
            {"tool":"expire_edge","args":{"source":"loop-reminder","target":"topic","rewritten_content":"closed"}},
            {"tool":"delete_node","args":{"name":"topic"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        assert_eq!(outcomes.len(), 4);
        assert!(!outcomes[0].1 && outcomes[0].0.contains("belongs to a loop node"));
        for (description, changed) in &outcomes[1..] {
            assert!(!changed && description.contains("open loops are heavy-owned"));
        }
        let stored = must_edge(&conn, "loop-reminder", "topic");
        assert_eq!(stored.content, "open loop: ask about the service");
        assert!(stored.expired_at.is_none());
        let Ok(Some(node)) = db::load_node(&conn, "loop-reminder") else {
            unreachable!("loop node remains")
        };
        assert_eq!(node.node_type, "loop");
    }

    fn assert_dream_edges_written(conn: &Connection) {
        assert_eq!(
            must_edge(conn, "game-chess", "queen-sorties").memory_type,
            "semantic"
        );
        assert_eq!(
            must_edge(conn, "game-chess", "develop-knights").content,
            "develop knights before bishops"
        );
        assert!(
            must_edge(conn, "game-chess", "castle-early")
                .expired_at
                .is_some(),
            "stale strategy expired"
        );
        // The denied record_move created no ply on the game hub.
        let Ok(chain) = db::load_task_chain(conn, "test", 240) else {
            unreachable!("chain loads")
        };
        assert!(chain.is_empty(), "record_move denied on the dream path");
    }
}
