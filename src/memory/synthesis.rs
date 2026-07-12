use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use serde_json::json;
use turso::{Connection, params};

use crate::api::LlmClient;
use crate::prompts;

use super::agent::{ToolCall, parse_tool_calls, stream_collect};
use super::db;
use super::graph::normalize_id;

const CLUSTER_MIN_EDGES: usize = 3; // replay-tunable.
const SYNTHESIS_CLUSTERS_PER_PASS: usize = 2; // replay-tunable.
const VIEW_MASS_WINDOW_DAYS: usize = 7; // replay-tunable.
const SYNTHESIS_STABILITY: f64 = 0.4; // replay-tunable.

pub struct SynthesisEdgeOverride {
    pub confidence: f64,
    pub stability: f64,
}

pub struct SynthesisPolicy {
    pub synthesis_nodes: HashSet<String>,
    pub edge_overrides: HashMap<(String, String), SynthesisEdgeOverride>,
}

impl SynthesisPolicy {
    // A policy that overrides nothing — used by passes (the strategy stage) that
    // apply through the dream path without a synthesis brainstorm behind them.
    pub fn empty() -> Self {
        Self {
            synthesis_nodes: HashSet::new(),
            edge_overrides: HashMap::new(),
        }
    }
}

pub struct SynthesisStageResult {
    pub proposal_text: String,
    pub policy: SynthesisPolicy,
}

struct Cluster {
    hub: String,
    edge_indices: Vec<usize>,
    node_ids: Vec<String>,
    view_mass: f64,
}

#[derive(Clone)]
struct ClusterEdge {
    source: String,
    target: String,
    content: String,
    confidence: f64,
    stability: f64,
    importance: f64,
    subject: bool,
    date: String,
    occurred_at: Option<String>,
    expired_at: Option<String>,
}

pub fn run_synthesis_stage(
    llm: Arc<dyn LlmClient>,
    conn: &Connection,
    log_line: impl Fn(String),
    append_text: impl Fn(String),
) -> Result<SynthesisStageResult, Box<dyn Error + Send + Sync>> {
    let edges = load_live_edges();
    let clusters = detect_clusters(conn, &edges)?;
    if clusters.is_empty() {
        log_line("no eligible clusters".to_string());
        return Ok(SynthesisStageResult {
            proposal_text: String::new(),
            policy: SynthesisPolicy {
                synthesis_nodes: HashSet::new(),
                edge_overrides: HashMap::new(),
            },
        });
    }
    log_cluster_ranking(&clusters, &log_line);
    let mut policy = SynthesisPolicy {
        synthesis_nodes: HashSet::new(),
        edge_overrides: HashMap::new(),
    };
    let mut proposal_blocks = Vec::new();
    let mut index = 0usize;
    while index < clusters.len() {
        let cluster = &clusters[index];
        let title = format!("cluster-{}", index + 1);
        let stage_calls =
            run_cluster_pipeline(Arc::clone(&llm), cluster, &edges, &log_line, &append_text)?;
        if stage_calls.is_empty() {
            log_line(format!("{title}: no synthesis survived"));
            index += 1;
            continue;
        }
        apply_policy_overrides(cluster, stage_calls.as_slice(), &mut policy);
        proposal_blocks.push((title, tool_calls_json(&stage_calls)));
        index += 1;
    }
    if proposal_blocks.is_empty() {
        log_line("no synthesis survived".to_string());
    }
    Ok(SynthesisStageResult {
        proposal_text: format_blocks(&proposal_blocks),
        policy,
    })
}

fn load_live_edges() -> Vec<ClusterEdge> {
    let rows = db::load_all_edges_semantic();
    let mut edges = Vec::new();
    for (
        source,
        target,
        content,
        confidence,
        stability,
        importance,
        subject,
        date,
        occurred_at,
        expired_at,
        _memory_type,
        _origin_session,
        _embedding,
    ) in rows
    {
        edges.push(ClusterEdge {
            source,
            target,
            content,
            confidence,
            stability,
            importance,
            subject,
            date,
            occurred_at,
            expired_at,
        });
    }
    edges
}

fn detect_clusters(
    conn: &Connection,
    edges: &[ClusterEdge],
) -> Result<Vec<Cluster>, Box<dyn Error + Send + Sync>> {
    let candidates = structural_candidates(edges);
    let mut clusters = dedupe_clusters(candidates);
    if clusters.is_empty() {
        return Ok(clusters);
    }
    let view_mass = load_recent_view_mass(conn)?;
    let mut index = 0usize;
    while index < clusters.len() {
        clusters[index].view_mass = cluster_view_mass(&clusters[index], edges, &view_mass);
        index += 1;
    }
    clusters.sort_by(cluster_rank_cmp);
    if clusters.len() > SYNTHESIS_CLUSTERS_PER_PASS {
        clusters.truncate(SYNTHESIS_CLUSTERS_PER_PASS);
    }
    Ok(clusters)
}

fn structural_candidates(edges: &[ClusterEdge]) -> Vec<Cluster> {
    let mut by_node: HashMap<String, Vec<usize>> = HashMap::new();
    let mut edge_index = 0usize;
    while edge_index < edges.len() {
        push_node_edge(&mut by_node, &edges[edge_index].source, edge_index);
        push_node_edge(&mut by_node, &edges[edge_index].target, edge_index);
        edge_index += 1;
    }
    let mut hubs: Vec<String> = by_node.keys().cloned().collect();
    hubs.sort();
    let mut candidates = Vec::new();
    let mut hub_index = 0usize;
    while hub_index < hubs.len() {
        let hub = &hubs[hub_index];
        let Some(incident) = by_node.get(hub) else {
            hub_index += 1;
            continue;
        };
        let candidate = build_hub_cluster(hub, incident, edges);
        if let Some(cluster) = candidate {
            candidates.push(cluster);
        }
        hub_index += 1;
    }
    candidates
}

fn push_node_edge(index: &mut HashMap<String, Vec<usize>>, node: &str, edge_idx: usize) {
    if let Some(list) = index.get_mut(node) {
        list.push(edge_idx);
        return;
    }
    index.insert(node.to_string(), vec![edge_idx]);
}

fn build_hub_cluster(hub: &str, incident: &[usize], edges: &[ClusterEdge]) -> Option<Cluster> {
    let (mut cluster_edges, neighbors) = incident_seed(hub, incident, edges);
    if neighbors.len() < 2 {
        return None;
    }
    include_neighbor_links(edges, &neighbors, &mut cluster_edges);
    if cluster_edges.len() < CLUSTER_MIN_EDGES {
        return None;
    }
    let mut edge_indices: Vec<usize> = cluster_edges.into_iter().collect();
    edge_indices.sort_unstable();
    let node_ids = cluster_nodes(&edge_indices, edges);
    if node_ids.len() < 3 {
        return None;
    }
    Some(Cluster {
        hub: hub.to_string(),
        edge_indices,
        node_ids,
        view_mass: 0.0,
    })
}

fn incident_seed(
    hub: &str,
    incident: &[usize],
    edges: &[ClusterEdge],
) -> (HashSet<usize>, HashSet<String>) {
    let mut cluster_edges: HashSet<usize> = HashSet::new();
    let mut neighbors: HashSet<String> = HashSet::new();
    let mut idx = 0usize;
    while idx < incident.len() {
        let edge_idx = incident[idx];
        cluster_edges.insert(edge_idx);
        collect_neighbor_nodes(hub, &edges[edge_idx], &mut neighbors);
        idx += 1;
    }
    (cluster_edges, neighbors)
}

fn collect_neighbor_nodes(hub: &str, edge: &ClusterEdge, neighbors: &mut HashSet<String>) {
    if edge.source != hub {
        neighbors.insert(edge.source.clone());
    }
    if edge.target != hub {
        neighbors.insert(edge.target.clone());
    }
}

fn include_neighbor_links(
    edges: &[ClusterEdge],
    neighbors: &HashSet<String>,
    cluster_edges: &mut HashSet<usize>,
) {
    let mut all_index = 0usize;
    while all_index < edges.len() {
        let edge = &edges[all_index];
        if neighbors.contains(&edge.source) && neighbors.contains(&edge.target) {
            cluster_edges.insert(all_index);
        }
        all_index += 1;
    }
}

fn cluster_nodes(edge_indices: &[usize], edges: &[ClusterEdge]) -> Vec<String> {
    let mut set = HashSet::new();
    let mut idx = 0usize;
    while idx < edge_indices.len() {
        let edge = &edges[edge_indices[idx]];
        set.insert(edge.source.clone());
        set.insert(edge.target.clone());
        idx += 1;
    }
    let mut nodes: Vec<String> = set.into_iter().collect();
    nodes.sort();
    nodes
}

fn dedupe_clusters(mut candidates: Vec<Cluster>) -> Vec<Cluster> {
    candidates.sort_by_key(|cluster| std::cmp::Reverse(cluster.edge_indices.len()));
    let mut kept = Vec::new();
    let mut idx = 0usize;
    while idx < candidates.len() {
        let candidate = &candidates[idx];
        if !overlaps_kept(candidate, &kept) {
            kept.push(Cluster {
                hub: candidate.hub.clone(),
                edge_indices: candidate.edge_indices.clone(),
                node_ids: candidate.node_ids.clone(),
                view_mass: 0.0,
            });
        }
        idx += 1;
    }
    kept
}

fn overlaps_kept(candidate: &Cluster, kept: &[Cluster]) -> bool {
    let mut idx = 0usize;
    while idx < kept.len() {
        if overlap_majority(candidate, &kept[idx]) {
            return true;
        }
        idx += 1;
    }
    false
}

fn overlap_majority(left: &Cluster, right: &Cluster) -> bool {
    let mut overlap = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.edge_indices.len() && j < right.edge_indices.len() {
        let left_idx = left.edge_indices[i];
        let right_idx = right.edge_indices[j];
        if left_idx == right_idx {
            overlap += 1;
            i += 1;
            j += 1;
            continue;
        }
        if left_idx < right_idx {
            i += 1;
        } else {
            j += 1;
        }
    }
    let smaller = left.edge_indices.len().min(right.edge_indices.len());
    overlap * 2 > smaller
}

fn load_recent_view_mass(
    conn: &Connection,
) -> Result<HashMap<(String, String), f64>, Box<dyn Error + Send + Sync>> {
    db::block_on(async {
        let window = format!("-{} days", VIEW_MASS_WINDOW_DAYS);
        let mut rows = conn
            .query(
                "SELECT source, target, SUM(weight) FROM events WHERE event_type IN ('injected', 'retrieved', 'marked') AND datetime(replace(ts, 'T', ' ')) > datetime('now', ?1) GROUP BY source, target",
                params![window.as_str()],
            )
            .await?;
        let mut totals = HashMap::new();
        while let Some(row) = rows.next().await? {
            let source = db::text(row.get_value(0)?);
            let target = db::text(row.get_value(1)?);
            let total = db::real(row.get_value(2)?);
            totals.insert((source, target), total);
        }
        Ok(totals)
    })
}

fn cluster_view_mass(
    cluster: &Cluster,
    edges: &[ClusterEdge],
    masses: &HashMap<(String, String), f64>,
) -> f64 {
    let mut total = 0.0;
    let mut idx = 0usize;
    while idx < cluster.edge_indices.len() {
        let edge = &edges[cluster.edge_indices[idx]];
        let key = (edge.source.clone(), edge.target.clone());
        total += *masses.get(&key).unwrap_or(&0.0);
        idx += 1;
    }
    total
}

fn cluster_rank_cmp(left: &Cluster, right: &Cluster) -> std::cmp::Ordering {
    right
        .view_mass
        .total_cmp(&left.view_mass)
        .then_with(|| right.edge_indices.len().cmp(&left.edge_indices.len()))
        .then_with(|| left.hub.cmp(&right.hub))
}

fn log_cluster_ranking(clusters: &[Cluster], log_line: &impl Fn(String)) {
    log_line(format!("clusters ranked: {}", clusters.len()));
    let mut index = 0usize;
    while index < clusters.len() {
        let cluster = &clusters[index];
        log_line(format!(
            "rank {} hub={} edges={} nodes={} view_mass={:.2}",
            index + 1,
            cluster.hub,
            cluster.edge_indices.len(),
            cluster.node_ids.len(),
            cluster.view_mass
        ));
        index += 1;
    }
}

fn run_cluster_pipeline(
    llm: Arc<dyn LlmClient>,
    cluster: &Cluster,
    edges: &[ClusterEdge],
    log_line: &impl Fn(String),
    append_text: &impl Fn(String),
) -> Result<Vec<ToolCall>, Box<dyn Error + Send + Sync>> {
    let cluster_toon = cluster_toon(cluster, edges);
    let stage_io = StageIo {
        log_line,
        append_text,
    };
    log_line(format!(
        "{}: start hub={} edges={}",
        cluster_label(cluster),
        cluster.hub,
        cluster.edge_indices.len()
    ));
    let text_outputs = run_brainstorm_text_stages(Arc::clone(&llm), &cluster_toon, &stage_io)?;
    let chairman = run_chairman_stage(llm, &cluster_toon, &text_outputs, &stage_io)?;
    let parsed = parse_tool_calls(&chairman);
    if parsed.is_empty() {
        return Ok(Vec::new());
    }
    Ok(normalize_calls_for_cluster(cluster, parsed))
}

struct StageIo<'stage_io> {
    log_line: &'stage_io dyn Fn(String),
    append_text: &'stage_io dyn Fn(String),
}

struct StageOutputs {
    drifter: String,
    advocate: String,
    devil: String,
}

struct StageSpec<'spec> {
    name: &'spec str,
    persona: &'spec str,
}

fn run_brainstorm_text_stages<'stage_io>(
    llm: Arc<dyn LlmClient>,
    cluster_toon: &str,
    stage_io: &StageIo<'stage_io>,
) -> Result<StageOutputs, Box<dyn Error + Send + Sync>> {
    let drifter = stage_text(
        Arc::clone(&llm),
        StageSpec {
            name: "drifter",
            persona: prompts::SYNTHESIS_DRIFTER,
        },
        cluster_toon,
        None,
        stage_io,
    )?;
    let advocate = stage_text(
        Arc::clone(&llm),
        StageSpec {
            name: "advocate",
            persona: prompts::SYNTHESIS_ADVOCATE,
        },
        cluster_toon,
        Some(&drifter),
        stage_io,
    )?;
    let devil = stage_text(
        llm,
        StageSpec {
            name: "devil",
            persona: prompts::SYNTHESIS_DEVIL,
        },
        cluster_toon,
        Some(&advocate),
        stage_io,
    )?;
    Ok(StageOutputs {
        drifter,
        advocate,
        devil,
    })
}

fn run_chairman_stage<'stage_io>(
    llm: Arc<dyn LlmClient>,
    cluster_toon: &str,
    outputs: &StageOutputs,
    stage_io: &StageIo<'stage_io>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let context = format!(
        "## Drifter\n{}\n\n## Advocate\n{}\n\n## Devil\n{}",
        outputs.drifter, outputs.advocate, outputs.devil
    );
    let prompt =
        prompts::build_dream_prompt(prompts::SYNTHESIS_CHAIRMAN, cluster_toon, Some(&context));
    stage_collect(llm, "chairman", prompt, stage_io)
}

fn stage_text<'stage_io, 'spec>(
    llm: Arc<dyn LlmClient>,
    spec: StageSpec<'spec>,
    toon: &str,
    prior: Option<&str>,
    stage_io: &StageIo<'stage_io>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let prompt = stage_prompt(spec.persona, toon, prior);
    stage_collect(llm, spec.name, prompt, stage_io)
}

fn stage_collect<'stage_io>(
    llm: Arc<dyn LlmClient>,
    stage: &str,
    prompt: String,
    stage_io: &StageIo<'stage_io>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let result = stream_collect(llm, prompt, |_| {}, stage_io.append_text);
    let output = match result {
        Ok(text) => text,
        Err(err) => {
            (stage_io.log_line)(format!("{stage}: error: {err}"));
            return Err(err);
        }
    };
    (stage_io.log_line)(format!("{stage}: {}", summarize(&output)));
    Ok(output)
}

fn stage_prompt(persona: &str, toon: &str, prior: Option<&str>) -> String {
    let mut prompt = String::new();
    prompt.push_str(persona);
    prompt.push_str("\n\nToday's date: ");
    prompt.push_str(&crate::timeutil::today_iso());
    prompt.push_str("\n\n## Cluster subgraph\n\n");
    prompt.push_str(toon);
    if let Some(text) = prior {
        prompt.push_str("\n\n## Previous stage output\n\n");
        prompt.push_str(text);
    }
    prompt
}

fn summarize(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "empty".to_string();
    }
    // Char-based truncation: byte slicing panics on multi-byte UTF-8.
    let max_chars = 160usize;
    let mut output = String::new();
    for (index, character) in trimmed.chars().enumerate() {
        if index >= max_chars {
            output.push('…');
            break;
        }
        output.push(character);
    }
    output
}

fn cluster_toon(cluster: &Cluster, edges: &[ClusterEdge]) -> String {
    let mut edges_json = Vec::new();
    let mut idx = 0usize;
    while idx < cluster.edge_indices.len() {
        let edge = &edges[cluster.edge_indices[idx]];
        edges_json.push(json!({
            "source": edge.source,
            "target": edge.target,
            "content": edge.content,
            "confidence": edge.confidence,
            "stability": edge.stability,
            "importance": edge.importance,
            "subject": edge.subject,
            "date": edge.date,
            "occurred_at": edge.occurred_at,
            "expired_at": edge.expired_at,
        }));
        idx += 1;
    }
    toon::encode(&json!({"nodes": [], "edges": edges_json}), None)
}

fn normalize_calls_for_cluster(cluster: &Cluster, calls: Vec<ToolCall>) -> Vec<ToolCall> {
    let confidence = (cluster.edge_indices.len() as f64 * 0.15).min(0.75);
    let mut normalized = Vec::new();
    let mut idx = 0usize;
    while idx < calls.len() {
        normalized.push(normalize_call(&calls[idx], confidence));
        idx += 1;
    }
    normalized
}

fn normalize_call(call: &ToolCall, confidence: f64) -> ToolCall {
    match call {
        ToolCall::UpsertNode { .. } => normalize_upsert_node(call),
        ToolCall::UpsertEdge { .. } => normalize_upsert_edge(call, confidence),
        ToolCall::DeleteEdge { .. } => normalize_delete_edge(call),
        ToolCall::DeleteNode { .. } => normalize_delete_node(call),
        ToolCall::ExpireEdge { .. } => normalize_expire_edge(call),
        ToolCall::PromoteEdge { .. } => normalize_promote_edge(call),
        ToolCall::GistEdge { .. } => normalize_gist_edge(call),
        ToolCall::RecordMove { .. } => normalize_record_move(call),
        ToolCall::DefineTask { name, description } => ToolCall::DefineTask {
            name: name.clone(),
            description: description.clone(),
        },
        ToolCall::RecordRule { rule } => ToolCall::RecordRule { rule: rule.clone() },
        ToolCall::RecordOutcome { result, comment } => ToolCall::RecordOutcome {
            result: result.clone(),
            comment: comment.clone(),
        },
        ToolCall::OpenLoop {
            intention,
            anchor,
            due,
        } => ToolCall::OpenLoop {
            intention: intention.clone(),
            anchor: anchor.clone(),
            due: due.clone(),
        },
        ToolCall::CloseLoop {
            loop_id,
            resolution,
            note,
        } => ToolCall::CloseLoop {
            loop_id: loop_id.clone(),
            resolution: resolution.clone(),
            note: note.clone(),
        },
        ToolCall::InboxAdd { .. } => normalize_inbox_add(call),
    }
}

fn normalize_inbox_add(call: &ToolCall) -> ToolCall {
    let ToolCall::InboxAdd { source, instruction } = call else {
        unreachable!("normalize_inbox_add called with non-inbox call");
    };
    ToolCall::InboxAdd { source: source.clone(), instruction: instruction.clone() }
}

// Synthesis never emits record_move (it is a scribe-only tool), but the match
// must stay exhaustive — pass it through unchanged.
fn normalize_record_move(call: &ToolCall) -> ToolCall {
    let ToolCall::RecordMove { move_text, actor } = call else {
        unreachable!("normalize_record_move called with non-record_move");
    };
    ToolCall::RecordMove {
        move_text: move_text.clone(),
        actor: actor.clone(),
    }
}

fn normalize_gist_edge(call: &ToolCall) -> ToolCall {
    let ToolCall::GistEdge {
        source,
        target,
        content,
        confidence,
        stability,
        supported_by,
    } = call
    else {
        unreachable!("normalize_gist_edge called with non-gist_edge");
    };
    ToolCall::GistEdge {
        source: source.clone(),
        target: target.clone(),
        content: content.clone(),
        confidence: *confidence,
        stability: *stability,
        supported_by: supported_by.clone(),
    }
}

fn normalize_promote_edge(call: &ToolCall) -> ToolCall {
    let ToolCall::PromoteEdge { source, target } = call else {
        unreachable!("normalize_promote_edge called with non-promote_edge");
    };
    ToolCall::PromoteEdge {
        source: source.clone(),
        target: target.clone(),
    }
}

fn normalize_upsert_node(call: &ToolCall) -> ToolCall {
    let ToolCall::UpsertNode {
        name, body, date, ..
    } = call
    else {
        unreachable!("normalize_upsert_node called with non-upsert_node");
    };
    ToolCall::UpsertNode {
        name: name.clone(),
        node_type: "synthesis".to_string(),
        body: body.clone(),
        date: date.clone(),
    }
}

fn normalize_upsert_edge(call: &ToolCall, confidence: f64) -> ToolCall {
    let ToolCall::UpsertEdge {
        source,
        target,
        content,
        importance,
        subject,
        date,
        memory_type,
        ..
    } = call
    else {
        unreachable!("normalize_upsert_edge called with non-upsert_edge");
    };
    ToolCall::UpsertEdge {
        source: source.clone(),
        target: target.clone(),
        content: content.clone(),
        stability: SYNTHESIS_STABILITY,
        confidence,
        importance: *importance,
        subject: *subject,
        date: date.clone(),
        memory_type: memory_type.clone(),
    }
}

fn normalize_delete_edge(call: &ToolCall) -> ToolCall {
    let ToolCall::DeleteEdge { source, target } = call else {
        unreachable!("normalize_delete_edge called with non-delete_edge");
    };
    ToolCall::DeleteEdge {
        source: source.clone(),
        target: target.clone(),
    }
}

fn normalize_delete_node(call: &ToolCall) -> ToolCall {
    let ToolCall::DeleteNode { name } = call else {
        unreachable!("normalize_delete_node called with non-delete_node");
    };
    ToolCall::DeleteNode { name: name.clone() }
}

fn normalize_expire_edge(call: &ToolCall) -> ToolCall {
    let ToolCall::ExpireEdge {
        source,
        target,
        rewritten_content,
    } = call
    else {
        unreachable!("normalize_expire_edge called with non-expire_edge");
    };
    ToolCall::ExpireEdge {
        source: source.clone(),
        target: target.clone(),
        rewritten_content: rewritten_content.clone(),
    }
}

fn apply_policy_overrides(cluster: &Cluster, calls: &[ToolCall], policy: &mut SynthesisPolicy) {
    let confidence = (cluster.edge_indices.len() as f64 * 0.15).min(0.75);
    let mut idx = 0usize;
    while idx < calls.len() {
        match &calls[idx] {
            ToolCall::UpsertNode { name, .. } => {
                policy.synthesis_nodes.insert(normalize_id(name));
            }
            ToolCall::UpsertEdge { source, target, .. } => {
                let key = (normalize_id(source), normalize_id(target));
                policy.edge_overrides.insert(
                    key,
                    SynthesisEdgeOverride {
                        confidence,
                        stability: SYNTHESIS_STABILITY,
                    },
                );
            }
            ToolCall::DeleteNode { .. }
            | ToolCall::DeleteEdge { .. }
            | ToolCall::ExpireEdge { .. }
            | ToolCall::PromoteEdge { .. }
            | ToolCall::GistEdge { .. }
            | ToolCall::RecordMove { .. }
            | ToolCall::DefineTask { .. }
            | ToolCall::RecordRule { .. }
            | ToolCall::RecordOutcome { .. }
            | ToolCall::OpenLoop { .. }
            | ToolCall::CloseLoop { .. }
            | ToolCall::InboxAdd { .. } => {}
        }
        idx += 1;
    }
}

fn gist_refs(supported_by: &[(String, String)]) -> Vec<String> {
    let mut refs = Vec::new();
    for (source, target) in supported_by {
        refs.push(format!("{source} → {target}"));
    }
    refs
}

fn tool_calls_json(calls: &[ToolCall]) -> String {
    let mut values = Vec::new();
    let mut idx = 0usize;
    while idx < calls.len() {
        values.push(tool_call_json(&calls[idx]));
        idx += 1;
    }
    serde_json::json!({ "tools": values }).to_string()
}

fn tool_call_json(call: &ToolCall) -> serde_json::Value {
    match call {
        ToolCall::UpsertNode {
            name,
            node_type,
            body,
            date,
        } => {
            json!({"tool":"upsert_node","args":{"name":name,"type":node_type,"body":body,"date":date}})
        }
        ToolCall::UpsertEdge { .. } => upsert_edge_json(call),
        ToolCall::DeleteNode { name } => json!({"tool":"delete_node","args":{"name":name}}),
        ToolCall::DeleteEdge { source, target } => {
            json!({"tool":"delete_edge","args":{"source":source,"target":target}})
        }
        ToolCall::ExpireEdge {
            source,
            target,
            rewritten_content,
        } => {
            json!({"tool":"expire_edge","args":{"source":source,"target":target,"rewritten_content":rewritten_content}})
        }
        ToolCall::PromoteEdge { source, target } => {
            json!({"tool":"promote_edge","args":{"source":source,"target":target}})
        }
        ToolCall::GistEdge { .. } => gist_edge_json(call),
        ToolCall::RecordMove { .. }
        | ToolCall::DefineTask { .. }
        | ToolCall::RecordRule { .. }
        | ToolCall::RecordOutcome { .. } => task_tool_json(call),
        ToolCall::OpenLoop { .. } | ToolCall::CloseLoop { .. } => loop_tool_json(call),
        ToolCall::InboxAdd { source, instruction } => {
            json!({"tool":"inbox_add","args":{"source":source,"instruction":instruction}})
        }
    }
}

fn task_tool_json(call: &ToolCall) -> serde_json::Value {
    match call {
        ToolCall::RecordMove { move_text, actor } => {
            json!({"tool":"record_move","args":{"move":move_text,"actor":actor}})
        }
        ToolCall::DefineTask { name, description } => {
            json!({"tool":"define_task","args":{"name":name,"description":description}})
        }
        ToolCall::RecordRule { rule } => json!({"tool":"record_rule","args":{"rule":rule}}),
        ToolCall::RecordOutcome { result, comment } => {
            json!({"tool":"record_outcome","args":{"result":result,"comment":comment}})
        }
        _ => unreachable!("task_tool_json called with a non-task tool"),
    }
}

fn loop_tool_json(call: &ToolCall) -> serde_json::Value {
    match call {
        ToolCall::OpenLoop {
            intention,
            anchor,
            due,
        } => {
            json!({"tool":"open_loop","args":{"intention":intention,"anchor":anchor,"due":due}})
        }
        ToolCall::CloseLoop {
            loop_id,
            resolution,
            note,
        } => {
            json!({"tool":"close_loop","args":{"loop":loop_id,"resolution":resolution,"note":note}})
        }
        _ => unreachable!("loop_tool_json called with a non-loop tool"),
    }
}

fn upsert_edge_json(call: &ToolCall) -> serde_json::Value {
    let ToolCall::UpsertEdge {
        source,
        target,
        content,
        stability,
        confidence,
        importance,
        subject,
        date,
        memory_type,
    } = call
    else {
        unreachable!("upsert_edge_json called with non-upsert_edge");
    };
    json!({"tool":"upsert_edge","args":{"source":source,"target":target,"content":content,"stability":stability,"confidence":confidence,"importance":importance,"subject":subject,"date":date,"memory_type":memory_type}})
}

fn gist_edge_json(call: &ToolCall) -> serde_json::Value {
    let ToolCall::GistEdge {
        source,
        target,
        content,
        confidence,
        stability,
        supported_by,
    } = call
    else {
        unreachable!("gist_edge_json called with non-gist_edge");
    };
    json!({"tool":"gist_edge","args":{"source":source,"target":target,"content":content,"confidence":confidence,"stability":stability,"supported_by":gist_refs(supported_by)}})
}

fn format_blocks(blocks: &[(String, String)]) -> String {
    let mut output = String::new();
    let mut idx = 0usize;
    while idx < blocks.len() {
        if idx > 0 {
            output.push_str("\n\n");
        }
        output.push_str("### ");
        output.push_str(&blocks[idx].0);
        output.push('\n');
        output.push_str(&blocks[idx].1);
        idx += 1;
    }
    output
}

fn cluster_label(cluster: &Cluster) -> String {
    format!("cluster:{}:{}", cluster.hub, cluster.edge_indices.len())
}
