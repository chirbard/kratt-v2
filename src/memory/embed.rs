use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use serde_json::json;

use crate::api::EmbeddingClient;
use crate::runlog;
use crate::timeutil;

use super::MemoryEvent;
use super::db;
use super::ledger;
use super::routing;

pub const EMBED_BATCH: usize = 50;
pub const TOP_K: usize = 60;
pub const SMALL_GRAPH_THRESHOLD: usize = 100;
pub const MAX_POOL: usize = 150;
pub const EXPLORATION_CANDIDATES: usize = 200;
pub const NOVELTY_WEIGHT: f64 = 0.3;
const W_SIM: f64 = 0.65; // replay-tunable.
const W_CONF: f64 = 0.2; // replay-tunable.
const W_RECENCY: f64 = 0.15; // replay-tunable.
// Novelty is an exploration signal, not a recall signal: penalizing
// often-seen edges suppresses exactly the beliefs a follow-up question
// needs. Keep at 0 for chat; exploration ranking has its own novelty.
const W_NOVELTY: f64 = 0.0; // replay-tunable.

#[derive(Clone)]
pub struct RoutedSemanticSelection {
    pub edges: Vec<SemanticEdge>,
    pub routing: routing::Routing,
}

#[derive(Clone)]
pub struct SemanticEdge {
    pub source: String,
    pub target: String,
    pub content: String,
    pub confidence: f64,
    pub stability: f64,
    pub importance: f64,
    pub subject: bool,
    pub date: String,
    pub occurred_at: Option<String>,
    pub expired_at: Option<String>,
    pub embedding: Option<Vec<f64>>,
}

pub fn embedding_string(source: &str, content: &str, target: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return format!("{source} {target}");
    }
    format!("{source} {trimmed} {target}")
}

pub fn embed_text(
    embed: &dyn EmbeddingClient,
    text: &str,
) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let texts = vec![text.to_string()];
    let vectors = embed.embed(&texts)?;
    match vectors.into_iter().next() {
        Some(vector) => Ok(vector),
        None => Err("missing embedding".into()),
    }
}

pub fn run_embed_pass(
    embed: Arc<dyn EmbeddingClient>,
    sender: &std::sync::mpsc::Sender<MemoryEvent>,
) {
    let _write_guard = db::acquire_write_lock();
    let pending = db::load_unembedded_edges();
    let mut start = 0usize;
    let mut embedded = 0usize;
    while start < pending.len() {
        let end = (start + EMBED_BATCH).min(pending.len());
        match embed_batch(&*embed, &pending[start..end]) {
            Ok(count) => embedded += count,
            Err(err) => send_log(sender, "embed", format!("batch failed — {err}")),
        }
        start = end;
    }
    send_log(sender, "embed", format!("{embedded} edges embedded"));
}

pub fn select_semantic_edges(embed: &dyn EmbeddingClient, query: &str) -> Vec<SemanticEdge> {
    if db::edge_count() < SMALL_GRAPH_THRESHOLD {
        return fallback_edges();
    }
    let Some(vector) = query_embedding(embed, query) else {
        return fallback_edges();
    };
    let Some(anchors) = anchor_edges_for_vector(&vector) else {
        return fallback_edges();
    };
    let all_edges = all_semantic_edges();
    expand_hops(anchors, all_edges)
}

pub fn select_semantic_edges_routed(
    embed: &dyn EmbeddingClient,
    query: &str,
) -> RoutedSemanticSelection {
    let query_embedding = match query_embedding(embed, query) {
        Some(vector) => vector,
        None => {
            return RoutedSemanticSelection {
                edges: fallback_edges(),
                routing: routing::Routing {
                    routing_vector: Vec::new(),
                    expert: None,
                    expert_score: 0.0,
                    expert_with_direction: false,
                },
            };
        }
    };
    let routing = routing::route(&query_embedding);
    if db::edge_count() < SMALL_GRAPH_THRESHOLD {
        return RoutedSemanticSelection {
            edges: fallback_edges(),
            routing,
        };
    }
    let Some(anchors) = anchor_edges_for_vector(&routing.routing_vector) else {
        return RoutedSemanticSelection {
            edges: fallback_edges(),
            routing,
        };
    };
    let all_edges = all_semantic_edges();
    let pool = expand_hops_pool(anchors, all_edges);
    let user_views = db::load_edge_actor_views("user", "local");
    let edges = rank_pool_for_chat(pool, &routing.routing_vector, &user_views);
    RoutedSemanticSelection { edges, routing }
}

pub fn select_exploration_edges(
    embed: &dyn EmbeddingClient,
    anchor: &str,
    agent_name: &str,
) -> Vec<SemanticEdge> {
    if db::edge_count() < SMALL_GRAPH_THRESHOLD {
        return fallback_edges();
    }
    let vector = match embed_text(embed, anchor) {
        Ok(vector) => vector,
        Err(_) => return fallback_edges(),
    };
    let rows = match db::query_edges_by_vector(&vector, EXPLORATION_CANDIDATES) {
        Ok(rows) => rows,
        Err(_) => return select_semantic_edges(embed, anchor),
    };
    if rows.is_empty() {
        return select_semantic_edges(embed, anchor);
    }
    let anchors = choose_exploration_anchors(rows_to_semantic_edges(rows), agent_name);
    let all_edges = all_semantic_edges();
    expand_hops(anchors, all_edges)
}

pub fn edges_to_toon(edges: &[SemanticEdge]) -> String {
    let mut edges_json = Vec::new();
    for edge in edges {
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
    }
    toon::encode(&json!({"nodes": [], "edges": edges_json}), None)
}

fn embed_batch(
    embed: &dyn EmbeddingClient,
    rows: &[(String, String, String)],
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let texts = batch_texts(rows);
    let vectors = embed.embed(&texts)?;
    save_batch_embeddings(rows, &vectors)
}

fn batch_texts(rows: &[(String, String, String)]) -> Vec<String> {
    let mut texts = Vec::new();
    for (source, target, content) in rows {
        texts.push(embedding_string(source, content, target));
    }
    texts
}

fn save_batch_embeddings(
    rows: &[(String, String, String)],
    vectors: &[Vec<f64>],
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    if rows.len() != vectors.len() {
        return Err("embedding count mismatch".into());
    }
    let mut row_index = 0usize;
    while row_index < rows.len() {
        let (source, target, _) = &rows[row_index];
        db::save_edge_embedding(source, target, &vectors[row_index])?;
        row_index += 1;
    }
    Ok(rows.len())
}

fn all_semantic_edges() -> Vec<SemanticEdge> {
    rows_to_semantic_edges(db::load_all_edges_semantic())
}

fn fallback_edges() -> Vec<SemanticEdge> {
    rank_and_cap(all_semantic_edges())
}

fn choose_exploration_anchors(
    candidates: Vec<SemanticEdge>,
    agent_name: &str,
) -> Vec<SemanticEdge> {
    if candidates.len() <= TOP_K {
        return candidates;
    }
    let views = db::load_edge_actor_views("agent", agent_name);
    rank_by_exploration(candidates, &views)
}

fn query_embedding(embed: &dyn EmbeddingClient, query: &str) -> Option<Vec<f64>> {
    embed_text(embed, query).ok()
}

fn anchor_edges_for_vector(vector: &[f64]) -> Option<Vec<SemanticEdge>> {
    let rows = match db::query_edges_by_vector(vector, TOP_K) {
        Ok(rows) => rows,
        Err(_) => return None,
    };
    if rows.is_empty() {
        return None;
    }
    Some(rows_to_semantic_edges(rows))
}

// Blend per-edge novelty with endpoint exposure so an unseen edge on an
// already-exhausted hub does not score maximum novelty.
fn novelty_score(
    edge: &SemanticEdge,
    views: &HashMap<(String, String), f64>,
    node_totals: &HashMap<String, f64>,
) -> f64 {
    let key = (edge.source.clone(), edge.target.clone());
    let edge_views = *views.get(&key).unwrap_or(&0.0);
    let source_views = *node_totals.get(&edge.source).unwrap_or(&0.0);
    let target_views = *node_totals.get(&edge.target).unwrap_or(&0.0);
    let endpoint_views = (source_views + target_views) / 2.0;
    0.5 / (1.0 + edge_views) + 0.5 / (1.0 + endpoint_views)
}

fn node_view_totals(views: &HashMap<(String, String), f64>) -> HashMap<String, f64> {
    let mut totals: HashMap<String, f64> = HashMap::new();
    for ((source, target), count) in views {
        *totals.entry(source.clone()).or_insert(0.0) += *count;
        *totals.entry(target.clone()).or_insert(0.0) += *count;
    }
    totals
}

fn rank_by_exploration(
    candidates: Vec<SemanticEdge>,
    views: &HashMap<(String, String), f64>,
) -> Vec<SemanticEdge> {
    let node_totals = node_view_totals(views);
    let mut scored = Vec::new();
    let total = EXPLORATION_CANDIDATES as f64;
    let mut rank = 0usize;
    while rank < candidates.len() {
        let edge = candidates[rank].clone();
        let semantic_pos = EXPLORATION_CANDIDATES.saturating_sub(rank) as f64 / total;
        let novelty = novelty_score(&edge, views, &node_totals);
        let score = (1.0 - NOVELTY_WEIGHT) * semantic_pos + NOVELTY_WEIGHT * novelty;
        scored.push(ScoredEdge { score, edge });
        rank += 1;
    }
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    top_ranked_edges(scored)
}

fn top_ranked_edges(mut scored: Vec<ScoredEdge>) -> Vec<SemanticEdge> {
    if scored.len() > TOP_K {
        scored.truncate(TOP_K);
    }
    let mut edges = Vec::new();
    for item in scored {
        edges.push(item.edge);
    }
    edges
}

fn expand_hops(anchors: Vec<SemanticEdge>, all_edges: Vec<SemanticEdge>) -> Vec<SemanticEdge> {
    rank_and_cap(expand_hops_pool(anchors, all_edges))
}

fn expand_hops_pool(anchors: Vec<SemanticEdge>, all_edges: Vec<SemanticEdge>) -> Vec<SemanticEdge> {
    let index = build_node_index(&all_edges);
    let anchor_nodes = collect_anchor_nodes(&anchors);
    let mut pool = Vec::new();
    let mut seen = HashSet::new();
    for edge in anchors {
        insert_pool_edge(&mut seen, &mut pool, edge);
    }
    let mut hop1_nodes = hop_expand(&anchor_nodes, &index, &all_edges, &mut seen, &mut pool);
    for node in &anchor_nodes {
        hop1_nodes.remove(node);
    }
    let _ = hop_expand(&hop1_nodes, &index, &all_edges, &mut seen, &mut pool);
    pool
}

fn collect_anchor_nodes(edges: &[SemanticEdge]) -> HashSet<String> {
    let mut nodes = HashSet::new();
    for edge in edges {
        nodes.insert(edge.source.clone());
        nodes.insert(edge.target.clone());
    }
    nodes
}

fn build_node_index(all_edges: &[SemanticEdge]) -> HashMap<String, Vec<usize>> {
    let mut index = HashMap::new();
    let mut edge_index = 0usize;
    while edge_index < all_edges.len() {
        let edge = &all_edges[edge_index];
        push_node_index(&mut index, &edge.source, edge_index);
        push_node_index(&mut index, &edge.target, edge_index);
        edge_index += 1;
    }
    index
}

fn push_node_index(index: &mut HashMap<String, Vec<usize>>, node: &str, edge_index: usize) {
    if let Some(edge_indices) = index.get_mut(node) {
        edge_indices.push(edge_index);
        return;
    }
    index.insert(node.to_string(), vec![edge_index]);
}

fn hop_expand(
    nodes: &HashSet<String>,
    index: &HashMap<String, Vec<usize>>,
    all_edges: &[SemanticEdge],
    seen: &mut HashSet<(String, String)>,
    pool: &mut Vec<SemanticEdge>,
) -> HashSet<String> {
    let mut new_nodes = HashSet::new();
    for node in nodes {
        let Some(edge_indices) = index.get(node) else {
            continue;
        };
        for edge_index in edge_indices {
            let edge = &all_edges[*edge_index];
            if !mark_seen(seen, edge) {
                continue;
            }
            pool.push(edge.clone());
            new_nodes.insert(edge.source.clone());
            new_nodes.insert(edge.target.clone());
        }
    }
    new_nodes
}

fn mark_seen(seen: &mut HashSet<(String, String)>, edge: &SemanticEdge) -> bool {
    seen.insert((edge.source.clone(), edge.target.clone()))
}

fn insert_pool_edge(
    seen: &mut HashSet<(String, String)>,
    pool: &mut Vec<SemanticEdge>,
    edge: SemanticEdge,
) {
    if !mark_seen(seen, &edge) {
        return;
    }
    pool.push(edge);
}

fn rank_and_cap(mut pool: Vec<SemanticEdge>) -> Vec<SemanticEdge> {
    pool.sort_by(|left, right| right.importance.total_cmp(&left.importance));
    if pool.len() > MAX_POOL {
        pool.truncate(MAX_POOL);
    }
    pool
}

fn rank_pool_for_chat(
    pool: Vec<SemanticEdge>,
    routing_vector: &[f64],
    user_views: &HashMap<(String, String), f64>,
) -> Vec<SemanticEdge> {
    let mut scored = Vec::new();
    let mut index = 0usize;
    while index < pool.len() {
        let edge = pool[index].clone();
        let has_embedding = edge.embedding.is_some();
        let score = rank_score(&edge, routing_vector, user_views);
        scored.push(ChatScoredEdge {
            has_embedding,
            score,
            original_index: index,
            edge,
        });
        index += 1;
    }
    scored.sort_by(chat_rank_cmp);
    if scored.len() > MAX_POOL {
        scored.truncate(MAX_POOL);
    }
    let mut edges = Vec::new();
    for row in scored {
        edges.push(row.edge);
    }
    edges
}

fn rank_score(
    edge: &SemanticEdge,
    routing_vector: &[f64],
    user_views: &HashMap<(String, String), f64>,
) -> f64 {
    let sim = similarity_term(routing_vector, edge.embedding.as_deref());
    let recency = recency_term(&edge.date);
    let novelty = novelty_term(edge, user_views);
    W_SIM * sim + W_CONF * edge.confidence + W_RECENCY * recency + W_NOVELTY * novelty
}

fn similarity_term(routing_vector: &[f64], edge_embedding: Option<&[f64]>) -> f64 {
    let Some(embedding) = edge_embedding else {
        return 0.0;
    };
    if embedding.len() != routing_vector.len() {
        return 0.0;
    }
    ledger::cosine_similarity(routing_vector, embedding).unwrap_or(0.0)
}

fn novelty_term(edge: &SemanticEdge, user_views: &HashMap<(String, String), f64>) -> f64 {
    let key = (edge.source.clone(), edge.target.clone());
    let views = *user_views.get(&key).unwrap_or(&0.0);
    1.0 / (1.0 + views)
}

fn recency_term(date: &str) -> f64 {
    let Some(days) = days_since_edge_date(date) else {
        return 0.0;
    };
    1.0 / (1.0 + days)
}

fn days_since_edge_date(date: &str) -> Option<f64> {
    let trimmed = date.trim();
    if trimmed.is_empty() {
        return None;
    }
    let epoch = if trimmed.contains('T') {
        timeutil::timestamp_to_epoch_seconds(trimmed)?
    } else {
        let stamped = format!("{trimmed}T00:00:00");
        timeutil::timestamp_to_epoch_seconds(&stamped)?
    };
    let now = timeutil::epoch_seconds_now() as i64;
    let seconds = now.saturating_sub(epoch);
    if seconds <= 0 {
        return Some(0.0);
    }
    Some(seconds as f64 / 86_400.0)
}

fn chat_rank_cmp(left: &ChatScoredEdge, right: &ChatScoredEdge) -> std::cmp::Ordering {
    if left.has_embedding != right.has_embedding {
        return right.has_embedding.cmp(&left.has_embedding);
    }
    if !left.has_embedding {
        return left.original_index.cmp(&right.original_index);
    }
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.original_index.cmp(&right.original_index))
}

struct ScoredEdge {
    score: f64,
    edge: SemanticEdge,
}

struct ChatScoredEdge {
    has_embedding: bool,
    score: f64,
    original_index: usize,
    edge: SemanticEdge,
}

fn rows_to_semantic_edges(rows: Vec<db::SemanticEdgeRow>) -> Vec<SemanticEdge> {
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
        embedding,
    ) in rows
    {
        edges.push(SemanticEdge {
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
            embedding,
        });
    }
    edges
}

fn send_log(sender: &std::sync::mpsc::Sender<MemoryEvent>, phase: &'static str, text: String) {
    runlog::log(phase, &text);
    let _ = sender.send(MemoryEvent::Log { phase, text });
}
