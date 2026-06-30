use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use serde_json::json;

use crate::api::EmbeddingClient;

use super::MemoryEvent;
use super::db;

pub const EMBED_BATCH: usize = 50;
pub const TOP_K: usize = 60;
pub const SMALL_GRAPH_THRESHOLD: usize = 100;
pub const MAX_POOL: usize = 150;
pub const EXPLORATION_CANDIDATES: usize = 200;
pub const NOVELTY_WEIGHT: f64 = 0.3;

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
    let Some(anchors) = anchor_edges(embed, query) else {
        return fallback_edges();
    };
    let all_edges = all_semantic_edges();
    expand_hops(anchors, all_edges)
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
    let views = db::load_all_agent_views(agent_name);
    rank_by_exploration(candidates, &views)
}

fn anchor_edges(embed: &dyn EmbeddingClient, query: &str) -> Option<Vec<SemanticEdge>> {
    let vector = match embed_text(embed, query) {
        Ok(vector) => vector,
        Err(_) => return None,
    };
    let rows = match db::query_edges_by_vector(&vector, TOP_K) {
        Ok(rows) => rows,
        Err(_) => return None,
    };
    if rows.is_empty() {
        return None;
    }
    Some(rows_to_semantic_edges(rows))
}

fn novelty_score(source: &str, target: &str, views: &HashMap<String, i64>) -> f64 {
    let source_views = *views.get(source).unwrap_or(&0) as f64;
    let target_views = *views.get(target).unwrap_or(&0) as f64;
    1.0 / (1.0 + (source_views + target_views) / 2.0)
}

fn rank_by_exploration(
    candidates: Vec<SemanticEdge>,
    views: &HashMap<String, i64>,
) -> Vec<SemanticEdge> {
    let mut scored = Vec::new();
    let total = EXPLORATION_CANDIDATES as f64;
    let mut rank = 0usize;
    while rank < candidates.len() {
        let edge = candidates[rank].clone();
        let semantic_pos = EXPLORATION_CANDIDATES.saturating_sub(rank) as f64 / total;
        let novelty = novelty_score(&edge.source, &edge.target, views);
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
    rank_and_cap(pool)
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

struct ScoredEdge {
    score: f64,
    edge: SemanticEdge,
}

fn rows_to_semantic_edges(rows: Vec<db::SemanticEdgeRow>) -> Vec<SemanticEdge> {
    let mut edges = Vec::new();
    for (source, target, content, confidence, stability, importance, subject, date) in rows {
        edges.push(SemanticEdge {
            source,
            target,
            content,
            confidence,
            stability,
            importance,
            subject,
            date,
        });
    }
    edges
}

fn send_log(sender: &std::sync::mpsc::Sender<MemoryEvent>, phase: &'static str, text: String) {
    let _ = sender.send(MemoryEvent::Log { phase, text });
}
