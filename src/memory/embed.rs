use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::api::EmbeddingClient;
use crate::runlog;
use crate::timeutil;

use super::MemoryEvent;
use super::db;
use super::graph::{Edge, Node, closure_nodes, closure_nodes_uncapped};
use super::ledger;
use super::routing;
use super::toon_display::{self, Table};

type ClosureFn = fn(&[(String, String)], &HashMap<String, Node>) -> Vec<(String, Node)>;

pub const EMBED_BATCH: usize = 50;
pub const TOP_K: usize = 60;
pub const CHAT_INITIAL_K: usize = 15; // replay-tunable
pub const SMALL_GRAPH_THRESHOLD: usize = 100;
pub const MAX_POOL: usize = 150;
pub const EXPLORATION_CANDIDATES: usize = 200;
pub const NOVELTY_WEIGHT: f64 = 0.3;
const SATIATION_WINDOW_DAYS: usize = 7; // replay-tunable.
const SATIATION_WEIGHT: f64 = 0.5; // replay-tunable.
const INTEREST_WEIGHT: f64 = 0.6; // replay-tunable.
const INTEREST_TARGETS: usize = 5; // replay-tunable.
const W_SIM: f64 = 0.65; // replay-tunable.
const W_CONF: f64 = 0.2; // replay-tunable.
const W_RECENCY: f64 = 0.15; // replay-tunable.
// Any episodic edge reaching rank_score has already survived session
// scoping (scope_episodic), so it is guaranteed same-session — rank it as
// maximally relevant recent experience.
const EPISODIC_RANK_BONUS: f64 = 1.0; // replay-tunable.

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
    pub memory_type: String,
    pub origin_session: String,
    pub provenance: String,
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
    let _timer = runlog::time_step("embed", "pass");
    let embedded = {
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
        match embed_interest_batch(&*embed, &db::load_unembedded_interests()) {
            Ok(count) => embedded += count,
            Err(err) => send_log(sender, "embed", format!("interest batch failed — {err}")),
        }
        embedded
    };
    if embedded > 0 {
        db::sync_push_best_effort("embed");
    }
    send_log(sender, "embed", format!("{embedded} edges embedded"));
}

fn embed_interest_batch(
    embed: &dyn EmbeddingClient,
    rows: &[(String, String)],
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    if rows.is_empty() {
        return Ok(0);
    }
    let mut texts = Vec::new();
    for (_, topic) in rows {
        texts.push(topic.clone());
    }
    let vectors = embed.embed(&texts)?;
    if rows.len() != vectors.len() {
        return Err("interest embedding count mismatch".into());
    }
    let mut index = 0usize;
    while index < rows.len() {
        db::save_interest_embedding(&rows[index].0, &vectors[index])?;
        index += 1;
    }
    Ok(rows.len())
}

pub fn select_semantic_edges(embed: &dyn EmbeddingClient, query: &str) -> Vec<SemanticEdge> {
    if db::edge_count() < SMALL_GRAPH_THRESHOLD {
        return fallback_edges(None);
    }
    let Some(vector) = query_embedding(embed, query) else {
        return fallback_edges(None);
    };
    let Some(anchors) = anchor_edges_for_vector(&vector, None) else {
        return fallback_edges(None);
    };
    let all_edges = all_semantic_edges(None);
    expand_hops(anchors, all_edges)
}

// Episodic edges are visible only within their originating session; across
// sessions they exist for decay and dream-promotion but never enter
// retrieval. session_id is the only anti-contamination guard here — decay
// alone is too slow to separate same-day sessions (see decay.rs).
pub fn select_semantic_edges_routed(
    embed: &dyn EmbeddingClient,
    query: &str,
    session_id: &str,
) -> RoutedSemanticSelection {
    let query_embedding = match query_embedding(embed, query) {
        Some(vector) => vector,
        None => {
            return RoutedSemanticSelection {
                edges: chat_fallback_edges(session_id),
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
            edges: chat_fallback_edges(session_id),
            routing,
        };
    }
    let Some(anchors) = anchor_edges_for_vector(&routing.routing_vector, Some(session_id)) else {
        return RoutedSemanticSelection {
            edges: chat_fallback_edges(session_id),
            routing,
        };
    };
    let all_edges = all_semantic_edges(Some(session_id));
    let pool = expand_hops_pool(anchors, all_edges);
    let edges = rank_pool_for_chat(pool, &routing.routing_vector);
    RoutedSemanticSelection { edges, routing }
}

pub fn select_exploration_edges(
    embed: &dyn EmbeddingClient,
    anchor: &str,
    agent_name: &str,
    session_id: &str,
) -> Vec<SemanticEdge> {
    if db::edge_count() < SMALL_GRAPH_THRESHOLD {
        return fallback_edges(Some(session_id));
    }
    let vector = match embed_text(embed, anchor) {
        Ok(vector) => vector,
        Err(_) => return fallback_edges(Some(session_id)),
    };
    let rows = match db::query_edges_by_vector(&vector, EXPLORATION_CANDIDATES) {
        Ok(rows) => rows,
        Err(_) => return select_semantic_edges(embed, anchor),
    };
    if rows.is_empty() {
        return select_semantic_edges(embed, anchor);
    }
    let candidates = scope_episodic(rows_to_semantic_edges(rows), Some(session_id));
    let anchors = choose_exploration_anchors(candidates, agent_name);
    let all_edges = all_semantic_edges(Some(session_id));
    expand_hops(anchors, all_edges)
}

pub fn edges_to_toon(edges: &[SemanticEdge]) -> String {
    let nodes = endpoint_nodes(edges);
    edges_to_toon_with_nodes(edges, &nodes)
}

pub(crate) fn packed_edges_to_toon(
    edges: &[SemanticEdge],
    nodes: &HashMap<String, Node>,
) -> String {
    edges_to_toon_with_closure(edges, closure_nodes_uncapped, nodes)
}

fn endpoint_nodes(edges: &[SemanticEdge]) -> HashMap<String, Node> {
    let mut node_ids = Vec::new();
    for edge in edges {
        node_ids.push(edge.source.clone());
        node_ids.push(edge.target.clone());
    }
    let Ok(conn) = db::open_connection() else {
        return HashMap::new();
    };
    match db::load_nodes_by_ids(&conn, &node_ids) {
        Ok(nodes) => nodes,
        Err(err) => {
            runlog::log("toon", &format!("node closure load failed: {err}"));
            HashMap::new()
        }
    }
}

fn edges_to_toon_with_nodes(edges: &[SemanticEdge], nodes: &HashMap<String, Node>) -> String {
    edges_to_toon_with_closure(edges, closure_nodes, nodes)
}

fn edges_to_toon_with_closure(
    edges: &[SemanticEdge],
    closure_fn: ClosureFn,
    nodes: &HashMap<String, Node>,
) -> String {
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_by(|left, right| {
        left.source
            .cmp(&right.source)
            .then_with(|| left.target.cmp(&right.target))
    });
    let mut edge_rows = Vec::new();
    let mut endpoints = Vec::new();
    for edge in &sorted_edges {
        endpoints.push((edge.source.clone(), edge.target.clone()));
        edge_rows.push(semantic_edge_row(edge));
    }
    let closure = closure_fn(&endpoints, nodes);
    let mut node_rows = Vec::new();
    for (node_id, node) in closure {
        node_rows.push(vec![
            json!(node_id),
            json!(node.node_type),
            json!(node.body),
            json!(node.date),
        ]);
    }
    encode_memory_tables(edge_rows, node_rows)
}

fn semantic_edge_row(edge: &SemanticEdge) -> Vec<Value> {
    vec![
        json!(edge.source),
        json!(edge.target),
        json!(edge.content),
        json!(edge.occurred_at),
        json!(edge.provenance),
        json!(edge.confidence),
        json!(edge.date),
        json!(edge.importance),
        json!(edge.stability),
        json!(edge.subject),
        json!(edge.expired_at),
    ]
}

fn encode_memory_tables(edge_rows: Vec<Vec<Value>>, node_rows: Vec<Vec<Value>>) -> String {
    const EDGE_COLUMNS: &[&str] = &[
        "source",
        "target",
        "content",
        "occurred_at",
        "provenance",
        "confidence",
        "date",
        "importance",
        "stability",
        "subject",
        "expired_at",
    ];
    const NODE_COLUMNS: &[&str] = &["id", "type", "body", "date"];
    toon_display::encode(&[
        Table {
            name: "edges",
            columns: EDGE_COLUMNS,
            rows: edge_rows,
        },
        Table {
            name: "nodes",
            columns: NODE_COLUMNS,
            rows: node_rows,
        },
    ])
}

pub(crate) fn semantic_edge(edge: &Edge) -> SemanticEdge {
    SemanticEdge {
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
    }
}

pub(crate) fn nearest_edges(
    embed: &dyn EmbeddingClient,
    query: &str,
    session_id: Option<&str>,
    limit: usize,
) -> Vec<SemanticEdge> {
    let Ok(vector) = embed_text(embed, query) else {
        return Vec::new();
    };
    let Ok(rows) = db::query_edges_by_vector(&vector, limit) else {
        return Vec::new();
    };
    scope_episodic(rows_to_semantic_edges(rows), session_id)
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

// session_id = None means "no session context" — the safe default, which
// excludes episodic edges entirely rather than leaking them across an
// unrelated retrieval path (crystal export and any future untagged caller
// inherit this behavior for free).
fn all_semantic_edges(session_id: Option<&str>) -> Vec<SemanticEdge> {
    scope_episodic(
        rows_to_semantic_edges(db::load_all_edges_semantic()),
        session_id,
    )
}

// See select_semantic_edges_routed: episodic edges from another session (or
// with no session context at all) are dropped before ranking ever sees them.
fn scope_episodic(edges: Vec<SemanticEdge>, session_id: Option<&str>) -> Vec<SemanticEdge> {
    let mut scoped = Vec::new();
    for edge in edges {
        if edge.memory_type != "episodic" {
            scoped.push(edge);
            continue;
        }
        if session_id == Some(edge.origin_session.as_str()) {
            scoped.push(edge);
        }
    }
    scoped
}

fn fallback_edges(session_id: Option<&str>) -> Vec<SemanticEdge> {
    rank_and_cap(all_semantic_edges(session_id))
}

fn chat_fallback_edges(session_id: &str) -> Vec<SemanticEdge> {
    let mut edges = fallback_edges(Some(session_id));
    if edges.len() > CHAT_INITIAL_K {
        edges.truncate(CHAT_INITIAL_K);
    }
    edges
}

fn choose_exploration_anchors(
    candidates: Vec<SemanticEdge>,
    agent_name: &str,
) -> Vec<SemanticEdge> {
    if candidates.len() <= TOP_K {
        return candidates;
    }
    let views = db::load_edge_actor_views("agent", agent_name);
    let keys = edge_keys(&candidates);
    let attention = db::load_recent_attention_counts(&keys, SATIATION_WINDOW_DAYS);
    let interests = db::densification_targets(INTEREST_TARGETS);
    rank_by_exploration(candidates, &views, &attention, &interests)
}

fn edge_keys(candidates: &[SemanticEdge]) -> Vec<(String, String)> {
    let mut keys = Vec::new();
    for edge in candidates {
        keys.push((edge.source.clone(), edge.target.clone()));
    }
    keys
}

fn query_embedding(embed: &dyn EmbeddingClient, query: &str) -> Option<Vec<f64>> {
    embed_text(embed, query).ok()
}

fn anchor_edges_for_vector(vector: &[f64], session_id: Option<&str>) -> Option<Vec<SemanticEdge>> {
    let rows = match db::query_edges_by_vector(vector, TOP_K) {
        Ok(rows) => rows,
        Err(_) => return None,
    };
    if rows.is_empty() {
        return None;
    }
    Some(scope_episodic(rows_to_semantic_edges(rows), session_id))
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
    attention: &HashMap<(String, String), usize>,
    interests: &[db::InterestTarget],
) -> Vec<SemanticEdge> {
    let node_totals = node_view_totals(views);
    let mut scored = Vec::new();
    let total = EXPLORATION_CANDIDATES as f64;
    let mut base_top: Option<(f64, (String, String))> = None;
    let mut unpulled_top: Option<(f64, (String, String))> = None;
    let mut rank = 0usize;
    while rank < candidates.len() {
        let edge = candidates[rank].clone();
        let semantic_pos = EXPLORATION_CANDIDATES.saturating_sub(rank) as f64 / total;
        let novelty = novelty_score(&edge, views, &node_totals);
        let base_score = (1.0 - NOVELTY_WEIGHT) * semantic_pos + NOVELTY_WEIGHT * novelty;
        let key = (edge.source.clone(), edge.target.clone());
        update_base_top(&mut base_top, base_score, &key);
        let recent_attention = *attention.get(&key).unwrap_or(&0);
        let satiation = satiation_factor(recent_attention);
        update_base_top(&mut unpulled_top, base_score * satiation, &key);
        let score = (base_score + interest_bonus(&edge, interests)) * satiation;
        scored.push(ScoredEdge { score, edge });
        rank += 1;
    }
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    log_attention_profile(attention);
    log_satiation_shift(&scored, &base_top, attention);
    log_interest_anchor(&scored, interests, &unpulled_top);
    top_ranked_edges(scored)
}

fn interest_bonus(edge: &SemanticEdge, interests: &[db::InterestTarget]) -> f64 {
    let Some(edge_embedding) = edge.embedding.as_deref() else {
        return 0.0;
    };
    let mut bonus: f64 = 0.0;
    for interest in interests {
        let similarity = cosine_similarity(edge_embedding, &interest.embedding).max(0.0);
        let density_pull = 1.0 / (1.0 + interest.density as f64);
        bonus = bonus.max(INTEREST_WEIGHT * similarity * density_pull);
    }
    bonus
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    let mut index = 0usize;
    while index < left.len() {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
        index += 1;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    dot / denominator
}

fn log_interest_anchor(
    scored: &[ScoredEdge],
    interests: &[db::InterestTarget],
    unpulled_top: &Option<(f64, (String, String))>,
) {
    let Some(first) = scored.first() else {
        return;
    };
    if matches!(
        unpulled_top,
        Some((_, key)) if key.0 == first.edge.source && key.1 == first.edge.target
    ) {
        return;
    }
    let mut best: Option<(&db::InterestTarget, f64)> = None;
    for interest in interests {
        let bonus = interest_bonus(&first.edge, std::slice::from_ref(interest));
        if bonus <= 0.0 {
            continue;
        }
        if best.is_none_or(|(_, best_bonus)| bonus > best_bonus) {
            best = Some((interest, bonus));
        }
    }
    if let Some((interest, _)) = best {
        runlog::log(
            "explore",
            &format!(
                "interest: densifying {} (density {})",
                interest.topic, interest.density
            ),
        );
    }
}

// Always-on profile: distinguishes "attention query returned nothing" from
// "pool uniformly saturated" when the shift log stays silent.
fn log_attention_profile(attention: &HashMap<(String, String), usize>) {
    let mut hit = 0usize;
    let mut max = 0usize;
    for count in attention.values() {
        if *count > 0 {
            hit += 1;
        }
        if *count > max {
            max = *count;
        }
    }
    runlog::log(
        "explore",
        &format!("satiation profile: {hit} candidates with attention, max {max}"),
    );
}

fn satiation_factor(recent_attention: usize) -> f64 {
    1.0 / (1.0 + SATIATION_WEIGHT * recent_attention as f64)
}

fn update_base_top(
    base_top: &mut Option<(f64, (String, String))>,
    score: f64,
    key: &(String, String),
) {
    let replace = match base_top {
        Some((top_score, _)) => score > *top_score,
        None => true,
    };
    if replace {
        *base_top = Some((score, key.clone()));
    }
}

fn log_satiation_shift(
    scored: &[ScoredEdge],
    base_top: &Option<(f64, (String, String))>,
    attention: &HashMap<(String, String), usize>,
) {
    let Some(first) = scored.first() else {
        return;
    };
    let Some((_, base_key)) = base_top else {
        return;
    };
    if first.edge.source == base_key.0 && first.edge.target == base_key.1 {
        return;
    }
    let mut penalized = 0usize;
    for count in attention.values() {
        if *count > 0 {
            penalized += 1;
        }
    }
    runlog::log(
        "explore",
        &format!("satiation: {penalized} candidates penalized, top anchor shifted"),
    );
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

fn rank_pool_for_chat(pool: Vec<SemanticEdge>, routing_vector: &[f64]) -> Vec<SemanticEdge> {
    let mut scored = Vec::new();
    let mut index = 0usize;
    while index < pool.len() {
        let edge = pool[index].clone();
        let has_embedding = edge.embedding.is_some();
        let score = rank_score(&edge, routing_vector);
        scored.push(ChatScoredEdge {
            has_embedding,
            score,
            original_index: index,
            edge,
        });
        index += 1;
    }
    scored.sort_by(chat_rank_cmp);
    // Keep the initial chat blob small enough that the model has to recall
    // embedding-distant user facts through retrieve_memory when needed.
    if scored.len() > CHAT_INITIAL_K {
        scored.truncate(CHAT_INITIAL_K);
    }
    let mut edges = Vec::new();
    for row in scored {
        edges.push(row.edge);
    }
    edges
}

fn rank_score(edge: &SemanticEdge, routing_vector: &[f64]) -> f64 {
    let sim = similarity_term(routing_vector, edge.embedding.as_deref());
    let recency = recency_term(&edge.date);
    weighted_rank_score(edge, sim, recency)
}

pub(crate) fn replay_rank_score(
    edge: &SemanticEdge,
    routing_vector: &[f64],
    reference_date: &str,
) -> f64 {
    let sim = similarity_term(routing_vector, edge.embedding.as_deref());
    let recency = recency_term_at(&edge.date, reference_date);
    weighted_rank_score(edge, sim, recency)
}

fn weighted_rank_score(edge: &SemanticEdge, sim: f64, recency: f64) -> f64 {
    let base = W_SIM * sim + W_CONF * edge.confidence + W_RECENCY * recency;
    if edge.memory_type == "episodic" {
        base + EPISODIC_RANK_BONUS
    } else {
        base
    }
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

fn recency_term_at(date: &str, reference_date: &str) -> f64 {
    let Some(edge_epoch) = date_epoch(date) else {
        return 0.0;
    };
    let Some(reference_epoch) = date_epoch(reference_date) else {
        return 0.0;
    };
    let seconds = reference_epoch.saturating_sub(edge_epoch);
    if seconds <= 0 {
        return 1.0;
    }
    1.0 / (1.0 + seconds as f64 / 86_400.0)
}

fn date_epoch(date: &str) -> Option<i64> {
    let trimmed = date.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.contains('T') {
        return timeutil::timestamp_to_epoch_seconds(trimmed);
    }
    timeutil::timestamp_to_epoch_seconds(&format!("{trimmed}T00:00:00"))
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
        memory_type,
        origin_session,
        provenance,
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
            memory_type,
            origin_session,
            provenance,
            embedding,
        });
    }
    edges
}

fn send_log(sender: &std::sync::mpsc::Sender<MemoryEvent>, phase: &'static str, text: String) {
    runlog::log(phase, &text);
    let _ = sender.send(MemoryEvent::Log { phase, text });
}

#[cfg(test)]
#[path = "embed_tests.rs"]
mod tests;
