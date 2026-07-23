use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub node_type: String,
    pub body: String,
    pub date: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub content: String,
    pub stability: f64,
    pub confidence: f64,
    pub importance: f64,
    pub date: String,
    pub subject: bool,
    pub occurred_at: Option<String>,
    pub expired_at: Option<String>,
    // "episodic" | "semantic" — episodic edges are lived experience scoped to
    // origin_session; semantic edges are durable beliefs. Anti-contamination
    // across sessions comes from retrieval scoping (embed.rs), not decay.
    pub memory_type: String,
    pub origin_session: String,
    pub provenance: String,
    // NULL in storage means the provenance tier owns the cap. Derived edges
    // snapshot a premise-inherited cap here at creation time.
    #[serde(default)]
    pub ceiling: Option<f64>,
    // Decay clock: the day decay was last charged for this edge. Empty means
    // never charged under the incremental scheme.
    pub last_decayed_at: String,
}

pub const PROVENANCE_USER_CONFIRMED: &str = "user-confirmed";
pub const PROVENANCE_WEB_CONFIRMED: &str = "web-confirmed";
pub const PROVENANCE_WEB_SURFACED: &str = "web-surfaced";
pub const PROVENANCE_DREAMED: &str = "dreamed";
pub const PROVENANCE_INFERRED: &str = "inferred";
pub const PROVENANCE_SYNTHESIZED: &str = "synthesized";
pub const PROVENANCE_LEGACY: &str = "legacy";
pub const INFERENCE_DISCOUNT: f64 = 0.9; // replay-tunable.
pub const CLOSURE_NODE_CAP: usize = 40; // replay-tunable.

pub fn node_type_is_loop(node_type: &str) -> bool {
    node_type == "loop"
}

pub fn edge_is_pinned(edge: &Edge, source_node_type: &str) -> bool {
    node_type_is_loop(source_node_type)
        && edge.memory_type == "semantic"
        && edge.expired_at.is_none()
}

pub fn stability_seed_max(provenance: &str) -> f64 {
    match provenance {
        PROVENANCE_USER_CONFIRMED | PROVENANCE_WEB_CONFIRMED => 0.5, // replay-tunable.
        PROVENANCE_WEB_SURFACED | PROVENANCE_INFERRED | PROVENANCE_SYNTHESIZED => 0.3, // replay-tunable.
        PROVENANCE_DREAMED => 0.15, // replay-tunable.
        _ => 0.3,                   // replay-tunable: legacy and unknown.
    }
}

pub fn clamp_stability_to_seed(stability: f64, provenance: &str) -> f64 {
    stability.min(stability_seed_max(provenance))
}

pub fn provenance_ceiling(tier: &str) -> f64 {
    match tier {
        PROVENANCE_USER_CONFIRMED => 1.0, // replay-tunable.
        PROVENANCE_WEB_CONFIRMED => 0.8,  // replay-tunable.
        PROVENANCE_WEB_SURFACED => 0.5,   // replay-tunable.
        PROVENANCE_DREAMED => 0.25,       // replay-tunable.
        PROVENANCE_INFERRED | PROVENANCE_SYNTHESIZED => 0.25,
        _ => 0.5, // replay-tunable: legacy and unknown.
    }
}

pub fn edge_ceiling(provenance: &str, ceiling: Option<f64>) -> f64 {
    ceiling.unwrap_or_else(|| provenance_ceiling(provenance))
}

pub fn inferred_ceiling(confidences: &[f64]) -> Option<f64> {
    let first = *confidences.first()?;
    let mut minimum = first;
    for confidence in &confidences[1..] {
        minimum = minimum.min(*confidence);
    }
    Some(round_ceiling(
        (minimum * INFERENCE_DISCOUNT).max(provenance_ceiling(PROVENANCE_DREAMED)),
    ))
}

fn round_ceiling(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

pub fn normalize_memory_type(raw: &str) -> String {
    if raw == "episodic" {
        "episodic".to_string()
    } else {
        "semantic".to_string()
    }
}

#[derive(Clone)]
pub struct Graph {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }
}

pub fn normalize_id(input: &str) -> String {
    input.trim().to_lowercase().replace(' ', "-")
}

pub fn node_carries_value(node: &Node) -> bool {
    node.node_type != "entity" || !node.body.trim().is_empty()
}

// Selected edges keep endpoint priority in their existing relevance order.
// The chosen closure is sorted only after the cap is applied so emitted bytes
// stay deterministic without losing the ranking signal at the boundary.
pub fn closure_nodes(
    endpoints: &[(String, String)],
    nodes: &HashMap<String, Node>,
) -> Vec<(String, Node)> {
    closure_nodes_with_limit(endpoints, nodes, Some(CLOSURE_NODE_CAP))
}

pub(crate) fn closure_nodes_uncapped(
    endpoints: &[(String, String)],
    nodes: &HashMap<String, Node>,
) -> Vec<(String, Node)> {
    closure_nodes_with_limit(endpoints, nodes, None)
}

fn closure_nodes_with_limit(
    endpoints: &[(String, String)],
    nodes: &HashMap<String, Node>,
    limit: Option<usize>,
) -> Vec<(String, Node)> {
    let mut state = ClosureState {
        seen: HashSet::new(),
        selected: Vec::new(),
        value_count: 0,
        limit,
    };
    for (source, target) in endpoints {
        collect_closure_node(source, nodes, &mut state);
        collect_closure_node(target, nodes, &mut state);
    }
    if limit.is_some_and(|cap| state.value_count > cap) {
        crate::runlog::log(
            "toon",
            &format!(
                "closure capped: {} of {} nodes",
                CLOSURE_NODE_CAP, state.value_count
            ),
        );
    }
    state.selected.sort_by(|left, right| left.0.cmp(&right.0));
    state.selected
}

struct ClosureState {
    seen: HashSet<String>,
    selected: Vec<(String, Node)>,
    value_count: usize,
    limit: Option<usize>,
}

fn collect_closure_node(node_id: &str, nodes: &HashMap<String, Node>, state: &mut ClosureState) {
    let Some(node) = nodes.get(node_id) else {
        return;
    };
    if !node_carries_value(node) || !state.seen.insert(node_id.to_string()) {
        return;
    }
    state.value_count += 1;
    if state.limit.is_none_or(|cap| state.selected.len() < cap) {
        state.selected.push((node_id.to_string(), node.clone()));
    }
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
