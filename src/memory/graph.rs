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

enum ToonSection {
    None,
    Edges,
    Nodes,
}

fn parse_toon_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut chars = line.trim().chars().peekable();
    while chars.peek().is_some() {
        if chars.peek() == Some(&'"') {
            let _ = chars.next();
            let mut field = String::new();
            loop {
                match chars.next() {
                    Some('"') | None => break,
                    Some(character) => field.push(character),
                }
            }
            if chars.peek() == Some(&',') {
                let _ = chars.next();
            }
            fields.push(field);
            continue;
        }
        let mut field = String::new();
        loop {
            match chars.peek() {
                Some(&',') | None => break,
                _ => {
                    if let Some(character) = chars.next() {
                        field.push(character);
                    }
                }
            }
        }
        if chars.peek() == Some(&',') {
            let _ = chars.next();
        }
        fields.push(field);
    }
    fields
}

fn parse_toon_edge(fields: &[String]) -> Option<Edge> {
    if fields.len() < 7 {
        return None;
    }
    // New 8-field format: confidence, content, date, importance, source, stability, subject, target
    // Old 8-field format: confidence, content, importance, reinforced_at, source, stability, subject, target
    // Old 7-field format: confidence, content, importance, source, stability, subject, target
    // Detect new format by whether fields[2] fails to parse as a float (date string or empty).
    let is_new = fields.len() >= 8 && fields[2].parse::<f64>().is_err();
    let (date, importance, col) = if is_new {
        (fields[2].clone(), fields[3].parse().unwrap_or(0.0), 4_usize)
    } else if fields.len() >= 8 {
        (fields[3].clone(), fields[2].parse().unwrap_or(0.0), 4_usize)
    } else {
        (String::new(), fields[2].parse().unwrap_or(0.0), 3_usize)
    };
    Some(Edge {
        confidence: fields[0].parse().unwrap_or(0.0),
        content: fields[1].clone(),
        importance,
        date,
        source: normalize_id(&fields[col]),
        stability: fields[col + 1].parse().unwrap_or(0.0),
        subject: fields[col + 2] == "true",
        target: normalize_id(&fields[col + 3]),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: PROVENANCE_LEGACY.to_string(),
        ceiling: None,
        last_decayed_at: String::new(),
    })
}

fn parse_toon_node(fields: &[String]) -> Option<(String, Node)> {
    if fields.len() < 4 {
        return None;
    }
    Some((
        normalize_id(&fields[2]),
        Node {
            body: fields[0].clone(),
            date: fields[1].clone(),
            node_type: fields[3].clone(),
        },
    ))
}

fn apply_toon_line(line: &str, graph: &mut Graph, section: &mut ToonSection) {
    if line.starts_with("edges[") {
        *section = ToonSection::Edges;
        return;
    }
    if line.starts_with("nodes[") {
        *section = ToonSection::Nodes;
        return;
    }
    match section {
        ToonSection::None => {}
        ToonSection::Edges => {
            if let Some(edge) = parse_toon_edge(&parse_toon_row(line)) {
                graph.edges.push(edge);
            }
        }
        ToonSection::Nodes => {
            if let Some((node_id, node)) = parse_toon_node(&parse_toon_row(line)) {
                graph.nodes.insert(node_id, node);
            }
        }
    }
}

pub fn normalize_id(input: &str) -> String {
    input.trim().to_lowercase().replace(' ', "-")
}

pub fn load_from_toon() -> Graph {
    let toon_text = match std::fs::read_to_string(crate::paths::data_file("kratt-graph.toon")) {
        Ok(contents) => contents,
        Err(_) => return Graph::new(),
    };
    let mut graph = Graph::new();
    let mut section = ToonSection::None;
    for line in toon_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        apply_toon_line(trimmed, &mut graph, &mut section);
    }
    graph
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
