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
}

pub const PROVENANCE_USER_CONFIRMED: &str = "user-confirmed";
pub const PROVENANCE_WEB_CONFIRMED: &str = "web-confirmed";
pub const PROVENANCE_WEB_SURFACED: &str = "web-surfaced";
pub const PROVENANCE_DREAMED: &str = "dreamed";
pub const PROVENANCE_INFERRED: &str = "inferred";
pub const PROVENANCE_SYNTHESIZED: &str = "synthesized";
pub const PROVENANCE_LEGACY: &str = "legacy";
pub const INFERENCE_DISCOUNT: f64 = 0.9; // replay-tunable.
pub const SYNTHESIS_DISCOUNT: f64 = 0.9; // replay-tunable.
pub const CLOSURE_NODE_CAP: usize = 40; // replay-tunable.

pub fn stability_seed_max(provenance: &str) -> f64 {
    match provenance {
        PROVENANCE_USER_CONFIRMED | PROVENANCE_WEB_CONFIRMED => 0.5, // replay-tunable.
        PROVENANCE_WEB_SURFACED | PROVENANCE_INFERRED | PROVENANCE_SYNTHESIZED => 0.3, // replay-tunable.
        PROVENANCE_DREAMED => 0.15, // replay-tunable.
        _ => 0.3,                   // replay-tunable: legacy and unknown.
    }
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

pub fn synthesized_ceiling(confidences: &[f64]) -> Option<f64> {
    if confidences.is_empty() {
        return None;
    }
    let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
    Some(round_ceiling(
        (mean * SYNTHESIS_DISCOUNT).max(provenance_ceiling(PROVENANCE_DREAMED)),
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

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
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
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    let mut value_count = 0usize;
    for (source, target) in endpoints {
        collect_closure_node(source, nodes, &mut seen, &mut selected, &mut value_count);
        collect_closure_node(target, nodes, &mut seen, &mut selected, &mut value_count);
    }
    if value_count > CLOSURE_NODE_CAP {
        crate::runlog::log(
            "toon",
            &format!(
                "closure capped: {} of {value_count} nodes",
                CLOSURE_NODE_CAP
            ),
        );
    }
    selected.sort_by(|left, right| left.0.cmp(&right.0));
    selected
}

fn collect_closure_node(
    node_id: &str,
    nodes: &HashMap<String, Node>,
    seen: &mut HashSet<String>,
    selected: &mut Vec<(String, Node)>,
    value_count: &mut usize,
) {
    let Some(node) = nodes.get(node_id) else {
        return;
    };
    if !node_carries_value(node) || !seen.insert(node_id.to_string()) {
        return;
    }
    *value_count += 1;
    if selected.len() < CLOSURE_NODE_CAP {
        selected.push((node_id.to_string(), node.clone()));
    }
}

pub fn graph_to_json(graph: &Graph) -> serde_json::Value {
    let mut node_names: Vec<&String> = graph.nodes.keys().collect();
    node_names.sort();
    let mut nodes_json = Vec::new();
    for node_name in node_names {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
        if !node_carries_value(node) {
            continue;
        }
        nodes_json.push(serde_json::json!({
            "body": node.body,
            "date": node.date,
            "id": node_name,
            "type": node.node_type,
        }));
    }
    let mut edges_json = Vec::new();
    for edge in &graph.edges {
        edges_json.push(serde_json::json!({
            "confidence": round2(edge.confidence),
            "content": edge.content,
            "date": edge.date,
            "expired_at": edge.expired_at,
            "occurred_at": edge.occurred_at,
            "importance": round2(edge.importance),
            "provenance": edge.provenance,
            "ceiling": edge.ceiling,
            "source": edge.source,
            "stability": round2(edge.stability),
            "subject": edge.subject,
            "target": edge.target,
        }));
    }
    serde_json::json!({
        "nodes": nodes_json,
        "edges": edges_json,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        CLOSURE_NODE_CAP, Edge, Graph, Node, PROVENANCE_DREAMED, PROVENANCE_INFERRED,
        PROVENANCE_LEGACY, PROVENANCE_SYNTHESIZED, PROVENANCE_USER_CONFIRMED,
        PROVENANCE_WEB_CONFIRMED, PROVENANCE_WEB_SURFACED, closure_nodes, edge_ceiling,
        graph_to_json, inferred_ceiling, node_carries_value, normalize_memory_type,
        provenance_ceiling, stability_seed_max, synthesized_ceiling,
    };

    fn node(node_type: &str, body: &str) -> Node {
        Node {
            node_type: node_type.to_string(),
            body: body.to_string(),
            date: String::new(),
        }
    }

    #[test]
    fn value_predicate_only_mutes_empty_entities() {
        for node_type in ["fact", "synthesis", "loop", "task"] {
            assert!(node_carries_value(&node(node_type, "")));
        }
        assert!(node_carries_value(&node("entity", "meaningful")));
        assert!(!node_carries_value(&node("entity", "")));
        assert!(!node_carries_value(&node("entity", " \n\t")));
    }

    #[test]
    fn closure_is_value_bearing_sorted_deduplicated_and_bounded() {
        let nodes = HashMap::from([
            ("a".to_string(), node("fact", "A fact")),
            ("b".to_string(), node("entity", "")),
            ("c".to_string(), node("loop", "")),
            ("outside".to_string(), node("fact", "not selected")),
        ]);
        let endpoints = vec![
            ("c".to_string(), "b".to_string()),
            ("a".to_string(), "c".to_string()),
        ];
        let closure = closure_nodes(&endpoints, &nodes);
        let mut node_ids = Vec::new();
        for (node_id, _) in &closure {
            node_ids.push(node_id.as_str());
        }
        assert_eq!(node_ids, vec!["a", "c"]);
    }

    #[test]
    fn closure_cap_prefers_earlier_edge_endpoints() {
        let mut nodes = HashMap::new();
        let mut endpoints = Vec::new();
        let total = CLOSURE_NODE_CAP + 2;
        for index in 0..total {
            let node_id = format!("node-{index:02}");
            nodes.insert(node_id.clone(), node("fact", &node_id));
            endpoints.push((node_id, "mute".to_string()));
        }
        nodes.insert("mute".to_string(), node("entity", ""));
        let closure = closure_nodes(&endpoints, &nodes);
        assert_eq!(closure.len(), CLOSURE_NODE_CAP);
        let mut includes_first = false;
        let mut includes_over_cap = false;
        for (node_id, _) in &closure {
            includes_first |= node_id == "node-00";
            includes_over_cap |= node_id == "node-40";
        }
        assert!(includes_first);
        assert!(!includes_over_cap);
    }

    #[test]
    fn full_graph_filters_only_mute_entities_deterministically() {
        let mut graph = Graph::new();
        graph.nodes.insert("mute".to_string(), node("entity", " "));
        graph
            .nodes
            .insert("entity".to_string(), node("entity", "body"));
        graph.nodes.insert("task".to_string(), node("task", ""));
        graph.edges.push(Edge {
            source: "entity".to_string(),
            target: "task".to_string(),
            content: String::new(),
            stability: 0.0,
            confidence: 0.0,
            importance: 0.0,
            date: String::new(),
            subject: false,
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: PROVENANCE_LEGACY.to_string(),
            ceiling: None,
        });
        let first = graph_to_json(&graph);
        let second = graph_to_json(&graph);
        let Some(emitted) = first["nodes"].as_array() else {
            panic!("nodes array missing");
        };
        assert_eq!(emitted.len(), 2);
        assert_eq!(first, second);
    }

    #[test]
    fn normalize_memory_type_accepts_episodic() {
        assert_eq!(normalize_memory_type("episodic"), "episodic");
    }

    #[test]
    fn normalize_memory_type_accepts_semantic() {
        assert_eq!(normalize_memory_type("semantic"), "semantic");
    }

    #[test]
    fn normalize_memory_type_rejects_garbage_to_semantic() {
        assert_eq!(normalize_memory_type("durable"), "semantic");
        assert_eq!(normalize_memory_type(""), "semantic");
        assert_eq!(normalize_memory_type("Episodic"), "semantic");
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn provenance_tiers_have_structural_ceilings() {
        assert_eq!(provenance_ceiling(PROVENANCE_USER_CONFIRMED), 1.0);
        assert_eq!(provenance_ceiling(PROVENANCE_WEB_CONFIRMED), 0.8);
        assert_eq!(provenance_ceiling(PROVENANCE_WEB_SURFACED), 0.5);
        assert_eq!(provenance_ceiling(PROVENANCE_DREAMED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_INFERRED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_SYNTHESIZED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_LEGACY), 0.5);
        assert_eq!(provenance_ceiling(""), 0.5);
        assert_eq!(provenance_ceiling("future-tier"), 0.5);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn provenance_tiers_have_stability_seed_maxima() {
        assert_eq!(stability_seed_max(PROVENANCE_USER_CONFIRMED), 0.5);
        assert_eq!(stability_seed_max(PROVENANCE_WEB_CONFIRMED), 0.5);
        assert_eq!(stability_seed_max(PROVENANCE_WEB_SURFACED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_INFERRED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_SYNTHESIZED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_DREAMED), 0.15);
        assert_eq!(stability_seed_max(PROVENANCE_LEGACY), 0.3);
        assert_eq!(stability_seed_max("future-tier"), 0.3);
    }

    #[test]
    fn edge_ceiling_prefers_the_stored_cap() {
        assert_eq!(edge_ceiling(PROVENANCE_USER_CONFIRMED, None), 1.0);
        assert_eq!(edge_ceiling(PROVENANCE_SYNTHESIZED, Some(0.72)), 0.72);
    }

    #[test]
    fn derived_ceiling_math_uses_premise_strength_and_floor() {
        assert_eq!(inferred_ceiling(&[1.0, 0.9]), Some(0.81));
        assert_eq!(inferred_ceiling(&[0.25, 0.25]), Some(0.25));
        assert_eq!(synthesized_ceiling(&[1.0, 0.5]), Some(0.68));
        assert_eq!(synthesized_ceiling(&[0.25; 10]), Some(0.25));
        assert_eq!(inferred_ceiling(&[]), None);
        assert_eq!(synthesized_ceiling(&[]), None);
    }
}
