use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
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
}

pub const PROVENANCE_USER_CONFIRMED: &str = "user-confirmed";
pub const PROVENANCE_WEB_SURFACED: &str = "web-surfaced";
pub const PROVENANCE_DREAMED: &str = "dreamed";
pub const PROVENANCE_LEGACY: &str = "legacy";

pub fn provenance_ceiling(tier: &str) -> f64 {
    match tier {
        PROVENANCE_USER_CONFIRMED => 1.0, // replay-tunable.
        PROVENANCE_WEB_SURFACED => 0.5,   // replay-tunable.
        PROVENANCE_DREAMED => 0.25,       // replay-tunable.
        _ => 0.5,                         // replay-tunable: legacy and unknown.
    }
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

pub fn graph_to_json(graph: &Graph) -> serde_json::Value {
    let mut node_names: Vec<&String> = graph.nodes.keys().collect();
    node_names.sort();
    let mut nodes_json = Vec::new();
    for node_name in node_names {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
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
    use super::{
        PROVENANCE_DREAMED, PROVENANCE_LEGACY, PROVENANCE_USER_CONFIRMED, PROVENANCE_WEB_SURFACED,
        normalize_memory_type, provenance_ceiling,
    };

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
    fn provenance_tiers_have_structural_ceilings() {
        assert_eq!(provenance_ceiling(PROVENANCE_USER_CONFIRMED), 1.0);
        assert_eq!(provenance_ceiling(PROVENANCE_WEB_SURFACED), 0.5);
        assert_eq!(provenance_ceiling(PROVENANCE_DREAMED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_LEGACY), 0.5);
        assert_eq!(provenance_ceiling(""), 0.5);
        assert_eq!(provenance_ceiling("future-tier"), 0.5);
    }
}
