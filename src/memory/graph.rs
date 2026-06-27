use std::collections::HashMap;
use std::error::Error;

const DATA_DIR: &str = "kratt-data";
const TOON_PATH: &str = "kratt-data/kratt-graph.toon";

pub struct Node {
    pub node_type: String,
    pub body: String,
    pub date: String,
}

pub struct Edge {
    pub source: String,
    pub target: String,
    pub content: String,
    pub stability: f64,
    pub confidence: f64,
    pub importance: f64,
    pub reinforced_at: String,
    pub subject: bool,
}

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
    let new_format = fields.len() >= 8;
    let (reinforced_at, source, stability, subject, target) = if new_format {
        (fields[3].clone(), &fields[4], &fields[5], &fields[6], &fields[7])
    } else {
        (String::new(), &fields[3], &fields[4], &fields[5], &fields[6])
    };
    Some(Edge {
        confidence: fields[0].parse().unwrap_or(0.0),
        content: fields[1].clone(),
        importance: fields[2].parse().unwrap_or(0.0),
        reinforced_at,
        source: normalize_id(source),
        stability: stability.parse().unwrap_or(0.0),
        subject: *subject == "true",
        target: normalize_id(target),
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

pub fn load_graph() -> Graph {
    let toon_text = match std::fs::read_to_string(TOON_PATH) {
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

pub fn save_graph(graph: &Graph) -> Result<(), Box<dyn Error + Send + Sync>> {
    std::fs::create_dir_all(DATA_DIR)?;
    std::fs::write(TOON_PATH, toon::encode(&graph_to_json(graph), None))?;
    Ok(())
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
            "importance": round2(edge.importance),
            "reinforced_at": edge.reinforced_at,
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

pub fn insert_edge(graph: &mut Graph, edge: Edge) {
    remove_edges(&mut graph.edges, &edge.source, &edge.target);
    graph.edges.push(edge);
}

pub fn remove_edges(edges: &mut Vec<Edge>, source: &str, target: &str) {
    let mut index = 0;
    while index < edges.len() {
        if edges[index].source == source && edges[index].target == target {
            edges.remove(index);
        } else {
            index += 1;
        }
    }
}

pub fn delete_node(graph: &mut Graph, name: &str) {
    graph.nodes.remove(name);
    let mut index = 0;
    while index < graph.edges.len() {
        if graph.edges[index].source == name || graph.edges[index].target == name {
            graph.edges.remove(index);
        } else {
            index += 1;
        }
    }
}
