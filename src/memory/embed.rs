use std::error::Error;
use std::sync::Arc;

use serde_json::json;

use crate::api::EmbeddingClient;

use super::MemoryEvent;
use super::db;

pub const EMBED_BATCH: usize = 50;
pub const TOP_K: usize = 60;
pub const SMALL_GRAPH_THRESHOLD: usize = 100;

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

pub fn run_embed_pass(embed: Arc<dyn EmbeddingClient>, sender: &std::sync::mpsc::Sender<MemoryEvent>) {
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
        return all_semantic_edges();
    }
    let vector = match embed_text(embed, query) {
        Ok(vector) => vector,
        Err(_) => return all_semantic_edges(),
    };
    match db::query_edges_by_vector(&vector, TOP_K) {
        Ok(rows) => {
            if rows.is_empty() {
                return all_semantic_edges();
            }
            rows_to_semantic_edges(rows)
        }
        Err(_) => all_semantic_edges(),
    }
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

fn rows_to_semantic_edges(
    rows: Vec<db::SemanticEdgeRow>,
) -> Vec<SemanticEdge> {
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
