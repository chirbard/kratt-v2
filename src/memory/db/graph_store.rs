use std::collections::{HashMap, HashSet};
use std::error::Error;

use turso::{Connection, Value, params};

use super::core::runtime;
use super::{
    backfill_edge_epistemics, clamp_stability_once, delete_edge_actor_views_async, int,
    migrate_entity_bodies_matching_ids, open_connection, optional_real, optional_text, real,
    round2, text, text_or,
};
use crate::memory::graph;
use crate::memory::graph::{
    Edge, Graph, Node, PROVENANCE_LEGACY, edge_ceiling, load_from_toon, normalize_id,
};

#[cfg(test)]
#[path = "graph_store_tests.rs"]
mod tests;

async fn seed_empty_graph(
    conn: &Connection,
    graph: &Graph,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute("BEGIN", ()).await?;
    for (node_id, node) in &graph.nodes {
        conn.execute(
            "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, ?2, ?3, ?4)",
            params![
                node_id.as_str(),
                node.node_type.as_str(),
                node.body.as_str(),
                node.date.as_str()
            ],
        )
        .await?;
    }
    for edge in &graph.edges {
        conn.execute(
            "INSERT INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                edge.source.as_str(),
                edge.target.as_str(),
                edge.content.as_str(),
                round2(edge.stability),
                clamped_confidence(edge.confidence, &edge.provenance, edge.ceiling),
                round2(edge.importance),
                edge.subject as i64,
                edge.date.as_str(),
                edge.occurred_at.as_deref(),
                edge.expired_at.as_deref(),
                edge.memory_type.as_str(),
                edge.origin_session.as_str(),
                edge.provenance.as_str(),
                edge.ceiling,
            ],
        )
        .await?;
    }
    conn.execute("COMMIT", ()).await?;
    Ok(())
}

pub fn db_init() -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    let count = runtime().block_on(node_count(&conn))?;
    if count == 0 {
        let graph = load_from_toon();
        if !graph.nodes.is_empty() || !graph.edges.is_empty() {
            runtime().block_on(seed_empty_graph(&conn, &graph))?;
        }
    }
    let _ = runtime().block_on(backfill_edge_epistemics(&conn))?;
    let today = crate::timeutil::today_iso();
    let _ = runtime().block_on(clamp_stability_once(&conn, &today))?;
    let _ = runtime().block_on(migrate_entity_bodies_matching_ids(&conn))?;
    Ok(())
}

pub(super) async fn node_count(conn: &Connection) -> Result<i64, Box<dyn Error + Send + Sync>> {
    let mut rows = conn.query("SELECT COUNT(*) FROM nodes", ()).await?;
    if let Some(row) = rows.next().await?
        && let Value::Integer(count) = row.get_value(0)?
    {
        return Ok(count);
    }
    Ok(0)
}

pub fn load_graph() -> Graph {
    let Ok(conn) = open_connection() else {
        return Graph::new();
    };
    runtime().block_on(load_graph_async(&conn))
}

async fn load_graph_async(conn: &Connection) -> Graph {
    let mut graph = Graph::new();
    load_nodes(conn, &mut graph).await;
    load_edges(conn, &mut graph).await;
    graph
}

async fn load_nodes(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn
        .query("SELECT id, node_type, body, date FROM nodes", ())
        .await
    else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        let (Ok(node_id), Ok(node_type), Ok(body), Ok(date)) = (
            row.get_value(0),
            row.get_value(1),
            row.get_value(2),
            row.get_value(3),
        ) else {
            continue;
        };
        graph.nodes.insert(
            text(node_id),
            Node {
                node_type: text(node_type),
                body: text(body),
                date: text(date),
            },
        );
    }
}

async fn load_edges(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn
        .query(
            // Stable order: TOON emission preserves row order, so an unordered scan
            // makes graph serializations engine-dependent — deterministic bytes are
            // a precondition for any prompt prefix reuse across calls.
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling, last_decayed_at FROM edges ORDER BY source, target",
            (),
        )
        .await
    else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        if let Some(edge) = parse_edge_row(&row) {
            graph.edges.push(edge);
        }
    }
}

pub fn begin_transaction(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("BEGIN", ()).await?;
        Ok(())
    })
}

pub fn commit_transaction(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("COMMIT", ()).await?;
        Ok(())
    })
}

pub fn write_node(
    conn: &Connection,
    node_id: &str,
    node: &Node,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, body, date) VALUES (?1, ?2, ?3, ?4)",
            params![
                node_id,
                node.node_type.as_str(),
                node.body.as_str(),
                node.date.as_str()
            ],
        )
        .await?;
        Ok(())
    })
}

pub(super) fn clamped_confidence(confidence: f64, provenance: &str, ceiling: Option<f64>) -> f64 {
    round2(confidence.min(edge_ceiling(provenance, ceiling)))
}

pub fn write_edge(
    conn: &Connection,
    edge: &Edge,
    source_message_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        ensure_entity_endpoints_async(conn, &edge.source, &edge.target).await?;
        conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id, memory_type, origin_session, provenance, ceiling, last_decayed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                edge.source.as_str(),
                edge.target.as_str(),
                edge.content.as_str(),
                round2(edge.stability),
                clamped_confidence(edge.confidence, &edge.provenance, edge.ceiling),
                round2(edge.importance),
                edge.subject as i64,
                edge.date.as_str(),
                edge.occurred_at.as_deref(),
                edge.expired_at.as_deref(),
                source_message_id,
                edge.memory_type.as_str(),
                edge.origin_session.as_str(),
                edge.provenance.as_str(),
                edge.ceiling,
                edge.last_decayed_at.as_str(),
            ],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn ensure_entity_endpoints(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(ensure_entity_endpoints_async(conn, source, target))
}

async fn ensure_entity_endpoints_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    ensure_entity_node(conn, source).await?;
    if target != source {
        ensure_entity_node(conn, target).await?;
    }
    Ok(())
}

async fn ensure_entity_node(
    conn: &Connection,
    node_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let inserted = conn
        .execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, body, date)
             VALUES (?1, 'entity', '', '')",
            params![node_id],
        )
        .await?;
    if inserted > 0 {
        crate::runlog::log("nodes", &format!("implicit entity: {node_id}"));
    }
    Ok(())
}

pub fn erase_node(conn: &Connection, node_id: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("DELETE FROM nodes WHERE id = ?1", params![node_id])
            .await?;
        conn.execute(
            "DELETE FROM edges WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM edge_actor_views WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
        Ok(())
    })
}

pub fn erase_edge(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "DELETE FROM edges WHERE source = ?1 AND target = ?2",
            params![source, target],
        )
        .await?;
        delete_edge_actor_views_async(conn, source, target).await?;
        Ok(())
    })
}

pub fn load_edge(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<Option<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_edge_async(conn, source, target))
}

pub fn load_edges_for_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_edges_for_node_async(conn, node_id))
}

pub fn load_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Option<Node>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT node_type, body, date FROM nodes WHERE id = ?1 LIMIT 1",
                params![node_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            return Ok(Some(Node {
                node_type: text(row.get_value(0)?),
                body: text(row.get_value(1)?),
                date: text(row.get_value(2)?),
            }));
        }
        Ok(None)
    })
}

pub fn load_nodes_by_ids(
    conn: &Connection,
    node_ids: &[String],
) -> Result<HashMap<String, Node>, Box<dyn Error + Send + Sync>> {
    if node_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut values = Vec::new();
    let mut placeholders = Vec::new();
    let mut seen = HashSet::new();
    for node_id in node_ids {
        if seen.insert(node_id) {
            values.push(Value::Text(node_id.clone()));
            placeholders.push(format!("?{}", values.len()));
        }
    }
    let query = format!(
        "SELECT id, node_type, body, date FROM nodes WHERE id IN ({})",
        placeholders.join(", ")
    );
    runtime().block_on(async {
        let mut rows = conn.query(&query, turso::params_from_iter(values)).await?;
        let mut nodes = HashMap::new();
        while let Some(row) = rows.next().await? {
            nodes.insert(
                text(row.get_value(0)?),
                Node {
                    node_type: text(row.get_value(1)?),
                    body: text(row.get_value(2)?),
                    date: text(row.get_value(3)?),
                },
            );
        }
        Ok(nodes)
    })
}

// The game-definition node a session's task hub is linked to (episodic
// hub→game edge), if any. Returns the normalized game id and its node so the
// caller can render the definition body and anchor rules/strategies.

pub fn update_edge_reinforcement(
    conn: &Connection,
    edge: &Edge,
    confidence: f64,
    stability: f64,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let changed = conn
            .execute(
                "UPDATE edges SET confidence = ?1, stability = ?2
                 WHERE source = ?3 AND target = ?4 AND expired_at IS NULL",
                params![
                    clamped_confidence(confidence, &edge.provenance, edge.ceiling),
                    round2(stability.clamp(0.0, 1.0)),
                    edge.source.as_str(),
                    edge.target.as_str()
                ],
            )
            .await?;
        Ok(changed > 0)
    })
}

#[allow(dead_code)]
pub(super) async fn update_edge_confidence_async(
    conn: &Connection,
    source: &str,
    target: &str,
    confidence: f64,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some(edge) = load_edge_async(conn, source, target).await? else {
        return Ok(false);
    };
    let changed = conn
        .execute(
            "UPDATE edges SET confidence = ?1 WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
            params![
                clamped_confidence(confidence, &edge.provenance, edge.ceiling),
                source,
                target
            ],
        )
        .await?;
    Ok(changed > 0)
}

async fn load_edge_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<Option<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling, last_decayed_at FROM edges WHERE source = ?1 AND target = ?2 LIMIT 1",
            params![source, target],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        return Ok(parse_edge_row(&row));
    }
    Ok(None)
}

// Retrieval resets the forgetting clock: an edge injected into a turn was
// just used, so its date moves to today and decay restarts from there.
// Expired edges are left alone — they are history, not live belief.
pub(crate) async fn touch_edge_dates(
    conn: &Connection,
    edges: &[Edge],
    today: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut refreshed = 0usize;
    for edge in edges {
        if edge.expired_at.is_some() {
            continue;
        }
        let changed = conn
            .execute(
                "UPDATE edges SET date = ?1, last_decayed_at = ?1 WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
                params![today, edge.source.as_str(), edge.target.as_str()],
            )
            .await?;
        if changed > 0 {
            refreshed += 1;
        }
    }
    Ok(refreshed)
}

async fn load_edges_for_node_async(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling, last_decayed_at FROM edges WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        if let Some(edge) = parse_edge_row(&row) {
            edges.push(edge);
        }
    }
    Ok(edges)
}

pub fn graph_diff_toon(pre: &Graph, post: &Graph) -> String {
    let mut lines: Vec<String> = Vec::new();
    node_diff_lines(pre, post, &mut lines);
    edge_diff_lines(pre, post, &mut lines);
    lines.sort();
    lines.join("\n")
}

fn node_diff_lines(pre: &Graph, post: &Graph, lines: &mut Vec<String>) {
    for (node_id, node) in &post.nodes {
        if !pre.nodes.contains_key(node_id) && graph::node_carries_value(node) {
            lines.push(format!(
                "+nodes[{node_id} | {} | {} | {}]",
                node.node_type, node.body, node.date
            ));
        }
    }
    for node_id in pre.nodes.keys() {
        if !post.nodes.contains_key(node_id) {
            lines.push(format!("-nodes[{node_id}]"));
        }
    }
}

fn edge_diff_lines(pre: &Graph, post: &Graph, lines: &mut Vec<String>) {
    let mut pre_keys: HashSet<(&str, &str)> = HashSet::new();
    for edge in &pre.edges {
        pre_keys.insert((edge.source.as_str(), edge.target.as_str()));
    }
    let mut post_keys: HashSet<(&str, &str)> = HashSet::new();
    for edge in &post.edges {
        post_keys.insert((edge.source.as_str(), edge.target.as_str()));
    }
    for edge in &post.edges {
        if !pre_keys.contains(&(edge.source.as_str(), edge.target.as_str())) {
            lines.push(format!(
                "+edges[{} → {} | {} | conf={} imp={} stab={}]",
                edge.source,
                edge.target,
                edge.content,
                edge.confidence,
                edge.importance,
                edge.stability
            ));
        }
    }
    for edge in &pre.edges {
        if !post_keys.contains(&(edge.source.as_str(), edge.target.as_str())) {
            lines.push(format!("-edges[{} → {}]", edge.source, edge.target));
        }
    }
}

pub(crate) fn parse_edge_row(row: &turso::Row) -> Option<Edge> {
    Some(Edge {
        source: normalize_id(&text(row.get_value(0).ok()?)),
        target: normalize_id(&text(row.get_value(1).ok()?)),
        content: text(row.get_value(2).ok()?),
        stability: real(row.get_value(3).ok()?),
        confidence: real(row.get_value(4).ok()?),
        importance: real(row.get_value(5).ok()?),
        subject: int(row.get_value(6).ok()?) != 0,
        date: text(row.get_value(7).ok()?),
        occurred_at: optional_text(row.get_value(8).ok()),
        expired_at: optional_text(row.get_value(9).ok()),
        memory_type: text_or(row.get_value(10), "semantic"),
        origin_session: text_or(row.get_value(11), ""),
        provenance: text_or(row.get_value(12), PROVENANCE_LEGACY),
        ceiling: optional_real(row.get_value(13).ok()),
        last_decayed_at: text_or(row.get_value(14), ""),
    })
}
