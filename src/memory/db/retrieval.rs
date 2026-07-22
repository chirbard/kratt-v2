use std::collections::{HashMap, HashSet};
use std::error::Error;

use turso::{Connection, Value, params};

use super::core::runtime;
use super::{
    add_interest_node_columns, block_on, int, open_connection, optional_text, parse_edge_row, real,
    text, text_or, timestamp_now,
};
use crate::memory::graph::{
    PROVENANCE_LEGACY, PROVENANCE_WEB_SURFACED, normalize_id, stability_seed_max,
};
use crate::memory::loops::OpenLoop;

#[cfg(test)]
#[path = "retrieval_tests.rs"]
mod tests;

pub type SemanticEdgeRow = (
    String,
    String,
    String,
    f64,
    f64,
    f64,
    bool,
    String,
    Option<String>,
    Option<String>,
    String,
    String,
    String,
    Option<Vec<f64>>,
);

pub type OpenCuriosityNoteRow = (String, String, String, String);

pub(crate) const INTEREST_NEIGHBOUR_DISTANCE: f64 = 0.25; // replay-tunable.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Interest {
    pub node_id: String,
    pub topic: String,
    pub origin: String,
    pub created_at: String,
    pub strength: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct InterestTarget {
    pub topic: String,
    pub density: usize,
    pub embedding: Vec<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecentLearning {
    pub content: String,
    pub source: String,
    pub target: String,
    pub event_type: String,
    pub timestamp: String,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NearestLiveEdge {
    pub source: String,
    pub target: String,
    pub provenance: String,
    pub confidence: f64,
    pub similarity: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct RegionTransition {
    pub target: String,
    pub timestamp: String,
    pub signal: String,
}

#[derive(Clone, Debug)]
pub(crate) struct SequenceSuggestion {
    pub turn_id: String,
    pub source: String,
    pub target: String,
}

pub fn vector32_json(vector: &[f64]) -> Result<String, Box<dyn Error + Send + Sync>> {
    Ok(serde_json::to_string(vector)?)
}

pub fn parse_vector32_json(json_text: &str) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let values = serde_json::from_str::<Vec<f64>>(json_text)?;
    Ok(values)
}

pub(crate) fn insert_interest(
    conn: &Connection,
    topic: &str,
    origin: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let interest_id = normalize_id(topic);
    if interest_id.is_empty() {
        return Err("interest topic is empty".into());
    }
    let created_at = crate::timeutil::timestamp_now();
    runtime().block_on(async {
        add_interest_node_columns(conn).await;
        conn.execute(
            "INSERT INTO nodes (id, node_type, body, date, origin, embedding)
             VALUES (?1, 'interest', ?2, ?3, ?4, NULL)
             ON CONFLICT(id) DO UPDATE SET node_type = 'interest', body = excluded.body,
                 date = excluded.date, origin = excluded.origin, embedding = NULL",
            params![interest_id.as_str(), topic.trim(), created_at, origin],
        )
        .await?;
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    })?;
    Ok(interest_id)
}

pub(crate) fn top_interests(limit: usize) -> Vec<Interest> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(top_interests_with(&conn, limit))
        .unwrap_or_default()
}

pub(crate) async fn top_interests_with(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<Interest>, Box<dyn Error + Send + Sync>> {
    add_interest_node_columns(conn).await;
    let mut interests = load_registered_interests(conn).await?;
    append_attention_interests(conn, &mut interests).await?;
    append_curiosity_interests(conn, &mut interests).await?;
    append_actor_position_interests(conn, &mut interests).await?;
    interests.sort_by(|left, right| {
        right
            .strength
            .total_cmp(&left.strength)
            .then_with(|| right.created_at.cmp(&left.created_at))
    });
    dedup_interests(&mut interests);
    interests.truncate(limit);
    Ok(interests)
}

async fn load_registered_interests(
    conn: &Connection,
) -> Result<Vec<Interest>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id, body, origin, date FROM nodes WHERE node_type = 'interest' ORDER BY date DESC",
            (),
        )
        .await?;
    let mut interests = Vec::new();
    while let Some(row) = rows.next().await? {
        interests.push(Interest {
            node_id: text(row.get_value(0)?),
            topic: text(row.get_value(1)?),
            origin: text(row.get_value(2)?),
            created_at: text(row.get_value(3)?),
            strength: 1.0,
        });
    }
    Ok(interests)
}

async fn append_attention_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT endpoint, MAX(ts) AS newest, COUNT(*) AS strength
             FROM (
                 SELECT source AS endpoint, ts FROM events
                 WHERE event_type IN ('reinforced', 'marked') AND source != ''
                 UNION ALL
                 SELECT target AS endpoint, ts FROM events
                 WHERE event_type IN ('reinforced', 'marked') AND target != ''
             ) attention
             JOIN nodes n ON n.id = endpoint
             WHERE n.node_type NOT IN ('interest', 'loop', 'note')
             GROUP BY endpoint ORDER BY COUNT(*) DESC, newest DESC LIMIT 12",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        let node_id = text(row.get_value(0)?);
        interests.push(derived_interest(
            &node_id,
            &node_id,
            text(row.get_value(1)?),
            real(row.get_value(2)?),
        ));
    }
    Ok(())
}

async fn append_curiosity_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT e.target, COALESCE(NULLIF(target.body, ''), e.target), note.date
             FROM nodes note JOIN edges e ON e.source = note.id
             LEFT JOIN nodes target ON target.id = e.target
             WHERE note.node_type = 'note' AND e.expired_at IS NULL
             ORDER BY note.date DESC LIMIT 12",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        interests.push(derived_interest(
            &text(row.get_value(0)?),
            &text(row.get_value(1)?),
            text(row.get_value(2)?),
            1.0,
        ));
    }
    Ok(())
}

async fn append_actor_position_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT vector_extract(sum_vector), updated_at, total_weight FROM actor_positions
             WHERE sum_vector IS NOT NULL ORDER BY updated_at DESC LIMIT 6",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        let vector_text = text(row.get_value(0)?);
        let Ok(vector) = parse_vector32_json(&vector_text) else {
            continue;
        };
        let Some(node_id) = nearest_node_to_vector(conn, &vector).await? else {
            continue;
        };
        interests.push(derived_interest(
            &node_id,
            &node_id,
            text(row.get_value(1)?),
            real(row.get_value(2)?),
        ));
    }
    Ok(())
}

async fn nearest_node_to_vector(
    conn: &Connection,
    vector: &[f64],
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT e.source FROM edges e JOIN nodes n ON n.id = e.source
             WHERE e.expired_at IS NULL AND e.embedding IS NOT NULL
               AND n.node_type NOT IN ('interest', 'loop', 'note')
             ORDER BY vector_distance_cos(e.embedding, vector32(?1)) LIMIT 1",
            params![json_str],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

fn derived_interest(node_id: &str, topic: &str, created_at: String, strength: f64) -> Interest {
    Interest {
        node_id: normalize_id(node_id),
        topic: topic.to_string(),
        origin: "derived".to_string(),
        created_at,
        strength,
    }
}

fn dedup_interests(interests: &mut Vec<Interest>) {
    let mut seen = HashSet::new();
    interests.retain(|interest| seen.insert(interest.node_id.clone()));
}

// Weights are stored at two decimals: full float precision survives every
// multiply-and-write cycle (decay, reinforce) and bloats the prompt/token
// side; nothing downstream needs more than 0.01 resolution.

pub(crate) fn nearest_live_edges(
    conn: &Connection,
    vector: &[f64],
    limit: usize,
) -> Result<Vec<NearestLiveEdge>, Box<dyn Error + Send + Sync>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let json_str = vector32_json(vector)?;
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, provenance, confidence,
                        vector_distance_cos(embedding, vector32(?1))
                 FROM edges WHERE embedding IS NOT NULL AND expired_at IS NULL
                 ORDER BY vector_distance_cos(embedding, vector32(?1)) LIMIT ?2",
                params![json_str, limit as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            let distance = row.get::<f64>(4)?;
            edges.push(NearestLiveEdge {
                source: text(row.get_value(0)?),
                target: text(row.get_value(1)?),
                provenance: text_or(row.get_value(2), PROVENANCE_LEGACY),
                confidence: real(row.get_value(3)?),
                similarity: (1.0 - distance).clamp(-1.0, 1.0),
            });
        }
        Ok(edges)
    })
}

pub(crate) fn record_news_graph_material(
    conn: &Connection,
    finding_id: i64,
    interest: &Interest,
    item: &crate::web::NewsItem,
    embedding: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let timestamp = timestamp_now();
    let today = crate::timeutil::today_iso();
    let session_id = format!("news:{today}");
    let target_id = format!("news-finding-{finding_id}");
    let content = format!("{} — {}", item.title, item.snippet);
    let json_str = vector32_json(embedding)?;
    let stability = stability_seed_max(PROVENANCE_WEB_SURFACED);
    block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, body, date)
             VALUES (?1, 'event', ?2, ?3)",
            params![target_id.as_str(), item.title.as_str(), timestamp.as_str()],
        )
        .await?;
        conn.execute(
            "INSERT OR REPLACE INTO edges
             (source, target, content, stability, confidence, importance, subject, date,
              occurred_at, memory_type, origin_session, provenance, embedding)
             VALUES (?1, ?2, ?3, ?4, 0.5, 0.4, 0, ?5, ?5, 'episodic', ?6, 'web-surfaced', vector32(?7))",
            params![
                interest.node_id.as_str(),
                target_id.as_str(),
                content.as_str(),
                stability,
                timestamp.as_str(),
                session_id.as_str(),
                json_str.as_str()
            ],
        )
        .await?;
        Ok(())
    })
}

pub fn load_unembedded_edges() -> Vec<(String, String, String)> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(load_unembedded_edges_async(&conn))
}

pub(crate) fn load_unembedded_interests() -> Vec<(String, String)> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(async {
        add_interest_node_columns(&conn).await;
        let Ok(mut rows) = conn
            .query(
                "SELECT id, body FROM nodes WHERE node_type = 'interest' AND embedding IS NULL",
                (),
            )
            .await
        else {
            return Vec::new();
        };
        let mut interests = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let (Ok(node_id), Ok(topic)) = (row.get_value(0), row.get_value(1)) else {
                continue;
            };
            interests.push((text(node_id), text(topic)));
        }
        interests
    })
}

pub(crate) fn save_interest_embedding(
    node_id: &str,
    vector: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(async {
        add_interest_node_columns(&conn).await;
        conn.execute(
            "UPDATE nodes SET embedding = vector32(?1) WHERE id = ?2 AND node_type = 'interest'",
            params![json_str, node_id],
        )
        .await?;
        Ok(())
    })
}

pub(super) async fn interest_density_with(
    conn: &Connection,
    vector: &[f64],
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM edges WHERE expired_at IS NULL AND embedding IS NOT NULL
             AND vector_distance_cos(embedding, vector32(?1)) <= ?2",
            params![json_str, max_distance],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(int(row.get_value(0)?) as usize)
}

pub(crate) fn densification_targets(limit: usize) -> Vec<InterestTarget> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(densification_targets_with(&conn, limit))
        .unwrap_or_default()
}

async fn densification_targets_with(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<InterestTarget>, Box<dyn Error + Send + Sync>> {
    add_interest_node_columns(conn).await;
    let mut rows = conn
        .query(
            "SELECT body, vector_extract(embedding) FROM nodes
             WHERE node_type = 'interest' AND embedding IS NOT NULL",
            (),
        )
        .await?;
    let mut targets = Vec::new();
    while let Some(row) = rows.next().await? {
        let topic = text(row.get_value(0)?);
        let vector_text = text(row.get_value(1)?);
        let Ok(embedding) = parse_vector32_json(&vector_text) else {
            continue;
        };
        let density = interest_density_with(conn, &embedding, INTEREST_NEIGHBOUR_DISTANCE).await?;
        targets.push(InterestTarget {
            topic,
            density,
            embedding,
        });
    }
    targets.sort_by_key(|target| target.density);
    targets.truncate(limit);
    Ok(targets)
}

pub(super) async fn load_unembedded_edges_async(
    conn: &Connection,
) -> Vec<(String, String, String)> {
    let Ok(mut rows) = conn
        .query(
            // Loop edges ARE embedded (the similarity trigger below needs
            // them); they stay excluded from ranked retrieval only.
            "SELECT e.source, e.target, e.content FROM edges e WHERE e.expired_at IS NULL AND e.embedding IS NULL",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut edges = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(edge) = parse_unembedded_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    edges
}

fn parse_unembedded_edge_row(row: &turso::Row) -> Option<(String, String, String)> {
    let source = text(row.get_value(0).ok()?);
    let target = text(row.get_value(1).ok()?);
    let content = text(row.get_value(2).ok()?);
    Some((source, target, content))
}

pub fn save_edge_embedding(
    source: &str,
    target: &str,
    vector: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(async {
        conn.execute(
            "UPDATE edges SET embedding = vector32(?1) WHERE source = ?2 AND target = ?3",
            params![json_str.as_str(), source, target],
        )
        .await?;
        Ok(())
    })
}

pub fn query_edges_by_vector(
    vector: &[f64],
    limit: usize,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(query_edges_by_vector_async(&conn, &json_str, limit))
}

async fn query_edges_by_vector_async(
    conn: &Connection,
    json_str: &str,
    limit: usize,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let limit_value = limit as i64;
    match query_edges_ann(conn, json_str, limit_value).await {
        Ok(rows) => Ok(rows),
        Err(_) => query_edges_scan(conn, json_str, limit_value).await,
    }
}

// Live loops nearest to the query embedding — the similarity trigger for the
// Open loops block. Loops are excluded from ranked TOP_K, so without this a
// loop whose fresh anchor has no other edges could never surface contextually.
pub fn load_loops_by_similarity(
    conn: &Connection,
    query_vec: &[f64],
    max_distance: f64,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_loops_by_similarity_async(
        conn,
        query_vec,
        max_distance,
        cap,
    ))
}

async fn load_loops_by_similarity_async(
    conn: &Connection,
    query_vec: &[f64],
    max_distance: f64,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(query_vec)?;
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL AND e.embedding IS NOT NULL AND vector_distance_cos(e.embedding, vector32(?1)) <= ?2 ORDER BY vector_distance_cos(e.embedding, vector32(?1)) ASC LIMIT ?3"
    );
    let rows = conn
        .query(&query, params![json_str, max_distance, cap as i64])
        .await?;
    collect_open_loops(rows).await
}

async fn query_edges_ann(
    conn: &Connection,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    execute_semantic_query(
        conn,
        "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, vector_extract(e.embedding) FROM vector_top_k('edges_vec', vector32(?1), ?2) AS vt JOIN edges AS e ON e.rowid = vt.id LEFT JOIN nodes AS source_node ON source_node.id = e.source LEFT JOIN nodes AS target_node ON target_node.id = e.target WHERE e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest')",
        json_str,
        limit,
    )
    .await
}

async fn query_edges_scan(
    conn: &Connection,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    execute_semantic_query(
        conn,
        "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, vector_extract(e.embedding) FROM edges e LEFT JOIN nodes source_node ON source_node.id = e.source LEFT JOIN nodes target_node ON target_node.id = e.target WHERE e.embedding IS NOT NULL AND e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest') ORDER BY vector_distance_cos(e.embedding, vector32(?1)) LIMIT ?2",
        json_str,
        limit,
    )
    .await
}

async fn execute_semantic_query(
    conn: &Connection,
    query: &str,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn.query(query, params![json_str, limit]).await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        let Some(edge) = parse_semantic_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    Ok(edges)
}

fn parse_semantic_edge_row(row: &turso::Row) -> Option<SemanticEdgeRow> {
    let edge = parse_edge_row(row)?;
    let embedding = parse_optional_vector32(row.get_value(14).ok());
    Some((
        edge.source,
        edge.target,
        edge.content,
        edge.confidence,
        edge.stability,
        edge.importance,
        edge.subject,
        edge.date,
        edge.occurred_at,
        edge.expired_at,
        edge.memory_type,
        edge.origin_session,
        edge.provenance,
        embedding,
    ))
}

pub fn load_all_edges_semantic() -> Vec<SemanticEdgeRow> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(load_all_edges_semantic_async(&conn))
}

pub(super) async fn load_all_edges_semantic_async(conn: &Connection) -> Vec<SemanticEdgeRow> {
    let Ok(mut rows) = conn
        .query(
            "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, CASE WHEN e.embedding IS NULL THEN NULL ELSE vector_extract(e.embedding) END FROM edges e LEFT JOIN nodes source_node ON source_node.id = e.source LEFT JOIN nodes target_node ON target_node.id = e.target WHERE e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest')",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut edges = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(edge) = parse_semantic_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    edges
}

pub fn load_edge_actor_views(actor_type: &str, actor_id: &str) -> HashMap<(String, String), f64> {
    let Ok(conn) = open_connection() else {
        return HashMap::new();
    };
    runtime().block_on(load_edge_actor_views_async(&conn, actor_type, actor_id))
}

pub fn load_recent_attention_counts(
    candidates: &[(String, String)],
    window_days: usize,
) -> HashMap<(String, String), usize> {
    if candidates.is_empty() {
        return HashMap::new();
    }
    let Ok(conn) = open_connection() else {
        return HashMap::new();
    };
    runtime()
        .block_on(load_recent_attention_counts_async(
            &conn,
            candidates,
            window_days,
        ))
        .unwrap_or_default()
}

pub(super) async fn load_recent_attention_counts_async(
    conn: &Connection,
    candidates: &[(String, String)],
    window_days: usize,
) -> Result<HashMap<(String, String), usize>, Box<dyn Error + Send + Sync>> {
    let mut values = Vec::new();
    let mut params = Vec::new();
    for (source, target) in candidates {
        values.push("(?, ?)");
        params.push(Value::Text(source.clone()));
        params.push(Value::Text(target.clone()));
    }
    params.push(Value::Integer(-(window_days as i64)));
    let query = format!(
        "WITH candidates(source, target) AS (VALUES {})
         SELECT ev.source, ev.target, COUNT(*)
         FROM events ev
         JOIN candidates c ON c.source = ev.source AND c.target = ev.target
         WHERE ev.event_type IN ('reinforced', 'marked')
           AND julianday(ev.ts) >= julianday('now', printf('%d days', ?))
         GROUP BY ev.source, ev.target",
        values.join(", ")
    );
    let mut rows = conn.query(&query, turso::params_from_iter(params)).await?;
    let mut counts = HashMap::new();
    while let Some(row) = rows.next().await? {
        let source = text(row.get_value(0)?);
        let target = text(row.get_value(1)?);
        counts.insert((source, target), int(row.get_value(2)?) as usize);
    }
    Ok(counts)
}

pub(crate) async fn load_turn_region(
    conn: &Connection,
    turn_id: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT ev.source
             FROM events ev
             JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
             LEFT JOIN nodes n ON n.id = ev.source
             WHERE ev.turn_id = ?1 AND ev.event_type = 'injected'
               AND ed.expired_at IS NULL
             ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC
             LIMIT 1",
            params![turn_id],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let region = text(row.get_value(0)?);
    if region.is_empty() {
        return Ok(None);
    }
    Ok(Some(region))
}

pub(crate) async fn load_previous_turn_region(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let Some(turn_id) = load_previous_user_turn(conn, session_id, current_turn).await? else {
        return Ok(None);
    };
    load_turn_region(conn, &turn_id).await
}

async fn load_previous_user_turn(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT previous.id
             FROM messages current
             JOIN messages previous ON previous.session_id = current.session_id
             WHERE current.id = ?1 AND current.session_id = ?2
               AND previous.role = 'user'
               AND (previous.created_at < current.created_at
                    OR (previous.created_at = current.created_at
                        AND previous.rowid < current.rowid))
             ORDER BY previous.created_at DESC, previous.rowid DESC
             LIMIT 1",
            params![current_turn, session_id],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

pub(crate) async fn load_region_transitions(
    conn: &Connection,
    source: &str,
    window_days: usize,
) -> Result<Vec<RegionTransition>, Box<dyn Error + Send + Sync>> {
    let mut signals = load_adjacent_region_transitions(conn, source, window_days).await?;
    load_sequence_feedback(conn, source, window_days, &mut signals).await?;
    Ok(signals)
}

const REGION_TRANSITIONS_QUERY: &str = "SELECT current.created_at,
           (SELECT ev.source FROM events ev
            JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
            LEFT JOIN nodes n ON n.id = ev.source
            WHERE ev.turn_id = (
                SELECT previous.id FROM messages previous
                WHERE previous.session_id = current.session_id
                  AND previous.role = 'user'
                  AND (previous.created_at < current.created_at
                       OR (previous.created_at = current.created_at
                           AND previous.rowid < current.rowid))
                ORDER BY previous.created_at DESC, previous.rowid DESC LIMIT 1
            )
              AND ev.event_type = 'injected' AND ed.expired_at IS NULL
            ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC LIMIT 1),
           (SELECT ev.source FROM events ev
            JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
            LEFT JOIN nodes n ON n.id = ev.source
            WHERE ev.turn_id = current.id
              AND ev.event_type = 'injected' AND ed.expired_at IS NULL
            ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC LIMIT 1)
    FROM messages current
    JOIN sessions session ON session.id = current.session_id
    WHERE current.role = 'user' AND session.session_type = 'conversation'
      AND julianday(current.created_at) >= julianday('now', printf('%d days', ?1))";

async fn load_adjacent_region_transitions(
    conn: &Connection,
    source: &str,
    window_days: usize,
) -> Result<Vec<RegionTransition>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(REGION_TRANSITIONS_QUERY, params![-(window_days as i64)])
        .await?;
    let mut signals = Vec::new();
    while let Some(row) = rows.next().await? {
        let previous = text(row.get_value(1)?);
        let current = text(row.get_value(2)?);
        if previous != source || current.is_empty() || current == previous {
            continue;
        }
        signals.push(RegionTransition {
            target: current,
            timestamp: text(row.get_value(0)?),
            signal: "transition".to_string(),
        });
    }
    Ok(signals)
}

async fn load_sequence_feedback(
    conn: &Connection,
    source: &str,
    window_days: usize,
    signals: &mut Vec<RegionTransition>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT target, ts, event_type FROM events
             WHERE actor_id = 'sequence' AND source = ?1
               AND event_type IN ('followed', 'rejected')
               AND julianday(ts) >= julianday('now', printf('%d days', ?2))
             ORDER BY id",
            params![source, -(window_days as i64)],
        )
        .await?;
    while let Some(row) = rows.next().await? {
        signals.push(RegionTransition {
            target: text(row.get_value(0)?),
            timestamp: text(row.get_value(1)?),
            signal: text(row.get_value(2)?),
        });
    }
    Ok(())
}

pub(crate) async fn last_suggestion_for_session(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<SequenceSuggestion>, Box<dyn Error + Send + Sync>> {
    let Some(previous_turn) = load_previous_user_turn(conn, session_id, current_turn).await? else {
        return Ok(None);
    };
    let mut rows = conn
        .query(
            "SELECT suggested.turn_id, suggested.source, suggested.target
             FROM events suggested
             WHERE suggested.session_id = ?1 AND suggested.turn_id = ?2
               AND suggested.actor_id = 'sequence'
               AND suggested.event_type = 'suggested'
               AND NOT EXISTS (
                   SELECT 1 FROM events outcome
                   WHERE outcome.session_id = ?1 AND outcome.turn_id = ?3
                     AND outcome.actor_id = 'sequence'
                     AND outcome.event_type IN ('followed', 'rejected')
                     AND outcome.source = suggested.source
                     AND outcome.target = suggested.target
               )
             ORDER BY suggested.id DESC LIMIT 1",
            params![session_id, previous_turn, current_turn],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(SequenceSuggestion {
        turn_id: text(row.get_value(0)?),
        source: text(row.get_value(1)?),
        target: text(row.get_value(2)?),
    }))
}

async fn load_edge_actor_views_async(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> HashMap<(String, String), f64> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source, target, views FROM edge_actor_views WHERE actor_type = ?1 AND actor_id = ?2",
            params![actor_type, actor_id],
        )
        .await
    else {
        return HashMap::new();
    };
    let mut views = HashMap::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some((source, target, count)) = parse_edge_actor_view_row(&row) else {
            continue;
        };
        views.insert((source, target), count);
    }
    views
}

fn parse_edge_actor_view_row(row: &turso::Row) -> Option<(String, String, f64)> {
    let source = text(row.get_value(0).ok()?);
    let target = text(row.get_value(1).ok()?);
    let count = real(row.get_value(2).ok()?);
    Some((source, target, count))
}

pub(super) async fn delete_edge_actor_views_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "DELETE FROM edge_actor_views WHERE source = ?1 AND target = ?2",
        params![source, target],
    )
    .await?;
    Ok(())
}

pub fn edge_count() -> usize {
    let Ok(conn) = open_connection() else {
        return 0;
    };
    let result = runtime().block_on(async {
        let mut rows = conn
            .query("SELECT COUNT(*) FROM edges WHERE expired_at IS NULL", ())
            .await?;
        if let Some(row) = rows.next().await?
            && let Value::Integer(count) = row.get_value(0)?
        {
            return Ok::<usize, Box<dyn Error + Send + Sync>>(count as usize);
        }
        Ok(0)
    });
    result.unwrap_or_default()
}

pub fn load_open_curiosity_notes(limit: usize) -> Vec<OpenCuriosityNoteRow> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(load_open_curiosity_notes_async(&conn, limit))
        .unwrap_or_default()
}

pub(crate) async fn load_open_curiosity_notes_async(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<OpenCuriosityNoteRow>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT n.id, n.body, e.target, n.date
                 FROM nodes n
                 JOIN edges e ON e.source = n.id
                 LEFT JOIN (
                     SELECT DISTINCT ev.source, ev.target
                     FROM events ev
                     JOIN sessions s ON s.id = ev.session_id
                     WHERE s.session_type = 'exploration' AND ev.event_type = 'retrieved'
                 ) consumed ON consumed.source = e.source AND consumed.target = e.target
                 WHERE n.node_type = 'note' AND e.expired_at IS NULL AND consumed.source IS NULL
                 ORDER BY n.date ASC, n.id ASC
                 LIMIT ?1",
            params![limit as i64],
        )
        .await?;
    let mut notes = Vec::new();
    while let Some(row) = rows.next().await? {
        notes.push((
            text(row.get_value(0)?),
            text(row.get_value(1)?),
            text(row.get_value(2)?),
            text(row.get_value(3)?),
        ));
    }
    Ok(notes)
}

pub(crate) async fn load_open_intentions_async(
    conn: &Connection,
    today: &str,
    limit: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL ORDER BY (n.date != '' AND n.date <= ?1) DESC, e.date DESC, e.rowid DESC LIMIT ?2"
    );
    let rows = conn.query(&query, params![today, limit as i64]).await?;
    collect_open_loops(rows).await
}

pub(crate) async fn load_recent_learning_async(
    conn: &Connection,
    since_days: usize,
    limit: usize,
) -> Result<Vec<RecentLearning>, Box<dyn Error + Send + Sync>> {
    let modifier = format!("-{} days", since_days.min(7));
    let mut rows = conn
        .query(
            "SELECT ed.content, ev.source, ev.target, ev.event_type, MAX(ev.ts), ed.provenance
             FROM events ev
             JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
             WHERE ev.event_type IN ('created', 'reinforced')
               AND ev.source != '' AND ev.target != ''
               AND ed.expired_at IS NULL
               AND julianday(ev.ts) >= julianday(date('now'), ?1)
             GROUP BY ed.source, ed.target, ed.content, ev.event_type, ed.provenance
             ORDER BY MAX(ev.ts) DESC, MAX(ev.id) DESC
             LIMIT ?2",
            params![modifier, limit as i64],
        )
        .await?;
    let mut learning = Vec::new();
    while let Some(row) = rows.next().await? {
        learning.push(RecentLearning {
            content: text(row.get_value(0)?),
            source: text(row.get_value(1)?),
            target: text(row.get_value(2)?),
            event_type: text(row.get_value(3)?),
            timestamp: text(row.get_value(4)?),
            provenance: text(row.get_value(5)?),
        });
    }
    Ok(learning)
}

pub(crate) async fn load_inbox_snapshot_async(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, status, result FROM inbox
             WHERE status IN ('queued', 'done', 'failed')
             ORDER BY (status = 'queued') DESC,
                      CASE WHEN status = 'queued' THEN accepted_at ELSE resolved_at END DESC,
                      id DESC
             LIMIT ?1",
            params![limit as i64],
        )
        .await?;
    let mut items = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = text(row.get_value(0)?);
        let status = text(row.get_value(1)?);
        let result = text(row.get_value(2)?);
        if status == "failed" && !result.is_empty() {
            items.push(format!("{source}: failed ({result})"));
        } else {
            items.push(format!("{source}: {status}"));
        }
    }
    Ok(items)
}

// The start of the current conversation "burst": scan recent messages newest
// first and return the timestamp on the older side of the first away-gap. That
// is the last thing said before the user stepped away, so a catch-up window
// from it spans whatever the background passes did overnight. None = no gap in
// the scanned window (continuously active, or first-ever chat).

pub fn count_new_embedding_neighbours(
    vector: &[f64],
    created_at: &str,
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    runtime().block_on(count_new_embedding_neighbours_async(
        &conn,
        vector,
        created_at,
        max_distance,
    ))
}

pub(super) async fn count_new_embedding_neighbours_async(
    conn: &Connection,
    vector: &[f64],
    created_at: &str,
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT COUNT(*)
             FROM edges e
             JOIN (
                 SELECT source, target, MIN(ts) AS created_at
                 FROM events
                 WHERE event_type = 'created' AND source != '' AND target != ''
                 GROUP BY source, target
             ) created ON created.source = e.source AND created.target = e.target
             WHERE e.expired_at IS NULL
               AND e.embedding IS NOT NULL
               AND created.created_at > ?2
               AND vector_distance_cos(e.embedding, vector32(?1)) <= ?3",
            params![json_str, created_at, max_distance],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(int(row.get_value(0)?) as usize)
}

const OPEN_LOOP_SELECT: &str = "SELECT e.source, e.target, n.body, n.date FROM nodes n JOIN edges e ON e.source = n.id JOIN nodes anchor ON anchor.id = e.target";

pub fn load_loops_for_anchors(
    conn: &Connection,
    anchor_ids: &[String],
    today: &str,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_loops_for_anchors_async(conn, anchor_ids, today, cap))
}

async fn load_loops_for_anchors_async(
    conn: &Connection,
    anchor_ids: &[String],
    today: &str,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let (query, values) = loop_anchor_query(anchor_ids, today, cap);
    let rows = conn.query(&query, turso::params_from_iter(values)).await?;
    collect_open_loops(rows).await
}

fn loop_anchor_query(anchor_ids: &[String], today: &str, cap: usize) -> (String, Vec<Value>) {
    let mut values = vec![Value::Text(today.to_string())];
    let mut placeholders = Vec::new();
    for anchor_id in anchor_ids {
        values.push(Value::Text(anchor_id.clone()));
        placeholders.push(format!("?{}", values.len()));
    }
    let anchor_filter = if placeholders.is_empty() {
        String::new()
    } else {
        format!(" OR e.target IN ({})", placeholders.join(", "))
    };
    values.push(Value::Integer(cap as i64));
    let limit_param = values.len();
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL AND ((n.date != '' AND n.date <= ?1){anchor_filter}) ORDER BY e.date ASC, e.rowid ASC LIMIT ?{limit_param}"
    );
    (query, values)
}

pub(crate) async fn load_due_loops_async(
    conn: &Connection,
    today: &str,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND n.date != '' AND n.date <= ?1 AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL ORDER BY n.date ASC, e.rowid ASC"
    );
    let rows = conn.query(&query, params![today]).await?;
    collect_open_loops(rows).await
}

async fn collect_open_loops(
    mut rows: turso::Rows,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let mut loops = Vec::new();
    while let Some(row) = rows.next().await? {
        loops.push(OpenLoop {
            source: normalize_id(&text(row.get_value(0)?)),
            target: normalize_id(&text(row.get_value(1)?)),
            intention: text(row.get_value(2)?),
            due: text(row.get_value(3)?),
        });
    }
    Ok(loops)
}

fn parse_optional_vector32(value: Option<Value>) -> Option<Vec<f64>> {
    let json_text = optional_text(value)?;
    if json_text.is_empty() {
        return None;
    }
    parse_vector32_json(&json_text).ok()
}
