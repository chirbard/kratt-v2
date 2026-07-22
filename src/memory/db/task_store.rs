use std::error::Error;

use turso::{Connection, params};

use super::core::runtime;
use super::{parse_edge_row, real, text};
use crate::memory::graph::{Edge, Node, normalize_id};

#[cfg(test)]
#[path = "task_store_tests.rs"]
mod tests;

pub fn load_session_game(
    conn: &Connection,
    hub_id: &str,
) -> Result<Option<(String, Node)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT n.id, n.node_type, n.body, n.date FROM nodes n JOIN edges e ON e.target = n.id WHERE e.source = ?1 AND n.node_type = 'game' AND e.expired_at IS NULL LIMIT 1",
                params![hub_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let game_id = normalize_id(&text(row.get_value(0)?));
            let node = Node {
                node_type: text(row.get_value(1)?),
                body: text(row.get_value(2)?),
                date: text(row.get_value(3)?),
            };
            return Ok(Some((game_id, node)));
        }
        Ok(None)
    })
}

// All live semantic edges incident to a game node (rules + strategies, both
// directions), highest-confidence first and capped. The episodic hub→game link
// is excluded by the semantic filter. This is not TOP_K-ranked: the rules of
// the game you are playing are mandatory context.
pub fn load_game_edges(
    conn: &Connection,
    game_id: &str,
    cap: usize,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling, last_decayed_at FROM edges WHERE (source = ?1 OR target = ?1) AND memory_type = 'semantic' AND expired_at IS NULL ORDER BY confidence DESC LIMIT ?2",
                params![game_id, cap as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            if let Some(edge) = parse_edge_row(&row) {
                edges.push(edge);
            }
        }
        Ok(edges)
    })
}

// Every game-definition node (node_type = "game"), each with its id and node.
// The strategy stage walks these to find games worth consolidating.
pub fn load_game_nodes(
    conn: &Connection,
) -> Result<Vec<(String, Node)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, node_type, body, date FROM nodes WHERE node_type = 'game'",
                (),
            )
            .await?;
        let mut games = Vec::new();
        while let Some(row) = rows.next().await? {
            let game_id = normalize_id(&text(row.get_value(0)?));
            let node = Node {
                node_type: text(row.get_value(1)?),
                body: text(row.get_value(2)?),
                date: text(row.get_value(3)?),
            };
            games.push((game_id, node));
        }
        Ok(games)
    })
}

// The live task hubs that played this game: sources of episodic hub→game edges
// (written by define_task). Expired links and other games' hubs are excluded by
// the target + expiry filter. Each hub id embeds its session as `task-<id>`.
pub fn load_game_hubs(
    conn: &Connection,
    game_id: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT DISTINCT source FROM edges WHERE target = ?1 AND memory_type = 'episodic' AND expired_at IS NULL AND source LIKE 'task-%'",
                params![game_id],
            )
            .await?;
        let mut hubs = Vec::new();
        while let Some(row) = rows.next().await? {
            hubs.push(text(row.get_value(0)?));
        }
        Ok(hubs)
    })
}

// The recorded game outcome for a session, if any: the (score, comment) of its
// most recent `outcome` event. score is the assistant-perspective weight
// (win +1 / draw 0 / loss -1); comment is pulled from the event payload.
pub fn load_session_outcome(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<(f64, String)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                // lower(): the session id arrives via the task hub id, which
                // normalize_id lowercased — the events table has the original
                // casing ("...T12-58-17" vs hub-derived "...t12-58-17").
                "SELECT weight, payload FROM events WHERE lower(session_id) = lower(?1) AND event_type = 'outcome' ORDER BY id DESC LIMIT 1",
                params![session_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let score = real(row.get_value(0)?);
            let comment = outcome_comment(&text(row.get_value(1)?));
            return Ok(Some((score, comment)));
        }
        Ok(None)
    })
}

// Pulls the human comment out of an outcome event payload; a missing or
// unparseable payload just yields an empty comment.
fn outcome_comment(payload: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return String::new();
    };
    let Some(comment) = value.get("comment") else {
        return String::new();
    };
    match comment.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

// Creates the task hub node on the first move of a game and is a no-op
// afterwards — INSERT OR IGNORE preserves the original row so repeated moves
// never reset it.
pub fn ensure_task_hub(
    conn: &Connection,
    hub_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, body, date) VALUES (?1, 'task', '', '')",
            params![hub_id],
        )
        .await?;
        Ok(())
    })
}

// Live ply count on a task hub, counted from the DB so ply numbering survives a
// restart (no in-memory counter). Only move edges (target `<hub>-ply-%`) count,
// so non-move task facts attached to the hub never inflate the number.
pub fn count_task_chain(
    conn: &Connection,
    hub_id: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    count_chain_edges(conn, hub_id, "AND expired_at IS NULL")
}

// Ply numbering must be monotonic: an expired ply still owns its number, or
// the next record_move would upsert over it and resurrect it. Count everything.
pub fn next_ply(conn: &Connection, hub_id: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
    Ok(count_chain_edges(conn, hub_id, "")? + 1)
}

fn count_chain_edges(
    conn: &Connection,
    hub_id: &str,
    expiry_filter: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let pattern = format!("{hub_id}-ply-%");
    let query =
        format!("SELECT COUNT(*) FROM edges WHERE source = ?1 AND target LIKE ?2 {expiry_filter}");
    runtime().block_on(async {
        let mut rows = conn
            .query(&query, params![hub_id, pattern.as_str()])
            .await?;
        if let Some(row) = rows.next().await? {
            return Ok(real(row.get_value(0)?) as usize);
        }
        Ok(0)
    })
}

// The current session's task-hub move chain, ordered by ply (zero-padded
// move-node ids sort lexically = numerically), expired edges excluded. Caps at
// `cap` keeping the most recent plies — a normal game never hits the cap; a
// pathological chain must not blow the context.
pub fn load_task_chain(
    conn: &Connection,
    session_id: &str,
    cap: usize,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let hub_id = normalize_id(&format!("task-{session_id}"));
    let pattern = format!("{hub_id}-ply-%");
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling, last_decayed_at FROM edges WHERE source = ?1 AND target LIKE ?2 AND memory_type = 'episodic' AND expired_at IS NULL ORDER BY target DESC LIMIT ?3",
                params![hub_id.as_str(), pattern.as_str(), cap as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            if let Some(edge) = parse_edge_row(&row) {
                edges.push(edge);
            }
        }
        // Loaded newest-first to honor the cap; flip back to ply order.
        edges.reverse();
        Ok(edges)
    })
}
