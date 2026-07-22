use std::error::Error;

use turso::{Connection, params};

use super::{block_on, timestamp_now};

#[derive(Debug, Clone, PartialEq)]
pub struct InboxItem {
    pub item_id: i64,
    pub source: String,
    pub instruction: String,
}

pub fn insert_inbox_item(
    conn: &Connection,
    source: &str,
    instruction: &str,
) -> Result<i64, Box<dyn Error + Send + Sync>> {
    let accepted_at = timestamp_now();
    block_on(async {
        conn.execute(
            "INSERT INTO inbox (accepted_at, source, instruction) VALUES (?1, ?2, ?3)",
            params![accepted_at, source, instruction],
        )
        .await?;
        Ok(conn.last_insert_rowid())
    })
}

pub fn oldest_queued_inbox(
    conn: &Connection,
) -> Result<Option<InboxItem>, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, source, instruction FROM inbox WHERE status = 'queued' ORDER BY id LIMIT 1",
                (),
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        Ok(Some(InboxItem {
            item_id: row.get::<i64>(0)?,
            source: row.get::<String>(1)?,
            instruction: row.get::<String>(2)?,
        }))
    })
}

// Short display names (last path/URL segment) of queued items, oldest first,
// capped — feeds the chat-visible internal-state line.
pub async fn load_queued_inbox_names(conn: &Connection) -> Vec<String> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source FROM inbox WHERE status = 'queued' ORDER BY id LIMIT 3",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut names = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let source = row.get::<String>(0).unwrap_or_default();
        let name = match source.trim_end_matches('/').rsplit('/').next() {
            Some(tail) if !tail.is_empty() => tail.to_string(),
            _ => source,
        };
        names.push(name);
    }
    names
}

pub fn inbox_source_is_queued(
    conn: &Connection,
    source: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT 1 FROM inbox WHERE status = 'queued' AND source = ?1 LIMIT 1",
                params![source],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    })
}

pub fn resolve_inbox_item(
    conn: &Connection,
    item_id: i64,
    status: &str,
    result: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let resolved_at = timestamp_now();
    block_on(async {
        conn.execute(
            "UPDATE inbox SET status = ?1, resolved_at = ?2, result = ?3 WHERE id = ?4",
            params![status, resolved_at, result, item_id],
        )
        .await?;
        Ok(())
    })
}

pub async fn count_queued_inbox(conn: &Connection) -> usize {
    let Ok(mut rows) = conn
        .query("SELECT COUNT(*) FROM inbox WHERE status = 'queued'", ())
        .await
    else {
        return 0;
    };
    match rows.next().await {
        Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0) as usize,
        _ => 0,
    }
}

pub async fn load_inbox_outcomes(
    conn: &Connection,
    from: &str,
    until: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn.query(
        "SELECT source, status, result FROM inbox WHERE status = 'queued' OR (resolved_at >= ?1 AND resolved_at < ?2) ORDER BY id",
        params![from, until],
    ).await else { return Ok(Vec::new()) };
    let mut outcomes = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = row.get::<String>(0)?;
        let status = row.get::<String>(1)?;
        let result = row.get::<String>(2)?;
        if status == "failed" {
            outcomes.push(format!("{source}: failed ({result})"));
        } else {
            outcomes.push(format!("{source}: {status}"));
        }
    }
    Ok(outcomes)
}
