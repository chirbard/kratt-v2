use std::error::Error;

use turso::{Connection, params};

use super::{begin_transaction, block_on, commit_transaction, timestamp_now};

pub(crate) const DEFAULT_PING_MUTED_UNTIL: &str = "9999-12-31T23:59:59";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Finding {
    pub finding_id: i64,
    pub interest_topic: String,
    pub title: String,
    pub url: String,
    pub novelty: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PingPolicy {
    pub muted_until: String,
    pub diverts: Vec<(String, String)>,
}

pub(crate) fn unsurfaced_findings(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<Finding>, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, interest_topic, title, url, novelty
                 FROM findings WHERE status = 'candidate' ORDER BY novelty DESC, found_at DESC LIMIT ?1",
                params![limit as i64],
            )
            .await?;
        let mut findings = Vec::new();
        while let Some(row) = rows.next().await? {
            findings.push(finding_from_row(&row)?);
        }
        Ok(findings)
    })
}

pub(crate) async fn unsurfaced_findings_in_windows(
    conn: &Connection,
    first: (&str, &str),
    second: (&str, &str),
    limit: usize,
) -> Result<Vec<Finding>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id, interest_topic, title, url, novelty
             FROM findings WHERE status = 'candidate' AND
             ((found_at >= ?1 AND found_at < ?2) OR (found_at >= ?3 AND found_at < ?4))
             ORDER BY novelty DESC, found_at DESC LIMIT ?5",
            params![first.0, first.1, second.0, second.1, limit as i64],
        )
        .await?;
    let mut findings = Vec::new();
    while let Some(row) = rows.next().await? {
        findings.push(finding_from_row(&row)?);
    }
    Ok(findings)
}

fn finding_from_row(row: &turso::Row) -> Result<Finding, Box<dyn Error + Send + Sync>> {
    Ok(Finding {
        finding_id: row.get::<i64>(0)?,
        interest_topic: row.get::<String>(1)?,
        title: row.get::<String>(2)?,
        url: row.get::<String>(3)?,
        novelty: row.get::<f64>(4)?,
    })
}

pub(crate) fn set_ping_mute(
    conn: &Connection,
    until: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    block_on(async {
        conn.execute(
            "INSERT INTO ping_policy (topic, muted_until) VALUES ('', ?1)
             ON CONFLICT(topic) DO UPDATE SET muted_until = excluded.muted_until",
            params![until],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn resume_pings(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    set_ping_mute(conn, "")
}

pub(crate) fn divert_ping_topic(
    conn: &Connection,
    topic: &str,
    until: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    block_on(async {
        conn.execute(
            "INSERT INTO ping_policy (topic, divert_until) VALUES (?1, ?2)
             ON CONFLICT(topic) DO UPDATE SET divert_until = excluded.divert_until",
            params![topic.trim(), until],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn load_ping_policy(
    conn: &Connection,
    now: &str,
) -> Result<PingPolicy, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut policy = PingPolicy {
            muted_until: DEFAULT_PING_MUTED_UNTIL.to_string(),
            diverts: Vec::new(),
        };
        let mut rows = conn
            .query(
                "SELECT topic, muted_until, divert_until FROM ping_policy",
                (),
            )
            .await?;
        while let Some(row) = rows.next().await? {
            let topic = row.get::<String>(0)?;
            if topic.is_empty() {
                policy.muted_until = row.get::<String>(1)?;
                continue;
            }
            let until = row.get::<String>(2)?;
            if until.as_str() > now {
                policy.diverts.push((topic, until));
            }
        }
        Ok(policy)
    })
}

pub(crate) fn mark_finding_surfaced(
    conn: &Connection,
    finding_id: i64,
    session_id: &str,
    turn_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let surfaced_at = timestamp_now();
    begin_transaction(conn)?;
    let result = block_on(async {
        conn.execute(
            "UPDATE findings SET status = 'surfaced', surfaced_at = ?1
             WHERE id = ?2 AND status = 'candidate'",
            params![surfaced_at.as_str(), finding_id],
        )
        .await?;
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'conversation', ?2)",
            params![session_id, surfaced_at.as_str()],
        )
        .await?;
        let payload = serde_json::json!({"finding_id": finding_id, "surfaced": true}).to_string();
        conn.execute(
            "INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, payload)
             VALUES (?1, ?2, ?3, 'system', 'ping', 'surfaced', ?4)",
            params![surfaced_at, session_id, turn_id, payload],
        )
        .await?;
        Ok(())
    });
    if let Err(err) = result {
        let _ = block_on(async { conn.execute("ROLLBACK", ()).await });
        return Err(err);
    }
    commit_transaction(conn)
}

pub(crate) async fn count_findings_in_windows(
    conn: &Connection,
    first_from: &str,
    first_until: &str,
    second_from: &str,
    second_until: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM findings
             WHERE (found_at >= ?1 AND found_at < ?2)
                OR (found_at >= ?3 AND found_at < ?4)",
            params![first_from, first_until, second_from, second_until],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(row.get::<i64>(0)? as usize)
}

#[cfg(test)]
#[path = "findings_tests.rs"]
mod tests;
