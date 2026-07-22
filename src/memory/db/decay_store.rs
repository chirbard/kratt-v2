use std::error::Error;

use turso::{Connection, params};

use super::{open_connection, round2, text};
use super::core::runtime;

pub fn load_last_decay() -> String {
    let Ok(conn) = open_connection() else {
        return String::new();
    };
    runtime().block_on(async {
        let Ok(mut rows) = conn
            .query("SELECT value FROM meta WHERE key = 'last_decay'", ())
            .await
        else {
            return String::new();
        };
        if let Ok(Some(row)) = rows.next().await
            && let Ok(val) = row.get_value(0)
        {
            return text(val);
        }
        String::new()
    })
}

// One decayed edge: only the two fields decay is allowed to change.
pub struct DecayChange {
    pub source: String,
    pub target: String,
    pub confidence: f64,
    pub last_decayed_at: String,
}

// Decay persistence is targeted UPDATEs plus the checkpoint in one transaction.
// It must never go through a whole-graph rewrite: the in-memory Graph carries
// neither embeddings nor source_message_id, so a rewrite silently destroyed
// both every night and left the embedding backfill to pay for it. Writing the
// checkpoint in the same transaction means a failed persist is retried rather
// than skipped until tomorrow.
pub fn persist_decay(
    changes: &[DecayChange],
    date: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    persist_decay_with(&conn, changes, date)
}

pub(crate) fn persist_decay_with(
    conn: &Connection,
    changes: &[DecayChange],
    date: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("BEGIN", ()).await?;
        // Atomicity is the whole promise here, so a failure must roll back
        // rather than return through `?` and leave the transaction open.
        match write_decay_rows(conn, changes, date).await {
            Ok(()) => {
                conn.execute("COMMIT", ()).await?;
                Ok(())
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    })
}

async fn write_decay_rows(
    conn: &Connection,
    changes: &[DecayChange],
    date: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for change in changes {
        conn.execute(
            "UPDATE edges SET confidence = ?1, last_decayed_at = ?2
             WHERE source = ?3 AND target = ?4",
            params![
                round2(change.confidence),
                change.last_decayed_at.as_str(),
                change.source.as_str(),
                change.target.as_str()
            ],
        )
        .await?;
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_decay', ?1)",
        params![date],
    )
    .await?;
    Ok(())
}

#[cfg(test)]
#[path = "decay_store_tests.rs"]
mod tests;
