use std::error::Error;

use turso::{Connection, params};

use super::{open_connection, round2, runtime, text};

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
mod tests {
    use super::{DecayChange, persist_decay_with};
    use crate::memory::db::block_on;
    use crate::memory::db::test_support::harness;
    use crate::memory::db::{int, real, text};
    use turso::Connection;

    // The decay persistence path is the one that silently broke: a parameter
    // count mismatch made every write fail, and the whole-graph rewrite it used
    // would have erased embeddings the moment the count was fixed. This drives
    // the real functions, not hand-written SQL.
    #[test]
    fn decay_persistence_updates_values_and_preserves_embeddings() {
        let conn = block_on(harness());
        block_on(seed_embedded_edge(&conn));

        let changes = vec![DecayChange {
            source: "a".to_string(),
            target: "b".to_string(),
            confidence: 0.42,
            last_decayed_at: "2026-07-20".to_string(),
        }];
        assert!(
            persist_decay_with(&conn, &changes, "2026-07-20").is_ok(),
            "decay persistence succeeds"
        );

        let (confidence, clock, has_embedding) = decayed_edge_state(&conn);
        assert!((confidence - 0.42).abs() < 1e-9);
        assert_eq!(clock, "2026-07-20");
        assert_eq!(
            has_embedding, 1,
            "embedding survives decay — the rewrite path used to drop it"
        );
        assert_eq!(
            decay_checkpoint(&conn),
            "2026-07-20",
            "the checkpoint commits with the values"
        );
        assert_eq!(
            edge_message_id(&conn),
            "msg-7",
            "message attribution survives decay"
        );
    }

    // Atomicity is the promise of the repair: if any statement fails, neither
    // the edge values nor the checkpoint may survive. Dropping the meta table
    // makes the checkpoint write fail after the edge update has been issued.
    #[test]
    fn a_failed_decay_write_commits_nothing() {
        let conn = block_on(harness());
        block_on(seed_embedded_edge(&conn));
        block_on(async {
            let _ = conn.execute("DROP TABLE meta", ()).await;
        });

        let changes = vec![DecayChange {
            source: "a".to_string(),
            target: "b".to_string(),
            confidence: 0.42,
            last_decayed_at: "2026-07-20".to_string(),
        }];
        assert!(
            persist_decay_with(&conn, &changes, "2026-07-20").is_err(),
            "a broken checkpoint write fails the whole persist"
        );

        block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT confidence, last_decayed_at FROM edges WHERE source = 'a'",
                    (),
                )
                .await
            else {
                unreachable!("edge query succeeds")
            };
            let Ok(Some(row)) = rows.next().await else {
                unreachable!("edge still exists")
            };
            let (Ok(conf), Ok(clock)) = (row.get_value(0), row.get_value(1)) else {
                unreachable!("row columns read")
            };
            assert!(
                (real(conf) - 0.9).abs() < 1e-9,
                "confidence rolled back to its original value"
            );
            assert_eq!(text(clock), "", "the clock rolled back too");
        });
    }

    fn decayed_edge_state(conn: &Connection) -> (f64, String, i64) {
        block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT confidence, last_decayed_at, embedding IS NOT NULL FROM edges WHERE source = 'a'",
                    (),
                )
                .await
            else {
                unreachable!("edge query succeeds")
            };
            let Ok(Some(row)) = rows.next().await else {
                unreachable!("edge still exists")
            };
            let (Ok(conf), Ok(clock), Ok(embedded)) =
                (row.get_value(0), row.get_value(1), row.get_value(2))
            else {
                unreachable!("row columns read")
            };
            (real(conf), text(clock), int(embedded))
        })
    }

    fn decay_checkpoint(conn: &Connection) -> String {
        block_on(async {
            let Ok(mut rows) = conn
                .query("SELECT value FROM meta WHERE key = 'last_decay'", ())
                .await
            else {
                return String::new();
            };
            match rows.next().await {
                Ok(Some(row)) => match row.get_value(0) {
                    Ok(value) => text(value),
                    Err(_) => String::new(),
                },
                _ => String::new(),
            }
        })
    }

    fn edge_message_id(conn: &Connection) -> String {
        block_on(async {
            let Ok(mut rows) = conn
                .query("SELECT source_message_id FROM edges WHERE source = 'a'", ())
                .await
            else {
                return String::new();
            };
            match rows.next().await {
                Ok(Some(row)) => match row.get_value(0) {
                    Ok(value) => text(value),
                    Err(_) => String::new(),
                },
                _ => String::new(),
            }
        })
    }

    async fn seed_embedded_edge(conn: &Connection) {
        let _ = conn
            .execute(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, stability, confidence, importance, subject, date, memory_type, origin_session, provenance, source_message_id, embedding)
                 VALUES ('a', 'b', 'belief', 0.5, 0.9, 0.5, 0, '2026-07-01', 'semantic', '', 'legacy', 'msg-7', vector32('[1.0, 0.0]'))",
                (),
            )
            .await;
    }
}
