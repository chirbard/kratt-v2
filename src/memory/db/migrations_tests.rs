use super::super::{block_on, test_support::harness};
use super::{
    add_edge_ceiling_column, backfill_edge_epistemics, clamp_stability_once,
    migrate_entity_bodies_matching_ids, stability_clamp_log_lines,
};
use turso::{Builder, Connection, params};

#[test]
fn ceiling_migration_is_idempotent_and_leaves_existing_edges_null() {
    let conn = block_on(migration_harness());
    block_on(add_edge_ceiling_column(&conn));
    block_on(add_edge_ceiling_column(&conn));
    let result = block_on(async {
        let mut rows = conn
            .query("SELECT ceiling FROM edges WHERE source = 'dream'", ())
            .await?;
        rows.next().await
    });
    assert!(matches!(
        result,
        Ok(Some(row)) if matches!(row.get_value(0), Ok(turso::Value::Null))
    ));
}

#[test]
fn epistemics_backfill_classifies_clamps_and_is_idempotent() {
    let conn = block_on(migration_harness());
    let Ok(first) = block_on(backfill_edge_epistemics(&conn)) else {
        unreachable!("backfill succeeds")
    };
    assert_eq!(first.total, 4);
    assert_eq!(first.dreamed, 1);
    assert_eq!(first.web, 1);
    assert_eq!(first.user, 1);
    assert_eq!(first.legacy, 1);
    assert_eq!(first.clamped, 3);

    block_on(async {
        let _ = conn
            .execute(
                "UPDATE edges SET confidence = 0.1 WHERE source = 'dream'",
                (),
            )
            .await;
    });
    let Ok(second) = block_on(backfill_edge_epistemics(&conn)) else {
        unreachable!("second backfill succeeds")
    };
    assert_eq!(second.total, 0);
    let values = block_on(backfilled_values(&conn));
    assert_eq!(
        values,
        vec![
            ("dream".to_string(), "dreamed".to_string(), 0.1),
            ("news".to_string(), "web-surfaced".to_string(), 0.5),
            ("turn".to_string(), "user-confirmed".to_string(), 0.9),
            ("unknown".to_string(), "legacy".to_string(), 0.5),
        ]
    );
}

#[test]
fn stability_clamp_changes_only_stability_and_reports_actual_rows() {
    let conn = block_on(stability_clamp_harness());
    let before = stability_non_target_values(&conn);
    let marker = crate::timeutil::today_iso();
    let Ok(Some(report)) = block_on(clamp_stability_once(&conn, &marker)) else {
        unreachable!("first clamp succeeds")
    };

    assert_eq!(
        stability_values(&conn),
        vec![
            ("below".to_string(), 0.2),
            ("dream".to_string(), 0.15),
            ("legacy".to_string(), 0.3),
            ("user".to_string(), 0.5),
        ]
    );
    assert_eq!(before, stability_non_target_values(&conn));
    assert_eq!(report.total, 4);
    assert_eq!(report.clamped, 3);
    assert_eq!(report.above_seed, 0);
    assert_eq!(
        stability_clamp_log_lines(&report),
        vec![
            format!("[stability] clamped 3 of 4 edges (marker {marker})"),
            "[stability] dreamed: 1 clamped, median 0.90 -> 0.15".to_string(),
            "[stability] legacy: 1 clamped, median 0.60 -> 0.25".to_string(),
            "[stability] user-confirmed: 1 clamped, median 1.00 -> 0.50".to_string(),
            "[stability] edges above seed after clamp: 0".to_string(),
        ]
    );
}

#[test]
fn stability_clamp_marker_preserves_later_growth() {
    let conn = block_on(stability_clamp_harness());
    let marker = crate::timeutil::today_iso();
    assert!(matches!(
        block_on(clamp_stability_once(&conn, &marker)),
        Ok(Some(_))
    ));
    block_on(async {
        let _ = conn
            .execute(
                "UPDATE edges SET stability = 0.8 WHERE source = 'legacy'",
                (),
            )
            .await;
    });

    assert!(matches!(
        block_on(clamp_stability_once(&conn, &marker)),
        Ok(None)
    ));
    assert_eq!(edge_stability(&conn, "legacy"), 0.8);
    assert_eq!(
        block_on(super::meta_value(&conn, super::STABILITY_CLAMP_MARKER)).unwrap_or_default(),
        Some(marker)
    );
}

#[test]
fn failed_stability_clamp_rolls_back_values_and_marker() {
    let conn = block_on(stability_clamp_harness());
    block_on(async {
        let _ = conn.execute("DROP TABLE meta", ()).await;
        let _ = conn
            .execute(
                "CREATE TABLE meta (
                        key TEXT PRIMARY KEY CHECK (key != 'stability_clamp'),
                        value TEXT NOT NULL DEFAULT ''
                    )",
                (),
            )
            .await;
    });
    let marker = crate::timeutil::today_iso();
    assert!(block_on(clamp_stability_once(&conn, &marker)).is_err());
    assert_eq!(edge_stability(&conn, "legacy"), 1.0);
    assert_eq!(
        block_on(super::meta_value(&conn, super::STABILITY_CLAMP_MARKER)).unwrap_or_default(),
        None
    );

    block_on(async {
        let _ = conn.execute("DROP TABLE meta", ()).await;
        let _ = conn
            .execute(
                "CREATE TABLE meta (
                        key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT ''
                    )",
                (),
            )
            .await;
    });
    assert!(matches!(
        block_on(clamp_stability_once(&conn, &marker)),
        Ok(Some(_))
    ));
    assert_eq!(edge_stability(&conn, "legacy"), 0.3);
}

#[test]
fn entity_body_id_migration_is_strict_and_idempotent() {
    let conn = block_on(harness());
    block_on(async {
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type, body) VALUES
                    ('some-name', 'entity', ' Some_Name '),
                    ('another-name', 'entity', 'ANOTHER-NAME'),
                    ('fact-name', 'fact', 'fact name'),
                    ('kept', 'entity', 'different')",
                (),
            )
            .await;
    });
    let first = block_on(migrate_entity_bodies_matching_ids(&conn)).unwrap_or_default();
    assert_eq!(first, vec!["another-name", "some-name"]);
    assert_eq!(node_body(&conn, "some-name"), "");
    assert_eq!(node_body(&conn, "another-name"), "");
    assert_eq!(node_body(&conn, "fact-name"), "fact name");
    assert_eq!(node_body(&conn, "kept"), "different");
    let second = block_on(migrate_entity_bodies_matching_ids(&conn)).unwrap_or_default();
    assert!(second.is_empty());
}

fn node_body(conn: &Connection, node_id: &str) -> String {
    block_on(async {
        let Ok(mut rows) = conn
            .query("SELECT body FROM nodes WHERE id = ?1", params![node_id])
            .await
        else {
            return String::new();
        };
        match rows.next().await {
            Ok(Some(row)) => row.get::<String>(0).unwrap_or_default(),
            _ => String::new(),
        }
    })
}

async fn migration_harness() -> Connection {
    let Ok(database) = Builder::new_local(":memory:").build().await else {
        unreachable!("in-memory database builds")
    };
    let Ok(conn) = database.connect() else {
        unreachable!("in-memory database connects")
    };
    let _ = conn
        .execute(
            "CREATE TABLE edges (
                source TEXT NOT NULL, target TEXT NOT NULL, confidence REAL NOT NULL,
                origin_session TEXT NOT NULL DEFAULT '',
                provenance TEXT NOT NULL DEFAULT '',
                last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target)
            )",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "INSERT INTO edges (source, target, confidence, origin_session) VALUES
                ('dream', 'x', 0.9, 'dream:1'),
                ('news', 'x', 0.9, 'news:2026-07-16'),
                ('turn', 'x', 0.9, '2026-07-16T12-00-00'),
                ('unknown', 'x', 0.9, '')",
            (),
        )
        .await;
    conn
}

async fn stability_clamp_harness() -> Connection {
    let conn = harness().await;
    let _ = conn
        .execute(
            "CREATE TABLE meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT ''
                )",
            (),
        )
        .await;
    let fixture_date = crate::timeutil::today_iso();
    let _ = conn
        .execute(
            "INSERT INTO edges (
                    source, target, content, stability, confidence, importance,
                    subject, date, occurred_at, expired_at, source_message_id,
                    memory_type, origin_session, provenance, ceiling, embedding,
                    last_decayed_at
                ) VALUES
                ('legacy', 'x', 'legacy belief', 1.0, 0.61, 0.7, 1, ?1, ?1, NULL,
                 'msg-legacy', 'semantic', 'session-a', 'legacy', 0.45,
                 vector32('[1.0, 0.0]'), ?1),
                ('user', 'x', 'user belief', 1.0, 0.91, 0.8, 0, ?1, NULL, NULL,
                 'msg-user', 'semantic', 'session-b', 'user-confirmed', NULL,
                 vector32('[0.0, 1.0]'), ?1),
                ('dream', 'x', 'dream belief', 0.9, 0.21, 0.2, 0, ?1, NULL, ?1,
                 'msg-dream', 'semantic', 'dream:1', 'dreamed', 0.25,
                 vector32('[0.5, 0.5]'), ?1),
                ('below', 'x', 'small belief', 0.2, 0.41, 0.3, 0, ?1, NULL, NULL,
                 'msg-below', 'semantic', '', 'legacy', NULL,
                 vector32('[0.25, 0.75]'), ?1)",
            params![fixture_date],
        )
        .await;
    conn
}

fn stability_values(conn: &Connection) -> Vec<(String, f64)> {
    block_on(async {
        let Ok(mut rows) = conn
            .query("SELECT source, stability FROM edges ORDER BY source", ())
            .await
        else {
            unreachable!("stability query succeeds")
        };
        let mut values = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let (Ok(source), Ok(stability)) = (row.get_value(0), row.get_value(1)) else {
                unreachable!("stability values read")
            };
            values.push((super::text(source), super::real(stability)));
        }
        values
    })
}

fn edge_stability(conn: &Connection, source: &str) -> f64 {
    let values = stability_values(conn);
    for (edge_source, stability) in values {
        if edge_source == source {
            return stability;
        }
    }
    -1.0
}

fn stability_non_target_values(conn: &Connection) -> Vec<Vec<String>> {
    block_on(async {
        let Ok(mut rows) = conn
            .query(
                "SELECT source, target, content, printf('%.17g', confidence),
                     printf('%.17g', importance), subject, date,
                     coalesce(occurred_at, '<null>'), coalesce(expired_at, '<null>'),
                     source_message_id, memory_type, origin_session, provenance,
                     coalesce(printf('%.17g', ceiling), '<null>'), hex(embedding),
                     last_decayed_at
                     FROM edges ORDER BY source, target",
                (),
            )
            .await
        else {
            unreachable!("non-target query succeeds")
        };
        let mut snapshots = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let mut values = Vec::new();
            for index in 0..16 {
                let Ok(value) = row.get_value(index) else {
                    unreachable!("non-target value reads")
                };
                values.push(match value {
                    turso::Value::Text(text) => text,
                    turso::Value::Integer(number) => number.to_string(),
                    _ => format!("{value:?}"),
                });
            }
            snapshots.push(values);
        }
        snapshots
    })
}

async fn backfilled_values(conn: &Connection) -> Vec<(String, String, f64)> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source, provenance, confidence FROM edges ORDER BY source",
            (),
        )
        .await
    else {
        unreachable!("query succeeds")
    };
    let mut values = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let (Ok(source), Ok(provenance), Ok(confidence)) =
            (row.get_value(0), row.get_value(1), row.get_value(2))
        else {
            unreachable!("backfill values read")
        };
        values.push((
            super::text(source),
            super::text(provenance),
            super::real(confidence),
        ));
    }
    values
}
