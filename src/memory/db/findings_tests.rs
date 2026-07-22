    use super::{
        DEFAULT_PING_MUTED_UNTIL, count_findings_in_windows, divert_ping_topic, insert_finding,
        load_ping_policy, mark_finding_surfaced, resume_pings, set_ping_mute, unsurfaced_findings,
    };
    use crate::memory::db::block_on;
    use crate::memory::db::test_support::{EVENTS_DDL, FINDINGS_DDL, PING_POLICY_DDL, harness};
    use crate::web::NewsItem;
    use turso::{Connection, params};

    #[test]
    fn ping_policy_defaults_muted_and_expired_controls_rearm() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(PING_POLICY_DDL, ()).await;
        });
        let now = "2026-07-14T12:00:00";
        let initial = load_ping_policy(&conn, now).unwrap_or_default();
        assert_eq!(initial.muted_until, DEFAULT_PING_MUTED_UNTIL);

        assert!(resume_pings(&conn).is_ok());
        assert_eq!(
            load_ping_policy(&conn, now).unwrap_or_default().muted_until,
            ""
        );
        assert!(set_ping_mute(&conn, "2026-07-15T00:00:00").is_ok());
        let muted = load_ping_policy(&conn, now).unwrap_or_default();
        assert!(muted.muted_until.as_str() > now);
        assert!(divert_ping_topic(&conn, "Rust", "2026-07-15T00:00:00").is_ok());
        assert!(divert_ping_topic(&conn, "Old", "2026-07-14T01:00:00").is_ok());
        let policy = load_ping_policy(&conn, now).unwrap_or_default();
        assert_eq!(
            policy.diverts,
            vec![("Rust".to_string(), "2026-07-15T00:00:00".to_string())]
        );
    }

    #[test]
    fn findings_queue_filters_status_and_counts_window() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(FINDINGS_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')", ()).await;
        });
        let item = NewsItem {
            title: "A new compiler".to_string(),
            url: "https://example.com/compiler".to_string(),
            snippet: "Details".to_string(),
        };
        let Ok(finding_id) = insert_finding(&conn, "Rust", &item, 0.8) else {
            unreachable!("finding inserts")
        };
        let candidates = unsurfaced_findings(&conn, 10).unwrap_or_default();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].finding_id, finding_id);
        block_on(async {
            let _ = conn
                .execute(
                    "UPDATE findings SET found_at = '2026-07-14T02:00:00Z' WHERE id = ?1",
                    params![finding_id],
                )
                .await;
        });
        assert!(mark_finding_surfaced(&conn, finding_id, "tg-2026-07-14", "ping-1").is_ok());
        assert!(
            unsurfaced_findings(&conn, 10)
                .unwrap_or_default()
                .is_empty()
        );
        let from = "2026-07-14T00:00:00Z";
        let until = "2026-07-15T00:00:00Z";
        let count =
            block_on(count_findings_in_windows(&conn, from, until, from, until)).unwrap_or(0);
        assert_eq!(count, 1);
        assert_eq!(count_surfaced_events(&conn, "ping-1"), 1);
    }

    fn count_surfaced_events(conn: &Connection, turn_id: &str) -> i64 {
        block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT COUNT(*) FROM events WHERE turn_id = ?1 AND event_type = 'surfaced'",
                    params![turn_id],
                )
                .await
            else {
                return 0;
            };
            match rows.next().await {
                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
                _ => 0,
            }
        })
    }
