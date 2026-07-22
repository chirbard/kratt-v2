    use super::{
        consecutive_failures_async, finish_pass_run_async, last_successful_pass_async,
        pass_runs_in_window_async, start_pass_run_async,
    };
    use crate::memory::db::block_on;
    use turso::{Builder, Connection};

    async fn pass_run_harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let Ok(()) = super::super::create_pass_run_table(&conn).await else {
            unreachable!("pass run table creates")
        };
        conn
    }

    #[test]
    fn pass_run_records_failure_reason_and_items() {
        let conn = block_on(pass_run_harness());
        let Ok(run_id) = block_on(start_pass_run_async(&conn, "dream")) else {
            unreachable!("pass starts")
        };
        assert!(
            block_on(finish_pass_run_async(
                &conn,
                run_id,
                "failed",
                "context_length",
                0,
            ))
            .is_ok()
        );
        let runs = block_on(pass_runs_in_window_async(&conn, "dream", "", "z")).unwrap_or_default();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].outcome, "failed");
        assert_eq!(runs[0].reason_class, "context_length");
    }

    #[test]
    fn pass_failure_streak_resets_after_success() {
        let conn = block_on(pass_run_harness());
        block_on(async {
            let _ = conn.execute(
                "INSERT INTO pass_runs (pass_type, started_at, finished_at, outcome, reason_class) VALUES
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-3 days'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-3 days'), 'ok', 'none'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 days'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 days'), 'failed', 'provider_error'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 day'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 day'), 'failed', 'provider_error'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'failed', 'context_length')",
                (),
            ).await;
        });
        assert_eq!(
            block_on(consecutive_failures_async(&conn, "dream")).unwrap_or_default(),
            3
        );
        let last = block_on(last_successful_pass_async(&conn, "dream")).unwrap_or_default();
        assert!(last.is_some());
        let Ok(run_id) = block_on(start_pass_run_async(&conn, "dream")) else {
            unreachable!("success starts")
        };
        assert!(block_on(finish_pass_run_async(&conn, run_id, "ok", "none", 4)).is_ok());
        assert_eq!(
            block_on(consecutive_failures_async(&conn, "dream")).unwrap_or_default(),
            0
        );
        let latest = block_on(last_successful_pass_async(&conn, "dream")).unwrap_or_default();
        let items_added = match latest {
            Some(run) => Some(run.items_added),
            None => None,
        };
        assert_eq!(items_added, Some(4));
    }
