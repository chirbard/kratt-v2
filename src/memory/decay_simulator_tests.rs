    use std::path::PathBuf;

    use turso::Builder;

    use super::{SimulationEdge, date_to_days, days_to_date, run_to_string, today_prune_eligible};
    use crate::memory::graph::Edge;

    #[test]
    fn clamp_uses_legacy_seed_maximum() {
        let edge = test_simulation_edge(1.0);
        let clamped = super::clamped_edges(&[edge]);
        assert_eq!(clamped[0].edge.stability, 0.3);
    }

    #[test]
    fn today_prune_count_matches_current_guard() {
        let mut edge = test_simulation_edge(0.6);
        edge.edge.importance = 0.5;
        edge.edge.confidence = 0.6;
        assert!(today_prune_eligible(&edge));
        edge.edge.stability = 0.7;
        assert!(today_prune_eligible(&edge));
        edge.edge.stability = 0.6;
        edge.edge.confidence = 0.7;
        assert!(!today_prune_eligible(&edge));
        edge.edge.confidence = 0.6;
        edge.edge.importance = 1.0;
        assert!(today_prune_eligible(&edge));
        edge.edge.subject = true;
        assert!(!today_prune_eligible(&edge));
        edge.edge.subject = false;
        edge.source_node_type = "loop".to_string();
        assert!(!today_prune_eligible(&edge));
    }

    #[test]
    fn date_round_trip_is_stable() {
        for date in ["2025-01-01", "2026-07-20", "2030-12-31"] {
            let Some(days) = date_to_days(date) else {
                panic!("valid date");
            };
            assert_eq!(days_to_date(days), date);
        }
    }

    #[test]
    fn simulator_is_read_only_and_produces_report() {
        let path = fixture_path();
        create_fixture(&path);
        let before = std::fs::read(&path).unwrap_or_default();
        let report = run_to_string(&path, None).unwrap_or_default();
        let after = std::fs::read(&path).unwrap_or_default();
        assert_eq!(before, after, "snapshot bytes changed");
        assert!(report.contains("STABILITY_DOUBLING 0.15"));
        assert!(report.contains("prune-eligible"));
        assert!(report.contains("rust-fact -> immune-response-node"));
        cleanup_fixture(&path);
    }

    fn fixture_path() -> PathBuf {
        std::env::temp_dir().join(format!("kratt-decay-simulator-{}.db", std::process::id()))
    }

    fn create_fixture(path: &std::path::Path) {
        cleanup_fixture(path);
        let Ok(runtime) = tokio::runtime::Runtime::new() else {
            panic!("runtime");
        };
        runtime.block_on(async {
            let Some(path_text) = path.to_str() else {
                panic!("path");
            };
            let Ok(database) = Builder::new_local(path_text)
                .experimental_index_method(true)
                .build()
                .await
            else {
                panic!("database");
            };
            let Ok(conn) = database.connect() else {
                panic!("connection");
            };
            create_fixture_schema(&conn).await;
            insert_fixture_rows(&conn).await;
        });
    }

    async fn create_fixture_schema(conn: &turso::Connection) {
        conn.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', embedding BLOB, PRIMARY KEY (source, target))",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '')",
            (),
        )
        .await
        .unwrap_or(0);
    }

    async fn insert_fixture_rows(conn: &turso::Connection) {
        conn.execute(
            "INSERT INTO nodes (id, node_type) VALUES ('rust-fact', 'fact'), ('immune-response-node', 'entity')",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "INSERT INTO edges (source, target, content, stability, confidence, importance, date, last_decayed_at, embedding) VALUES ('rust-fact', 'immune-response-node', 'bad analogy', 1.0, 0.6, 0.4, '2026-07-20', '2026-07-20', vector32('[1.0,0.0]'))",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "INSERT INTO events (ts, event_type, source, target) VALUES ('2026-07-20T12:00:00', 'injected', 'rust-fact', 'immune-response-node')",
            (),
        )
        .await
        .unwrap_or(0);
    }

    fn cleanup_fixture(path: &std::path::Path) {
        for suffix in ["", "-wal", "-shm"] {
            let candidate = PathBuf::from(format!("{}{suffix}", path.display()));
            let _ = std::fs::remove_file(candidate);
        }
    }

    fn test_simulation_edge(stability: f64) -> SimulationEdge {
        SimulationEdge {
            edge: Edge {
                source: "rust-fact".to_string(),
                target: "immune-response-node".to_string(),
                content: "bad analogy".to_string(),
                stability,
                confidence: 0.6,
                importance: 0.4,
                date: "2026-07-20".to_string(),
                subject: false,
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
                ceiling: None,
                last_decayed_at: "2026-07-20".to_string(),
            },
            embedding: Some(vec![1.0, 0.0]),
            source_node_type: "fact".to_string(),
            target_node_type: "entity".to_string(),
        }
    }
