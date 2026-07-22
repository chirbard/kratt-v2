    use super::{
        HygieneReport, collapse_separators, importance_score, recency_score, run_hygiene,
        run_importance_pass,
    };
    use crate::memory::db;
    use turso::{Builder, Connection, params};

    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL DEFAULT '', embedding BLOB)";
    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    // record_events_in_tx also writes actor views and positions.
    const VIEWS_DDL: &str = "CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))";
    const POSITIONS_DDL: &str = "CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))";
    const HISTORY_DDL: &str = "CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)";

    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        for ddl in [
            NODES_DDL,
            EDGES_DDL,
            EVENTS_DDL,
            VIEWS_DDL,
            POSITIONS_DDL,
            HISTORY_DDL,
        ] {
            let _ = conn.execute(ddl, ()).await;
        }
        conn
    }

    async fn add_node(conn: &Connection, node_id: &str, node_type: &str) {
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type) VALUES (?1, ?2)",
                params![node_id, node_type],
            )
            .await;
    }

    async fn add_edge(conn: &Connection, source: &str, target: &str, expired_at: Option<&str>) {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, confidence, stability, importance, expired_at) VALUES (?1, ?2, 0.5, 0.5, 0.5, ?3)",
                params![source, target, expired_at],
            )
            .await;
    }

    async fn expired_at_of(conn: &Connection, source: &str, target: &str) -> Option<String> {
        let Ok(mut rows) = conn
            .query(
                "SELECT expired_at FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("expiry query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(value) = row.get_value(0) else {
            unreachable!("column reads")
        };
        match value {
            turso::Value::Text(text) => Some(text),
            _ => None,
        }
    }

    async fn edge_weights(conn: &Connection, source: &str, target: &str) -> (f64, f64, f64) {
        let Ok(mut rows) = conn
            .query(
                "SELECT confidence, stability, importance FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("weights query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(confidence) = row.get_value(0) else {
            unreachable!("column reads")
        };
        let Ok(stability) = row.get_value(1) else {
            unreachable!("column reads")
        };
        let Ok(importance) = row.get_value(2) else {
            unreachable!("column reads")
        };
        (
            db::real(confidence),
            db::real(stability),
            db::real(importance),
        )
    }

    async fn subject_of(conn: &Connection, source: &str, target: &str) -> (i64, f64) {
        let Ok(mut rows) = conn
            .query(
                "SELECT subject, importance FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("subject query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(subject) = row.get_value(0) else {
            unreachable!("column reads")
        };
        let Ok(importance) = row.get_value(1) else {
            unreachable!("column reads")
        };
        (db::int(subject), db::real(importance))
    }

    fn hygiene(conn: &Connection) -> HygieneReport {
        match run_hygiene(conn) {
            Ok(report) => report,
            Err(err) => panic!("hygiene runs: {err}"),
        }
    }

    #[test]
    fn importance_blend_and_recency_follow_the_declared_formula() {
        assert_eq!(importance_score(0.0, 0.0, 0.0), 0.0);
        assert_eq!(importance_score(1.0, 1.0, 1.0), 1.0);
        assert_eq!(importance_score(1.0, 0.0, 0.0), 0.4);
        assert!((recency_score(14.0) - (-1.0_f64).exp()).abs() < 1e-9);
        assert_eq!(importance_score(0.0, recency_score(0.0), 0.0), 0.4);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn importance_pass_combines_interest_recency_and_missing_embeddings() {
        let conn = db::block_on(harness());
        db::block_on(async {
            for node_id in [
                "interest-rust",
                "near",
                "near-target",
                "far",
                "far-target",
                "plain",
                "plain-target",
            ] {
                add_node(&conn, node_id, "entity").await;
            }
            let _ = conn
                .execute(
                    "UPDATE nodes SET node_type = 'interest', body = 'Rust', origin = 'user',
                         embedding = vector32('[1.0,0.0]') WHERE id = 'interest-rust'",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO edges (source, target, embedding)
                     VALUES ('near', 'near-target', vector32('[1.0,0.0]')),
                            ('far', 'far-target', vector32('[0.0,1.0]')),
                            ('plain', 'plain-target', NULL)",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO events (ts, event_type, source, target)
                     VALUES (datetime('now'), 'created', 'near', 'near-target'),
                            (datetime('now', '-60 days'), 'created', 'far', 'far-target'),
                            (datetime('now'), 'injected', 'plain', 'plain-target')",
                    (),
                )
                .await;
            let count = run_importance_pass(&conn).await.unwrap_or_default();
            assert_eq!(count, 3);
        });
        let near = db::block_on(edge_weights(&conn, "near", "near-target")).2;
        let far = db::block_on(edge_weights(&conn, "far", "far-target")).2;
        let plain = db::block_on(edge_weights(&conn, "plain", "plain-target")).2;
        assert!(near > plain);
        assert!(plain > far);
        assert_eq!(plain, 0.4);
        assert_eq!(near, db::round2(near));
        assert_eq!(far, db::round2(far));
    }

    #[test]
    fn orphan_edges_are_expired_and_intact_edges_are_left_alone() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        // Missing target node.
        db::block_on(add_edge(&conn, "alice", "ghost", None));
        // Missing source node.
        db::block_on(add_edge(&conn, "phantom", "berlin", None));
        // Already expired orphan — must not be written again.
        db::block_on(add_edge(&conn, "wraith", "berlin", Some("2020-01-01")));

        let report = hygiene(&conn);
        assert_eq!(report.expired_orphan_edges, 2);
        assert!(db::block_on(expired_at_of(&conn, "alice", "ghost")).is_some());
        assert!(db::block_on(expired_at_of(&conn, "phantom", "berlin")).is_some());
        assert!(db::block_on(expired_at_of(&conn, "alice", "berlin")).is_none());
        assert_eq!(
            db::block_on(expired_at_of(&conn, "wraith", "berlin")),
            Some("2020-01-01".to_string())
        );
    }

    #[test]
    fn live_loop_edge_survives_hygiene_pass() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "ask-about-backup", "loop"));
        db::block_on(add_node(&conn, "backup", "entity"));
        db::block_on(add_edge(&conn, "ask-about-backup", "backup", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9, stability = 0.9, importance = 1.0,
             provenance = 'user-confirmed' WHERE source = 'ask-about-backup'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.expired_orphan_edges, 0);
        assert!(db::block_on(expired_at_of(&conn, "ask-about-backup", "backup")).is_none());
        assert_eq!(
            db::block_on(edge_weights(&conn, "ask-about-backup", "backup")),
            (0.9, 0.9, 0.0)
        );
    }

    #[test]
    fn out_of_range_weights_are_clamped_into_the_unit_interval() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_node(&conn, "carol", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        db::block_on(add_edge(&conn, "carol", "berlin", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 1.4, stability = -0.2 WHERE source = 'alice'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.clamped_edges, 1);
        assert_eq!(
            db::block_on(edge_weights(&conn, "alice", "berlin")),
            (0.5, 0.0, 0.0)
        );
        // Confidence and stability were already in range; salience is recomputed.
        assert_eq!(
            db::block_on(edge_weights(&conn, "carol", "berlin")),
            (0.5, 0.5, 0.0)
        );
    }

    #[test]
    fn derived_edge_is_clamped_to_its_own_ceiling() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "summary", "entity"));
        db::block_on(add_node(&conn, "topic", "entity"));
        db::block_on(add_edge(&conn, "summary", "topic", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9, provenance = 'synthesized', ceiling = 0.72
             WHERE source = 'summary'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.clamped_edges, 1);
        assert_eq!(
            db::block_on(edge_weights(&conn, "summary", "topic")).0,
            0.72
        );
    }

    #[test]
    fn unambiguous_missing_subject_edge_is_repaired() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 1);
        assert!(report.ambiguous_subject_facts.is_empty());
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (1, 0.0)
        );
    }

    #[test]
    fn fact_with_two_entity_candidates_is_reported_not_fixed() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));
        db::block_on(add_edge(&conn, "fact-moved", "berlin", None));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 0);
        assert_eq!(
            report.ambiguous_subject_facts,
            vec!["fact-moved".to_string()]
        );
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (0, 0.0)
        );
    }

    #[test]
    fn fact_that_already_has_a_subject_edge_is_untouched() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));
        let _ = db::block_on(conn.execute("UPDATE edges SET subject = 1", ()));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 0);
        assert!(report.ambiguous_subject_facts.is_empty());
        // Subject state stays untouched; nightly salience is still recomputed.
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (1, 0.0)
        );
    }

    #[test]
    fn separator_variants_are_reported_as_duplicate_pairs() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "tinyllama-1.1b", "entity"));
        db::block_on(add_node(&conn, "tinyllama-1-1b", "entity"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));

        let report = hygiene(&conn);
        assert_eq!(
            report.duplicate_pairs,
            vec![("tinyllama-1-1b".to_string(), "tinyllama-1.1b".to_string())]
        );
    }

    #[test]
    fn orphan_nodes_are_reported_and_notes_are_excluded() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        // Zero edges at all.
        db::block_on(add_node(&conn, "lonely", "entity"));
        // Only an expired edge — invisible to retrieval.
        db::block_on(add_node(&conn, "faded", "entity"));
        db::block_on(add_edge(&conn, "faded", "berlin", Some("2020-01-01")));
        db::block_on(add_node(&conn, "scratch", "note"));

        let report = hygiene(&conn);
        assert_eq!(report.orphan_node_count, 2);
        assert!(report.orphan_node_examples.contains(&"lonely".to_string()));
        assert!(report.orphan_node_examples.contains(&"faded".to_string()));
        assert!(!report.orphan_node_examples.contains(&"scratch".to_string()));
    }

    #[test]
    fn sessionless_episodic_edges_are_counted() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET memory_type = 'episodic', origin_session = ''",
            (),
        ));

        assert_eq!(hygiene(&conn).episodic_without_session, 1);
    }

    #[test]
    fn a_clean_graph_reports_nothing() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));

        let report = hygiene(&conn);
        assert!(report.is_clean());
        assert!(report.summary().is_empty());
    }

    #[test]
    fn summary_names_each_non_zero_finding() {
        let report = HygieneReport {
            expired_orphan_edges: 195,
            clamped_edges: 1,
            ..HygieneReport::default()
        };
        assert_eq!(
            report.summary(),
            "expired 195 orphan edges, clamped 1 edge weight"
        );
    }

    #[test]
    fn separator_collapse_ignores_dots_hyphens_and_underscores() {
        assert_eq!(collapse_separators("tinyllama-1.1b"), "tinyllama11b");
        assert_eq!(collapse_separators("claude_pro"), "claudepro");
        assert_eq!(collapse_separators("alice"), "alice");
    }
