    use super::*;
    use crate::api::EmbeddingClient;
    use turso::{Builder, params};

    struct TestEmbed;

    impl EmbeddingClient for TestEmbed {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
            let mut vectors = Vec::new();
            for text in texts {
                if text.contains("near") {
                    vectors.push(vec![1.0, 0.0]);
                } else if text.contains("mid") {
                    vectors.push(vec![0.7, 0.714_142_842_9]);
                } else {
                    vectors.push(vec![0.0, 1.0]);
                }
            }
            Ok(vectors)
        }
    }

    #[test]
    fn threshold_boundary_is_kept() {
        assert_eq!(novelty_from_similarity(Some(0.6)), NEWS_NOVELTY_MIN);
        assert!(novelty_from_similarity(Some(0.61)) < NEWS_NOVELTY_MIN);
    }

    #[test]
    fn mid_similarity_is_discarded_by_novelty_gate() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        set_known_edge(&conn, "legacy", 0.4, 0.4);
        let count = process_interest_results(
            &conn,
            &TestEmbed,
            &test_interest(),
            &[news_item("mid update")],
        );
        assert_eq!(count, 0);
        let Ok(Some(edge)) = db::load_edge(&conn, "known", "fact") else {
            unreachable!("known edge exists")
        };
        assert_eq!(edge.provenance, "legacy");
        assert_eq!(edge.confidence, 0.4);
        assert_eq!(edge.stability, 0.4);
    }

    #[test]
    fn nearest_live_edges_return_identity_in_similarity_order() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        db::block_on(async {
            let _ = conn.execute("INSERT INTO edges (source, target, provenance, confidence, embedding) VALUES ('second', 'claim', 'web-surfaced', 0.4, vector32('[0.8,0.6]'))", ()).await;
        });
        let edges = db::nearest_live_edges(&conn, &[1.0, 0.0], 2).unwrap_or_default();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].source, "known");
        assert_eq!(edges[0].target, "fact");
        assert_eq!(edges[0].provenance, "legacy");
        assert_eq!(edges[0].confidence, 0.0);
        assert!(edges[0].similarity > edges[1].similarity);
        assert_eq!(edges[1].source, "second");
        assert_eq!(edges[1].target, "claim");
    }

    async fn test_connection() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        conn
    }

    fn seed_test_graph(conn: &Connection) {
        db::block_on(async {
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))", ()).await;
            let _ = conn.execute("CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))", ()).await;
            let _ = conn.execute("CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))", ()).await;
            let _ = conn.execute("CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)", ()).await;
            let _ = conn
                .execute(
                    "INSERT INTO nodes (id, node_type) VALUES ('rust', 'interest')",
                    (),
                )
                .await;
            let _ = conn.execute("INSERT INTO edges (source, target, embedding) VALUES ('known', 'fact', vector32('[1.0,0.0]'))", ()).await;
        });
    }

    fn set_known_edge(conn: &Connection, provenance: &str, confidence: f64, stability: f64) {
        db::block_on(async {
            let _ = conn
                .execute(
                    "UPDATE edges SET provenance = ?1, confidence = ?2, stability = ?3
                     WHERE source = 'known' AND target = 'fact'",
                    params![provenance, confidence, stability],
                )
                .await;
        });
    }

    fn test_interest() -> Interest {
        Interest {
            node_id: "rust".to_string(),
            topic: "Rust".to_string(),
            origin: "user".to_string(),
            created_at: "2026-07-14T00:00:00Z".to_string(),
            strength: 1.0,
        }
    }

    fn news_item(title: &str) -> NewsItem {
        NewsItem {
            title: title.to_string(),
            url: format!("https://example.com/{title}"),
            snippet: "details".to_string(),
        }
    }
