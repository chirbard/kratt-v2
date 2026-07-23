    use super::{Connection, Edge, load_injected_edges};
    use crate::memory::db;
    use turso::{Builder, params};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";

    // Fresh in-memory graph: three edges, one of them already expired.
    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let _ = conn.execute(EDGES_DDL, ()).await;
        let _ = conn.execute(EVENTS_DDL, ()).await;
        insert_edge(&conn, "alice", "berlin", "2020-01-01", None).await;
        insert_edge(&conn, "carol", "delhi", "2020-01-01", Some("2024-05-05")).await;
        insert_edge(&conn, "erin", "faro", "2020-01-01", None).await;
        conn
    }

    async fn insert_edge(
        conn: &Connection,
        source: &str,
        target: &str,
        date: &str,
        expired_at: Option<&str>,
    ) {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, date, expired_at) VALUES (?1, ?2, 'x', ?3, ?4)",
                params![source, target, date, expired_at],
            )
            .await;
    }

    async fn insert_injected(conn: &Connection, turn_id: &str, source: &str, target: &str) {
        let _ = conn
            .execute(
                "INSERT INTO events (turn_id, event_type, source, target) VALUES (?1, 'injected', ?2, ?3)",
                params![turn_id, source, target],
            )
            .await;
    }

    async fn edge_date(conn: &Connection, source: &str, target: &str) -> String {
        let Ok(mut rows) = conn
            .query(
                "SELECT date FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("date query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(value) = row.get_value(0) else {
            unreachable!("date column reads")
        };
        db::text(value)
    }

    fn injected_for(conn: &Connection, turn_id: &str) -> Vec<Edge> {
        let Ok(edges) = db::block_on(load_injected_edges(conn, turn_id)) else {
            unreachable!("loader succeeds")
        };
        edges
    }

    #[test]
    fn loader_returns_deduplicated_live_edges_for_the_turn_only() {
        let conn = db::block_on(harness());
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        // Same edge injected twice in one turn — retrieval hit it more than once.
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        db::block_on(insert_injected(&conn, "turn-1", "carol", "delhi"));
        db::block_on(insert_injected(&conn, "turn-2", "erin", "faro"));

        let injected = injected_for(&conn, "turn-1");
        assert_eq!(injected.len(), 1, "expired edge and duplicate are excluded");
        assert_eq!(injected[0].source, "alice");
        assert_eq!(injected[0].target, "berlin");

        assert_eq!(injected_for(&conn, "turn-2").len(), 1);
        assert!(injected_for(&conn, "turn-unknown").is_empty());
    }

    #[test]
    fn refresh_sets_today_on_injected_edges_and_spares_the_rest() {
        let conn = db::block_on(harness());
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        db::block_on(insert_injected(&conn, "turn-2", "erin", "faro"));

        let today = crate::prompts::today_iso();
        let injected = injected_for(&conn, "turn-1");
        let Ok(refreshed) = db::block_on(db::touch_edge_dates(&conn, &injected, &today)) else {
            unreachable!("refresh succeeds")
        };
        assert_eq!(refreshed, 1);

        assert_eq!(db::block_on(edge_date(&conn, "alice", "berlin")), today);
        // Injected in a different turn — its clock keeps running.
        assert_eq!(db::block_on(edge_date(&conn, "erin", "faro")), "2020-01-01");
    }

    #[test]
    fn refresh_never_touches_an_expired_edge() {
        let conn = db::block_on(harness());
        let expired = Edge {
            source: "carol".to_string(),
            target: "delhi".to_string(),
            content: "x".to_string(),
            stability: 0.0,
            confidence: 0.0,
            importance: 0.0,
            date: "2020-01-01".to_string(),
            subject: false,
            occurred_at: None,
            expired_at: Some("2024-05-05".to_string()),
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        };
        let today = crate::prompts::today_iso();
        let Ok(refreshed) = db::block_on(db::touch_edge_dates(&conn, &[expired], &today)) else {
            unreachable!("refresh succeeds")
        };
        assert_eq!(refreshed, 0);
        assert_eq!(
            db::block_on(edge_date(&conn, "carol", "delhi")),
            "2020-01-01"
        );
    }

    const SESSIONS_DDL: &str = "CREATE TABLE sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')";
    const MESSAGES_DDL: &str = "CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '')";
    const PASS_RUNS_DDL: &str = "CREATE TABLE pass_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, pass_type TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT 'failed', reason_class TEXT NOT NULL DEFAULT 'interrupted', items_added INTEGER NOT NULL DEFAULT 0)";
    const FINDINGS_DDL: &str = "CREATE TABLE findings (id INTEGER PRIMARY KEY, interest_topic TEXT, title TEXT, url TEXT, snippet TEXT, novelty REAL, found_at TEXT, status TEXT, surfaced_at TEXT)";

    const YESTERDAY: &str = "2026-07-10";
    const TODAY: &str = "2026-07-11";

    // Bare events+sessions harness for the digest queries.
    async fn digest_harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let _ = conn.execute(EVENTS_DDL, ()).await;
        let _ = conn.execute(SESSIONS_DDL, ()).await;
        let _ = conn.execute(MESSAGES_DDL, ()).await;
        let _ = conn.execute(PASS_RUNS_DDL, ()).await;
        let _ = conn.execute(EDGES_DDL, ()).await;
        let _ = conn.execute(NODES_DDL, ()).await;
        let _ = conn.execute(FINDINGS_DDL, ()).await;
        conn
    }

    async fn insert_digest_loop(
        conn: &Connection,
        loop_id: &str,
        body: &str,
        due: &str,
        expired_at: Option<&str>,
    ) {
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO nodes (id, node_type) VALUES ('topic', 'entity')",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, 'loop', ?2, ?3)",
                params![loop_id, body, due],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, stability, confidence, importance, expired_at) VALUES (?1, 'topic', ?2, 0.9, 0.9, 1.0, ?3)",
                params![loop_id, format!("open loop: {body}"), expired_at],
            )
            .await;
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_event(
        conn: &Connection,
        day: &str,
        session_id: &str,
        turn_id: &str,
        event_type: &str,
        source: &str,
        target: &str,
        payload: &str,
    ) {
        let timestamp = format!("{day}T04:00:00");
        let _ = conn
            .execute(
                "INSERT INTO events (ts, session_id, turn_id, event_type, source, target, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![timestamp, session_id, turn_id, event_type, source, target, payload],
            )
            .await;
    }

    async fn insert_digest_message(
        conn: &Connection,
        details: (&str, &str, &str, &str),
        created_at: &str,
    ) {
        let (message_id, session_id, session_type, role) = details;
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, ?2, ?3)",
                params![session_id, session_type, created_at],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?1, ?2, ?3, 'x', ?4)",
                params![message_id, session_id, role, created_at],
            )
            .await;
    }

    fn digest(conn: &Connection, day: &str) -> super::DayDigest {
        // Whole-civil-day night window: matches the old slice for tests that
        // don't probe the boundary.
        let night = super::NightWindow {
            from: format!("{day}T00"),
            until: format!("{day}T24"),
        };
        digest_with_night(conn, day, &night)
    }

    fn digest_with_night(
        conn: &Connection,
        day: &str,
        night: &super::NightWindow,
    ) -> super::DayDigest {
        let activity = super::NightWindow {
            from: format!("{day}T"),
            until: format!("{day}U"),
        };
        let Ok(digest) =
            db::block_on(super::window_digest_async(conn, &activity, night, TODAY))
        else {
            unreachable!("digest query runs")
        };
        digest
    }

    #[test]
    fn digest_counts_only_yesterdays_slice() {
        let conn = db::block_on(digest_harness());
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "tg-2026-07-10",
            "turn-a",
            "created",
            "a",
            "b",
            "",
        ));
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "tg-2026-07-10",
            "turn-b",
            "updated",
            "c",
            "d",
            "",
        ));
        // Today's events must not leak into yesterday's digest.
        db::block_on(insert_event(
            &conn,
            TODAY,
            "tg-2026-07-11",
            "turn-c",
            "created",
            "e",
            "f",
            "",
        ));

        let digest = digest(&conn, YESTERDAY);
        assert_eq!(digest.turns, 2);
        assert_eq!(digest.created, 1);
        assert_eq!(digest.updated, 1);
    }

    #[test]
    fn digest_counts_only_user_messages_from_conversation_sessions() {
        let conn = db::block_on(digest_harness());
        let day = crate::timeutil::today_iso();
        let timestamp = format!("{day}T04:00:00");
        db::block_on(insert_digest_message(
            &conn,
            ("user-message", "chat-session", "conversation", "user"),
            &timestamp,
        ));
        db::block_on(insert_digest_message(
            &conn,
            (
                "assistant-message",
                "chat-session",
                "conversation",
                "assistant",
            ),
            &timestamp,
        ));
        db::block_on(insert_digest_message(
            &conn,
            ("dream-user-message", "dream-session", "dream", "user"),
            &timestamp,
        ));
        let digest = digest(&conn, &day);
        assert_eq!(digest.conversation_user_messages, 1);
    }

    #[test]
    fn digest_counts_dream_events_and_surfaces_gists() {
        let conn = db::block_on(digest_harness());
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "dream-100",
            "",
            "created",
            "x",
            "y",
            "",
        ));
        let gist_payload = serde_json::json!({"gist_of": ["g3 → m5"]}).to_string();
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "dream-100",
            "",
            "updated",
            "lesson",
            "topic",
            &gist_payload,
        ));

        let digest = digest(&conn, YESTERDAY);
        assert_eq!(digest.dream_events, 2);
        // Dream events carry no turn_id, so they never count as turns.
        assert_eq!(digest.turns, 0);
        assert_eq!(digest.gists, vec!["lesson → topic".to_string()]);
    }

    // The production window is yesterday's report hour → today's: a dream 47
    // seconds after midnight belongs to the night that consolidated yesterday.
    #[test]
    fn night_window_catches_the_post_midnight_dream() {
        let conn = db::block_on(digest_harness());
        db::block_on(insert_event(
            &conn,
            TODAY,
            "dream-200",
            "",
            "created",
            "x",
            "y",
            "",
        ));
        let night = super::NightWindow {
            from: format!("{YESTERDAY}T07"),
            until: format!("{TODAY}T07"),
        };
        let digest = digest_with_night(&conn, YESTERDAY, &night);
        assert_eq!(digest.dream_events, 1);
    }

    #[test]
    fn digest_ranks_injected_edges_by_frequency() {
        let conn = db::block_on(digest_harness());
        for _ in 0..3 {
            db::block_on(insert_event(
                &conn,
                YESTERDAY,
                "tg-2026-07-10",
                "turn-a",
                "injected",
                "hot",
                "node",
                "",
            ));
        }
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "tg-2026-07-10",
            "turn-a",
            "injected",
            "cold",
            "node",
            "",
        ));

        let digest = digest(&conn, YESTERDAY);
        assert_eq!(digest.top_injected.first(), Some(&"hot → node".to_string()));
        assert_eq!(digest.top_injected.len(), 2);
    }

    #[test]
    fn only_surfaced_reactions_count_toward_the_hit_rate() {
        // A plain chat reaction (no surfaced marker) must not count.
        assert_eq!(
            super::surfaced_reaction_kind(r#"{"kind":"reaction","reaction":"positive"}"#),
            None
        );
        // A reaction to a surfaced item counts.
        assert_eq!(
            super::surfaced_reaction_kind(
                r#"{"kind":"reaction","reaction":"negative","surfaced":true}"#
            ),
            Some("negative")
        );
        // A non-reaction outcome never counts.
        assert_eq!(
            super::surfaced_reaction_kind(r#"{"kind":"day_verdict","surfaced":true}"#),
            None
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn digest_counts_autonomy_yield_and_reactions() {
        let conn = db::block_on(digest_harness());
        db::block_on(async {
            let _ = conn.execute("INSERT INTO sessions (id, session_type, created_at) VALUES ('explore-1', 'exploration', '2026-07-10T02:00:00'), ('dream-1', 'dream', '2026-07-10T03:00:00'), ('chat-1', 'conversation', '2026-07-10T04:00:00'), ('news:2026-07-10', 'news', '2026-07-10T05:00:00')", ()).await;
            let _ = conn.execute("INSERT INTO findings (interest_topic, title, url, snippet, novelty, found_at, status, surfaced_at) VALUES ('Rust', 'Compiler news', 'https://example.com', '', 0.8, '2026-07-10T05:00:00', 'candidate', '')", ()).await;
        });
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "explore-1",
            "",
            "created",
            "new",
            "edge",
            "",
        ));
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "chat-1",
            "turn-1",
            "outcome",
            "",
            "",
            r#"{"kind":"reaction","reaction":"positive","surfaced":true}"#,
        ));
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "chat-1",
            "turn-2",
            "outcome",
            "",
            "",
            r#"{"kind":"reaction","reaction":"negative","surfaced":true}"#,
        ));
        db::block_on(insert_event(
            &conn,
            YESTERDAY,
            "news:2026-07-10",
            "",
            "promoted",
            "known",
            "fact",
            r#"{"from":"legacy","to":"web-confirmed"}"#,
        ));

        let digest = digest(&conn, YESTERDAY);
        assert_eq!(digest.autonomy.actions, 3);
        assert_eq!(digest.autonomy.yielded, 1);
        assert_eq!(digest.autonomy.news_findings, 1);
        assert_eq!(digest.autonomy.promoted, 1);
        assert_eq!(digest.autonomy.positive_reactions, 1);
        assert_eq!(digest.autonomy.negative_reactions, 1);
        let Some(line) = digest.autonomy.line() else {
            unreachable!("non-empty autonomy renders")
        };
        assert!(line.contains("1 of 2 surfaced items useful"));
        assert!(line.contains("1 news finding on your interests"));
        assert!(line.contains("1 promoted belief"));
        assert_eq!(
            digest.held_findings,
            vec!["Compiler news on Rust".to_string()]
        );
    }

    #[test]
    fn autonomy_line_omits_hit_rate_without_reactions() {
        let summary = super::AutonomySummary {
            actions: 1,
            yielded: 0,
            ..super::AutonomySummary::default()
        };
        let Some(line) = summary.line() else {
            unreachable!("one pass renders")
        };
        assert_eq!(
            line,
            "Autonomy: Ran 1 self-directed pass (0 added knowledge)."
        );
        assert!(super::AutonomySummary::default().line().is_none());
    }

    fn insert_digest_loop_cases(conn: &Connection) {
        let cases = [
            ("past", "ask about the file", YESTERDAY, None),
            ("today", "check the service", TODAY, None),
            ("future", "review the logs", "2026-07-12", None),
            ("context", "mention the reboot", "", None),
            ("closed", "confirm the backup", YESTERDAY, Some(TODAY)),
        ];
        for (loop_id, body, due, expired_at) in cases {
            db::block_on(insert_digest_loop(conn, loop_id, body, due, expired_at));
        }
    }

    #[test]
    fn digest_surfaces_only_due_live_loops() {
        let conn = db::block_on(digest_harness());
        insert_digest_loop_cases(&conn);

        let digest = digest(&conn, YESTERDAY);
        assert_eq!(
            digest.due_loops,
            vec![
                "ask about the file (due 2026-07-10)".to_string(),
                "check the service (due 2026-07-11)".to_string(),
            ]
        );
    }
