    use super::{
        MAX_TOOL_ROUNDS, MemorySections, ToolExecutionContext, assemble_memory_block,
        build_memory_events, build_tools, catch_up_since, collect_anchor_ids,
        execute_catch_up_with, execute_recall_curiosities_with, execute_recall_intentions_with,
        execute_recall_interests_with, execute_recall_ping_policy_with, execute_recall_queue_with,
        execute_recall_recent_learning_with, execute_tool, parse_since_days, tools_for_round,
    };
    use crate::api::{Chunk, EmbeddingClient, ToolCallRequest};
    use crate::memory::{EdgeRef, GameKnowledge, OpenLoops, SemanticEdge, TaskChain};
    use turso::{Builder, Connection};

    struct TestEmbed;

    impl EmbeddingClient for TestEmbed {
        fn embed(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
            Ok(vec![vec![1.0]; texts.len()])
        }
    }

    fn test_tool_context<'context>(
        embed: &'context TestEmbed,
        sender: &'context std::sync::mpsc::Sender<Chunk>,
    ) -> ToolExecutionContext<'context> {
        ToolExecutionContext {
            embed,
            search: None,
            session_path: std::path::Path::new("missing"),
            sender,
        }
    }

    async fn introspection_harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let statements = [
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))",
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE inbox (id INTEGER PRIMARY KEY AUTOINCREMENT, accepted_at TEXT NOT NULL, source TEXT NOT NULL, instruction TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'queued', resolved_at TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')",
            "CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)",
            "CREATE TABLE pass_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, pass_type TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT 'failed', reason_class TEXT NOT NULL DEFAULT 'interrupted', items_added INTEGER NOT NULL DEFAULT 0)",
        ];
        for statement in statements {
            let _ = conn.execute(statement, ()).await;
        }
        conn
    }

    fn loops(block: &str) -> OpenLoops {
        OpenLoops {
            block: block.to_string(),
            refs: Vec::new(),
        }
    }

    fn knowledge(block: &str) -> GameKnowledge {
        GameKnowledge {
            block: block.to_string(),
            refs: Vec::new(),
        }
    }

    fn chain(block: &str) -> TaskChain {
        TaskChain {
            block: block.to_string(),
            refs: Vec::new(),
        }
    }

    #[test]
    fn game_block_and_chain_block_coexist_when_both_present() {
        let block = assemble_memory_block(
            &[],
            &loops(""),
            &knowledge("## Game: chess\nthe rules\n"),
            &chain("## Current task (in order)\nply 1 (assistant): e4\n"),
        );
        assert_eq!(
            block,
            "## Game: chess\nthe rules\n\n\n## Current task (in order)\nply 1 (assistant): e4\n"
        );
    }

    #[test]
    fn recall_ping_policy_reports_default_mute_diverts_and_resumed_state() {
        let conn = crate::memory::block_on(introspection_harness());
        let now = "2026-07-14T12:00:00";
        assert!(execute_recall_ping_policy_with(&conn, now).contains("9999-12-31"));
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO ping_policy (topic, muted_until, divert_until) VALUES ('', '', ''), ('Rust', '', '2026-07-15T00:00:00')", ()).await;
        });
        let diverted = execute_recall_ping_policy_with(&conn, now);
        assert!(diverted.contains("Rust is held for the digest"));
        crate::memory::block_on(async {
            let _ = conn
                .execute("DELETE FROM ping_policy WHERE topic = 'Rust'", ())
                .await;
        });
        assert_eq!(
            execute_recall_ping_policy_with(&conn, now),
            "Pinging is on; nothing is suppressed."
        );
    }

    #[test]
    fn no_game_link_adds_no_game_block() {
        let block = assemble_memory_block(
            &[],
            &loops(""),
            &knowledge(""),
            &chain("## Current task (in order)\n"),
        );
        assert_eq!(block, "## Current task (in order)\n");
    }

    #[test]
    fn open_loop_block_is_present_only_when_non_empty() {
        let present = assemble_memory_block(
            &[],
            &loops("## Open loops\n- ask about the server\n"),
            &knowledge(""),
            &chain(""),
        );
        let absent = assemble_memory_block(&[], &loops(""), &knowledge(""), &chain(""));
        assert_eq!(present, "## Open loops\n- ask about the server\n");
        assert!(absent.is_empty());
    }

    fn semantic_edge(source: &str, target: &str) -> SemanticEdge {
        SemanticEdge {
            source: source.to_string(),
            target: target.to_string(),
            content: String::new(),
            confidence: 1.0,
            stability: 1.0,
            importance: 1.0,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            embedding: None,
        }
    }

    #[test]
    fn loop_anchors_include_both_ranked_edge_endpoints_once() {
        let edges = vec![
            semantic_edge("home-server", "systemd"),
            semantic_edge("systemd", "reboot"),
        ];
        assert_eq!(
            collect_anchor_ids(&edges),
            vec!["home-server", "reboot", "systemd"]
        );
    }

    #[test]
    fn surfaced_loop_edges_record_local_injections() {
        let loop_ref = EdgeRef::new("loop-reminder", "home-server");
        let sections = MemorySections {
            loops: OpenLoops {
                block: "## Open loops\n- ask later\n".to_string(),
                refs: vec![loop_ref],
            },
            knowledge: knowledge(""),
            chain: chain(""),
        };
        let events = build_memory_events(("session", "turn"), &[], &sections);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "injected");
        assert_eq!(events[0].actor_id, "local");
        assert_eq!(events[0].source, "loop-reminder");
        assert_eq!(events[0].target, "home-server");
    }

    #[test]
    fn curiosity_recall_renders_real_questions_and_honest_empty_state() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(
            execute_recall_curiosities_with(&conn, "2026-07-14"),
            "No open curiosities right now."
        );
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO nodes VALUES ('note-1', 'note', 'Why does sleep improve recall?', '2026-07-12')", ()).await;
            let _ = conn
                .execute("INSERT INTO nodes VALUES ('sleep', 'entity', '', '')", ())
                .await;
            let _ = conn.execute("INSERT INTO edges (source, target, content, date) VALUES ('note-1', 'sleep', 'curious about sleep', '2026-07-12')", ()).await;
        });
        let rendered = execute_recall_curiosities_with(&conn, "2026-07-14");
        assert!(rendered.contains("Why does sleep improve recall?"));
        assert!(rendered.contains("2 days old"));
        assert!(rendered.contains("about sleep"));
    }

    #[test]
    fn interest_recall_surfaces_registration_and_honest_empty_state() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(
            execute_recall_interests_with(&conn),
            "No interests have emerged yet."
        );
        let inserted = crate::memory::insert_interest(&conn, "libsql internals", "user");
        assert!(inserted.is_ok());
        let rendered = execute_recall_interests_with(&conn);
        assert!(rendered.contains("libsql internals"));
        assert!(rendered.contains("user"));
    }

    #[test]
    fn intention_recall_renders_trigger_and_due_date() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(
            execute_recall_intentions_with(&conn, "2026-07-14"),
            "No open loops."
        );
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO nodes VALUES ('loop-1', 'loop', 'ask whether the deploy worked', '2026-07-14')", ()).await;
            let _ = conn
                .execute("INSERT INTO nodes VALUES ('deploy', 'entity', '', '')", ())
                .await;
            let _ = conn.execute("INSERT INTO edges (source, target, content, date) VALUES ('loop-1', 'deploy', 'open loop: ask whether the deploy worked', '2026-07-13')", ()).await;
        });
        let rendered = execute_recall_intentions_with(&conn, "2026-07-14");
        assert!(rendered.contains("Waiting on deploy"));
        assert!(rendered.contains("due 2026-07-14"));
    }

    #[test]
    fn recent_learning_respects_window_and_clamps_arguments() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(
            execute_recall_recent_learning_with(&conn, 0),
            "Nothing new consolidated recently."
        );
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO edges (source, target, content) VALUES ('new', 'belief', 'a new belief'), ('old', 'belief', 'an old belief')", ()).await;
            let _ = conn.execute("INSERT INTO events (ts, event_type, source, target) VALUES (strftime('%Y-%m-%dT%H:%M:%S', 'now'), 'created', 'new', 'belief'), (strftime('%Y-%m-%dT%H:%M:%S', 'now', '-8 days'), 'created', 'old', 'belief')", ()).await;
        });
        let rendered = execute_recall_recent_learning_with(&conn, 7);
        assert!(rendered.contains("a new belief"));
        assert!(rendered.contains("New (legacy)"));
        assert!(!rendered.contains("an old belief"));
        assert_eq!(parse_since_days(r#"{"since_days":99}"#), 7);
        assert_eq!(parse_since_days("{}"), 0);
    }

    #[test]
    fn queue_recall_renders_queued_and_recent_outcomes() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(execute_recall_queue_with(&conn), "Inbox is empty.");
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO inbox (accepted_at, source) VALUES ('2026-07-14T08:00:00', 'paper.pdf')", ()).await;
            let _ = conn.execute("INSERT INTO inbox (accepted_at, source, status, resolved_at, result) VALUES ('2026-07-13T08:00:00', 'broken.html', 'failed', '2026-07-14T07:00:00', '404')", ()).await;
        });
        let rendered = execute_recall_queue_with(&conn);
        assert!(rendered.contains("paper.pdf: queued"));
        assert!(rendered.contains("broken.html: failed (404)"));
    }

    #[test]
    fn catch_up_uses_windowed_digest_and_handles_a_quiet_window() {
        let conn = crate::memory::block_on(introspection_harness());
        assert_eq!(
            execute_catch_up_with(
                &conn,
                "2026-07-14T08:00:00",
                "2026-07-14T12:00:00",
                "2026-07-14"
            ),
            "Nothing has happened since we last talked."
        );
        crate::memory::block_on(async {
            let _ = conn.execute("INSERT INTO sessions (id, session_type, created_at) VALUES ('explore-1', 'exploration', '2026-07-14T08:30:00')", ()).await;
            let _ = conn.execute("INSERT INTO events (ts, session_id, turn_id, event_type, source, target) VALUES ('2026-07-14T09:00:00', 'explore-1', 'turn-1', 'created', 'new', 'belief')", ()).await;
            let _ = conn.execute("INSERT INTO inbox (accepted_at, source, status, resolved_at, result) VALUES ('2026-07-14T08:30:00', 'paper.pdf', 'done', '2026-07-14T11:00:00', 'study-1')", ()).await;
        });
        let rendered = execute_catch_up_with(
            &conn,
            "2026-07-14T08:00:00",
            "2026-07-14T12:00:00",
            "2026-07-14",
        );
        assert!(rendered.contains("1 beliefs added"));
        assert!(rendered.contains("Inbox: paper.pdf: done"));
        assert!(rendered.contains("Autonomy: Ran 1 self-directed pass (1 added knowledge)."));
    }

    #[test]
    fn catch_up_window_starts_at_the_last_message_before_an_away_gap() {
        let conn = crate::memory::block_on(introspection_harness());
        crate::memory::block_on(async {
            // Prior sitting ends 08:00; a 15h gap; then the current burst.
            for (idx, stamp) in [
                "2026-07-13T07:59:00",
                "2026-07-13T08:00:00",
                "2026-07-14T09:15:00",
                "2026-07-14T09:16:00",
            ]
            .iter()
            .enumerate()
            {
                let _ = conn
                    .execute(
                        "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?1, 's', 'user', 'x', ?2)",
                        turso::params![format!("m{idx}"), *stamp],
                    )
                    .await;
            }
        });
        // Boundary is the last message before the gap, not the previous keystroke.
        assert_eq!(catch_up_since(&conn), "2026-07-13T08:00:00");
    }

    #[test]
    fn catch_up_window_collapses_when_conversation_is_continuous() {
        let conn = crate::memory::block_on(introspection_harness());
        crate::memory::block_on(async {
            for (idx, stamp) in ["2026-07-14T09:15:00", "2026-07-14T09:16:00"]
                .iter()
                .enumerate()
            {
                let _ = conn
                    .execute(
                        "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?1, 's', 'assistant', 'x', ?2)",
                        turso::params![format!("m{idx}"), *stamp],
                    )
                    .await;
            }
        });
        // No away-gap → falls back to the latest message (empty window).
        assert_eq!(catch_up_since(&conn), "2026-07-14T09:16:00");
    }

    #[test]
    fn chat_tool_registry_and_dispatch_include_introspection_tools() {
        let tools = build_tools(false);
        let Some(array) = tools.as_array() else {
            unreachable!("tools are an array")
        };
        let mut names = Vec::new();
        for tool in array {
            let Some(value) = tool.pointer("/function/name") else {
                continue;
            };
            if let Some(name) = value.as_str() {
                names.push(name);
            }
        }
        let (sender, _receiver) = std::sync::mpsc::channel::<Chunk>();
        let embed = TestEmbed;
        let context = test_tool_context(&embed, &sender);
        for expected in [
            "retrieve_memory",
            "recall_curiosities",
            "recall_intentions",
            "recall_recent_learning",
            "recall_queue",
            "catch_up",
        ] {
            assert!(names.contains(&expected));
            let call = ToolCallRequest {
                id: "1".to_string(),
                name: expected.to_string(),
                args: "{}".to_string(),
            };
            assert_ne!(execute_tool(&call, &context), "unknown tool");
        }
        let bogus = ToolCallRequest {
            id: "1".to_string(),
            name: "bogus".to_string(),
            args: "{}".to_string(),
        };
        assert_eq!(execute_tool(&bogus, &context), "unknown tool");
    }

    #[test]
    fn tool_round_cap_removes_tools_from_final_generation() {
        let tools = build_tools(false);
        assert!(tools_for_round(&tools, MAX_TOOL_ROUNDS - 1).is_some());
        assert!(tools_for_round(&tools, MAX_TOOL_ROUNDS).is_none());
    }
