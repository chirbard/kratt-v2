    use super::{SynthesisPolicy, assemble_strategy_context, run_strategy_stage};
    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};
    use crate::memory::db;
    use crate::memory::graph::{Edge, Node};
    use std::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use turso::{Builder, Connection};

    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";

    // Replays one canned response per stream() call and counts the calls so a
    // test can assert the stage never reached the LLM.
    struct ScriptedClient {
        responses: Mutex<Vec<String>>,
        calls: AtomicUsize,
    }

    impl ScriptedClient {
        fn new(responses: &[&str]) -> Arc<ScriptedClient> {
            let mut queued: Vec<String> = Vec::new();
            for text in responses.iter().rev() {
                queued.push((*text).to_string());
            }
            Arc::new(ScriptedClient {
                responses: Mutex::new(queued),
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let text = match self.responses.lock() {
                Ok(mut queued) => queued.pop().unwrap_or_default(),
                Err(_) => String::new(),
            };
            let _ = sender.send(Chunk::Content(text));
            let _ = sender.send(Chunk::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }));
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }
    }

    fn harness() -> Connection {
        db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            conn
        })
    }

    fn write_node(conn: &Connection, node_id: &str, node_type: &str, body: &str) {
        let node = Node {
            node_type: node_type.to_string(),
            body: body.to_string(),
            date: String::new(),
        };
        let Ok(()) = db::write_node(conn, node_id, &node) else {
            unreachable!("node written")
        };
    }

    fn write_edge(conn: &Connection, edge: &Edge) {
        let Ok(()) = db::write_edge(conn, edge, "") else {
            unreachable!("edge written")
        };
    }

    fn load_hubs(conn: &Connection, game_id: &str) -> Vec<String> {
        let Ok(hubs) = db::load_game_hubs(conn, game_id) else {
            unreachable!("hubs load")
        };
        hubs
    }

    fn assemble(conn: &Connection, game_id: &str, node: &Node) -> super::GameStrategyContext {
        let hubs = load_hubs(conn, game_id);
        let Ok(ctx) = assemble_strategy_context(conn, game_id, node, &hubs) else {
            unreachable!("context assembles")
        };
        ctx
    }

    fn must_edge(conn: &Connection, source: &str, target: &str) -> Edge {
        let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
            unreachable!("edge exists")
        };
        edge
    }

    fn index_of(text: &str, needle: &str) -> usize {
        match text.find(needle) {
            Some(index) => index,
            None => unreachable!("substring present"),
        }
    }

    fn link_edge(source: &str, target: &str, memory_type: &str, expired: bool) -> Edge {
        Edge {
            source: source.to_string(),
            target: target.to_string(),
            content: format!("{source} → {target}"),
            stability: 0.3,
            confidence: 1.0,
            importance: 1.0,
            subject: false,
            date: "2020-01-01".to_string(),
            occurred_at: None,
            expired_at: if expired {
                Some("2020-01-02".to_string())
            } else {
                None
            },
            memory_type: memory_type.to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    // A hub→game define_task link plus a two-ply move chain for one session.
    fn play_game(conn: &Connection, session: &str, game_id: &str, moves: &[(&str, &str)]) {
        let hub = format!("task-{session}");
        write_node(conn, &hub, "task", "");
        write_edge(conn, &link_edge(&hub, game_id, "episodic", false));
        for (idx, (actor, text)) in moves.iter().enumerate() {
            let ply = idx + 1;
            let target = format!("{hub}-ply-{ply:03}");
            let mut edge = link_edge(&hub, &target, "episodic", false);
            edge.content = format!("ply {ply} ({actor}): {text}");
            write_edge(conn, &edge);
        }
    }

    fn record_outcome(conn: &Connection, session: &str, score: f64, comment: &str) {
        let payload = serde_json::json!({ "comment": comment }).to_string();
        db::block_on(async {
            let _ = conn
                .execute(
                    "INSERT INTO events (event_type, session_id, weight, payload) VALUES ('outcome', ?1, ?2, ?3)",
                    turso::params![session, score, payload.as_str()],
                )
                .await;
        });
    }

    fn channel() -> (Sender<super::DreamEvent>, Receiver<super::DreamEvent>) {
        std::sync::mpsc::channel()
    }

    #[test]
    fn context_holds_definition_both_chains_in_order_and_the_outcome() {
        let conn = harness();
        write_node(
            &conn,
            "game-chess",
            "game",
            "user plays White, reply move only",
        );
        play_game(
            &conn,
            "s1",
            "game-chess",
            &[("assistant", "e4"), ("user", "e5")],
        );
        play_game(
            &conn,
            "s2",
            "game-chess",
            &[("assistant", "d4"), ("user", "d5")],
        );
        record_outcome(&conn, "s1", -1.0, "resigned");
        let node = game_node("user plays White, reply move only");
        let ctx = assemble(&conn, "game-chess", &node);
        // Definition body, both chains in ply order, and the outcome score.
        assert_all_present(
            &ctx.text,
            &[
                "user plays White, reply move only",
                "ply 1 (assistant): e4",
                "ply 2 (user): e5",
                "ply 1 (assistant): d4",
                "outcome: score -1",
                "resigned",
            ],
        );
        assert!(
            index_of(&ctx.text, "e4") < index_of(&ctx.text, "e5"),
            "ply order"
        );
        assert_eq!((ctx.chains, ctx.outcomes), (2, 1));
    }

    fn assert_all_present(text: &str, needles: &[&str]) {
        for needle in needles {
            assert!(text.contains(needle), "missing: {needle}");
        }
    }

    fn game_node(body: &str) -> Node {
        Node {
            node_type: "game".to_string(),
            body: body.to_string(),
            date: String::new(),
        }
    }

    #[test]
    fn a_game_with_no_chains_skips_the_stage_without_an_llm_call() {
        let conn = harness();
        // A defined game but no hub ever linked/played it.
        write_node(&conn, "game-go", "game", "19x19 board");
        let llm: Arc<ScriptedClient> = ScriptedClient::new(&["{}"]);
        let dyn_llm: Arc<dyn LlmClient> = llm.clone();
        let (sender, receiver) = channel();
        let Ok(total) = run_strategy_stage(&dyn_llm, &conn, "dream-1", &sender) else {
            unreachable!("stage runs")
        };
        drop(receiver);
        assert_eq!(total, 0);
        assert_eq!(
            llm.calls.load(Ordering::SeqCst),
            0,
            "no game reached the LLM"
        );
    }

    #[test]
    fn hub_discovery_excludes_expired_links_and_other_games() {
        let conn = harness();
        write_node(&conn, "game-chess", "game", "");
        write_node(&conn, "game-go", "game", "");
        write_node(&conn, "task-live", "task", "");
        write_node(&conn, "task-dead", "task", "");
        write_node(&conn, "task-other", "task", "");
        write_edge(
            &conn,
            &link_edge("task-live", "game-chess", "episodic", false),
        );
        write_edge(
            &conn,
            &link_edge("task-dead", "game-chess", "episodic", true),
        );
        write_edge(
            &conn,
            &link_edge("task-other", "game-go", "episodic", false),
        );
        assert_eq!(
            load_hubs(&conn, "game-chess"),
            vec!["task-live".to_string()]
        );
    }

    // Drives the strategy response through the dream apply path one call at a
    // time (as the existing gist tests do), so the ledger's position machinery
    // is out of scope — this test is about which calls the dream path honors.
    fn apply_all(conn: &Connection, response: &str) -> Vec<(String, bool)> {
        let policy = SynthesisPolicy::empty();
        let mut events: Vec<super::Event> = Vec::new();
        let mut apply_state = super::DreamApplyState::new(&policy);
        let mut ctx = super::DreamCallContext {
            conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let mut outcomes = Vec::new();
        for call in super::parse_tool_calls(response) {
            let Ok(outcome) = super::apply_dream_call(&mut ctx, call) else {
                unreachable!("apply runs")
            };
            outcomes.push(outcome);
        }
        outcomes
    }

    #[test]
    fn a_strategy_pass_applies_gist_upsert_expire_and_denies_record_move() {
        let conn = harness();
        write_node(&conn, "game-chess", "game", "rules");
        // Two episodic plies for a gist to cite.
        play_game(
            &conn,
            "s1",
            "game-chess",
            &[("assistant", "Qh5"), ("user", "Nc6")],
        );
        // A pre-existing strategy the pass expires on contradicting evidence.
        let mut stale = link_edge("game-chess", "castle-early", "semantic", false);
        stale.content = "castle early".to_string();
        write_edge(&conn, &stale);
        let response = r#"{"tools":[
            {"tool":"gist_edge","args":{"source":"game-chess","target":"queen-sorties","content":"early queen sorties get punished","confidence":0.6,"stability":0.4,"supported_by":["task-s1 → task-s1-ply-001","task-s1 → task-s1-ply-002"]}},
            {"tool":"upsert_edge","args":{"source":"game-chess","target":"develop-knights","content":"develop knights before bishops","stability":0.4,"confidence":0.5,"importance":0.5,"subject":false,"date":"2026-07-11"}},
            {"tool":"expire_edge","args":{"source":"game-chess","target":"castle-early","rewritten_content":"castling early is not always best"}},
            {"tool":"record_move","args":{"move":"e4","actor":"assistant"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        // gist + upsert report an added edge; expire and the denied move do not.
        let mut added: Vec<bool> = Vec::new();
        for outcome in &outcomes {
            added.push(outcome.1);
        }
        assert_eq!(added, vec![true, true, false, false]);
        assert!(outcomes[3].0.contains("denied"), "record_move denied");
        assert_dream_edges_written(&conn);
    }

    #[test]
    fn dream_path_denies_open_and_close_loop() {
        let conn = harness();
        let response = r#"{"tools":[
            {"tool":"open_loop","args":{"intention":"ask later","anchor":"topic","due":""}},
            {"tool":"close_loop","args":{"loop":"ask-later","resolution":"done","note":"asked"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        assert_eq!(outcomes.len(), 2);
        assert!(!outcomes[0].1 && outcomes[0].0.contains("open_loop denied"));
        assert!(!outcomes[1].1 && outcomes[1].0.contains("close_loop denied"));
    }

    #[test]
    fn dream_generic_mutations_cannot_rewrite_or_remove_loops() {
        let conn = harness();
        write_node(&conn, "loop-reminder", "loop", "ask about the service");
        write_node(&conn, "topic", "entity", "");
        let mut anchor = link_edge("loop-reminder", "topic", "semantic", false);
        anchor.content = "open loop: ask about the service".to_string();
        write_edge(&conn, &anchor);
        let response = r#"{"tools":[
            {"tool":"upsert_node","args":{"name":"loop-reminder","type":"fact","body":"rewritten","date":""}},
            {"tool":"upsert_edge","args":{"source":"loop-reminder","target":"topic","content":"rewritten","stability":0.2,"confidence":0.2,"importance":0.2,"subject":false,"date":"2026-07-12"}},
            {"tool":"expire_edge","args":{"source":"loop-reminder","target":"topic","rewritten_content":"closed"}},
            {"tool":"delete_node","args":{"name":"topic"}}
        ]}"#;
        let outcomes = apply_all(&conn, response);
        assert_eq!(outcomes.len(), 4);
        assert!(!outcomes[0].1 && outcomes[0].0.contains("belongs to a loop node"));
        for (description, changed) in &outcomes[1..] {
            assert!(!changed && description.contains("open loops are heavy-owned"));
        }
        let stored = must_edge(&conn, "loop-reminder", "topic");
        assert_eq!(stored.content, "open loop: ask about the service");
        assert!(stored.expired_at.is_none());
        let Ok(Some(node)) = db::load_node(&conn, "loop-reminder") else {
            unreachable!("loop node remains")
        };
        assert_eq!(node.node_type, "loop");
    }

    fn assert_dream_edges_written(conn: &Connection) {
        assert_eq!(
            must_edge(conn, "game-chess", "queen-sorties").memory_type,
            "semantic"
        );
        assert_eq!(
            must_edge(conn, "game-chess", "develop-knights").content,
            "develop knights before bishops"
        );
        assert!(
            must_edge(conn, "game-chess", "castle-early")
                .expired_at
                .is_some(),
            "stale strategy expired"
        );
        // The denied record_move created no ply on the game hub.
        let Ok(chain) = db::load_task_chain(conn, "test", 240) else {
            unreachable!("chain loads")
        };
        assert!(chain.is_empty(), "record_move denied on the dream path");
    }
