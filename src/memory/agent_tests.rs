    use super::{
        AgentRun, IterationOutcome, MAX_PASS_ITERATIONS, MemoryEvent, ToolCall, TurnMode,
        dropped_tool_reason, memory_mutation_tools, parse_tool_calls, parse_turn_mode, says_done,
        stream_collect,
    };
    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};
    use crate::memory::ledger::Event;
    use std::error::Error;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender};

    // Replays one canned response per stream() call; the loop under test never
    // reaches the DB, so only the LLM half of an iteration is exercised.
    struct ScriptedClient {
        responses: Mutex<Vec<String>>,
    }

    impl ScriptedClient {
        fn scripted(responses: &[&str]) -> Arc<dyn LlmClient> {
            let mut queued: Vec<String> = Vec::new();
            for text in responses.iter().rev() {
                queued.push((*text).to_string());
            }
            Arc::new(ScriptedClient {
                responses: Mutex::new(queued),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            let Ok(mut queued) = self.responses.lock() else {
                return;
            };
            // Running dry means the loop asked for more turns than scripted;
            // an empty reply reads as "no tool calls" and ends the pass.
            let text = queued.pop().unwrap_or_default();
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

    const ONE_CALL: &str =
        r#"{"tools":[{"tool":"delete_edge","args":{"source":"a","target":"b"}}]}"#;

    // Counts iterations by driving drive_pass with an iterate closure that does
    // the real LLM + parse work but stubs out the DB write.
    fn count_iterations(responses: &[&str]) -> usize {
        let (sender, receiver): (Sender<MemoryEvent>, Receiver<MemoryEvent>) =
            std::sync::mpsc::channel();
        let llm = ScriptedClient::scripted(responses);
        let agent_run = AgentRun {
            seats: super::MemorySeats::uniform(llm),
            user_msg: "user",
            assistant_msg: "assistant",
            sender: &sender,
            user_message_id: "msg-1",
        };
        let mut iterations = 0usize;
        let pass = super::heavy_pass("extract");
        let result = agent_run.drive_pass(&pass, "first prompt".to_string(), |_index, prompt| {
            iterations += 1;
            let call_text = stream_collect(Arc::clone(&agent_run.seats.heavy), prompt, drop, drop)?;
            let tool_calls = parse_tool_calls(&call_text);
            let done = tool_calls.is_empty() || says_done(&call_text);
            let mut descriptions = Vec::new();
            for _call in &tool_calls {
                descriptions.push("applied".to_string());
            }
            Ok(IterationOutcome {
                parsed: descriptions.len(),
                descriptions,
                done,
            })
        });
        assert!(result.is_ok());
        drop(receiver);
        iterations
    }

    // Same as count_iterations but drives the scribe pass, whose tighter cap
    // (MAX_SCRIBE_ITERATIONS) is what these tests exercise.
    fn count_scribe_iterations(responses: &[&str]) -> usize {
        let (sender, receiver): (Sender<MemoryEvent>, Receiver<MemoryEvent>) =
            std::sync::mpsc::channel();
        let llm = ScriptedClient::scripted(responses);
        let agent_run = AgentRun {
            seats: super::MemorySeats::uniform(llm),
            user_msg: "user",
            assistant_msg: "assistant",
            sender: &sender,
            user_message_id: "msg-1",
        };
        let mut iterations = 0usize;
        let pass = super::scribe_pass();
        let result = agent_run.drive_pass(&pass, "first prompt".to_string(), |_index, prompt| {
            iterations += 1;
            let call_text =
                stream_collect(Arc::clone(&agent_run.seats.scribe), prompt, drop, drop)?;
            let tool_calls = parse_tool_calls(&call_text);
            let done = tool_calls.is_empty() || says_done(&call_text);
            let mut descriptions = Vec::new();
            for _call in &tool_calls {
                descriptions.push("recorded".to_string());
            }
            Ok(IterationOutcome {
                parsed: descriptions.len(),
                descriptions,
                done,
            })
        });
        assert!(result.is_ok());
        drop(receiver);
        iterations
    }

    #[test]
    fn scribe_continues_once_to_catch_an_under_emitted_turn() {
        // First response records something, the continuation says DONE: two
        // iterations, exactly the recovery the cap is meant to allow.
        assert_eq!(count_scribe_iterations(&[ONE_CALL, "DONE"]), 2);
    }

    #[test]
    fn scribe_stops_at_its_two_iteration_cap() {
        // The model keeps emitting calls and never says DONE — the tighter
        // scribe cap halts it at two, not the heavy MAX_PASS_ITERATIONS.
        let responses = vec![ONE_CALL; MAX_PASS_ITERATIONS];
        assert_eq!(count_scribe_iterations(&responses), 2);
    }

    #[test]
    fn scribe_single_shot_stops_at_one_iteration() {
        // A setup turn the first response fully captures: DONE on the first
        // pass ends it immediately, no continuation latency.
        let done = format!("{ONE_CALL}\nDONE");
        assert_eq!(count_scribe_iterations(&[&done]), 1);
        // Zero tool calls also reads as finished on the first iteration.
        assert_eq!(count_scribe_iterations(&["nothing to record"]), 1);
    }

    #[test]
    fn pass_stops_when_the_model_reports_done() {
        assert_eq!(count_iterations(&[ONE_CALL, "DONE"]), 2);
    }

    #[test]
    fn pass_stops_immediately_when_no_tool_calls_are_emitted() {
        assert_eq!(count_iterations(&["nothing to record here"]), 1);
    }

    #[test]
    fn pass_stops_at_the_iteration_cap_when_the_model_never_finishes() {
        let responses = vec![ONE_CALL; MAX_PASS_ITERATIONS + 2];
        assert_eq!(count_iterations(&responses), MAX_PASS_ITERATIONS);
    }

    #[test]
    fn done_detection_tolerates_prose_and_casing() {
        assert!(says_done("done"));
        assert!(says_done("  Done.\n"));
        assert!(says_done("Nothing else remains — DONE"));
        assert!(!says_done(ONE_CALL));
        assert!(!says_done("the plan was abandoned, work undone"));
    }

    #[test]
    fn a_response_with_both_tool_calls_and_done_applies_then_stops() {
        let mixed = format!("{ONE_CALL}\nDONE");
        let calls = parse_tool_calls(&mixed);
        assert_eq!(calls.len(), 1);
        assert!(says_done(&mixed));
        assert_eq!(count_iterations(&[&mixed]), 1);
    }

    #[test]
    fn turn_mode_parses_task_and_chat_and_defaults_to_conversation() {
        assert_eq!(parse_turn_mode("TASK"), TurnMode::TaskMove);
        assert_eq!(parse_turn_mode("task"), TurnMode::TaskMove);
        assert_eq!(parse_turn_mode("CHAT"), TurnMode::Conversation);
        assert_eq!(parse_turn_mode("garbage nonsense"), TurnMode::Conversation);
        assert_eq!(parse_turn_mode(""), TurnMode::Conversation);
        // Ambiguous replies mentioning both bias to the safe heavy path.
        assert_eq!(
            parse_turn_mode("this is chat, not a task"),
            TurnMode::Conversation
        );
    }

    #[test]
    fn truncated_tool_array_salvages_complete_objects() {
        let truncated = r#"{"tools":[{"tool":"upsert_node","args":{"name":"a","type":"entity","body":"","date":""}},{"tool":"upsert_edge","args":{"source":"a","target":"b","content":"a relates to b","stability":0.5,"confidence":0.5,"importance":0.5,"subject":false,"date":"2026-07-06"}},{"tool":"upsert_node","args":{"name":"trunc"#;
        let calls = parse_tool_calls(truncated);
        assert_eq!(calls.len(), 2);
        assert!(matches!(calls[0], ToolCall::UpsertNode { .. }));
        assert!(matches!(calls[1], ToolCall::UpsertEdge { .. }));
    }

    #[test]
    fn dropped_tool_calls_distinguish_unknown_names_from_bad_shapes() {
        let unknown = serde_json::json!({"tool": "invent_memory", "args": {}});
        let bad_shape = serde_json::json!({"tool": "upsert_edge", "args": {}});
        assert_eq!(dropped_tool_reason(&unknown), "unknown tool");
        assert_eq!(dropped_tool_reason(&bad_shape), "bad shape");
        assert!(parse_tool_calls(&unknown.to_string()).is_empty());
        assert!(parse_tool_calls(&bad_shape.to_string()).is_empty());
    }

    #[test]
    fn semantic_tool_schema_exposes_occurred_at_but_not_model_dates() {
        let tools = memory_mutation_tools();
        let node_parameters = &tools[0]["function"]["parameters"];
        let edge_parameters = &tools[2]["function"]["parameters"];
        assert!(node_parameters["properties"].get("date").is_none());
        assert!(edge_parameters["properties"].get("date").is_none());
        assert!(edge_parameters["properties"].get("occurred_at").is_some());
        let required = edge_parameters["required"].as_array().cloned().unwrap_or_default();
        assert!(!required.contains(&serde_json::json!("occurred_at")));
    }

    #[test]
    fn upsert_edge_parses_inferred_from_pairs() {
        let text = r#"{"tool":"upsert_edge","args":{"source":"result","target":"fact","inferred_from":[["Premise A","Fact"],["premise-b","fact"]]}}"#;
        let calls = parse_tool_calls(text);
        let ToolCall::UpsertEdge {
            inferred_from: Some(refs),
            ..
        } = &calls[0]
        else {
            unreachable!("parsed inferred refs")
        };
        assert_eq!(
            refs,
            &vec![
                ("premise-a".to_string(), "fact".to_string()),
                ("premise-b".to_string(), "fact".to_string()),
            ]
        );
    }

    #[test]
    fn upsert_edge_parses_optional_occurred_at() {
        let historical = parse_tool_calls(
            r#"{"tool":"upsert_edge","args":{"source":"release","target":"product","occurred_at":"2022"}}"#,
        );
        let ToolCall::UpsertEdge { occurred_at, .. } = &historical[0] else {
            unreachable!("parsed historical edge")
        };
        assert_eq!(occurred_at.as_deref(), Some("2022"));

        let current = parse_tool_calls(
            r#"{"tool":"upsert_edge","args":{"source":"current","target":"fact","occurred_at":""}}"#,
        );
        let ToolCall::UpsertEdge { occurred_at, .. } = &current[0] else {
            unreachable!("parsed current edge")
        };
        assert_eq!(occurred_at, &None);
    }

    #[test]
    fn prose_wrapped_json_is_recovered() {
        let wrapped = r#"Here are the extracted facts: {"tools":[{"tool":"delete_edge","args":{"source":"a","target":"b"}}]} done."#;
        let calls = parse_tool_calls(wrapped);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn memory_upsert_restamps_an_existing_legacy_edge_as_user_confirmed() {
        use super::{ApplyToolOptions, apply_tool_call};
        use crate::memory::db;
        use crate::memory::graph::Edge;
        use crate::memory::ledger::{Actor, EventContext};
        use turso::Builder;

        let conn = db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))", ()).await;
            conn
        });
        let legacy = Edge {
            source: "rust".to_string(),
            target: "memory".to_string(),
            content: "old claim".to_string(),
            stability: 0.5,
            confidence: 0.5,
            importance: 0.4,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        };
        assert!(db::write_edge(&conn, &legacy, "old").is_ok());
        let call = ToolCall::UpsertEdge {
            source: "rust".to_string(),
            target: "memory".to_string(),
            content: "user restated claim".to_string(),
            stability: 0.3,
            confidence: 0.9,
            importance: 0.7,
            subject: false,
            occurred_at: None,
            memory_type: Some("semantic".to_string()),
            inferred_from: None,
        };
        let options = ApplyToolOptions {
            actor: Actor::new("system", "extract"),
            context: EventContext::new("chat-1", "turn-1"),
            source_message_id: "chat-1:user:1",
            event_payload: "",
            confidence_cap: None,
            provenance: "user-confirmed",
            default_memory_type: "semantic",
        };
        let mut events: Vec<Event> = Vec::new();
        assert!(apply_tool_call(call, &conn, &options, &mut events).is_ok());
        let Ok(Some(edge)) = db::load_edge(&conn, "rust", "memory") else {
            unreachable!("re-upserted edge exists")
        };
        assert_eq!(edge.provenance, "user-confirmed");
        assert_eq!(edge.confidence, 0.9);
        assert_eq!(edge.stability, 0.55);
    }

    #[test]
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    fn heavy_upsert_inherits_an_inferred_cap_or_falls_back() {
        use super::{ApplyToolOptions, apply_tool_call};
        use crate::memory::db;
        use crate::memory::graph::Edge;
        use crate::memory::ledger::{Actor, EventContext};
        use turso::Builder;

        let conn = db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))", ()).await;
            conn
        });
        for (source, confidence) in [("premise-a", 1.0), ("premise-b", 0.9)] {
            let premise = Edge {
                source: source.to_string(),
                target: "fact".to_string(),
                content: "premise".to_string(),
                stability: 0.5,
                confidence,
                importance: 0.5,
                subject: false,
                date: String::new(),
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "user-confirmed".to_string(),
                ceiling: None,
                last_decayed_at: String::new(),
            };
            assert!(db::write_edge(&conn, &premise, "").is_ok());
        }
        let options = ApplyToolOptions {
            actor: Actor::new("system", "extract"),
            context: EventContext::new("chat-1", "turn-1"),
            source_message_id: "",
            event_payload: "",
            confidence_cap: None,
            provenance: "user-confirmed",
            default_memory_type: "semantic",
        };
        let mut events: Vec<Event> = Vec::new();
        let inferred = ToolCall::UpsertEdge {
            source: "conclusion".to_string(),
            target: "fact".to_string(),
            content: "combined".to_string(),
            stability: 0.5,
            confidence: 0.95,
            importance: 0.5,
            subject: false,
            occurred_at: None,
            memory_type: None,
            inferred_from: Some(vec![
                ("premise-a".to_string(), "fact".to_string()),
                ("premise-b".to_string(), "fact".to_string()),
            ]),
        };
        assert!(apply_tool_call(inferred, &conn, &options, &mut events).is_ok());
        let Ok(Some(stored)) = db::load_edge(&conn, "conclusion", "fact") else {
            unreachable!("inferred edge exists")
        };
        assert_eq!(stored.provenance, "inferred");
        assert_eq!(stored.ceiling, Some(0.81));
        assert_eq!(stored.confidence, 0.81);
        assert_eq!(stored.stability, 0.3);

        let fallback = ToolCall::UpsertEdge {
            source: "fallback".to_string(),
            target: "fact".to_string(),
            content: "unsupported".to_string(),
            stability: 0.5,
            confidence: 0.9,
            importance: 0.5,
            subject: false,
            occurred_at: None,
            memory_type: None,
            inferred_from: Some(vec![("ghost".to_string(), "fact".to_string())]),
        };
        assert!(apply_tool_call(fallback, &conn, &options, &mut events).is_ok());
        let Ok(Some(stored)) = db::load_edge(&conn, "fallback", "fact") else {
            unreachable!("fallback edge exists")
        };
        assert_eq!(stored.provenance, "user-confirmed");
        assert_eq!(stored.ceiling, None);
        assert_eq!(stored.stability, 0.5);
    }

    mod record_move {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            conn
        }

        fn options_for_phase(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("game-1", "game-1:user:ts"),
                source_message_id: "game-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn record(move_text: &str, actor: &str) -> ToolCall {
            ToolCall::RecordMove {
                move_text: move_text.to_string(),
                actor: actor.to_string(),
            }
        }

        fn apply<'opt>(
            conn: &turso::Connection,
            options: &ApplyToolOptions<'opt>,
            call: ToolCall,
        ) -> String {
            let mut events: Vec<Event> = Vec::new();
            match apply_tool_call(call, conn, options, &mut events) {
                Ok(description) => description,
                Err(err) => panic!("apply succeeds: {err}"),
            }
        }

        fn chain(conn: &turso::Connection) -> Vec<crate::memory::graph::Edge> {
            let Ok(chain) = db::load_task_chain(conn, "game-1", 240) else {
                unreachable!("chain loads")
            };
            chain
        }

        #[test]
        fn valid_args_parse_and_missing_move_is_rejected() {
            let calls = parse_tool_calls(
                r#"{"tool":"record_move","args":{"move":"Nc3","actor":"assistant"}}"#,
            );
            assert_eq!(calls.len(), 1);
            assert!(matches!(calls[0], ToolCall::RecordMove { .. }));
            assert!(
                parse_tool_calls(r#"{"tool":"record_move","args":{"actor":"user"}}"#).is_empty()
            );
        }

        // Two moves applied as two separate calls: the second call reads the ply
        // counter back from the DB (simulated restart), so it appends rather than
        // restarting at 1.
        fn two_move_chain(conn: &turso::Connection) -> Vec<crate::memory::graph::Edge> {
            let options = options_for_phase("scribe");
            let first = apply(conn, &options, record("e4", "assistant"));
            assert!(first.contains("ply 1"), "{first}");
            let second = apply(conn, &options, record("e5", "user"));
            assert!(second.contains("ply 2"), "{second}");
            chain(conn)
        }

        fn targets(chain: &[crate::memory::graph::Edge]) -> Vec<String> {
            let mut ids = Vec::new();
            for edge in chain {
                ids.push(edge.target.clone());
            }
            ids
        }

        fn contents(chain: &[crate::memory::graph::Edge]) -> Vec<String> {
            let mut lines = Vec::new();
            for edge in chain {
                lines.push(edge.content.clone());
            }
            lines
        }

        #[test]
        fn scribe_numbers_plies_in_order_from_the_db() {
            let conn = db::block_on(harness());
            let chain = two_move_chain(&conn);
            assert_eq!(
                targets(&chain),
                vec!["task-game-1-ply-001", "task-game-1-ply-002"]
            );
        }

        #[test]
        fn scribe_writes_episodic_edges_with_actor_content() {
            let conn = db::block_on(harness());
            let chain = two_move_chain(&conn);
            assert_eq!(
                contents(&chain),
                vec!["ply 1 (assistant): e4", "ply 2 (user): e5"]
            );
            assert_eq!(chain[0].memory_type, "episodic");
            assert_eq!(chain[0].origin_session, "game-1");
        }

        #[test]
        fn unknown_actor_maps_to_user_without_panicking() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(&conn, &options, record("Qh5", "white"));
            assert!(description.contains("(user)"), "{description}");
        }

        #[test]
        fn heavy_and_dream_paths_deny_the_tool() {
            let conn = db::block_on(harness());
            for phase in ["extract", "consolidate", "consolidation"] {
                let options = options_for_phase(phase);
                let description = apply(&conn, &options, record("e4", "assistant"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            assert!(chain(&conn).is_empty(), "denied calls write nothing");
        }
    }

    mod game {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', embedding BLOB, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
        const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
        const VIEWS_DDL: &str = "CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))";
        const POSITIONS_DDL: &str = "CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))";
        const POSITION_HISTORY_DDL: &str = "CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn.execute(VIEWS_DDL, ()).await;
            let _ = conn.execute(POSITIONS_DDL, ()).await;
            let _ = conn.execute(POSITION_HISTORY_DDL, ()).await;
            conn
        }

        fn options_for_phase(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("game-1", "game-1:user:ts"),
                source_message_id: "game-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn apply<'opt>(
            conn: &turso::Connection,
            options: &ApplyToolOptions<'opt>,
            call: ToolCall,
        ) -> String {
            let mut events: Vec<Event> = Vec::new();
            match apply_tool_call(call, conn, options, &mut events) {
                Ok(description) => description,
                Err(err) => panic!("apply succeeds: {err}"),
            }
        }

        fn define(name: &str, description: &str) -> ToolCall {
            ToolCall::DefineTask {
                name: name.to_string(),
                description: description.to_string(),
            }
        }

        fn edge(
            conn: &turso::Connection,
            source: &str,
            target: &str,
        ) -> crate::memory::graph::Edge {
            let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
                unreachable!("edge {source} → {target} exists")
            };
            edge
        }

        #[test]
        fn define_task_creates_game_node_and_session_hub_link() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(
                &conn,
                &options,
                define("Chess", "user White, replies move only"),
            );
            assert!(description.contains("chess"), "{description}");
            let Ok(Some(node)) = db::load_node(&conn, "game-chess") else {
                unreachable!("game node exists")
            };
            assert_eq!(node.node_type, "game");
            assert_eq!(node.body, "user White, replies move only");
            let link = edge(&conn, "task-game-1", "game-chess");
            assert_eq!(link.content, "this session plays chess");
            assert_eq!(link.memory_type, "episodic");
            assert_eq!(link.origin_session, "game-1");
        }

        #[test]
        fn redefining_with_a_shorter_description_keeps_the_longer_body() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            apply(
                &conn,
                &options,
                define("chess", "the long, detailed rules of the game"),
            );
            apply(&conn, &options, define("chess", "chess"));
            let Ok(Some(node)) = db::load_node(&conn, "game-chess") else {
                unreachable!("game node exists")
            };
            assert_eq!(node.body, "the long, detailed rules of the game");
        }

        #[test]
        fn define_task_is_scribe_only() {
            let conn = db::block_on(harness());
            for phase in ["extract", "consolidate", "consolidation"] {
                let options = options_for_phase(phase);
                let description = apply(&conn, &options, define("chess", "rules"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            let Ok(missing) = db::load_node(&conn, "game-chess") else {
                unreachable!("load_node succeeds")
            };
            assert!(missing.is_none());
        }

        fn rule(text: &str) -> ToolCall {
            ToolCall::RecordRule {
                rule: text.to_string(),
            }
        }

        #[test]
        fn record_rule_writes_a_durable_semantic_fact_anchored_to_the_game() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            apply(&conn, &options, define("chess", "rules"));
            let description = apply(&conn, &options, rule("the king moves one square"));
            assert!(description.contains("record_rule"), "{description}");
            let rule_edge = edge(&conn, "game-chess", "the-king-moves-one-square");
            assert!((rule_edge.confidence - 0.9).abs() < 1e-9);
            assert!((rule_edge.stability - 0.5).abs() < 1e-9);
            assert_eq!(rule_edge.memory_type, "semantic");
        }

        #[test]
        fn record_rule_is_denied_without_a_defined_game() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(&conn, &options, rule("some rule"));
            assert!(description.contains("define_task first"), "{description}");
        }

        #[test]
        fn record_rule_is_allowed_from_scribe_and_heavy_but_denied_from_dream() {
            let conn = db::block_on(harness());
            apply(
                &conn,
                &options_for_phase("scribe"),
                define("chess", "rules"),
            );
            for phase in ["scribe", "extract", "consolidate"] {
                let description = apply(&conn, &options_for_phase(phase), rule("shared rule"));
                assert!(
                    description.contains("record_rule"),
                    "{phase}: {description}"
                );
            }
            let denied = apply(
                &conn,
                &options_for_phase("consolidation"),
                rule("shared rule"),
            );
            assert!(denied.contains("denied"), "{denied}");
        }

        fn outcome(result: &str) -> ToolCall {
            ToolCall::RecordOutcome {
                result: result.to_string(),
                comment: "the game ended".to_string(),
            }
        }

        fn reaction(value: &str) -> ToolCall {
            ToolCall::RecordReaction {
                reaction: value.to_string(),
                comment: "evaluated the previous answer".to_string(),
            }
        }

        async fn seed_reaction_edge(conn: &turso::Connection) {
            let _ = conn
                .execute("INSERT INTO nodes (id) VALUES ('a'), ('b')", ())
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO edges (source, target, confidence, provenance)
                     VALUES ('a', 'b', 0.5, 'user-confirmed')",
                    (),
                )
                .await;
            let _ = conn.execute("INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, source, target) VALUES ('', 'game-1', 'turn-old', 'system', 'local', 'injected', 'a', 'b'), ('', 'game-1', 'turn-latest', 'system', 'local', 'injected', 'a', 'b')", ()).await;
        }

        fn outcome_turns(conn: &turso::Connection) -> Vec<String> {
            db::block_on(async {
                let Ok(mut rows) = conn
                    .query(
                        "SELECT turn_id FROM events WHERE event_type = 'outcome' ORDER BY id",
                        (),
                    )
                    .await
                else {
                    unreachable!("outcomes query")
                };
                let mut turns = Vec::new();
                while let Ok(Some(row)) = rows.next().await {
                    let Ok(value) = row.get_value(0) else {
                        unreachable!("turn value")
                    };
                    turns.push(db::text(value));
                }
                turns
            })
        }

        fn outcome_payloads(conn: &turso::Connection) -> Vec<String> {
            db::block_on(async {
                let Ok(mut rows) = conn
                    .query(
                        "SELECT payload FROM events WHERE event_type = 'outcome' ORDER BY id",
                        (),
                    )
                    .await
                else {
                    unreachable!("outcomes query")
                };
                let mut payloads = Vec::new();
                while let Ok(Some(row)) = rows.next().await {
                    let Ok(value) = row.get_value(0) else {
                        unreachable!("payload value")
                    };
                    payloads.push(db::text(value));
                }
                payloads
            })
        }

        #[test]
        fn reaction_parse_accepts_binary_values_only() {
            for value in ["positive", "negative"] {
                let input = format!(
                    r#"{{"tool":"record_reaction","args":{{"reaction":"{value}","comment":"clear evaluation"}}}}"#
                );
                assert!(matches!(
                    parse_tool_calls(&input)[0],
                    ToolCall::RecordReaction { .. }
                ));
            }
            assert!(
                parse_tool_calls(r#"{"tool":"record_reaction","args":{"reaction":"meh"}}"#)
                    .is_empty()
            );
        }

        #[test]
        fn reaction_targets_latest_prior_turn_and_is_seat_symmetric() {
            let conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&conn));
            for phase in ["scribe", "extract"] {
                let description = apply(&conn, &options_for_phase(phase), reaction("positive"));
                assert!(
                    description.contains("turn-latest"),
                    "{phase}: {description}"
                );
            }
            for phase in ["consolidation", "ingest"] {
                let description = apply(&conn, &options_for_phase(phase), reaction("positive"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            assert_eq!(outcome_turns(&conn), vec!["turn-latest"]);
        }

        #[test]
        fn reaction_to_ping_is_stamped_surfaced_but_normal_reaction_is_not() {
            let ping_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&ping_conn));
            db::block_on(async {
                let _ = ping_conn
                    .execute(
                        r#"INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, payload)
                         VALUES ('', 'game-1', 'ping-1', 'system', 'ping', 'surfaced', '{"surfaced":true}')"#,
                        (),
                    )
                    .await;
            });
            apply(
                &ping_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(outcome_payloads(&ping_conn)[0].contains(r#""surfaced":true"#));

            let normal_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&normal_conn));
            apply(
                &normal_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(outcome_payloads(&normal_conn)[0].contains(r#""surfaced":false"#));
        }

        #[test]
        fn reaction_denies_without_prior_turn() {
            let conn = db::block_on(harness());
            let description = apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            assert_eq!(
                description,
                "record_reaction denied: no previous turn in this session"
            );
            assert!(outcome_turns(&conn).is_empty());
        }

        #[test]
        fn reaction_score_moves_confidence_and_session_outcome_coexists() {
            let positive_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&positive_conn));
            let positive_before = edge(&positive_conn, "a", "b").confidence;
            apply(
                &positive_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(edge(&positive_conn, "a", "b").confidence > positive_before);

            let conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&conn));
            let before = edge(&conn, "a", "b").confidence;
            apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            assert!(edge(&conn, "a", "b").confidence < before);
            apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            apply(&conn, &options_for_phase("scribe"), outcome("draw"));
            assert_eq!(outcome_turns(&conn), vec!["turn-latest", ""]);
        }

        #[test]
        fn record_outcome_works_from_scribe_and_heavy_but_not_dream_or_ingest() {
            let conn = db::block_on(harness());
            apply(
                &conn,
                &options_for_phase("scribe"),
                define("chess", "rules"),
            );
            let description = apply(&conn, &options_for_phase("scribe"), outcome("loss"));
            assert!(
                description.contains("loss") && description.contains("chess"),
                "{description}"
            );
            // Game-end turns legitimately classify chat ("I resign — good
            // game" has no new move), so the heavy pass must be able to score.
            let from_extract = apply(&conn, &options_for_phase("extract"), outcome("win"));
            assert!(from_extract.contains("win"), "{from_extract}");
            let denied = apply(&conn, &options_for_phase("ingest"), outcome("win"));
            assert!(denied.contains("denied"), "{denied}");
        }

        #[test]
        fn invalid_outcome_result_is_parse_rejected() {
            assert!(
                parse_tool_calls(r#"{"tool":"record_outcome","args":{"result":"forfeit"}}"#)
                    .is_empty()
            );
            let calls = parse_tool_calls(r#"{"tool":"record_outcome","args":{"result":"draw"}}"#);
            assert_eq!(calls.len(), 1);
            assert!(matches!(calls[0], ToolCall::RecordOutcome { .. }));
        }

        #[test]
        fn outcome_score_maps_the_three_verdicts() {
            assert!((super::super::outcome_score("win") - 1.0).abs() < 1e-9);
            assert!((super::super::outcome_score("draw")).abs() < 1e-9);
            assert!((super::super::outcome_score("loss") + 1.0).abs() < 1e-9);
        }
    }

    mod inbox {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const INBOX_DDL: &str = "CREATE TABLE inbox (id INTEGER PRIMARY KEY AUTOINCREMENT, accepted_at TEXT NOT NULL, source TEXT NOT NULL, instruction TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'queued', resolved_at TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '')";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(INBOX_DDL, ()).await;
            conn
        }

        fn options(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "chat-1:user:ts"),
                source_message_id: "chat-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn add(conn: &turso::Connection, phase: &'static str, source: &str) -> String {
            let call = ToolCall::InboxAdd {
                source: source.to_string(),
                instruction: "focus on X".to_string(),
            };
            let mut events: Vec<Event> = Vec::new();
            let Ok(description) = apply_tool_call(call, conn, &options(phase), &mut events) else {
                unreachable!("apply succeeds")
            };
            description
        }

        #[test]
        fn inbox_add_queues_urls_from_scribe_and_heavy_seats() {
            let conn = db::block_on(harness());
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued"));
            assert!(add(&conn, "extract", "http://example.com/b").contains("queued"));
            let count = db::block_on(db::count_queued_inbox(&conn));
            assert_eq!(count, 2);
        }

        // Looped passes re-emit across iterations; the apply layer holds the line.
        #[test]
        fn inbox_add_dedups_a_source_already_queued() {
            let conn = db::block_on(harness());
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued item"));
            let second = add(&conn, "scribe", "https://example.com/a");
            assert!(second.contains("already queued"), "{second}");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 1);
            // A resolved source may be queued again (re-study is legitimate).
            let Ok(Some(item)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("item exists")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, item.item_id, "done", "ingest-1") else {
                unreachable!("resolve succeeds")
            };
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued item"));
        }

        #[test]
        fn inbox_add_denied_from_dream_and_ingest_seats() {
            let conn = db::block_on(harness());
            for phase in ["consolidation", "ingest"] {
                let result = add(&conn, phase, "https://example.com/a");
                assert!(result.contains("denied"), "{phase}: {result}");
            }
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 0);
        }

        #[test]
        fn inbox_add_rejects_a_source_that_is_neither_url_nor_file() {
            let conn = db::block_on(harness());
            let result = add(&conn, "extract", "definitely/not/a/real/path.txt");
            assert!(result.contains("denied"), "{result}");
            let empty = add(&conn, "extract", "");
            assert!(empty.contains("denied"), "{empty}");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 0);
        }

        #[test]
        fn queue_is_fifo_and_resolution_updates_status() {
            let conn = db::block_on(harness());
            let Ok(first) = db::insert_inbox_item(&conn, "https://a", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(_second) = db::insert_inbox_item(&conn, "https://b", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(Some(oldest)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("oldest exists")
            };
            assert_eq!(oldest.item_id, first);
            assert_eq!(oldest.source, "https://a");
            let Ok(()) = db::resolve_inbox_item(&conn, first, "done", "ingest-1") else {
                unreachable!("resolve succeeds")
            };
            let Ok(Some(next)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("next exists")
            };
            assert_eq!(next.source, "https://b");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 1);
        }

        #[test]
        #[allow(clippy::too_many_lines)]
        fn outcomes_cover_window_resolutions_and_all_queued_items() {
            let conn = db::block_on(harness());
            let Ok(done_id) = db::insert_inbox_item(&conn, "https://done", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(failed_id) = db::insert_inbox_item(&conn, "https://failed", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(old_id) = db::insert_inbox_item(&conn, "https://old", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(_queued) = db::insert_inbox_item(&conn, "https://queued", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, done_id, "done", "ingest-9") else {
                unreachable!("resolve succeeds")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, failed_id, "failed", "HTTP 404") else {
                unreachable!("resolve succeeds")
            };
            // Resolved long before the window: must not re-report.
            let _ = db::block_on(conn.execute(
                "UPDATE inbox SET status = 'done', resolved_at = '2020-01-01T00:00:00', result = 'x' WHERE id = ?1",
                turso::params![old_id],
            ));
            let Ok(outcomes) =
                db::block_on(db::load_inbox_outcomes(&conn, "2026-01-01", "2099-01-01"))
            else {
                unreachable!("outcomes load")
            };
            assert!(
                outcomes.contains(&"https://done: done".to_string()),
                "{outcomes:?}"
            );
            assert!(
                outcomes.contains(&"https://failed: failed (HTTP 404)".to_string()),
                "{outcomes:?}"
            );
            assert!(
                outcomes.contains(&"https://queued: queued".to_string()),
                "{outcomes:?}"
            );
            for line in &outcomes {
                assert!(!line.starts_with("https://old"), "{outcomes:?}");
            }
        }
    }

    mod interests {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            conn
        }

        fn register(conn: &turso::Connection, phase: &'static str, topic: &str) -> String {
            let options = ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "turn-1"),
                source_message_id: "message-1",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "semantic",
            };
            let mut events: Vec<Event> = Vec::new();
            let call = ToolCall::RegisterInterest {
                topic: topic.to_string(),
            };
            apply_tool_call(call, conn, &options, &mut events).unwrap_or_default()
        }

        #[test]
        fn registration_is_seat_symmetric_and_idempotent() {
            let conn = db::block_on(harness());
            for phase in ["scribe", "extract", "consolidate"] {
                assert!(register(&conn, phase, "libsql internals").contains("libsql-internals"));
            }
            let interests = db::block_on(db::top_interests_with(&conn, 10)).unwrap_or_default();
            assert_eq!(interests.len(), 1);
            assert_eq!(interests[0].topic, "libsql internals");
            assert_eq!(interests[0].origin, "user");
        }

        #[test]
        fn registration_is_denied_from_dream_and_ingest() {
            let conn = db::block_on(harness());
            for phase in ["consolidation", "ingest"] {
                assert!(register(&conn, phase, "libsql internals").contains("denied"));
            }
            let interests = db::block_on(db::top_interests_with(&conn, 10)).unwrap_or_default();
            assert!(interests.is_empty());
        }

        #[test]
        fn register_interest_parses_only_non_empty_topics() {
            let calls = parse_tool_calls(
                r#"{"tool":"register_interest","args":{"topic":"libsql internals"}}"#,
            );
            assert!(matches!(
                calls.first(),
                Some(ToolCall::RegisterInterest { .. })
            ));
            assert!(
                parse_tool_calls(r#"{"tool":"register_interest","args":{"topic":""}}"#).is_empty()
            );
        }
    }

    mod ping_policy {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')", ()).await;
            conn
        }

        fn apply_policy(conn: &turso::Connection, phase: &str, call: ToolCall) -> String {
            let options = ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "turn-1"),
                source_message_id: "message-1",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "semantic",
            };
            let mut events: Vec<Event> = Vec::new();
            apply_tool_call(call, conn, &options, &mut events).unwrap_or_default()
        }

        fn resume() -> ToolCall {
            ToolCall::SetPingPolicy {
                action: "resume".to_string(),
                topic: String::new(),
                until: String::new(),
            }
        }

        #[test]
        fn policy_parser_rejects_unknown_and_incomplete_actions() {
            assert!(matches!(
                parse_tool_calls(r#"{"tool":"set_ping_policy","args":{"action":"resume"}}"#)
                    .first(),
                Some(ToolCall::SetPingPolicy { .. })
            ));
            for input in [
                r#"{"tool":"set_ping_policy","args":{"action":"pause"}}"#,
                r#"{"tool":"set_ping_policy","args":{"action":"mute"}}"#,
                r#"{"tool":"set_ping_policy","args":{"action":"divert","until":"2026-07-15T00:00:00"}}"#,
            ] {
                assert!(parse_tool_calls(input).is_empty(), "{input}");
            }
        }

        #[test]
        fn policy_tool_is_seat_symmetric_and_denied_to_autonomous_seats() {
            let conn = db::block_on(harness());
            for phase in ["scribe", "extract", "consolidate"] {
                assert!(apply_policy(&conn, phase, resume()).contains("set_ping_policy"));
            }
            for phase in ["consolidation", "ingest"] {
                assert!(apply_policy(&conn, phase, resume()).contains("denied"));
            }
        }
    }

    mod node_contract {
        use super::super::{ApplyToolOptions, ToolCall, apply_preflighted_batch, apply_tool_call};
        use crate::memory::db;
        use crate::memory::graph::{Node, PROVENANCE_USER_CONFIRMED};
        use crate::memory::ledger::{Actor, EventContext};
        use turso::{Builder, Connection};

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";

        fn harness() -> Connection {
            db::block_on(async {
                let Ok(database) = Builder::new_local(":memory:").build().await else {
                    unreachable!("database builds")
                };
                let Ok(conn) = database.connect() else {
                    unreachable!("database connects")
                };
                let _ = conn.execute(NODES_DDL, ()).await;
                let _ = conn.execute(EDGES_DDL, ()).await;
                conn
            })
        }

        fn options() -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", "extract"),
                context: EventContext::new("session", "turn"),
                source_message_id: "message",
                event_payload: "",
                confidence_cap: None,
                provenance: PROVENANCE_USER_CONFIRMED,
                default_memory_type: "semantic",
            }
        }

        fn node(name: &str, node_type: &str, body: &str) -> ToolCall {
            ToolCall::UpsertNode {
                name: name.to_string(),
                node_type: node_type.to_string(),
                body: body.to_string(),
            }
        }

        fn edge(source: &str, target: &str, subject: bool) -> ToolCall {
            ToolCall::UpsertEdge {
                source: source.to_string(),
                target: target.to_string(),
                content: if subject {
                    String::new()
                } else {
                    "relates to".to_string()
                },
                stability: 0.7,
                confidence: 1.0,
                importance: 1.0,
                subject,
                occurred_at: None,
                memory_type: None,
                inferred_from: None,
            }
        }

        fn write_node(conn: &Connection, node_id: &str, node_type: &str, body: &str) {
            let stored = Node {
                node_type: node_type.to_string(),
                body: body.to_string(),
                date: String::new(),
            };
            assert!(db::write_node(conn, node_id, &stored).is_ok());
        }

        fn apply_one(conn: &Connection, call: ToolCall) -> String {
            let mut events = Vec::new();
            apply_tool_call(call, conn, &options(), &mut events).unwrap_or_default()
        }

        #[test]
        fn declared_nodes_are_fact_only() {
            let conn = harness();
            for denied_type in ["entity", "game", "anything"] {
                let result = apply_one(&conn, node(denied_type, denied_type, ""));
                assert!(
                    result.contains("only fact nodes may be declared"),
                    "{result}"
                );
            }
            let accepted = apply_one(&conn, node("claim", "fact", "one claim"));
            assert!(accepted.contains("upsert_node:"), "{accepted}");
            let Ok(Some(stored)) = db::load_node(&conn, "claim") else {
                unreachable!("fact exists")
            };
            assert_eq!(
                (stored.node_type.as_str(), stored.body.as_str()),
                ("fact", "one claim")
            );
            assert_eq!(stored.date, crate::prompts::today_iso());
        }

        #[test]
        fn semantic_edges_separate_learned_date_from_occurred_time() {
            let conn = harness();
            let mut historical = edge("release", "product", false);
            let ToolCall::UpsertEdge { occurred_at, .. } = &mut historical else {
                unreachable!("edge helper returns upsert_edge")
            };
            *occurred_at = Some("2022".to_string());
            assert!(apply_one(&conn, historical).contains("upsert_edge:"));
            let Ok(Some(stored)) = db::load_edge(&conn, "release", "product") else {
                unreachable!("historical edge exists")
            };
            assert_eq!(stored.date, crate::prompts::today_iso());
            assert_eq!(stored.occurred_at.as_deref(), Some("2022"));

            assert!(apply_one(&conn, edge("current", "fact", false)).contains("upsert_edge:"));
            let Ok(Some(current)) = db::load_edge(&conn, "current", "fact") else {
                unreachable!("current edge exists")
            };
            assert_eq!(current.date, crate::prompts::today_iso());
            assert_eq!(current.occurred_at, None);

            let mut episodic = edge("session", "event", false);
            let ToolCall::UpsertEdge {
                occurred_at,
                memory_type,
                ..
            } = &mut episodic
            else {
                unreachable!("edge helper returns upsert_edge")
            };
            *occurred_at = Some("2022".to_string());
            *memory_type = Some("episodic".to_string());
            assert!(apply_one(&conn, episodic).contains("upsert_edge:"));
            let Ok(Some(stored)) = db::load_edge(&conn, "session", "event") else {
                unreachable!("episodic edge exists")
            };
            assert_eq!(stored.occurred_at, None);
        }

        #[test]
        fn fact_collision_matrix_never_overwrites_other_node_types() {
            let conn = harness();
            write_node(&conn, "fact-id", "fact", "old");
            write_node(&conn, "entity-id", "entity", "");
            for node_type in ["game", "loop", "interest", "task"] {
                write_node(&conn, node_type, node_type, "protected");
            }
            assert!(apply_one(&conn, node("new-id", "fact", "new")).contains("upsert_node:"));
            assert!(apply_one(&conn, node("fact-id", "fact", "updated")).contains("upsert_node:"));
            let entity_denial = apply_one(&conn, node("entity-id", "fact", "claim"));
            assert!(
                entity_denial.contains("an entity is an identity"),
                "{entity_denial}"
            );
            for node_type in ["game", "loop", "interest", "task"] {
                let denial = apply_one(&conn, node(node_type, "fact", "overwrite"));
                assert!(
                    denial.contains(&format!("belongs to a {node_type} node")),
                    "{denial}"
                );
                let Ok(Some(stored)) = db::load_node(&conn, node_type) else {
                    unreachable!("protected node exists")
                };
                assert_eq!(
                    (stored.node_type.as_str(), stored.body.as_str()),
                    (node_type, "protected")
                );
            }
            let Ok(Some(entity)) = db::load_node(&conn, "entity-id") else {
                unreachable!("entity remains")
            };
            assert_eq!(entity.node_type, "entity");
        }

        #[test]
        fn preflight_applies_valid_calls_individually_and_facts_before_edges() {
            let conn = harness();
            let calls = vec![
                edge("later-fact", "topic", true),
                node("invalid", "entity", ""),
                node("later-fact", "fact", "the later declaration"),
                edge("plain-entity", "other", true),
            ];
            let mut events = Vec::new();
            let outcomes =
                apply_preflighted_batch(&conn, calls, &options(), &mut events).unwrap_or_default();
            assert!(outcomes[0].contains("upsert_edge:"), "{outcomes:?}");
            assert!(outcomes[1].contains("only fact nodes"), "{outcomes:?}");
            assert!(outcomes[2].contains("upsert_node:"), "{outcomes:?}");
            assert!(outcomes[3].contains("subject source"), "{outcomes:?}");
            assert!(
                db::load_edge(&conn, "later-fact", "topic")
                    .ok()
                    .flatten()
                    .is_some()
            );
            assert!(
                db::load_edge(&conn, "plain-entity", "other")
                    .ok()
                    .flatten()
                    .is_none()
            );
        }

        #[test]
        fn implicit_endpoints_create_entities_and_preserve_typed_nodes() {
            let conn = harness();
            assert!(apply_one(&conn, edge("left", "right", false)).contains("upsert_edge:"));
            for node_id in ["left", "right"] {
                let Ok(Some(stored)) = db::load_node(&conn, node_id) else {
                    unreachable!("implicit endpoint exists")
                };
                assert_eq!(stored.node_type, "entity");
                assert!(stored.body.is_empty());
            }
            write_node(&conn, "fact", "fact", "claim");
            write_node(&conn, "game", "game", "rules");
            assert!(apply_one(&conn, edge("fact", "game", false)).contains("upsert_edge:"));
            for (node_id, expected_type, expected_body) in
                [("fact", "fact", "claim"), ("game", "game", "rules")]
            {
                let Ok(Some(stored)) = db::load_node(&conn, node_id) else {
                    unreachable!("endpoint remains")
                };
                assert_eq!(
                    (stored.node_type.as_str(), stored.body.as_str()),
                    (expected_type, expected_body)
                );
            }
        }
    }

    mod loops {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::graph::{Edge, Node};
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            conn
        }

        fn options(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "chat-1:user:ts"),
                source_message_id: "chat-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn apply(conn: &turso::Connection, phase: &'static str, call: ToolCall) -> String {
            let mut events: Vec<Event> = Vec::new();
            let Ok(description) = apply_tool_call(call, conn, &options(phase), &mut events) else {
                unreachable!("apply succeeds")
            };
            description
        }

        fn open(intention: &str, anchor: &str, due: &str) -> ToolCall {
            ToolCall::OpenLoop {
                intention: intention.to_string(),
                anchor: anchor.to_string(),
                due: due.to_string(),
            }
        }

        fn close(loop_id: &str, resolution: &str, note: &str) -> ToolCall {
            ToolCall::CloseLoop {
                loop_id: loop_id.to_string(),
                resolution: resolution.to_string(),
                note: note.to_string(),
            }
        }

        fn write_anchor(conn: &turso::Connection, anchor: &str) {
            let node = Node {
                node_type: "entity".to_string(),
                body: String::new(),
                date: String::new(),
            };
            let Ok(()) = db::write_node(conn, anchor, &node) else {
                unreachable!("anchor writes")
            };
        }

        fn edge(conn: &turso::Connection, source: &str, target: &str) -> Edge {
            let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
                unreachable!("edge exists")
            };
            edge
        }

        fn write_raw_edge(conn: &turso::Connection, source: &str, target: &str, content: &str) {
            let raw_edge = Edge {
                source: source.to_string(),
                target: target.to_string(),
                content: content.to_string(),
                stability: 0.9,
                confidence: 0.9,
                importance: 1.0,
                subject: false,
                date: crate::prompts::today_iso(),
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
                ceiling: None,
                last_decayed_at: String::new(),
            };
            let Ok(()) = db::write_edge(conn, &raw_edge, "test") else {
                unreachable!("raw edge writes")
            };
        }

        #[test]
        fn loop_calls_parse_and_unknown_resolution_is_rejected() {
            let text = r#"{"tools":[{"tool":"open_loop","args":{"intention":"ask Markus","anchor":"home-server","due":"2026-07-15"}},{"tool":"close_loop","args":{"loop":"ask-markus","resolution":"done","note":"asked"}}]}"#;
            let calls = parse_tool_calls(text);
            assert_eq!(calls.len(), 2);
            assert!(matches!(calls[0], ToolCall::OpenLoop { .. }));
            assert!(matches!(calls[1], ToolCall::CloseLoop { .. }));
            let invalid = r#"{"tool":"close_loop","args":{"loop":"ask-markus","resolution":"forfeit","note":"no"}}"#;
            assert!(parse_tool_calls(invalid).is_empty());
        }

        #[test]
        fn open_loop_writes_fixed_semantic_node_and_edge() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            let intention = "ask Markus whether the systemd unit is running";
            let description = apply(
                &conn,
                "extract",
                open(intention, "home-server", "2026-07-15"),
            );
            assert!(description.contains("open_loop"), "{description}");
            let Ok(Some(node)) =
                db::load_node(&conn, "ask-markus-whether-the-systemd-unit-is-running")
            else {
                unreachable!("loop node exists")
            };
            assert_eq!(
                (
                    node.node_type.as_str(),
                    node.body.as_str(),
                    node.date.as_str()
                ),
                ("loop", intention, "2026-07-15")
            );
            let edge = edge(
                &conn,
                "ask-markus-whether-the-systemd-unit-is-running",
                "home-server",
            );
            assert_eq!(edge.content, format!("open loop: {intention}"));
            assert!((edge.confidence - 0.9).abs() < 1e-9);
            assert!((edge.stability - 0.9).abs() < 1e-9);
            assert_eq!(
                (edge.importance, edge.memory_type.as_str()),
                (1.0, "semantic")
            );
            assert_eq!(edge.provenance, "user-confirmed");
        }

        #[test]
        fn invalid_due_date_is_dropped_but_loop_still_opens() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(
                &conn,
                "extract",
                open("check the unit", "home-server", "Friday"),
            );
            let Ok(Some(node)) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("loop node exists")
            };
            assert!(node.date.is_empty());
            let loop_edge = edge(&conn, "check-the-unit", "home-server");
            assert!(loop_edge.expired_at.is_none());
            apply(
                &conn,
                "extract",
                open("check the calendar", "home-server", "2026-02-29"),
            );
            let Ok(Some(calendar)) = db::load_node(&conn, "check-the-calendar") else {
                unreachable!("calendar loop exists")
            };
            assert!(calendar.date.is_empty(), "non-leap February 29 is invalid");
        }

        #[test]
        fn close_loop_expires_anchor_with_code_built_resolution() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let description = apply(
                &conn,
                "consolidate",
                close("check-the-unit", "done", "confirmed running"),
            );
            assert!(description.contains("close_loop"), "{description}");
            let loop_edge = edge(&conn, "check-the-unit", "home-server");
            let expected = format!(
                "done: confirmed running (closed {})",
                crate::prompts::today_iso()
            );
            assert_eq!(loop_edge.content, expected);
            assert!(loop_edge.expired_at.is_some());
        }

        #[test]
        fn unknown_loop_returns_a_friendly_denial() {
            let conn = db::block_on(harness());
            let description = apply(&conn, "extract", close("missing-loop", "done", "finished"));
            assert!(description.contains("denied"));
            assert!(description.contains("not found"));
        }

        #[test]
        // Scribe is allowed: intention statements legitimately classify as
        // task turns (live evidence 2026-07-12), so loop tools are
        // seat-symmetric like record_outcome. Dream/ingest stay denied.
        fn dream_and_ingest_seats_cannot_mutate_loops() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            for phase in ["consolidation", "ingest"] {
                let opened = apply(&conn, phase, open("check the unit", "home-server", ""));
                let closed = apply(&conn, phase, close("check-the-unit", "done", "finished"));
                assert!(opened.contains("denied"), "{phase}: {opened}");
                assert!(closed.contains("denied"), "{phase}: {closed}");
            }
            let Ok(node) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("node lookup succeeds")
            };
            assert!(node.is_none());

            let opened = apply(&conn, "scribe", open("check the unit", "home-server", ""));
            assert!(opened.contains("open_loop:"), "{opened}");
            let closed = apply(&conn, "scribe", close("check-the-unit", "done", "finished"));
            assert!(closed.contains("close_loop:"), "{closed}");
        }

        #[test]
        fn open_loop_creates_a_missing_anchor() {
            let conn = db::block_on(harness());
            let description = apply(&conn, "extract", open("check the unit", "missing", ""));
            assert!(description.contains("open_loop:"), "{description}");
            let Ok(Some(anchor)) = db::load_node(&conn, "missing") else {
                unreachable!("anchor exists")
            };
            assert_eq!(anchor.node_type, "entity");
            assert!(anchor.body.is_empty());
        }

        #[test]
        fn open_loop_preserves_an_existing_typed_anchor() {
            let conn = db::block_on(harness());
            let fact = Node {
                node_type: "fact".to_string(),
                body: "the service is important".to_string(),
                date: String::new(),
            };
            assert!(db::write_node(&conn, "home-server", &fact).is_ok());
            let opened = apply(&conn, "extract", open("check the unit", "home-server", ""));
            assert!(opened.contains("open_loop:"), "{opened}");
            let Ok(Some(anchor)) = db::load_node(&conn, "home-server") else {
                unreachable!("anchor remains")
            };
            assert_eq!(
                (anchor.node_type.as_str(), anchor.body.as_str()),
                ("fact", "the service is important")
            );
        }

        #[test]
        fn open_loop_denies_node_collisions_and_self_anchors() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "check-the-unit");
            let collision = apply(&conn, "extract", open("check the unit", "home-server", ""));
            assert!(collision.contains("non-loop node"), "{collision}");
            let self_anchor = apply(&conn, "extract", open("home server", "home-server", ""));
            assert!(self_anchor.contains("same id"), "{self_anchor}");
            let Ok(Some(node)) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("colliding entity remains")
            };
            assert_eq!(node.node_type, "entity");
        }

        #[test]
        fn reopening_cannot_add_a_second_live_anchor() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "other-topic");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let denied = apply(&conn, "extract", open("check the unit", "other-topic", ""));
            assert!(
                denied.contains("already has anchor home-server"),
                "{denied}"
            );
            let Ok(missing) = db::load_edge(&conn, "check-the-unit", "other-topic") else {
                unreachable!("edge lookup succeeds")
            };
            assert!(missing.is_none());
        }

        #[test]
        fn close_loop_expires_only_actual_anchor_edges() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "other-topic");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            write_raw_edge(&conn, "check-the-unit", "other-topic", "related fact");
            apply(
                &conn,
                "extract",
                close("check-the-unit", "done", "finished"),
            );
            assert!(
                edge(&conn, "check-the-unit", "home-server")
                    .expired_at
                    .is_some()
            );
            assert!(
                edge(&conn, "check-the-unit", "other-topic")
                    .expired_at
                    .is_none()
            );
        }

        #[test]
        fn generic_mutations_cannot_bypass_loop_lifecycle() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let expire = ToolCall::ExpireEdge {
                source: "check-the-unit".to_string(),
                target: "home-server".to_string(),
                rewritten_content: "gone".to_string(),
            };
            let denied = apply(&conn, "ingest", expire);
            assert!(denied.contains("loop lifecycle"), "{denied}");
            assert!(
                edge(&conn, "check-the-unit", "home-server")
                    .expired_at
                    .is_none()
            );
        }
    }
