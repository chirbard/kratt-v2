    #[test]
    fn dream_graph_context_prefers_importance_and_respects_budget() {
        let mut graph = Graph::new();
        let mut edges = Vec::new();
        for index in 0..3000 {
            edges.push(Edge {
                source: format!("s{index:05}"),
                target: format!("t{index:05}"),
                content: "x".repeat(200),
                stability: 0.3,
                confidence: 0.5,
                importance: if index == 0 { 1.0 } else { 0.1 },
                date: String::new(),
                subject: false,
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
                ceiling: None,
                last_decayed_at: String::new(),
            });
        }
        graph.edges = edges;
        let ranked = super::ranked_live_edges(&graph);
        assert_eq!(ranked[0].source, "s00000");
        let (selected, _nodes) = super::pack_dream_edges(&ranked, &graph.nodes);
        assert!(!selected.is_empty());
        assert!(selected.len() < 3000);
        assert_eq!(selected[0].source, "s00000");
        let toon_a = super::bounded_global_toon(&graph);
        let toon_b = super::bounded_global_toon(&graph);
        assert_eq!(toon_a, toon_b);
        assert!(super::super::heavy::estimate_tokens(&toon_a) <= super::DREAM_GRAPH_BUDGET_TOKENS + 2000);
    }

    use super::{
        DreamApplyState, DreamCallContext, DreamGistCall, DreamNodeCall, Edge, Event, Graph,
        SynthesisPolicy, apply_dream_gist_edge, apply_dream_upsert_node, dream_edge_values,
    };
    use crate::memory::agent::{self, ToolCall};
    use crate::memory::db;
    use crate::memory::ledger::{Actor, EventContext};
    use std::collections::{HashMap, HashSet};
    use turso::{Builder, Connection};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";

    fn harness() -> Connection {
        db::block_on(async {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn.execute(NODES_DDL, ()).await;
            conn
        })
    }

    fn edge(source: &str, target: &str, memory_type: &str) -> Edge {
        Edge {
            source: source.to_string(),
            target: target.to_string(),
            content: "x".to_string(),
            stability: 0.2,
            confidence: 0.3,
            importance: 0.1,
            subject: false,
            date: "2020-01-01".to_string(),
            occurred_at: None,
            expired_at: None,
            memory_type: memory_type.to_string(),
            origin_session: "session-1".to_string(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    #[test]
    fn prune_guard_uses_belief_and_structure_not_durability_proxies() {
        let mut candidate = edge("fact", "topic", "semantic");
        candidate.stability = 0.95;
        candidate.importance = 1.0;
        assert!(super::prune_block_reason(&candidate, "fact", 0).is_none());

        candidate.subject = true;
        candidate.importance = 0.2;
        assert_eq!(
            super::prune_block_reason(&candidate, "fact", 0).as_deref(),
            Some("subject edge")
        );

        candidate.subject = false;
        candidate.confidence = super::PRUNE_GUARD_THRESHOLD;
        assert!(
            super::prune_block_reason(&candidate, "fact", 0)
                .is_some_and(|reason| reason.starts_with("confidence"))
        );

        candidate.confidence = 0.3;
        assert_eq!(
            super::prune_block_reason(&candidate, "loop", 0).as_deref(),
            Some("pinned edge")
        );
    }

    fn insert(conn: &Connection, source: &str, target: &str, memory_type: &str) {
        let stored = edge(source, target, memory_type);
        let Ok(()) = db::write_edge(conn, &stored, "") else {
            unreachable!("edge written")
        };
    }

    fn gist_call(supports: &[(&str, &str)], confidence: f64) -> DreamGistCall {
        let mut supported_by = Vec::new();
        for (source, target) in supports {
            supported_by.push(((*source).to_string(), (*target).to_string()));
        }
        DreamGistCall {
            source: "queen-sorties".to_string(),
            target: "chess-openings".to_string(),
            content: "early queen sorties get punished".to_string(),
            confidence,
            stability: 0.5,
            supported_by,
        }
    }

    fn empty_policy() -> SynthesisPolicy {
        SynthesisPolicy {
            synthesis_nodes: HashSet::new(),
            edge_overrides: HashMap::new(),
        }
    }

    fn run_gist(conn: &Connection, call: DreamGistCall) -> ((String, bool), Vec<Event>) {
        let policy = empty_policy();
        let mut events: Vec<Event> = Vec::new();
        let mut apply_state = DreamApplyState::new(&policy);
        let mut ctx = DreamCallContext {
            conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let Ok(outcome) = apply_dream_gist_edge(&mut ctx, call) else {
            unreachable!("apply runs")
        };
        (outcome, events)
    }

    fn loaded(conn: &Connection, source: &str, target: &str) -> Option<Edge> {
        let Ok(edge) = db::load_edge(conn, source, target) else {
            unreachable!("load runs")
        };
        edge
    }

    fn must_load(conn: &Connection, source: &str, target: &str) -> Edge {
        let Some(edge) = loaded(conn, source, target) else {
            unreachable!("edge exists")
        };
        edge
    }

    // Two 0.3-confidence supports, proposed at 0.6 and capped at mean × 0.9.
    fn gist_two_supports(conn: &Connection) -> Vec<Event> {
        insert(conn, "game3-hub", "move-qh5", "episodic");
        insert(conn, "game4-hub", "move-qf3", "episodic");
        let ((description, added), events) = run_gist(
            conn,
            gist_call(&[("game3-hub", "move-qh5"), ("game4-hub", "move-qf3")], 0.6),
        );
        assert!(added, "a gist edge was written");
        assert!(description.starts_with("gist_edge:"));
        events
    }

    #[test]
    fn two_episodic_supports_write_a_semantic_gist() {
        let conn = harness();
        let _ = gist_two_supports(&conn);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.memory_type, "semantic");
        assert_eq!(written.date, crate::prompts::today_iso());
        assert!((written.confidence - 0.27).abs() < 1e-9);
        assert_eq!(written.provenance, "synthesized");
        assert_eq!(written.ceiling, Some(0.27));
    }

    #[test]
    fn a_gist_records_provenance_and_spares_its_supporters() {
        let conn = harness();
        let events = gist_two_supports(&conn);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "updated");
        assert!(events[0].payload.contains("gist_of"));
        assert!(events[0].payload.contains("game3-hub → move-qh5"));
        assert!(events[0].payload.contains("game4-hub → move-qf3"));

        // Supporters are left untouched — still episodic, not expired.
        let support = must_load(&conn, "game3-hub", "move-qh5");
        assert_eq!(support.memory_type, "episodic");
        assert!(support.expired_at.is_none());
    }

    #[test]
    fn gist_confidence_is_clamped_to_the_inherited_ceiling() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "game4-hub", "move-qf3", "episodic");
        let (_, _) = run_gist(
            &conn,
            gist_call(
                &[("game3-hub", "move-qh5"), ("game4-hub", "move-qf3")],
                0.95,
            ),
        );
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert!((written.confidence - 0.27).abs() < 1e-9);
    }

    #[test]
    fn a_single_resolvable_support_can_synthesize() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), events) =
            run_gist(&conn, gist_call(&[("game3-hub", "move-qh5")], 0.6));
        assert!(added);
        assert!(description.contains("from 1 supports"));
        assert_eq!(events.len(), 1);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "synthesized");
        assert_eq!(written.ceiling, Some(0.27));
    }

    #[test]
    fn an_unknown_ref_is_skipped_when_another_support_resolves() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        let ((description, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("ghost", "nowhere")], 0.6),
        );
        assert!(added);
        assert!(description.contains("from 1 supports"));
    }

    #[test]
    fn a_semantic_support_contributes_to_a_gist() {
        let conn = harness();
        insert(&conn, "game3-hub", "move-qh5", "episodic");
        insert(&conn, "old-belief", "chess", "semantic");
        let ((_, added), _) = run_gist(
            &conn,
            gist_call(&[("game3-hub", "move-qh5"), ("old-belief", "chess")], 0.6),
        );
        assert!(added);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "synthesized");
    }

    #[test]
    fn gist_ceiling_uses_the_mean_of_mixed_supporters() {
        let conn = harness();
        insert(&conn, "episode", "fact", "episodic");
        insert(&conn, "belief", "fact", "semantic");
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9 WHERE source = 'belief'",
            (),
        ));
        let _ = run_gist(
            &conn,
            gist_call(&[("episode", "fact"), ("belief", "fact")], 0.9),
        );
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.ceiling, Some(0.54));
        assert_eq!(written.confidence, 0.54);
    }

    #[test]
    fn zero_resolvable_gist_supports_fall_back_to_dreamed() {
        let conn = harness();
        let ((_, added), _) = run_gist(&conn, gist_call(&[("ghost", "nowhere")], 0.9));
        assert!(added);
        let written = must_load(&conn, "queen-sorties", "chess-openings");
        assert_eq!(written.provenance, "dreamed");
        assert_eq!(written.ceiling, None);
        assert_eq!(written.confidence, 0.25);
        assert_eq!(written.stability, 0.15);
    }

    #[test]
    fn dream_inference_inherits_the_dream_premise_floor() {
        let conn = harness();
        let mut first = edge("dream-a", "fact", "semantic");
        first.provenance = "dreamed".to_string();
        first.confidence = 0.25;
        let mut second = edge("dream-b", "fact", "semantic");
        second.provenance = "dreamed".to_string();
        second.confidence = 0.25;
        assert!(db::write_edge(&conn, &first, "").is_ok());
        assert!(db::write_edge(&conn, &second, "").is_ok());
        let refs = Some(vec![
            ("dream-a".to_string(), "fact".to_string()),
            ("dream-b".to_string(), "fact".to_string()),
        ]);
        let values = dream_edge_values(
            &empty_policy(),
            &("result".to_string(), "fact".to_string()),
            None,
            &super::DreamWeights {
                stability: 0.9,
                confidence: 0.9,
            },
            super::dream_declared_ceiling(&conn, &refs).unwrap_or_default(),
        );
        assert_eq!(values.2, "inferred");
        assert_eq!(values.3, Some(0.25));
        assert_eq!(values.0, 0.3);
    }

    #[test]
    fn dream_creation_clamps_model_stability_to_dream_seed() {
        let values = dream_edge_values(
            &empty_policy(),
            &("dream".to_string(), "fact".to_string()),
            None,
            &super::DreamWeights {
                stability: 0.9,
                confidence: 0.9,
            },
            None,
        );
        assert_eq!(values.0, 0.15);
        assert_eq!(values.2, "dreamed");
    }

    #[test]
    fn synthesis_policy_remains_the_only_non_fact_declaration_exception() {
        let conn = harness();
        let mut policy = empty_policy();
        policy.synthesis_nodes.insert("higher-order".to_string());
        let mut apply_state = DreamApplyState::new(&policy);
        let mut events = Vec::new();
        let ctx = DreamCallContext {
            conn: &conn,
            source_id: "dream:test",
            events: &mut events,
            apply_state: &mut apply_state,
        };
        let call = DreamNodeCall::new(
            "higher-order".to_string(),
            "synthesis".to_string(),
            "a supported synthesis".to_string(),
            String::new(),
        );
        let Ok((description, changed_edge)) = apply_dream_upsert_node(&ctx, call) else {
            unreachable!("synthesis applies")
        };
        assert!(description.contains("(synthesis)"), "{description}");
        assert!(!changed_edge);
        let Ok(Some(stored)) = db::load_node(&conn, "higher-order") else {
            unreachable!("synthesis exists")
        };
        assert_eq!(stored.node_type, "synthesis");
    }

    #[test]
    fn the_chat_apply_path_denies_gist_edge() {
        let conn = harness();
        let call = ToolCall::GistEdge {
            source: "a".to_string(),
            target: "b".to_string(),
            content: "c".to_string(),
            confidence: 0.6,
            stability: 0.5,
            supported_by: vec![("x".to_string(), "y".to_string())],
        };
        let options = agent::ApplyToolOptions {
            actor: Actor::new("system", "chat"),
            context: EventContext::new("session-1", "turn-1"),
            source_message_id: "msg-1",
            event_payload: "",
            confidence_cap: None,
            provenance: "user-confirmed",
            default_memory_type: "semantic",
        };
        let mut events: Vec<Event> = Vec::new();
        let Ok(description) = agent::apply_tool_call(call, &conn, &options, &mut events) else {
            unreachable!("apply runs")
        };
        assert!(description.contains("gist_edge is a dreaming tool"));
        assert!(events.is_empty());
    }
