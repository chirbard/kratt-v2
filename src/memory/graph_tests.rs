    use std::collections::HashMap;

    use super::{
        CLOSURE_NODE_CAP, Edge, Node, PROVENANCE_DREAMED, PROVENANCE_INFERRED,
        PROVENANCE_LEGACY, PROVENANCE_SYNTHESIZED, PROVENANCE_USER_CONFIRMED,
        PROVENANCE_WEB_CONFIRMED, PROVENANCE_WEB_SURFACED, closure_nodes, edge_ceiling,
        edge_is_pinned, inferred_ceiling, node_carries_value, normalize_memory_type,
        provenance_ceiling, stability_seed_max,
    };

    fn node(node_type: &str, body: &str) -> Node {
        Node {
            node_type: node_type.to_string(),
            body: body.to_string(),
            date: String::new(),
        }
    }

    #[test]
    fn value_predicate_only_mutes_empty_entities() {
        for node_type in ["fact", "synthesis", "loop", "task"] {
            assert!(node_carries_value(&node(node_type, "")));
        }
        assert!(node_carries_value(&node("entity", "meaningful")));
        assert!(!node_carries_value(&node("entity", "")));
        assert!(!node_carries_value(&node("entity", " \n\t")));
    }

    #[test]
    fn closure_is_value_bearing_sorted_deduplicated_and_bounded() {
        let nodes = HashMap::from([
            ("a".to_string(), node("fact", "A fact")),
            ("b".to_string(), node("entity", "")),
            ("c".to_string(), node("loop", "")),
            ("outside".to_string(), node("fact", "not selected")),
        ]);
        let endpoints = vec![
            ("c".to_string(), "b".to_string()),
            ("a".to_string(), "c".to_string()),
        ];
        let closure = closure_nodes(&endpoints, &nodes);
        let mut node_ids = Vec::new();
        for (node_id, _) in &closure {
            node_ids.push(node_id.as_str());
        }
        assert_eq!(node_ids, vec!["a", "c"]);
    }

    #[test]
    fn closure_cap_prefers_earlier_edge_endpoints() {
        let mut nodes = HashMap::new();
        let mut endpoints = Vec::new();
        let total = CLOSURE_NODE_CAP + 2;
        for index in 0..total {
            let node_id = format!("node-{index:02}");
            nodes.insert(node_id.clone(), node("fact", &node_id));
            endpoints.push((node_id, "mute".to_string()));
        }
        nodes.insert("mute".to_string(), node("entity", ""));
        let closure = closure_nodes(&endpoints, &nodes);
        assert_eq!(closure.len(), CLOSURE_NODE_CAP);
        let mut includes_first = false;
        let mut includes_over_cap = false;
        for (node_id, _) in &closure {
            includes_first |= node_id == "node-00";
            includes_over_cap |= node_id == "node-40";
        }
        assert!(includes_first);
        assert!(!includes_over_cap);
    }

    #[test]
    fn normalize_memory_type_accepts_episodic() {
        assert_eq!(normalize_memory_type("episodic"), "episodic");
    }

    #[test]
    fn normalize_memory_type_accepts_semantic() {
        assert_eq!(normalize_memory_type("semantic"), "semantic");
    }

    #[test]
    fn normalize_memory_type_rejects_garbage_to_semantic() {
        assert_eq!(normalize_memory_type("durable"), "semantic");
        assert_eq!(normalize_memory_type(""), "semantic");
        assert_eq!(normalize_memory_type("Episodic"), "semantic");
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn provenance_tiers_have_structural_ceilings() {
        assert_eq!(provenance_ceiling(PROVENANCE_USER_CONFIRMED), 1.0);
        assert_eq!(provenance_ceiling(PROVENANCE_WEB_CONFIRMED), 0.8);
        assert_eq!(provenance_ceiling(PROVENANCE_WEB_SURFACED), 0.5);
        assert_eq!(provenance_ceiling(PROVENANCE_DREAMED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_INFERRED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_SYNTHESIZED), 0.25);
        assert_eq!(provenance_ceiling(PROVENANCE_LEGACY), 0.5);
        assert_eq!(provenance_ceiling(""), 0.5);
        assert_eq!(provenance_ceiling("future-tier"), 0.5);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn provenance_tiers_have_stability_seed_maxima() {
        assert_eq!(stability_seed_max(PROVENANCE_USER_CONFIRMED), 0.5);
        assert_eq!(stability_seed_max(PROVENANCE_WEB_CONFIRMED), 0.5);
        assert_eq!(stability_seed_max(PROVENANCE_WEB_SURFACED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_INFERRED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_SYNTHESIZED), 0.3);
        assert_eq!(stability_seed_max(PROVENANCE_DREAMED), 0.15);
        assert_eq!(stability_seed_max(PROVENANCE_LEGACY), 0.3);
        assert_eq!(stability_seed_max("future-tier"), 0.3);
    }

    #[test]
    fn edge_ceiling_prefers_the_stored_cap() {
        assert_eq!(edge_ceiling(PROVENANCE_USER_CONFIRMED, None), 1.0);
        assert_eq!(edge_ceiling(PROVENANCE_SYNTHESIZED, Some(0.72)), 0.72);
    }

    #[test]
    fn derived_ceiling_math_uses_premise_strength_and_floor() {
        assert_eq!(inferred_ceiling(&[1.0, 0.9]), Some(0.81));
        assert_eq!(inferred_ceiling(&[0.25, 0.25]), Some(0.25));
        assert_eq!(inferred_ceiling(&[]), None);
    }

    #[test]
    fn only_live_semantic_loop_source_edges_are_pinned() {
        let mut edge = Edge {
            source: "loop".to_string(),
            target: "topic".to_string(),
            content: "open loop: check service".to_string(),
            stability: 0.9,
            confidence: 0.9,
            importance: 1.0,
            date: String::new(),
            subject: false,
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: PROVENANCE_USER_CONFIRMED.to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        };
        assert!(edge_is_pinned(&edge, "loop"));
        assert!(!edge_is_pinned(&edge, "interest"));
        assert!(!edge_is_pinned(&edge, "fact"));
        edge.target = "loop".to_string();
        assert!(!edge_is_pinned(&edge, "fact"));

        edge.memory_type = "episodic".to_string();
        assert!(!edge_is_pinned(&edge, "loop"));
        edge.memory_type = "semantic".to_string();
        edge.expired_at = Some("2026-07-20".to_string());
        assert!(!edge_is_pinned(&edge, "loop"));
    }
