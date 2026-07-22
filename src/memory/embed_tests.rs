    use std::collections::HashMap;

    use super::{
        CHAT_INITIAL_K, SemanticEdge, edges_to_toon, edges_to_toon_with_nodes, interest_bonus,
        rank_by_exploration, rank_pool_for_chat, satiation_factor, scope_episodic,
    };
    use crate::memory::db::InterestTarget;
    use crate::memory::graph::Node;

    fn test_edge(memory_type: &str, origin_session: &str) -> SemanticEdge {
        SemanticEdge {
            source: "a".to_string(),
            target: "b".to_string(),
            content: String::new(),
            confidence: 1.0,
            stability: 0.5,
            importance: 0.5,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
            memory_type: memory_type.to_string(),
            origin_session: origin_session.to_string(),
            provenance: "legacy".to_string(),
            embedding: None,
        }
    }

    #[test]
    fn initial_chat_ranking_is_capped_separately_from_candidate_pool() {
        let mut edges = Vec::new();
        for index in 0..(CHAT_INITIAL_K + 5) {
            let mut edge = test_edge("semantic", "");
            edge.source = format!("source-{index}");
            edge.target = format!("target-{index}");
            edges.push(edge);
        }
        let ranked = rank_pool_for_chat(edges, &[]);
        assert_eq!(ranked.len(), CHAT_INITIAL_K);
    }

    #[test]
    fn same_session_episodic_edge_survives_scoping() {
        let edges = vec![test_edge("episodic", "game-1")];
        let scoped = scope_episodic(edges, Some("game-1"));
        assert_eq!(scoped.len(), 1);
    }

    #[test]
    fn injected_context_includes_provenance() {
        let mut edge = test_edge("semantic", "");
        edge.provenance = "dreamed".to_string();
        let toon = edges_to_toon(&[edge]);
        assert!(toon.contains("provenance"));
        assert!(toon.contains("dreamed"));
    }

    #[test]
    fn injected_context_renders_learned_and_occurred_dates() {
        let mut edge = test_edge("semantic", "");
        edge.date = crate::prompts::today_iso();
        edge.occurred_at = Some("2022".to_string());
        let toon = edges_to_toon(&[edge]);
        assert!(toon.contains("date"));
        assert!(toon.contains("occurred_at"));
        assert!(toon.contains("2022"));
    }

    #[test]
    fn injected_context_is_source_first_and_groups_source_rows() {
        let mut second_kratt = test_edge("semantic", "");
        second_kratt.source = "kratt".to_string();
        second_kratt.target = "zeta".to_string();
        second_kratt.content = "second".to_string();
        let mut other = test_edge("semantic", "");
        other.source = "other".to_string();
        other.target = "middle".to_string();
        other.content = "other".to_string();
        let mut first_kratt = test_edge("semantic", "");
        first_kratt.source = "kratt".to_string();
        first_kratt.target = "alpha".to_string();
        first_kratt.content = "first".to_string();

        let toon = edges_to_toon(&[second_kratt, other, first_kratt]);
        let lines = toon.lines().collect::<Vec<_>>();

        assert!(lines[0].starts_with("edges[3]{source,target,content,"));
        assert!(lines[1].starts_with("  kratt,alpha,first,"));
        assert!(lines[2].starts_with("  kratt,zeta,second,"));
        assert!(lines[3].starts_with("  other,middle,other,"));
        assert!(toon.contains("nodes[0]{id,type,body,date}:"));
    }

    #[test]
    fn bounded_toon_includes_fact_endpoint_body() {
        let edge = test_edge("semantic", "");
        let nodes = HashMap::from([
            (
                "a".to_string(),
                Node {
                    node_type: "fact".to_string(),
                    body: "Atomic sentence".to_string(),
                    date: "2026-07-17".to_string(),
                },
            ),
            (
                "b".to_string(),
                Node {
                    node_type: "entity".to_string(),
                    body: String::new(),
                    date: String::new(),
                },
            ),
        ]);
        let first = edges_to_toon_with_nodes(std::slice::from_ref(&edge), &nodes);
        let second = edges_to_toon_with_nodes(std::slice::from_ref(&edge), &nodes);
        assert!(first.contains("Atomic sentence"));
        assert!(!first.contains("\"b\""));
        assert_eq!(first, second);
    }

    #[test]
    fn bounded_toon_has_empty_nodes_for_mute_entities() {
        let edge = test_edge("semantic", "");
        let nodes = HashMap::from([
            (
                "a".to_string(),
                Node {
                    node_type: "entity".to_string(),
                    body: " ".to_string(),
                    date: String::new(),
                },
            ),
            (
                "b".to_string(),
                Node {
                    node_type: "entity".to_string(),
                    body: String::new(),
                    date: String::new(),
                },
            ),
        ]);
        let toon = edges_to_toon_with_nodes(&[edge], &nodes);
        assert!(toon.contains("nodes[0]"));
    }

    #[test]
    fn cross_session_episodic_edge_is_filtered_out() {
        let edges = vec![test_edge("episodic", "game-1")];
        let scoped = scope_episodic(edges, Some("game-2"));
        assert!(scoped.is_empty());
    }

    #[test]
    fn no_session_context_excludes_all_episodic_edges() {
        let edges = vec![test_edge("episodic", "game-1")];
        let scoped = scope_episodic(edges, None);
        assert!(scoped.is_empty());
    }

    #[test]
    fn semantic_edges_are_never_filtered() {
        let edges = vec![test_edge("semantic", "")];
        let scoped = scope_episodic(edges, Some("game-2"));
        assert_eq!(scoped.len(), 1);
    }

    #[test]
    fn satiation_factor_decreases_but_never_reaches_zero() {
        assert_eq!(satiation_factor(0), 1.0);
        assert!(satiation_factor(1) < satiation_factor(0));
        assert!(satiation_factor(4) < satiation_factor(1));
        assert!(satiation_factor(usize::MAX) > 0.0);
    }

    #[test]
    fn satiation_shifts_ranking_without_removing_candidate() {
        let mut penalized = test_edge("semantic", "");
        penalized.source = "penalized".to_string();
        penalized.target = "region".to_string();
        let mut untouched = test_edge("semantic", "");
        untouched.source = "untouched".to_string();
        untouched.target = "region".to_string();
        let mut attention = HashMap::new();
        attention.insert(("penalized".to_string(), "region".to_string()), 8);

        let ranked =
            rank_by_exploration(vec![penalized, untouched], &HashMap::new(), &attention, &[]);

        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].source, "untouched");
        let mut found_penalized = false;
        for edge in ranked {
            if edge.source == "penalized" {
                found_penalized = true;
            }
        }
        assert!(found_penalized);
    }

    #[test]
    fn sparse_interest_pulls_nearby_candidate_more_strongly() {
        let mut nearby = test_edge("semantic", "");
        nearby.embedding = Some(vec![1.0, 0.0]);
        let sparse = InterestTarget {
            topic: "libsql internals".to_string(),
            density: 0,
            embedding: vec![1.0, 0.0],
        };
        let dense = InterestTarget {
            topic: "rust".to_string(),
            density: 8,
            embedding: vec![1.0, 0.0],
        };
        assert!(interest_bonus(&nearby, &[sparse]) > interest_bonus(&nearby, &[dense]));

        let mut outside = nearby.clone();
        outside.embedding = Some(vec![0.0, 1.0]);
        let target = InterestTarget {
            topic: "libsql internals".to_string(),
            density: 0,
            embedding: vec![1.0, 0.0],
        };
        assert!(
            interest_bonus(&nearby, std::slice::from_ref(&target))
                > interest_bonus(&outside, &[target])
        );

        outside.source = "outside".to_string();
        nearby.source = "nearby".to_string();
        let target = InterestTarget {
            topic: "libsql internals".to_string(),
            density: 0,
            embedding: vec![1.0, 0.0],
        };
        let ranked = rank_by_exploration(
            vec![outside, nearby],
            &HashMap::new(),
            &HashMap::new(),
            &[target],
        );
        assert_eq!(ranked[0].source, "nearby");
    }
