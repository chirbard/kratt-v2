    use super::{HeavyInputs, estimate_tokens, select_with_nearest};
    use crate::memory::embed::semantic_edge;
    use crate::memory::graph::{Edge, Graph, Node, PROVENANCE_LEGACY};

    fn edge(source: &str, target: &str, importance: f64) -> Edge {
        Edge {
            source: source.to_string(),
            target: target.to_string(),
            content: format!("{source} relates to {target}"),
            stability: 0.5,
            confidence: 0.5,
            importance,
            date: String::new(),
            subject: false,
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: PROVENANCE_LEGACY.to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    fn graph(edges: Vec<Edge>) -> Graph {
        let mut graph = Graph::new();
        for edge in &edges {
            graph.nodes.insert(edge.source.clone(), entity());
            graph.nodes.insert(edge.target.clone(), entity());
        }
        graph.edges = edges;
        graph
    }

    fn entity() -> Node {
        Node {
            node_type: "entity".to_string(),
            body: String::new(),
            date: String::new(),
        }
    }

    #[test]
    fn lexical_semantic_hop_and_contradiction_tiers_are_deduplicated() {
        let exact = edge("rust", "compiler", 0.2);
        let near = edge("vector", "anchor", 0.2);
        let hop = edge("anchor", "neighbor", 0.2);
        let contradiction = edge("old-belief", "risk", 0.9);
        let graph = graph(vec![exact, near.clone(), hop, contradiction.clone()]);
        let mut nearest = vec![semantic_edge(&near)];
        while nearest.len() < super::super::embed::TOP_K {
            nearest.push(semantic_edge(&edge(
                &format!("missing-{}", nearest.len()),
                "none",
                0.0,
            )));
        }
        nearest.push(semantic_edge(&contradiction));
        let input = HeavyInputs {
            graph: &graph,
            injected: &[],
            exchange: "Rust vectors",
            session_id: None,
            active_ids: &[],
            reserved_text: "",
        };
        let selection = select_with_nearest(&input, &nearest, 10_000);
        assert_eq!(selection.tier_fill, [0, 1, 2, 1]);
        assert_eq!(selection.edge_count, 4);
    }

    #[test]
    fn mandatory_edge_survives_before_semantic_filler() {
        let mandatory = edge("must", "stay", 0.2);
        let filler = edge("semantic", "filler", 0.2);
        let graph = graph(vec![mandatory.clone(), filler.clone()]);
        let input = HeavyInputs {
            graph: &graph,
            injected: std::slice::from_ref(&mandatory),
            exchange: "",
            session_id: None,
            active_ids: &[],
            reserved_text: "",
        };
        let mandatory_only = select_with_nearest(&input, &[], 10_000);
        let budget = mandatory_only.estimated_tokens;
        let selection = select_with_nearest(&input, &[semantic_edge(&filler)], budget);
        assert_eq!(selection.tier_fill[0], 1);
        assert_eq!(selection.tier_fill[2], 0);
    }

    #[test]
    fn value_node_that_exceeds_remaining_budget_skips_the_whole_edge() {
        let first = edge("a", "b", 0.2);
        let second = edge("c", "large-fact", 0.2);
        let mut graph = graph(vec![first.clone(), second.clone()]);
        graph.nodes.insert(
            "large-fact".to_string(),
            Node {
                node_type: "fact".to_string(),
                body: "x".repeat(800),
                date: String::new(),
            },
        );
        let input = HeavyInputs {
            graph: &graph,
            injected: &[first.clone(), second],
            exchange: "",
            session_id: None,
            active_ids: &[],
            reserved_text: "",
        };
        let first_only = HeavyInputs {
            injected: std::slice::from_ref(&first),
            ..input
        };
        let budget = select_with_nearest(&first_only, &[], 10_000).estimated_tokens + 5;
        let selection = select_with_nearest(&input, &[], budget);
        assert_eq!(selection.edge_count, 1);
        assert!(!selection.toon.contains("large-fact"));
        assert!(selection.estimated_tokens <= budget);
    }

    #[test]
    fn identical_inputs_serialize_identically() {
        let graph = graph(vec![edge("b", "c", 0.2), edge("a", "b", 0.2)]);
        let input = HeavyInputs {
            graph: &graph,
            injected: &[],
            exchange: "a",
            session_id: None,
            active_ids: &[],
            reserved_text: "",
        };
        let first = select_with_nearest(&input, &[], 10_000);
        let second = select_with_nearest(&input, &[], 10_000);
        assert_eq!(first.toon, second.toon);
        assert_eq!(estimate_tokens(&first.toon), estimate_tokens(&second.toon));
    }

    #[test]
    fn packed_selection_does_not_use_the_chat_closure_cap() {
        let mut edges = Vec::new();
        let mut index = 0usize;
        while index < 45 {
            edges.push(edge("hub", &format!("fact-{index:02}"), 0.2));
            index += 1;
        }
        let mut graph = graph(edges.clone());
        for edge in &edges {
            graph.nodes.insert(
                edge.target.clone(),
                Node {
                    node_type: "fact".to_string(),
                    body: format!("value carried by {}", edge.target),
                    date: String::new(),
                },
            );
        }
        let input = HeavyInputs {
            graph: &graph,
            injected: &edges,
            exchange: "",
            session_id: None,
            active_ids: &[],
            reserved_text: "",
        };
        let selection = select_with_nearest(&input, &[], 100_000);
        assert_eq!(selection.node_count, 45);
        assert!(selection.toon.contains("fact-44"));
    }
