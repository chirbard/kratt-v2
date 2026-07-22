    use std::collections::HashMap;

    use super::{Cluster, ClusterEdge, cluster_ceiling, cluster_toon};
    use crate::memory::graph::Node;

    fn cluster_edge(confidence: f64) -> ClusterEdge {
        ClusterEdge {
            source: "source".to_string(),
            target: "target".to_string(),
            content: "belief".to_string(),
            confidence,
            stability: 0.5,
            importance: 0.5,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
        }
    }

    #[test]
    fn cluster_ceiling_uses_mean_confidence_not_cluster_size() {
        let edges = vec![cluster_edge(0.25); 10];
        let cluster = Cluster {
            hub: "hub".to_string(),
            edge_indices: (0..10).collect(),
            node_ids: Vec::new(),
            view_mass: 0.0,
        };
        assert_eq!(cluster_ceiling(&cluster, &edges), 0.25);
    }

    #[test]
    fn cluster_toon_includes_value_bearing_endpoint_nodes() {
        let edges = vec![cluster_edge(0.5)];
        let cluster = Cluster {
            hub: "source".to_string(),
            edge_indices: vec![0],
            node_ids: vec!["source".to_string(), "target".to_string()],
            view_mass: 0.0,
        };
        let nodes = HashMap::from([
            (
                "source".to_string(),
                Node {
                    node_type: "synthesis".to_string(),
                    body: "An abstraction".to_string(),
                    date: String::new(),
                },
            ),
            (
                "target".to_string(),
                Node {
                    node_type: "entity".to_string(),
                    body: String::new(),
                    date: String::new(),
                },
            ),
        ]);
        let first = cluster_toon(&cluster, &edges, &nodes);
        let second = cluster_toon(&cluster, &edges, &nodes);
        assert!(first.contains("An abstraction"));
        assert_eq!(first, second);
    }
