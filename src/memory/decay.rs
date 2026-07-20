use super::db;
use super::graph::{Edge, Graph};
use crate::prompts::today_iso;

pub fn apply_decay_if_needed(graph: &mut Graph) -> bool {
    let today = today_iso();
    if db::load_last_decay() == today {
        return false;
    }
    let changes = decay_graph(graph, &today);
    // The checkpoint lives inside the same transaction as the values, so a
    // failed persist retries on the next pass instead of being skipped until
    // tomorrow — the previous order wrote the checkpoint first and swallowed
    // the error, which silently disabled decay entirely.
    if let Err(err) = db::persist_decay(&changes, &today) {
        crate::runlog::log("decay", &format!("persist failed, retrying later: {err}"));
        return false;
    }
    true
}

fn decay_graph(graph: &mut Graph, today: &str) -> Vec<db::DecayChange> {
    let nodes = &graph.nodes;
    let mut changes = Vec::new();
    for edge in &mut graph.edges {
        if edge.expired_at.is_none() && edge_touches_standing_node(nodes, edge) {
            continue;
        }
        let before = (edge.confidence, edge.last_decayed_at.clone());
        decay_edge(edge, today);
        if before.0 != edge.confidence || before.1 != edge.last_decayed_at {
            changes.push(db::DecayChange {
                source: edge.source.clone(),
                target: edge.target.clone(),
                confidence: edge.confidence,
                last_decayed_at: edge.last_decayed_at.clone(),
            });
        }
    }
    changes
}

fn edge_touches_standing_node(
    nodes: &std::collections::HashMap<String, super::graph::Node>,
    edge: &Edge,
) -> bool {
    for node_id in [&edge.source, &edge.target] {
        if matches!(nodes.get(node_id), Some(node) if matches!(node.node_type.as_str(), "loop" | "interest"))
        {
            return true;
        }
    }
    false
}

// Episodic edges are memory of lived experience, not durable belief — they
// must be forgettable within days. This is the long-term forgetting
// mechanism; it does NOT prevent cross-session contamination by itself (a
// same-day second game would still see stale episodic edges from the first
// one if decay were the only guard). That guarantee comes from retrieval
// scoping in embed.rs, which excludes episodic edges outside their
// originating session regardless of how much they have decayed.
const EPISODIC_DECAY_BASE: f64 = 0.5; // replay-tunable.

fn decay_edge(edge: &mut Edge, today: &str) {
    if edge.stability >= 1.0 || edge.date.is_empty() {
        return;
    }
    // Charge elapsed time since the last charge, not the edge's whole age.
    // Without this the same history was re-charged every night and the
    // exponent compounded as n(n+1)/14 instead of n/7 — an untouched 30-day
    // edge reached the confidence floor in about ten nights. Edges that have
    // never been charged under this scheme start their clock today: current
    // damage is frozen where it is, not repaired here (repair is part of the
    // reviewed decay redesign, kratt-roadmap/epistemics/frozen-graph.md).
    if edge.last_decayed_at.is_empty() {
        edge.last_decayed_at = today.to_string();
        return;
    }
    let days = days_between(&edge.last_decayed_at, today);
    if days <= 0 {
        return;
    }
    edge.last_decayed_at = today.to_string();
    let factor = if edge.memory_type == "episodic" {
        EPISODIC_DECAY_BASE.powf(days as f64)
    } else {
        0.95_f64.powf(days as f64 / 7.0)
    };
    edge.confidence = (edge.confidence * factor).clamp(0.05, 1.0);
}

fn days_between(from: &str, until: &str) -> i64 {
    let Some(from_days) = date_to_days(from) else {
        return 0;
    };
    let Some(until_days) = date_to_days(until) else {
        return 0;
    };
    until_days - from_days
}

fn date_to_days(date: &str) -> Option<i64> {
    let mut parts = date.splitn(3, '-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    let (adj_year, adj_month) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = adj_year / 400;
    let year_of_era = adj_year - era * 400;
    let day_of_year = (153 * adj_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

#[cfg(test)]
mod tests {

    // Regression: decay used to charge each edge's full age on every run,
    // compounding as n(n+1)/14. One night must cost exactly one night.
    #[test]
    fn repeated_runs_charge_one_night_each_not_the_whole_history() {
        let mut edge = test_edge("semantic", "2026-06-01");
        edge.confidence = 0.9;
        edge.last_decayed_at = String::new();
        // First run only starts the clock: existing damage is frozen, not repaired.
        decay_edge(&mut edge, "2026-07-01");
        assert_eq!(edge.confidence, 0.9, "clock start does not decay");
        assert_eq!(edge.last_decayed_at, "2026-07-01");

        decay_edge(&mut edge, "2026-07-02");
        let after_one_night = edge.confidence;
        let expected = 0.9 * 0.95_f64.powf(1.0 / 7.0);
        assert!(
            (after_one_night - expected).abs() < 1e-9,
            "one night charges one night: {after_one_night} vs {expected}"
        );

        decay_edge(&mut edge, "2026-07-03");
        let expected_two = 0.9 * 0.95_f64.powf(2.0 / 7.0);
        assert!(
            (edge.confidence - expected_two).abs() < 1e-9,
            "two nights charge two nights, not 1+2"
        );
        // The compounded-history value would be far lower.
        assert!(edge.confidence > 0.9 * 0.95_f64.powf(31.0 / 7.0));
    }

    #[test]
    fn same_day_second_run_charges_nothing() {
        let mut edge = test_edge("semantic", "2026-06-01");
        edge.confidence = 0.9;
        edge.last_decayed_at = "2026-07-02".to_string();
        decay_edge(&mut edge, "2026-07-02");
        assert_eq!(edge.confidence, 0.9);
    }

    use super::super::graph::{Edge, Graph, Node};
    use super::{decay_edge, decay_graph};

    fn test_edge(memory_type: &str, date: &str) -> Edge {
        Edge {
            source: "a".to_string(),
            target: "b".to_string(),
            content: String::new(),
            stability: 0.3,
            confidence: 1.0,
            importance: 0.5,
            date: date.to_string(),
            subject: false,
            occurred_at: None,
            expired_at: None,
            memory_type: memory_type.to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            // Already on the incremental scheme, charged up to its own date —
            // the common case once the clock has been running.
            last_decayed_at: date.to_string(),
        }
    }

    #[test]
    fn episodic_edges_decay_far_faster_than_semantic_over_same_span() {
        let mut episodic = test_edge("episodic", "2026-07-01");
        let mut semantic = test_edge("semantic", "2026-07-01");
        decay_edge(&mut episodic, "2026-07-04");
        decay_edge(&mut semantic, "2026-07-04");
        assert!(episodic.confidence < semantic.confidence);
        assert!(episodic.confidence <= 0.15);
        assert!(semantic.confidence > 0.9);
    }

    #[test]
    fn episodic_decay_never_drops_below_floor() {
        let mut episodic = test_edge("episodic", "2026-01-01");
        decay_edge(&mut episodic, "2026-07-01");
        assert!(episodic.confidence >= 0.05);
    }

    #[test]
    fn live_loop_anchor_keeps_fixed_confidence() {
        let mut graph = Graph::new();
        graph.nodes.insert(
            "loop-reminder".to_string(),
            Node {
                node_type: "loop".to_string(),
                body: "ask later".to_string(),
                date: String::new(),
            },
        );
        let mut loop_edge = test_edge("semantic", "2026-01-01");
        loop_edge.source = "loop-reminder".to_string();
        loop_edge.confidence = 0.9;
        loop_edge.stability = 0.9;
        graph.edges.push(loop_edge.clone());
        graph.edges.push(test_edge("semantic", "2026-01-01"));
        loop_edge.expired_at = Some("2026-01-02".to_string());
        graph.edges.push(loop_edge);
        decay_graph(&mut graph, "2026-07-12");
        assert!((graph.edges[0].confidence - 0.9).abs() < 1e-9);
        assert!(graph.edges[1].confidence < 1.0);
        assert!(graph.edges[2].confidence < 0.9);
    }
}
