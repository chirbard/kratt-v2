use super::db;
use super::graph::{Edge, Graph};
use crate::prompts::today_iso;

pub fn apply_decay_if_needed(graph: &mut Graph) -> bool {
    let today = today_iso();
    if db::load_last_decay() == today {
        return false;
    }
    for edge in &mut graph.edges {
        decay_edge(edge, &today);
    }
    db::save_last_decay(&today);
    true
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
    let days = days_between(&edge.date, today);
    if days <= 0 {
        return;
    }
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
    use super::decay_edge;
    use super::super::graph::Edge;

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
}
