use super::db;
use super::decay_curve;
use super::graph::{Edge, Graph, edge_is_pinned};
use crate::prompts::today_iso;

pub fn apply_decay_if_needed(graph: &mut Graph) -> bool {
    let today = today_iso();
    if db::load_last_decay() == today {
        return false;
    }
    let (changes, initialized) = decay_graph(graph, &today);
    // The checkpoint lives inside the same transaction as the values, so a
    // failed persist retries on the next pass instead of being skipped until
    // tomorrow — the previous order wrote the checkpoint first and swallowed
    // the error, which silently disabled decay entirely.
    if let Err(err) = db::persist_decay(&changes, &today) {
        crate::runlog::log("decay", &format!("persist failed, retrying later: {err}"));
        return false;
    }
    if initialized > 0 {
        crate::runlog::log("decay", &format!("initialized {initialized} clocks"));
    }
    true
}

fn decay_graph(graph: &mut Graph, today: &str) -> (Vec<db::DecayChange>, usize) {
    let nodes = &graph.nodes;
    let mut changes = Vec::new();
    let mut initialized = 0usize;
    for edge in &mut graph.edges {
        let source_type = match nodes.get(&edge.source) {
            Some(node) => node.node_type.as_str(),
            None => "",
        };
        if edge_is_pinned(edge, source_type) {
            continue;
        }
        let before = (edge.confidence, edge.last_decayed_at.clone());
        initialized += usize::from(decay_edge(edge, today));
        if before.0 != edge.confidence || before.1 != edge.last_decayed_at {
            changes.push(db::DecayChange {
                source: edge.source.clone(),
                target: edge.target.clone(),
                confidence: edge.confidence,
                last_decayed_at: edge.last_decayed_at.clone(),
            });
        }
    }
    (changes, initialized)
}

fn decay_edge(edge: &mut Edge, today: &str) -> bool {
    // Charge elapsed time since the last charge, not the edge's whole age.
    // Without this the same history was re-charged every night and the
    // exponent compounded as n(n+1)/14 instead of n/7 — an untouched 30-day
    // edge reached the confidence floor in about ten nights. Edges that have
    // never been charged under this scheme start their clock today: current
    // damage is frozen where it is, not repaired here (repair is part of the
    // reviewed decay redesign, kratt-roadmap/epistemics/frozen-graph.md).
    if date_to_days(&edge.last_decayed_at).is_none() {
        edge.last_decayed_at = today.to_string();
        return true;
    }
    let days = days_between(&edge.last_decayed_at, today);
    if days <= 0 {
        return false;
    }
    edge.last_decayed_at = today.to_string();
    let factor = decay_curve::decay_factor(&edge.memory_type, edge.stability, days as f64);
    edge.confidence = (edge.confidence * factor).clamp(0.05, 1.0);
    false
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
#[path = "decay_tests.rs"]
mod tests;
