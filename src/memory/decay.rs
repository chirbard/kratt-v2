use super::db;
use super::graph::{Edge, Graph};
use crate::prompts::today_iso;

pub fn apply_decay_if_needed(graph: &mut Graph) {
    let today = today_iso();
    if db::load_last_decay() == today {
        return;
    }
    for edge in &mut graph.edges {
        decay_edge(edge, &today);
    }
    db::save_last_decay(&today);
}

fn decay_edge(edge: &mut Edge, today: &str) {
    if edge.stability >= 1.0 || edge.date.is_empty() {
        return;
    }
    let days = days_between(&edge.date, today);
    if days <= 0 {
        return;
    }
    let factor = 0.95_f64.powf(days as f64 / 7.0);
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
