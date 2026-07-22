use std::collections::BTreeMap;
use std::error::Error;

use turso::Connection;

use super::db::{self, RegionTransition};
use super::ledger::{self, EdgeRef, Event, EventContext};

const SEQUENCE_HALF_LIFE_DAYS: f64 = 14.0; // replay-tunable.
const SEQUENCE_WINDOW_DAYS: usize = 60; // replay-tunable.
const SUGGEST_MIN_DECAYED_COUNT: f64 = 1.5; // replay-tunable.
const FOLLOW_BONUS: f64 = 1.0; // replay-tunable.
const REJECT_PENALTY: f64 = 0.5; // replay-tunable.

struct Continuation {
    target: String,
    count: f64,
}

pub fn sequence_line(session_id: &str, turn_id: &str) -> Option<String> {
    let conn = db::open_connection().ok()?;
    match db::block_on(sequence_line_async(&conn, session_id, turn_id)) {
        Ok(line) => line,
        Err(err) => {
            crate::runlog::log("chat", &format!("sequence lookup failed: {err}"));
            None
        }
    }
}

async fn sequence_line_async(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let Some(source) = db::load_previous_turn_region(conn, session_id, turn_id).await? else {
        return Ok(None);
    };
    let signals = db::load_region_transitions(conn, &source, SEQUENCE_WINDOW_DAYS).await?;
    let Some(next) = best_continuation(&signals, db::timestamp_now()) else {
        return Ok(None);
    };
    if next.target == source {
        return Ok(None);
    }
    let payload = format!("{:.6}", next.count);
    let event = Event::sequence(
        "suggested",
        EdgeRef::new(&source, &next.target),
        EventContext::new(session_id, turn_id),
        payload,
    );
    ledger::record_events_in_tx(conn, &[event]).await?;
    Ok(Some(render_sequence_line(&source, &next)))
}

pub fn record_feedback(turn_id: &str) {
    let Some(session_id) = db::source_session_id(turn_id) else {
        return;
    };
    let Ok(conn) = db::open_connection() else {
        return;
    };
    if let Err(err) = db::block_on(record_feedback_async(&conn, &session_id, turn_id)) {
        crate::runlog::log("ledger", &format!("sequence feedback failed: {err}"));
    }
}

async fn record_feedback_async(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let suggestion = db::last_suggestion_for_session(conn, session_id, turn_id).await?;
    let Some(suggestion) = suggestion else {
        return Ok(());
    };
    let current_region = db::load_turn_region(conn, turn_id).await?;
    let event_type = match current_region {
        Some(region) if region == suggestion.target => "followed",
        _ => "rejected",
    };
    let payload = serde_json::json!({"suggested_turn": suggestion.turn_id}).to_string();
    let event = Event::sequence(
        event_type,
        EdgeRef::new(&suggestion.source, &suggestion.target),
        EventContext::new(session_id, turn_id),
        payload,
    );
    ledger::record_events_in_tx(conn, &[event]).await
}

fn best_continuation(signals: &[RegionTransition], now: String) -> Option<Continuation> {
    let now_epoch = crate::timeutil::timestamp_to_epoch_seconds(&now)?;
    let mut counts = BTreeMap::<String, f64>::new();
    for signal in signals {
        let Some(weight) = decayed_signal_weight(signal, now_epoch) else {
            continue;
        };
        let count = counts.entry(signal.target.clone()).or_default();
        *count += weight;
    }
    select_best_count(counts)
}

fn decayed_signal_weight(signal: &RegionTransition, now_epoch: i64) -> Option<f64> {
    let signal_epoch = crate::timeutil::timestamp_to_epoch_seconds(&signal.timestamp)?;
    let age_days = ((now_epoch - signal_epoch).max(0) as f64) / 86_400.0;
    if age_days > SEQUENCE_WINDOW_DAYS as f64 {
        return None;
    }
    let decay = 0.5_f64.powf(age_days / SEQUENCE_HALF_LIFE_DAYS);
    Some(signal_weight(&signal.signal) * decay)
}

fn signal_weight(signal: &str) -> f64 {
    match signal {
        "followed" => FOLLOW_BONUS,
        "rejected" => -REJECT_PENALTY,
        _ => 1.0,
    }
}

fn select_best_count(counts: BTreeMap<String, f64>) -> Option<Continuation> {
    let mut best: Option<Continuation> = None;
    for (target, count) in counts {
        if count < SUGGEST_MIN_DECAYED_COUNT {
            continue;
        }
        let should_replace = match &best {
            Some(current) => count > current.count,
            None => true,
        };
        if should_replace {
            best = Some(Continuation { target, count });
        }
    }
    best
}

fn render_sequence_line(source: &str, next: &Continuation) -> String {
    format!(
        "Sequence memory: after {source}, this user has previously moved on to {} (unprompted continuations: {:.1}).",
        next.target, next.count
    )
}

#[cfg(test)]
#[path = "sequence_tests.rs"]
mod tests;
