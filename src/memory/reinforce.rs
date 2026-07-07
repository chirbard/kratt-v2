use std::error::Error;

use turso::{Connection, params};

use super::db;
use super::ledger::{EdgeRef, Event, EventContext};

const REINFORCE_RATE: f64 = 0.05; // replay-tunable.
const TEMPORAL_FLOOR: f64 = 0.3; // replay-tunable.
const CONFIDENCE_MIN: f64 = 0.05; // replay-tunable.
const CONFIDENCE_MAX: f64 = 0.95; // replay-tunable.

struct SliceEvent {
    source: String,
    target: String,
    activity_weight: f64,
}

struct EdgeDelta {
    delta: f64,
    activity_weight: f64,
    temporal_weight: f64,
    raw_weighted: f64,
    norm_factor: f64,
}

pub fn record_outcome(
    session_id: &str,
    outcome_score: f64,
    payload: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    apply_outcome(session_id, "", outcome_score, payload)
}

// Per-move credit assignment for multi-move games (see credit-assignment.md).
// No caller yet — wired up by the future game harness (Demo 1).
#[allow(dead_code)]
pub fn record_move_outcome(
    session_id: &str,
    turn_id: &str,
    outcome_score: f64,
    payload: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    apply_outcome(session_id, turn_id, outcome_score, payload)
}

fn apply_outcome(
    session_id: &str,
    turn_id: &str,
    outcome_score: f64,
    payload: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let clamped_score = outcome_score.clamp(-1.0, 1.0);
    let _write_guard = db::acquire_write_lock();
    let conn = db::open_connection()?;
    db::begin_transaction(&conn)?;
    match run_reinforcement(&conn, session_id, turn_id, clamped_score, payload) {
        Ok(summary) => {
            db::commit_transaction(&conn)?;
            db::sync_push_best_effort("reinforce");
            crate::runlog::log("reinforce", &summary);
            Ok(summary)
        }
        Err(err) => {
            let _ = db::block_on(async { conn.execute("ROLLBACK", ()).await });
            Err(err)
        }
    }
}

fn run_reinforcement(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
    outcome_score: f64,
    payload: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if already_reinforced(conn, session_id, turn_id)? {
        return Ok(format!("session {session_id}: already reinforced"));
    }
    let context = EventContext::new(session_id, turn_id);
    let mut events = vec![Event::outcome(
        session_id,
        turn_id,
        outcome_score,
        payload.to_string(),
    )];
    let slice = load_session_slice(conn, session_id, turn_id)?;
    let deltas = compute_edge_deltas(&slice, outcome_score);
    let mut applied = 0usize;
    let mut delta_sum = 0.0;
    for (edge, edge_delta) in deltas {
        let Some(event) = apply_edge_delta(conn, &edge, &context, &edge_delta, outcome_score)?
        else {
            continue;
        };
        delta_sum += edge_delta.delta;
        applied += 1;
        events.push(event);
    }
    db::block_on(super::ledger::record_events_in_tx(conn, &events))?;
    Ok(format!(
        "session {session_id}: outcome {outcome_score:.2}, {applied} edges, Σdelta {delta_sum:.4}"
    ))
}

fn apply_edge_delta(
    conn: &Connection,
    edge: &EdgeRef,
    context: &EventContext,
    edge_delta: &EdgeDelta,
    outcome_score: f64,
) -> Result<Option<Event>, Box<dyn Error + Send + Sync>> {
    let Some(current) = db::load_edge(conn, &edge.source, &edge.target)? else {
        return Ok(None);
    };
    if current.expired_at.is_some() {
        return Ok(None);
    }
    let confidence_before = current.confidence;
    let confidence_after = clamp_confidence(confidence_before, edge_delta.delta);
    if !db::update_edge_confidence(conn, &edge.source, &edge.target, confidence_after)? {
        return Ok(None);
    }
    let payload = serde_json::json!({
        "session_id": context.session_id,
        "delta": edge_delta.delta,
        "activity_weight": edge_delta.activity_weight,
        "temporal_weight": edge_delta.temporal_weight,
        "raw_weighted": edge_delta.raw_weighted,
        "norm_factor": edge_delta.norm_factor,
        "outcome_score": outcome_score,
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
    })
    .to_string();
    let event = Event::reinforced(
        EdgeRef::new(&edge.source, &edge.target),
        &context.session_id,
        &context.turn_id,
        edge_delta.delta,
        payload,
    );
    Ok(Some(event))
}

fn clamp_confidence(current: f64, delta: f64) -> f64 {
    let after = (current + delta).clamp(CONFIDENCE_MIN, CONFIDENCE_MAX);
    // Reinforcement must never move confidence against its own sign: an edge
    // already above the ceiling must not be pulled down by a positive outcome
    // (nor raised by a negative one).
    if delta > 0.0 && after < current {
        return current;
    }
    if delta < 0.0 && after > current {
        return current;
    }
    after
}

fn temporal_weight(pos: usize, slice_len: usize) -> f64 {
    if slice_len <= 1 {
        return 1.0;
    }
    TEMPORAL_FLOOR + (1.0 - TEMPORAL_FLOOR) * (pos as f64 / (slice_len - 1) as f64)
}

fn compute_edge_deltas(slice: &[SliceEvent], outcome_score: f64) -> Vec<(EdgeRef, EdgeDelta)> {
    let slice_len = slice.len();
    let mut sums: Vec<(EdgeRef, f64, f64)> = Vec::new();
    let mut pos = 0usize;
    while pos < slice.len() {
        let event = &slice[pos];
        let weight = temporal_weight(pos, slice_len);
        let contribution = event.activity_weight * weight;
        accumulate_edge(&mut sums, event, contribution);
        pos += 1;
    }
    build_deltas(sums, outcome_score)
}

fn accumulate_edge(sums: &mut Vec<(EdgeRef, f64, f64)>, event: &SliceEvent, contribution: f64) {
    let mut index = 0usize;
    while index < sums.len() {
        let (edge, activity_weight, weighted) = &mut sums[index];
        if edge.source == event.source && edge.target == event.target {
            *activity_weight += event.activity_weight;
            *weighted += contribution;
            return;
        }
        index += 1;
    }
    sums.push((
        EdgeRef::new(&event.source, &event.target),
        event.activity_weight,
        contribution,
    ));
}

// Per-outcome normalization: scale every edge's raw weighted activity by the
// session's maximum so the most-central edge moves by exactly REINFORCE_RATE
// and the rest proportionally less. Without this a long slice (an edge injected
// or retrieved dozens of times) sums to a huge weight and one outcome saturates
// confidence — normalization makes the nudge slice-size independent.
fn build_deltas(sums: Vec<(EdgeRef, f64, f64)>, outcome_score: f64) -> Vec<(EdgeRef, EdgeDelta)> {
    let mut max_weighted = 0.0;
    for (_, _, weighted) in &sums {
        if *weighted > max_weighted {
            max_weighted = *weighted;
        }
    }
    let mut deltas = Vec::with_capacity(sums.len());
    for (edge, activity_weight, weighted) in sums {
        let norm = if max_weighted > 0.0 {
            weighted / max_weighted
        } else {
            0.0
        };
        let temporal = if activity_weight > 0.0 {
            weighted / activity_weight
        } else {
            0.0
        };
        deltas.push((
            edge,
            EdgeDelta {
                delta: outcome_score * REINFORCE_RATE * norm,
                activity_weight,
                temporal_weight: temporal,
                raw_weighted: weighted,
                norm_factor: norm,
            },
        ));
    }
    deltas
}

fn already_reinforced(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    db::block_on(already_reinforced_async(conn, session_id, turn_id))
}

async fn already_reinforced_async(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT 1 FROM events WHERE event_type = 'reinforced' AND session_id = ?1 AND turn_id = ?2 LIMIT 1",
            params![session_id, turn_id],
        )
        .await?;
    Ok(rows.next().await?.is_some())
}

fn load_session_slice(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<Vec<SliceEvent>, Box<dyn Error + Send + Sync>> {
    db::block_on(load_session_slice_async(conn, session_id, turn_id))
}

async fn load_session_slice_async(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<Vec<SliceEvent>, Box<dyn Error + Send + Sync>> {
    let mut rows = if turn_id.is_empty() {
        conn.query(
            "SELECT source, target, weight FROM events WHERE session_id = ?1 AND event_type IN ('retrieved','marked') ORDER BY id",
            params![session_id],
        )
        .await?
    } else {
        conn.query(
            "SELECT source, target, weight FROM events WHERE session_id = ?1 AND turn_id = ?2 AND event_type IN ('retrieved','marked') ORDER BY id",
            params![session_id, turn_id],
        )
        .await?
    };
    let mut slice = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = db::text(row.get_value(0)?);
        let target = db::text(row.get_value(1)?);
        if source.is_empty() || target.is_empty() {
            continue;
        }
        let activity_weight = db::real(row.get_value(2)?);
        slice.push(SliceEvent {
            source,
            target,
            activity_weight,
        });
    }
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::{
        CONFIDENCE_MAX, CONFIDENCE_MIN, REINFORCE_RATE, SliceEvent, TEMPORAL_FLOOR,
        clamp_confidence, compute_edge_deltas, temporal_weight,
    };

    fn edge_event(activity_weight: f64) -> SliceEvent {
        named_edge_event("a", "b", activity_weight)
    }

    fn named_edge_event(source: &str, target: &str, activity_weight: f64) -> SliceEvent {
        SliceEvent {
            source: source.to_string(),
            target: target.to_string(),
            activity_weight,
        }
    }

    #[test]
    fn temporal_weight_single_event_is_full_weight() {
        assert!((temporal_weight(0, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn temporal_weight_first_event_is_floor() {
        assert!((temporal_weight(0, 5) - TEMPORAL_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn temporal_weight_last_event_is_full_weight() {
        assert!((temporal_weight(4, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_edge_sums_activity_weight() {
        let slice = vec![edge_event(1.0), edge_event(2.0)];
        let deltas = compute_edge_deltas(&slice, 1.0);
        assert_eq!(deltas.len(), 1);
        assert!((deltas[0].1.activity_weight - 3.0).abs() < 1e-9);
    }

    #[test]
    fn delta_sign_follows_outcome_sign() {
        let slice = vec![edge_event(1.0)];
        let positive = compute_edge_deltas(&slice, 1.0);
        let negative = compute_edge_deltas(&slice, -1.0);
        assert!(positive[0].1.delta > 0.0);
        assert!(negative[0].1.delta < 0.0);
    }

    #[test]
    fn clamp_confidence_bounds_both_ends() {
        assert!((clamp_confidence(0.9, 10.0) - CONFIDENCE_MAX).abs() < 1e-9);
        assert!((clamp_confidence(0.1, -10.0) - CONFIDENCE_MIN).abs() < 1e-9);
    }

    #[test]
    fn positive_delta_never_lowers_over_ceiling_edge() {
        // An edge already above the ceiling must stay put under a positive nudge.
        assert!((clamp_confidence(0.98, 0.05) - 0.98).abs() < 1e-9);
        assert!((clamp_confidence(0.02, -0.05) - 0.02).abs() < 1e-9);
    }

    #[test]
    fn normalization_caps_max_delta_at_rate() {
        // Regardless of slice size, the most-active edge moves by REINFORCE_RATE.
        let slice = vec![
            named_edge_event("a", "b", 2.0),
            named_edge_event("a", "b", 2.0),
            named_edge_event("c", "d", 3.0),
        ];
        let deltas = compute_edge_deltas(&slice, 1.0);
        let mut max_delta = 0.0f64;
        for (_, edge_delta) in &deltas {
            if edge_delta.delta > max_delta {
                max_delta = edge_delta.delta;
            }
        }
        assert!((max_delta - REINFORCE_RATE).abs() < 1e-9);
    }

    #[test]
    fn normalization_scales_less_active_edge_proportionally() {
        let slice = vec![
            named_edge_event("a", "b", 1.0),
            named_edge_event("c", "d", 2.0),
        ];
        // Single-event positions both weight 1.0 temporally (n=2: pos0=floor, pos1=1.0),
        // so the ratio is driven by activity weight — the weaker edge stays under RATE.
        let deltas = compute_edge_deltas(&slice, 1.0);
        for (edge, edge_delta) in &deltas {
            assert!(edge_delta.delta <= REINFORCE_RATE + 1e-9);
            if edge.source == "a" {
                assert!(edge_delta.delta < REINFORCE_RATE);
            }
        }
    }
}
