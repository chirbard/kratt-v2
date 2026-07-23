use std::collections::HashMap;
use std::path::PathBuf;

use turso::Connection;

use super::db;
use super::ledger;

// replay-tunable.
const RE_READ_NOTE_THRESHOLD: usize = 3;
// replay-tunable.
const DRIFT_WARN_DISTANCE: f64 = 0.10;

/// Action that the scheduler decides to take
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerAction {
    NewsScan { reason: String },
    InboxStudy { reason: String },
    Ingest { path: PathBuf, reason: String },
    Rest { reason: String },
}

/// Input data for the decision function (pure, testable)
#[derive(Debug, Clone, Default)]
pub struct SchedulerInputs {
    pub queued_inbox: usize,
    pub re_read_candidates: Vec<(String, usize, String)>, // (path, note_count, source_node_id)
    pub actor_positions: Vec<(String, String, Option<Vec<f64>>)>, // (actor_type, actor_id, position)
    pub news_scan_due: bool,
}

/// Gather all inputs from the database for scheduler decision
pub fn gather_inputs() -> SchedulerInputs {
    let Ok(conn) = db::open_connection() else {
        return SchedulerInputs::default();
    };

    db::block_on(gather_inputs_async(&conn))
}

async fn gather_inputs_async(conn: &Connection) -> SchedulerInputs {
    let queued_inbox = db::count_queued_inbox(conn).await;
    let re_read_candidates = find_re_read_candidates(conn).await;
    let actor_positions = load_all_actor_positions(conn).await;

    SchedulerInputs {
        queued_inbox,
        re_read_candidates,
        actor_positions,
        news_scan_due: true,
    }
}

// The chat-visible slice of interoception: real backlog counters, surfaced
// only when queued study exists. None = no line, zero cost in the prompt.
pub fn pressure_line() -> Option<String> {
    let Ok(conn) = db::open_connection() else {
        return None;
    };
    let queued = db::block_on(db::load_queued_inbox_names(&conn));
    render_pressure(&queued)
}

// Names, not counts: a bare count made the model guess wrongly at WHICH items
// were unread (observed live: it refused to synthesize an already-studied
// document, assuming it was among the queued ones).
fn render_pressure(queued: &[String]) -> Option<String> {
    if queued.is_empty() {
        return None;
    }
    Some(format!(
        "Internal state: queued study items not yet read: {}.",
        queued.join(", ")
    ))
}

/// Pure decision function - first matching rule wins (priority order)
pub fn decide(inputs: &SchedulerInputs) -> SchedulerAction {
    if inputs.queued_inbox > 0 {
        return SchedulerAction::InboxStudy {
            reason: format!("{} inbox items queued", inputs.queued_inbox),
        };
    }
    if let Some(action) = check_re_read_candidates(inputs) {
        return action;
    }
    if inputs.news_scan_due {
        return SchedulerAction::NewsScan {
            reason: "daily interest-driven scan".to_string(),
        };
    }
    rest_action(inputs)
}

fn check_re_read_candidates(inputs: &SchedulerInputs) -> Option<SchedulerAction> {
    for (path, note_count, _) in &inputs.re_read_candidates {
        if *note_count >= RE_READ_NOTE_THRESHOLD && std::path::Path::new(path).exists() {
            return Some(SchedulerAction::Ingest {
                path: PathBuf::from(path.clone()),
                reason: format!(
                    "document '{}' has {} open curiosity notes >= {}",
                    path, note_count, RE_READ_NOTE_THRESHOLD
                ),
            });
        }
    }
    None
}

fn rest_action(_inputs: &SchedulerInputs) -> SchedulerAction {
    SchedulerAction::Rest {
        reason: "nothing to do".to_string(),
    }
}

/// Compute and log drift between all actor positions
pub fn log_drift_info(actor_positions: &[(String, String, Option<Vec<f64>>)]) {
    if actor_positions.len() < 2 {
        return;
    }

    let mut min_distance: Option<(f64, String, String)> = None;
    let position_map = build_position_map(actor_positions);
    compute_min_distance(&position_map, &mut min_distance);

    if let Some((distance, actor_a, actor_b)) = min_distance {
        crate::runlog::log(
            "sched",
            &format!(
                "drift: min pair {}-{} distance {:.4}",
                actor_a, actor_b, distance
            ),
        );
        log_drift_warning(distance, actor_a.as_str(), actor_b.as_str());
    }
}

fn build_position_map(
    actor_positions: &[(String, String, Option<Vec<f64>>)],
) -> HashMap<(String, String), &Vec<f64>> {
    let mut position_map = HashMap::new();
    for (actor_type, actor_id, position) in actor_positions {
        if let Some(pos) = position {
            position_map.insert((actor_type.clone(), actor_id.clone()), pos);
        }
    }
    position_map
}

fn compute_min_distance(
    position_map: &HashMap<(String, String), &Vec<f64>>,
    min_distance: &mut Option<(f64, String, String)>,
) {
    let positions: Vec<_> = position_map.iter().collect();
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let ((at1, ai1), pos1) = positions[i];
            let ((at2, ai2), pos2) = positions[j];

            if pos1.len() != pos2.len() {
                continue;
            }

            let distance = ledger::cosine_distance(pos1, pos2).unwrap_or(1.0);
            let actor_pair = (format!("{}:{}", at1, ai1), format!("{}:{}", at2, ai2));
            update_min_distance_if_needed(min_distance, distance, actor_pair);
        }
    }
}

fn update_min_distance_if_needed(
    min_distance: &mut Option<(f64, String, String)>,
    distance: f64,
    actors: (String, String),
) {
    match min_distance.as_ref() {
        Some((current_min, _, _)) => {
            if distance < *current_min {
                *min_distance = Some((distance, actors.0, actors.1));
            }
        }
        None => {
            *min_distance = Some((distance, actors.0, actors.1));
        }
    }
}

fn log_drift_warning(distance: f64, actor_a: &str, actor_b: &str) {
    if distance < DRIFT_WARN_DISTANCE {
        crate::runlog::log(
            "sched",
            &format!(
                "WARNING: agents {} and {} collapsing onto same region (distance {:.4} < {})",
                actor_a, actor_b, distance, DRIFT_WARN_DISTANCE
            ),
        );
    }
}

/// Log the scheduler action decision
pub fn log_scheduler_decision(action: &SchedulerAction) {
    match action {
        SchedulerAction::NewsScan { reason } => {
            crate::runlog::log("sched", &format!("[autonomy] news scan — {reason}"));
        }
        SchedulerAction::InboxStudy { reason } => {
            crate::runlog::log("sched", &format!("[autonomy] inbox study — {reason}"));
        }
        SchedulerAction::Ingest { path, reason } => {
            crate::runlog::log(
                "sched",
                &format!("[autonomy] ingest — {reason} ({:?})", path.display()),
            );
        }
        SchedulerAction::Rest { reason } => {
            crate::runlog::log("sched", &format!("[autonomy] rest — {reason}"));
        }
    }
}

// ==================== Input Gathering Helpers ====================

/// Find documents that have >= RE_READ_NOTE_THRESHOLD open curiosity notes.
/// The document path lives in the note's `created` event payload
/// (`{"source": path, ...}`) — note bodies are plain question text.
/// Returns (path, note_count, source_node_id).
async fn find_re_read_candidates(conn: &Connection) -> Vec<(String, usize, String)> {
    let Some(note_sources) = load_open_note_sources(conn).await else {
        return Vec::new();
    };
    let mut per_path: HashMap<String, Vec<String>> = HashMap::new();
    for (note_id, path) in note_sources {
        let notes = per_path.entry(path).or_default();
        if !notes.contains(&note_id) {
            notes.push(note_id);
        }
    }
    let mut candidates = Vec::new();
    for (path, notes) in per_path {
        if notes.len() >= RE_READ_NOTE_THRESHOLD {
            candidates.push((path, notes.len(), notes[0].clone()));
        }
    }
    candidates
}

/// (note_id, source_path) pairs for open notes, parsed from event payloads.
async fn load_open_note_sources(conn: &Connection) -> Option<Vec<(String, String)>> {
    let mut rows = conn
        .query(
            "SELECT n.id, ev.payload
             FROM nodes n
             JOIN edges e ON e.source = n.id AND e.expired_at IS NULL
             JOIN events ev ON ev.source = e.source AND ev.target = e.target
                 AND ev.event_type = 'created'
             LEFT JOIN (
                 SELECT DISTINCT c.source, c.target
                 FROM events c
                 JOIN sessions s ON s.id = c.session_id
                 WHERE s.session_type = 'exploration' AND c.event_type = 'retrieved'
             ) consumed ON consumed.source = e.source AND consumed.target = e.target
             WHERE n.node_type = 'note' AND consumed.source IS NULL",
            (),
        )
        .await
        .ok()?;
    let mut pairs = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Ok(note_value) = row.get_value(0) else {
            return None;
        };
        let Ok(payload_value) = row.get_value(1) else {
            return None;
        };
        let note_id = db::text(note_value);
        let payload = db::text(payload_value);
        if let Some(path) = payload_source_path(&payload) {
            pairs.push((note_id, path));
        }
    }
    Some(pairs)
}

fn payload_source_path(payload: &str) -> Option<String> {
    let json = serde_json::from_str::<serde_json::Value>(payload).ok()?;
    Some(json.get("source")?.as_str()?.to_string())
}


/// Load all actor positions for drift monitoring
async fn load_all_actor_positions(conn: &Connection) -> Vec<(String, String, Option<Vec<f64>>)> {
    let mut positions = Vec::new();

    let mut rows = match conn
        .query(
            "SELECT actor_type, actor_id, CASE WHEN sum_vector IS NULL THEN NULL ELSE vector_extract(sum_vector) END 
             FROM actor_positions",
            (),
        )
        .await {
        Ok(rows) => rows,
        Err(_) => return positions,
    };

    while let Ok(Some(row)) = rows.next().await {
        let actor_type = match row.get_value(0) {
            Ok(value) => db::text(value),
            Err(_) => continue,
        };
        let actor_id = match row.get_value(1) {
            Ok(value) => db::text(value),
            Err(_) => continue,
        };
        let position = match row.get_value(2) {
            Ok(value) => super::ledger::parse_vector_blob(value),
            Err(_) => None,
        };
        positions.push((actor_type, actor_id, position));
    }

    positions
}

// ==================== Tests ====================

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
