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
mod tests {
    use super::{
        RegionTransition, best_continuation, decayed_signal_weight, record_feedback_async,
        sequence_line_async,
    };
    use crate::memory::db;
    use turso::{Builder, Connection, params};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    const SESSIONS_DDL: &str = "CREATE TABLE sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')";
    const MESSAGES_DDL: &str = "CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '')";
    const VIEWS_DDL: &str = "CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))";
    const POSITIONS_DDL: &str = "CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))";
    const HISTORY_DDL: &str = "CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)";
    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";

    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        for ddl in [
            EDGES_DDL,
            EVENTS_DDL,
            SESSIONS_DDL,
            MESSAGES_DDL,
            VIEWS_DDL,
            POSITIONS_DDL,
            HISTORY_DDL,
            NODES_DDL,
        ] {
            let _ = conn.execute(ddl, ()).await;
        }
        conn
    }

    // A fact node outranking an entity by confidence must still lose the
    // region to the entity (live finding: reverse fact->entity edges made
    // 'ghostty-supports-windows' a region, fragmenting 'ghostty' counts).
    #[test]
    fn region_prefers_entity_sources_over_higher_confidence_facts() {
        db::block_on(async {
            let conn = harness().await;
            add_turn(&conn, ("s1", "t1", "2026-07-13T10:00:00", Some("ghostty-supports-windows"), 0.9)).await;
            let _ = conn
                .execute(
                    "INSERT OR REPLACE INTO edges (source, target, confidence) VALUES ('ghostty', 'ghostty-detail', 0.4)",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO events (turn_id, event_type, source, target) VALUES ('t1', 'injected', 'ghostty', 'ghostty-detail')",
                    (),
                )
                .await;
            for (node_id, node_type) in [("ghostty", "entity"), ("ghostty-supports-windows", "fact")] {
                let _ = conn
                    .execute(
                        "INSERT OR REPLACE INTO nodes (id, node_type) VALUES (?1, ?2)",
                        params![node_id, node_type],
                    )
                    .await;
            }
            let Ok(region) = db::load_turn_region(&conn, "t1").await else {
                unreachable!("region query succeeds")
            };
            assert_eq!(region.as_deref(), Some("ghostty"));
        });
    }

    async fn add_turn(conn: &Connection, details: (&str, &str, &str, Option<&str>, f64)) {
        let (session, turn, timestamp, region, confidence) = details;
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO sessions (id) VALUES (?1)",
                params![session],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO messages (id, session_id, role, created_at) VALUES (?1, ?2, 'user', ?3)",
                params![turn, session, timestamp],
            )
            .await;
        let Some(source) = region else {
            return;
        };
        let target = format!("{source}-detail");
        let _ = conn
            .execute(
                "INSERT OR REPLACE INTO edges (source, target, confidence) VALUES (?1, ?2, ?3)",
                params![source, target.as_str(), confidence],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO events (ts, session_id, turn_id, event_type, source, target) VALUES (?1, ?2, ?3, 'injected', ?4, ?5)",
                params![timestamp, session, turn, source, target.as_str()],
            )
            .await;
    }

    fn recent_signal(target: &str, signal: &str) -> RegionTransition {
        RegionTransition {
            target: target.to_string(),
            timestamp: db::timestamp_now(),
            signal: signal.to_string(),
        }
    }

    fn continuation_count(signals: &[RegionTransition], now: String) -> Option<f64> {
        match best_continuation(signals, now) {
            Some(item) => Some(item.count),
            None => None,
        }
    }

    #[test]
    fn decay_and_feedback_adjust_counts() {
        let now = db::timestamp_now();
        let half_life_old = RegionTransition {
            target: "b".to_string(),
            timestamp: "2026-06-29T00:00:00".to_string(),
            signal: "transition".to_string(),
        };
        let half_life_now = "2026-07-13T00:00:00".to_string();
        let now_epoch = crate::timeutil::timestamp_to_epoch_seconds(&half_life_now).unwrap_or(0);
        let weight = decayed_signal_weight(&half_life_old, now_epoch).unwrap_or(0.0);
        assert!((weight - 0.5).abs() < 0.000_001);
        let best = best_continuation(&[half_life_old], half_life_now);
        assert!(
            best.is_none(),
            "one half-decayed occurrence is below the gate"
        );

        let base = [
            recent_signal("b", "transition"),
            recent_signal("b", "transition"),
        ];
        let followed = [
            base[0].clone(),
            base[1].clone(),
            recent_signal("b", "followed"),
        ];
        let rejected = [
            base[0].clone(),
            base[1].clone(),
            recent_signal("b", "rejected"),
        ];
        let base_count = continuation_count(&base, now.clone());
        let followed_count = continuation_count(&followed, now.clone());
        let rejected_count = continuation_count(&rejected, now);
        assert!(followed_count > base_count);
        assert!(rejected_count < base_count);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn regions_and_transitions_respect_turn_and_session_boundaries() {
        let conn = db::block_on(harness());
        let cases = [
            ("s1", "s1-1", "2026-07-13T01:00:01", Some("a")),
            ("s1", "s1-2", "2026-07-13T01:00:02", Some("b")),
            ("s1", "s1-3", "2026-07-13T01:00:03", Some("b")),
            ("s1", "s1-4", "2026-07-13T01:00:04", None),
            ("s1", "s1-5", "2026-07-13T01:00:05", Some("c")),
            ("s1", "s1-6", "2026-07-13T01:00:06", Some("a")),
            ("s1", "s1-7", "2026-07-13T01:00:07", Some("b")),
            ("s2", "s2-1", "2026-07-13T01:00:08", Some("b")),
            ("old", "old-1", "2026-04-01T01:00:01", Some("a")),
            ("old", "old-2", "2026-04-01T01:00:02", Some("b")),
        ];
        for (session, turn, timestamp, region) in cases {
            db::block_on(add_turn(&conn, (session, turn, timestamp, region, 0.6)));
        }
        db::block_on(add_turn(
            &conn,
            ("top", "top-1", "2026-07-13T01:00:09", Some("low"), 0.2),
        ));
        let _ = db::block_on(conn.execute(
            "INSERT INTO edges (source, target, confidence) VALUES ('high', 'detail', 0.9)",
            (),
        ));
        let _ = db::block_on(conn.execute(
            "INSERT INTO events (ts, session_id, turn_id, event_type, source, target) VALUES ('2026-07-13T01:00:09', 'top', 'top-1', 'injected', 'high', 'detail')",
            (),
        ));
        let Ok(region) = db::block_on(db::load_turn_region(&conn, "top-1")) else {
            unreachable!("region loads")
        };
        assert_eq!(region.as_deref(), Some("high"));
        let Ok(missing) = db::block_on(db::load_turn_region(&conn, "s1-4")) else {
            unreachable!("empty region loads")
        };
        assert!(missing.is_none());

        let Ok(signals) = db::block_on(db::load_region_transitions(&conn, "a", 60)) else {
            unreachable!("transitions load")
        };
        let mut transitions = Vec::new();
        for signal in &signals {
            if signal.signal == "transition" {
                transitions.push(signal);
            }
        }
        assert_eq!(transitions.len(), 2);
        for signal in transitions {
            assert_eq!(signal.target, "b");
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn suggestion_gate_writes_event_and_regionless_turn_suppresses_it() {
        let conn = db::block_on(harness());
        let timestamps = [
            "2026-07-13T00:00:01",
            "2026-07-13T00:00:02",
            "2026-07-13T00:00:03",
            "2026-07-13T00:00:04",
            "2026-07-13T00:00:05",
            "2026-07-13T00:00:06",
        ];
        let regions = [Some("a"), Some("b"), Some("a"), Some("b"), Some("a"), None];
        for index in 0..regions.len() {
            db::block_on(add_turn(
                &conn,
                (
                    "chat",
                    &format!("turn-{index}"),
                    timestamps[index],
                    regions[index],
                    0.8,
                ),
            ));
        }
        let line = match db::block_on(sequence_line_async(&conn, "chat", "turn-5")) {
            Ok(line) => line,
            Err(err) => unreachable!("suggestion computes: {err}"),
        };
        assert!(line.is_some_and(|text| text.contains("after a")));
        let count = db::block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT COUNT(*) FROM events WHERE turn_id = 'turn-5' AND event_type = 'suggested'",
                    (),
                )
                .await
            else {
                return 0;
            };
            let Ok(Some(row)) = rows.next().await else {
                return 0;
            };
            db::int(row.get_value(0).unwrap_or(turso::Value::Integer(0)))
        });
        assert_eq!(count, 1);

        db::block_on(add_turn(
            &conn,
            ("chat", "turn-6", "2026-07-13T00:00:07", None, 0.0),
        ));
        let Ok(no_line) = db::block_on(sequence_line_async(&conn, "chat", "turn-6")) else {
            unreachable!("regionless lookup succeeds")
        };
        assert!(no_line.is_none());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn feedback_follows_or_rejects_once() {
        let conn = db::block_on(harness());
        db::block_on(add_turn(
            &conn,
            ("chat", "turn-a", "2026-07-13T00:00:01", Some("a"), 0.8),
        ));
        let _ = db::block_on(conn.execute(
            "INSERT INTO events (ts, session_id, turn_id, actor_id, event_type, source, target) VALUES ('2026-07-13T00:00:01', 'chat', 'turn-a', 'sequence', 'suggested', 'a', 'b')",
            (),
        ));
        db::block_on(add_turn(
            &conn,
            ("chat", "turn-b", "2026-07-13T00:00:02", Some("b"), 0.8),
        ));
        let Ok(()) = db::block_on(record_feedback_async(&conn, "chat", "turn-b")) else {
            unreachable!("feedback records")
        };
        let Ok(()) = db::block_on(record_feedback_async(&conn, "chat", "turn-b")) else {
            unreachable!("feedback retry is safe")
        };
        let outcomes = db::block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT event_type FROM events WHERE turn_id = 'turn-b' AND actor_id = 'sequence'",
                    (),
                )
                .await
            else {
                return Vec::new();
            };
            let mut values = Vec::new();
            while let Ok(Some(row)) = rows.next().await {
                values.push(db::text(row.get_value(0).unwrap_or(turso::Value::Null)));
            }
            values
        });
        assert_eq!(outcomes, vec!["followed".to_string()]);

        db::block_on(add_turn(
            &conn,
            ("other", "other-a", "2026-07-13T00:00:03", Some("a"), 0.8),
        ));
        let _ = db::block_on(conn.execute(
            "INSERT INTO events (ts, session_id, turn_id, actor_id, event_type, source, target) VALUES ('2026-07-13T00:00:03', 'other', 'other-a', 'sequence', 'suggested', 'a', 'b')",
            (),
        ));
        db::block_on(add_turn(
            &conn,
            ("other", "other-c", "2026-07-13T00:00:04", Some("c"), 0.8),
        ));
        let Ok(()) = db::block_on(record_feedback_async(&conn, "other", "other-c")) else {
            unreachable!("rejection records")
        };
        let rejected = db::block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT COUNT(*) FROM events WHERE turn_id = 'other-c' AND event_type = 'rejected'",
                    (),
                )
                .await
            else {
                return 0;
            };
            let Ok(Some(row)) = rows.next().await else {
                return 0;
            };
            row.get::<i64>(0).unwrap_or(0)
        });
        assert_eq!(rejected, 1);
    }
}
