use std::collections::HashMap;
use std::error::Error;

use serde_json::Value as JsonValue;
use turso::{Connection, Value, params};

use crate::memory::db;
use crate::memory::graph::Edge;

const HALF_LIFE_CYCLES: f64 = 20.0; // replay-tunable.
const POSITION_EPSILON: f64 = 0.05; // replay-tunable.
const DIRECTION_SNAPSHOTS: usize = 5; // replay-tunable.
const DIRECTION_NORM_EPSILON: f64 = 1e-9; // replay-tunable.
const TIER_ACTIVE: f64 = 2.0; // replay-tunable.
const TIER_MARKED: f64 = 3.0; // replay-tunable.

#[derive(Clone)]
pub struct Event {
    pub timestamp: String,
    pub session_id: String,
    pub turn_id: String,
    pub actor_type: String,
    pub actor_id: String,
    pub event_type: String,
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub payload: String,
}

#[derive(Clone)]
pub struct Actor {
    pub actor_type: String,
    pub actor_id: String,
}

#[derive(Clone)]
pub struct EdgeRef {
    pub source: String,
    pub target: String,
}

#[derive(Clone)]
pub struct EventContext {
    pub session_id: String,
    pub turn_id: String,
}

#[derive(Default)]
pub struct Attribution {
    pub expert: Option<String>,
    pub expert_score: f64,
    pub injected_edges: usize,
    pub top_edges: Vec<String>,
}

struct EventBody {
    weight: f64,
    payload: String,
}

impl Event {
    pub fn injected(actor: Actor, edge: EdgeRef, context: EventContext) -> Self {
        Self::new(
            actor,
            "injected",
            edge,
            context,
            EventBody::new(1.0, String::new()),
        )
    }

    pub fn upsert(actor: Actor, edge: EdgeRef, context: EventContext, existed: bool) -> Self {
        let event_type = if existed { "updated" } else { "created" };
        Self::new(
            actor,
            event_type,
            edge,
            context,
            EventBody::new(1.0, String::new()),
        )
    }

    pub fn upsert_with_payload(
        actor: Actor,
        edge: EdgeRef,
        context: EventContext,
        existed: bool,
        payload: String,
    ) -> Self {
        let event_type = if existed { "updated" } else { "created" };
        Self::new(
            actor,
            event_type,
            edge,
            context,
            EventBody::new(1.0, payload),
        )
    }

    pub fn updated(actor: Actor, edge: EdgeRef, context: EventContext, payload: String) -> Self {
        Self::new(
            actor,
            "updated",
            edge,
            context,
            EventBody::new(1.0, payload),
        )
    }

    pub fn retrieved(actor: Actor, edge: EdgeRef, context: EventContext) -> Self {
        Self::new(
            actor,
            "retrieved",
            edge,
            context,
            EventBody::new(TIER_ACTIVE, String::new()),
        )
    }

    pub fn marked(actor: Actor, edge: EdgeRef, context: EventContext, reason: &str) -> Self {
        let payload = serde_json::json!({"reason": reason}).to_string();
        Self::new(
            actor,
            "marked",
            edge,
            context,
            EventBody::new(TIER_MARKED, payload),
        )
    }

    pub fn pruned(actor: Actor, edge: EdgeRef, context: EventContext, payload: String) -> Self {
        Self::new(actor, "pruned", edge, context, EventBody::new(1.0, payload))
    }

    pub fn routed(context: EventContext, payload: String) -> Self {
        Self::new(
            Actor::new("system", "router"),
            "routed",
            EdgeRef::new("", ""),
            context,
            EventBody::new(1.0, payload),
        )
    }

    // turn_id is empty for a session-level outcome and set for per-move credit
    // (record_move_outcome in reinforce.rs); both scopes share this constructor.
    pub fn outcome(session_id: &str, turn_id: &str, score: f64, payload: String) -> Self {
        Self::new(
            Actor::new("system", "reinforcement"),
            "outcome",
            EdgeRef::new("", ""),
            EventContext::new(session_id, turn_id),
            EventBody::new(score, payload),
        )
    }

    pub fn reinforced(
        edge: EdgeRef,
        session_id: &str,
        turn_id: &str,
        delta: f64,
        payload: String,
    ) -> Self {
        Self::new(
            Actor::new("system", "reinforcement"),
            "reinforced",
            edge,
            EventContext::new(session_id, turn_id),
            EventBody::new(delta, payload),
        )
    }

    fn new(
        actor: Actor,
        event_type: &str,
        edge: EdgeRef,
        context: EventContext,
        body: EventBody,
    ) -> Self {
        Self {
            timestamp: db::timestamp_now(),
            session_id: context.session_id,
            turn_id: context.turn_id,
            actor_type: actor.actor_type,
            actor_id: actor.actor_id,
            event_type: event_type.to_string(),
            source: edge.source,
            target: edge.target,
            weight: body.weight,
            payload: body.payload,
        }
    }
}

pub fn pruned_events(
    actor: Actor,
    context: EventContext,
    edges: &[Edge],
) -> Result<Vec<Event>, Box<dyn Error + Send + Sync>> {
    let mut events = Vec::new();
    for edge in edges {
        let payload = serde_json::to_string(edge)?;
        events.push(Event::pruned(
            actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            context.clone(),
            payload,
        ));
    }
    Ok(events)
}

impl Actor {
    pub fn new(actor_type: &str, actor_id: &str) -> Self {
        Self {
            actor_type: actor_type.to_string(),
            actor_id: actor_id.to_string(),
        }
    }
}

impl EdgeRef {
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source: source.to_string(),
            target: target.to_string(),
        }
    }
}

impl EventContext {
    pub fn new(session_id: &str, turn_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
        }
    }
}

impl EventBody {
    fn new(weight: f64, payload: String) -> Self {
        Self { weight, payload }
    }
}

// Per-batch cache: one edges-table lookup per unique (source, target).
// Row presence means the edge is live (not expired); the embedding may still be NULL.
struct EdgeInfo {
    live: bool,
    embedding: Option<Vec<f64>>,
}

type EdgeCache = HashMap<(String, String), EdgeInfo>;

pub fn record_events(events: &[Event]) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = db::open_connection()?;
    db::block_on(async {
        conn.execute("BEGIN", ()).await?;
        let result = record_events_in_tx(&conn, events).await;
        match result {
            Ok(()) => {
                conn.execute("COMMIT", ()).await?;
                Ok(())
            }
            Err(err) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(err)
            }
        }
    })
}

pub(crate) async fn record_events_in_tx(
    conn: &Connection,
    events: &[Event],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if events.is_empty() {
        return Ok(());
    }
    let mut cache: EdgeCache = HashMap::new();
    let mut batches: Vec<ActorBatch> = Vec::new();
    for event in events {
        insert_event(conn, event).await?;
        if event.source.is_empty() || event.target.is_empty() {
            continue;
        }
        push_actor_event(&mut batches, event.clone());
        if !edge_info(conn, &mut cache, &event.source, &event.target)
            .await?
            .live
        {
            continue;
        }
        upsert_edge_view(conn, event).await?;
    }
    let mut index = 0usize;
    while index < batches.len() {
        let batch = &batches[index];
        update_actor_position(conn, &mut cache, batch).await?;
        index += 1;
    }
    Ok(())
}

// Audit only: folds the full ledger (undecayed, dead edges skipped) and reports
// how far each actor's incremental position sits from that fold. It never
// writes — the incremental decayed values stay authoritative; a large distance
// on a low-activity actor is a bug signal, on an active actor it mostly
// reflects decay.
pub fn recompute_positions() -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let conn = db::open_connection()?;
    db::block_on(async {
        let recomputed = load_recomputed_positions(&conn).await?;
        let mut logs = audit_positions(&conn, recomputed).await?;
        logs.sort();
        for line in &logs {
            crate::runlog::log("recompute", line);
        }
        Ok(logs)
    })
}

pub fn load_actor_position(actor_type: &str, actor_id: &str) -> Option<Vec<f64>> {
    let conn = match db::open_connection() {
        Ok(conn) => conn,
        Err(_) => return None,
    };
    db::block_on(load_actor_position_async(&conn, actor_type, actor_id)).unwrap_or_default()
}

pub fn load_actor_direction(actor_type: &str, actor_id: &str) -> Option<Vec<f64>> {
    let conn = match db::open_connection() {
        Ok(conn) => conn,
        Err(_) => return None,
    };
    db::block_on(load_actor_direction_async(&conn, actor_type, actor_id)).unwrap_or_default()
}

pub fn turn_attribution(turn_id: &str) -> Attribution {
    let conn = match db::open_connection() {
        Ok(conn) => conn,
        Err(_) => return Attribution::default(),
    };
    db::block_on(turn_attribution_async(&conn, turn_id)).unwrap_or_default()
}

pub(crate) fn cosine_similarity(
    left: &[f64],
    right: &[f64],
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    Ok(1.0 - cosine_distance(left, right)?)
}

pub(crate) fn normalized_vector(vector: &[f64]) -> Option<Vec<f64>> {
    let mut norm = 0.0;
    let mut index = 0usize;
    while index < vector.len() {
        norm += vector[index] * vector[index];
        index += 1;
    }
    if norm <= DIRECTION_NORM_EPSILON {
        return None;
    }
    let scale = norm.sqrt();
    let mut normalized = Vec::with_capacity(vector.len());
    let mut normalize_index = 0usize;
    while normalize_index < vector.len() {
        normalized.push(vector[normalize_index] / scale);
        normalize_index += 1;
    }
    Some(normalized)
}

async fn turn_attribution_async(
    conn: &Connection,
    turn_id: &str,
) -> Result<Attribution, Box<dyn Error + Send + Sync>> {
    let mut attribution = Attribution::default();
    let mut rows = conn
        .query(
            "SELECT event_type, payload FROM events WHERE turn_id = ?1 ORDER BY id",
            params![turn_id],
        )
        .await?;
    while let Some(row) = rows.next().await? {
        let event_type = db::text(row.get_value(0)?);
        if event_type == "injected" {
            attribution.injected_edges += 1;
            continue;
        }
        if event_type != "routed" {
            continue;
        }
        parse_routed_payload(&db::text(row.get_value(1)?), &mut attribution);
    }
    attribution.top_edges = load_top_injected_edges(conn, turn_id).await?;
    Ok(attribution)
}

fn parse_routed_payload(payload: &str, attribution: &mut Attribution) {
    let Ok(json) = serde_json::from_str::<JsonValue>(payload) else {
        return;
    };
    let mut expert = None;
    if let Some(value) = json.get("expert")
        && let Some(text) = value.as_str()
    {
        expert = Some(text.to_string());
    }
    attribution.expert = expert;
    attribution.expert_score = match json.get("score") {
        Some(value) => value.as_f64().unwrap_or(0.0),
        None => 0.0,
    };
}

// The full set of live beliefs retrieval put in front of the model this turn —
// memory's prediction for the turn. The memory pass uses it to refresh decay
// clocks and to extract only what the prediction missed.
pub(super) async fn load_injected_edges(
    conn: &Connection,
    turn_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT ed.source, ed.target, ed.content, ed.stability, ed.confidence, ed.importance, ed.subject, ed.date, ed.occurred_at, ed.expired_at, ed.memory_type, ed.origin_session FROM events ev JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target WHERE ev.turn_id = ?1 AND ev.event_type = 'injected' AND ed.expired_at IS NULL GROUP BY ed.source, ed.target",
            params![turn_id],
        )
        .await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        if let Some(edge) = db::parse_edge_row(&row) {
            edges.push(edge);
        }
    }
    Ok(edges)
}

async fn load_top_injected_edges(
    conn: &Connection,
    turn_id: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT ed.content, MAX(ed.confidence) FROM events ev JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target WHERE ev.turn_id = ?1 AND ev.event_type = 'injected' AND ed.expired_at IS NULL GROUP BY ed.source, ed.target, ed.content ORDER BY MAX(ed.confidence) DESC LIMIT 3",
            params![turn_id],
        )
        .await?;
    let mut top = Vec::new();
    while let Some(row) = rows.next().await? {
        let content = db::text(row.get_value(0)?);
        if !content.trim().is_empty() {
            top.push(content);
        }
    }
    Ok(top)
}

struct ActorBatch {
    actor_type: String,
    actor_id: String,
    events: Vec<Event>,
}

struct PositionState {
    sum_vector: Option<Vec<f64>>,
    total_weight: f64,
    updated_at: String,
}

async fn insert_event(
    conn: &Connection,
    event: &Event,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, source, target, weight, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            event.timestamp.as_str(),
            event.session_id.as_str(),
            event.turn_id.as_str(),
            event.actor_type.as_str(),
            event.actor_id.as_str(),
            event.event_type.as_str(),
            event.source.as_str(),
            event.target.as_str(),
            event.weight,
            event.payload.as_str(),
        ],
    )
    .await?;
    Ok(())
}

async fn edge_info<'cache>(
    conn: &Connection,
    cache: &'cache mut EdgeCache,
    source: &str,
    target: &str,
) -> Result<&'cache EdgeInfo, Box<dyn Error + Send + Sync>> {
    let key = (source.to_string(), target.to_string());
    if !cache.contains_key(&key) {
        let info = load_edge_info(conn, source, target).await?;
        cache.insert(key.clone(), info);
    }
    match cache.get(&key) {
        Some(info) => Ok(info),
        None => Err("edge cache lookup failed".into()),
    }
}

// NULL-embedding columns must be guarded with CASE WHEN everywhere:
// turso's vector_extract(NULL) throws "Expected blob value" AND aborts the
// enclosing transaction — this silently killed every memory/dream write
// batch containing a freshly created (not-yet-embedded) edge.
async fn load_edge_info(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<EdgeInfo, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT CASE WHEN embedding IS NULL THEN NULL ELSE vector_extract(embedding) END FROM edges WHERE source = ?1 AND target = ?2 AND expired_at IS NULL LIMIT 1",
            params![source, target],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        let embedding = match row.get_value(0).ok() {
            Some(value) => parse_vector_blob(value),
            None => None,
        };
        return Ok(EdgeInfo {
            live: true,
            embedding,
        });
    }
    Ok(EdgeInfo {
        live: false,
        embedding: None,
    })
}

async fn upsert_edge_view(
    conn: &Connection,
    event: &Event,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "INSERT INTO edge_actor_views (actor_type, actor_id, source, target, views, last_visited_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6) ON CONFLICT(actor_type, actor_id, source, target) DO UPDATE SET views = views + excluded.views, last_visited_at = excluded.last_visited_at",
        params![
            event.actor_type.as_str(),
            event.actor_id.as_str(),
            event.source.as_str(),
            event.target.as_str(),
            event.weight,
            event.timestamp.as_str(),
        ],
    )
    .await?;
    Ok(())
}

fn push_actor_event(batches: &mut Vec<ActorBatch>, event: Event) {
    let mut index = 0usize;
    while index < batches.len() {
        let batch = &mut batches[index];
        if batch.actor_type == event.actor_type && batch.actor_id == event.actor_id {
            batch.events.push(event);
            return;
        }
        index += 1;
    }
    batches.push(ActorBatch {
        actor_type: event.actor_type.clone(),
        actor_id: event.actor_id.clone(),
        events: vec![event],
    });
}

async fn update_actor_position(
    conn: &Connection,
    cache: &mut EdgeCache,
    batch: &ActorBatch,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut state = load_position_state(conn, &batch.actor_type, &batch.actor_id).await?;
    let mut should_persist = decay_position_state(&mut state);
    should_persist |= add_event_embeddings(conn, cache, &mut state, &batch.events).await?;
    if !should_persist {
        return Ok(());
    }
    state.updated_at = db::timestamp_now();
    let position = position_from_state(&state);
    persist_position(conn, &batch.actor_type, &batch.actor_id, &state, &position).await?;
    maybe_snapshot_position(
        conn,
        &batch.actor_type,
        &batch.actor_id,
        &position,
        &state.updated_at,
    )
    .await?;
    Ok(())
}

fn decay_position_state(state: &mut PositionState) -> bool {
    let decay = 0.5_f64.powf(1.0 / HALF_LIFE_CYCLES);
    let mut changed = false;
    if let Some(sum_vector) = state.sum_vector.as_mut() {
        scale_vector(sum_vector, decay);
        changed = true;
    }
    if state.total_weight > 0.0 {
        state.total_weight *= decay;
        changed = true;
    }
    changed
}

async fn add_event_embeddings(
    conn: &Connection,
    cache: &mut EdgeCache,
    state: &mut PositionState,
    events: &[Event],
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let mut changed = false;
    for event in events {
        let info = edge_info(conn, cache, &event.source, &event.target).await?;
        let Some(embedding) = info.embedding.clone() else {
            continue;
        };
        add_to_state(state, &embedding, event.weight)?;
        changed = true;
    }
    Ok(changed)
}

fn add_to_state(
    state: &mut PositionState,
    embedding: &[f64],
    weight: f64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if let Some(sum_vector) = state.sum_vector.as_mut() {
        add_weighted_vector(sum_vector, embedding, weight)?;
    } else {
        state.sum_vector = Some(weighted_vector(embedding, weight));
    }
    state.total_weight += weight;
    Ok(())
}

async fn load_position_state(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> Result<PositionState, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT CASE WHEN sum_vector IS NULL THEN NULL ELSE vector_extract(sum_vector) END, total_weight, updated_at FROM actor_positions WHERE actor_type = ?1 AND actor_id = ?2 LIMIT 1",
            params![actor_type, actor_id],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        let sum_vector = match row.get_value(0).ok() {
            Some(value) => parse_vector_blob(value),
            None => None,
        };
        let total_weight = match row.get_value(1).ok() {
            Some(value) => db::real(value),
            None => 0.0,
        };
        let updated_at = match row.get_value(2).ok() {
            Some(value) => db::text(value),
            None => db::timestamp_now(),
        };
        return Ok(PositionState {
            sum_vector,
            total_weight,
            updated_at,
        });
    }
    Ok(PositionState {
        sum_vector: None,
        total_weight: 0.0,
        updated_at: db::timestamp_now(),
    })
}

async fn load_actor_position_async(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> Result<Option<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let state = load_position_state(conn, actor_type, actor_id).await?;
    Ok(position_from_state(&state))
}

async fn load_actor_direction_async(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> Result<Option<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT CASE WHEN embedding IS NULL THEN NULL ELSE vector_extract(embedding) END FROM actor_position_history WHERE actor_type = ?1 AND actor_id = ?2 ORDER BY recorded_at DESC LIMIT ?3",
            params![actor_type, actor_id, DIRECTION_SNAPSHOTS as i64],
        )
        .await?;
    let mut snapshots: Vec<Vec<f64>> = Vec::new();
    while let Some(row) = rows.next().await? {
        let embedding = match row.get_value(0).ok() {
            Some(value) => parse_vector_blob(value),
            None => None,
        };
        if let Some(vector) = embedding {
            snapshots.push(vector);
        }
    }
    if snapshots.len() < 2 {
        return Ok(None);
    }
    let newest = &snapshots[0];
    let oldest = &snapshots[snapshots.len() - 1];
    if newest.len() != oldest.len() {
        return Ok(None);
    }
    let mut delta = Vec::with_capacity(newest.len());
    let mut index = 0usize;
    while index < newest.len() {
        delta.push(newest[index] - oldest[index]);
        index += 1;
    }
    Ok(normalized_vector(&delta))
}

pub(crate) fn parse_vector_blob(value: Value) -> Option<Vec<f64>> {
    let json_text = db::text(value);
    if json_text.is_empty() {
        return None;
    }
    db::parse_vector32_json(&json_text).ok()
}

fn weighted_vector(vector: &[f64], weight: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(vector.len());
    let mut index = 0usize;
    while index < vector.len() {
        result.push(vector[index] * weight);
        index += 1;
    }
    result
}

fn scale_vector(vector: &mut [f64], scale: f64) {
    let mut index = 0usize;
    while index < vector.len() {
        vector[index] *= scale;
        index += 1;
    }
}

fn add_weighted_vector(
    sum_vector: &mut [f64],
    vector: &[f64],
    weight: f64,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if sum_vector.len() != vector.len() {
        return Err("position vector dimension mismatch".into());
    }
    let mut index = 0usize;
    while index < sum_vector.len() {
        sum_vector[index] += vector[index] * weight;
        index += 1;
    }
    Ok(())
}

fn position_from_state(state: &PositionState) -> Option<Vec<f64>> {
    if state.total_weight <= 0.0 {
        return None;
    }
    let sum_vector = state.sum_vector.as_ref()?;
    let mut position = Vec::with_capacity(sum_vector.len());
    let mut index = 0usize;
    while index < sum_vector.len() {
        position.push(sum_vector[index] / state.total_weight);
        index += 1;
    }
    Some(position)
}

async fn persist_position(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
    state: &PositionState,
    position: &Option<Vec<f64>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let embedding = match position {
        Some(vector) => Some(db::vector32_json(vector)?),
        None => None,
    };
    conn.execute(
        "INSERT INTO actor_positions (actor_type, actor_id, sum_vector, total_weight, updated_at) VALUES (?1, ?2, CASE WHEN ?3 IS NULL THEN NULL ELSE vector32(?3) END, ?4, ?5) ON CONFLICT(actor_type, actor_id) DO UPDATE SET sum_vector = excluded.sum_vector, total_weight = excluded.total_weight, updated_at = excluded.updated_at",
        params![
            actor_type,
            actor_id,
            embedding.as_deref(),
            state.total_weight,
            state.updated_at.as_str(),
        ],
    )
    .await?;
    Ok(())
}

async fn maybe_snapshot_position(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
    position: &Option<Vec<f64>>,
    recorded_at: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(position) = position else {
        return Ok(());
    };
    let Some(last_snapshot) = load_last_snapshot(conn, actor_type, actor_id).await? else {
        snapshot_position(conn, actor_type, actor_id, position, recorded_at).await?;
        return Ok(());
    };
    if cosine_distance(&last_snapshot, position)? > POSITION_EPSILON {
        snapshot_position(conn, actor_type, actor_id, position, recorded_at).await?;
    }
    Ok(())
}

async fn load_last_snapshot(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> Result<Option<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT CASE WHEN embedding IS NULL THEN NULL ELSE vector_extract(embedding) END FROM actor_position_history WHERE actor_type = ?1 AND actor_id = ?2 ORDER BY recorded_at DESC LIMIT 1",
            params![actor_type, actor_id],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        return Ok(match row.get_value(0).ok() {
            Some(value) => parse_vector_blob(value),
            None => None,
        });
    }
    Ok(None)
}

async fn snapshot_position(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
    position: &[f64],
    recorded_at: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let json = db::vector32_json(position)?;
    conn.execute(
        "INSERT INTO actor_position_history (actor_type, actor_id, embedding, recorded_at) VALUES (?1, ?2, vector32(?3), ?4)",
        params![actor_type, actor_id, json.as_str(), recorded_at],
    )
    .await?;
    Ok(())
}

pub(crate) fn cosine_distance(
    left: &[f64],
    right: &[f64],
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    if left.len() != right.len() {
        return Err("position vector dimension mismatch".into());
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    let mut index = 0usize;
    while index < left.len() {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
        index += 1;
    }
    if left_norm <= 0.0 || right_norm <= 0.0 {
        return Ok(1.0);
    }
    Ok(1.0 - dot / (left_norm.sqrt() * right_norm.sqrt()))
}

async fn load_recomputed_positions(
    conn: &Connection,
) -> Result<HashMap<(String, String), PositionState>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT actor_type, actor_id, source, target, weight FROM events WHERE source != '' AND target != '' ORDER BY id",
            (),
        )
        .await?;
    let mut event_rows: Vec<(String, String, String, String, f64)> = Vec::new();
    while let Some(row) = rows.next().await? {
        event_rows.push((
            db::text(row.get_value(0)?),
            db::text(row.get_value(1)?),
            db::text(row.get_value(2)?),
            db::text(row.get_value(3)?),
            db::real(row.get_value(4)?),
        ));
    }
    let mut cache: EdgeCache = HashMap::new();
    let mut positions: HashMap<(String, String), PositionState> = HashMap::new();
    for (actor_type, actor_id, source, target, weight) in event_rows {
        let info = edge_info(conn, &mut cache, &source, &target).await?;
        let Some(embedding) = info.embedding.clone() else {
            continue;
        };
        let state = positions
            .entry((actor_type, actor_id))
            .or_insert(PositionState {
                sum_vector: None,
                total_weight: 0.0,
                updated_at: String::new(),
            });
        add_to_state(state, &embedding, weight)?;
    }
    Ok(positions)
}

async fn audit_positions(
    conn: &Connection,
    recomputed: HashMap<(String, String), PositionState>,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut logs = Vec::new();
    let mut current_rows = load_current_positions(conn).await?;
    for ((actor_type, actor_id), state) in recomputed {
        let folded_position = position_from_state(&state);
        match current_rows.remove(&(actor_type.clone(), actor_id.clone())) {
            Some(current) => {
                let current_position = position_from_state(&current);
                if let (Some(left), Some(right)) =
                    (current_position.as_ref(), folded_position.as_ref())
                {
                    let distance = cosine_distance(left, right)?;
                    logs.push(format!(
                        "{actor_type}:{actor_id} undecayed-fold distance {distance:.4}"
                    ));
                }
            }
            None => logs.push(format!(
                "{actor_type}:{actor_id} has events but no position row"
            )),
        }
    }
    for ((actor_type, actor_id), _) in current_rows {
        logs.push(format!(
            "{actor_type}:{actor_id} has a position but no embedded events"
        ));
    }
    Ok(logs)
}

async fn load_current_positions(
    conn: &Connection,
) -> Result<HashMap<(String, String), PositionState>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT actor_type, actor_id, CASE WHEN sum_vector IS NULL THEN NULL ELSE vector_extract(sum_vector) END, total_weight, updated_at FROM actor_positions",
            (),
        )
        .await?;
    let mut positions = HashMap::new();
    while let Some(row) = rows.next().await? {
        let actor_type = db::text(row.get_value(0)?);
        let actor_id = db::text(row.get_value(1)?);
        let state = PositionState {
            sum_vector: match row.get_value(2).ok() {
                Some(value) => parse_vector_blob(value),
                None => None,
            },
            total_weight: db::real(row.get_value(3)?),
            updated_at: db::text(row.get_value(4)?),
        };
        positions.insert((actor_type, actor_id), state);
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::{Connection, Edge, load_injected_edges};
    use crate::memory::db;
    use turso::{Builder, params};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";

    // Fresh in-memory graph: three edges, one of them already expired.
    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let _ = conn.execute(EDGES_DDL, ()).await;
        let _ = conn.execute(EVENTS_DDL, ()).await;
        insert_edge(&conn, "alice", "berlin", "2020-01-01", None).await;
        insert_edge(&conn, "carol", "delhi", "2020-01-01", Some("2024-05-05")).await;
        insert_edge(&conn, "erin", "faro", "2020-01-01", None).await;
        conn
    }

    async fn insert_edge(
        conn: &Connection,
        source: &str,
        target: &str,
        date: &str,
        expired_at: Option<&str>,
    ) {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, date, expired_at) VALUES (?1, ?2, 'x', ?3, ?4)",
                params![source, target, date, expired_at],
            )
            .await;
    }

    async fn insert_injected(conn: &Connection, turn_id: &str, source: &str, target: &str) {
        let _ = conn
            .execute(
                "INSERT INTO events (turn_id, event_type, source, target) VALUES (?1, 'injected', ?2, ?3)",
                params![turn_id, source, target],
            )
            .await;
    }

    async fn edge_date(conn: &Connection, source: &str, target: &str) -> String {
        let Ok(mut rows) = conn
            .query(
                "SELECT date FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("date query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(value) = row.get_value(0) else {
            unreachable!("date column reads")
        };
        db::text(value)
    }

    fn injected_for(conn: &Connection, turn_id: &str) -> Vec<Edge> {
        let Ok(edges) = db::block_on(load_injected_edges(conn, turn_id)) else {
            unreachable!("loader succeeds")
        };
        edges
    }

    #[test]
    fn loader_returns_deduplicated_live_edges_for_the_turn_only() {
        let conn = db::block_on(harness());
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        // Same edge injected twice in one turn — retrieval hit it more than once.
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        db::block_on(insert_injected(&conn, "turn-1", "carol", "delhi"));
        db::block_on(insert_injected(&conn, "turn-2", "erin", "faro"));

        let injected = injected_for(&conn, "turn-1");
        assert_eq!(injected.len(), 1, "expired edge and duplicate are excluded");
        assert_eq!(injected[0].source, "alice");
        assert_eq!(injected[0].target, "berlin");

        assert_eq!(injected_for(&conn, "turn-2").len(), 1);
        assert!(injected_for(&conn, "turn-unknown").is_empty());
    }

    #[test]
    fn refresh_sets_today_on_injected_edges_and_spares_the_rest() {
        let conn = db::block_on(harness());
        db::block_on(insert_injected(&conn, "turn-1", "alice", "berlin"));
        db::block_on(insert_injected(&conn, "turn-2", "erin", "faro"));

        let today = crate::prompts::today_iso();
        let injected = injected_for(&conn, "turn-1");
        let Ok(refreshed) = db::block_on(db::touch_edge_dates(&conn, &injected, &today)) else {
            unreachable!("refresh succeeds")
        };
        assert_eq!(refreshed, 1);

        assert_eq!(db::block_on(edge_date(&conn, "alice", "berlin")), today);
        // Injected in a different turn — its clock keeps running.
        assert_eq!(db::block_on(edge_date(&conn, "erin", "faro")), "2020-01-01");
    }

    #[test]
    fn refresh_never_touches_an_expired_edge() {
        let conn = db::block_on(harness());
        let expired = Edge {
            source: "carol".to_string(),
            target: "delhi".to_string(),
            content: "x".to_string(),
            stability: 0.0,
            confidence: 0.0,
            importance: 0.0,
            date: "2020-01-01".to_string(),
            subject: false,
            occurred_at: None,
            expired_at: Some("2024-05-05".to_string()),
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
        };
        let today = crate::prompts::today_iso();
        let Ok(refreshed) = db::block_on(db::touch_edge_dates(&conn, &[expired], &today)) else {
            unreachable!("refresh succeeds")
        };
        assert_eq!(refreshed, 0);
        assert_eq!(
            db::block_on(edge_date(&conn, "carol", "delhi")),
            "2020-01-01"
        );
    }
}
