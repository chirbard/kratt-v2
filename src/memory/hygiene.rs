// Deterministic graph hygiene. No LLM calls: every check is a SQL query plus
// imperative Rust. Checks that need judgment (which spelling of a duplicate is
// canonical, whether an orphan node is worth keeping) are report-only — the
// dreaming agents decide those. Nothing here hard-deletes; expiry is the
// graph's reversible tombstone.

use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;

use turso::{Connection, params};

use super::db;
use super::graph::edge_ceiling;
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

// Enough examples to recognise the pattern in a log line, not a full dump.
const MAX_REPORTED_EXAMPLES: usize = 10; // replay-tunable
const W_INTEREST: f64 = 0.4; // replay-tunable.
const W_RECENCY: f64 = 0.4; // replay-tunable.
const W_DENSITY: f64 = 0.2; // replay-tunable.
const RECENCY_HALF_LIFE_DAYS: f64 = 14.0; // replay-tunable.
const DENSITY_SATURATION: usize = 10; // replay-tunable.

#[derive(Default)]
pub struct HygieneReport {
    pub expired_orphan_edges: usize,
    pub clamped_edges: usize,
    pub fixed_subject_edges: usize,
    pub orphan_node_count: usize,
    pub orphan_node_examples: Vec<String>,
    pub ambiguous_subject_facts: Vec<String>,
    pub duplicate_pairs: Vec<(String, String)>,
    pub episodic_without_session: usize,
}

impl HygieneReport {
    pub fn is_clean(&self) -> bool {
        self.expired_orphan_edges == 0
            && self.clamped_edges == 0
            && self.fixed_subject_edges == 0
            && self.orphan_node_count == 0
            && self.ambiguous_subject_facts.is_empty()
            && self.duplicate_pairs.is_empty()
            && self.episodic_without_session == 0
    }

    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        push_count(
            &mut parts,
            self.expired_orphan_edges,
            "expired",
            "orphan edge",
        );
        push_count(&mut parts, self.clamped_edges, "clamped", "edge weight");
        push_count(
            &mut parts,
            self.fixed_subject_edges,
            "fixed",
            "subject edge",
        );
        push_count(&mut parts, self.orphan_node_count, "found", "orphan node");
        push_count(
            &mut parts,
            self.ambiguous_subject_facts.len(),
            "found",
            "ambiguous fact",
        );
        push_count(
            &mut parts,
            self.duplicate_pairs.len(),
            "found",
            "duplicate-pair candidate",
        );
        push_count(
            &mut parts,
            self.episodic_without_session,
            "found",
            "sessionless episodic edge",
        );
        parts.join(", ")
    }
}

fn push_count(parts: &mut Vec<String>, count: usize, verb: &str, noun: &str) {
    if count == 0 {
        return;
    }
    let plural = if count == 1 { "" } else { "s" };
    parts.push(format!("{verb} {count} {noun}{plural}"));
}

// Auto-fixes run in one transaction; the report-only checks read afterwards.
pub fn run_hygiene(conn: &Connection) -> Result<HygieneReport, Box<dyn Error + Send + Sync>> {
    let mut report = HygieneReport::default();
    db::begin_transaction(conn)?;
    let result = db::block_on(apply_fixes(conn, &mut report));
    if let Err(err) = result {
        if let Err(rollback_err) = db::block_on(async { conn.execute("ROLLBACK", ()).await }) {
            crate::runlog::log("hygiene", &format!("rollback failed: {rollback_err}"));
        }
        return Err(err);
    }
    db::commit_transaction(conn)?;
    db::block_on(collect_reports(conn, &mut report))?;
    Ok(report)
}

async fn apply_fixes(
    conn: &Connection,
    report: &mut HygieneReport,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut events: Vec<Event> = Vec::new();
    report.expired_orphan_edges = expire_orphan_edges(conn, &mut events).await?;
    report.clamped_edges = clamp_edge_weights(conn).await?;
    fix_subject_edges(conn, report).await?;
    run_importance_pass(conn).await?;
    ledger::record_events_in_tx(conn, &events).await
}

struct ImportanceSignal {
    source: String,
    target: String,
    embedding: Option<Vec<f64>>,
    days_since_touch: Option<f64>,
}

pub(crate) async fn run_importance_pass(
    conn: &Connection,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let interests = load_interest_embeddings(conn).await?;
    let signals = load_importance_signals(conn).await?;
    let _timer =
        crate::runlog::time_step("importance", format!("recomputed {} edges", signals.len()));
    for signal in &signals {
        let alignment = max_alignment(signal.embedding.as_deref(), &interests);
        let recency = match signal.days_since_touch {
            Some(days) => recency_score(days),
            None => 0.0,
        };
        let density = neighbourhood_density(signal, &signals);
        let importance = importance_score(alignment, recency, density);
        conn.execute(
            "UPDATE edges SET importance = ?1
             WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
            params![importance, signal.source.as_str(), signal.target.as_str()],
        )
        .await?;
    }
    Ok(signals.len())
}

async fn load_interest_embeddings(
    conn: &Connection,
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let interests = db::top_interests_with(conn, usize::MAX).await?;
    let mut ids: HashSet<String> = HashSet::new();
    for interest in interests {
        ids.insert(interest.node_id);
    }
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut rows = conn
        .query(
            "SELECT id, vector_extract(embedding) FROM nodes WHERE embedding IS NOT NULL",
            (),
        )
        .await?;
    let mut embeddings = Vec::new();
    while let Some(row) = rows.next().await? {
        let node_id = db::text(row.get_value(0)?);
        if !ids.contains(&node_id) {
            continue;
        }
        let vector_text = db::text(row.get_value(1)?);
        if let Ok(vector) = db::parse_vector32_json(&vector_text) {
            embeddings.push(vector);
        }
    }
    Ok(embeddings)
}

async fn load_importance_signals(
    conn: &Connection,
) -> Result<Vec<ImportanceSignal>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT e.source, e.target,
                    CASE WHEN e.embedding IS NULL THEN '' ELSE vector_extract(e.embedding) END,
                    julianday('now') - julianday(MAX(ev.ts))
             FROM edges e
             LEFT JOIN events ev ON ev.source = e.source AND ev.target = e.target
                  AND ev.event_type IN ('reinforced', 'injected', 'created')
             WHERE e.expired_at IS NULL
             GROUP BY e.source, e.target, e.embedding",
            (),
        )
        .await?;
    let mut signals = Vec::new();
    while let Some(row) = rows.next().await? {
        let vector_text = db::text(row.get_value(2)?);
        let embedding = db::parse_vector32_json(&vector_text).ok();
        let days_since_touch = match row.get_value(3)? {
            turso::Value::Real(value) => Some(value.max(0.0)),
            turso::Value::Integer(value) => Some((value as f64).max(0.0)),
            _ => None,
        };
        signals.push(ImportanceSignal {
            source: db::text(row.get_value(0)?),
            target: db::text(row.get_value(1)?),
            embedding,
            days_since_touch,
        });
    }
    Ok(signals)
}

fn max_alignment(edge: Option<&[f64]>, interests: &[Vec<f64>]) -> f64 {
    let Some(edge_embedding) = edge else {
        return 0.0;
    };
    let mut alignment: f64 = 0.0;
    for interest in interests {
        alignment = alignment.max(cosine_similarity(edge_embedding, interest).max(0.0));
    }
    alignment.min(1.0)
}

fn neighbourhood_density(signal: &ImportanceSignal, signals: &[ImportanceSignal]) -> f64 {
    let Some(edge_embedding) = signal.embedding.as_deref() else {
        return 0.0;
    };
    let mut neighbours = 0usize;
    for candidate in signals {
        if candidate.source == signal.source && candidate.target == signal.target {
            continue;
        }
        let Some(candidate_embedding) = candidate.embedding.as_deref() else {
            continue;
        };
        let distance = 1.0 - cosine_similarity(edge_embedding, candidate_embedding);
        if distance <= db::INTEREST_NEIGHBOUR_DISTANCE {
            neighbours += 1;
        }
    }
    neighbours.min(DENSITY_SATURATION) as f64 / DENSITY_SATURATION as f64
}

fn cosine_similarity(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
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
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    dot / denominator
}

fn recency_score(days_since_touch: f64) -> f64 {
    (-days_since_touch.max(0.0) / RECENCY_HALF_LIFE_DAYS).exp()
}

fn importance_score(alignment: f64, recency: f64, density: f64) -> f64 {
    db::round2((W_INTEREST * alignment + W_RECENCY * recency + W_DENSITY * density).clamp(0.0, 1.0))
}

async fn collect_reports(
    conn: &Connection,
    report: &mut HygieneReport,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (count, examples) = find_orphan_nodes(conn).await?;
    report.orphan_node_count = count;
    report.orphan_node_examples = examples;
    report.duplicate_pairs = find_duplicate_pairs(conn).await?;
    report.episodic_without_session = count_sessionless_episodic(conn).await?;
    Ok(())
}

// An edge whose endpoint has no node row is unreachable by retrieval but still
// carries provenance, so it is expired rather than deleted.
async fn expire_orphan_edges(
    conn: &Connection,
    events: &mut Vec<Event>,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target FROM edges WHERE expired_at IS NULL AND (source NOT IN (SELECT id FROM nodes) OR target NOT IN (SELECT id FROM nodes))",
            (),
        )
        .await?;
    let mut orphans: Vec<(String, String)> = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = db::text(row.get_value(0)?);
        let target = db::text(row.get_value(1)?);
        orphans.push((source, target));
    }
    let now = db::timestamp_now();
    for (source, target) in &orphans {
        conn.execute(
            "UPDATE edges SET expired_at = ?1 WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
            params![now.as_str(), source.as_str(), target.as_str()],
        )
        .await?;
        events.push(expiry_event(source, target));
    }
    Ok(orphans.len())
}

// Matches what expire_edge's apply path records — same event type, same payload.
fn expiry_event(source: &str, target: &str) -> Event {
    Event::updated(
        Actor::new("system", "hygiene"),
        EdgeRef::new(source, target),
        EventContext::new("", ""),
        serde_json::json!({"expired": true}).to_string(),
    )
}

async fn clamp_edge_weights(conn: &Connection) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let clamped = load_clamped_weights(conn).await?;
    for (source, target, confidence, stability, importance) in &clamped {
        conn.execute(
            "UPDATE edges SET confidence = ?1, stability = ?2, importance = ?3 WHERE source = ?4 AND target = ?5",
            params![
                db::round2(*confidence),
                db::round2(*stability),
                db::round2(*importance),
                source.as_str(),
                target.as_str()
            ],
        )
        .await?;
    }
    Ok(clamped.len())
}

async fn load_clamped_weights(
    conn: &Connection,
) -> Result<Vec<(String, String, f64, f64, f64)>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, confidence, stability, importance, provenance, ceiling
             FROM edges
             WHERE confidence < 0 OR confidence > 1
                OR confidence > COALESCE(ceiling, CASE provenance
                       WHEN 'user-confirmed' THEN 1.0
                       WHEN 'web-confirmed' THEN 0.8
                       WHEN 'dreamed' THEN 0.25
                       WHEN 'inferred' THEN 0.25
                       WHEN 'synthesized' THEN 0.25
                       ELSE 0.5
                   END)
                OR stability < 0 OR stability > 1
                OR importance < 0 OR importance > 1",
            (),
        )
        .await?;
    let mut clamped: Vec<(String, String, f64, f64, f64)> = Vec::new();
    while let Some(row) = rows.next().await? {
        let provenance = db::text(row.get_value(5)?);
        let ceiling = match row.get_value(6)? {
            turso::Value::Real(value) => Some(value),
            turso::Value::Integer(value) => Some(value as f64),
            _ => None,
        };
        clamped.push((
            db::text(row.get_value(0)?),
            db::text(row.get_value(1)?),
            clamp_unit(db::real(row.get_value(2)?)).min(edge_ceiling(&provenance, ceiling)),
            clamp_unit(db::real(row.get_value(3)?)),
            clamp_unit(db::real(row.get_value(4)?)),
        ));
    }
    Ok(clamped)
}

fn clamp_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

// A fact node needs exactly one subject edge. When it has none but exactly one
// live outgoing edge to an entity, the subject is unambiguous. Any other shape
// is a judgment call and only gets reported.
async fn fix_subject_edges(
    conn: &Connection,
    report: &mut HygieneReport,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for fact in facts_without_subject(conn).await? {
        let candidates = entity_targets(conn, &fact).await?;
        if candidates.len() != 1 {
            report.ambiguous_subject_facts.push(fact);
            continue;
        }
        let Some(target) = candidates.first() else {
            continue;
        };
        conn.execute(
            "UPDATE edges SET subject = 1, importance = 1.0 WHERE source = ?1 AND target = ?2",
            params![fact.as_str(), target.as_str()],
        )
        .await?;
        report.fixed_subject_edges += 1;
    }
    Ok(())
}

async fn facts_without_subject(
    conn: &Connection,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id FROM nodes WHERE node_type = 'fact' AND id NOT IN (SELECT source FROM edges WHERE subject = 1 AND expired_at IS NULL)",
            (),
        )
        .await?;
    let mut facts = Vec::new();
    while let Some(row) = rows.next().await? {
        facts.push(db::text(row.get_value(0)?));
    }
    Ok(facts)
}

async fn entity_targets(
    conn: &Connection,
    fact: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT ed.target FROM edges ed JOIN nodes nd ON nd.id = ed.target WHERE ed.source = ?1 AND ed.expired_at IS NULL AND nd.node_type = 'entity'",
            params![fact],
        )
        .await?;
    let mut targets = Vec::new();
    while let Some(row) = rows.next().await? {
        targets.push(db::text(row.get_value(0)?));
    }
    Ok(targets)
}

// Edges are the retrieval unit, so a node with no live edge is invisible —
// silent knowledge loss worth surfacing. Notes are standalone by design.
async fn find_orphan_nodes(
    conn: &Connection,
) -> Result<(usize, Vec<String>), Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id FROM nodes WHERE node_type NOT IN ('note', 'interest') AND id NOT IN (SELECT source FROM edges WHERE expired_at IS NULL UNION SELECT target FROM edges WHERE expired_at IS NULL)",
            (),
        )
        .await?;
    let mut count = 0usize;
    let mut examples = Vec::new();
    while let Some(row) = rows.next().await? {
        count += 1;
        if examples.len() < MAX_REPORTED_EXAMPLES {
            examples.push(db::text(row.get_value(0)?));
        }
    }
    Ok((count, examples))
}

// Node ids are normalized on write (lowercased, spaces hyphenated), so casing
// duplicates cannot exist. Separator variants can: tinyllama-1.1b vs
// tinyllama-1-1b. Merging needs a canonical choice, so only report the pairs.
async fn find_duplicate_pairs(
    conn: &Connection,
) -> Result<Vec<(String, String)>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn.query("SELECT id FROM nodes", ()).await?;
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(row) = rows.next().await? {
        let node_id = db::text(row.get_value(0)?);
        groups
            .entry(collapse_separators(&node_id))
            .or_default()
            .push(node_id);
    }
    let mut pairs = Vec::new();
    for (_key, mut ids) in groups {
        if ids.len() < 2 {
            continue;
        }
        ids.sort();
        for left in 0..ids.len() {
            for right in left + 1..ids.len() {
                pairs.push((ids[left].clone(), ids[right].clone()));
            }
        }
    }
    pairs.sort();
    Ok(pairs)
}

fn collapse_separators(node_id: &str) -> String {
    let mut collapsed = String::new();
    for character in node_id.chars() {
        if character == '-' || character == '_' || character == '.' {
            continue;
        }
        collapsed.push(character);
    }
    collapsed
}

// The originating session cannot be reconstructed after the fact, so this is a
// count only — it tells us whether the episodic write path is still leaking.
async fn count_sessionless_episodic(
    conn: &Connection,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM edges WHERE memory_type = 'episodic' AND origin_session = '' AND expired_at IS NULL",
            (),
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    let count = db::int(row.get_value(0)?);
    Ok(usize::try_from(count).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::{
        HygieneReport, collapse_separators, importance_score, recency_score, run_hygiene,
        run_importance_pass,
    };
    use crate::memory::db;
    use turso::{Builder, Connection, params};

    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL DEFAULT '', embedding BLOB)";
    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    // record_events_in_tx also writes actor views and positions.
    const VIEWS_DDL: &str = "CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))";
    const POSITIONS_DDL: &str = "CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))";
    const HISTORY_DDL: &str = "CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)";

    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        for ddl in [
            NODES_DDL,
            EDGES_DDL,
            EVENTS_DDL,
            VIEWS_DDL,
            POSITIONS_DDL,
            HISTORY_DDL,
        ] {
            let _ = conn.execute(ddl, ()).await;
        }
        conn
    }

    async fn add_node(conn: &Connection, node_id: &str, node_type: &str) {
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type) VALUES (?1, ?2)",
                params![node_id, node_type],
            )
            .await;
    }

    async fn add_edge(conn: &Connection, source: &str, target: &str, expired_at: Option<&str>) {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, confidence, stability, importance, expired_at) VALUES (?1, ?2, 0.5, 0.5, 0.5, ?3)",
                params![source, target, expired_at],
            )
            .await;
    }

    async fn expired_at_of(conn: &Connection, source: &str, target: &str) -> Option<String> {
        let Ok(mut rows) = conn
            .query(
                "SELECT expired_at FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("expiry query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(value) = row.get_value(0) else {
            unreachable!("column reads")
        };
        match value {
            turso::Value::Text(text) => Some(text),
            _ => None,
        }
    }

    async fn edge_weights(conn: &Connection, source: &str, target: &str) -> (f64, f64, f64) {
        let Ok(mut rows) = conn
            .query(
                "SELECT confidence, stability, importance FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("weights query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(confidence) = row.get_value(0) else {
            unreachable!("column reads")
        };
        let Ok(stability) = row.get_value(1) else {
            unreachable!("column reads")
        };
        let Ok(importance) = row.get_value(2) else {
            unreachable!("column reads")
        };
        (
            db::real(confidence),
            db::real(stability),
            db::real(importance),
        )
    }

    async fn subject_of(conn: &Connection, source: &str, target: &str) -> (i64, f64) {
        let Ok(mut rows) = conn
            .query(
                "SELECT subject, importance FROM edges WHERE source = ?1 AND target = ?2",
                params![source, target],
            )
            .await
        else {
            unreachable!("subject query runs")
        };
        let Ok(Some(row)) = rows.next().await else {
            unreachable!("edge exists")
        };
        let Ok(subject) = row.get_value(0) else {
            unreachable!("column reads")
        };
        let Ok(importance) = row.get_value(1) else {
            unreachable!("column reads")
        };
        (db::int(subject), db::real(importance))
    }

    fn hygiene(conn: &Connection) -> HygieneReport {
        match run_hygiene(conn) {
            Ok(report) => report,
            Err(err) => panic!("hygiene runs: {err}"),
        }
    }

    #[test]
    fn importance_blend_and_recency_follow_the_declared_formula() {
        assert_eq!(importance_score(0.0, 0.0, 0.0), 0.0);
        assert_eq!(importance_score(1.0, 1.0, 1.0), 1.0);
        assert_eq!(importance_score(1.0, 0.0, 0.0), 0.4);
        assert!((recency_score(14.0) - (-1.0_f64).exp()).abs() < 1e-9);
        assert_eq!(importance_score(0.0, recency_score(0.0), 0.0), 0.4);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn importance_pass_combines_interest_recency_and_missing_embeddings() {
        let conn = db::block_on(harness());
        db::block_on(async {
            for node_id in [
                "interest-rust",
                "near",
                "near-target",
                "far",
                "far-target",
                "plain",
                "plain-target",
            ] {
                add_node(&conn, node_id, "entity").await;
            }
            let _ = conn
                .execute(
                    "UPDATE nodes SET node_type = 'interest', body = 'Rust', origin = 'user',
                         embedding = vector32('[1.0,0.0]') WHERE id = 'interest-rust'",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO edges (source, target, embedding)
                     VALUES ('near', 'near-target', vector32('[1.0,0.0]')),
                            ('far', 'far-target', vector32('[0.0,1.0]')),
                            ('plain', 'plain-target', NULL)",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO events (ts, event_type, source, target)
                     VALUES (datetime('now'), 'created', 'near', 'near-target'),
                            (datetime('now', '-60 days'), 'created', 'far', 'far-target'),
                            (datetime('now'), 'injected', 'plain', 'plain-target')",
                    (),
                )
                .await;
            let count = run_importance_pass(&conn).await.unwrap_or_default();
            assert_eq!(count, 3);
        });
        let near = db::block_on(edge_weights(&conn, "near", "near-target")).2;
        let far = db::block_on(edge_weights(&conn, "far", "far-target")).2;
        let plain = db::block_on(edge_weights(&conn, "plain", "plain-target")).2;
        assert!(near > plain);
        assert!(plain > far);
        assert_eq!(plain, 0.4);
        assert_eq!(near, db::round2(near));
        assert_eq!(far, db::round2(far));
    }

    #[test]
    fn orphan_edges_are_expired_and_intact_edges_are_left_alone() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        // Missing target node.
        db::block_on(add_edge(&conn, "alice", "ghost", None));
        // Missing source node.
        db::block_on(add_edge(&conn, "phantom", "berlin", None));
        // Already expired orphan — must not be written again.
        db::block_on(add_edge(&conn, "wraith", "berlin", Some("2020-01-01")));

        let report = hygiene(&conn);
        assert_eq!(report.expired_orphan_edges, 2);
        assert!(db::block_on(expired_at_of(&conn, "alice", "ghost")).is_some());
        assert!(db::block_on(expired_at_of(&conn, "phantom", "berlin")).is_some());
        assert!(db::block_on(expired_at_of(&conn, "alice", "berlin")).is_none());
        assert_eq!(
            db::block_on(expired_at_of(&conn, "wraith", "berlin")),
            Some("2020-01-01".to_string())
        );
    }

    #[test]
    fn live_loop_edge_survives_hygiene_pass() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "ask-about-backup", "loop"));
        db::block_on(add_node(&conn, "backup", "entity"));
        db::block_on(add_edge(&conn, "ask-about-backup", "backup", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9, stability = 0.9, importance = 1.0,
             provenance = 'user-confirmed' WHERE source = 'ask-about-backup'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.expired_orphan_edges, 0);
        assert!(db::block_on(expired_at_of(&conn, "ask-about-backup", "backup")).is_none());
        assert_eq!(
            db::block_on(edge_weights(&conn, "ask-about-backup", "backup")),
            (0.9, 0.9, 0.0)
        );
    }

    #[test]
    fn out_of_range_weights_are_clamped_into_the_unit_interval() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_node(&conn, "carol", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        db::block_on(add_edge(&conn, "carol", "berlin", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 1.4, stability = -0.2 WHERE source = 'alice'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.clamped_edges, 1);
        assert_eq!(
            db::block_on(edge_weights(&conn, "alice", "berlin")),
            (0.5, 0.0, 0.0)
        );
        // Confidence and stability were already in range; salience is recomputed.
        assert_eq!(
            db::block_on(edge_weights(&conn, "carol", "berlin")),
            (0.5, 0.5, 0.0)
        );
    }

    #[test]
    fn derived_edge_is_clamped_to_its_own_ceiling() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "summary", "entity"));
        db::block_on(add_node(&conn, "topic", "entity"));
        db::block_on(add_edge(&conn, "summary", "topic", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET confidence = 0.9, provenance = 'synthesized', ceiling = 0.72
             WHERE source = 'summary'",
            (),
        ));

        let report = hygiene(&conn);
        assert_eq!(report.clamped_edges, 1);
        assert_eq!(
            db::block_on(edge_weights(&conn, "summary", "topic")).0,
            0.72
        );
    }

    #[test]
    fn unambiguous_missing_subject_edge_is_repaired() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 1);
        assert!(report.ambiguous_subject_facts.is_empty());
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (1, 0.0)
        );
    }

    #[test]
    fn fact_with_two_entity_candidates_is_reported_not_fixed() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));
        db::block_on(add_edge(&conn, "fact-moved", "berlin", None));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 0);
        assert_eq!(
            report.ambiguous_subject_facts,
            vec!["fact-moved".to_string()]
        );
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (0, 0.0)
        );
    }

    #[test]
    fn fact_that_already_has_a_subject_edge_is_untouched() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "fact-moved", "fact"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_edge(&conn, "fact-moved", "alice", None));
        let _ = db::block_on(conn.execute("UPDATE edges SET subject = 1", ()));

        let report = hygiene(&conn);
        assert_eq!(report.fixed_subject_edges, 0);
        assert!(report.ambiguous_subject_facts.is_empty());
        // Subject state stays untouched; nightly salience is still recomputed.
        assert_eq!(
            db::block_on(subject_of(&conn, "fact-moved", "alice")),
            (1, 0.0)
        );
    }

    #[test]
    fn separator_variants_are_reported_as_duplicate_pairs() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "tinyllama-1.1b", "entity"));
        db::block_on(add_node(&conn, "tinyllama-1-1b", "entity"));
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));

        let report = hygiene(&conn);
        assert_eq!(
            report.duplicate_pairs,
            vec![("tinyllama-1-1b".to_string(), "tinyllama-1.1b".to_string())]
        );
    }

    #[test]
    fn orphan_nodes_are_reported_and_notes_are_excluded() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        // Zero edges at all.
        db::block_on(add_node(&conn, "lonely", "entity"));
        // Only an expired edge — invisible to retrieval.
        db::block_on(add_node(&conn, "faded", "entity"));
        db::block_on(add_edge(&conn, "faded", "berlin", Some("2020-01-01")));
        db::block_on(add_node(&conn, "scratch", "note"));

        let report = hygiene(&conn);
        assert_eq!(report.orphan_node_count, 2);
        assert!(report.orphan_node_examples.contains(&"lonely".to_string()));
        assert!(report.orphan_node_examples.contains(&"faded".to_string()));
        assert!(!report.orphan_node_examples.contains(&"scratch".to_string()));
    }

    #[test]
    fn sessionless_episodic_edges_are_counted() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));
        let _ = db::block_on(conn.execute(
            "UPDATE edges SET memory_type = 'episodic', origin_session = ''",
            (),
        ));

        assert_eq!(hygiene(&conn).episodic_without_session, 1);
    }

    #[test]
    fn a_clean_graph_reports_nothing() {
        let conn = db::block_on(harness());
        db::block_on(add_node(&conn, "alice", "entity"));
        db::block_on(add_node(&conn, "berlin", "entity"));
        db::block_on(add_edge(&conn, "alice", "berlin", None));

        let report = hygiene(&conn);
        assert!(report.is_clean());
        assert!(report.summary().is_empty());
    }

    #[test]
    fn summary_names_each_non_zero_finding() {
        let report = HygieneReport {
            expired_orphan_edges: 195,
            clamped_edges: 1,
            ..HygieneReport::default()
        };
        assert_eq!(
            report.summary(),
            "expired 195 orphan edges, clamped 1 edge weight"
        );
    }

    #[test]
    fn separator_collapse_ignores_dots_hyphens_and_underscores() {
        assert_eq!(collapse_separators("tinyllama-1.1b"), "tinyllama11b");
        assert_eq!(collapse_separators("claude_pro"), "claudepro");
        assert_eq!(collapse_separators("alice"), "alice");
    }
}
