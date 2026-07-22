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
#[path = "hygiene_tests.rs"]
mod tests;
