use std::collections::HashMap;
use std::error::Error;

use turso::{Builder, Connection, Value, params};

use super::{clamped_confidence, node_count, real, text};
use crate::memory::graph::{
    PROVENANCE_DREAMED, PROVENANCE_LEGACY, PROVENANCE_USER_CONFIRMED, PROVENANCE_WEB_SURFACED,
    clamp_stability_to_seed, edge_ceiling,
};
use crate::paths;

#[cfg(test)]
#[path = "migrations_tests.rs"]
mod tests;

pub(super) const STABILITY_CLAMP_MARKER: &str = "stability_clamp";

const MIGRATE_TABLES: [(&str, &str); 9] = [
    ("nodes", "id, node_type, body, date"),
    (
        "edges",
        "source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id, embedding, memory_type, origin_session",
    ),
    ("meta", "key, value"),
    ("sessions", "id, session_type, created_at"),
    ("messages", "id, session_id, role, content, created_at"),
    (
        "events",
        "id, ts, session_id, turn_id, actor_type, actor_id, event_type, source, target, weight, payload",
    ),
    (
        "edge_actor_views",
        "actor_type, actor_id, source, target, views, last_visited_at",
    ),
    (
        "actor_positions",
        "actor_type, actor_id, sum_vector, total_weight, updated_at",
    ),
    (
        "actor_position_history",
        "actor_type, actor_id, embedding, recorded_at",
    ),
];

pub(super) async fn migrate_legacy_if_empty(
    conn: &Connection,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if node_count(conn).await? > 0 {
        return Ok(());
    }
    let legacy_path = paths::data_file("kratt.db");
    if !std::path::Path::new(&legacy_path).exists() {
        return Ok(());
    }
    crate::runlog::log("sync", "migrating legacy kratt.db into synced database");
    let legacy = Builder::new_local(&legacy_path)
        .experimental_index_method(true)
        .build()
        .await?;
    let legacy_conn = legacy.connect()?;
    let mut total = 0usize;
    for (table, columns) in MIGRATE_TABLES {
        total += copy_table(&legacy_conn, conn, table, columns).await?;
    }
    crate::runlog::log("sync", &format!("legacy migration done — {total} rows"));
    Ok(())
}

async fn copy_table(
    source_conn: &Connection,
    target_conn: &Connection,
    table: &str,
    columns: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let column_count = columns.split(',').count();
    let mut placeholders = Vec::new();
    for index in 0..column_count {
        placeholders.push(format!("?{}", index + 1));
    }
    let insert_sql = format!(
        "INSERT OR IGNORE INTO {table} ({columns}) VALUES ({})",
        placeholders.join(", ")
    );
    let mut rows = source_conn
        .query(&format!("SELECT {columns} FROM {table}"), ())
        .await?;
    let mut copied = 0usize;
    while let Some(row) = rows.next().await? {
        let mut values: Vec<Value> = Vec::with_capacity(column_count);
        for index in 0..column_count {
            values.push(row.get_value(index)?);
        }
        target_conn
            .execute(&insert_sql, turso::params_from_iter(values))
            .await?;
        copied += 1;
    }
    Ok(copied)
}

pub(super) async fn add_interest_node_columns(conn: &Connection) {
    let _ = conn
        .execute(
            "ALTER TABLE nodes ADD COLUMN origin TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
    let _ = conn
        .execute("ALTER TABLE nodes ADD COLUMN embedding BLOB", ())
        .await;
}

async fn add_edge_attribution_column(conn: &Connection) {
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN source_message_id TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
}

async fn add_edge_embedding_column(conn: &Connection) {
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN embedding BLOB", ())
        .await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS edges_vec ON edges (libsql_vector_idx(embedding, 1024))",
            (),
        )
        .await;
}

pub(super) async fn add_edge_columns(conn: &Connection) {
    add_edge_attribution_column(conn).await;
    add_edge_embedding_column(conn).await;
    add_edge_time_columns(conn).await;
    add_edge_memory_type_columns(conn).await;
    add_edge_epistemics_column(conn).await;
    add_edge_decay_clock_column(conn).await;
    add_edge_ceiling_column(conn).await;
}

async fn add_edge_decay_clock_column(conn: &Connection) {
    // Decay charged each edge's full age on every run because nothing advanced
    // a per-edge clock — the exponent compounded as n(n+1)/14 instead of n/7.
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN last_decayed_at TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
}

async fn add_edge_epistemics_column(conn: &Connection) {
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN provenance TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
}

pub(super) async fn add_edge_ceiling_column(conn: &Connection) {
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN ceiling REAL", ())
        .await;
}

#[derive(Default)]
pub(super) struct EpistemicsBackfill {
    pub(super) total: usize,
    pub(super) dreamed: usize,
    pub(super) web: usize,
    pub(super) user: usize,
    pub(super) legacy: usize,
    pub(super) clamped: usize,
}

#[derive(Debug, PartialEq)]
struct StabilityClampProvenance {
    provenance: String,
    clamped: usize,
    median_before: f64,
    median_after: f64,
}

#[derive(Debug, PartialEq)]
pub(super) struct StabilityClampReport {
    pub(super) marker: String,
    pub(super) total: usize,
    pub(super) clamped: usize,
    pub(super) above_seed: usize,
    provenances: Vec<StabilityClampProvenance>,
}

struct StabilityClampRow {
    source: String,
    target: String,
    provenance: String,
    before: f64,
    after: f64,
    changed: bool,
}

pub(super) async fn backfill_edge_epistemics(
    conn: &Connection,
) -> Result<EpistemicsBackfill, Box<dyn Error + Send + Sync>> {
    let pending = load_unclassified_edges(conn).await?;
    let mut summary = EpistemicsBackfill::default();
    for edge in pending {
        backfill_one_edge(conn, edge, &mut summary).await?;
    }
    log_epistemics_backfill(&summary);
    Ok(summary)
}

pub(super) async fn clamp_stability_once(
    conn: &Connection,
    marker: &str,
) -> Result<Option<StabilityClampReport>, Box<dyn Error + Send + Sync>> {
    conn.execute("BEGIN", ()).await?;
    let outcome = clamp_stability_transaction(conn, marker).await;
    finish_stability_transaction(conn, outcome).await
}

async fn finish_stability_transaction(
    conn: &Connection,
    outcome: Result<Option<StabilityClampReport>, Box<dyn Error + Send + Sync>>,
) -> Result<Option<StabilityClampReport>, Box<dyn Error + Send + Sync>> {
    match outcome {
        Ok(report) => {
            if let Err(err) = conn.execute("COMMIT", ()).await {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(err.into());
            }
            if let Some(summary) = &report {
                log_stability_clamp(summary);
            }
            Ok(report)
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK", ()).await;
            Err(err)
        }
    }
}

async fn clamp_stability_transaction(
    conn: &Connection,
    marker: &str,
) -> Result<Option<StabilityClampReport>, Box<dyn Error + Send + Sync>> {
    if meta_value(conn, STABILITY_CLAMP_MARKER).await?.is_some() {
        return Ok(None);
    }
    let mut rows = load_stability_clamp_rows(conn).await?;
    apply_stability_clamp(conn, &mut rows).await?;
    let above_seed = count_edges_above_seed(conn).await?;
    if above_seed > 0 {
        return Err(format!("stability clamp left {above_seed} edges above seed").into());
    }
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)",
        params![STABILITY_CLAMP_MARKER, marker],
    )
    .await?;
    Ok(Some(stability_clamp_report(marker, &rows, above_seed)))
}

pub(super) async fn meta_value(
    conn: &Connection,
    key: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query("SELECT value FROM meta WHERE key = ?1", params![key])
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

async fn load_stability_clamp_rows(
    conn: &Connection,
) -> Result<Vec<StabilityClampRow>, Box<dyn Error + Send + Sync>> {
    let mut query = conn
        .query(
            "SELECT source, target, provenance, stability FROM edges ORDER BY source, target",
            (),
        )
        .await?;
    let mut rows = Vec::new();
    while let Some(row) = query.next().await? {
        let before = real(row.get_value(3)?);
        let provenance = text(row.get_value(2)?);
        rows.push(StabilityClampRow {
            source: text(row.get_value(0)?),
            target: text(row.get_value(1)?),
            after: clamp_stability_to_seed(before, &provenance),
            provenance,
            before,
            changed: false,
        });
    }
    Ok(rows)
}

async fn apply_stability_clamp(
    conn: &Connection,
    rows: &mut [StabilityClampRow],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for row in rows {
        if row.after >= row.before {
            continue;
        }
        let changed = conn
            .execute(
                "UPDATE edges SET stability = ?1 WHERE source = ?2 AND target = ?3",
                params![row.after, row.source.as_str(), row.target.as_str()],
            )
            .await?;
        row.changed = changed > 0;
    }
    Ok(())
}

async fn count_edges_above_seed(conn: &Connection) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let rows = load_stability_clamp_rows(conn).await?;
    let mut count = 0usize;
    for row in rows {
        if row.before > row.after {
            count += 1;
        }
    }
    Ok(count)
}

fn stability_clamp_report(
    marker: &str,
    rows: &[StabilityClampRow],
    above_seed: usize,
) -> StabilityClampReport {
    let mut groups: HashMap<String, (usize, Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut clamped = 0usize;
    for row in rows {
        let group = groups
            .entry(row.provenance.clone())
            .or_insert_with(|| (0, Vec::new(), Vec::new()));
        group.1.push(row.before);
        group.2.push(row.after);
        if row.changed {
            group.0 += 1;
            clamped += 1;
        }
    }
    StabilityClampReport {
        marker: marker.to_string(),
        total: rows.len(),
        clamped,
        above_seed,
        provenances: stability_provenance_summaries(groups),
    }
}

fn stability_provenance_summaries(
    groups: HashMap<String, (usize, Vec<f64>, Vec<f64>)>,
) -> Vec<StabilityClampProvenance> {
    let mut summaries = Vec::new();
    for (provenance, (clamped, mut before, mut after)) in groups {
        summaries.push(StabilityClampProvenance {
            provenance,
            clamped,
            median_before: median(&mut before),
            median_after: median(&mut after),
        });
    }
    summaries.sort_by(|left, right| left.provenance.cmp(&right.provenance));
    summaries
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    values[middle]
}

pub(super) fn stability_clamp_log_lines(report: &StabilityClampReport) -> Vec<String> {
    let mut lines = vec![format!(
        "[stability] clamped {} of {} edges (marker {})",
        report.clamped, report.total, report.marker
    )];
    for summary in &report.provenances {
        lines.push(format!(
            "[stability] {}: {} clamped, median {:.2} -> {:.2}",
            summary.provenance, summary.clamped, summary.median_before, summary.median_after
        ));
    }
    lines.push(format!(
        "[stability] edges above seed after clamp: {}",
        report.above_seed
    ));
    lines
}

fn log_stability_clamp(report: &StabilityClampReport) {
    for line in stability_clamp_log_lines(report) {
        crate::runlog::log("stability", &line);
    }
    // Baseline from simulation-2026-07-20: this clamp plus the shipped curves
    // predicts rust/immune median 0.128 and user-confirmed median 0.749 at 90
    // nights, with junk prune-eligible by 30 nights.
    crate::runlog::log(
        "stability",
        "[stability] prediction: at 90 nights rust/immune median 0.128, \
         user-confirmed median 0.749; junk prune-eligible by 30 nights",
    );
}

pub(super) async fn migrate_entity_bodies_matching_ids(
    conn: &Connection,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let predicate = "node_type = 'entity' AND body != ''
        AND lower(trim(replace(replace(body, '-', ' '), '_', ' ')))
          = lower(trim(replace(replace(id, '-', ' '), '_', ' ')))";
    let mut rows = conn
        .query(
            &format!("SELECT id FROM nodes WHERE {predicate} ORDER BY id"),
            (),
        )
        .await?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().await? {
        ids.push(text(row.get_value(0)?));
    }
    conn.execute(&format!("UPDATE nodes SET body = '' WHERE {predicate}"), ())
        .await?;
    log_entity_body_migration(&ids);
    Ok(ids)
}

fn log_entity_body_migration(ids: &[String]) {
    if ids.is_empty() {
        return;
    }
    for node_id in ids {
        crate::runlog::log("nodes", &format!("migrated body≈id: {node_id}"));
    }
    crate::runlog::log("nodes", &format!("migrated body≈id: {} nodes", ids.len()));
}

async fn load_unclassified_edges(
    conn: &Connection,
) -> Result<Vec<(String, String, String, f64)>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, origin_session, confidence FROM edges WHERE provenance = ''",
            (),
        )
        .await?;
    let mut pending = Vec::new();
    while let Some(row) = rows.next().await? {
        pending.push((
            text(row.get_value(0)?),
            text(row.get_value(1)?),
            text(row.get_value(2)?),
            real(row.get_value(3)?),
        ));
    }
    Ok(pending)
}

async fn backfill_one_edge(
    conn: &Connection,
    edge: (String, String, String, f64),
    summary: &mut EpistemicsBackfill,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (source, target, origin_session, confidence) = edge;
    let provenance = provenance_from_origin_session(&origin_session);
    let clamped = clamped_confidence(confidence, provenance, None);
    conn.execute(
        "UPDATE edges SET provenance = ?1, confidence = ?2
         WHERE source = ?3 AND target = ?4 AND provenance = ''",
        params![provenance, clamped, source, target],
    )
    .await?;
    count_backfill(summary, provenance, confidence);
    Ok(())
}

fn log_epistemics_backfill(summary: &EpistemicsBackfill) {
    if summary.total > 0 {
        crate::runlog::log(
            "epistemics",
            &format!(
                "[epistemics] backfilled {} edges ({} dreamed, {} web, {} user, {} legacy), clamped {}",
                summary.total,
                summary.dreamed,
                summary.web,
                summary.user,
                summary.legacy,
                summary.clamped
            ),
        );
    }
}

fn count_backfill(summary: &mut EpistemicsBackfill, provenance: &str, confidence: f64) {
    summary.total += 1;
    match provenance {
        PROVENANCE_DREAMED => summary.dreamed += 1,
        PROVENANCE_WEB_SURFACED => summary.web += 1,
        PROVENANCE_USER_CONFIRMED => summary.user += 1,
        _ => summary.legacy += 1,
    }
    if confidence > edge_ceiling(provenance, None) {
        summary.clamped += 1;
    }
}

fn provenance_from_origin_session(origin_session: &str) -> &'static str {
    // Observed/current shapes: dream-<epoch> (older notes also used dream:*),
    // explore-<epoch>, ingest-<epoch>, news:<date>, tg-<date>, and TUI
    // timestamps like 2026-07-07T19-26-14. Blank or unknown stays legacy.
    if origin_session.starts_with("dream-") || origin_session.starts_with("dream:") {
        return PROVENANCE_DREAMED;
    }
    if origin_session.starts_with("news:")
        || origin_session.starts_with("explore-")
        || origin_session.starts_with("explore:")
        || origin_session.starts_with("ingest-")
        || origin_session.starts_with("ingest:")
    {
        return PROVENANCE_WEB_SURFACED;
    }
    if origin_session.starts_with("tg-") || timestamp_session_id(origin_session) {
        return PROVENANCE_USER_CONFIRMED;
    }
    PROVENANCE_LEGACY
}

fn timestamp_session_id(origin_session: &str) -> bool {
    let bytes = origin_session.as_bytes();
    bytes.len() >= 10
        && ascii_digits(bytes, 0, 4)
        && bytes[4] == b'-'
        && ascii_digits(bytes, 5, 7)
        && bytes[7] == b'-'
        && ascii_digits(bytes, 8, 10)
}

fn ascii_digits(bytes: &[u8], start: usize, end: usize) -> bool {
    let mut index = start;
    while index < end {
        if !bytes[index].is_ascii_digit() {
            return false;
        }
        index += 1;
    }
    true
}

async fn add_edge_memory_type_columns(conn: &Connection) {
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'semantic'",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN origin_session TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
}

async fn add_edge_time_columns(conn: &Connection) {
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN occurred_at TEXT", ())
        .await;
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN expired_at TEXT", ())
        .await;
}

pub(super) async fn drop_node_agent_views_table(conn: &Connection) {
    let _ = conn
        .execute("DROP TABLE IF EXISTS node_agent_views", ())
        .await;
}
