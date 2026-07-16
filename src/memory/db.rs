use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tokio::runtime::Runtime;
use turso::{Builder, Connection, Database, Value, params};

use super::graph::{
    Edge, Graph, Node, PROVENANCE_DREAMED, PROVENANCE_LEGACY, PROVENANCE_USER_CONFIRMED,
    PROVENANCE_WEB_SURFACED, edge_ceiling, load_from_toon, normalize_id, stability_seed_max,
};
use super::loops::OpenLoop;

use crate::paths;

pub type SemanticEdgeRow = (
    String,
    String,
    String,
    f64,
    f64,
    f64,
    bool,
    String,
    Option<String>,
    Option<String>,
    String,
    String,
    String,
    Option<Vec<f64>>,
);

pub type OpenCuriosityNoteRow = (String, String, String, String);

pub(crate) const INTEREST_NEIGHBOUR_DISTANCE: f64 = 0.25; // replay-tunable.
pub(crate) const DEFAULT_PING_MUTED_UNTIL: &str = "9999-12-31T23:59:59";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Interest {
    pub node_id: String,
    pub topic: String,
    pub origin: String,
    pub created_at: String,
    pub strength: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct InterestTarget {
    pub topic: String,
    pub density: usize,
    pub embedding: Vec<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecentLearning {
    pub content: String,
    pub source: String,
    pub target: String,
    pub event_type: String,
    pub timestamp: String,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NearestLiveEdge {
    pub source: String,
    pub target: String,
    pub provenance: String,
    pub confidence: f64,
    pub similarity: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct RegionTransition {
    pub target: String,
    pub timestamp: String,
    pub signal: String,
}

#[derive(Clone, Debug)]
pub(crate) struct SequenceSuggestion {
    pub turn_id: String,
    pub source: String,
    pub target: String,
}

#[allow(clippy::expect_used)]
fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

pub(crate) fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    runtime().block_on(future)
}

// Synced mode: when TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set, the app
// runs on a Turso-Cloud-synced local file (separate from the legacy plain
// file so the old database stays intact as a fallback). On first synced run
// with an empty synced database, legacy data is copied over automatically.
enum DbHandle {
    Local(Database),
    Synced(turso::sync::Database),
}

static DATABASE: OnceLock<DbHandle> = OnceLock::new();

fn sync_credentials() -> Option<(String, String)> {
    // Dev isolation, safe by default: a custom KRATT_DATA_DIR never syncs
    // against the cloud primary — one shared TURSO_DATABASE_URL in .env must
    // not let a dev run write into the live database. KRATT_SYNC=1 opts in.
    if paths::data_dir() != "kratt-data" && std::env::var("KRATT_SYNC").is_err() {
        return None;
    }
    let url = std::env::var("TURSO_DATABASE_URL").ok()?;
    let token = std::env::var("TURSO_AUTH_TOKEN").ok()?;
    if url.trim().is_empty() || token.trim().is_empty() {
        return None;
    }
    Some((url, token))
}

#[allow(clippy::expect_used)]
fn database() -> &'static DbHandle {
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors
    DATABASE.get_or_init(|| {
        runtime().block_on(async {
            std::fs::create_dir_all(paths::data_dir()).expect("create data dir");
            match sync_credentials() {
                Some((url, token)) => DbHandle::Synced(open_synced(&url, &token).await),
                None => DbHandle::Local(open_local().await),
            }
        })
    })
}

pub fn startup_database() -> Result<(), Box<dyn Error + Send + Sync>> {
    std::fs::create_dir_all(paths::data_dir())?;
    let Some((url, token)) = sync_credentials() else {
        let _ = database();
        return Ok(());
    };
    if DATABASE.get().is_some() {
        return Ok(());
    }
    let replica_path = PathBuf::from(paths::data_file("kratt-synced.db"));
    let today = crate::timeutil::today_iso();
    let handle =
        recover_replica_once(
            &replica_path,
            &today,
            |recovering| match open_and_probe_synced(&url, &token, recovering) {
                Ok(handle) => Ok(handle),
                Err(err) => Err(err.to_string()),
            },
        )?;
    match DATABASE.set(handle) {
        Ok(()) => Ok(()),
        Err(_) => Err("database initialized concurrently during startup".into()),
    }
}

fn open_and_probe_synced(
    url: &str,
    token: &str,
    recovering: bool,
) -> Result<DbHandle, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let database = open_synced_result(url, token, !recovering).await?;
        let conn = database.connect().await?;
        let _ = node_count(&conn).await?;
        if recovering {
            let changed = database.pull().await?;
            let _ = node_count(&conn).await?;
            record_startup_pull_success(changed);
        } else {
            startup_pull_best_effort(&database, &conn).await?;
        }
        Ok(DbHandle::Synced(database))
    })
}

async fn startup_pull_best_effort(
    database: &turso::sync::Database,
    conn: &Connection,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match database.pull().await {
        Ok(changed) => {
            let _ = node_count(conn).await?;
            record_startup_pull_success(changed);
        }
        Err(err) => {
            let _ = node_count(conn).await?;
            crate::runlog::log("sync", &format!("pull failed: {err}"));
            record_sync_outcome(Some("pull failed".to_string()));
        }
    }
    Ok(())
}

fn record_startup_pull_success(changed: bool) {
    crate::runlog::log("sync", &format!("pull ok — changed: {changed}"));
    record_sync_outcome(None);
}

fn recover_replica_once<Output>(
    replica_path: &Path,
    today: &str,
    mut probe: impl FnMut(bool) -> Result<Output, String>,
) -> Result<Output, Box<dyn Error + Send + Sync>> {
    let first_failure = match run_startup_probe(&mut probe, false) {
        Ok(output) => return Ok(output),
        Err(failure) => failure,
    };
    let message = "replica unusable — re-pulling from cloud";
    eprintln!("[recover] {message}");
    crate::runlog::log("recover", message);
    move_replica_aside(replica_path, today)?;
    match run_startup_probe(&mut probe, true) {
        Ok(output) => Ok(output),
        Err(failure) => Err(format!(
            "replica recovery failed after one retry: {failure}; initial failure: {first_failure}"
        )
        .into()),
    }
}

fn run_startup_probe<Output>(
    probe: &mut impl FnMut(bool) -> Result<Output, String>,
    recovering: bool,
) -> Result<Output, String> {
    match catch_unwind(AssertUnwindSafe(|| probe(recovering))) {
        Ok(result) => result,
        Err(_) => Err("startup database probe panicked".to_string()),
    }
}

fn move_replica_aside(
    replica_path: &Path,
    today: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(parent) = replica_path.parent() else {
        return Err("synced replica path has no parent".into());
    };
    let Some(file_name) = replica_path.file_name() else {
        return Err("synced replica path has no filename".into());
    };
    let Some(replica_name) = file_name.to_str() else {
        return Err("synced replica path is not valid UTF-8".into());
    };
    let backup_name = available_backup_name(parent, replica_name, today);
    let mut sources = Vec::new();
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name_text) = name.to_str() else {
            continue;
        };
        if replica_file_matches(name_text, replica_name) {
            sources.push((entry.path(), name_text.to_string()));
        }
    }
    for (source, name) in sources {
        let suffix = &name[replica_name.len()..];
        let destination = parent.join(format!("{backup_name}{suffix}"));
        std::fs::rename(source, destination)?;
    }
    Ok(())
}

fn replica_file_matches(candidate: &str, replica_name: &str) -> bool {
    if candidate == replica_name {
        return true;
    }
    candidate.starts_with(replica_name) && !candidate[replica_name.len()..].starts_with(".corrupt-")
}

fn available_backup_name(parent: &Path, replica_name: &str, today: &str) -> String {
    let first = format!("{replica_name}.corrupt-{today}");
    if !parent.join(&first).exists() {
        return first;
    }
    let mut sequence = 2usize;
    loop {
        let candidate = format!("{replica_name}.corrupt-{today}-{sequence}");
        if !parent.join(&candidate).exists() {
            return candidate;
        }
        sequence += 1;
    }
}

#[allow(clippy::expect_used)]
async fn open_local() -> Database {
    let database = Builder::new_local(&paths::data_file("kratt.db"))
        .experimental_index_method(true)
        .build()
        .await
        .expect("open db");
    let conn = database.connect().expect("db connect");
    init_schema(&conn, true).await.expect("init schema");
    database
}

#[allow(clippy::expect_used)]
async fn open_synced(url: &str, token: &str) -> turso::sync::Database {
    open_synced_result(url, token, true)
        .await
        .expect("open synced db")
}

async fn open_synced_result(
    url: &str,
    token: &str,
    migrate_legacy: bool,
) -> Result<turso::sync::Database, Box<dyn Error + Send + Sync>> {
    let database = turso::sync::Builder::new_remote(&paths::data_file("kratt-synced.db"))
        .with_remote_url(url)
        .with_auth_token(token)
        .experimental_index_method(true)
        .build()
        .await?;
    let conn = database.connect().await?;
    init_schema(&conn, false).await?;
    if migrate_legacy && let Err(err) = migrate_legacy_if_empty(&conn).await {
        return Err(format!("legacy migration failed: {err}").into());
    }
    let _ = backfill_edge_epistemics(&conn).await?;
    Ok(database)
}

pub fn open_connection() -> Result<Connection, Box<dyn Error + Send + Sync>> {
    match database() {
        DbHandle::Local(database) => Ok(database.connect()?),
        DbHandle::Synced(database) => Ok(runtime().block_on(database.connect())?),
    }
}

// Push local changes to Turso Cloud. No-op in local mode. Best-effort at all
// call sites — sync must never break the app. Checkpoints afterwards to keep
// the local WAL bounded (it grows quickly with embedding blobs).
pub fn sync_push() -> Result<(), Box<dyn Error + Send + Sync>> {
    with_write_lock(|| match database() {
        DbHandle::Local(_) => Ok(()),
        DbHandle::Synced(database) => {
            let _timer = crate::runlog::time_step("sync", "push and checkpoint");
            runtime().block_on(database.push())?;
            if let Err(err) = runtime().block_on(database.checkpoint()) {
                crate::runlog::log("sync", &format!("checkpoint failed: {err}"));
            }
            Ok(())
        }
    })
}

// Pull remote changes from Turso Cloud; returns whether anything changed.
pub fn sync_pull() -> Result<bool, Box<dyn Error + Send + Sync>> {
    with_write_lock(|| match database() {
        DbHandle::Local(_) => Ok(false),
        DbHandle::Synced(database) => {
            let _timer = crate::runlog::time_step("sync", "pull");
            Ok(runtime().block_on(database.pull())?)
        }
    })
}

pub fn sync_enabled() -> bool {
    matches!(database(), DbHandle::Synced(_))
}

// Last sync failure, for the TUI status line. Background passes push from
// their own threads, so this cell is the handoff point to the renderer.
// Set on failure, cleared by the next success.
fn sync_error_cell() -> &'static std::sync::Mutex<Option<String>> {
    static SYNC_ERROR: OnceLock<std::sync::Mutex<Option<String>>> = OnceLock::new();
    SYNC_ERROR.get_or_init(|| std::sync::Mutex::new(None))
}

fn record_sync_outcome(error: Option<String>) {
    if let Ok(mut cell) = sync_error_cell().lock() {
        *cell = error;
    }
}

pub fn last_sync_error() -> Option<String> {
    match sync_error_cell().lock() {
        Ok(cell) => cell.clone(),
        Err(_) => None,
    }
}

// Best-effort push for the end of background write passes and app exit.
pub fn sync_push_best_effort(context: &str) {
    if !sync_enabled() {
        return;
    }
    match sync_push() {
        Ok(()) => {
            crate::runlog::log("sync", &format!("push ok ({context})"));
            record_sync_outcome(None);
        }
        Err(err) => {
            crate::runlog::log("sync", &format!("push failed ({context}): {err}"));
            record_sync_outcome(Some(format!("push failed ({context})")));
        }
    }
}

// Manual full sync (TUI 'S' key): pull remote changes, push local ones.
// Returns a human-readable summary line.
pub fn sync_now() -> String {
    if !sync_enabled() {
        return "sync disabled (no Turso Cloud credentials)".to_string();
    }
    let mut failure = None;
    let pull_summary = match sync_pull() {
        Ok(changed) => format!("pull ok (changed: {changed})"),
        Err(err) => {
            failure = Some("pull failed (manual sync)".to_string());
            format!("pull failed: {err}")
        }
    };
    let push_summary = match sync_push() {
        Ok(()) => "push ok".to_string(),
        Err(err) => {
            failure = Some("push failed (manual sync)".to_string());
            format!("push failed: {err}")
        }
    };
    record_sync_outcome(failure);
    let summary = format!("{pull_summary} · {push_summary}");
    crate::runlog::log("sync", &summary);
    summary
}

// Explicit column lists: the legacy file gained columns via ALTERs in a
// different order than a fresh schema creates them, so SELECT * would
// scramble the mapping.
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

async fn migrate_legacy_if_empty(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
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

// Turso sync and checkpoint operations mutate database-wide state and can
// dissolve a transaction on another connection. Every write pass and every
// sync operation must serialize on this lock.
static WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn acquire_write_lock() -> std::sync::MutexGuard<'static, ()> {
    match WRITE_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn with_write_lock<Output>(operation: impl FnOnce() -> Output) -> Output {
    let _write_guard = acquire_write_lock();
    operation()
}

async fn init_schema(
    conn: &Connection,
    create_fts_index: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    create_base_tables(conn).await?;
    create_history_tables(conn, create_fts_index).await?;
    add_interest_node_columns(conn).await;
    add_edge_columns(conn).await;
    let _ = backfill_edge_epistemics(conn).await?;
    add_ledger_tables(conn).await;
    drop_node_agent_views_table(conn).await;
    Ok(())
}

async fn add_interest_node_columns(conn: &Connection) {
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

async fn create_base_tables(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS inbox (id INTEGER PRIMARY KEY AUTOINCREMENT, accepted_at TEXT NOT NULL, source TEXT NOT NULL, instruction TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'queued', resolved_at TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    conn.execute(
        "INSERT OR IGNORE INTO ping_policy (topic, muted_until) VALUES ('', ?1)",
        params![DEFAULT_PING_MUTED_UNTIL],
    )
    .await?;
    Ok(())
}

async fn create_history_tables(
    conn: &Connection,
    create_fts_index: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    // The FTS index is local-mode only. The cloud server cannot parse
    // `CREATE INDEX … USING fts` DDL: a push carrying it fails, and a pull
    // can clobber the local index record, restarting the cycle on every
    // machine. In synced mode message search falls back to LIKE.
    if !create_fts_index {
        return Ok(());
    }
    if !fts_index_exists(conn).await {
        let _ = conn
            .execute(
                "CREATE INDEX fts_messages_content ON messages USING fts (content)",
                (),
            )
            .await;
    }
    Ok(())
}

async fn fts_index_exists(conn: &Connection) -> bool {
    let Ok(mut rows) = conn
        .query(
            "SELECT name FROM sqlite_schema WHERE name = 'fts_messages_content' LIMIT 1",
            (),
        )
        .await
    else {
        return false;
    };
    matches!(rows.next().await, Ok(Some(_)))
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

async fn add_edge_columns(conn: &Connection) {
    add_edge_attribution_column(conn).await;
    add_edge_embedding_column(conn).await;
    add_edge_time_columns(conn).await;
    add_edge_memory_type_columns(conn).await;
    add_edge_epistemics_column(conn).await;
    add_edge_ceiling_column(conn).await;
}

async fn add_edge_epistemics_column(conn: &Connection) {
    let _ = conn
        .execute(
            "ALTER TABLE edges ADD COLUMN provenance TEXT NOT NULL DEFAULT ''",
            (),
        )
        .await;
}

async fn add_edge_ceiling_column(conn: &Connection) {
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN ceiling REAL", ())
        .await;
}

#[derive(Default)]
struct EpistemicsBackfill {
    total: usize,
    dreamed: usize,
    web: usize,
    user: usize,
    legacy: usize,
    clamped: usize,
}

async fn backfill_edge_epistemics(
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

async fn add_ledger_tables(conn: &Connection) {
    add_events_table(conn).await;
    add_edge_actor_views_table(conn).await;
    add_actor_positions_table(conn).await;
    add_actor_position_history_table(conn).await;
}

async fn add_edge_time_columns(conn: &Connection) {
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN occurred_at TEXT", ())
        .await;
    let _ = conn
        .execute("ALTER TABLE edges ADD COLUMN expired_at TEXT", ())
        .await;
}

async fn add_events_table(conn: &Connection) {
    // event_type values: created, updated, pruned, injected, retrieved, marked, routed
    let _ = conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')", ()).await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS events_actor_idx ON events (actor_type, actor_id, id)",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS events_turn_idx ON events (turn_id)",
            (),
        )
        .await;
    let _ = conn
        .execute(
            "CREATE INDEX IF NOT EXISTS events_session_idx ON events (session_id)",
            (),
        )
        .await;
}

async fn add_edge_actor_views_table(conn: &Connection) {
    let _ = conn.execute("CREATE TABLE IF NOT EXISTS edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))", ()).await;
}

async fn add_actor_positions_table(conn: &Connection) {
    let _ = conn.execute("CREATE TABLE IF NOT EXISTS actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))", ()).await;
}

async fn add_actor_position_history_table(conn: &Connection) {
    let _ = conn.execute("CREATE TABLE IF NOT EXISTS actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)", ()).await;
    let _ = conn.execute("CREATE INDEX IF NOT EXISTS actor_position_history_idx ON actor_position_history (actor_type, actor_id, recorded_at)", ()).await;
}

async fn drop_node_agent_views_table(conn: &Connection) {
    let _ = conn
        .execute("DROP TABLE IF EXISTS node_agent_views", ())
        .await;
}

pub fn vector32_json(vector: &[f64]) -> Result<String, Box<dyn Error + Send + Sync>> {
    Ok(serde_json::to_string(vector)?)
}

pub fn parse_vector32_json(json_text: &str) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let values = serde_json::from_str::<Vec<f64>>(json_text)?;
    Ok(values)
}

pub fn timestamp_now() -> String {
    crate::timeutil::timestamp_now()
}

#[derive(Debug, Clone, PartialEq)]
pub struct InboxItem {
    pub item_id: i64,
    pub source: String,
    pub instruction: String,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // Full queue rows retain metadata beyond the ping renderer's fields.
pub(crate) struct Finding {
    pub finding_id: i64,
    pub interest_topic: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub novelty: f64,
    pub found_at: String,
    pub status: String,
    pub surfaced_at: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PingPolicy {
    pub muted_until: String,
    pub diverts: Vec<(String, String)>,
}

pub fn insert_inbox_item(
    conn: &Connection,
    source: &str,
    instruction: &str,
) -> Result<i64, Box<dyn Error + Send + Sync>> {
    let accepted_at = timestamp_now();
    block_on(async {
        conn.execute(
            "INSERT INTO inbox (accepted_at, source, instruction) VALUES (?1, ?2, ?3)",
            params![accepted_at, source, instruction],
        )
        .await?;
        Ok(conn.last_insert_rowid())
    })
}

pub fn oldest_queued_inbox(
    conn: &Connection,
) -> Result<Option<InboxItem>, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, source, instruction FROM inbox WHERE status = 'queued' ORDER BY id LIMIT 1",
                (),
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        Ok(Some(InboxItem {
            item_id: row.get::<i64>(0)?,
            source: row.get::<String>(1)?,
            instruction: row.get::<String>(2)?,
        }))
    })
}

// Short display names (last path/URL segment) of queued items, oldest first,
// capped — feeds the chat-visible internal-state line.
pub async fn load_queued_inbox_names(conn: &Connection) -> Vec<String> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source FROM inbox WHERE status = 'queued' ORDER BY id LIMIT 3",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut names = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let source = row.get::<String>(0).unwrap_or_default();
        let name = match source.trim_end_matches('/').rsplit('/').next() {
            Some(tail) if !tail.is_empty() => tail.to_string(),
            _ => source,
        };
        names.push(name);
    }
    names
}

pub fn inbox_source_is_queued(
    conn: &Connection,
    source: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT 1 FROM inbox WHERE status = 'queued' AND source = ?1 LIMIT 1",
                params![source],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    })
}

pub fn resolve_inbox_item(
    conn: &Connection,
    item_id: i64,
    status: &str,
    result: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let resolved_at = timestamp_now();
    block_on(async {
        conn.execute(
            "UPDATE inbox SET status = ?1, resolved_at = ?2, result = ?3 WHERE id = ?4",
            params![status, resolved_at, result, item_id],
        )
        .await?;
        Ok(())
    })
}

pub async fn count_queued_inbox(conn: &Connection) -> usize {
    let Ok(mut rows) = conn
        .query("SELECT COUNT(*) FROM inbox WHERE status = 'queued'", ())
        .await
    else {
        return 0;
    };
    match rows.next().await {
        Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0) as usize,
        _ => 0,
    }
}

pub(crate) fn insert_finding(
    conn: &Connection,
    interest_topic: &str,
    item: &crate::web::NewsItem,
    novelty: f64,
) -> Result<i64, Box<dyn Error + Send + Sync>> {
    let found_at = timestamp_now();
    block_on(async {
        conn.execute(
            "INSERT INTO findings (interest_topic, title, url, snippet, novelty, found_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                interest_topic,
                item.title.as_str(),
                item.url.as_str(),
                item.snippet.as_str(),
                novelty,
                found_at
            ],
        )
        .await?;
        Ok(conn.last_insert_rowid())
    })
}

pub(crate) fn unsurfaced_findings(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<Finding>, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, interest_topic, title, url, snippet, novelty, found_at, status, surfaced_at
                 FROM findings WHERE status = 'candidate' ORDER BY novelty DESC, found_at DESC LIMIT ?1",
                params![limit as i64],
            )
            .await?;
        let mut findings = Vec::new();
        while let Some(row) = rows.next().await? {
            findings.push(finding_from_row(&row)?);
        }
        Ok(findings)
    })
}

pub(crate) async fn unsurfaced_findings_in_windows(
    conn: &Connection,
    first: (&str, &str),
    second: (&str, &str),
    limit: usize,
) -> Result<Vec<Finding>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id, interest_topic, title, url, snippet, novelty, found_at, status, surfaced_at
             FROM findings WHERE status = 'candidate' AND
             ((found_at >= ?1 AND found_at < ?2) OR (found_at >= ?3 AND found_at < ?4))
             ORDER BY novelty DESC, found_at DESC LIMIT ?5",
            params![first.0, first.1, second.0, second.1, limit as i64],
        )
        .await?;
    let mut findings = Vec::new();
    while let Some(row) = rows.next().await? {
        findings.push(finding_from_row(&row)?);
    }
    Ok(findings)
}

fn finding_from_row(row: &turso::Row) -> Result<Finding, Box<dyn Error + Send + Sync>> {
    Ok(Finding {
        finding_id: row.get::<i64>(0)?,
        interest_topic: row.get::<String>(1)?,
        title: row.get::<String>(2)?,
        url: row.get::<String>(3)?,
        snippet: row.get::<String>(4)?,
        novelty: row.get::<f64>(5)?,
        found_at: row.get::<String>(6)?,
        status: row.get::<String>(7)?,
        surfaced_at: row.get::<String>(8)?,
    })
}

pub(crate) fn set_ping_mute(
    conn: &Connection,
    until: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    block_on(async {
        conn.execute(
            "INSERT INTO ping_policy (topic, muted_until) VALUES ('', ?1)
             ON CONFLICT(topic) DO UPDATE SET muted_until = excluded.muted_until",
            params![until],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn resume_pings(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    set_ping_mute(conn, "")
}

pub(crate) fn divert_ping_topic(
    conn: &Connection,
    topic: &str,
    until: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    block_on(async {
        conn.execute(
            "INSERT INTO ping_policy (topic, divert_until) VALUES (?1, ?2)
             ON CONFLICT(topic) DO UPDATE SET divert_until = excluded.divert_until",
            params![topic.trim(), until],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn load_ping_policy(
    conn: &Connection,
    now: &str,
) -> Result<PingPolicy, Box<dyn Error + Send + Sync>> {
    block_on(async {
        let mut policy = PingPolicy {
            muted_until: DEFAULT_PING_MUTED_UNTIL.to_string(),
            diverts: Vec::new(),
        };
        let mut rows = conn
            .query(
                "SELECT topic, muted_until, divert_until FROM ping_policy",
                (),
            )
            .await?;
        while let Some(row) = rows.next().await? {
            let topic = row.get::<String>(0)?;
            if topic.is_empty() {
                policy.muted_until = row.get::<String>(1)?;
                continue;
            }
            let until = row.get::<String>(2)?;
            if until.as_str() > now {
                policy.diverts.push((topic, until));
            }
        }
        Ok(policy)
    })
}

pub(crate) fn mark_finding_surfaced(
    conn: &Connection,
    finding_id: i64,
    session_id: &str,
    turn_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let surfaced_at = timestamp_now();
    begin_transaction(conn)?;
    let result = block_on(async {
        conn.execute(
            "UPDATE findings SET status = 'surfaced', surfaced_at = ?1
             WHERE id = ?2 AND status = 'candidate'",
            params![surfaced_at.as_str(), finding_id],
        )
        .await?;
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'conversation', ?2)",
            params![session_id, surfaced_at.as_str()],
        )
        .await?;
        let payload = serde_json::json!({"finding_id": finding_id, "surfaced": true}).to_string();
        conn.execute(
            "INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, payload)
             VALUES (?1, ?2, ?3, 'system', 'ping', 'surfaced', ?4)",
            params![surfaced_at, session_id, turn_id, payload],
        )
        .await?;
        Ok(())
    });
    if let Err(err) = result {
        let _ = block_on(async { conn.execute("ROLLBACK", ()).await });
        return Err(err);
    }
    commit_transaction(conn)
}

#[allow(dead_code)] // Single-window API is part of the findings queue contract.
pub(crate) async fn count_findings_in_window(
    conn: &Connection,
    from: &str,
    until: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    count_findings_in_windows(conn, from, until, from, until).await
}

pub(crate) async fn count_findings_in_windows(
    conn: &Connection,
    first_from: &str,
    first_until: &str,
    second_from: &str,
    second_until: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM findings
             WHERE (found_at >= ?1 AND found_at < ?2)
                OR (found_at >= ?3 AND found_at < ?4)",
            params![first_from, first_until, second_from, second_until],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(row.get::<i64>(0)? as usize)
}

pub async fn load_inbox_outcomes(
    conn: &Connection,
    from: &str,
    until: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn.query(
        "SELECT source, status, result FROM inbox WHERE status = 'queued' OR (resolved_at >= ?1 AND resolved_at < ?2) ORDER BY id",
        params![from, until],
    ).await else { return Ok(Vec::new()) };
    let mut outcomes = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = row.get::<String>(0)?;
        let status = row.get::<String>(1)?;
        let result = row.get::<String>(2)?;
        if status == "failed" {
            outcomes.push(format!("{source}: failed ({result})"));
        } else {
            outcomes.push(format!("{source}: {status}"));
        }
    }
    Ok(outcomes)
}

pub fn db_init() -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    let count = runtime().block_on(node_count(&conn))?;
    if count > 0 {
        return Ok(());
    }
    let graph = load_from_toon();
    if graph.nodes.is_empty() && graph.edges.is_empty() {
        return Ok(());
    }
    runtime().block_on(write_graph(&conn, &graph))
}

async fn node_count(conn: &Connection) -> Result<i64, Box<dyn Error + Send + Sync>> {
    let mut rows = conn.query("SELECT COUNT(*) FROM nodes", ()).await?;
    if let Some(row) = rows.next().await?
        && let Value::Integer(count) = row.get_value(0)?
    {
        return Ok(count);
    }
    Ok(0)
}

pub fn load_graph() -> Graph {
    let Ok(conn) = open_connection() else {
        return Graph::new();
    };
    runtime().block_on(load_graph_async(&conn))
}

async fn load_graph_async(conn: &Connection) -> Graph {
    let mut graph = Graph::new();
    load_nodes(conn, &mut graph).await;
    load_edges(conn, &mut graph).await;
    graph
}

async fn load_nodes(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn
        .query("SELECT id, node_type, body, date FROM nodes", ())
        .await
    else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        let (Ok(node_id), Ok(node_type), Ok(body), Ok(date)) = (
            row.get_value(0),
            row.get_value(1),
            row.get_value(2),
            row.get_value(3),
        ) else {
            continue;
        };
        graph.nodes.insert(
            text(node_id),
            Node {
                node_type: text(node_type),
                body: text(body),
                date: text(date),
            },
        );
    }
}

async fn load_edges(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling FROM edges",
            (),
        )
        .await
    else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        if let Some(edge) = parse_edge_row(&row) {
            graph.edges.push(edge);
        }
    }
}

pub fn save_graph(graph: &Graph) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    runtime().block_on(write_graph(&conn, graph))
}

pub fn begin_transaction(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("BEGIN", ()).await?;
        Ok(())
    })
}

pub fn commit_transaction(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("COMMIT", ()).await?;
        Ok(())
    })
}

pub fn write_node(
    conn: &Connection,
    node_id: &str,
    node: &Node,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, body, date) VALUES (?1, ?2, ?3, ?4)",
            params![
                node_id,
                node.node_type.as_str(),
                node.body.as_str(),
                node.date.as_str()
            ],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn insert_interest(
    conn: &Connection,
    topic: &str,
    origin: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let interest_id = normalize_id(topic);
    if interest_id.is_empty() {
        return Err("interest topic is empty".into());
    }
    let created_at = crate::timeutil::timestamp_now();
    runtime().block_on(async {
        add_interest_node_columns(conn).await;
        conn.execute(
            "INSERT INTO nodes (id, node_type, body, date, origin, embedding)
             VALUES (?1, 'interest', ?2, ?3, ?4, NULL)
             ON CONFLICT(id) DO UPDATE SET node_type = 'interest', body = excluded.body,
                 date = excluded.date, origin = excluded.origin, embedding = NULL",
            params![interest_id.as_str(), topic.trim(), created_at, origin],
        )
        .await?;
        Ok::<(), Box<dyn Error + Send + Sync>>(())
    })?;
    Ok(interest_id)
}

pub(crate) fn top_interests(limit: usize) -> Vec<Interest> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(top_interests_with(&conn, limit))
        .unwrap_or_default()
}

pub(crate) async fn top_interests_with(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<Interest>, Box<dyn Error + Send + Sync>> {
    add_interest_node_columns(conn).await;
    let mut interests = load_registered_interests(conn).await?;
    append_attention_interests(conn, &mut interests).await?;
    append_curiosity_interests(conn, &mut interests).await?;
    append_actor_position_interests(conn, &mut interests).await?;
    interests.sort_by(|left, right| {
        right
            .strength
            .total_cmp(&left.strength)
            .then_with(|| right.created_at.cmp(&left.created_at))
    });
    dedup_interests(&mut interests);
    interests.truncate(limit);
    Ok(interests)
}

async fn load_registered_interests(
    conn: &Connection,
) -> Result<Vec<Interest>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT id, body, origin, date FROM nodes WHERE node_type = 'interest' ORDER BY date DESC",
            (),
        )
        .await?;
    let mut interests = Vec::new();
    while let Some(row) = rows.next().await? {
        interests.push(Interest {
            node_id: text(row.get_value(0)?),
            topic: text(row.get_value(1)?),
            origin: text(row.get_value(2)?),
            created_at: text(row.get_value(3)?),
            strength: 1.0,
        });
    }
    Ok(interests)
}

async fn append_attention_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT endpoint, MAX(ts) AS newest, COUNT(*) AS strength
             FROM (
                 SELECT source AS endpoint, ts FROM events
                 WHERE event_type IN ('reinforced', 'marked') AND source != ''
                 UNION ALL
                 SELECT target AS endpoint, ts FROM events
                 WHERE event_type IN ('reinforced', 'marked') AND target != ''
             ) attention
             JOIN nodes n ON n.id = endpoint
             WHERE n.node_type NOT IN ('interest', 'loop', 'note')
             GROUP BY endpoint ORDER BY COUNT(*) DESC, newest DESC LIMIT 12",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        let node_id = text(row.get_value(0)?);
        interests.push(derived_interest(
            &node_id,
            &node_id,
            text(row.get_value(1)?),
            real(row.get_value(2)?),
        ));
    }
    Ok(())
}

async fn append_curiosity_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT e.target, COALESCE(NULLIF(target.body, ''), e.target), note.date
             FROM nodes note JOIN edges e ON e.source = note.id
             LEFT JOIN nodes target ON target.id = e.target
             WHERE note.node_type = 'note' AND e.expired_at IS NULL
             ORDER BY note.date DESC LIMIT 12",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        interests.push(derived_interest(
            &text(row.get_value(0)?),
            &text(row.get_value(1)?),
            text(row.get_value(2)?),
            1.0,
        ));
    }
    Ok(())
}

async fn append_actor_position_interests(
    conn: &Connection,
    interests: &mut Vec<Interest>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(mut rows) = conn
        .query(
            "SELECT vector_extract(sum_vector), updated_at, total_weight FROM actor_positions
             WHERE sum_vector IS NOT NULL ORDER BY updated_at DESC LIMIT 6",
            (),
        )
        .await
    else {
        return Ok(());
    };
    while let Some(row) = rows.next().await? {
        let vector_text = text(row.get_value(0)?);
        let Ok(vector) = parse_vector32_json(&vector_text) else {
            continue;
        };
        let Some(node_id) = nearest_node_to_vector(conn, &vector).await? else {
            continue;
        };
        interests.push(derived_interest(
            &node_id,
            &node_id,
            text(row.get_value(1)?),
            real(row.get_value(2)?),
        ));
    }
    Ok(())
}

async fn nearest_node_to_vector(
    conn: &Connection,
    vector: &[f64],
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT e.source FROM edges e JOIN nodes n ON n.id = e.source
             WHERE e.expired_at IS NULL AND e.embedding IS NOT NULL
               AND n.node_type NOT IN ('interest', 'loop', 'note')
             ORDER BY vector_distance_cos(e.embedding, vector32(?1)) LIMIT 1",
            params![json_str],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

fn derived_interest(node_id: &str, topic: &str, created_at: String, strength: f64) -> Interest {
    Interest {
        node_id: normalize_id(node_id),
        topic: topic.to_string(),
        origin: "derived".to_string(),
        created_at,
        strength,
    }
}

fn dedup_interests(interests: &mut Vec<Interest>) {
    let mut seen = HashSet::new();
    interests.retain(|interest| seen.insert(interest.node_id.clone()));
}

// Weights are stored at two decimals: full float precision survives every
// multiply-and-write cycle (decay, reinforce) and bloats the prompt/token
// side; nothing downstream needs more than 0.01 resolution.
pub(crate) fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn clamped_confidence(confidence: f64, provenance: &str, ceiling: Option<f64>) -> f64 {
    round2(confidence.min(edge_ceiling(provenance, ceiling)))
}

pub fn write_edge(
    conn: &Connection,
    edge: &Edge,
    source_message_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id, memory_type, origin_session, provenance, ceiling) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                edge.source.as_str(),
                edge.target.as_str(),
                edge.content.as_str(),
                round2(edge.stability),
                clamped_confidence(edge.confidence, &edge.provenance, edge.ceiling),
                round2(edge.importance),
                edge.subject as i64,
                edge.date.as_str(),
                edge.occurred_at.as_deref(),
                edge.expired_at.as_deref(),
                source_message_id,
                edge.memory_type.as_str(),
                edge.origin_session.as_str(),
                edge.provenance.as_str(),
                edge.ceiling,
            ],
        )
        .await?;
        Ok(())
    })
}

pub(crate) fn nearest_live_edges(
    conn: &Connection,
    vector: &[f64],
    limit: usize,
) -> Result<Vec<NearestLiveEdge>, Box<dyn Error + Send + Sync>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let json_str = vector32_json(vector)?;
    block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, provenance, confidence,
                        vector_distance_cos(embedding, vector32(?1))
                 FROM edges WHERE embedding IS NOT NULL AND expired_at IS NULL
                 ORDER BY vector_distance_cos(embedding, vector32(?1)) LIMIT ?2",
                params![json_str, limit as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            let distance = row.get::<f64>(4)?;
            edges.push(NearestLiveEdge {
                source: text(row.get_value(0)?),
                target: text(row.get_value(1)?),
                provenance: text_or(row.get_value(2), PROVENANCE_LEGACY),
                confidence: real(row.get_value(3)?),
                similarity: (1.0 - distance).clamp(-1.0, 1.0),
            });
        }
        Ok(edges)
    })
}

pub(crate) fn record_news_graph_material(
    conn: &Connection,
    finding_id: i64,
    interest: &Interest,
    item: &crate::web::NewsItem,
    embedding: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let timestamp = timestamp_now();
    let today = crate::timeutil::today_iso();
    let session_id = format!("news:{today}");
    let target_id = format!("news-finding-{finding_id}");
    let content = format!("{} — {}", item.title, item.snippet);
    let json_str = vector32_json(embedding)?;
    let stability = stability_seed_max(PROVENANCE_WEB_SURFACED);
    block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, node_type, body, date)
             VALUES (?1, 'event', ?2, ?3)",
            params![target_id.as_str(), item.title.as_str(), timestamp.as_str()],
        )
        .await?;
        conn.execute(
            "INSERT OR REPLACE INTO edges
             (source, target, content, stability, confidence, importance, subject, date,
              occurred_at, memory_type, origin_session, provenance, embedding)
             VALUES (?1, ?2, ?3, ?4, 0.5, 0.4, 0, ?5, ?5, 'episodic', ?6, 'web-surfaced', vector32(?7))",
            params![
                interest.node_id.as_str(),
                target_id.as_str(),
                content.as_str(),
                stability,
                timestamp.as_str(),
                session_id.as_str(),
                json_str.as_str()
            ],
        )
        .await?;
        Ok(())
    })
}

pub fn erase_node(conn: &Connection, node_id: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute("DELETE FROM nodes WHERE id = ?1", params![node_id])
            .await?;
        conn.execute(
            "DELETE FROM edges WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM edge_actor_views WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
        Ok(())
    })
}

pub fn erase_edge(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "DELETE FROM edges WHERE source = ?1 AND target = ?2",
            params![source, target],
        )
        .await?;
        delete_edge_actor_views_async(conn, source, target).await?;
        Ok(())
    })
}

pub fn load_edge(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<Option<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_edge_async(conn, source, target))
}

pub fn load_edges_for_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_edges_for_node_async(conn, node_id))
}

pub fn load_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Option<Node>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT node_type, body, date FROM nodes WHERE id = ?1 LIMIT 1",
                params![node_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            return Ok(Some(Node {
                node_type: text(row.get_value(0)?),
                body: text(row.get_value(1)?),
                date: text(row.get_value(2)?),
            }));
        }
        Ok(None)
    })
}

// The game-definition node a session's task hub is linked to (episodic
// hub→game edge), if any. Returns the normalized game id and its node so the
// caller can render the definition body and anchor rules/strategies.
pub fn load_session_game(
    conn: &Connection,
    hub_id: &str,
) -> Result<Option<(String, Node)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT n.id, n.node_type, n.body, n.date FROM nodes n JOIN edges e ON e.target = n.id WHERE e.source = ?1 AND n.node_type = 'game' AND e.expired_at IS NULL LIMIT 1",
                params![hub_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let game_id = normalize_id(&text(row.get_value(0)?));
            let node = Node {
                node_type: text(row.get_value(1)?),
                body: text(row.get_value(2)?),
                date: text(row.get_value(3)?),
            };
            return Ok(Some((game_id, node)));
        }
        Ok(None)
    })
}

// All live semantic edges incident to a game node (rules + strategies, both
// directions), highest-confidence first and capped. The episodic hub→game link
// is excluded by the semantic filter. This is not TOP_K-ranked: the rules of
// the game you are playing are mandatory context.
pub fn load_game_edges(
    conn: &Connection,
    game_id: &str,
    cap: usize,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling FROM edges WHERE (source = ?1 OR target = ?1) AND memory_type = 'semantic' AND expired_at IS NULL ORDER BY confidence DESC LIMIT ?2",
                params![game_id, cap as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            if let Some(edge) = parse_edge_row(&row) {
                edges.push(edge);
            }
        }
        Ok(edges)
    })
}

// Every game-definition node (node_type = "game"), each with its id and node.
// The strategy stage walks these to find games worth consolidating.
pub fn load_game_nodes(
    conn: &Connection,
) -> Result<Vec<(String, Node)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT id, node_type, body, date FROM nodes WHERE node_type = 'game'",
                (),
            )
            .await?;
        let mut games = Vec::new();
        while let Some(row) = rows.next().await? {
            let game_id = normalize_id(&text(row.get_value(0)?));
            let node = Node {
                node_type: text(row.get_value(1)?),
                body: text(row.get_value(2)?),
                date: text(row.get_value(3)?),
            };
            games.push((game_id, node));
        }
        Ok(games)
    })
}

// The live task hubs that played this game: sources of episodic hub→game edges
// (written by define_task). Expired links and other games' hubs are excluded by
// the target + expiry filter. Each hub id embeds its session as `task-<id>`.
pub fn load_game_hubs(
    conn: &Connection,
    game_id: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT DISTINCT source FROM edges WHERE target = ?1 AND memory_type = 'episodic' AND expired_at IS NULL AND source LIKE 'task-%'",
                params![game_id],
            )
            .await?;
        let mut hubs = Vec::new();
        while let Some(row) = rows.next().await? {
            hubs.push(text(row.get_value(0)?));
        }
        Ok(hubs)
    })
}

// The recorded game outcome for a session, if any: the (score, comment) of its
// most recent `outcome` event. score is the assistant-perspective weight
// (win +1 / draw 0 / loss -1); comment is pulled from the event payload.
pub fn load_session_outcome(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<(f64, String)>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let mut rows = conn
            .query(
                // lower(): the session id arrives via the task hub id, which
                // normalize_id lowercased — the events table has the original
                // casing ("...T12-58-17" vs hub-derived "...t12-58-17").
                "SELECT weight, payload FROM events WHERE lower(session_id) = lower(?1) AND event_type = 'outcome' ORDER BY id DESC LIMIT 1",
                params![session_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            let score = real(row.get_value(0)?);
            let comment = outcome_comment(&text(row.get_value(1)?));
            return Ok(Some((score, comment)));
        }
        Ok(None)
    })
}

// Pulls the human comment out of an outcome event payload; a missing or
// unparseable payload just yields an empty comment.
fn outcome_comment(payload: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return String::new();
    };
    let Some(comment) = value.get("comment") else {
        return String::new();
    };
    match comment.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

// Creates the task hub node on the first move of a game and is a no-op
// afterwards — INSERT OR IGNORE preserves the original row so repeated moves
// never reset it.
pub fn ensure_task_hub(
    conn: &Connection,
    hub_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR IGNORE INTO nodes (id, node_type, body, date) VALUES (?1, 'task', '', '')",
            params![hub_id],
        )
        .await?;
        Ok(())
    })
}

// Live ply count on a task hub, counted from the DB so ply numbering survives a
// restart (no in-memory counter). Only move edges (target `<hub>-ply-%`) count,
// so non-move task facts attached to the hub never inflate the number.
pub fn count_task_chain(
    conn: &Connection,
    hub_id: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    count_chain_edges(conn, hub_id, "AND expired_at IS NULL")
}

// Ply numbering must be monotonic: an expired ply still owns its number, or
// the next record_move would upsert over it and resurrect it. Count everything.
pub fn next_ply(conn: &Connection, hub_id: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
    Ok(count_chain_edges(conn, hub_id, "")? + 1)
}

fn count_chain_edges(
    conn: &Connection,
    hub_id: &str,
    expiry_filter: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let pattern = format!("{hub_id}-ply-%");
    let query =
        format!("SELECT COUNT(*) FROM edges WHERE source = ?1 AND target LIKE ?2 {expiry_filter}");
    runtime().block_on(async {
        let mut rows = conn
            .query(&query, params![hub_id, pattern.as_str()])
            .await?;
        if let Some(row) = rows.next().await? {
            return Ok(real(row.get_value(0)?) as usize);
        }
        Ok(0)
    })
}

// The current session's task-hub move chain, ordered by ply (zero-padded
// move-node ids sort lexically = numerically), expired edges excluded. Caps at
// `cap` keeping the most recent plies — a normal game never hits the cap; a
// pathological chain must not blow the context.
pub fn load_task_chain(
    conn: &Connection,
    session_id: &str,
    cap: usize,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let hub_id = normalize_id(&format!("task-{session_id}"));
    let pattern = format!("{hub_id}-ply-%");
    runtime().block_on(async {
        let mut rows = conn
            .query(
                "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling FROM edges WHERE source = ?1 AND target LIKE ?2 AND memory_type = 'episodic' AND expired_at IS NULL ORDER BY target DESC LIMIT ?3",
                params![hub_id.as_str(), pattern.as_str(), cap as i64],
            )
            .await?;
        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            if let Some(edge) = parse_edge_row(&row) {
                edges.push(edge);
            }
        }
        // Loaded newest-first to honor the cap; flip back to ply order.
        edges.reverse();
        Ok(edges)
    })
}

#[allow(dead_code)]
pub fn update_edge_confidence(
    conn: &Connection,
    source: &str,
    target: &str,
    confidence: f64,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    runtime().block_on(update_edge_confidence_async(
        conn, source, target, confidence,
    ))
}

pub fn update_edge_reinforcement(
    conn: &Connection,
    edge: &Edge,
    confidence: f64,
    stability: f64,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        let changed = conn
            .execute(
                "UPDATE edges SET confidence = ?1, stability = ?2
                 WHERE source = ?3 AND target = ?4 AND expired_at IS NULL",
                params![
                    clamped_confidence(confidence, &edge.provenance, edge.ceiling),
                    round2(stability.clamp(0.0, 1.0)),
                    edge.source.as_str(),
                    edge.target.as_str()
                ],
            )
            .await?;
        Ok(changed > 0)
    })
}

#[allow(dead_code)]
async fn update_edge_confidence_async(
    conn: &Connection,
    source: &str,
    target: &str,
    confidence: f64,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some(edge) = load_edge_async(conn, source, target).await? else {
        return Ok(false);
    };
    let changed = conn
        .execute(
            "UPDATE edges SET confidence = ?1 WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
            params![
                clamped_confidence(confidence, &edge.provenance, edge.ceiling),
                source,
                target
            ],
        )
        .await?;
    Ok(changed > 0)
}

async fn load_edge_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<Option<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling FROM edges WHERE source = ?1 AND target = ?2 LIMIT 1",
            params![source, target],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        return Ok(parse_edge_row(&row));
    }
    Ok(None)
}

// Retrieval resets the forgetting clock: an edge injected into a turn was
// just used, so its date moves to today and decay restarts from there.
// Expired edges are left alone — they are history, not live belief.
pub(super) async fn touch_edge_dates(
    conn: &Connection,
    edges: &[Edge],
    today: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let mut refreshed = 0usize;
    for edge in edges {
        if edge.expired_at.is_some() {
            continue;
        }
        let changed = conn
            .execute(
                "UPDATE edges SET date = ?1 WHERE source = ?2 AND target = ?3 AND expired_at IS NULL",
                params![today, edge.source.as_str(), edge.target.as_str()],
            )
            .await?;
        if changed > 0 {
            refreshed += 1;
        }
    }
    Ok(refreshed)
}

async fn load_edges_for_node_async(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, memory_type, origin_session, provenance, ceiling FROM edges WHERE source = ?1 OR target = ?1",
            params![node_id],
        )
        .await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        if let Some(edge) = parse_edge_row(&row) {
            edges.push(edge);
        }
    }
    Ok(edges)
}

pub fn load_unembedded_edges() -> Vec<(String, String, String)> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(load_unembedded_edges_async(&conn))
}

pub(crate) fn load_unembedded_interests() -> Vec<(String, String)> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(async {
        add_interest_node_columns(&conn).await;
        let Ok(mut rows) = conn
            .query(
                "SELECT id, body FROM nodes WHERE node_type = 'interest' AND embedding IS NULL",
                (),
            )
            .await
        else {
            return Vec::new();
        };
        let mut interests = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let (Ok(node_id), Ok(topic)) = (row.get_value(0), row.get_value(1)) else {
                continue;
            };
            interests.push((text(node_id), text(topic)));
        }
        interests
    })
}

pub(crate) fn save_interest_embedding(
    node_id: &str,
    vector: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(async {
        add_interest_node_columns(&conn).await;
        conn.execute(
            "UPDATE nodes SET embedding = vector32(?1) WHERE id = ?2 AND node_type = 'interest'",
            params![json_str, node_id],
        )
        .await?;
        Ok(())
    })
}

#[allow(dead_code)]
pub(crate) fn interest_density(
    vector: &[f64],
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    runtime().block_on(interest_density_with(&conn, vector, max_distance))
}

async fn interest_density_with(
    conn: &Connection,
    vector: &[f64],
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM edges WHERE expired_at IS NULL AND embedding IS NOT NULL
             AND vector_distance_cos(embedding, vector32(?1)) <= ?2",
            params![json_str, max_distance],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(int(row.get_value(0)?) as usize)
}

pub(crate) fn densification_targets(limit: usize) -> Vec<InterestTarget> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(densification_targets_with(&conn, limit))
        .unwrap_or_default()
}

async fn densification_targets_with(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<InterestTarget>, Box<dyn Error + Send + Sync>> {
    add_interest_node_columns(conn).await;
    let mut rows = conn
        .query(
            "SELECT body, vector_extract(embedding) FROM nodes
             WHERE node_type = 'interest' AND embedding IS NOT NULL",
            (),
        )
        .await?;
    let mut targets = Vec::new();
    while let Some(row) = rows.next().await? {
        let topic = text(row.get_value(0)?);
        let vector_text = text(row.get_value(1)?);
        let Ok(embedding) = parse_vector32_json(&vector_text) else {
            continue;
        };
        let density = interest_density_with(conn, &embedding, INTEREST_NEIGHBOUR_DISTANCE).await?;
        targets.push(InterestTarget {
            topic,
            density,
            embedding,
        });
    }
    targets.sort_by_key(|target| target.density);
    targets.truncate(limit);
    Ok(targets)
}

async fn load_unembedded_edges_async(conn: &Connection) -> Vec<(String, String, String)> {
    let Ok(mut rows) = conn
        .query(
            // Loop edges ARE embedded (the similarity trigger below needs
            // them); they stay excluded from ranked retrieval only.
            "SELECT e.source, e.target, e.content FROM edges e WHERE e.expired_at IS NULL AND e.embedding IS NULL",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut edges = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(edge) = parse_unembedded_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    edges
}

fn parse_unembedded_edge_row(row: &turso::Row) -> Option<(String, String, String)> {
    let source = text(row.get_value(0).ok()?);
    let target = text(row.get_value(1).ok()?);
    let content = text(row.get_value(2).ok()?);
    Some((source, target, content))
}

pub fn save_edge_embedding(
    source: &str,
    target: &str,
    vector: &[f64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(async {
        conn.execute(
            "UPDATE edges SET embedding = vector32(?1) WHERE source = ?2 AND target = ?3",
            params![json_str.as_str(), source, target],
        )
        .await?;
        Ok(())
    })
}

pub fn query_edges_by_vector(
    vector: &[f64],
    limit: usize,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let conn = open_connection()?;
    runtime().block_on(query_edges_by_vector_async(&conn, &json_str, limit))
}

async fn query_edges_by_vector_async(
    conn: &Connection,
    json_str: &str,
    limit: usize,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let limit_value = limit as i64;
    match query_edges_ann(conn, json_str, limit_value).await {
        Ok(rows) => Ok(rows),
        Err(_) => query_edges_scan(conn, json_str, limit_value).await,
    }
}

// Live loops nearest to the query embedding — the similarity trigger for the
// Open loops block. Loops are excluded from ranked TOP_K, so without this a
// loop whose fresh anchor has no other edges could never surface contextually.
pub fn load_loops_by_similarity(
    conn: &Connection,
    query_vec: &[f64],
    max_distance: f64,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_loops_by_similarity_async(
        conn,
        query_vec,
        max_distance,
        cap,
    ))
}

async fn load_loops_by_similarity_async(
    conn: &Connection,
    query_vec: &[f64],
    max_distance: f64,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(query_vec)?;
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL AND e.embedding IS NOT NULL AND vector_distance_cos(e.embedding, vector32(?1)) <= ?2 ORDER BY vector_distance_cos(e.embedding, vector32(?1)) ASC LIMIT ?3"
    );
    let rows = conn
        .query(&query, params![json_str, max_distance, cap as i64])
        .await?;
    collect_open_loops(rows).await
}

async fn query_edges_ann(
    conn: &Connection,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    execute_semantic_query(
        conn,
        "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, vector_extract(e.embedding) FROM vector_top_k('edges_vec', vector32(?1), ?2) AS vt JOIN edges AS e ON e.rowid = vt.id LEFT JOIN nodes AS source_node ON source_node.id = e.source LEFT JOIN nodes AS target_node ON target_node.id = e.target WHERE e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest')",
        json_str,
        limit,
    )
    .await
}

async fn query_edges_scan(
    conn: &Connection,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    execute_semantic_query(
        conn,
        "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, vector_extract(e.embedding) FROM edges e LEFT JOIN nodes source_node ON source_node.id = e.source LEFT JOIN nodes target_node ON target_node.id = e.target WHERE e.embedding IS NOT NULL AND e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest') ORDER BY vector_distance_cos(e.embedding, vector32(?1)) LIMIT ?2",
        json_str,
        limit,
    )
    .await
}

async fn execute_semantic_query(
    conn: &Connection,
    query: &str,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn.query(query, params![json_str, limit]).await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        let Some(edge) = parse_semantic_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    Ok(edges)
}

fn parse_semantic_edge_row(row: &turso::Row) -> Option<SemanticEdgeRow> {
    let edge = parse_edge_row(row)?;
    let embedding = parse_optional_vector32(row.get_value(14).ok());
    Some((
        edge.source,
        edge.target,
        edge.content,
        edge.confidence,
        edge.stability,
        edge.importance,
        edge.subject,
        edge.date,
        edge.occurred_at,
        edge.expired_at,
        edge.memory_type,
        edge.origin_session,
        edge.provenance,
        embedding,
    ))
}

pub fn load_all_edges_semantic() -> Vec<SemanticEdgeRow> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime().block_on(load_all_edges_semantic_async(&conn))
}

async fn load_all_edges_semantic_async(conn: &Connection) -> Vec<SemanticEdgeRow> {
    let Ok(mut rows) = conn
        .query(
            "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, e.memory_type, e.origin_session, e.provenance, e.ceiling, CASE WHEN e.embedding IS NULL THEN NULL ELSE vector_extract(e.embedding) END FROM edges e LEFT JOIN nodes source_node ON source_node.id = e.source LEFT JOIN nodes target_node ON target_node.id = e.target WHERE e.expired_at IS NULL AND COALESCE(source_node.node_type, '') NOT IN ('loop', 'interest') AND COALESCE(target_node.node_type, '') NOT IN ('loop', 'interest')",
            (),
        )
        .await
    else {
        return Vec::new();
    };
    let mut edges = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(edge) = parse_semantic_edge_row(&row) else {
            continue;
        };
        edges.push(edge);
    }
    edges
}

pub fn load_edge_actor_views(actor_type: &str, actor_id: &str) -> HashMap<(String, String), f64> {
    let Ok(conn) = open_connection() else {
        return HashMap::new();
    };
    runtime().block_on(load_edge_actor_views_async(&conn, actor_type, actor_id))
}

pub fn load_recent_attention_counts(
    candidates: &[(String, String)],
    window_days: usize,
) -> HashMap<(String, String), usize> {
    if candidates.is_empty() {
        return HashMap::new();
    }
    let Ok(conn) = open_connection() else {
        return HashMap::new();
    };
    runtime()
        .block_on(load_recent_attention_counts_async(
            &conn,
            candidates,
            window_days,
        ))
        .unwrap_or_default()
}

async fn load_recent_attention_counts_async(
    conn: &Connection,
    candidates: &[(String, String)],
    window_days: usize,
) -> Result<HashMap<(String, String), usize>, Box<dyn Error + Send + Sync>> {
    let mut values = Vec::new();
    let mut params = Vec::new();
    for (source, target) in candidates {
        values.push("(?, ?)");
        params.push(Value::Text(source.clone()));
        params.push(Value::Text(target.clone()));
    }
    params.push(Value::Integer(-(window_days as i64)));
    let query = format!(
        "WITH candidates(source, target) AS (VALUES {})
         SELECT ev.source, ev.target, COUNT(*)
         FROM events ev
         JOIN candidates c ON c.source = ev.source AND c.target = ev.target
         WHERE ev.event_type IN ('reinforced', 'marked')
           AND julianday(ev.ts) >= julianday('now', printf('%d days', ?))
         GROUP BY ev.source, ev.target",
        values.join(", ")
    );
    let mut rows = conn.query(&query, turso::params_from_iter(params)).await?;
    let mut counts = HashMap::new();
    while let Some(row) = rows.next().await? {
        let source = text(row.get_value(0)?);
        let target = text(row.get_value(1)?);
        counts.insert((source, target), int(row.get_value(2)?) as usize);
    }
    Ok(counts)
}

pub(crate) async fn load_turn_region(
    conn: &Connection,
    turn_id: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT ev.source
             FROM events ev
             JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
             LEFT JOIN nodes n ON n.id = ev.source
             WHERE ev.turn_id = ?1 AND ev.event_type = 'injected'
               AND ed.expired_at IS NULL
             ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC
             LIMIT 1",
            params![turn_id],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let region = text(row.get_value(0)?);
    if region.is_empty() {
        return Ok(None);
    }
    Ok(Some(region))
}

pub(crate) async fn load_previous_turn_region(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let Some(turn_id) = load_previous_user_turn(conn, session_id, current_turn).await? else {
        return Ok(None);
    };
    load_turn_region(conn, &turn_id).await
}

async fn load_previous_user_turn(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT previous.id
             FROM messages current
             JOIN messages previous ON previous.session_id = current.session_id
             WHERE current.id = ?1 AND current.session_id = ?2
               AND previous.role = 'user'
               AND (previous.created_at < current.created_at
                    OR (previous.created_at = current.created_at
                        AND previous.rowid < current.rowid))
             ORDER BY previous.created_at DESC, previous.rowid DESC
             LIMIT 1",
            params![current_turn, session_id],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

pub(crate) async fn load_region_transitions(
    conn: &Connection,
    source: &str,
    window_days: usize,
) -> Result<Vec<RegionTransition>, Box<dyn Error + Send + Sync>> {
    let mut signals = load_adjacent_region_transitions(conn, source, window_days).await?;
    load_sequence_feedback(conn, source, window_days, &mut signals).await?;
    Ok(signals)
}

const REGION_TRANSITIONS_QUERY: &str = "SELECT current.created_at,
           (SELECT ev.source FROM events ev
            JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
            LEFT JOIN nodes n ON n.id = ev.source
            WHERE ev.turn_id = (
                SELECT previous.id FROM messages previous
                WHERE previous.session_id = current.session_id
                  AND previous.role = 'user'
                  AND (previous.created_at < current.created_at
                       OR (previous.created_at = current.created_at
                           AND previous.rowid < current.rowid))
                ORDER BY previous.created_at DESC, previous.rowid DESC LIMIT 1
            )
              AND ev.event_type = 'injected' AND ed.expired_at IS NULL
            ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC LIMIT 1),
           (SELECT ev.source FROM events ev
            JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
            LEFT JOIN nodes n ON n.id = ev.source
            WHERE ev.turn_id = current.id
              AND ev.event_type = 'injected' AND ed.expired_at IS NULL
            ORDER BY (n.node_type = 'entity') DESC, ed.confidence DESC, ev.id ASC LIMIT 1)
    FROM messages current
    JOIN sessions session ON session.id = current.session_id
    WHERE current.role = 'user' AND session.session_type = 'conversation'
      AND julianday(current.created_at) >= julianday('now', printf('%d days', ?1))";

async fn load_adjacent_region_transitions(
    conn: &Connection,
    source: &str,
    window_days: usize,
) -> Result<Vec<RegionTransition>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(REGION_TRANSITIONS_QUERY, params![-(window_days as i64)])
        .await?;
    let mut signals = Vec::new();
    while let Some(row) = rows.next().await? {
        let previous = text(row.get_value(1)?);
        let current = text(row.get_value(2)?);
        if previous != source || current.is_empty() || current == previous {
            continue;
        }
        signals.push(RegionTransition {
            target: current,
            timestamp: text(row.get_value(0)?),
            signal: "transition".to_string(),
        });
    }
    Ok(signals)
}

async fn load_sequence_feedback(
    conn: &Connection,
    source: &str,
    window_days: usize,
    signals: &mut Vec<RegionTransition>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT target, ts, event_type FROM events
             WHERE actor_id = 'sequence' AND source = ?1
               AND event_type IN ('followed', 'rejected')
               AND julianday(ts) >= julianday('now', printf('%d days', ?2))
             ORDER BY id",
            params![source, -(window_days as i64)],
        )
        .await?;
    while let Some(row) = rows.next().await? {
        signals.push(RegionTransition {
            target: text(row.get_value(0)?),
            timestamp: text(row.get_value(1)?),
            signal: text(row.get_value(2)?),
        });
    }
    Ok(())
}

pub(crate) async fn last_suggestion_for_session(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<SequenceSuggestion>, Box<dyn Error + Send + Sync>> {
    let Some(previous_turn) = load_previous_user_turn(conn, session_id, current_turn).await? else {
        return Ok(None);
    };
    let mut rows = conn
        .query(
            "SELECT suggested.turn_id, suggested.source, suggested.target
             FROM events suggested
             WHERE suggested.session_id = ?1 AND suggested.turn_id = ?2
               AND suggested.actor_id = 'sequence'
               AND suggested.event_type = 'suggested'
               AND NOT EXISTS (
                   SELECT 1 FROM events outcome
                   WHERE outcome.session_id = ?1 AND outcome.turn_id = ?3
                     AND outcome.actor_id = 'sequence'
                     AND outcome.event_type IN ('followed', 'rejected')
                     AND outcome.source = suggested.source
                     AND outcome.target = suggested.target
               )
             ORDER BY suggested.id DESC LIMIT 1",
            params![session_id, previous_turn, current_turn],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(SequenceSuggestion {
        turn_id: text(row.get_value(0)?),
        source: text(row.get_value(1)?),
        target: text(row.get_value(2)?),
    }))
}

async fn load_edge_actor_views_async(
    conn: &Connection,
    actor_type: &str,
    actor_id: &str,
) -> HashMap<(String, String), f64> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source, target, views FROM edge_actor_views WHERE actor_type = ?1 AND actor_id = ?2",
            params![actor_type, actor_id],
        )
        .await
    else {
        return HashMap::new();
    };
    let mut views = HashMap::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some((source, target, count)) = parse_edge_actor_view_row(&row) else {
            continue;
        };
        views.insert((source, target), count);
    }
    views
}

fn parse_edge_actor_view_row(row: &turso::Row) -> Option<(String, String, f64)> {
    let source = text(row.get_value(0).ok()?);
    let target = text(row.get_value(1).ok()?);
    let count = real(row.get_value(2).ok()?);
    Some((source, target, count))
}

async fn delete_edge_actor_views_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "DELETE FROM edge_actor_views WHERE source = ?1 AND target = ?2",
        params![source, target],
    )
    .await?;
    Ok(())
}

pub fn edge_count() -> usize {
    let Ok(conn) = open_connection() else {
        return 0;
    };
    let result = runtime().block_on(async {
        let mut rows = conn
            .query("SELECT COUNT(*) FROM edges WHERE expired_at IS NULL", ())
            .await?;
        if let Some(row) = rows.next().await?
            && let Value::Integer(count) = row.get_value(0)?
        {
            return Ok::<usize, Box<dyn Error + Send + Sync>>(count as usize);
        }
        Ok(0)
    });
    result.unwrap_or_default()
}

async fn write_graph(conn: &Connection, graph: &Graph) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute("BEGIN", ()).await?;
    conn.execute("DELETE FROM nodes", ()).await?;
    conn.execute("DELETE FROM edges", ()).await?;
    for (node_id, node) in &graph.nodes {
        insert_graph_node(conn, node_id, node).await?;
    }
    for edge in &graph.edges {
        insert_graph_edge(conn, edge).await?;
    }
    conn.execute("COMMIT", ()).await?;
    Ok(())
}

async fn insert_graph_node(
    conn: &Connection,
    node_id: &str,
    node: &Node,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, ?2, ?3, ?4)",
        params![
            node_id,
            node.node_type.as_str(),
            node.body.as_str(),
            node.date.as_str()
        ],
    )
    .await?;
    Ok(())
}

async fn insert_graph_edge(
    conn: &Connection,
    edge: &Edge,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
            "INSERT INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id, memory_type, origin_session, provenance, ceiling) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            edge.source.as_str(),
            edge.target.as_str(),
            edge.content.as_str(),
            round2(edge.stability),
            clamped_confidence(edge.confidence, &edge.provenance, edge.ceiling),
            round2(edge.importance),
            edge.subject as i64,
            edge.date.as_str(),
            edge.occurred_at.as_deref(),
            edge.expired_at.as_deref(),
            "",
            edge.memory_type.as_str(),
            edge.origin_session.as_str(),
            edge.provenance.as_str(),
            edge.ceiling,
        ],
    )
    .await?;
    Ok(())
}

pub fn register_session(session_id: &str, created_at: &str) {
    let Ok(conn) = open_connection() else {
        return;
    };
    runtime().block_on(async {
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?1, ?2)",
                params![session_id, created_at],
            )
            .await;
    });
}

pub fn register_dream_session(session_id: &str, created_at: &str) {
    let Ok(conn) = open_connection() else {
        return;
    };
    runtime().block_on(async {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'dream', ?2)",
            params![session_id, created_at],
        )
        .await;
    });
}

pub fn register_exploration_session(session_id: &str, created_at: &str) {
    let Ok(conn) = open_connection() else {
        return;
    };
    runtime().block_on(async {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'exploration', ?2)",
            params![session_id, created_at],
        )
        .await;
    });
}

pub fn register_ingestion_session(session_id: &str, created_at: &str) {
    let Ok(conn) = open_connection() else {
        return;
    };
    runtime().block_on(async {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'ingestion', ?2)",
            params![session_id, created_at],
        )
        .await;
    });
}

pub(crate) fn register_news_session(session_id: &str, created_at: &str) {
    let Ok(conn) = open_connection() else {
        return;
    };
    runtime().block_on(async {
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO sessions (id, session_type, created_at) VALUES (?1, 'news', ?2)",
                params![session_id, created_at],
            )
            .await;
    });
}

pub fn latest_session_id(session_type: &str) -> Option<String> {
    let conn = open_connection().ok()?;
    runtime().block_on(latest_session_id_async(&conn, session_type))
}

async fn latest_session_id_async(conn: &Connection, session_type: &str) -> Option<String> {
    let mut rows = conn
        .query(
            "SELECT id FROM sessions WHERE session_type = ?1 ORDER BY created_at DESC, rowid DESC LIMIT 1",
            params![session_type],
        )
        .await
        .ok()?;
    let row = match rows.next().await {
        Ok(Some(row)) => row,
        _ => return None,
    };
    match row.get_value(0) {
        Ok(value) => Some(text(value)),
        Err(_) => None,
    }
}

pub fn load_open_curiosity_notes(limit: usize) -> Vec<OpenCuriosityNoteRow> {
    let Ok(conn) = open_connection() else {
        return Vec::new();
    };
    runtime()
        .block_on(load_open_curiosity_notes_async(&conn, limit))
        .unwrap_or_default()
}

pub(crate) async fn load_open_curiosity_notes_async(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<OpenCuriosityNoteRow>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT n.id, n.body, e.target, n.date
                 FROM nodes n
                 JOIN edges e ON e.source = n.id
                 LEFT JOIN (
                     SELECT DISTINCT ev.source, ev.target
                     FROM events ev
                     JOIN sessions s ON s.id = ev.session_id
                     WHERE s.session_type = 'exploration' AND ev.event_type = 'retrieved'
                 ) consumed ON consumed.source = e.source AND consumed.target = e.target
                 WHERE n.node_type = 'note' AND e.expired_at IS NULL AND consumed.source IS NULL
                 ORDER BY n.date ASC, n.id ASC
                 LIMIT ?1",
            params![limit as i64],
        )
        .await?;
    let mut notes = Vec::new();
    while let Some(row) = rows.next().await? {
        notes.push((
            text(row.get_value(0)?),
            text(row.get_value(1)?),
            text(row.get_value(2)?),
            text(row.get_value(3)?),
        ));
    }
    Ok(notes)
}

pub(crate) async fn load_open_intentions_async(
    conn: &Connection,
    today: &str,
    limit: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL ORDER BY (n.date != '' AND n.date <= ?1) DESC, e.date DESC, e.rowid DESC LIMIT ?2"
    );
    let rows = conn.query(&query, params![today, limit as i64]).await?;
    collect_open_loops(rows).await
}

pub(crate) async fn load_recent_learning_async(
    conn: &Connection,
    since_days: usize,
    limit: usize,
) -> Result<Vec<RecentLearning>, Box<dyn Error + Send + Sync>> {
    let modifier = format!("-{} days", since_days.min(7));
    let mut rows = conn
        .query(
            "SELECT ed.content, ev.source, ev.target, ev.event_type, MAX(ev.ts), ed.provenance
             FROM events ev
             JOIN edges ed ON ed.source = ev.source AND ed.target = ev.target
             WHERE ev.event_type IN ('created', 'reinforced')
               AND ev.source != '' AND ev.target != ''
               AND ed.expired_at IS NULL
               AND julianday(ev.ts) >= julianday(date('now'), ?1)
             GROUP BY ed.source, ed.target, ed.content, ev.event_type, ed.provenance
             ORDER BY MAX(ev.ts) DESC, MAX(ev.id) DESC
             LIMIT ?2",
            params![modifier, limit as i64],
        )
        .await?;
    let mut learning = Vec::new();
    while let Some(row) = rows.next().await? {
        learning.push(RecentLearning {
            content: text(row.get_value(0)?),
            source: text(row.get_value(1)?),
            target: text(row.get_value(2)?),
            event_type: text(row.get_value(3)?),
            timestamp: text(row.get_value(4)?),
            provenance: text(row.get_value(5)?),
        });
    }
    Ok(learning)
}

pub(crate) async fn load_inbox_snapshot_async(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, status, result FROM inbox
             WHERE status IN ('queued', 'done', 'failed')
             ORDER BY (status = 'queued') DESC,
                      CASE WHEN status = 'queued' THEN accepted_at ELSE resolved_at END DESC,
                      id DESC
             LIMIT ?1",
            params![limit as i64],
        )
        .await?;
    let mut items = Vec::new();
    while let Some(row) = rows.next().await? {
        let source = text(row.get_value(0)?);
        let status = text(row.get_value(1)?);
        let result = text(row.get_value(2)?);
        if status == "failed" && !result.is_empty() {
            items.push(format!("{source}: failed ({result})"));
        } else {
            items.push(format!("{source}: {status}"));
        }
    }
    Ok(items)
}

// The start of the current conversation "burst": scan recent messages newest
// first and return the timestamp on the older side of the first away-gap. That
// is the last thing said before the user stepped away, so a catch-up window
// from it spans whatever the background passes did overnight. None = no gap in
// the scanned window (continuously active, or first-ever chat).
pub(crate) async fn conversation_boundary_async(
    conn: &Connection,
    gap_seconds: i64,
    scan_limit: usize,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT created_at FROM messages ORDER BY created_at DESC, rowid DESC LIMIT ?1",
            params![scan_limit as i64],
        )
        .await?;
    let mut stamps = Vec::new();
    while let Some(row) = rows.next().await? {
        stamps.push(text(row.get_value(0)?));
    }
    let mut index = 0usize;
    while index + 1 < stamps.len() {
        let newer = crate::timeutil::timestamp_to_epoch_seconds(&stamps[index]);
        let older = crate::timeutil::timestamp_to_epoch_seconds(&stamps[index + 1]);
        if let (Some(newer), Some(older)) = (newer, older)
            && newer - older > gap_seconds
        {
            return Ok(Some(stamps[index + 1].clone()));
        }
        index += 1;
    }
    Ok(None)
}

pub(crate) async fn latest_assistant_message_timestamp_async(
    conn: &Connection,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT created_at FROM messages WHERE role = 'assistant' ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (),
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(text(row.get_value(0)?)))
}

pub fn count_new_embedding_neighbours(
    vector: &[f64],
    created_at: &str,
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let conn = open_connection()?;
    runtime().block_on(count_new_embedding_neighbours_async(
        &conn,
        vector,
        created_at,
        max_distance,
    ))
}

async fn count_new_embedding_neighbours_async(
    conn: &Connection,
    vector: &[f64],
    created_at: &str,
    max_distance: f64,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let json_str = vector32_json(vector)?;
    let mut rows = conn
        .query(
            "SELECT COUNT(*)
             FROM edges e
             JOIN (
                 SELECT source, target, MIN(ts) AS created_at
                 FROM events
                 WHERE event_type = 'created' AND source != '' AND target != ''
                 GROUP BY source, target
             ) created ON created.source = e.source AND created.target = e.target
             WHERE e.expired_at IS NULL
               AND e.embedding IS NOT NULL
               AND created.created_at > ?2
               AND vector_distance_cos(e.embedding, vector32(?1)) <= ?3",
            params![json_str, created_at, max_distance],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(0);
    };
    Ok(int(row.get_value(0)?) as usize)
}

const OPEN_LOOP_SELECT: &str = "SELECT e.source, e.target, n.body, n.date FROM nodes n JOIN edges e ON e.source = n.id JOIN nodes anchor ON anchor.id = e.target";

pub fn load_loops_for_anchors(
    conn: &Connection,
    anchor_ids: &[String],
    today: &str,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_loops_for_anchors_async(conn, anchor_ids, today, cap))
}

async fn load_loops_for_anchors_async(
    conn: &Connection,
    anchor_ids: &[String],
    today: &str,
    cap: usize,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let (query, values) = loop_anchor_query(anchor_ids, today, cap);
    let rows = conn.query(&query, turso::params_from_iter(values)).await?;
    collect_open_loops(rows).await
}

fn loop_anchor_query(anchor_ids: &[String], today: &str, cap: usize) -> (String, Vec<Value>) {
    let mut values = vec![Value::Text(today.to_string())];
    let mut placeholders = Vec::new();
    for anchor_id in anchor_ids {
        values.push(Value::Text(anchor_id.clone()));
        placeholders.push(format!("?{}", values.len()));
    }
    let anchor_filter = if placeholders.is_empty() {
        String::new()
    } else {
        format!(" OR e.target IN ({})", placeholders.join(", "))
    };
    values.push(Value::Integer(cap as i64));
    let limit_param = values.len();
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL AND ((n.date != '' AND n.date <= ?1){anchor_filter}) ORDER BY e.date ASC, e.rowid ASC LIMIT ?{limit_param}"
    );
    (query, values)
}

#[allow(dead_code)]
pub fn load_due_loops(
    conn: &Connection,
    today: &str,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    runtime().block_on(load_due_loops_async(conn, today))
}

pub(super) async fn load_due_loops_async(
    conn: &Connection,
    today: &str,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let query = format!(
        "{OPEN_LOOP_SELECT} WHERE n.node_type = 'loop' AND n.date != '' AND n.date <= ?1 AND e.memory_type = 'semantic' AND e.content LIKE 'open loop:%' AND e.expired_at IS NULL ORDER BY n.date ASC, e.rowid ASC"
    );
    let rows = conn.query(&query, params![today]).await?;
    collect_open_loops(rows).await
}

async fn collect_open_loops(
    mut rows: turso::Rows,
) -> Result<Vec<OpenLoop>, Box<dyn Error + Send + Sync>> {
    let mut loops = Vec::new();
    while let Some(row) = rows.next().await? {
        loops.push(OpenLoop {
            source: normalize_id(&text(row.get_value(0)?)),
            target: normalize_id(&text(row.get_value(1)?)),
            intention: text(row.get_value(2)?),
            due: text(row.get_value(3)?),
        });
    }
    Ok(loops)
}

pub fn insert_message(session_id: &str, role: &str, content: &str, created_at: &str) -> String {
    let message_id = format!("{session_id}:{role}:{created_at}");
    let Ok(conn) = open_connection() else {
        return message_id;
    };
    runtime().block_on(async {
        let _ = conn.execute(
            "INSERT OR IGNORE INTO messages (id, session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![message_id.as_str(), session_id, role, content, created_at],
        )
        .await;
    });
    message_id
}

pub fn search_messages_fts(queries: &[String]) -> String {
    let fts_query = build_fts_query(queries);
    if fts_query.is_empty() {
        return String::new();
    }
    let Ok(conn) = open_connection() else {
        return String::new();
    };
    runtime().block_on(search_messages_async(&conn, &fts_query, queries))
}

// MATCH needs the FTS index, which exists only in local mode (the cloud
// server cannot store its DDL) — synced mode falls back to LIKE.
async fn search_messages_async(conn: &Connection, fts_query: &str, queries: &[String]) -> String {
    let rows = conn.query(
        "SELECT id, role, content, created_at FROM messages WHERE content MATCH ?1 ORDER BY created_at DESC LIMIT 10",
        params![fts_query],
    )
    .await;
    match rows {
        Ok(rows) => collect_message_rows(rows).await,
        Err(_) => search_messages_like(conn, queries).await,
    }
}

async fn search_messages_like(conn: &Connection, queries: &[String]) -> String {
    let mut clauses = Vec::new();
    let mut values: Vec<Value> = Vec::new();
    for query in queries {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            continue;
        }
        clauses.push(format!("content LIKE ?{}", values.len() + 1));
        values.push(Value::Text(format!("%{trimmed}%")));
    }
    if clauses.is_empty() {
        return String::new();
    }
    let sql = format!(
        "SELECT id, role, content, created_at FROM messages WHERE {} ORDER BY created_at DESC LIMIT 10",
        clauses.join(" OR ")
    );
    let Ok(rows) = conn.query(&sql, turso::params_from_iter(values)).await else {
        return String::new();
    };
    collect_message_rows(rows).await
}

async fn collect_message_rows(mut rows: turso::Rows) -> String {
    let mut lines = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(line) = format_message_search_row(&row) else {
            continue;
        };
        lines.push(line);
    }
    join_lines(&lines)
}

fn build_fts_query(queries: &[String]) -> String {
    let mut fts_query = String::new();
    for query in queries {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !fts_query.is_empty() {
            fts_query.push_str(" OR ");
        }
        fts_query.push_str(trimmed);
    }
    fts_query
}

fn format_message_search_row(row: &turso::Row) -> Option<String> {
    let created_at = text(row.get_value(3).ok()?);
    let role = text(row.get_value(1).ok()?);
    let content = text(row.get_value(2).ok()?);
    Some(format!("[{created_at}] {role}: {content}"))
}

pub fn read_message_context(edge_source: &str, edge_target: &str) -> String {
    let Ok(conn) = open_connection() else {
        return "no provenance recorded for this edge".to_string();
    };
    runtime().block_on(read_message_context_async(&conn, edge_source, edge_target))
}

async fn read_message_context_async(
    conn: &Connection,
    edge_source: &str,
    edge_target: &str,
) -> String {
    let source_message_id = lookup_source_message_id(conn, edge_source, edge_target).await;
    if source_message_id.is_empty() {
        return "no provenance recorded for this edge".to_string();
    }
    let Some(session_id) = source_session_id(&source_message_id) else {
        return "no provenance recorded for this edge".to_string();
    };
    let messages = load_session_messages(conn, &session_id).await;
    format_message_context(&messages, &source_message_id)
}

async fn lookup_source_message_id(
    conn: &Connection,
    edge_source: &str,
    edge_target: &str,
) -> String {
    let normalized_source = normalize_id(edge_source);
    let normalized_target = normalize_id(edge_target);
    let Ok(mut rows) = conn
        .query(
            "SELECT source_message_id FROM edges WHERE source = ?1 AND target = ?2",
            params![normalized_source.as_str(), normalized_target.as_str()],
        )
        .await
    else {
        return String::new();
    };
    if let Ok(Some(row)) = rows.next().await
        && let Ok(value) = row.get_value(0)
    {
        return text(value);
    }
    String::new()
}

pub fn source_session_id(source_message_id: &str) -> Option<String> {
    match source_message_id.split_once(':') {
        Some((session_id, _)) if !session_id.is_empty() => Some(session_id.to_string()),
        _ => None,
    }
}

struct MessageContextRow {
    message_id: String,
    role: String,
    content: String,
}

async fn load_session_messages(conn: &Connection, session_id: &str) -> Vec<MessageContextRow> {
    let Ok(mut rows) = conn.query(
        "SELECT id, role, content FROM messages WHERE session_id = ?1 ORDER BY created_at LIMIT 20",
        params![session_id],
    )
    .await else {
        return Vec::new();
    };
    let mut messages = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let Some(message) = parse_message_context_row(&row) else {
            continue;
        };
        messages.push(message);
    }
    messages
}

fn parse_message_context_row(row: &turso::Row) -> Option<MessageContextRow> {
    Some(MessageContextRow {
        message_id: text(row.get_value(0).ok()?),
        role: text(row.get_value(1).ok()?),
        content: text(row.get_value(2).ok()?),
    })
}

fn format_message_context(messages: &[MessageContextRow], source_message_id: &str) -> String {
    let mut msg_index = 0usize;
    while msg_index < messages.len() {
        if messages[msg_index].message_id == source_message_id {
            return format_context_pair(messages, msg_index);
        }
        msg_index += 1;
    }
    "no provenance recorded for this edge".to_string()
}

fn format_context_pair(messages: &[MessageContextRow], msg_index: usize) -> String {
    let mut text = format_context_line(&messages[msg_index]);
    if msg_index + 1 < messages.len() {
        text.push('\n');
        text.push_str(&format_context_line(&messages[msg_index + 1]));
    }
    text
}

fn format_context_line(message: &MessageContextRow) -> String {
    let role = match message.role.as_str() {
        "user" => "User",
        "assistant" => "Assistant",
        _ => message.role.as_str(),
    };
    format!("{role}: {}", message.content)
}

fn join_lines(lines: &[String]) -> String {
    let mut text = String::new();
    let mut line_index = 0usize;
    while line_index < lines.len() {
        if line_index > 0 {
            text.push('\n');
        }
        text.push_str(&lines[line_index]);
        line_index += 1;
    }
    text
}

pub fn load_last_decay() -> String {
    let Ok(conn) = open_connection() else {
        return String::new();
    };
    runtime().block_on(async {
        let Ok(mut rows) = conn
            .query("SELECT value FROM meta WHERE key = 'last_decay'", ())
            .await
        else {
            return String::new();
        };
        if let Ok(Some(row)) = rows.next().await
            && let Ok(val) = row.get_value(0)
        {
            return text(val);
        }
        String::new()
    })
}

pub fn save_last_decay(date: &str) {
    let Ok(conn) = open_connection() else { return };
    let _ = runtime().block_on(conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_decay', ?1)",
        params![date],
    ));
}

pub fn graph_diff_toon(pre: &Graph, post: &Graph) -> String {
    let mut lines: Vec<String> = Vec::new();
    node_diff_lines(pre, post, &mut lines);
    edge_diff_lines(pre, post, &mut lines);
    lines.sort();
    lines.join("\n")
}

fn node_diff_lines(pre: &Graph, post: &Graph, lines: &mut Vec<String>) {
    for (node_id, node) in &post.nodes {
        if !pre.nodes.contains_key(node_id) {
            lines.push(format!(
                "+nodes[{node_id} | {} | {} | {}]",
                node.node_type, node.body, node.date
            ));
        }
    }
    for node_id in pre.nodes.keys() {
        if !post.nodes.contains_key(node_id) {
            lines.push(format!("-nodes[{node_id}]"));
        }
    }
}

fn edge_diff_lines(pre: &Graph, post: &Graph, lines: &mut Vec<String>) {
    let mut pre_keys: HashSet<(&str, &str)> = HashSet::new();
    for edge in &pre.edges {
        pre_keys.insert((edge.source.as_str(), edge.target.as_str()));
    }
    let mut post_keys: HashSet<(&str, &str)> = HashSet::new();
    for edge in &post.edges {
        post_keys.insert((edge.source.as_str(), edge.target.as_str()));
    }
    for edge in &post.edges {
        if !pre_keys.contains(&(edge.source.as_str(), edge.target.as_str())) {
            lines.push(format!(
                "+edges[{} → {} | {} | conf={} imp={} stab={}]",
                edge.source,
                edge.target,
                edge.content,
                edge.confidence,
                edge.importance,
                edge.stability
            ));
        }
    }
    for edge in &pre.edges {
        if !post_keys.contains(&(edge.source.as_str(), edge.target.as_str())) {
            lines.push(format!("-edges[{} → {}]", edge.source, edge.target));
        }
    }
}

pub(super) fn parse_edge_row(row: &turso::Row) -> Option<Edge> {
    Some(Edge {
        source: normalize_id(&text(row.get_value(0).ok()?)),
        target: normalize_id(&text(row.get_value(1).ok()?)),
        content: text(row.get_value(2).ok()?),
        stability: real(row.get_value(3).ok()?),
        confidence: real(row.get_value(4).ok()?),
        importance: real(row.get_value(5).ok()?),
        subject: int(row.get_value(6).ok()?) != 0,
        date: text(row.get_value(7).ok()?),
        occurred_at: optional_text(row.get_value(8).ok()),
        expired_at: optional_text(row.get_value(9).ok()),
        memory_type: text_or(row.get_value(10), "semantic"),
        origin_session: text_or(row.get_value(11), ""),
        provenance: text_or(row.get_value(12), PROVENANCE_LEGACY),
        ceiling: optional_real(row.get_value(13).ok()),
    })
}

fn text_or(value: Result<Value, turso::Error>, default: &str) -> String {
    match value {
        Ok(Value::Text(text)) => text,
        _ => default.to_string(),
    }
}

fn optional_text(value: Option<Value>) -> Option<String> {
    match value? {
        Value::Text(string) => Some(string),
        Value::Null => None,
        _ => None,
    }
}

fn optional_real(value: Option<Value>) -> Option<f64> {
    match value? {
        Value::Real(number) => Some(number),
        Value::Integer(number) => Some(number as f64),
        Value::Null => None,
        _ => None,
    }
}

fn parse_optional_vector32(value: Option<Value>) -> Option<Vec<f64>> {
    let json_text = optional_text(value)?;
    if json_text.is_empty() {
        return None;
    }
    parse_vector32_json(&json_text).ok()
}

pub(crate) fn text(val: Value) -> String {
    match val {
        Value::Text(string) => string,
        _ => String::new(),
    }
}

pub(crate) fn real(val: Value) -> f64 {
    match val {
        Value::Real(float) => float,
        Value::Integer(int) => int as f64,
        _ => 0.0,
    }
}

pub(crate) fn int(val: Value) -> i64 {
    match val {
        Value::Integer(int) => int,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PING_MUTED_UNTIL, add_edge_ceiling_column, backfill_edge_epistemics, block_on,
        count_findings_in_window, count_new_embedding_neighbours_async, divert_ping_topic,
        insert_finding, insert_interest, interest_density_with, load_all_edges_semantic_async,
        load_due_loops, load_game_edges, load_loops_for_anchors, load_ping_policy,
        load_recent_attention_counts_async, load_session_game, load_task_chain,
        load_unembedded_edges_async, mark_finding_surfaced, recover_replica_once, resume_pings,
        set_ping_mute, top_interests_with, unsurfaced_findings, update_edge_confidence,
        with_write_lock, write_edge,
    };
    use crate::memory::graph::Edge;
    use crate::web::NewsItem;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use turso::{Builder, Connection, params};

    const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, PRIMARY KEY (source, target))";
    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
    const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
    const FINDINGS_DDL: &str = "CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')";
    const PING_POLICY_DDL: &str = "CREATE TABLE ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')";

    #[test]
    fn startup_recovery_moves_replica_and_retries_once() {
        let directory = recovery_test_dir("retry-success");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"dirty").is_ok());
        assert!(std::fs::write(directory.join("kratt-synced.db-wal"), b"wal").is_ok());
        let mut calls = 0usize;
        let result = recover_replica_once(&replica, "2026-07-16", |recovering| {
            calls += 1;
            if calls == 1 {
                assert!(!recovering);
                panic!("dirty replica");
            }
            assert!(recovering);
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(calls, 2);
        assert!(!replica.exists());
        assert!(
            directory
                .join("kratt-synced.db.corrupt-2026-07-16")
                .exists()
        );
        assert!(
            directory
                .join("kratt-synced.db.corrupt-2026-07-16-wal")
                .exists()
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn startup_recovery_stops_after_second_failure() {
        let directory = recovery_test_dir("retry-failure");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"dirty").is_ok());
        let mut calls = 0usize;
        let result: Result<(), Box<dyn std::error::Error + Send + Sync>> =
            recover_replica_once(&replica, "2026-07-16", |_recovering| {
                calls += 1;
                Err(format!("probe failure {calls}"))
            });
        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert!(!replica.exists());
        let message = match result {
            Err(err) => err.to_string(),
            Ok(()) => String::new(),
        };
        assert!(message.contains("after one retry"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn startup_probe_success_does_not_touch_replica() {
        let directory = recovery_test_dir("no-recovery");
        assert!(std::fs::create_dir_all(&directory).is_ok());
        let replica = directory.join("kratt-synced.db");
        assert!(std::fs::write(&replica, b"healthy").is_ok());
        let mut calls = 0usize;
        let result = recover_replica_once(&replica, "2026-07-16", |recovering| {
            calls += 1;
            assert!(!recovering);
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(calls, 1);
        assert!(replica.exists());
        assert!(
            !directory
                .join("kratt-synced.db.corrupt-2026-07-16")
                .exists()
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    fn recovery_test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("kratt-{label}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn ping_policy_defaults_muted_and_expired_controls_rearm() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(PING_POLICY_DDL, ()).await;
        });
        let now = "2026-07-14T12:00:00";
        let initial = load_ping_policy(&conn, now).unwrap_or_default();
        assert_eq!(initial.muted_until, DEFAULT_PING_MUTED_UNTIL);

        assert!(resume_pings(&conn).is_ok());
        assert_eq!(
            load_ping_policy(&conn, now).unwrap_or_default().muted_until,
            ""
        );
        assert!(set_ping_mute(&conn, "2026-07-15T00:00:00").is_ok());
        let muted = load_ping_policy(&conn, now).unwrap_or_default();
        assert!(muted.muted_until.as_str() > now);
        assert!(divert_ping_topic(&conn, "Rust", "2026-07-15T00:00:00").is_ok());
        assert!(divert_ping_topic(&conn, "Old", "2026-07-14T01:00:00").is_ok());
        let policy = load_ping_policy(&conn, now).unwrap_or_default();
        assert_eq!(
            policy.diverts,
            vec![("Rust".to_string(), "2026-07-15T00:00:00".to_string())]
        );
    }

    #[test]
    fn findings_queue_filters_status_and_counts_window() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(FINDINGS_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, session_type TEXT NOT NULL DEFAULT 'conversation', created_at TEXT NOT NULL DEFAULT '')", ()).await;
        });
        let item = NewsItem {
            title: "A new compiler".to_string(),
            url: "https://example.com/compiler".to_string(),
            snippet: "Details".to_string(),
        };
        let Ok(finding_id) = insert_finding(&conn, "Rust", &item, 0.8) else {
            unreachable!("finding inserts")
        };
        let candidates = unsurfaced_findings(&conn, 10).unwrap_or_default();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].finding_id, finding_id);
        block_on(async {
            let _ = conn
                .execute(
                    "UPDATE findings SET found_at = '2026-07-14T02:00:00Z' WHERE id = ?1",
                    params![finding_id],
                )
                .await;
        });
        assert!(mark_finding_surfaced(&conn, finding_id, "tg-2026-07-14", "ping-1").is_ok());
        assert!(
            unsurfaced_findings(&conn, 10)
                .unwrap_or_default()
                .is_empty()
        );
        let count = block_on(count_findings_in_window(
            &conn,
            "2026-07-14T00:00:00Z",
            "2026-07-15T00:00:00Z",
        ))
        .unwrap_or(0);
        assert_eq!(count, 1);
        assert_eq!(count_surfaced_events(&conn, "ping-1"), 1);
    }

    fn count_surfaced_events(conn: &Connection, turn_id: &str) -> i64 {
        block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT COUNT(*) FROM events WHERE turn_id = ?1 AND event_type = 'surfaced'",
                    params![turn_id],
                )
                .await
            else {
                return 0;
            };
            match rows.next().await {
                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
                _ => 0,
            }
        })
    }

    #[test]
    fn top_interests_unifies_registered_and_derived_regions_and_caps() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn
                .execute(
                    "INSERT INTO nodes (id, node_type) VALUES ('rust', 'entity')",
                    (),
                )
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO events (ts, actor_type, actor_id, event_type, source) VALUES
                ('2026-07-14T01:00:00Z', 'agent', 'local', 'marked', 'rust'),
                ('2026-07-14T02:00:00Z', 'agent', 'local', 'reinforced', 'rust')",
                    (),
                )
                .await;
        });
        assert!(insert_interest(&conn, "libsql internals", "user").is_ok());
        let interests = block_on(top_interests_with(&conn, 10)).unwrap_or_default();
        let mut found_registered = false;
        let mut found_derived = false;
        for item in interests {
            if item.topic == "libsql internals" {
                found_registered = true;
            }
            if item.topic == "rust" {
                found_derived = true;
            }
        }
        assert!(found_registered);
        assert!(found_derived);
        let capped = block_on(top_interests_with(&conn, 1)).unwrap_or_default();
        assert_eq!(capped.len(), 1);
        assert_eq!(capped[0].topic, "rust");
    }

    #[test]
    fn interest_density_counts_live_embedding_neighbours() {
        let conn = block_on(harness());
        block_on(async {
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute("INSERT INTO edges (source, target, embedding) VALUES ('a', 'b', vector32('[1.0,0.0]')), ('c', 'd', vector32('[0.0,1.0]'))", ()).await;
        });
        let density = block_on(interest_density_with(&conn, &[1.0, 0.0], 0.1)).unwrap_or(0);
        assert_eq!(density, 1);
    }

    #[test]
    fn sync_operation_waits_for_write_pass() {
        let write_guard = super::acquire_write_lock();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let (done_sender, done_receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = started_sender.send(());
            with_write_lock(|| {
                let _ = done_sender.send(());
            });
        });
        assert!(started_receiver.recv().is_ok());
        assert!(
            done_receiver
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err()
        );
        drop(write_guard);
        assert!(
            done_receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .is_ok()
        );
        assert!(handle.join().is_ok());
    }

    async fn harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let _ = conn.execute(EDGES_DDL, ()).await;
        conn
    }

    fn epistemic_edge(source: &str, provenance: &str, confidence: f64) -> Edge {
        Edge {
            source: source.to_string(),
            target: "target".to_string(),
            content: "belief".to_string(),
            stability: 0.5,
            confidence,
            importance: 0.5,
            subject: false,
            date: "2026-07-16".to_string(),
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: provenance.to_string(),
            ceiling: None,
        }
    }

    #[test]
    fn writes_and_confidence_updates_obey_provenance_ceiling() {
        let conn = block_on(harness());
        let dreamed = epistemic_edge("dream", "dreamed", 0.9);
        let confirmed = epistemic_edge("user", "user-confirmed", 0.9);
        assert!(write_edge(&conn, &dreamed, "").is_ok());
        assert!(write_edge(&conn, &confirmed, "").is_ok());
        let Ok(Some(dreamed_stored)) = super::load_edge(&conn, "dream", "target") else {
            unreachable!("dreamed edge loads")
        };
        let Ok(Some(confirmed_stored)) = super::load_edge(&conn, "user", "target") else {
            unreachable!("confirmed edge loads")
        };
        assert_eq!(dreamed_stored.confidence, 0.25);
        assert_eq!(confirmed_stored.confidence, 0.9);

        assert!(update_edge_confidence(&conn, "dream", "target", 0.8).unwrap_or(false));
        let Ok(Some(updated)) = super::load_edge(&conn, "dream", "target") else {
            unreachable!("updated edge loads")
        };
        assert_eq!(updated.confidence, 0.25);
    }

    #[test]
    fn writes_obey_a_derived_edges_own_ceiling() {
        let conn = block_on(harness());
        let mut edge = epistemic_edge("synthesis", "synthesized", 0.9);
        edge.ceiling = Some(0.72);
        assert!(write_edge(&conn, &edge, "").is_ok());
        let Ok(Some(stored)) = super::load_edge(&conn, "synthesis", "target") else {
            unreachable!("derived edge loads")
        };
        assert_eq!(stored.confidence, 0.72);
        assert_eq!(stored.ceiling, Some(0.72));
    }

    #[test]
    fn ceiling_migration_is_idempotent_and_leaves_existing_edges_null() {
        let conn = block_on(migration_harness());
        block_on(add_edge_ceiling_column(&conn));
        block_on(add_edge_ceiling_column(&conn));
        let result = block_on(async {
            let mut rows = conn
                .query("SELECT ceiling FROM edges WHERE source = 'dream'", ())
                .await?;
            rows.next().await
        });
        assert!(matches!(
            result,
            Ok(Some(row)) if matches!(row.get_value(0), Ok(turso::Value::Null))
        ));
    }

    #[test]
    fn epistemics_backfill_classifies_clamps_and_is_idempotent() {
        let conn = block_on(migration_harness());
        let Ok(first) = block_on(backfill_edge_epistemics(&conn)) else {
            unreachable!("backfill succeeds")
        };
        assert_eq!(first.total, 4);
        assert_eq!(first.dreamed, 1);
        assert_eq!(first.web, 1);
        assert_eq!(first.user, 1);
        assert_eq!(first.legacy, 1);
        assert_eq!(first.clamped, 3);

        block_on(async {
            let _ = conn
                .execute(
                    "UPDATE edges SET confidence = 0.1 WHERE source = 'dream'",
                    (),
                )
                .await;
        });
        let Ok(second) = block_on(backfill_edge_epistemics(&conn)) else {
            unreachable!("second backfill succeeds")
        };
        assert_eq!(second.total, 0);
        let values = block_on(backfilled_values(&conn));
        assert_eq!(
            values,
            vec![
                ("dream".to_string(), "dreamed".to_string(), 0.1),
                ("news".to_string(), "web-surfaced".to_string(), 0.5),
                ("turn".to_string(), "user-confirmed".to_string(), 0.9),
                ("unknown".to_string(), "legacy".to_string(), 0.5),
            ]
        );
    }

    async fn migration_harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let _ = conn
            .execute(
                "CREATE TABLE edges (
                source TEXT NOT NULL, target TEXT NOT NULL, confidence REAL NOT NULL,
                origin_session TEXT NOT NULL DEFAULT '',
                provenance TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (source, target)
            )",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, confidence, origin_session) VALUES
                ('dream', 'x', 0.9, 'dream:1'),
                ('news', 'x', 0.9, 'news:2026-07-16'),
                ('turn', 'x', 0.9, '2026-07-16T12-00-00'),
                ('unknown', 'x', 0.9, '')",
                (),
            )
            .await;
        conn
    }

    async fn backfilled_values(conn: &Connection) -> Vec<(String, String, f64)> {
        let Ok(mut rows) = conn
            .query(
                "SELECT source, provenance, confidence FROM edges ORDER BY source",
                (),
            )
            .await
        else {
            unreachable!("query succeeds")
        };
        let mut values = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let (Ok(source), Ok(provenance), Ok(confidence)) =
                (row.get_value(0), row.get_value(1), row.get_value(2))
            else {
                unreachable!("backfill values read")
            };
            values.push((
                super::text(source),
                super::text(provenance),
                super::real(confidence),
            ));
        }
        values
    }

    async fn game_harness() -> Connection {
        let conn = harness().await;
        let _ = conn.execute(NODES_DDL, ()).await;
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type, body) VALUES ('game-chess', 'game', 'the rules')",
                (),
            )
            .await;
        // Episodic hub → game link (excluded from the semantic knowledge load).
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, memory_type, origin_session) VALUES ('task-game-1', 'game-chess', 'this session plays chess', 'episodic', 'game-1')",
                (),
            )
            .await;
        conn
    }

    async fn loop_harness() -> Connection {
        let conn = harness().await;
        let _ = conn.execute(NODES_DDL, ()).await;
        conn
    }

    async fn policy_harness() -> Connection {
        let conn = harness().await;
        let _ = conn.execute(EVENTS_DDL, ()).await;
        conn
    }

    async fn insert_policy_edge(conn: &Connection, details: (&str, &str, &str, &str)) {
        let (source, target, vector, created_at) = details;
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, embedding) VALUES (?1, ?2, vector32(?3))",
                params![source, target, vector],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO events (ts, event_type, source, target) VALUES (?1, 'created', ?2, ?3)",
                params![created_at, source, target],
            )
            .await;
    }

    async fn insert_loop(
        conn: &Connection,
        loop_id: &str,
        anchor_id: &str,
        details: (&str, &str, &str, Option<&str>),
    ) {
        let (intention, due, edge_date, expired_at) = details;
        let _ = conn
            .execute(
                "INSERT OR IGNORE INTO nodes (id, node_type) VALUES (?1, 'entity')",
                params![anchor_id],
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, 'loop', ?2, ?3)",
                params![loop_id, intention, due],
            )
            .await;
        let content = format!("open loop: {intention}");
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, stability, confidence, importance, memory_type, date, expired_at) VALUES (?1, ?2, ?3, 0.9, 0.9, 1.0, 'semantic', ?4, ?5)",
                params![loop_id, anchor_id, content.as_str(), edge_date, expired_at],
            )
            .await;
    }

    async fn seed_anchor_loop_cases(conn: &Connection) {
        let cases = [
            (
                "loop-context",
                "home-server",
                ("ask about the service", "", "2026-07-02", None),
            ),
            (
                "loop-due",
                "file-transfer",
                (
                    "check whether the file arrived",
                    "2026-07-10",
                    "2026-07-01",
                    None,
                ),
            ),
            (
                "loop-future",
                "other-topic",
                ("ask about the future", "2026-07-12", "2026-07-03", None),
            ),
            (
                "loop-closed",
                "home-server",
                (
                    "closed intention",
                    "",
                    "2026-06-30",
                    Some("2026-07-09T12:00:00Z"),
                ),
            ),
        ];
        for (loop_id, anchor_id, details) in cases {
            insert_loop(conn, loop_id, anchor_id, details).await;
        }
    }

    async fn seed_due_loop_cases(conn: &Connection) {
        let cases = [
            (
                "loop-past",
                "past-topic",
                ("past intention", "2026-07-10", "2026-07-01", None),
            ),
            (
                "loop-today",
                "today-topic",
                ("today intention", "2026-07-11", "2026-07-02", None),
            ),
            (
                "loop-context",
                "context-topic",
                ("context only", "", "2026-07-03", None),
            ),
            (
                "loop-future",
                "future-topic",
                ("future intention", "2026-07-12", "2026-07-04", None),
            ),
            (
                "loop-closed",
                "closed-topic",
                (
                    "closed intention",
                    "2026-07-09",
                    "2026-07-05",
                    Some("2026-07-10T00:00:00Z"),
                ),
            ),
        ];
        for (loop_id, anchor_id, details) in cases {
            insert_loop(conn, loop_id, anchor_id, details).await;
        }
    }

    async fn insert_game_edge(
        conn: &Connection,
        target: &str,
        confidence: f64,
        expired: Option<&str>,
    ) {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, confidence, memory_type, expired_at) VALUES ('game-chess', ?1, ?1, ?2, 'semantic', ?3)",
                params![target, confidence, expired],
            )
            .await;
    }

    #[test]
    fn session_game_returns_the_linked_definition_and_none_without_a_link() {
        let conn = block_on(game_harness());
        let Ok(Some((game_id, node))) = load_session_game(&conn, "task-game-1") else {
            unreachable!("game is linked")
        };
        assert_eq!(game_id, "game-chess");
        assert_eq!(node.body, "the rules");
        let Ok(missing) = load_session_game(&conn, "task-game-2") else {
            unreachable!("load succeeds")
        };
        assert!(missing.is_none());
    }

    #[test]
    fn game_edges_load_highest_confidence_first_and_exclude_expired() {
        let conn = block_on(game_harness());
        block_on(insert_game_edge(&conn, "rule-low", 0.4, None));
        block_on(insert_game_edge(&conn, "rule-high", 0.9, None));
        block_on(insert_game_edge(
            &conn,
            "rule-gone",
            0.95,
            Some("2026-01-01"),
        ));
        let Ok(edges) = load_game_edges(&conn, "game-chess", 60) else {
            unreachable!("edges load")
        };
        assert_eq!(edges.len(), 2, "expired excluded, episodic link excluded");
        assert_eq!(edges[0].target, "rule-high");
        assert_eq!(edges[1].target, "rule-low");
    }

    #[test]
    fn game_edges_cap_keeps_the_highest_confidence() {
        let conn = block_on(game_harness());
        block_on(insert_game_edge(&conn, "rule-a", 0.3, None));
        block_on(insert_game_edge(&conn, "rule-b", 0.9, None));
        let Ok(edges) = load_game_edges(&conn, "game-chess", 1) else {
            unreachable!("edges load")
        };
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, "rule-b");
    }

    #[test]
    fn loop_loader_matches_anchors_includes_due_and_excludes_closed() {
        let conn = block_on(loop_harness());
        block_on(seed_anchor_loop_cases(&conn));

        let anchors = vec!["home-server".to_string()];
        let Ok(loops) = load_loops_for_anchors(&conn, &anchors, "2026-07-11", 5) else {
            unreachable!("loops load")
        };
        assert_eq!(loops.len(), 2);
        assert_eq!(loops[0].source, "loop-due", "oldest loop comes first");
        assert_eq!(loops[1].source, "loop-context");
    }

    #[test]
    fn loop_loader_cap_keeps_oldest_edge_dates_first() {
        let conn = block_on(loop_harness());
        block_on(insert_loop(
            &conn,
            "loop-third",
            "topic-third",
            ("third", "2026-07-08", "2026-07-10", None),
        ));
        block_on(insert_loop(
            &conn,
            "loop-first",
            "topic-first",
            ("first", "2026-07-10", "2026-07-08", None),
        ));
        block_on(insert_loop(
            &conn,
            "loop-second",
            "topic-second",
            ("second", "2026-07-09", "2026-07-09", None),
        ));

        let Ok(loops) = load_loops_for_anchors(&conn, &[], "2026-07-11", 2) else {
            unreachable!("loops load")
        };
        assert_eq!(loops.len(), 2);
        assert_eq!(loops[0].source, "loop-first");
        assert_eq!(loops[1].source, "loop-second");
    }

    #[test]
    fn loop_loader_preserves_insertion_order_for_same_day() {
        let conn = block_on(loop_harness());
        block_on(insert_loop(
            &conn,
            "z-first",
            "topic-first",
            ("first", "2026-07-10", "2026-07-10", None),
        ));
        block_on(insert_loop(
            &conn,
            "a-second",
            "topic-second",
            ("second", "2026-07-10", "2026-07-10", None),
        ));
        let Ok(loops) = load_loops_for_anchors(&conn, &[], "2026-07-11", 1) else {
            unreachable!("loops load")
        };
        assert_eq!(loops[0].source, "z-first");
    }

    #[test]
    fn due_loop_loader_formats_only_live_due_loops() {
        let conn = block_on(loop_harness());
        block_on(seed_due_loop_cases(&conn));

        let Ok(loops) = load_due_loops(&conn, "2026-07-11") else {
            unreachable!("due loops load")
        };
        let mut displays = Vec::new();
        for open_loop in &loops {
            displays.push(open_loop.display());
        }
        assert_eq!(
            displays,
            vec![
                "past intention (due 2026-07-10)",
                "today intention (due 2026-07-11)"
            ]
        );
    }

    #[test]
    fn ordinary_retrieval_excludes_loop_anchor_edges() {
        let conn = block_on(loop_harness());
        block_on(insert_loop(
            &conn,
            "loop-reminder",
            "topic",
            ("ask about the service", "", "2026-07-10", None),
        ));
        assert!(insert_interest(&conn, "libsql internals", "user").is_ok());
        block_on(async {
            let _ = conn
                .execute(
                    "INSERT INTO edges (source, target, content) VALUES
                     ('ordinary', 'topic', 'ordinary belief'),
                     ('libsql-internals', 'topic', 'standing interest edge')",
                    (),
                )
                .await;
        });
        // Standing-node edges ARE embedded for their dedicated similarity
        // paths…
        let pending = block_on(load_unembedded_edges_async(&conn));
        assert_eq!(pending.len(), 3);
        // …but stay excluded from ranked retrieval.
        let semantic = block_on(load_all_edges_semantic_async(&conn));
        assert_eq!(semantic.len(), 1);
        assert_eq!(semantic[0].0, "ordinary");
    }

    #[test]
    fn recent_attention_counts_only_candidate_events_inside_window() {
        let conn = block_on(policy_harness());
        block_on(async {
            let _ = conn
                .execute(
                    "INSERT INTO events (ts, event_type, source, target) VALUES
                     (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'reinforced', 'a', 'b'),
                     (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'marked', 'a', 'b'),
                     (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'retrieved', 'a', 'b'),
                     (strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-8 days'), 'reinforced', 'a', 'b'),
                     (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'marked', 'other', 'edge')",
                    (),
                )
                .await;
        });
        let candidates = vec![("a".to_string(), "b".to_string())];
        let Ok(counts) = block_on(load_recent_attention_counts_async(&conn, &candidates, 7)) else {
            unreachable!("attention counts load")
        };
        assert_eq!(counts.get(&("a".to_string(), "b".to_string())), Some(&2));
        assert_eq!(counts.len(), 1);
    }

    #[test]
    fn incubation_counts_only_new_near_live_edges() {
        let conn = block_on(policy_harness());
        let note_created = "2026-07-01T12:00:00Z";
        let cases = [
            ("near-1", "topic-1", "[1.0,0.0]", "2026-07-02T12:00:00Z"),
            ("near-2", "topic-2", "[0.99,0.01]", "2026-07-03T12:00:00Z"),
            ("near-3", "topic-3", "[0.98,0.02]", "2026-07-04T12:00:00Z"),
            ("old-near", "topic-4", "[1.0,0.0]", "2026-06-30T12:00:00Z"),
            ("new-far", "topic-5", "[0.0,1.0]", "2026-07-05T12:00:00Z"),
            (
                "expired-near",
                "topic-6",
                "[1.0,0.0]",
                "2026-07-06T12:00:00Z",
            ),
        ];
        for details in cases {
            block_on(insert_policy_edge(&conn, details));
        }
        block_on(async {
            let _ = conn
                .execute(
                    "UPDATE edges SET expired_at = '2026-07-07T12:00:00Z' WHERE source = 'expired-near'",
                    (),
                )
                .await;
        });
        let Ok(count) = block_on(count_new_embedding_neighbours_async(
            &conn,
            &[1.0, 0.0],
            note_created,
            0.45,
        )) else {
            unreachable!("incubation neighbours count")
        };
        assert_eq!(count, 3);
    }

    async fn insert_ply(
        conn: &Connection,
        hub: &str,
        ply: usize,
        session: &str,
        expired: Option<&str>,
    ) {
        let target = format!("{hub}-ply-{ply:03}");
        let content = format!("ply {ply} (assistant): move{ply}");
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, memory_type, origin_session, expired_at) VALUES (?1, ?2, ?3, 'episodic', ?4, ?5)",
                params![hub, target.as_str(), content.as_str(), session, expired],
            )
            .await;
    }

    #[test]
    fn chain_returns_plies_in_order_for_this_session_only() {
        let conn = block_on(harness());
        block_on(insert_ply(&conn, "task-game-1", 2, "game-1", None));
        block_on(insert_ply(&conn, "task-game-1", 1, "game-1", None));
        // A different session's hub must never leak in.
        block_on(insert_ply(&conn, "task-game-2", 1, "game-2", None));

        let Ok(chain) = load_task_chain(&conn, "game-1", 240) else {
            unreachable!("chain loads")
        };
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].target, "task-game-1-ply-001");
        assert_eq!(chain[1].target, "task-game-1-ply-002");
    }

    #[test]
    fn chain_excludes_expired_plies() {
        let conn = block_on(harness());
        block_on(insert_ply(&conn, "task-game-1", 1, "game-1", None));
        block_on(insert_ply(
            &conn,
            "task-game-1",
            2,
            "game-1",
            Some("2026-01-01"),
        ));

        let Ok(chain) = load_task_chain(&conn, "game-1", 240) else {
            unreachable!("chain loads")
        };
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].target, "task-game-1-ply-001");
    }

    #[test]
    fn cap_keeps_the_most_recent_plies_in_order() {
        let conn = block_on(harness());
        for ply in 1..=5 {
            block_on(insert_ply(&conn, "task-game-1", ply, "game-1", None));
        }
        let Ok(chain) = load_task_chain(&conn, "game-1", 2) else {
            unreachable!("chain loads")
        };
        assert_eq!(chain.len(), 2);
        // Newest two plies, still in ascending order.
        assert_eq!(chain[0].target, "task-game-1-ply-004");
        assert_eq!(chain[1].target, "task-game-1-ply-005");
    }
}
