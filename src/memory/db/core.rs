use std::error::Error;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use tokio::runtime::Runtime;
use turso::{Builder, Connection, Database};

use super::{
    backfill_edge_epistemics, init_schema, migrate_entity_bodies_matching_ids,
    migrate_legacy_if_empty, node_count,
};
use crate::paths;

#[allow(clippy::expect_used)]
pub(super) fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors.
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

pub(crate) fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    runtime().block_on(future)
}

// Turso sync and checkpoint operations mutate database-wide state and can
// dissolve a transaction on another connection. Every write pass and every
// sync operation must serialize on this lock.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn acquire_write_lock() -> MutexGuard<'static, ()> {
    match WRITE_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub(super) fn with_write_lock<Output>(operation: impl FnOnce() -> Output) -> Output {
    let _write_guard = acquire_write_lock();
    operation()
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
        let _ = migrate_entity_bodies_matching_ids(&conn).await?;
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

pub(super) fn recover_replica_once<Output>(
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
    migrate_entity_bodies_matching_ids(&conn)
        .await
        .expect("entity body migration");
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
    let _ = migrate_entity_bodies_matching_ids(&conn).await?;
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

#[cfg(test)]
mod tests {
    use super::recover_replica_once;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

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
}

// Explicit column lists: the legacy file gained columns via ALTERs in a
// different order than a fresh schema creates them, so SELECT * would
// scramble the mapping.
