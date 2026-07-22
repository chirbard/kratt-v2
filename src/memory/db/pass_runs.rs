use std::error::Error;

use turso::{Connection, params};

use super::{acquire_write_lock, block_on, int, open_connection, real, text, timestamp_now};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassRun {
    pub run_id: i64,
    pub pass_type: String,
    pub started_at: String,
    pub finished_at: String,
    pub outcome: String,
    pub reason_class: String,
    pub items_added: usize,
}

pub fn start_pass_run(pass_type: &str) -> Result<i64, Box<dyn Error + Send + Sync>> {
    validate_pass_type(pass_type)?;
    let _write_guard = acquire_write_lock();
    let conn = open_connection()?;
    block_on(start_pass_run_async(&conn, pass_type))
}

pub fn finish_pass_run(
    run_id: i64,
    outcome: &str,
    reason_class: &str,
    items_added: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    validate_pass_outcome(outcome)?;
    validate_reason_class(reason_class)?;
    let _write_guard = acquire_write_lock();
    let conn = open_connection()?;
    block_on(finish_pass_run_async(
        &conn,
        run_id,
        outcome,
        reason_class,
        items_added,
    ))
}

fn validate_pass_type(pass_type: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    if matches!(pass_type, "dream" | "exploration" | "news" | "ingestion") {
        return Ok(());
    }
    Err(format!("invalid pass type: {pass_type}").into())
}

fn validate_pass_outcome(outcome: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    if matches!(outcome, "ok" | "failed" | "skipped") {
        return Ok(());
    }
    Err(format!("invalid pass outcome: {outcome}").into())
}

fn validate_reason_class(reason_class: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    if matches!(
        reason_class,
        "none"
            | "context_length"
            | "provider_error"
            | "database_error"
            | "no_material"
            | "not_eligible"
            | "unavailable"
            | "interrupted"
    ) {
        return Ok(());
    }
    Err(format!("invalid pass reason class: {reason_class}").into())
}

pub(crate) async fn start_pass_run_async(
    conn: &Connection,
    pass_type: &str,
) -> Result<i64, Box<dyn Error + Send + Sync>> {
    validate_pass_type(pass_type)?;
    conn.execute(
        "INSERT INTO pass_runs (pass_type, started_at) VALUES (?1, ?2)",
        params![pass_type, timestamp_now()],
    )
    .await?;
    let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
    let Some(row) = rows.next().await? else {
        return Err("pass run insert did not return an id".into());
    };
    Ok(int(row.get_value(0)?))
}

pub(crate) async fn finish_pass_run_async(
    conn: &Connection,
    run_id: i64,
    outcome: &str,
    reason_class: &str,
    items_added: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    validate_pass_outcome(outcome)?;
    validate_reason_class(reason_class)?;
    conn.execute(
        "UPDATE pass_runs SET finished_at = ?1, outcome = ?2, reason_class = ?3, items_added = ?4 WHERE id = ?5",
        params![
            timestamp_now(),
            outcome,
            reason_class,
            items_added as i64,
            run_id
        ],
    )
    .await?;
    Ok(())
}

pub(crate) async fn pass_runs_in_window_async(
    conn: &Connection,
    pass_type: &str,
    from: &str,
    until: &str,
) -> Result<Vec<PassRun>, Box<dyn Error + Send + Sync>> {
    validate_pass_type(pass_type)?;
    let mut rows = conn
        .query(
            "SELECT id, pass_type, started_at, finished_at, outcome, reason_class, items_added
             FROM pass_runs
             WHERE pass_type = ?1 AND started_at >= ?2 AND started_at < ?3
             ORDER BY id",
            params![pass_type, from, until],
        )
        .await?;
    collect_pass_runs(&mut rows).await
}

pub(crate) async fn last_successful_pass_async(
    conn: &Connection,
    pass_type: &str,
) -> Result<Option<PassRun>, Box<dyn Error + Send + Sync>> {
    validate_pass_type(pass_type)?;
    let mut rows = conn
        .query(
            "SELECT id, pass_type, started_at, finished_at, outcome, reason_class, items_added
             FROM pass_runs
             WHERE pass_type = ?1 AND outcome = 'ok'
             ORDER BY id DESC LIMIT 1",
            params![pass_type],
        )
        .await?;
    let mut runs = collect_pass_runs(&mut rows).await?;
    Ok(runs.pop())
}

pub(crate) async fn consecutive_failures_async(
    conn: &Connection,
    pass_type: &str,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    validate_pass_type(pass_type)?;
    let mut rows = conn
        .query(
            "SELECT
                 MAX(CASE WHEN outcome = 'ok' THEN 1 ELSE 0 END),
                 MAX(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END)
             FROM pass_runs
             WHERE pass_type = ?1
             GROUP BY substr(started_at, 1, 10)
             ORDER BY substr(started_at, 1, 10) DESC",
            params![pass_type],
        )
        .await?;
    let mut failures = 0usize;
    while let Some(row) = rows.next().await? {
        let had_success = real(row.get_value(0)?) > 0.0;
        let had_failure = real(row.get_value(1)?) > 0.0;
        if had_success {
            break;
        }
        if had_failure {
            failures += 1;
        }
    }
    Ok(failures)
}

async fn collect_pass_runs(
    rows: &mut turso::Rows,
) -> Result<Vec<PassRun>, Box<dyn Error + Send + Sync>> {
    let mut runs = Vec::new();
    while let Some(row) = rows.next().await? {
        runs.push(PassRun {
            run_id: int(row.get_value(0)?),
            pass_type: text(row.get_value(1)?),
            started_at: text(row.get_value(2)?),
            finished_at: text(row.get_value(3)?),
            outcome: text(row.get_value(4)?),
            reason_class: text(row.get_value(5)?),
            items_added: real(row.get_value(6)?) as usize,
        });
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::{
        consecutive_failures_async, finish_pass_run_async, last_successful_pass_async,
        pass_runs_in_window_async, start_pass_run_async,
    };
    use crate::memory::db::block_on;
    use turso::{Builder, Connection};

    async fn pass_run_harness() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        let Ok(()) = super::super::create_pass_run_table(&conn).await else {
            unreachable!("pass run table creates")
        };
        conn
    }

    #[test]
    fn pass_run_records_failure_reason_and_items() {
        let conn = block_on(pass_run_harness());
        let Ok(run_id) = block_on(start_pass_run_async(&conn, "dream")) else {
            unreachable!("pass starts")
        };
        assert!(
            block_on(finish_pass_run_async(
                &conn,
                run_id,
                "failed",
                "context_length",
                0,
            ))
            .is_ok()
        );
        let runs = block_on(pass_runs_in_window_async(&conn, "dream", "", "z")).unwrap_or_default();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].outcome, "failed");
        assert_eq!(runs[0].reason_class, "context_length");
    }

    #[test]
    fn pass_failure_streak_resets_after_success() {
        let conn = block_on(pass_run_harness());
        block_on(async {
            let _ = conn.execute(
                "INSERT INTO pass_runs (pass_type, started_at, finished_at, outcome, reason_class) VALUES
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-3 days'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-3 days'), 'ok', 'none'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 days'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 days'), 'failed', 'provider_error'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 day'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 day'), 'failed', 'provider_error'),
                 ('dream', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'failed', 'context_length')",
                (),
            ).await;
        });
        assert_eq!(
            block_on(consecutive_failures_async(&conn, "dream")).unwrap_or_default(),
            3
        );
        let last = block_on(last_successful_pass_async(&conn, "dream")).unwrap_or_default();
        assert!(last.is_some());
        let Ok(run_id) = block_on(start_pass_run_async(&conn, "dream")) else {
            unreachable!("success starts")
        };
        assert!(block_on(finish_pass_run_async(&conn, run_id, "ok", "none", 4)).is_ok());
        assert_eq!(
            block_on(consecutive_failures_async(&conn, "dream")).unwrap_or_default(),
            0
        );
        let latest = block_on(last_successful_pass_async(&conn, "dream")).unwrap_or_default();
        let items_added = match latest {
            Some(run) => Some(run.items_added),
            None => None,
        };
        assert_eq!(items_added, Some(4));
    }
}
