use std::error::Error;

use turso::{Connection, params};

use super::{int, real, text};

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

fn validate_pass_type(pass_type: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    if matches!(pass_type, "dream" | "exploration" | "news" | "ingestion") {
        return Ok(());
    }
    Err(format!("invalid pass type: {pass_type}").into())
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
#[path = "pass_runs_tests.rs"]
mod tests;
