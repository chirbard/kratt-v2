use std::error::Error;

use turso::{Connection, params};

use super::{
    DEFAULT_PING_MUTED_UNTIL, add_edge_columns, add_interest_node_columns,
    backfill_edge_epistemics, drop_node_agent_views_table,
};

pub(super) async fn init_schema(
    conn: &Connection,
    create_fts_index: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    create_base_tables(conn).await?;
    create_history_tables(conn, create_fts_index).await?;
    create_pass_run_table(conn).await?;
    add_interest_node_columns(conn).await;
    add_edge_columns(conn).await;
    let _ = backfill_edge_epistemics(conn).await?;
    add_ledger_tables(conn).await;
    drop_node_agent_views_table(conn).await;
    Ok(())
}

async fn create_base_tables(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))",
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

pub(super) async fn create_pass_run_table(
    conn: &Connection,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pass_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pass_type TEXT NOT NULL CHECK (pass_type IN ('dream', 'exploration', 'news', 'ingestion')),
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            outcome TEXT NOT NULL DEFAULT 'failed' CHECK (outcome IN ('ok', 'failed', 'skipped')),
            reason_class TEXT NOT NULL DEFAULT 'interrupted' CHECK (reason_class IN ('none', 'context_length', 'provider_error', 'database_error', 'no_material', 'not_eligible', 'unavailable', 'interrupted')),
            items_added INTEGER NOT NULL DEFAULT 0 CHECK (items_added >= 0)
        )",
        (),
    )
    .await?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pass_runs_type_id ON pass_runs (pass_type, id)",
        (),
    )
    .await?;
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

async fn add_ledger_tables(conn: &Connection) {
    add_events_table(conn).await;
    add_edge_actor_views_table(conn).await;
    add_actor_positions_table(conn).await;
    add_actor_position_history_table(conn).await;
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
