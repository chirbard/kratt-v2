use turso::{Builder, Connection};

pub(super) const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, last_decayed_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))";
pub(super) const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
pub(super) const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
pub(super) const FINDINGS_DDL: &str = "CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')";
pub(super) const PING_POLICY_DDL: &str = "CREATE TABLE ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')";

pub(super) async fn harness() -> Connection {
    let Ok(database) = Builder::new_local(":memory:").build().await else {
        unreachable!("in-memory database builds")
    };
    let Ok(conn) = database.connect() else {
        unreachable!("in-memory database connects")
    };
    let _ = conn.execute(NODES_DDL, ()).await;
    let _ = conn.execute(EDGES_DDL, ()).await;
    conn
}
