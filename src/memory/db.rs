use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::future::Future;
use std::sync::OnceLock;

use tokio::runtime::Runtime;
use turso::{Builder, Connection, Database, Value, params};

use super::graph::{Edge, Graph, Node, load_from_toon, normalize_id};

const DATA_DIR: &str = "kratt-data";
const DB_PATH: &str = "kratt-data/kratt.db";
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
    Option<Vec<f64>>,
);

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

#[allow(clippy::expect_used)]
fn database() -> &'static Database {
    static DATABASE: OnceLock<Database> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors
    DATABASE.get_or_init(|| {
        runtime().block_on(async {
            std::fs::create_dir_all(DATA_DIR).expect("create data dir");
            let database = Builder::new_local(DB_PATH)
                .experimental_index_method(true)
                .build()
                .await
                .expect("open db");
            let conn = database.connect().expect("db connect");
            init_schema(&conn).await.expect("init schema");
            database
        })
    })
}

pub fn open_connection() -> Result<Connection, Box<dyn Error + Send + Sync>> {
    Ok(database().connect()?)
}

async fn init_schema(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    create_base_tables(conn).await?;
    create_history_tables(conn).await?;
    add_edge_columns(conn).await;
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
        "CREATE TABLE IF NOT EXISTS edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, PRIMARY KEY (source, target))",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')",
        (),
    )
    .await?;
    Ok(())
}

async fn create_history_tables(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
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
    // fts index has no IF NOT EXISTS in turso — try and ignore if already exists
    let _ = conn
        .execute(
            "CREATE INDEX fts_messages_content ON messages USING fts (content)",
            (),
        )
        .await;
    Ok(())
}

async fn add_edge_provenance_column(conn: &Connection) {
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
    add_edge_provenance_column(conn).await;
    add_edge_embedding_column(conn).await;
    add_edge_time_columns(conn).await;
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
    // event_type values: created, updated, pruned, injected, routed
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
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at FROM edges",
            (),
        )
        .await
    else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        let (Ok(src), Ok(tgt), Ok(content), Ok(stab), Ok(conf), Ok(imp), Ok(subj), Ok(date)) = (
            row.get_value(0),
            row.get_value(1),
            row.get_value(2),
            row.get_value(3),
            row.get_value(4),
            row.get_value(5),
            row.get_value(6),
            row.get_value(7),
        ) else {
            continue;
        };
        graph.edges.push(Edge {
            source: normalize_id(&text(src)),
            target: normalize_id(&text(tgt)),
            content: text(content),
            stability: real(stab),
            confidence: real(conf),
            importance: real(imp),
            subject: int(subj) != 0,
            date: text(date),
            occurred_at: optional_text(row.get_value(8).ok()),
            expired_at: optional_text(row.get_value(9).ok()),
        });
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

pub fn write_edge(
    conn: &Connection,
    edge: &Edge,
    source_message_id: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    runtime().block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                edge.source.as_str(),
                edge.target.as_str(),
                edge.content.as_str(),
                edge.stability,
                edge.confidence,
                edge.importance,
                edge.subject as i64,
                edge.date.as_str(),
                edge.occurred_at.as_deref(),
                edge.expired_at.as_deref(),
                source_message_id,
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

async fn load_edge_async(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<Option<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at FROM edges WHERE source = ?1 AND target = ?2 LIMIT 1",
            params![source, target],
        )
        .await?;
    if let Some(row) = rows.next().await? {
        return Ok(parse_edge_row(&row));
    }
    Ok(None)
}

async fn load_edges_for_node_async(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let mut rows = conn
        .query(
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at FROM edges WHERE source = ?1 OR target = ?1",
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

async fn load_unembedded_edges_async(conn: &Connection) -> Vec<(String, String, String)> {
    let Ok(mut rows) = conn
        .query(
            "SELECT source, target, content FROM edges WHERE expired_at IS NULL AND embedding IS NULL",
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

async fn query_edges_ann(
    conn: &Connection,
    json_str: &str,
    limit: i64,
) -> Result<Vec<SemanticEdgeRow>, Box<dyn Error + Send + Sync>> {
    execute_semantic_query(
        conn,
        "SELECT e.source, e.target, e.content, e.stability, e.confidence, e.importance, e.subject, e.date, e.occurred_at, e.expired_at, vector_extract(e.embedding) FROM vector_top_k('edges_vec', vector32(?1), ?2) AS vt JOIN edges AS e ON e.rowid = vt.id WHERE e.expired_at IS NULL",
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
        "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, vector_extract(embedding) FROM edges WHERE embedding IS NOT NULL AND expired_at IS NULL ORDER BY vector_distance_cos(embedding, vector32(?1)) LIMIT ?2",
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
    let embedding = parse_optional_vector32(row.get_value(10).ok());
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
            "SELECT source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, vector_extract(embedding) FROM edges WHERE expired_at IS NULL",
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
        "INSERT INTO edges (source, target, content, stability, confidence, importance, subject, date, occurred_at, expired_at, source_message_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            edge.source.as_str(),
            edge.target.as_str(),
            edge.content.as_str(),
            edge.stability,
            edge.confidence,
            edge.importance,
            edge.subject as i64,
            edge.date.as_str(),
            edge.occurred_at.as_deref(),
            edge.expired_at.as_deref(),
            "",
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
    runtime().block_on(search_messages_fts_async(&conn, &fts_query))
}

async fn search_messages_fts_async(conn: &Connection, fts_query: &str) -> String {
    let mut lines = Vec::new();
    let Ok(mut rows) = conn.query(
        "SELECT id, role, content, created_at FROM messages WHERE content MATCH ?1 ORDER BY created_at DESC LIMIT 10",
        params![fts_query],
    )
    .await else {
        return String::new();
    };
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

fn parse_edge_row(row: &turso::Row) -> Option<Edge> {
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
    })
}

fn optional_text(value: Option<Value>) -> Option<String> {
    match value? {
        Value::Text(string) => Some(string),
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

fn int(val: Value) -> i64 {
    match val {
        Value::Integer(int) => int,
        _ => 0,
    }
}
