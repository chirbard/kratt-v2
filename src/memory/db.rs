use std::collections::HashSet;
use std::error::Error;
use std::sync::OnceLock;

use tokio::runtime::Runtime;
use turso::{Builder, Connection, Database, Value, params};

use super::graph::{Edge, Graph, Node, load_from_toon, normalize_id};

const DATA_DIR: &str = "kratt-data";
const DB_PATH: &str = "kratt-data/kratt.db";

#[allow(clippy::expect_used)]
fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

#[allow(clippy::expect_used)]
fn database() -> &'static Database {
    static DATABASE: OnceLock<Database> = OnceLock::new();
    // OnceLock::get_or_init requires FnOnce() -> T — no way to propagate errors
    DATABASE.get_or_init(|| {
        runtime().block_on(async {
            std::fs::create_dir_all(DATA_DIR).expect("create data dir");
            let database = Builder::new_local(DB_PATH).build().await.expect("open db");
            let conn = database.connect().expect("db connect");
            init_schema(&conn).await.expect("init schema");
            database
        })
    })
}

fn connect() -> Result<Connection, Box<dyn Error + Send + Sync>> {
    Ok(database().connect()?)
}

async fn init_schema(conn: &Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, target))",
        (),
    ).await?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')",
        (),
    ).await?;
    Ok(())
}

pub fn db_init() -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = connect()?;
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
    let Ok(conn) = connect() else { return Graph::new() };
    runtime().block_on(load_graph_async(&conn))
}

async fn load_graph_async(conn: &Connection) -> Graph {
    let mut graph = Graph::new();
    load_nodes(conn, &mut graph).await;
    load_edges(conn, &mut graph).await;
    graph
}

async fn load_nodes(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn.query("SELECT id, node_type, body, date FROM nodes", ()).await else {
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
        graph.nodes.insert(text(node_id), Node { node_type: text(node_type), body: text(body), date: text(date) });
    }
}

async fn load_edges(conn: &Connection, graph: &mut Graph) {
    let Ok(mut rows) = conn.query(
        "SELECT source, target, content, stability, confidence, importance, subject, date FROM edges",
        (),
    ).await else {
        return;
    };
    while let Ok(Some(row)) = rows.next().await {
        let (Ok(src), Ok(tgt), Ok(content), Ok(stab), Ok(conf), Ok(imp), Ok(subj), Ok(date)) = (
            row.get_value(0), row.get_value(1), row.get_value(2),
            row.get_value(3), row.get_value(4), row.get_value(5),
            row.get_value(6), row.get_value(7),
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
        });
    }
}

pub fn save_graph(graph: &Graph) -> Result<(), Box<dyn Error + Send + Sync>> {
    let conn = connect()?;
    runtime().block_on(write_graph(&conn, graph))
}

async fn write_graph(conn: &Connection, graph: &Graph) -> Result<(), Box<dyn Error + Send + Sync>> {
    conn.execute("BEGIN", ()).await?;
    conn.execute("DELETE FROM nodes", ()).await?;
    conn.execute("DELETE FROM edges", ()).await?;
    for (node_id, node) in &graph.nodes {
        conn.execute(
            "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, ?2, ?3, ?4)",
            params![node_id.as_str(), node.node_type.as_str(), node.body.as_str(), node.date.as_str()],
        ).await?;
    }
    for edge in &graph.edges {
        conn.execute(
            "INSERT INTO edges (source, target, content, stability, confidence, importance, subject, date) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![edge.source.as_str(), edge.target.as_str(), edge.content.as_str(), edge.stability, edge.confidence, edge.importance, edge.subject as i64, edge.date.as_str()],
        ).await?;
    }
    conn.execute("COMMIT", ()).await?;
    Ok(())
}

pub fn load_last_decay() -> String {
    let Ok(conn) = connect() else { return String::new() };
    runtime().block_on(async {
        let Ok(mut rows) = conn.query("SELECT value FROM meta WHERE key = 'last_decay'", ()).await
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
    let Ok(conn) = connect() else { return };
    let _ = runtime().block_on(
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_decay', ?1)", params![date]),
    );
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
            lines.push(format!("+nodes[{node_id} | {} | {} | {}]", node.node_type, node.body, node.date));
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
            lines.push(format!("+edges[{} → {} | {} | conf={} imp={} stab={}]", edge.source, edge.target, edge.content, edge.confidence, edge.importance, edge.stability));
        }
    }
    for edge in &pre.edges {
        if !post_keys.contains(&(edge.source.as_str(), edge.target.as_str())) {
            lines.push(format!("-edges[{} → {}]", edge.source, edge.target));
        }
    }
}

fn text(val: Value) -> String {
    match val {
        Value::Text(string) => string,
        _ => String::new(),
    }
}

fn real(val: Value) -> f64 {
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
