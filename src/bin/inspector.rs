use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

use tokio::runtime::Runtime;
use turso::{Builder, Connection, Database, Value};

const DB_PATH: &str = "kratt-data/kratt.db";
const SYNCED_DB_PATH: &str = "kratt-data/kratt-synced.db";
const ADDR: &str = "127.0.0.1:7070";
const PAGE_SIZE: usize = 100;
const TABLES: &[&str] = &[
    "nodes",
    "edges",
    "events",
    "edge_actor_views",
    "actor_positions",
    "actor_position_history",
    "sessions",
    "messages",
    "meta",
];

#[allow(clippy::expect_used)]
fn main() {
    // Same cwd pinning as the main binary: data paths are relative.
    let _ = std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"));
    let rt = Runtime::new().expect("tokio runtime");
    // Prefer the Turso-Cloud-synced file when it exists (read-only browsing
    // of the file the app actually uses in synced mode).
    let path = if std::path::Path::new(SYNCED_DB_PATH).exists() {
        SYNCED_DB_PATH
    } else {
        DB_PATH
    };
    let db = rt.block_on(async {
        Builder::new_local(path)
            .experimental_index_method(true)
            .build()
            .await
            .expect("open db")
    });
    let listener = TcpListener::bind(ADDR).expect("bind");
    println!("inspector: http://{ADDR} ({path})");
    loop {
        if let Ok((stream, _)) = listener.accept() {
            handle(&rt, &db, stream);
        }
    }
}

fn handle(rt: &Runtime, db: &Database, mut stream: TcpStream) {
    let request = read_request(&stream);
    let path = request_path(&request);
    let (clean_path, offset) = split_path_offset(path);
    let conn = match db.connect() {
        Ok(conn) => conn,
        Err(err) => {
            send(&mut stream, 500, &format!("db error: {err}"));
            return;
        }
    };
    let body = rt.block_on(route(&conn, clean_path, offset));
    send(&mut stream, 200, &body);
}

fn read_request(stream: &TcpStream) -> String {
    let reader = BufReader::new(stream);
    let mut lines = Vec::new();
    for line in reader.lines() {
        match line {
            Ok(ln) if ln.is_empty() => break,
            Ok(ln) => lines.push(ln),
            Err(_) => break,
        }
    }
    lines.join("\n")
}

fn request_path(request: &str) -> &str {
    let first = match request.lines().next() {
        Some(line) => line,
        None => return "/",
    };
    let mut parts = first.splitn(3, ' ');
    parts.next();
    match parts.next() {
        Some(path) => path,
        None => "/",
    }
}

fn split_path_offset(path: &str) -> (&str, usize) {
    match path.split_once('?') {
        Some((clean, query)) => (clean, parse_offset_param(query)),
        None => (path, 0),
    }
}

fn parse_offset_param(query: &str) -> usize {
    for param in query.split('&') {
        if let Some(val) = param.strip_prefix("offset=")
            && let Ok(n) = val.parse::<usize>()
        {
            return n;
        }
    }
    0
}

async fn route(conn: &Connection, path: &str, offset: usize) -> String {
    if path == "/" {
        return render_index(conn).await;
    }
    if let Some(table) = path.strip_prefix("/table/") {
        return render_table(conn, table, offset).await;
    }
    page("not found", "<p>not found</p>")
}

async fn render_index(conn: &Connection) -> String {
    let mut rows = String::new();
    for table in TABLES {
        let count = row_count(conn, table).await;
        rows.push_str(&format!(
            "<tr><td><a href='/table/{table}'>{table}</a></td><td class='num'>{count}</td></tr>\n"
        ));
    }
    let body = format!("<table><tr><th>table</th><th>rows</th></tr>\n{rows}</table>");
    page("tables", &body)
}

async fn row_count(conn: &Connection, table: &str) -> i64 {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let mut rows = match conn.query(&sql, ()).await {
        Ok(rows) => rows,
        Err(_) => return 0,
    };
    match rows.next().await {
        Ok(Some(row)) => match row.get_value(0) {
            Ok(Value::Integer(n)) => n,
            _ => 0,
        },
        _ => 0,
    }
}

async fn render_table(conn: &Connection, table: &str, offset: usize) -> String {
    let columns = table_columns(conn, table).await;
    if columns.is_empty() {
        return page("error", "<p>table not found or has no columns</p>");
    }
    let col_list = columns.join(", ");
    let sql = format!("SELECT {col_list} FROM {table} LIMIT {PAGE_SIZE} OFFSET {offset}");
    let mut result = match conn.query(&sql, ()).await {
        Ok(result) => result,
        Err(err) => return page(table, &format!("<p>query error: {err}</p>")),
    };
    let header = build_header(&columns);
    let (body_rows, row_count) = build_rows(&mut result, columns.len()).await;
    let nav = pagination_nav(table, offset, row_count);
    let body = format!("<table>{header}{body_rows}</table>{nav}");
    page(table, &body)
}

async fn table_columns(conn: &Connection, table: &str) -> Vec<String> {
    let sql = format!("PRAGMA table_info({table})");
    let mut rows = match conn.query(&sql, ()).await {
        Ok(rows) => rows,
        Err(_) => return Vec::new(),
    };
    let mut cols = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        if let Ok(Value::Text(name)) = row.get_value(1) {
            cols.push(name);
        }
    }
    cols
}

fn build_header(columns: &[String]) -> String {
    let mut header = String::from("<tr>");
    for col in columns {
        header.push_str(&format!("<th>{col}</th>"));
    }
    header.push_str("</tr>\n");
    header
}

async fn build_rows(result: &mut turso::Rows, col_count: usize) -> (String, usize) {
    let mut html = String::new();
    let mut count = 0usize;
    while let Ok(Some(row)) = result.next().await {
        html.push_str("<tr>");
        for idx in 0..col_count {
            let val = match row.get_value(idx) {
                Ok(val) => format_value(val),
                Err(_) => String::new(),
            };
            let escaped = escape_html(&val);
            html.push_str(&format!("<td title='{escaped}'>{escaped}</td>"));
        }
        html.push_str("</tr>\n");
        count += 1;
    }
    (html, count)
}

fn format_value(val: Value) -> String {
    match val {
        Value::Text(s) => s,
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => format!("{f:.3}"),
        Value::Null => String::new(),
        Value::Blob(b) => format!("<blob {} bytes>", b.len()),
    }
}

fn escape_html(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

fn pagination_nav(table: &str, offset: usize, row_count: usize) -> String {
    let mut nav = String::from("<p class='nav'>");
    if offset > 0 {
        let prev = offset.saturating_sub(PAGE_SIZE);
        nav.push_str(&format!(
            "<a href='/table/{table}?offset={prev}'>&larr; prev</a> "
        ));
    }
    if row_count == PAGE_SIZE {
        let next = offset + PAGE_SIZE;
        nav.push_str(&format!(
            "<a href='/table/{table}?offset={next}'>next &rarr;</a>"
        ));
    }
    nav.push_str("</p>");
    nav
}

fn send(stream: &mut TcpStream, status: u16, body: &str) {
    let status_text = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn page(title: &str, content: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>kratt / {title}</title><style>
body{{font-family:monospace;max-width:1400px;margin:20px auto;background:#111;color:#ccc;padding:0 20px}}
a{{color:#7af;text-decoration:none}}a:hover{{text-decoration:underline}}
h1{{font-size:1rem;color:#555;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;font-size:0.82rem}}
th{{background:#1e1e1e;padding:6px 12px;text-align:left;position:sticky;top:0;border-bottom:1px solid #333}}
td{{padding:4px 12px;border-bottom:1px solid #1a1a1a;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
tr:hover td{{background:#161616}}
td.num,th.num{{text-align:right}}
p.nav{{color:#666;margin-top:12px}}
</style></head>
<body><h1><a href="/">kratt inspector</a> / {title}</h1>
{content}
</body></html>"#
    )
}
