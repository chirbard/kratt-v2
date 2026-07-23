use std::error::Error;

use turso::{Connection, Value, params};

use super::core::runtime;
use super::{open_connection, text};
use crate::memory::graph::normalize_id;

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
