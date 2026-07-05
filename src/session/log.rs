use crate::memory;
use crate::timeutil;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
struct SessionEntry {
    role: String,
    content: String,
    #[serde(rename = "ts")]
    timestamp: String,
}

#[derive(Clone)]
pub struct RecentEntry {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

pub fn new_session_path() -> PathBuf {
    let created_at = timestamp_now();
    let filename = created_at.replace(':', "-");
    let path = PathBuf::from("kratt-data")
        .join("sessions")
        .join(format!("{filename}.jsonl"));
    let session_id = session_id_from_path(&path);
    memory::register_session(&session_id, &created_at);
    path
}

pub fn append_entry(path: &Path, role: &str, content: &str) -> String {
    let created_at = timestamp_now();
    let entry = SessionEntry {
        role: role.to_string(),
        content: content.to_string(),
        timestamp: created_at.clone(),
    };
    write_entry_jsonl(path, &entry);
    let session_id = session_id_from_path(path);
    memory::insert_message(&session_id, role, content, &created_at)
}

fn write_entry_jsonl(path: &Path, entry: &SessionEntry) {
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    if serde_json::to_writer(&mut file, &entry).is_err() {
        return;
    }
    let _ = file.write_all(b"\n");
}

pub fn load_recent_entries(path: &Path, turn_count: usize) -> Vec<RecentEntry> {
    if turn_count == 0 {
        return Vec::new();
    }
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(_) => continue,
        };
        let session_entry = match serde_json::from_str::<SessionEntry>(&line) {
            Ok(session_entry) => session_entry,
            Err(_) => continue,
        };
        entries.push(RecentEntry {
            role: session_entry.role,
            content: session_entry.content,
            timestamp: session_entry.timestamp,
        });
    }
    select_recent_turns(entries, turn_count)
}

fn select_recent_turns(entries: Vec<RecentEntry>, turn_count: usize) -> Vec<RecentEntry> {
    if entries.len() < 2 {
        return Vec::new();
    }
    let mut tail_end = entries.len();
    while tail_end > 0 && !is_assistant_entry(&entries[tail_end - 1]) {
        tail_end -= 1;
    }
    if tail_end < 2 {
        return Vec::new();
    }
    let mut start_index = tail_end;
    let mut scan_end = tail_end;
    let mut pair_count = 0usize;
    while scan_end >= 2 && pair_count < turn_count {
        let user_index = scan_end - 2;
        let assistant_index = scan_end - 1;
        if !is_user_entry(&entries[user_index]) || !is_assistant_entry(&entries[assistant_index]) {
            break;
        }
        start_index = user_index;
        pair_count += 1;
        scan_end -= 2;
    }
    if pair_count == 0 {
        return Vec::new();
    }
    entries[start_index..tail_end].to_vec()
}

fn is_user_entry(entry: &RecentEntry) -> bool {
    entry.role == "user"
}

fn is_assistant_entry(entry: &RecentEntry) -> bool {
    entry.role == "assistant"
}

pub fn timestamp_now() -> String {
    timeutil::timestamp_now()
}

pub fn session_id_from_path(path: &Path) -> String {
    match path.file_stem() {
        Some(stem) => stem.to_string_lossy().into_owned(),
        None => String::new(),
    }
}
