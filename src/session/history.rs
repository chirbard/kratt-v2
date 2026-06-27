use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct SessionEntry {
    role: String,
    content: String,
    #[serde(rename = "ts", default)]
    timestamp: String,
}

pub fn read_history(session_path: &Path, offset: usize) -> String {
    let entries = load_entries(session_path);
    if entries.len() < 2 {
        return String::new();
    }
    let mut scan_end = entries.len();
    let mut pair_offset = 0usize;
    while scan_end >= 2 {
        let assistant_index = scan_end - 1;
        let user_index = scan_end - 2;
        if !is_assistant_entry(&entries[assistant_index]) {
            scan_end -= 1;
            continue;
        }
        if !is_user_entry(&entries[user_index]) {
            scan_end -= 1;
            continue;
        }
        if pair_offset == offset {
            return format_turn(&entries[user_index], &entries[assistant_index]);
        }
        pair_offset += 1;
        scan_end -= 2;
    }
    String::new()
}

pub fn search_history(queries: &[String]) -> String {
    if queries.is_empty() {
        return String::new();
    }
    let lower_queries = lowercase_queries(queries);
    if lower_queries.is_empty() {
        return String::new();
    }
    let session_files = list_session_files();
    if session_files.is_empty() {
        return String::new();
    }
    collect_matches(&session_files, &lower_queries)
}

fn lowercase_queries(queries: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    for query in queries {
        let trimmed = query.trim();
        if !trimmed.is_empty() {
            result.push(trimmed.to_ascii_lowercase());
        }
    }
    result
}

fn collect_matches(session_files: &[PathBuf], lower_queries: &[String]) -> String {
    let mut matches = Vec::new();
    let mut file_index = session_files.len();
    while file_index > 0 {
        file_index -= 1;
        let entries = load_entries(&session_files[file_index]);
        let mut entry_index = entries.len();
        while entry_index > 0 {
            entry_index -= 1;
            let content_lower = entries[entry_index].content.to_ascii_lowercase();
            if !matches_any_query(&content_lower, lower_queries) {
                continue;
            }
            matches.push(format_match(&entries[entry_index]));
            if matches.len() == 5 {
                return join_lines(&matches);
            }
        }
    }
    join_lines(&matches)
}

fn matches_any_query(content: &str, lower_queries: &[String]) -> bool {
    for query in lower_queries {
        if content.contains(query.as_str()) {
            return true;
        }
    }
    false
}

fn load_entries(path: &Path) -> Vec<SessionEntry> {
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
        let entry = match serde_json::from_str::<SessionEntry>(&line) {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        entries.push(entry);
    }
    entries
}

fn list_session_files() -> Vec<PathBuf> {
    let read_dir = match std::fs::read_dir(Path::new("kratt-data").join("sessions")) {
        Ok(read_dir) => read_dir,
        Err(_) => return Vec::new(),
    };
    let mut paths = Vec::new();
    for dir_entry_result in read_dir {
        let dir_entry = match dir_entry_result {
            Ok(dir_entry) => dir_entry,
            Err(_) => continue,
        };
        let path = dir_entry.path();
        let extension_value = match path.extension() {
            Some(extension_value) => extension_value,
            None => continue,
        };
        let extension = match extension_value.to_str() {
            Some(extension) => extension,
            None => continue,
        };
        if extension != "jsonl" {
            continue;
        }
        paths.push(path);
    }
    paths.sort();
    paths
}

fn format_turn(user_entry: &SessionEntry, assistant_entry: &SessionEntry) -> String {
    let mut text = String::new();
    text.push_str("User: ");
    text.push_str(&user_entry.content);
    text.push_str("\nAssistant: ");
    text.push_str(&assistant_entry.content);
    text
}

fn format_match(entry: &SessionEntry) -> String {
    let mut line = String::new();
    line.push('[');
    line.push_str(&entry.timestamp);
    line.push_str("] ");
    line.push_str(&entry.role);
    line.push_str(": ");
    line.push_str(&entry.content);
    line
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

fn is_user_entry(entry: &SessionEntry) -> bool {
    entry.role == "user"
}

fn is_assistant_entry(entry: &SessionEntry) -> bool {
    entry.role == "assistant"
}
