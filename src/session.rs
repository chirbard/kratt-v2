use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize)]
struct SessionEntry {
    role: String,
    content: String,
    #[serde(rename = "ts")]
    timestamp: String,
}

pub fn new_session_path() -> PathBuf {
    let timestamp = timestamp_parts();
    PathBuf::from("kratt-data")
        .join("sessions")
        .join(format!(
            "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}.jsonl",
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second
        ))
}

pub fn append_entry(path: &Path, role: &str, content: &str) {
    let entry = SessionEntry {
        role: role.to_string(),
        content: content.to_string(),
        timestamp: format_timestamp(':'),
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    if serde_json::to_writer(&mut file, &entry).is_err() {
        return;
    }
    let _ = file.write_all(b"\n");
}

struct TimestampParts {
    year: i64,
    month: i64,
    day: i64,
    hour: u64,
    minute: u64,
    second: u64,
}

fn timestamp_parts() -> TimestampParts {
    let seconds_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_since_epoch = (seconds_since_epoch / 86_400) as i64;
    let second_of_day = seconds_since_epoch % 86_400;
    let (year, month, day) = civil_from_days(days_since_epoch);
    TimestampParts {
        year,
        month,
        day,
        hour: second_of_day / 3_600,
        minute: (second_of_day % 3_600) / 60,
        second: second_of_day % 60,
    }
}

fn format_timestamp(time_separator: char) -> String {
    let timestamp = timestamp_parts();
    format!(
        "{:04}-{:02}-{:02}T{:02}{}{:02}{}{:02}",
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        time_separator,
        timestamp.minute,
        time_separator,
        timestamp.second
    )
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let shifted_days = days_since_epoch + 719_468;
    let era = (if shifted_days >= 0 {
        shifted_days
    } else {
        shifted_days - 146_096
    }) / 146_097;
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year_base = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 { month_prime + 3 } else { month_prime - 9 };
    let year = if month <= 2 { year_base + 1 } else { year_base };
    (year, month, day)
}
