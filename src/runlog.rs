use std::io::Write;

use crate::paths;
use crate::timeutil;

pub fn log(area: &str, text: &str) {
    if area.trim().is_empty() || text.trim().is_empty() {
        return;
    }
    let day = timeutil::today_iso();
    let log_dir = paths::data_file("logs");
    let path = format!("{log_dir}/kratt-{day}.log");
    if std::fs::create_dir_all(&log_dir).is_err() {
        return;
    }
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(file) => file,
        Err(_) => return,
    };
    let line = format!("{} [{}] {}\n", timeutil::timestamp_now(), area, text);
    let _ = file.write_all(line.as_bytes());
}
