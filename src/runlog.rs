use std::io::Write;
use std::time::Instant;

use crate::paths;
use crate::timeutil;

pub struct StepTimer {
    area: &'static str,
    step: String,
    start: Instant,
}

pub fn time_step(area: &'static str, step: impl Into<String>) -> StepTimer {
    StepTimer {
        area,
        step: step.into(),
        start: Instant::now(),
    }
}

impl Drop for StepTimer {
    fn drop(&mut self) {
        let timing = format_timing(&self.step, self.start.elapsed().as_millis());
        log(self.area, &timing);
    }
}

fn format_timing(step: &str, elapsed_ms: u128) -> String {
    format!("{step} took {elapsed_ms}ms")
}

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

#[cfg(test)]
mod tests {
    #[test]
    fn timing_format_includes_step_and_zero_milliseconds() {
        assert_eq!(super::format_timing("test step", 0), "test step took 0ms");
    }
}
