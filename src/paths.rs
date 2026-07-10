use std::sync::OnceLock;

// The dev-isolation axis: KRATT_DATA_DIR overrides the data root once at
// startup, so a dev bot token + a separate data dir keep the live brain and
// the dev harness fully apart. Read once — both front-ends share this value.
const DEFAULT_DATA_DIR: &str = "kratt-data";

pub fn data_dir() -> &'static str {
    static DATA_DIR: OnceLock<String> = OnceLock::new();
    DATA_DIR.get_or_init(|| match std::env::var("KRATT_DATA_DIR") {
        Ok(dir) if !dir.trim().is_empty() => dir,
        _ => DEFAULT_DATA_DIR.to_string(),
    })
}

// Data-root-relative path as a string (turso builders and fs calls want owned
// strings, not PathBuf).
pub fn data_file(relative: &str) -> String {
    format!("{}/{relative}", data_dir())
}
