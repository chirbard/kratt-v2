use std::error::Error;
use std::path::Path;
use std::process::{Command, Stdio};

const DATA_DIR: &str = "kratt-data";
const GRAPH_PATH: &str = "kratt-graph.toon";

fn run_git(args: &[&str]) -> Result<(), Box<dyn Error + Send + Sync>> {
    let status = Command::new("git")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        return Err(format!("git {args:?} exited with {status}").into());
    }
    Ok(())
}

fn has_staged_changes() -> Result<bool, Box<dyn Error + Send + Sync>> {
    let status = Command::new("git")
        .args(["-C", DATA_DIR, "diff", "--cached", "--quiet"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        return Ok(false);
    }
    match status.code() {
        Some(1) => Ok(true),
        _ => Err(format!(
            "git {:?} exited with {status}",
            ["-C", DATA_DIR, "diff", "--cached", "--quiet"]
        )
        .into()),
    }
}

pub fn git_init_if_needed() -> Result<(), Box<dyn Error + Send + Sync>> {
    let git_dir = Path::new(DATA_DIR).join(".git");
    if git_dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(DATA_DIR)?;
    run_git(&["init", DATA_DIR])?;
    run_git(&["-C", DATA_DIR, "config", "user.email", "kratt@local"])?;
    run_git(&["-C", DATA_DIR, "config", "user.name", "kratt"])?;
    run_git(&["-C", DATA_DIR, "commit", "--allow-empty", "-m", "init: memory state"])?;
    Ok(())
}

pub fn git_commit(label: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
    run_git(&["-C", DATA_DIR, "add", GRAPH_PATH])?;
    if !has_staged_changes()? {
        return Ok(false);
    }
    run_git(&["-C", DATA_DIR, "commit", "-m", &format!("mem: {label}")])?;
    Ok(true)
}

pub fn git_diff_last() -> String {
    let output = Command::new("git")
        .args(["-C", DATA_DIR, "diff", "HEAD~1..HEAD", "--", GRAPH_PATH])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        _ => String::new(),
    }
}
