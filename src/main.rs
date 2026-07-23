#![deny(clippy::disallowed_methods)]
// Strip A leaves the lower layers compiled but intentionally unreachable until
// Strip B removes or reconnects them for the eval harness.
#![allow(dead_code, unused_imports)]
#![warn(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::min_ident_chars
)]

mod api;
mod memory;
mod paths;
mod prompts;
mod runlog;
mod session;
mod timeutil;
mod trace;
mod turn;
mod web;

type MainResult<Output> = Result<Output, Box<dyn std::error::Error>>;

// Panics land in the runlog before the default hook prints them — under a
// daemon (systemd, detached terminal) the runlog is the only place they are
// seen at all. Observed live: turso pager panics inside tokio tasks survived
// the process but aborted DB operations invisibly (backlog: turso pager panic).
fn install_panic_runlog() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        runlog::log("panic", &info.to_string());
        default_hook(info);
    }));
}

fn main() -> MainResult<()> {
    let trace_view = trace_view_target()?;
    // All data paths (kratt-data/, logs, .env) are relative — pin the working
    // directory to the crate root so `cargo run` from a subdirectory doesn't
    // scatter kratt-data/ copies around the repo.
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;
    install_panic_runlog();
    if let Some(target) = trace_view {
        return trace::view(&target);
    }
    println!("kratt: no interactive entry point yet — eval harness pending");
    Ok(())
}

fn trace_view_target() -> MainResult<Option<String>> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--trace-view" {
            return match args.next() {
                Some(target) => Ok(Some(target)),
                None => Err("--trace-view requires a flow folder or day".into()),
            };
        }
    }
    Ok(None)
}
