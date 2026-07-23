use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use std::path::PathBuf;

use super::app::{self, App, IngestState, MemoryState, Mode, State};
use super::stream::start_stream;

pub(crate) fn handle_key(key: KeyEvent, app: &mut App) -> bool {
    match app.mode {
        Mode::Insert => handle_insert_key(key, app),
        Mode::Normal => handle_normal_key(key, app),
    }
}

fn handle_insert_key(key: KeyEvent, app: &mut App) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Esc => app.mode = Mode::Normal,
        // Chat and ingestion both require no running ingestion pass: it holds
        // a transaction open across LLM calls, and concurrent chat memory
        // writes abort it (turso allows only one writer).
        KeyCode::Enter if app.state == State::Idle && !app.input.is_empty() => {
            let path = ingest_command_path(&app.input);
            if let Some(path) = path {
                if maintenance_allowed(app) {
                    let path = path.to_string();
                    app.input.clear();
                    app::start_ingestion(app, PathBuf::from(path));
                }
            } else if background_passes_idle(app) {
                start_stream(app);
            }
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Char(chr) => app.input.push(chr),
        _ => {}
    }
    false
}

fn handle_normal_key(key: KeyEvent, app: &mut App) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('q') => return true,
        KeyCode::Char('i') => app.mode = Mode::Insert,
        KeyCode::Char('j') => scroll_down(app),
        KeyCode::Char('k') => scroll_up(app),
        KeyCode::Char('G') => jump_to_end(app),
        KeyCode::Char('m') => app.show_memory_panel = !app.show_memory_panel,
        KeyCode::Char('R') => app.show_reasoning = !app.show_reasoning,
        KeyCode::Char('d') => app.show_debug = !app.show_debug,
        KeyCode::Char('t') => app.show_tools = !app.show_tools,
        KeyCode::Char('a') => app.show_attribution = !app.show_attribution,
        KeyCode::Char('P') if maintenance_allowed(app) => app::start_recompute_positions(app),
        KeyCode::Char('S') if maintenance_allowed(app) => app::start_sync(app),
        KeyCode::Char('A') => toggle_autonomous_mode(app),
        _ => {}
    }
    false
}

fn toggle_autonomous_mode(app: &mut App) {
    app::toggle_autonomous_mode(app);
    app::reset_scheduler_idle(app);
}

// Background maintenance actions are mutually exclusive with ingestion and
// must not run during streaming or memory updates.
fn maintenance_allowed(app: &App) -> bool {
    app.state == State::Idle
        && app.memory_state == MemoryState::Idle
        && app.ingest_state != IngestState::Running
}

fn background_passes_idle(app: &App) -> bool {
    app.ingest_state != IngestState::Running
}

fn ingest_command_path(input: &str) -> Option<&str> {
    let trimmed = input.trim();
    let path = trimmed.strip_prefix("/ingest ")?;
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    Some(path)
}

fn scroll_down(app: &mut App) {
    if app.show_memory_panel {
        app.memory_auto_scroll = false;
        app.memory_scroll = app.memory_scroll.saturating_add(3);
    } else {
        app.auto_scroll = false;
        app.scroll = app.scroll.saturating_add(3);
    }
}

fn scroll_up(app: &mut App) {
    if app.show_memory_panel {
        app.memory_auto_scroll = false;
        app.memory_scroll = app.memory_scroll.saturating_sub(3);
    } else {
        app.auto_scroll = false;
        app.scroll = app.scroll.saturating_sub(3);
    }
}

fn jump_to_end(app: &mut App) {
    if app.show_memory_panel {
        app.memory_auto_scroll = true;
    } else {
        app.auto_scroll = true;
    }
}
