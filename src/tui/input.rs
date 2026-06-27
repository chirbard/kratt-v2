use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app::{App, Mode, State};
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
        KeyCode::Enter if app.state == State::Idle && !app.input.is_empty() => {
            start_stream(app);
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
        KeyCode::Char('j') => {
            if app.show_memory_panel {
                app.memory_auto_scroll = false;
                app.memory_scroll = app.memory_scroll.saturating_add(3);
            } else {
                app.auto_scroll = false;
                app.scroll = app.scroll.saturating_add(3);
            }
        }
        KeyCode::Char('k') => {
            if app.show_memory_panel {
                app.memory_auto_scroll = false;
                app.memory_scroll = app.memory_scroll.saturating_sub(3);
            } else {
                app.auto_scroll = false;
                app.scroll = app.scroll.saturating_sub(3);
            }
        }
        KeyCode::Char('G') => {
            if app.show_memory_panel {
                app.memory_auto_scroll = true;
            } else {
                app.auto_scroll = true;
            }
        }
        KeyCode::Char('m') => app.show_memory_panel = !app.show_memory_panel,
        KeyCode::Char('R') => app.show_reasoning = !app.show_reasoning,
        KeyCode::Char('d') => app.show_debug = !app.show_debug,
        KeyCode::Char('t') => app.show_tools = !app.show_tools,
        _ => {}
    }
    false
}
