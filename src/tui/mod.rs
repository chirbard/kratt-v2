mod app;
mod input;
mod render;
mod stream;

use std::io::Stdout;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::api::LlmClient;
use crate::api::EmbeddingClient;
use crate::web::SearchClient;

use app::{App, drain_dream, drain_embed, drain_memory, drain_stream};

type Term = Terminal<CrosstermBackend<Stdout>>;

pub fn run(
    chat_llm: Arc<dyn LlmClient>,
    memory_llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    session_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    tui_enter(&mut stdout)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new(chat_llm, memory_llm, embed, search, session_path);
    let result = run_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    tui_exit(&mut terminal)?;
    terminal.show_cursor()?;
    result
}

fn run_loop(terminal: &mut Term, app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        if event::poll(Duration::from_millis(16))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && input::handle_key(key, app)
        {
            break;
        }
        drain_stream(app);
        drain_memory(app);
        drain_dream(app);
        drain_embed(app);
        app.tick += 1;
        terminal.draw(|frame| render::render(frame, app))?;
    }
    Ok(())
}

#[allow(clippy::disallowed_methods)]
fn tui_enter(stdout: &mut Stdout) -> std::io::Result<()> {
    execute!(stdout, EnterAlternateScreen)
}

#[allow(clippy::disallowed_methods)]
fn tui_exit(terminal: &mut Term) -> std::io::Result<()> {
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
}
