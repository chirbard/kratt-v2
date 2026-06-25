#![deny(clippy::disallowed_methods)]
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

use std::io::Stdout;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use api::{Chunk, Stats};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

const DEFAULT_MODEL: &str = "deepseek/deepseek-v4-flash";

#[derive(PartialEq)]
enum State {
    Idle,
    Streaming,
    Done,
}

struct App {
    input: String,
    reasoning: String,
    content: String,
    state: State,
    stats: Option<Stats>,
    scroll: u16,
    auto_scroll: bool,
    stream_rx: Option<Receiver<Chunk>>,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            reasoning: String::new(),
            content: String::new(),
            state: State::Idle,
            stats: None,
            scroll: 0,
            auto_scroll: true,
            stream_rx: None,
        }
    }
}

type Term = Terminal<CrosstermBackend<Stdout>>;

#[allow(clippy::disallowed_methods)]
fn tui_enter(stdout: &mut Stdout) -> std::io::Result<()> {
    execute!(stdout, EnterAlternateScreen)
}

#[allow(clippy::disallowed_methods)]
fn tui_exit(terminal: &mut Term) -> std::io::Result<()> {
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv()?;
    let api_key = std::env::var("OPENROUTER_API_KEY")?;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    tui_enter(&mut stdout)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal, &api_key);

    disable_raw_mode()?;
    tui_exit(&mut terminal)?;
    terminal.show_cursor()?;

    result
}

fn run(terminal: &mut Term, api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new();
    loop {
        if event::poll(Duration::from_millis(16))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && handle_key(key, &mut app, api_key)
        {
            break;
        }
        drain_stream(&mut app);
        terminal.draw(|frame| render(frame, &app))?;
    }
    Ok(())
}

fn start_stream(app: &mut App, api_key: &str) {
    let prompt = std::mem::take(&mut app.input);
    app.reasoning.clear();
    app.content.clear();
    app.stats = None;
    app.scroll = 0;
    app.auto_scroll = true;
    app.state = State::Streaming;

    let (sender, recvr) = std::sync::mpsc::channel();
    app.stream_rx = Some(recvr);

    let key = api_key.to_string();
    let model = DEFAULT_MODEL.to_string();
    std::thread::spawn(move || api::stream(key, model, prompt, sender));
}

fn handle_key(key: KeyEvent, app: &mut App, api_key: &str) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Esc => return true,
        KeyCode::Enter if app.state != State::Streaming && !app.input.is_empty() => {
            start_stream(app, api_key);
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Char(chr) => app.input.push(chr),
        KeyCode::Up => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(3);
        }
        KeyCode::Down => app.scroll = app.scroll.saturating_add(3),
        KeyCode::PageUp => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(20);
        }
        KeyCode::PageDown => app.scroll = app.scroll.saturating_add(20),
        _ => {}
    }
    false
}

fn drain_stream(app: &mut App) {
    let recv = match app.stream_rx.take() {
        Some(recv) => recv,
        None => return,
    };
    while let Ok(chunk) = recv.try_recv() {
        match chunk {
            Chunk::Reasoning(text) => app.reasoning.push_str(&text),
            Chunk::Content(text) => app.content.push_str(&text),
            Chunk::Done(stats) => {
                app.stats = Some(stats);
                app.state = State::Done;
            }
            Chunk::Err(err) => {
                app.content = format!("Error: {err}");
                app.state = State::Done;
            }
        }
    }
    if app.state == State::Streaming {
        app.stream_rx = Some(recv);
    }
}

fn build_output_text(app: &App) -> Text<'static> {
    let mut lines: Vec<Line> = Vec::new();
    for line in app.reasoning.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
    }
    if !app.reasoning.is_empty() && !app.content.is_empty() {
        lines.push(Line::default());
    }
    for line in app.content.lines() {
        lines.push(Line::from(line.to_string()));
    }
    Text::from(lines)
}

fn render_output(frame: &mut Frame, area: Rect, app: &App) {
    let text = build_output_text(app);
    let total_lines = text.lines.len() as u16;
    let visible_height = area.height.saturating_sub(2);
    let scroll = if app.auto_scroll {
        total_lines.saturating_sub(visible_height)
    } else {
        app.scroll.min(total_lines.saturating_sub(visible_height))
    };
    let title = match app.state {
        State::Idle => " kratt ",
        State::Streaming => " ... ",
        State::Done => " done ",
    };
    let output = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(output, area);
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let input_widget = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title(" > "));
    frame.render_widget(input_widget, area);

    let cursor_x = area.x + 1 + app.input.chars().count() as u16;
    let cursor_y = area.y + 1;
    if cursor_x < area.x + area.width.saturating_sub(1) {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn render_stats(frame: &mut Frame, area: Rect, stats: &Stats) {
    let cost_str = match stats.cost_usd {
        Some(cost) => format!("${cost:.6}"),
        None => "N/A".to_string(),
    };
    let text = format!(" ↑{} ↓{}  {}  TTFT {}ms", stats.tokens_in, stats.tokens_out, cost_str, stats.ttft_ms);
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn render(frame: &mut Frame, app: &App) {
    let areas = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .split(frame.area());

    render_output(frame, areas[0], app);
    render_input(frame, areas[1], app);

    if let Some(stats) = &app.stats {
        render_stats(frame, areas[2], stats);
    }
}
