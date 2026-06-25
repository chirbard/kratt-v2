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
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(PartialEq)]
enum Mode {
    Insert,
    Normal,
}

#[derive(PartialEq)]
enum State {
    Idle,
    Streaming,
}

enum Role {
    User,
    Assistant,
}

struct Message {
    role: Role,
    reasoning: String,
    content: String,
}

struct App {
    input: String,
    mode: Mode,
    history: Vec<Message>,
    tick: u64,
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
            mode: Mode::Insert,
            history: Vec::new(),
            tick: 0,
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
        app.tick += 1;
        terminal.draw(|frame| render(frame, &app))?;
    }
    Ok(())
}

fn start_stream(app: &mut App, api_key: &str) {
    let prompt = std::mem::take(&mut app.input);
    app.history.push(Message { role: Role::User, reasoning: String::new(), content: prompt.clone() });
    app.history.push(Message { role: Role::Assistant, reasoning: String::new(), content: String::new() });
    app.stats = None;
    app.auto_scroll = true;
    app.state = State::Streaming;
    let (sender, recvr) = std::sync::mpsc::channel();
    app.stream_rx = Some(recvr);
    let key = api_key.to_string();
    let model = DEFAULT_MODEL.to_string();
    std::thread::spawn(move || api::stream(key, model, prompt, sender));
}

fn handle_key(key: KeyEvent, app: &mut App, api_key: &str) -> bool {
    match app.mode {
        Mode::Insert => handle_insert_key(key, app, api_key),
        Mode::Normal => handle_normal_key(key, app),
    }
}

fn handle_insert_key(key: KeyEvent, app: &mut App, api_key: &str) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Esc => app.mode = Mode::Normal,
        KeyCode::Enter if app.state == State::Idle && !app.input.is_empty() => {
            start_stream(app, api_key);
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
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_add(3);
        }
        KeyCode::Char('k') => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(3);
        }
        KeyCode::Char('G') => app.auto_scroll = true,
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
            Chunk::Reasoning(text) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.reasoning.push_str(&text);
                }
            }
            Chunk::Content(text) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.content.push_str(&text);
                }
            }
            Chunk::Done(stats) => {
                app.stats = Some(stats);
                app.state = State::Idle;
            }
            Chunk::Err(err) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.content = format!("Error: {err}");
                }
                app.state = State::Idle;
            }
        }
    }
    if app.state == State::Streaming {
        app.stream_rx = Some(recv);
    }
}

fn push_user_lines(lines: &mut Vec<Line>, msg: &Message) {
    lines.push(Line::from(Span::styled(
        "You",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    for line in msg.content.lines() {
        lines.push(Line::from(line.to_string()));
    }
}

fn push_assistant_lines(lines: &mut Vec<Line>, msg: &Message, is_streaming: bool, tick: u64) {
    lines.push(Line::from(Span::styled(
        "kratt",
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
    )));
    for line in msg.reasoning.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
    }
    let show_spinner = is_streaming && msg.content.is_empty() && msg.reasoning.is_empty();
    if show_spinner {
        let frame = SPINNER[(tick as usize / 4) % SPINNER.len()];
        lines.push(Line::from(Span::styled(
            format!("{frame} waiting..."),
            Style::default().fg(Color::DarkGray),
        )));
        return;
    }
    if !msg.reasoning.is_empty() && !msg.content.is_empty() {
        lines.push(Line::default());
    }
    for line in msg.content.lines() {
        lines.push(Line::from(line.to_string()));
    }
}

fn build_output_text(app: &App) -> Text<'static> {
    let mut lines: Vec<Line> = Vec::new();
    let msg_count = app.history.len();
    for (idx, msg) in app.history.iter().enumerate() {
        let is_last = idx + 1 == msg_count;
        let streaming_here = is_last && app.state == State::Streaming;
        match msg.role {
            Role::User => push_user_lines(&mut lines, msg),
            Role::Assistant => push_assistant_lines(&mut lines, msg, streaming_here, app.tick),
        }
        if !is_last {
            lines.push(Line::default());
        }
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
    let title = if app.state == State::Streaming { " ... " } else { " kratt " };
    let output = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(output, area);
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let in_normal = app.mode == Mode::Normal;
    let style = if in_normal { Style::default().fg(Color::DarkGray) } else { Style::default() };
    let block = Block::default().borders(Borders::ALL).title(" > ").border_style(style);
    let input_widget = Paragraph::new(app.input.as_str()).style(style).block(block);
    frame.render_widget(input_widget, area);
    if app.mode == Mode::Insert {
        let cursor_x = area.x + 1 + app.input.chars().count() as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width.saturating_sub(1) {
            frame.set_cursor_position((cursor_x, cursor_y));
        }
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

fn render_mode(frame: &mut Frame, area: Rect, app: &App) {
    let text = match app.mode {
        Mode::Insert => "-- INSERT --",
        Mode::Normal => "-- NORMAL --",
    };
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn render(frame: &mut Frame, app: &App) {
    let areas = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(frame.area());

    render_output(frame, areas[0], app);
    render_input(frame, areas[1], app);

    if let Some(stats) = &app.stats {
        render_stats(frame, areas[2], stats);
    }

    render_mode(frame, areas[3], app);
}
