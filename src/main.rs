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
mod graph;
mod memory_agent;
mod memory_prompt;
mod session;

use std::io::Stdout;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use api::{Chunk, Stats};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use memory_agent::MemoryEvent;
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

#[derive(PartialEq)]
enum MemoryState {
    Idle,
    Updating,
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
    api_key: String,
    input: String,
    mode: Mode,
    history: Vec<Message>,
    tick: u64,
    state: State,
    memory_state: MemoryState,
    stats: Option<Stats>,
    scroll: u16,
    auto_scroll: bool,
    stream_rx: Option<Receiver<Chunk>>,
    memory_rx: Option<Receiver<MemoryEvent>>,
    memory_log: Vec<(String, String)>,
    show_memory_panel: bool,
    show_reasoning: bool,
    show_debug: bool,
    session_path: PathBuf,
    memory_toon: String,
}

impl App {
    fn new(api_key: String, memory_toon: String, session_path: PathBuf) -> Self {
        Self {
            api_key,
            input: String::new(),
            mode: Mode::Insert,
            history: Vec::new(),
            tick: 0,
            state: State::Idle,
            memory_state: MemoryState::Idle,
            stats: None,
            scroll: 0,
            auto_scroll: true,
            stream_rx: None,
            memory_rx: None,
            memory_log: Vec::new(),
            show_memory_panel: false,
            show_reasoning: true,
            show_debug: false,
            session_path,
            memory_toon,
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
    let graph = graph::load_graph();
    let memory_toon = toon::encode(&graph::graph_to_json(&graph), None);
    let session_path = session::new_session_path();
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    tui_enter(&mut stdout)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run(&mut terminal, api_key, memory_toon, session_path);
    disable_raw_mode()?;
    tui_exit(&mut terminal)?;
    terminal.show_cursor()?;
    result
}

fn run(
    terminal: &mut Term,
    api_key: String,
    memory_toon: String,
    session_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new(api_key, memory_toon, session_path);
    loop {
        if event::poll(Duration::from_millis(16))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && handle_key(key, &mut app)
        {
            break;
        }
        drain_stream(&mut app);
        drain_memory(&mut app);
        app.tick += 1;
        terminal.draw(|frame| render(frame, &app))?;
    }
    Ok(())
}

fn start_stream(app: &mut App) {
    let prompt = std::mem::take(&mut app.input);
    session::append_entry(&app.session_path, "user", &prompt);
    app.history.push(Message {
        role: Role::User,
        reasoning: String::new(),
        content: prompt.clone(),
    });
    app.history.push(Message {
        role: Role::Assistant,
        reasoning: String::new(),
        content: String::new(),
    });
    app.stats = None;
    app.auto_scroll = true;
    app.state = State::Streaming;
    let (sender, recvr) = std::sync::mpsc::channel();
    app.stream_rx = Some(recvr);
    let system = memory_prompt::build_chat_system_prompt(&app.memory_toon);
    let system_opt = if system.is_empty() {
        None
    } else {
        Some(system)
    };
    let key = app.api_key.clone();
    let model = DEFAULT_MODEL.to_string();
    std::thread::spawn(move || api::stream(key, model, prompt, system_opt, sender));
}

fn handle_key(key: KeyEvent, app: &mut App) -> bool {
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
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_add(3);
        }
        KeyCode::Char('k') => {
            app.auto_scroll = false;
            app.scroll = app.scroll.saturating_sub(3);
        }
        KeyCode::Char('G') => app.auto_scroll = true,
        KeyCode::Char('m') => app.show_memory_panel = !app.show_memory_panel,
        KeyCode::Char('R') => app.show_reasoning = !app.show_reasoning,
        KeyCode::Char('d') => app.show_debug = !app.show_debug,
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
                start_memory_update(app);
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

fn start_memory_update(app: &mut App) {
    let assistant_content = assistant_content(app);
    let user_content = user_content(app);
    session::append_entry(&app.session_path, "assistant", &assistant_content);
    app.memory_log.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let api_key = app.api_key.clone();
    let model = DEFAULT_MODEL.to_string();
    std::thread::spawn(move || {
        let _ = memory_agent::run_memory_agent(
            &api_key,
            &model,
            &user_content,
            &assistant_content,
            sender,
        );
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

fn assistant_content(app: &App) -> String {
    match app.history.last() {
        Some(message) => message.content.clone(),
        None => String::new(),
    }
}

fn user_content(app: &App) -> String {
    match app.history.iter().rev().nth(1) {
        Some(message) => message.content.clone(),
        None => String::new(),
    }
}

fn drain_memory(app: &mut App) {
    let memory_receiver = match app.memory_rx.take() {
        Some(memory_receiver) => memory_receiver,
        None => return,
    };
    loop {
        match memory_receiver.try_recv() {
            Ok(MemoryEvent::Log { phase, text }) => {
                app.memory_log.push((phase.to_string(), text));
            }
            Ok(MemoryEvent::Done) => {
                let graph = graph::load_graph();
                app.memory_toon = toon::encode(&graph::graph_to_json(&graph), None);
                app.memory_state = MemoryState::Idle;
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.memory_rx = Some(memory_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.memory_state = MemoryState::Idle;
                return;
            }
        }
    }
}

fn push_user_lines(lines: &mut Vec<Line>, msg: &Message) {
    lines.push(Line::from(Span::styled(
        "You",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )));
    for line in msg.content.lines() {
        lines.push(Line::from(line.to_string()));
    }
}

fn push_assistant_lines(
    lines: &mut Vec<Line>,
    msg: &Message,
    is_streaming: bool,
    tick: u64,
    show_reasoning: bool,
) {
    lines.push(Line::from(Span::styled(
        "kratt",
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
    )));
    let reasoning_visible = show_reasoning && !msg.reasoning.is_empty();
    if reasoning_visible {
        for line in msg.reasoning.lines() {
            lines.push(Line::from(Span::styled(
                line.to_string(),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
        }
    }
    let show_spinner = is_streaming && msg.content.is_empty() && !reasoning_visible;
    if show_spinner {
        let frame = SPINNER[((tick / 4) as usize) % SPINNER.len()];
        lines.push(Line::from(Span::styled(
            format!("{frame} waiting..."),
            Style::default().fg(Color::DarkGray),
        )));
        return;
    }
    if reasoning_visible && !msg.content.is_empty() {
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
            Role::Assistant => push_assistant_lines(
                &mut lines,
                msg,
                streaming_here,
                app.tick,
                app.show_reasoning,
            ),
        }
        if !is_last {
            lines.push(Line::default());
        }
    }
    Text::from(lines)
}

fn count_visual_rows(text: &Text, area_width: u16) -> u16 {
    let inner = area_width.saturating_sub(2).max(1) as usize;
    let mut total: u16 = 0;
    for line in &text.lines {
        let mut chars: usize = 0;
        for span in &line.spans {
            chars += span.content.chars().count();
        }
        total += chars.max(1).div_ceil(inner) as u16;
    }
    total
}

fn render_output(frame: &mut Frame, area: Rect, app: &App) {
    let text = build_output_text(app);
    let total_rows = count_visual_rows(&text, area.width);
    let visible_height = area.height.saturating_sub(2);
    let scroll = if app.auto_scroll {
        total_rows.saturating_sub(visible_height)
    } else {
        app.scroll.min(total_rows.saturating_sub(visible_height))
    };
    let title = if app.state == State::Streaming {
        " ... "
    } else {
        " kratt "
    };
    let output = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(output, area);
}

fn phase_color(phase: &str) -> Color {
    match phase {
        "extract" => Color::Cyan,
        "consolidate" => Color::Yellow,
        "memory" => Color::Green,
        "debug" => Color::DarkGray,
        _ => Color::White,
    }
}

fn render_memory_panel(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();
    for (phase, text) in &app.memory_log {
        if phase == "debug" && !app.show_debug {
            continue;
        }
        let label = format!("[{phase}] ");
        let line = Line::from(vec![
            Span::styled(label, Style::default().fg(phase_color(phase))),
            Span::raw(text.clone()),
        ]);
        lines.push(line);
    }
    let text = Text::from(lines);
    let total_rows = count_visual_rows(&text, area.width);
    let visible_height = area.height.saturating_sub(2);
    let scroll = total_rows.saturating_sub(visible_height);
    let spinner_ch = SPINNER[((app.tick / 6) as usize) % SPINNER.len()];
    let title = if app.memory_state == MemoryState::Updating {
        format!(" memory {spinner_ch} ")
    } else {
        " memory ".to_string()
    };
    let panel = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(panel, area);
}

fn render_input(frame: &mut Frame, area: Rect, app: &App) {
    let in_normal = app.mode == Mode::Normal;
    let style = if in_normal {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" > ")
        .border_style(style);
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

fn render_stats(frame: &mut Frame, area: Rect, app: &App) {
    let mut text = match &app.stats {
        Some(stats) => {
            let cost = match stats.cost_usd {
                Some(value) => format!("${value:.6}"),
                None => "N/A".to_string(),
            };
            format!(
                " ↑{} ↓{}  {}  TTFT {}ms",
                stats.tokens_in, stats.tokens_out, cost, stats.ttft_ms
            )
        }
        None => String::new(),
    };
    if app.memory_state == MemoryState::Updating {
        let spinner = SPINNER[((app.tick / 6) as usize) % SPINNER.len()];
        if !text.is_empty() {
            text.push_str("  ");
        }
        text.push_str(&format!("mem {spinner}"));
    }
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn render_mode(frame: &mut Frame, area: Rect, app: &App) {
    let text = match app.mode {
        Mode::Insert => "-- INSERT --".to_string(),
        Mode::Normal => {
            let mem_hint = if app.show_memory_panel {
                "[m] mem ✓"
            } else {
                "[m] mem"
            };
            let rea_hint = if app.show_reasoning {
                "[R] reasoning ✓"
            } else {
                "[R] reasoning"
            };
            let dbg_hint = if app.show_debug { "[d] debug ✓" } else { "[d] debug" };
            format!("-- NORMAL --  {mem_hint}  {rea_hint}  {dbg_hint}")
        }
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

    if app.show_memory_panel {
        let cols = Layout::horizontal([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(areas[0]);
        render_output(frame, cols[0], app);
        render_memory_panel(frame, cols[1], app);
    } else {
        render_output(frame, areas[0], app);
    }
    render_input(frame, areas[1], app);
    render_stats(frame, areas[2], app);
    render_mode(frame, areas[3], app);
}
