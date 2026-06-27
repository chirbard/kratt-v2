use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::app::{App, MemoryState, Message, Mode, Role, State};

pub(crate) const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) fn render(frame: &mut Frame, app: &App) {
    use ratatui::layout::{Constraint, Layout};
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
    let inner_width = area.width.saturating_sub(2);
    let char_count = app.input.chars().count() as u16;
    let scroll_col = if char_count >= inner_width {
        char_count - inner_width + 1
    } else {
        0
    };
    let input_widget = Paragraph::new(app.input.as_str())
        .style(style)
        .block(block)
        .scroll((0, scroll_col));
    frame.render_widget(input_widget, area);
    if app.mode == Mode::Insert {
        let cursor_x = area.x + 1 + char_count - scroll_col;
        let cursor_y = area.y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
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
            let dbg_hint = if app.show_debug {
                "[d] debug ✓"
            } else {
                "[d] debug"
            };
            let tool_hint = if app.show_tools {
                "[t] tools ✓"
            } else {
                "[t] tools"
            };
            format!("-- NORMAL --  {mem_hint}  {rea_hint}  {dbg_hint}  {tool_hint}")
        }
    };
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn build_output_text(app: &App) -> Text<'static> {
    let mut lines: Vec<Line> = Vec::new();
    let msg_count = app.history.len();
    for (idx, msg) in app.history.iter().enumerate() {
        let is_last = idx + 1 == msg_count;
        let streaming_here = is_last && app.state == State::Streaming;
        match msg.role {
            Role::User => push_user_lines(&mut lines, msg),
            Role::Assistant => push_assistant_lines(&mut lines, msg, streaming_here, app),
        }
        if !is_last {
            lines.push(Line::default());
        }
    }
    Text::from(lines)
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

fn push_tool_line(lines: &mut Vec<Line>, tool_call: &str) {
    lines.push(Line::from(Span::styled(
        format!("[tool] {tool_call}"),
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    )));
}

fn push_reasoning_lines(lines: &mut Vec<Line>, reasoning: &str) {
    for line in reasoning.lines() {
        lines.push(Line::from(Span::styled(
            line.to_string(),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    }
}

fn push_assistant_lines(lines: &mut Vec<Line>, msg: &Message, is_streaming: bool, app: &App) {
    lines.push(Line::from(Span::styled(
        "kratt",
        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
    )));
    if app.show_tools && !msg.tool_call.is_empty() {
        push_tool_line(lines, &msg.tool_call);
    }
    let reasoning_visible = app.show_reasoning && !msg.reasoning.is_empty();
    if reasoning_visible {
        push_reasoning_lines(lines, &msg.reasoning);
    }
    let show_spinner = is_streaming && msg.content.is_empty() && !reasoning_visible;
    if show_spinner {
        let frame = SPINNER[((app.tick / 4) as usize) % SPINNER.len()];
        let status = if msg.tool_call.is_empty() {
            "waiting…"
        } else {
            msg.tool_call.as_str()
        };
        lines.push(Line::from(Span::styled(
            format!("{frame} {status}"),
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

pub(crate) fn count_visual_rows(text: &Text, area_width: u16) -> u16 {
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

fn phase_color(phase: &str) -> Color {
    match phase {
        "extract" => Color::Cyan,
        "consolidate" => Color::Yellow,
        "memory" => Color::Green,
        "debug" => Color::DarkGray,
        _ => Color::White,
    }
}
