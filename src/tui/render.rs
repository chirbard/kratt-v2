use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use super::app::{App, DreamState, ExploreState, MemoryState, Message, Mode, Role, State};

pub(crate) const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) fn render(frame: &mut Frame, app: &mut App) {
    use ratatui::layout::{Constraint, Layout};
    let areas = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(frame.area());

    if app.show_memory_panel {
        render_memory_panel(frame, areas[0], app);
    } else {
        render_output(frame, areas[0], app);
    }
    render_input(frame, areas[1], app);
    render_stats(frame, areas[2], app);
    render_mode(frame, areas[3], app);
}

fn render_output(frame: &mut Frame, area: Rect, app: &mut App) {
    let text = build_output_text(app);
    let is_streaming = app.state == State::Streaming;
    let block = Block::default().borders(Borders::TOP | Borders::BOTTOM);
    let inner = block.inner(area);
    let total_rows = count_visual_rows(&text, inner.width);
    let max_scroll = total_rows.saturating_sub(inner.height);
    let scroll = if app.auto_scroll {
        max_scroll
    } else {
        app.scroll.min(max_scroll)
    };
    app.scroll = scroll;
    let pos_str = scroll_label(scroll, max_scroll);
    let state_str = if is_streaming { "..." } else { "kratt" };
    let title = format!(" {state_str}  {pos_str} ");
    let output = Paragraph::new(text)
        .block(block.title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(output, area);
    let mut scroll_state = ScrollbarState::new(max_scroll as usize).position(scroll as usize);
    frame.render_stateful_widget(
        Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight),
        area,
        &mut scroll_state,
    );
}

fn build_memory_text(app: &App) -> Text<'static> {
    let mut lines: Vec<Line> = Vec::new();
    append_log_section(&mut lines, &app.memory_log, None, app);
    append_log_section(&mut lines, &app.dream_log, Some("dream"), app);
    append_log_section(&mut lines, &app.explore_log, Some("explore"), app);
    Text::from(lines)
}

fn append_log_section(
    lines: &mut Vec<Line<'static>>,
    entries: &[(String, String)],
    title: Option<&str>,
    app: &App,
) {
    if entries.is_empty() {
        return;
    }
    if let Some(title_text) = title
        && !lines.is_empty()
    {
        lines.push(Line::from(Span::styled(
            format!("── {title_text} ──"),
            Style::default().fg(Color::DarkGray),
        )));
    }
    for (phase, text) in entries {
        if should_hide_phase(phase, app) {
            continue;
        }
        push_phase_line(lines, phase, text);
    }
}

fn should_hide_phase(phase: &str, app: &App) -> bool {
    if phase == "think" && !app.show_reasoning {
        return true;
    }
    phase == "debug" && !app.show_debug
}

fn push_phase_line(lines: &mut Vec<Line<'static>>, phase: &str, text: &str) {
    let label = format!("[{phase}] ");
    lines.push(Line::from(vec![
        Span::styled(label, Style::default().fg(phase_color(phase))),
        Span::raw(text.to_string()),
    ]));
}

fn render_memory_panel(frame: &mut Frame, area: Rect, app: &mut App) {
    let text = build_memory_text(app);
    let spinner_ch = SPINNER[((app.tick / 6) as usize) % SPINNER.len()];
    let state_str = if app.memory_state == MemoryState::Updating {
        format!("memory {spinner_ch}")
    } else {
        "memory".to_string()
    };
    let block = Block::default().borders(Borders::TOP | Borders::BOTTOM);
    let inner = block.inner(area);
    let total_rows = count_visual_rows(&text, inner.width);
    let max_scroll = total_rows.saturating_sub(inner.height);
    let scroll = if app.memory_auto_scroll {
        max_scroll
    } else {
        app.memory_scroll.min(max_scroll)
    };
    app.memory_scroll = scroll;
    let pos_str = scroll_label(scroll, max_scroll);
    let title = format!(" {state_str}  {pos_str} ");
    let panel = Paragraph::new(text)
        .block(block.title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(panel, area);
    let mut scroll_state = ScrollbarState::new(max_scroll as usize).position(scroll as usize);
    frame.render_stateful_widget(
        Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight),
        area,
        &mut scroll_state,
    );
}

fn scroll_label(scroll: u16, max_scroll: u16) -> String {
    if max_scroll == 0 || scroll >= max_scroll {
        "end".to_string()
    } else {
        let pct = (scroll as u32 * 100) / max_scroll as u32;
        format!("{pct}%")
    }
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
    let mut text = stats_line(app);
    append_memory_status(&mut text, app);
    append_dream_status(&mut text, app);
    append_explore_status(&mut text, app);
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn stats_line(app: &App) -> String {
    match &app.stats {
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
    }
}

fn append_memory_status(text: &mut String, app: &App) {
    if app.memory_state != MemoryState::Updating {
        return;
    }
    append_status_text(text, format!("mem {}", spinner_char(app.tick)));
}

fn append_dream_status(text: &mut String, app: &App) {
    if app.dream_state == DreamState::Running {
        append_status_text(text, format!("dream {}", spinner_char(app.tick)));
        return;
    }
    if app.dream_state == DreamState::Done && !app.dream_summary.is_empty() {
        append_status_text(text, app.dream_summary.clone());
    }
}

fn append_explore_status(text: &mut String, app: &App) {
    if app.explore_state == ExploreState::Running {
        append_status_text(text, format!("explore {}", spinner_char(app.tick)));
        return;
    }
    if app.explore_state == ExploreState::Done && !app.explore_summary.is_empty() {
        append_status_text(text, app.explore_summary.clone());
    }
}

fn append_status_text(text: &mut String, status: String) {
    if !text.is_empty() {
        text.push_str("  ");
    }
    text.push_str(&status);
}

fn spinner_char(tick: u64) -> &'static str {
    SPINNER[((tick / 6) as usize) % SPINNER.len()]
}

fn render_mode(frame: &mut Frame, area: Rect, app: &App) {
    let text = match app.mode {
        Mode::Insert => "-- INSERT --".to_string(),
        Mode::Normal => normal_mode_hint(app),
    };
    let widget = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(widget, area);
}

fn normal_mode_hint(app: &App) -> String {
    let mem_hint = toggle_hint("m", "mem", app.show_memory_panel);
    let rea_hint = toggle_hint("R", "reasoning", app.show_reasoning);
    let dbg_hint = toggle_hint("d", "debug", app.show_debug);
    let tool_hint = toggle_hint("t", "tools", app.show_tools);
    let attr_hint = toggle_hint("a", "attribution", app.show_attribution);
    let dream_hint = dream_hint(
        app.dream_state == DreamState::Running,
        app.dream_state == DreamState::Done,
    );
    let explore_hint = explore_hint(
        app.explore_state == ExploreState::Running,
        app.explore_state == ExploreState::Done,
    );
    format!(
        "-- NORMAL --  {mem_hint}  {rea_hint}  {dbg_hint}  {tool_hint}  {attr_hint}  {dream_hint}  {explore_hint}"
    )
}

fn toggle_hint(key: &str, label: &str, enabled: bool) -> String {
    if enabled {
        format!("[{key}] {label} ✓")
    } else {
        format!("[{key}] {label}")
    }
}

fn dream_hint(running: bool, done: bool) -> &'static str {
    if running {
        return "[D] dreaming…";
    }
    if done {
        return "[D] dream ✓";
    }
    "[D] dream"
}

fn explore_hint(running: bool, done: bool) -> &'static str {
    if running {
        return "[E] exploring…";
    }
    if done {
        return "[E] explore ✓";
    }
    "[E] explore"
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
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
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
    if app.show_attribution && !msg.attribution.is_empty() {
        lines.push(Line::from(Span::styled(
            msg.attribution.clone(),
            Style::default().fg(Color::DarkGray),
        )));
    }
}

pub(crate) fn count_visual_rows(text: &Text, inner_width: u16) -> u16 {
    let col_width = inner_width.max(1) as usize;
    let mut total: u16 = 0;
    for line in &text.lines {
        let mut chars: usize = 0;
        for span in &line.spans {
            chars += span.content.chars().count();
        }
        total += chars.max(1).div_ceil(col_width) as u16;
    }
    total
}

fn phase_color(phase: &str) -> Color {
    match phase {
        "extract" => Color::Cyan,
        "consolidate" => Color::Yellow,
        "memory" => Color::Green,
        "embed" => Color::Green,
        "biologist" | "historian" | "economist" | "child" => Color::Blue,
        "connector" => Color::Cyan,
        "skeptic" => Color::Red,
        "lateral" => Color::Magenta,
        "consolidation" => Color::Yellow,
        "dream" => Color::Green,
        "explore" => Color::Green,
        "think" => Color::Magenta,
        "debug" => Color::DarkGray,
        _ => Color::White,
    }
}
