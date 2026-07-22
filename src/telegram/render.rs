// Telegram renders no markdown: answers arrive as LLM markdown and must be
// converted to Telegram's HTML parse mode. Headers become bold lines, bullets
// become •, **bold** and `code` map to tags, fenced code and tables (which
// Telegram has no concept of) become <pre> blocks so alignment survives.

pub(crate) fn markdown_to_telegram_html(text: &str) -> String {
    let mut out = String::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim_start().starts_with("```") {
            render_code_fence(&mut out, &mut lines);
            continue;
        }
        if is_table_line(line) {
            render_table(&mut out, line, &mut lines);
            continue;
        }
        render_text_line(&mut out, line);
    }
    trim_trailing_newlines(&mut out);
    out
}

fn render_code_fence(
    out: &mut String,
    lines: &mut std::iter::Peekable<std::str::Lines>,
) {
    out.push_str("<pre>");
    for line in lines.by_ref() {
        if line.trim_start().starts_with("```") {
            break;
        }
        out.push_str(&escape_html(line));
        out.push('\n');
    }
    trim_trailing_newlines(out);
    out.push_str("</pre>\n");
}

// Tables keep their raw pipe layout inside <pre> — monospace alignment is the
// only table rendering Telegram can do.
fn render_table(
    out: &mut String,
    first: &str,
    lines: &mut std::iter::Peekable<std::str::Lines>,
) {
    out.push_str("<pre>");
    out.push_str(&escape_html(first.trim_end()));
    while let Some(line) = lines.peek() {
        if !is_table_line(line) {
            break;
        }
        out.push('\n');
        out.push_str(&escape_html(line.trim_end()));
        lines.next();
    }
    out.push_str("</pre>\n");
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.len() > 1
}

fn render_text_line(out: &mut String, line: &str) {
    if let Some(header) = strip_header(line) {
        out.push_str("<b>");
        out.push_str(&render_inline(header));
        out.push_str("</b>\n");
        return;
    }
    out.push_str(&render_inline(&bulletize(line)));
    out.push('\n');
}

fn strip_header(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let after_hashes = trimmed.trim_start_matches('#');
    if after_hashes.len() == trimmed.len() {
        return None;
    }
    let header = after_hashes.strip_prefix(' ')?;
    Some(header.trim())
}

fn bulletize(line: &str) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let (indent, rest) = line.split_at(indent_len);
    for marker in ["- ", "* "] {
        if let Some(item) = rest.strip_prefix(marker) {
            return format!("{indent}• {item}");
        }
    }
    line.to_string()
}

// Escape first, then wrap complete **…** and `…` pairs; unpaired markers are
// left as literal text rather than producing broken tags.
fn render_inline(line: &str) -> String {
    let escaped = escape_html(line);
    let bolded = wrap_pairs(&escaped, "**", "<b>", "</b>");
    wrap_pairs(&bolded, "`", "<code>", "</code>")
}

fn wrap_pairs(text: &str, marker: &str, open: &str, close: &str) -> String {
    let parts: Vec<&str> = text.split(marker).collect();
    if parts.len() < 3 {
        return text.to_string();
    }
    let mut out = String::new();
    let mut inside = false;
    let last = parts.len() - 1;
    for (index, part) in parts.iter().enumerate() {
        out.push_str(part);
        if index == last {
            break;
        }
        // A trailing unpaired marker stays literal.
        if !inside && index == last - 1 && parts.len().is_multiple_of(2) {
            out.push_str(marker);
            continue;
        }
        out.push_str(if inside { close } else { open });
        inside = !inside;
    }
    out
}

fn escape_html(text: &str) -> String {
    let mut out = String::new();
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(character),
        }
    }
    out
}

fn trim_trailing_newlines(out: &mut String) {
    while out.ends_with('\n') {
        out.pop();
    }
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
