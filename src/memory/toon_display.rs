use serde_json::Value;

pub(crate) struct Table<'table> {
    pub name: &'table str,
    pub columns: &'table [&'table str],
    pub rows: Vec<Vec<Value>>,
}

pub(crate) fn encode<'table>(tables: &[Table<'table>]) -> String {
    let mut lines = Vec::new();
    for table in tables {
        lines.push(format!(
            "{}[{}]{{{}}}:",
            table.name,
            table.rows.len(),
            table.columns.join(",")
        ));
        for row in &table.rows {
            debug_assert_eq!(row.len(), table.columns.len());
            lines.push(format!("  {}", encode_row(row)));
        }
    }
    lines.join("\n")
}

fn encode_row(row: &[Value]) -> String {
    let mut values = Vec::new();
    for value in row {
        values.push(encode_value(value));
    }
    values.join(",")
}

fn encode_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => encode_number(number.as_f64()),
        Value::String(text) => encode_string(text),
        Value::Array(_) | Value::Object(_) => "null".to_string(),
    }
}

fn encode_number(number: Option<f64>) -> String {
    let Some(number) = number else {
        return "null".to_string();
    };
    if !number.is_finite() {
        return "null".to_string();
    }
    if number == 0.0 {
        return "0".to_string();
    }
    if number.fract() == 0.0 && number.abs() < 1e15 {
        return format!("{number:.0}");
    }
    format!("{number}")
}

fn encode_string(value: &str) -> String {
    if is_safe_unquoted(value) {
        return value.to_string();
    }
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

fn is_safe_unquoted(value: &str) -> bool {
    if value.is_empty() || value != value.trim() {
        return false;
    }
    if matches!(value, "true" | "false" | "null") || is_numeric_like(value) {
        return false;
    }
    if value.contains([':', '"', '\\', '[', ']', '{', '}', '\n', '\r', '\t', ',']) {
        return false;
    }
    !value.starts_with("- ")
}

fn is_numeric_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = usize::from(bytes.first() == Some(&b'-'));
    if !consume_digits(bytes, &mut index) {
        return false;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        if !consume_digits(bytes, &mut index) {
            return false;
        }
    }
    if bytes.get(index) == Some(&b'e') {
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        if !consume_digits(bytes, &mut index) {
            return false;
        }
    }
    index == bytes.len()
}

fn consume_digits(bytes: &[u8], index: &mut usize) -> bool {
    let start = *index;
    while bytes.get(*index).is_some_and(u8::is_ascii_digit) {
        *index += 1;
    }
    *index > start
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Table, encode};

    #[test]
    fn strings_follow_toon_comma_delimiter_escaping() {
        let columns = ["value"];
        let rows = vec![
            vec![json!("plain")],
            vec![json!("05")],
            vec![json!("a,b")],
            vec![json!("line\nquote\"slash\\")],
        ];
        let output = encode(&[Table {
            name: "values",
            columns: &columns,
            rows,
        }]);
        assert_eq!(
            output,
            "values[4]{value}:\n  plain\n  \"05\"\n  \"a,b\"\n  \"line\\nquote\\\"slash\\\\\""
        );
    }
}
