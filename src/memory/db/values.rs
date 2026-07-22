use turso::Value;

pub(crate) fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

pub(super) fn text_or(value: Result<Value, turso::Error>, default: &str) -> String {
    match value {
        Ok(Value::Text(text)) => text,
        _ => default.to_string(),
    }
}

pub(super) fn optional_text(value: Option<Value>) -> Option<String> {
    match value? {
        Value::Text(string) => Some(string),
        Value::Null => None,
        _ => None,
    }
}

pub(super) fn optional_real(value: Option<Value>) -> Option<f64> {
    match value? {
        Value::Real(number) => Some(number),
        Value::Integer(number) => Some(number as f64),
        Value::Null => None,
        _ => None,
    }
}

pub(crate) fn text(value: Value) -> String {
    match value {
        Value::Text(string) => string,
        _ => String::new(),
    }
}

pub(crate) fn real(value: Value) -> f64 {
    match value {
        Value::Real(float) => float,
        Value::Integer(integer) => integer as f64,
        _ => 0.0,
    }
}

pub(crate) fn int(value: Value) -> i64 {
    match value {
        Value::Integer(integer) => integer,
        _ => 0,
    }
}
