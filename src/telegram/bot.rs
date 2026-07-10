use std::error::Error;
use std::time::Duration;

use serde_json::{Value, json};

// Telegram enforces a 4096-character cap per message; longer answers are split
// on paragraph boundaries into a sequence of messages.
pub(crate) const TELEGRAM_MAX_CHARS: usize = 4096;

const API_BASE: &str = "https://api.telegram.org";
// Long-poll window: getUpdates blocks server-side up to this many seconds when
// no update is pending, so the daemon holds no inbound port and rarely spins.
const LONG_POLL_SECONDS: u64 = 50;
// HTTP read timeout must outlast the long-poll window or every idle poll errors.
const HTTP_TIMEOUT_SLACK_SECONDS: u64 = 15;

pub(crate) type BotResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

// A message we can actually act on: text from a known user in a chat.
pub(crate) struct Update {
    pub user_id: i64,
    pub chat_id: i64,
    pub text: String,
}

// One entry from getUpdates. update_id always advances the poll offset (even
// when message is None), so a malformed or unsupported update is skipped
// without being re-delivered forever.
pub(crate) struct RawUpdate {
    pub update_id: i64,
    pub message: Option<Update>,
}

// Sync: the typing keep-alive thread shares the transport with the turn.
pub(crate) trait BotTransport: Sync {
    fn get_updates(&self, offset: i64) -> BotResult<Value>;
    fn send_message(&self, chat_id: i64, text: &str) -> BotResult<()>;
    // parse_mode: HTML — fails with HTTP 400 on malformed tags; callers fall
    // back to plain send_message so the answer always arrives.
    fn send_message_html(&self, chat_id: i64, html: &str) -> BotResult<()>;
    fn send_chat_action(&self, chat_id: i64, action: &str) -> BotResult<()>;
}

pub(crate) fn parse_updates(value: &Value) -> Vec<RawUpdate> {
    let results = match value.get("result") {
        Some(result) => match result.as_array() {
            Some(array) => array,
            None => return Vec::new(),
        },
        None => return Vec::new(),
    };
    let mut updates = Vec::new();
    for item in results {
        if let Some(raw) = parse_raw_update(item) {
            updates.push(raw);
        }
    }
    updates
}

// Without an update_id we cannot advance the offset safely, so drop the whole
// entry; a missing/non-text message just yields message = None.
fn parse_raw_update(item: &Value) -> Option<RawUpdate> {
    let update_id = item.get("update_id")?.as_i64()?;
    Some(RawUpdate {
        update_id,
        message: parse_message(item),
    })
}

fn parse_message(item: &Value) -> Option<Update> {
    let message = item.get("message")?;
    let user_id = message.get("from")?.get("id")?.as_i64()?;
    let chat_id = message.get("chat")?.get("id")?.as_i64()?;
    let text = message.get("text")?.as_str()?;
    Some(Update {
        user_id,
        chat_id,
        text: text.to_string(),
    })
}

pub(crate) fn split_message(text: &str, limit: usize) -> Vec<String> {
    if text.chars().count() <= limit {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n") {
        append_paragraph(&mut chunks, &mut current, paragraph, limit);
    }
    flush(&mut chunks, &mut current);
    chunks
}

fn append_paragraph(chunks: &mut Vec<String>, current: &mut String, paragraph: &str, limit: usize) {
    if paragraph.chars().count() > limit {
        flush(chunks, current);
        for piece in hard_split(paragraph, limit) {
            chunks.push(piece);
        }
        return;
    }
    if fits(current, paragraph, limit) {
        join_paragraph(current, paragraph);
    } else {
        flush(chunks, current);
        *current = paragraph.to_string();
    }
}

fn fits(current: &str, paragraph: &str, limit: usize) -> bool {
    if current.is_empty() {
        return true;
    }
    current.chars().count() + 2 + paragraph.chars().count() <= limit
}

fn join_paragraph(current: &mut String, paragraph: &str) {
    if !current.is_empty() {
        current.push_str("\n\n");
    }
    current.push_str(paragraph);
}

fn flush(chunks: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        chunks.push(std::mem::take(current));
    }
}

// Last resort for a single paragraph longer than the whole limit: cut it into
// limit-sized character pieces so every message stays under the cap.
fn hard_split(text: &str, limit: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if current.chars().count() == limit {
            pieces.push(std::mem::take(&mut current));
        }
        current.push(character);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

pub(crate) struct HttpTransport {
    token: String,
    http: reqwest::blocking::Client,
}

impl HttpTransport {
    pub(crate) fn new(token: String) -> BotResult<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(LONG_POLL_SECONDS + HTTP_TIMEOUT_SLACK_SECONDS))
            .build()?;
        Ok(Self { token, http })
    }

    fn method_url(&self, method: &str) -> String {
        format!("{API_BASE}/bot{}/{method}", self.token)
    }
}

impl BotTransport for HttpTransport {
    fn get_updates(&self, offset: i64) -> BotResult<Value> {
        let body = json!({"offset": offset, "timeout": LONG_POLL_SECONDS});
        post_json(&self.http, &self.method_url("getUpdates"), &body)
    }

    fn send_message(&self, chat_id: i64, text: &str) -> BotResult<()> {
        let body = json!({"chat_id": chat_id, "text": text});
        post_json(&self.http, &self.method_url("sendMessage"), &body)?;
        Ok(())
    }

    fn send_message_html(&self, chat_id: i64, html: &str) -> BotResult<()> {
        let body = json!({"chat_id": chat_id, "text": html, "parse_mode": "HTML"});
        post_json(&self.http, &self.method_url("sendMessage"), &body)?;
        Ok(())
    }

    fn send_chat_action(&self, chat_id: i64, action: &str) -> BotResult<()> {
        let body = json!({"chat_id": chat_id, "action": action});
        post_json(&self.http, &self.method_url("sendChatAction"), &body)?;
        Ok(())
    }
}

fn post_json(http: &reqwest::blocking::Client, url: &str, body: &Value) -> BotResult<Value> {
    let resp = http.post(url).json(body).send()?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        return Err(format!("HTTP {status}: {text}").into());
    }
    Ok(resp.json::<Value>()?)
}

#[cfg(test)]
mod tests {
    use super::{TELEGRAM_MAX_CHARS, parse_updates, split_message};
    use serde_json::json;

    #[test]
    fn parses_user_chat_and_text_from_an_update() {
        let value = json!({"result": [{
            "update_id": 10,
            "message": {"from": {"id": 42}, "chat": {"id": 7}, "text": "hello"}
        }]});
        let updates = parse_updates(&value);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 10);
        let Some(message) = updates[0].message.as_ref() else {
            unreachable!("message parsed")
        };
        assert_eq!(message.user_id, 42);
        assert_eq!(message.chat_id, 7);
        assert_eq!(message.text, "hello");
    }

    #[test]
    fn a_malformed_update_is_skipped_but_still_advances_the_offset() {
        // No text field — nothing to answer, but update_id is intact.
        let value = json!({"result": [{
            "update_id": 5,
            "message": {"from": {"id": 42}, "chat": {"id": 7}}
        }]});
        let updates = parse_updates(&value);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 5);
        assert!(updates[0].message.is_none());
    }

    #[test]
    fn an_entry_without_an_update_id_is_dropped_entirely() {
        let value = json!({"result": [{"message": {"text": "x"}}]});
        assert!(parse_updates(&value).is_empty());
    }

    #[test]
    fn a_short_answer_stays_one_message() {
        let chunks = split_message("just a short reply", TELEGRAM_MAX_CHARS);
        assert_eq!(chunks, vec!["just a short reply".to_string()]);
    }

    #[test]
    fn a_long_answer_splits_on_paragraph_boundaries_in_order() {
        let para = "x".repeat(3000);
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = split_message(&text, TELEGRAM_MAX_CHARS);
        assert!(chunks.len() >= 2, "must split");
        for chunk in &chunks {
            assert!(chunk.chars().count() <= TELEGRAM_MAX_CHARS);
        }
        // Order preserved: concatenating back yields the paragraphs in sequence.
        let rejoined = chunks.join("\n\n");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn a_single_oversized_paragraph_is_hard_split() {
        let text = "y".repeat(TELEGRAM_MAX_CHARS * 2 + 50);
        let chunks = split_message(&text, TELEGRAM_MAX_CHARS);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= TELEGRAM_MAX_CHARS);
        }
        assert_eq!(chunks.concat(), text);
    }
}
