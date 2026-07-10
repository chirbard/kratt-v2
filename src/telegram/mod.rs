mod bot;
mod render;

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::api::{EmbeddingClient, LlmClient};
use crate::memory::{self, MemoryEvent};
use crate::runlog;
use crate::session;
use crate::timeutil;
use crate::turn;
use crate::web::SearchClient;

use bot::{
    BotTransport, HttpTransport, TELEGRAM_MAX_CHARS, Update, parse_updates, split_message,
};

// Backoff after a poll failure (network down, non-200, bad JSON): log, sleep,
// re-poll — the daemon must never crash on a bad update.
const POLL_ERROR_BACKOFF: Duration = Duration::from_secs(3);
const BROKEN_TURN_REPLY: &str = "something broke on my end — it's logged.";
// The model has no idea it is talking to a phone unless told: tables wrap
// into unreadable soup at phone width even inside <pre>.
const TELEGRAM_STYLE_HINT: &str = "Formatting: you are answering inside Telegram on a phone screen. NEVER use markdown tables — present tabular information as short labeled bullet lists (one item per row, 'label: value' pairs). Keep answers compact; prefer bullet lists and short paragraphs over long headers. Use code blocks only for actual code.";
const EMPTY_ANSWER_REPLY: &str = "(no response)";

struct Config {
    token: String,
    allowed_user_id: i64,
}

impl Config {
    fn from_env() -> Result<Self, Box<dyn Error>> {
        let token = match std::env::var("TELEGRAM_BOT_TOKEN") {
            Ok(token) if !token.trim().is_empty() => token,
            _ => return Err("TELEGRAM_BOT_TOKEN is required in --telegram mode".into()),
        };
        let raw = match std::env::var("TELEGRAM_USER_ID") {
            Ok(raw) => raw,
            Err(_) => return Err("TELEGRAM_USER_ID is required in --telegram mode".into()),
        };
        let allowed_user_id = match raw.trim().parse::<i64>() {
            Ok(user_id) => user_id,
            Err(_) => return Err("TELEGRAM_USER_ID must be an integer user id".into()),
        };
        Ok(Self {
            token,
            allowed_user_id,
        })
    }
}

pub fn run(
    chat_llm: Arc<dyn LlmClient>,
    memory_llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
) -> Result<(), Box<dyn Error>> {
    let config = Config::from_env()?;
    let transport = match HttpTransport::new(config.token) {
        Ok(transport) => transport,
        Err(err) => return Err(err.to_string().into()),
    };
    let mut handler = LiveTurnHandler::new(chat_llm, memory_llm, embed, search);
    let shutdown = install_shutdown_flag();
    let banner = "kratt telegram gateway ready — long polling (Ctrl-C to stop)";
    println!("{banner}");
    runlog::log("tg", banner);
    let mut gateway = Gateway::new(config.allowed_user_id);
    gateway.poll_loop(&transport, &mut handler, &shutdown);
    memory::sync_push_best_effort("telegram exit");
    runlog::log("tg", "gateway stopped");
    Ok(())
}

// SIGTERM/SIGINT flip the flag; the loop finishes the in-flight turn (or the
// current long poll, up to ~50s) and exits on its next boundary check. A
// second signal while the flag is set force-quits immediately. A watcher
// thread announces the graceful shutdown the moment the flag flips, because
// the main thread may be blocked in a poll or a memory pass at that point.
fn install_shutdown_flag() -> Arc<AtomicBool> {
    let shutdown = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        // Registration order matters: the conditional shutdown runs first and
        // only fires when the flag is already set — i.e. on the second signal.
        let conditional =
            signal_hook::flag::register_conditional_shutdown(signal, 1, Arc::clone(&shutdown));
        if let Err(err) = conditional {
            runlog::log("tg", &format!("signal register failed: {err}"));
        }
        if let Err(err) = signal_hook::flag::register(signal, Arc::clone(&shutdown)) {
            runlog::log("tg", &format!("signal register failed: {err}"));
        }
    }
    spawn_shutdown_announcer(Arc::clone(&shutdown));
    shutdown
}

fn spawn_shutdown_announcer(shutdown: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
        }
        let notice = "shutting down gracefully — finishing in-flight turn and saving memory (Ctrl-C again to force quit)";
        eprintln!("{notice}");
        runlog::log("tg", notice);
    });
}

trait TurnHandler {
    fn handle(&mut self, text: &str) -> Result<String, Box<dyn Error + Send + Sync>>;
}

struct Gateway {
    allowed_user_id: i64,
    offset: i64,
    backoff: Duration,
}

impl Gateway {
    fn new(allowed_user_id: i64) -> Self {
        Self {
            allowed_user_id,
            offset: 0,
            backoff: POLL_ERROR_BACKOFF,
        }
    }

    fn poll_loop(
        &mut self,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        shutdown: &AtomicBool,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            self.poll_once(transport, handler, shutdown);
        }
    }

    fn poll_once(
        &mut self,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        shutdown: &AtomicBool,
    ) {
        match transport.get_updates(self.offset) {
            Ok(value) => self.handle_batch(transport, handler, &value, shutdown),
            Err(err) => {
                runlog::log("tg", &format!("poll error: {err}"));
                std::thread::sleep(self.backoff);
            }
        }
    }

    fn handle_batch(
        &mut self,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        value: &serde_json::Value,
        shutdown: &AtomicBool,
    ) {
        for raw in parse_updates(value) {
            // Leave the offset untouched on shutdown so an unhandled update is
            // re-delivered when the daemon restarts.
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            self.offset = self.offset.max(raw.update_id + 1);
            if let Some(message) = raw.message {
                self.dispatch(transport, handler, &message);
            }
        }
    }

    fn dispatch(
        &self,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        message: &Update,
    ) {
        if message.user_id != self.allowed_user_id {
            runlog::log(
                "tg",
                &format!("dropped update from unauthorized user {}", message.user_id),
            );
            return;
        }
        runlog::log(
            "tg",
            &format!("update from {} in chat {}", message.user_id, message.chat_id),
        );
        let outcome = with_typing_indicator(transport, message.chat_id, || {
            handler.handle(&message.text)
        });
        match outcome {
            Ok(answer) => send_answer(transport, message.chat_id, &answer),
            Err(err) => {
                runlog::log("tg", &format!("turn failed: {err}"));
                let _ = transport.send_message(message.chat_id, BROKEN_TURN_REPLY);
            }
        }
    }
}

// Telegram's typing status expires after ~5 seconds, and a turn (LLM + tool
// loop) can run for minutes — re-send the action on a keep-alive cadence for
// the whole turn so the user sees the bot working the moment they hit send.
const TYPING_KEEPALIVE: Duration = Duration::from_secs(4);

fn with_typing_indicator<Answer>(
    transport: &dyn BotTransport,
    chat_id: i64,
    run_turn: impl FnOnce() -> Answer,
) -> Answer {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                let _ = transport.send_chat_action(chat_id, "typing");
                sleep_until_done(&done, TYPING_KEEPALIVE);
            }
        });
        let answer = run_turn();
        done.store(true, Ordering::Relaxed);
        answer
    })
}

// Sleep in short slices so the keep-alive thread exits promptly when the turn
// finishes instead of holding the scope open for a full cadence interval.
fn sleep_until_done(done: &AtomicBool, total: Duration) {
    let slice = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < total && !done.load(Ordering::Relaxed) {
        std::thread::sleep(slice);
        slept += slice;
    }
}

// Split on the raw markdown (stable paragraph boundaries), render each chunk
// to Telegram HTML, and fall back to the plain chunk when Telegram rejects
// the HTML (a chunk boundary can cut a tag pair) — the answer always arrives.
fn send_answer(transport: &dyn BotTransport, chat_id: i64, answer: &str) {
    let text = if answer.trim().is_empty() {
        EMPTY_ANSWER_REPLY
    } else {
        answer
    };
    for chunk in split_message(text, TELEGRAM_MAX_CHARS) {
        if let Err(err) = send_chunk(transport, chat_id, &chunk) {
            runlog::log("tg", &format!("send failed: {err}"));
            return;
        }
    }
    runlog::log("tg", "answer sent");
}

fn send_chunk(
    transport: &dyn BotTransport,
    chat_id: i64,
    chunk: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let html = render::markdown_to_telegram_html(chunk);
    match transport.send_message_html(chat_id, &html) {
        Ok(()) => Ok(()),
        Err(err) => {
            runlog::log("tg", &format!("html send rejected, falling back to plain: {err}"));
            transport.send_message(chat_id, chunk)
        }
    }
}

struct LiveTurnHandler {
    chat_llm: Arc<dyn LlmClient>,
    memory_llm: Arc<dyn LlmClient>,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    day: DaySession,
}

impl LiveTurnHandler {
    fn new(
        chat_llm: Arc<dyn LlmClient>,
        memory_llm: Arc<dyn LlmClient>,
        embed: Arc<dyn EmbeddingClient>,
        search: Option<Arc<dyn SearchClient>>,
    ) -> Self {
        Self {
            chat_llm,
            memory_llm,
            embed,
            search,
            day: DaySession::new(),
        }
    }

    fn session_path(&mut self) -> PathBuf {
        let (session_id, new_day) = self.day.resolve(&timeutil::today_iso());
        if new_day {
            memory::register_session(&session_id, &timeutil::timestamp_now());
            runlog::log("tg", &format!("opened session {session_id}"));
        }
        session_path_for_id(&session_id)
    }
}

impl TurnHandler for LiveTurnHandler {
    // Strictly sequential: answer, then run the memory pass to completion,
    // before the caller polls again. Serializes all DB writes like the TUI.
    fn handle(&mut self, text: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let session_path = self.session_path();
        let turn_id = session::append_entry(&session_path, "user", text);
        let clients = turn::TurnClients {
            llm: &*self.chat_llm,
            embed: &*self.embed,
            search: self.search.as_deref(),
            style_hint: Some(TELEGRAM_STYLE_HINT),
        };
        let answer = turn::run_turn_collecting(&clients, text, &session_path, &turn_id)?;
        session::append_entry(&session_path, "assistant", &answer);
        run_memory_to_completion(Arc::clone(&self.memory_llm), text, &answer, &turn_id);
        Ok(answer)
    }
}

// Runs the same memory classifier/pass as the TUI, but inline: the events are
// drained to completion here rather than into a panel.
fn run_memory_to_completion(
    llm: Arc<dyn LlmClient>,
    user_msg: &str,
    assistant_msg: &str,
    turn_id: &str,
) {
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    if let Err(err) = memory::run_memory_agent(llm, user_msg, assistant_msg, turn_id, sender) {
        runlog::log("tg", &format!("memory pass failed: {err}"));
    }
    // The agent already returned and dropped its sender; drain the buffered
    // events so the pass is fully accounted for before the next poll.
    for _event in receiver {}
}

// Session = calendar day: episodic origin_session scoping and the future daily
// 1:1 get a natural "day" unit. Pure derivation — DB registration happens in
// the handler only when the day actually rolls over.
struct DaySession {
    day: String,
    session_id: String,
}

impl DaySession {
    fn new() -> Self {
        Self {
            day: String::new(),
            session_id: String::new(),
        }
    }

    // Returns (session_id, is_new_day). Same local day → same session id.
    fn resolve(&mut self, today: &str) -> (String, bool) {
        if self.day == today {
            return (self.session_id.clone(), false);
        }
        self.day = today.to_string();
        self.session_id = format!("tg-{today}");
        (self.session_id.clone(), true)
    }
}

fn session_path_for_id(session_id: &str) -> PathBuf {
    session::sessions_dir().join(format!("{session_id}.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::{DaySession, Gateway, TurnHandler};
    use super::bot::BotTransport;
    use std::error::Error;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;
    use serde_json::{Value, json};

    // Replays queued getUpdates results and records everything the gateway
    // tries to send — same fake-client shape as agent.rs's ScriptedClient.
    struct ScriptedTransport {
        batches: Mutex<Vec<Result<Value, String>>>,
        sent: Mutex<Vec<(i64, String)>>,
        actions: Mutex<Vec<(i64, String)>>,
    }

    impl ScriptedTransport {
        fn new(batches: Vec<Result<Value, String>>) -> Self {
            let mut queued = batches;
            queued.reverse();
            Self {
                batches: Mutex::new(queued),
                sent: Mutex::new(Vec::new()),
                actions: Mutex::new(Vec::new()),
            }
        }

        fn sent_count(&self) -> usize {
            match self.sent.lock() {
                Ok(sent) => sent.len(),
                Err(_) => 0,
            }
        }
    }

    impl BotTransport for ScriptedTransport {
        fn get_updates(&self, _offset: i64) -> Result<Value, Box<dyn Error + Send + Sync>> {
            let Ok(mut batches) = self.batches.lock() else {
                unreachable!("lock held")
            };
            match batches.pop() {
                Some(Ok(value)) => Ok(value),
                Some(Err(err)) => Err(err.into()),
                None => Ok(json!({"result": []})),
            }
        }

        fn send_message(&self, chat_id: i64, text: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Ok(mut sent) = self.sent.lock() {
                sent.push((chat_id, text.to_string()));
            }
            Ok(())
        }

        fn send_message_html(&self, chat_id: i64, html: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Ok(mut sent) = self.sent.lock() {
                sent.push((chat_id, html.to_string()));
            }
            Ok(())
        }

        fn send_chat_action(&self, chat_id: i64, action: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Ok(mut actions) = self.actions.lock() {
                actions.push((chat_id, action.to_string()));
            }
            Ok(())
        }
    }

    struct FakeHandler {
        seen: Vec<String>,
        reply: String,
    }

    impl TurnHandler for FakeHandler {
        fn handle(&mut self, text: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            self.seen.push(text.to_string());
            Ok(self.reply.clone())
        }
    }

    fn fast_gateway(allowed_user_id: i64) -> Gateway {
        Gateway {
            allowed_user_id,
            offset: 0,
            backoff: Duration::ZERO,
        }
    }

    fn update_batch(update_id: i64, user_id: i64, text: &str) -> Value {
        json!({"result": [{
            "update_id": update_id,
            "message": {"from": {"id": user_id}, "chat": {"id": 100}, "text": text}
        }]})
    }

    #[test]
    fn an_update_from_the_wrong_user_is_dropped_with_no_reply() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 999, "hi"))]);
        let mut handler = FakeHandler {
            seen: Vec::new(),
            reply: "answer".to_string(),
        };
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);
        gateway.poll_once(&transport, &mut handler, &stop);
        assert!(handler.seen.is_empty(), "handler never ran");
        assert_eq!(transport.sent_count(), 0, "nothing sent");
    }

    #[test]
    fn an_update_from_the_allowed_user_is_processed_and_answered() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 42, "hi"))]);
        let mut handler = FakeHandler {
            seen: Vec::new(),
            reply: "answer".to_string(),
        };
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);
        gateway.poll_once(&transport, &mut handler, &stop);
        assert_eq!(handler.seen, vec!["hi".to_string()]);
        assert_eq!(transport.sent_count(), 1);
    }

    #[test]
    fn the_offset_advances_past_a_skipped_malformed_update() {
        // update_id 8, but no text — skipped, offset must still move past it.
        let batch = json!({"result": [{
            "update_id": 8,
            "message": {"from": {"id": 42}, "chat": {"id": 100}}
        }]});
        let transport = ScriptedTransport::new(vec![Ok(batch)]);
        let mut handler = FakeHandler {
            seen: Vec::new(),
            reply: "answer".to_string(),
        };
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);
        gateway.poll_once(&transport, &mut handler, &stop);
        assert!(handler.seen.is_empty());
        assert_eq!(gateway.offset, 9);
    }

    #[test]
    fn the_poll_loop_survives_an_error_then_processes_the_next_batch() {
        let transport = ScriptedTransport::new(vec![
            Err("network down".to_string()),
            Ok(update_batch(3, 42, "after error")),
        ]);
        let mut handler = FakeHandler {
            seen: Vec::new(),
            reply: "answer".to_string(),
        };
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);
        // First poll errors (and sleeps zero), second delivers the update.
        gateway.poll_once(&transport, &mut handler, &stop);
        gateway.poll_once(&transport, &mut handler, &stop);
        assert_eq!(handler.seen, vec!["after error".to_string()]);
    }

    #[test]
    fn the_same_day_reuses_the_session_id_and_a_rollover_makes_a_new_one() {
        let mut day = DaySession::new();
        let (first, new_first) = day.resolve("2026-07-10");
        let (second, new_second) = day.resolve("2026-07-10");
        assert_eq!(first, "tg-2026-07-10");
        assert!(new_first, "first resolve opens a day");
        assert_eq!(first, second, "same day → same session id");
        assert!(!new_second, "same day is not a new day");

        let (third, new_third) = day.resolve("2026-07-11");
        assert_eq!(third, "tg-2026-07-11");
        assert!(new_third, "rollover opens a new day");
        assert_ne!(second, third);
    }
}
