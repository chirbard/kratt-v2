mod bot;
mod ping;
mod render;
mod report;

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::api::{EmbeddingClient, LlmClient};
use crate::memory::{self, MemoryEvent, MemorySeats};
use crate::runlog;
use crate::session;
use crate::timeutil;
use crate::trace::{TraceContext, TracingClient};
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
    seats: MemorySeats,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
) -> Result<(), Box<dyn Error>> {
    let config = Config::from_env()?;
    let transport = match HttpTransport::new(config.token) {
        Ok(transport) => transport,
        Err(err) => return Err(err.to_string().into()),
    };
    let mut report = report::ReportCycle::from_env(
        Arc::clone(&seats.heavy),
        config.allowed_user_id,
    )?;
    let mut ping = ping::PingCycle::new(config.allowed_user_id);
    let mut handler = LiveTurnHandler::new(chat_llm, seats, embed, search);
    let shutdown = install_shutdown_flag();
    let banner = "kratt telegram gateway ready — long polling (Ctrl-C to stop)";
    println!("{banner}");
    runlog::log("tg", banner);
    let mut gateway = Gateway::new(config.allowed_user_id);
    let mut ctx = PollCtx {
        transport: &transport,
        handler: &mut handler,
        report: &mut report,
        ping: &mut ping,
        shutdown: &shutdown,
    };
    gateway.poll_loop(&mut ctx);
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
    fn handle(&mut self, text: &str) -> Result<CompletedTurn, Box<dyn Error + Send + Sync>>;
}

struct CompletedTurn {
    answer: String,
    memory_pass: PendingMemoryPass,
}

struct PendingMemoryPass {
    seats: MemorySeats,
    user_msg: String,
    answer: String,
    turn_id: String,
}

trait MemoryPassRunner {
    fn run(&self, pass: PendingMemoryPass) -> Result<(), Box<dyn Error + Send + Sync>>;
}

struct LiveMemoryPassRunner;

impl MemoryPassRunner for LiveMemoryPassRunner {
    fn run(&self, pass: PendingMemoryPass) -> Result<(), Box<dyn Error + Send + Sync>> {
        run_memory_to_completion(pass)
    }
}

struct Gateway {
    allowed_user_id: i64,
    offset: i64,
    backoff: Duration,
    memory_runner: Box<dyn MemoryPassRunner>,
}

// The per-poll references threaded down the dispatch chain, bundled so the
// chain stays under the argument cap.
struct PollCtx<'scope> {
    transport: &'scope dyn BotTransport,
    handler: &'scope mut dyn TurnHandler,
    report: &'scope mut report::ReportCycle,
    ping: &'scope mut ping::PingCycle,
    shutdown: &'scope AtomicBool,
}

impl Gateway {
    fn new(allowed_user_id: i64) -> Self {
        Self {
            allowed_user_id,
            offset: 0,
            backoff: POLL_ERROR_BACKOFF,
            memory_runner: Box::new(LiveMemoryPassRunner),
        }
    }

    fn poll_loop(&mut self, ctx: &mut PollCtx) {
        while !ctx.shutdown.load(Ordering::Relaxed) {
            if self.poll_once(ctx) {
                ctx.ping.maybe_send(ctx.transport);
                ctx.report.maybe_send(ctx.transport);
            }
        }
    }

    // Returns true when the poll was idle (no updates delivered).
    fn poll_once(&mut self, ctx: &mut PollCtx) -> bool {
        match ctx.transport.get_updates(self.offset) {
            Ok(value) => self.handle_batch(ctx, &value),
            Err(err) => {
                runlog::log("tg", &format!("poll error: {err}"));
                std::thread::sleep(self.backoff);
                false
            }
        }
    }

    fn handle_batch(&mut self, ctx: &mut PollCtx, value: &serde_json::Value) -> bool {
        let updates = parse_updates(value);
        let idle = updates.is_empty();
        for raw in updates {
            // Leave the offset untouched on shutdown so an unhandled update is
            // re-delivered when the daemon restarts.
            if ctx.shutdown.load(Ordering::Relaxed) {
                return idle;
            }
            self.offset = self.offset.max(raw.update_id + 1);
            if let Some(message) = raw.message {
                self.dispatch(ctx, &message);
            }
        }
        idle
    }

    fn dispatch(&self, ctx: &mut PollCtx, message: &Update) {
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
        // A bare-number reply to a pending report is a control message, not a
        // chat turn: score it and stop, skipping the turn and memory pipeline.
        if ctx.report.try_handle_verdict(ctx.transport, &message.text) {
            return;
        }
        let outcome = with_typing_indicator(ctx.transport, message.chat_id, || {
            ctx.handler.handle(&message.text)
        });
        match outcome {
            Ok(completed) => self.finish_turn(ctx.transport, message.chat_id, completed),
            Err(err) => {
                runlog::log("tg", &format!("turn failed: {err}"));
                let _ = ctx.transport.send_message(message.chat_id, BROKEN_TURN_REPLY);
            }
        }
    }

    fn finish_turn(&self, transport: &dyn BotTransport, chat_id: i64, completed: CompletedTurn) {
        send_answer(transport, chat_id, &completed.answer);
        if let Err(err) = self.memory_runner.run(completed.memory_pass) {
            runlog::log("tg", &format!("memory pass failed: {err}"));
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
    seats: MemorySeats,
    embed: Arc<dyn EmbeddingClient>,
    search: Option<Arc<dyn SearchClient>>,
    day: DaySession,
}

impl LiveTurnHandler {
    fn new(
        chat_llm: Arc<dyn LlmClient>,
        seats: MemorySeats,
        embed: Arc<dyn EmbeddingClient>,
        search: Option<Arc<dyn SearchClient>>,
    ) -> Self {
        Self {
            chat_llm,
            seats,
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
    fn handle(&mut self, text: &str) -> Result<CompletedTurn, Box<dyn Error + Send + Sync>> {
        let session_path = self.session_path();
        let prior_path = prior_session_path();
        let turn_id = session::append_entry(&session_path, "user", text);
        let trace_context = TraceContext::new("chat");
        let traced_llm = TracingClient::wrap(Arc::clone(&self.chat_llm), trace_context.clone());
        let clients = turn::TurnClients {
            llm: &*traced_llm,
            embed: &*self.embed,
            search: self.search.as_deref(),
            style_hint: Some(TELEGRAM_STYLE_HINT),
            trace: &trace_context,
        };
        let answer = turn::run_turn_collecting(
            &clients,
            text,
            (&session_path, prior_path.as_deref()),
            &turn_id,
        )?;
        session::append_entry(&session_path, "assistant", &answer);
        let memory_pass = PendingMemoryPass {
            seats: self.seats.clone(),
            user_msg: text.to_string(),
            answer: answer.clone(),
            turn_id,
        };
        Ok(CompletedTurn {
            answer,
            memory_pass,
        })
    }
}

// Runs the same memory classifier/pass as the TUI after delivery. The caller
// stays blocked until the buffered events are drained, preserving turn order.
fn run_memory_to_completion(pass: PendingMemoryPass) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let result = memory::run_memory_agent(
        pass.seats,
        &pass.user_msg,
        &pass.answer,
        &pass.turn_id,
        sender,
    );
    // The agent already returned and dropped its sender; drain the buffered
    // events so the pass is fully accounted for before the next poll.
    for _event in receiver {}
    result
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

// Yesterday's day-session file, if any — the Telegram page shows one
// continuous conversation across midnight, so a fresh morning session tops up
// its recent context from this tail (see turn::load_recent_with_prior).
fn prior_session_path() -> Option<PathBuf> {
    let path = session_path_for_id(&format!("tg-{}", timeutil::yesterday_iso()));
    if path.exists() { Some(path) } else { None }
}

fn session_path_for_id(session_id: &str) -> PathBuf {
    session::sessions_dir().join(format!("{session_id}.jsonl"))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
