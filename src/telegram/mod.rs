mod bot;
mod night;
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
    // Dream, exploration and the daily report all run on the heavy seat
    // (MEMORY_MODEL); only the live memory pass consults the split seats.
    let mut night = night::NightCycle::from_env(
        Arc::clone(&seats.heavy),
        Arc::clone(&embed),
        search.clone(),
    )?;
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
    gateway.poll_loop(&mut ctx, &mut night);
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
// chain stays under the argument cap. The night cycle is not in here: it is
// used only at the top of the loop, and keeping it out lets the tests drive
// poll_once without constructing one.
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

    fn poll_loop(&mut self, ctx: &mut PollCtx, night: &mut night::NightCycle) {
        while !ctx.shutdown.load(Ordering::Relaxed) {
            // The idle path (no updates) is where the night cycle and the daily
            // report run: a poll that returned messages spent this cycle
            // answering them — and changed state, so the night cycle must
            // re-evaluate next chance.
            if self.poll_once(ctx) {
                night.maybe_run(ctx.shutdown);
                ctx.ping.maybe_send(ctx.transport);
                ctx.report.maybe_send(ctx.transport);
            } else {
                night.mark_dirty();
            }
        }
    }

    // Returns true when the poll was idle (no updates delivered) — the signal
    // the night cycle gates on.
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
        let clients = turn::TurnClients {
            llm: &*self.chat_llm,
            embed: &*self.embed,
            search: self.search.as_deref(),
            style_hint: Some(TELEGRAM_STYLE_HINT),
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
mod tests {
    use super::bot::BotTransport;
    use super::{
        CompletedTurn, DaySession, Gateway, MemoryPassRunner, PendingMemoryPass, PollCtx,
        TurnHandler, ping, report,
    };
    use crate::api::{Chunk, EmbeddingClient, LlmClient, TurnOutcome};
    use crate::memory::MemorySeats;
    use serde_json::{Value, json};
    use std::error::Error;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    // The gateway poll tests never trigger a report or verdict, so the LLM is
    // never called — this stub only exists to construct a ReportCycle.
    struct StubLlm;

    impl LlmClient for StubLlm {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unreachable!("poll tests never call the LLM")
        }
        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {}
        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unreachable!("poll tests never call the LLM")
        }
    }

    struct StubEmbed;

    impl EmbeddingClient for StubEmbed {
        fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
            unreachable!("gateway tests never embed")
        }
    }

    fn fake_report() -> report::ReportCycle {
        report::ReportCycle::for_test(Arc::new(StubLlm), 42)
    }

    // Drives one poll, scoping the PollCtx borrows so the tests can assert on
    // the handler and transport afterwards.
    fn poll(
        gateway: &mut Gateway,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        report: &mut report::ReportCycle,
        stop: &AtomicBool,
    ) -> bool {
        let mut ping = ping::PingCycle::new(42);
        let mut ctx = PollCtx {
            transport,
            handler,
            report,
            ping: &mut ping,
            shutdown: stop,
        };
        gateway.poll_once(&mut ctx)
    }

    // Replays queued getUpdates results and records everything the gateway
    // tries to send — same fake-client shape as agent.rs's ScriptedClient.
    struct ScriptedTransport {
        batches: Mutex<Vec<Result<Value, String>>>,
        sent: Mutex<Vec<(i64, String)>>,
        actions: Mutex<Vec<(i64, String)>>,
        events: Option<Arc<Mutex<Vec<String>>>>,
    }

    impl ScriptedTransport {
        fn new(batches: Vec<Result<Value, String>>) -> Self {
            let mut queued = batches;
            queued.reverse();
            Self {
                batches: Mutex::new(queued),
                sent: Mutex::new(Vec::new()),
                actions: Mutex::new(Vec::new()),
                events: None,
            }
        }

        fn recording(batches: Vec<Result<Value, String>>, events: Arc<Mutex<Vec<String>>>) -> Self {
            let mut transport = Self::new(batches);
            transport.events = Some(events);
            transport
        }

        fn sent_count(&self) -> usize {
            match self.sent.lock() {
                Ok(sent) => sent.len(),
                Err(_) => 0,
            }
        }

        fn sent_texts(&self) -> Vec<String> {
            match self.sent.lock() {
                Ok(sent) => {
                    let mut texts = Vec::with_capacity(sent.len());
                    for (_, text) in &*sent {
                        texts.push(text.clone());
                    }
                    texts
                }
                Err(_) => Vec::new(),
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

        fn send_message(
            &self,
            chat_id: i64,
            text: &str,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            self.record_send();
            if let Ok(mut sent) = self.sent.lock() {
                sent.push((chat_id, text.to_string()));
            }
            Ok(())
        }

        fn send_message_html(
            &self,
            chat_id: i64,
            html: &str,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            self.record_send();
            if let Ok(mut sent) = self.sent.lock() {
                sent.push((chat_id, html.to_string()));
            }
            Ok(())
        }

        fn send_chat_action(
            &self,
            chat_id: i64,
            action: &str,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Ok(mut actions) = self.actions.lock() {
                actions.push((chat_id, action.to_string()));
            }
            Ok(())
        }
    }

    impl ScriptedTransport {
        fn record_send(&self) {
            let Some(events) = &self.events else {
                return;
            };
            if let Ok(mut events) = events.lock() {
                events.push("send".to_string());
            }
        }
    }

    struct FakeHandler {
        seen: Vec<String>,
        reply: String,
        events: Option<Arc<Mutex<Vec<String>>>>,
    }

    impl TurnHandler for FakeHandler {
        fn handle(&mut self, text: &str) -> Result<CompletedTurn, Box<dyn Error + Send + Sync>> {
            self.seen.push(text.to_string());
            if let Some(events) = &self.events
                && let Ok(mut events) = events.lock()
            {
                events.push(format!("context:{text}"));
                events.push(format!("assistant-appended:{text}"));
            }
            Ok(fake_completed(&self.reply, text))
        }
    }

    struct FailingHandler;

    impl TurnHandler for FailingHandler {
        fn handle(&mut self, _text: &str) -> Result<CompletedTurn, Box<dyn Error + Send + Sync>> {
            Err("generation failed".into())
        }
    }

    struct NoopMemoryRunner;

    impl MemoryPassRunner for NoopMemoryRunner {
        fn run(&self, _pass: PendingMemoryPass) -> Result<(), Box<dyn Error + Send + Sync>> {
            Ok(())
        }
    }

    struct RecordingMemoryRunner {
        events: Arc<Mutex<Vec<String>>>,
        fail: bool,
    }

    impl MemoryPassRunner for RecordingMemoryRunner {
        fn run(&self, pass: PendingMemoryPass) -> Result<(), Box<dyn Error + Send + Sync>> {
            if let Ok(mut events) = self.events.lock() {
                events.push(format!("memory:{}", pass.user_msg));
            }
            if self.fail {
                return Err("memory failed".into());
            }
            Ok(())
        }
    }

    fn fake_completed(answer: &str, text: &str) -> CompletedTurn {
        CompletedTurn {
            answer: answer.to_string(),
            memory_pass: PendingMemoryPass {
                seats: fake_seats(),
                user_msg: text.to_string(),
                answer: answer.to_string(),
                turn_id: "turn-test".to_string(),
            },
        }
    }

    fn fake_seats() -> MemorySeats {
        MemorySeats {
            classifier: Arc::new(StubLlm),
            scribe: Arc::new(StubLlm),
            heavy: Arc::new(StubLlm),
            embed: Arc::new(StubEmbed),
        }
    }

    fn fake_handler(reply: &str) -> FakeHandler {
        FakeHandler {
            seen: Vec::new(),
            reply: reply.to_string(),
            events: None,
        }
    }

    fn recorded_events(events: &Mutex<Vec<String>>) -> Vec<String> {
        let Ok(events) = events.lock() else {
            unreachable!("events lock held")
        };
        events.clone()
    }

    fn fast_gateway(allowed_user_id: i64) -> Gateway {
        Gateway {
            allowed_user_id,
            offset: 0,
            backoff: Duration::ZERO,
            memory_runner: Box::new(NoopMemoryRunner),
        }
    }

    fn recording_gateway(events: Arc<Mutex<Vec<String>>>, fail: bool) -> Gateway {
        Gateway {
            allowed_user_id: 42,
            offset: 0,
            backoff: Duration::ZERO,
            memory_runner: Box::new(RecordingMemoryRunner { events, fail }),
        }
    }

    fn update_batch(update_id: i64, user_id: i64, text: &str) -> Value {
        json!({"result": [{
            "update_id": update_id,
            "message": {"from": {"id": user_id}, "chat": {"id": 100}, "text": text}
        }]})
    }

    fn two_update_batch() -> Value {
        json!({"result": [
            {
                "update_id": 1,
                "message": {"from": {"id": 42}, "chat": {"id": 100}, "text": "first"}
            },
            {
                "update_id": 2,
                "message": {"from": {"id": 42}, "chat": {"id": 100}, "text": "second"}
            }
        ]})
    }

    #[test]
    fn an_update_from_the_wrong_user_is_dropped_with_no_reply() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 999, "hi"))]);
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);
        assert!(handler.seen.is_empty(), "handler never ran");
        assert_eq!(transport.sent_count(), 0, "nothing sent");
    }

    #[test]
    fn an_update_from_the_allowed_user_is_processed_and_answered() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 42, "hi"))]);
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);
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
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);
        assert!(handler.seen.is_empty());
        assert_eq!(gateway.offset, 9);
    }

    #[test]
    fn the_poll_loop_survives_an_error_then_processes_the_next_batch() {
        let transport = ScriptedTransport::new(vec![
            Err("network down".to_string()),
            Ok(update_batch(3, 42, "after error")),
        ]);
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = fast_gateway(42);
        // First poll errors (and sleeps zero), second delivers the update.
        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);
        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);
        assert_eq!(handler.seen, vec!["after error".to_string()]);
    }

    #[test]
    fn answer_is_sent_before_memory_and_memory_finishes_before_the_next_turn() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let transport =
            ScriptedTransport::recording(vec![Ok(two_update_batch())], Arc::clone(&events));
        let mut handler = FakeHandler {
            seen: Vec::new(),
            reply: "answer".to_string(),
            events: Some(Arc::clone(&events)),
        };
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = recording_gateway(Arc::clone(&events), false);

        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);

        let recorded = recorded_events(&events);
        assert_eq!(
            recorded,
            vec![
                "context:first",
                "assistant-appended:first",
                "send",
                "memory:first",
                "context:second",
                "assistant-appended:second",
                "send",
                "memory:second",
            ]
        );
    }

    #[test]
    fn memory_failure_after_send_does_not_send_a_broken_turn_reply() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let transport =
            ScriptedTransport::recording(vec![Ok(update_batch(1, 42, "hi"))], Arc::clone(&events));
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = recording_gateway(Arc::clone(&events), true);

        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);

        assert_eq!(transport.sent_texts().len(), 1);
        assert!(transport.sent_texts()[0].contains("answer"));
        assert_eq!(recorded_events(&events), ["send", "memory:hi"]);
    }

    #[test]
    fn generation_failure_still_sends_the_broken_turn_reply() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 42, "hi"))]);
        let mut handler = FailingHandler;
        let stop = AtomicBool::new(false);
        let mut report = fake_report();
        let mut gateway = fast_gateway(42);

        poll(&mut gateway, &transport, &mut handler, &mut report, &stop);

        assert_eq!(transport.sent_texts(), vec![super::BROKEN_TURN_REPLY]);
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
