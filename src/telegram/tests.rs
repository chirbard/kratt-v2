    use super::bot::BotTransport;
    use super::{
        CompletedTurn, DaySession, Gateway, MemoryPassRunner, PendingMemoryPass, PollCtx,
        TurnHandler, ping,
    };
    use crate::api::{Chunk, EmbeddingClient, LlmClient, TurnOutcome};
    use crate::memory::MemorySeats;
    use serde_json::{Value, json};
    use std::error::Error;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct StubLlm;

    impl LlmClient for StubLlm {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unreachable!("gateway tests never collect")
        }

        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {}

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unreachable!("gateway tests never stream")
        }
    }

    struct StubEmbed;

    impl EmbeddingClient for StubEmbed {
        fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
            unreachable!("gateway tests never embed")
        }
    }

    // Drives one poll, scoping the PollCtx borrows so the tests can assert on
    // the handler and transport afterwards.
    fn poll(
        gateway: &mut Gateway,
        transport: &dyn BotTransport,
        handler: &mut dyn TurnHandler,
        stop: &AtomicBool,
    ) -> bool {
        let mut ping = ping::PingCycle::new(42);
        let mut ctx = PollCtx {
            transport,
            handler,
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
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &stop);
        assert!(handler.seen.is_empty(), "handler never ran");
        assert_eq!(transport.sent_count(), 0, "nothing sent");
    }

    #[test]
    fn an_update_from_the_allowed_user_is_processed_and_answered() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 42, "hi"))]);
        let mut handler = fake_handler("answer");
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &stop);
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
        let mut gateway = fast_gateway(42);
        poll(&mut gateway, &transport, &mut handler, &stop);
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
        let mut gateway = fast_gateway(42);
        // First poll errors (and sleeps zero), second delivers the update.
        poll(&mut gateway, &transport, &mut handler, &stop);
        poll(&mut gateway, &transport, &mut handler, &stop);
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
        let mut gateway = recording_gateway(Arc::clone(&events), false);

        poll(&mut gateway, &transport, &mut handler, &stop);

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
        let mut gateway = recording_gateway(Arc::clone(&events), true);

        poll(&mut gateway, &transport, &mut handler, &stop);

        assert_eq!(transport.sent_texts().len(), 1);
        assert!(transport.sent_texts()[0].contains("answer"));
        assert_eq!(recorded_events(&events), ["send", "memory:hi"]);
    }

    #[test]
    fn generation_failure_still_sends_the_broken_turn_reply() {
        let transport = ScriptedTransport::new(vec![Ok(update_batch(1, 42, "hi"))]);
        let mut handler = FailingHandler;
        let stop = AtomicBool::new(false);
        let mut gateway = fast_gateway(42);

        poll(&mut gateway, &transport, &mut handler, &stop);

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
