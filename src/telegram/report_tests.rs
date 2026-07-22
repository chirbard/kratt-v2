    use std::error::Error;
    use std::sync::Arc;
    use std::sync::mpsc::Sender;

    use crate::api::{Chunk, LlmClient, TurnOutcome};
    use crate::memory::PassRun;

    use super::{DayDigest, ReportGate, parse_verdict, render_digest, verdict_session_ids};

    // The classification tests never phrase a report, so the LLM is never
    // called — this stub exists only to construct a ReportCycle.
    struct StubLlm;

    impl LlmClient for StubLlm {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unreachable!("classification tests never call the LLM")
        }
        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {}
        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unreachable!("classification tests never call the LLM")
        }
    }

    // A ReportCycle with a stub LLM, armed on `yesterday`, for verdict
    // classification tests that must not touch the network or the database.
    fn armed_cycle(awaiting_verdict: Option<String>) -> super::ReportCycle {
        super::ReportCycle {
            gate: ReportGate::new(7),
            memory_llm: Arc::new(StubLlm),
            chat_id: 42,
            awaiting_verdict,
        }
    }

    #[test]
    fn a_verdict_is_taken_only_while_armed_and_then_disarms() {
        let mut cycle = armed_cycle(Some("2026-07-10".to_string()));
        let taken = cycle.take_verdict("2");
        assert_eq!(taken, Some(("2026-07-10".to_string(), 1.0)));
        // The flag is cleared, so the same reply is no longer a verdict.
        assert_eq!(cycle.take_verdict("2"), None);
    }

    #[test]
    fn a_bare_number_is_not_a_verdict_when_unarmed() {
        let mut cycle = armed_cycle(None);
        assert_eq!(cycle.take_verdict("2"), None);
    }

    #[test]
    fn a_non_verdict_reply_leaves_the_flag_armed() {
        let mut cycle = armed_cycle(Some("2026-07-10".to_string()));
        assert_eq!(cycle.take_verdict("2 but great"), None);
        // Still armed: the user can chat first and score later that day.
        assert_eq!(
            cycle.take_verdict("1"),
            Some(("2026-07-10".to_string(), 0.5))
        );
    }

    #[test]
    fn a_verdict_scores_the_day_session_and_the_dream_when_present() {
        assert_eq!(
            verdict_session_ids("2026-07-10", None),
            vec!["tg-2026-07-10".to_string()]
        );
        assert_eq!(
            verdict_session_ids("2026-07-10", Some("dream-123".to_string())),
            vec!["tg-2026-07-10".to_string(), "dream-123".to_string()]
        );
    }

    #[test]
    fn gate_fires_once_at_the_report_hour() {
        let mut gate = ReportGate::new(7);
        assert!(!gate.should_fire(6, "2026-07-11"), "before the hour");
        assert!(gate.should_fire(7, "2026-07-11"), "at the hour");
        assert!(!gate.should_fire(7, "2026-07-11"), "not twice the same day");
    }

    #[test]
    fn gate_fires_again_on_the_next_day() {
        let mut gate = ReportGate::new(7);
        assert!(gate.should_fire(7, "2026-07-11"));
        assert!(
            gate.should_fire(7, "2026-07-12"),
            "a new day re-arms the gate"
        );
    }

    #[test]
    fn verdict_parsing_accepts_only_bare_integers_in_range() {
        assert_eq!(parse_verdict("2"), Some(2));
        assert_eq!(parse_verdict("-1"), Some(-1));
        assert_eq!(parse_verdict("0"), Some(0));
        assert_eq!(parse_verdict(" 2 "), Some(2));
        assert_eq!(parse_verdict("3"), None, "out of range");
        assert_eq!(parse_verdict("2 but great"), None);
        assert_eq!(parse_verdict("yesterday was good"), None);
        assert_eq!(parse_verdict(""), None);
    }

    #[test]
    fn verdict_maps_to_the_reinforcement_range() {
        for (raw, expected) in [("2", 1.0), ("-1", -0.5), ("0", 0.0)] {
            let Some(value) = parse_verdict(raw) else {
                unreachable!("valid verdict")
            };
            assert!((value as f64 / 2.0 - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn a_quiet_digest_is_recognised() {
        assert!(DayDigest::default().is_quiet());
        let busy = DayDigest {
            turns: 3,
            ..DayDigest::default()
        };
        assert!(!busy.is_quiet());
        let due_only = DayDigest {
            due_loops: vec!["ask about the file (due 2026-07-10)".to_string()],
            ..DayDigest::default()
        };
        assert!(!due_only.is_quiet());
    }

    #[test]
    fn render_omits_empty_ref_lines_but_keeps_counts() {
        let digest = DayDigest {
            day: "2026-07-10".to_string(),
            turns: 4,
            conversation_user_messages: 4,
            created: 2,
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains("Conversation: 4 messages, but no durable learning was recorded."));
        assert!(text.contains("Edges created 2"));
        assert!(!text.contains("Gists consolidated"));
        assert!(!text.contains("Due loops"));
    }

    fn pass_run(outcome: &str, reason_class: &str, items_added: usize) -> PassRun {
        PassRun {
            run_id: 1,
            pass_type: "dream".to_string(),
            started_at: String::new(),
            finished_at: String::new(),
            outcome: outcome.to_string(),
            reason_class: reason_class.to_string(),
            items_added,
        }
    }

    #[test]
    fn render_asserts_all_three_conversation_states() {
        let none = render_digest(&DayDigest::default());
        assert!(none.contains("Conversation: none today."));
        assert!(!none.contains("Thread 1:"));

        let messages_only = DayDigest {
            conversation_user_messages: 2,
            ..DayDigest::default()
        };
        let messages_text = render_digest(&messages_only);
        assert!(
            messages_text
                .contains("Conversation: 2 messages, but no durable learning was recorded.")
        );

        let learned = DayDigest {
            conversation_user_messages: 2,
            turn_groups: vec!["a → b".to_string()],
            ..DayDigest::default()
        };
        let learned_text = render_digest(&learned);
        assert!(learned_text.contains("Thread 1: a → b"));
        assert!(!learned_text.contains("no durable learning"));
        assert!(!learned_text.contains("none today"));
    }

    #[test]
    fn failed_dream_renders_outcome_and_omits_event_count() {
        let last_success = crate::timeutil::yesterday_iso();
        let digest = DayDigest {
            dream_events: 451,
            dream_runs: vec![pass_run("failed", "context_length", 0)],
            dream_consecutive_failures: 3,
            dream_last_successful_at: Some(format!("{last_success}T04:00:00Z")),
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains(&format!(
            "Dream: failed (context_length), 3rd consecutive night, last success {last_success}."
        )));
        assert!(!text.contains("Overnight dream events"));
    }

    #[test]
    fn successful_dream_renders_written_edges_and_event_count() {
        let digest = DayDigest {
            dream_events: 12,
            dream_runs: vec![pass_run("ok", "none", 3)],
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains("Dream: ok — 3 edges written."));
        assert!(text.contains("Overnight dream events: 12"));
    }

    #[test]
    fn render_includes_due_loops() {
        let digest = DayDigest {
            due_loops: vec![
                "ask about the file (due 2026-07-10)".to_string(),
                "check the service (due 2026-07-11)".to_string(),
            ],
            ..DayDigest::default()
        };
        let text = render_digest(&digest);
        assert!(text.contains(
            "Due loops: ask about the file (due 2026-07-10); check the service (due 2026-07-11)"
        ));
    }

    #[test]
    fn render_includes_autonomy_only_when_non_empty() {
        let empty = render_digest(&DayDigest::default());
        assert!(!empty.contains("Autonomy:"));
        let mut digest = DayDigest::default();
        digest.autonomy.actions = 2;
        digest.autonomy.yielded = 1;
        assert!(render_digest(&digest).contains("Autonomy: Ran 2 self-directed passes"));
    }
