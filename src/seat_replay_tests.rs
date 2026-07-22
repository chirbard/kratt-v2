    use super::{
        ReplayEntry, extract_turn_pairs, format_classifier_report, format_scribe_report,
        run_classifier_replay, run_scribe_replay,
    };
    use crate::api::{Chunk, LlmClient, TurnOutcome};
    use std::error::Error;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::mpsc::Sender;

    fn entry(role: &str, content: &str) -> ReplayEntry {
        ReplayEntry {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn pairs_are_ordered_and_last_n_respected() {
        let entries = vec![
            entry("user", "u1"),
            entry("assistant", "a1"),
            entry("user", "u2"),
            entry("assistant", "a2"),
            entry("user", "u3"),
            entry("assistant", "a3"),
        ];
        let pairs = extract_turn_pairs(&entries, 2);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("u2".to_string(), "a2".to_string()));
        assert_eq!(pairs[1], ("u3".to_string(), "a3".to_string()));
    }

    #[test]
    fn unpaired_trailing_user_entry_is_skipped() {
        let entries = vec![
            entry("user", "u1"),
            entry("assistant", "a1"),
            entry("user", "dangling"),
        ];
        let pairs = extract_turn_pairs(&entries, 30);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("u1".to_string(), "a1".to_string()));
    }

    // Hands out one canned reply per collect() call, in order.
    struct ScriptedClient {
        replies: Mutex<Vec<String>>,
    }

    impl ScriptedClient {
        fn scripted(replies: &[&str]) -> Arc<dyn LlmClient> {
            let mut queued = Vec::new();
            for reply in replies.iter().rev() {
                queued.push((*reply).to_string());
            }
            Arc::new(ScriptedClient {
                replies: Mutex::new(queued),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            let Ok(mut queued) = self.replies.lock() else {
                return Ok(String::new());
            };
            Ok(queued.pop().unwrap_or_default())
        }

        fn stream(&self, _prompt: String, _system: Option<String>, _sender: Sender<Chunk>) {
            unimplemented!("scripted client collects only")
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client collects only")
        }
    }

    fn two_pairs() -> Vec<(String, String)> {
        vec![
            ("u1".to_string(), "a1".to_string()),
            ("u2".to_string(), "a2".to_string()),
        ]
    }

    #[test]
    fn classifier_agreement_line_counts_matches() {
        let baseline = ScriptedClient::scripted(&["TASK", "CHAT"]);
        let candidate = ScriptedClient::scripted(&["TASK", "TASK"]);
        let result = run_classifier_replay(&baseline, &candidate, &two_pairs());
        let report = format_classifier_report(&result);
        assert!(report.contains("classifier agreement: 1/2 (50%)"), "{report}");
        assert!(report.contains("disagree [base=chat cand=task]"), "{report}");
    }

    const ONE_CALL: &str =
        r#"{"tools":[{"tool":"upsert_edge","args":{"source":"game","target":"move-e4","content":"played e4","stability":0.3,"confidence":1.0,"importance":1.0,"subject":false,"date":"2026-07-11","memory_type":"episodic"}}]}"#;

    #[test]
    fn scribe_parse_rate_line_counts_valid_responses() {
        let baseline = ScriptedClient::scripted(&[ONE_CALL, ONE_CALL]);
        let candidate = ScriptedClient::scripted(&[ONE_CALL, "not json at all"]);
        let result = run_scribe_replay(&baseline, &candidate, &two_pairs());
        let report = format_scribe_report(&result);
        assert!(report.contains("scribe baseline: parse 2/2 (100%)"), "{report}");
        assert!(
            report.contains("scribe candidate: parse 1/2 (50%)"),
            "{report}"
        );
        assert!(report.contains("malformed 1"), "{report}");
    }
