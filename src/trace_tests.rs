    use std::collections::HashMap;
    use std::error::Error;
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};

    use super::{
        TraceContext, TraceKind, TraceRecord, TracingClient, WriterState, assign_sequence,
        flow_text_files, new_id, prune_traces, record_paths, redact_value, render_human_record,
        write_record_to,
    };

    struct PassClient;

    impl LlmClient for PassClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            Ok("ok".to_string())
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            let _ = sender.send(Chunk::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }));
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            Ok(TurnOutcome::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }))
        }
    }

    fn record(context: &TraceContext, kind: TraceKind) -> TraceRecord {
        TraceRecord::new(&context.llm_span(), kind)
    }

    fn test_context(activity: &str) -> TraceContext {
        let trace_id = new_id("trace");
        TraceContext {
            parent_span: trace_id.clone(),
            trace_id,
            activity: activity.to_string(),
            seat: None,
            last_span: Arc::new(Mutex::new(String::new())),
        }
    }

    #[test]
    fn writer_sequence_is_global_within_each_trace() {
        let context = test_context("dream");
        let mut first = record(&context, TraceKind::Request);
        let mut second = record(&context, TraceKind::Request);
        let mut third = record(&context, TraceKind::Response);
        let mut sequences = HashMap::new();
        assign_sequence(&mut first, &mut sequences);
        assign_sequence(&mut second, &mut sequences);
        assign_sequence(&mut third, &mut sequences);
        assert_eq!([first.sequence, second.sequence, third.sequence], [1, 2, 3]);
    }

    #[test]
    fn tool_spans_attach_to_the_preceding_model_call() {
        let context = test_context("chat");
        let model_span = context.llm_span();
        let tool_span = context.child_span();
        assert_eq!(tool_span.context.parent_span, model_span.span_id);
    }

    #[test]
    fn tracing_off_returns_the_original_client_and_keeps_calls_unchanged() {
        if std::env::var("KRATT_TRACE").is_ok_and(|value| value == "1") {
            return;
        }
        let inner: Arc<dyn LlmClient> = Arc::new(PassClient);
        let wrapped = TracingClient::wrap(Arc::clone(&inner), TraceContext::new("test"));
        assert!(Arc::ptr_eq(&inner, &wrapped));
        assert_eq!(wrapped.collect("prompt").unwrap_or_default(), "ok");
    }

    #[test]
    fn writer_creates_readable_flow_and_agent_files() {
        let context = test_context("chat").with_seat("answering");
        let span = context.llm_span();
        let mut request_record = TraceRecord::new(&span, TraceKind::Request);
        request_record.timestamp = "2026-07-22T14:28:03".to_string();
        request_record.provider = Some("openrouter".to_string());
        request_record.model = Some("openai/gpt-test".to_string());
        request_record.content = Some(json!({
            "messages": [
                {"role": "system", "content": "line one\nline two"},
                {"role": "user", "content": "hello"}
            ],
            "tools": [{"type": "function", "function": {"name": "search_history"}}]
        }));
        let root = std::env::temp_dir().join(new_id("trace-layout"));
        let mut state = WriterState::default();
        assert!(write_record_to(&request_record, &root, &mut state).is_ok());
        let day = root.join("2026-07-22");
        let flow = only_directory(&day).unwrap_or_default();
        assert!(flow.file_name().unwrap_or_default().to_string_lossy().starts_with("142803-chat-"));
        let human_path = flow.join("01-answering-142803.txt");
        let raw_path = flow.join("01-answering-142803.jsonl");
        let human = std::fs::read_to_string(human_path).unwrap_or_default();
        let raw = std::fs::read_to_string(raw_path).unwrap_or_default();
        assert!(human.contains("[system]\nline one\nline two"));
        assert!(human.contains("[user]\nhello"));
        assert!(human.contains("tools: search_history   (1 total)"));
        assert!(!human.contains("line one\\nline two"));
        assert!(raw.contains("line one\\nline two"));
        let _ = std::fs::remove_dir_all(root);
    }

    fn only_directory(parent: &std::path::Path) -> Option<std::path::PathBuf> {
        let mut directories = Vec::new();
        let entries = std::fs::read_dir(parent).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                directories.push(path);
            }
        }
        assert_eq!(directories.len(), 1);
        directories.into_iter().next()
    }

    #[test]
    fn tool_records_share_the_parent_agent_file_and_rounds_advance() {
        let context = test_context("chat");
        let first_span = context.llm_span();
        let mut request = TraceRecord::new(&first_span, TraceKind::Request);
        request.timestamp = "2026-07-22T14:28:03".to_string();
        let tool_span = context.child_span();
        let mut tool = TraceRecord::new(&tool_span, TraceKind::ToolCall);
        tool.timestamp = "2026-07-22T14:28:05".to_string();
        let second_span = context.llm_span();
        let mut second = TraceRecord::new(&second_span, TraceKind::Request);
        second.timestamp = "2026-07-22T14:28:09".to_string();
        let root = std::path::Path::new("traces");
        let mut state = WriterState::default();
        let request_paths = record_paths(&request, root, &mut state);
        let tool_paths = record_paths(&tool, root, &mut state);
        let second_paths = record_paths(&second, root, &mut state);
        assert_eq!(request_paths.human, tool_paths.human);
        assert!(request_paths.human.ends_with("01-answering-142803.txt"));
        assert!(second_paths.human.ends_with("02-answering-142809.txt"));
    }

    #[test]
    fn response_rendering_preserves_content_reasoning_and_tool_calls() {
        let context = test_context("chat");
        let mut response = record(&context, TraceKind::Response);
        response.timestamp = "2026-07-22T14:28:05".to_string();
        response.finish_reason = Some("stop".to_string());
        response.duration_ms = Some(1_200);
        response.usage = Some(json!({"prompt_tokens": 400, "completion_tokens": 112}));
        response.content = Some(json!({
            "content": "answer line one\nanswer line two",
            "reasoning": "because this follows",
            "tool_calls": [{"name": "retrieve_memory", "args": "{\"query\":\"camera\"}"}]
        }));
        let rendered = render_human_record(&response);
        assert!(rendered.contains("finish=stop, 512 tok, 1.2s"));
        assert!(rendered.contains("answer line one\nanswer line two"));
        assert!(rendered.contains("[reasoning]\nbecause this follows"));
        assert!(rendered.contains("retrieve_memory({\"query\":\"camera\"})"));
    }

    #[test]
    fn retention_prunes_whole_oldest_day_folder() {
        let root = std::env::temp_dir().join(new_id("trace-retention"));
        for offset in 0..=100 {
            let day = root.join(format!("{:04}-01-01", 2_000 + offset));
            assert!(std::fs::create_dir_all(day.join("flow")).is_ok());
        }
        assert!(prune_traces(&root).is_ok());
        assert!(!root.join("2000-01-01").exists());
        assert!(root.join("2100-01-01").is_dir());
        let day_count = match std::fs::read_dir(&root) {
            Ok(entries) => entries.count(),
            Err(_) => 0,
        };
        assert_eq!(day_count, 100);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn writer_failure_is_isolated_from_a_model_call() {
        let context = test_context("test");
        let record = record(&context, TraceKind::Request);
        let blocker = std::env::temp_dir().join(new_id("trace-blocker"));
        let created = std::fs::File::create(&blocker);
        assert!(created.is_ok());
        let mut state = WriterState::default();
        assert!(write_record_to(&record, &blocker, &mut state).is_err());
        let client = PassClient;
        assert_eq!(client.collect("still runs").unwrap_or_default(), "ok");
        let _ = std::fs::remove_file(blocker);
    }

    #[test]
    fn redaction_scrubs_only_exact_secrets() {
        let mut value = json!({
            "prompt": "secret-value https://example.com sk-ordinary-looking",
            "nested": ["secret-value-suffix"]
        });
        redact_value(&mut value, &["secret-value".to_string()]);
        assert_eq!(
            value["prompt"],
            "[REDACTED] https://example.com sk-ordinary-looking"
        );
        assert_eq!(value["nested"][0], "[REDACTED]-suffix");
    }

    #[test]
    fn viewer_reads_agent_files_in_span_order() {
        let flow = std::env::temp_dir().join(new_id("trace-view"));
        assert!(std::fs::create_dir_all(&flow).is_ok());
        assert!(std::fs::write(flow.join("02-scribe-142810.txt"), "second\n").is_ok());
        assert!(std::fs::write(flow.join("01-classifier-142803.txt"), "first\n").is_ok());
        let files = flow_text_files(&flow).unwrap_or_default();
        assert_eq!(files.len(), 2);
        assert!(files[0].ends_with("01-classifier-142803.txt"));
        assert!(files[1].ends_with("02-scribe-142810.txt"));
        let _ = std::fs::remove_dir_all(flow);
    }
