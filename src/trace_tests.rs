    use std::collections::HashMap;
    use std::error::Error;
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};

    use super::{
        ResponseRecord, TraceContext, TraceKind, TraceRecord, TracingClient, assign_sequence,
        enabled, new_id, redact_value, render_tree, request, response, trace_dir, write_record_to,
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
    fn enabled_writer_persists_request_and_response_shape() {
        if !enabled() {
            return;
        }
        let context = TraceContext::new("test");
        let span = context.llm_span();
        request(
            &span,
            "provider",
            "model",
            json!({"messages": [], "tools": [{"name": "search"}]}),
        );
        response(
            &span,
            "provider",
            "model",
            ResponseRecord {
                content: json!({"content": "answer", "reasoning": "", "tool_calls": []}),
                finish_reason: Some("stop".to_string()),
                usage: json!({"prompt_tokens": 3, "completion_tokens": 1}),
                duration_ms: 4,
            },
        );
        let path = trace_dir().join(format!("{}.jsonl", context.trace_id));
        let raw = std::fs::read_to_string(&path).unwrap_or_default();
        let mut lines = raw.lines();
        let request_line = lines.next().unwrap_or_default();
        let response_line = lines.next().unwrap_or_default();
        assert!(request_line.contains("\"kind\":\"request\""));
        assert!(request_line.contains("\"tools\""));
        assert!(response_line.contains("\"kind\":\"response\""));
        assert!(response_line.contains("\"finish_reason\":\"stop\""));
        assert!(response_line.contains("\"prompt_tokens\":3"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_failure_is_isolated_from_a_model_call() {
        let context = test_context("test");
        let record = record(&context, TraceKind::Request);
        let blocker = std::env::temp_dir().join(new_id("trace-blocker"));
        let created = std::fs::File::create(&blocker);
        assert!(created.is_ok());
        assert!(write_record_to(&record, &blocker).is_err());
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
    fn viewer_orders_and_indents_a_small_trace_tree() {
        let context = test_context("chat");
        let mut response = record(&context, TraceKind::Response);
        response.sequence = 2;
        let mut request = record(&context, TraceKind::Request);
        request.sequence = 1;
        response.parent = request.span_id.clone();
        let rendered = render_tree(&[request, response]);
        let lines: Vec<&str> = rendered.lines().collect();
        assert!(lines[0].starts_with("chat trace-"));
        assert!(lines[1].starts_with("  0001 request"));
        assert!(lines[2].starts_with("    0002 response"));
    }
