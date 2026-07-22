    use std::time::Instant;

    use serde_json::json;

    use super::{
        StreamState, apply_usage, cached_usage_line, request_area, request_body,
        send_content_delta, send_reasoning_delta,
    };

    fn stream_state() -> StreamState {
        StreamState {
            start: Instant::now(),
            ttft: None,
            tokens_in: 0,
            tokens_out: 0,
            cost_usd: None,
            cached_tokens: None,
            saw_prompt_tokens: false,
            tool_calls: Vec::new(),
            finish_reason: None,
            content: String::new(),
            reasoning: String::new(),
        }
    }

    #[test]
    fn traced_request_body_is_the_full_wire_payload_with_tools() {
        let messages = vec![json!({"role": "user", "content": "hello"})];
        let tools = json!([{"type": "function", "function": {"name": "search"}}]);
        let body = request_body("model-name", &messages, Some(&tools));
        assert_eq!(body["model"], "model-name");
        assert_eq!(body["messages"], serde_json::Value::Array(messages));
        assert_eq!(body["stream"], true);
        assert_eq!(body["tools"], tools);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn response_accumulator_keeps_forwarded_content_and_reasoning() {
        let (sender, _receiver) = std::sync::mpsc::channel();
        let mut state = stream_state();
        send_reasoning_delta(Some("because ".to_string()), &mut state, &sender);
        send_reasoning_delta(Some("facts".to_string()), &mut state, &sender);
        send_content_delta(Some("final ".to_string()), &mut state, &sender);
        send_content_delta(Some("answer".to_string()), &mut state, &sender);
        assert_eq!(state.reasoning, "because facts");
        assert_eq!(state.content, "final answer");
    }

    #[test]
    fn cached_prompt_tokens_are_parsed_when_reported() {
        let usage_result = serde_json::from_value(json!({
            "prompt_tokens": 1200,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 900}
        }));
        let Ok(usage) = usage_result else {
            unreachable!("usage fixture parses");
        };
        let mut state = stream_state();
        apply_usage(usage, &mut state);
        assert_eq!(
            cached_usage_line(&state).as_deref(),
            Some("cached 900/1200 prompt tokens")
        );
    }

    #[test]
    fn absent_cached_token_details_produce_no_log_line() {
        let usage_result = serde_json::from_value(json!({"prompt_tokens": 1200}));
        let Ok(usage) = usage_result else {
            unreachable!("usage fixture without details parses");
        };
        let mut state = stream_state();
        apply_usage(usage, &mut state);
        assert!(cached_usage_line(&state).is_none());
    }

    #[test]
    fn request_area_distinguishes_chat_memory_and_dream_calls() {
        let chat = vec![
            json!({"role": "system", "content": "chat rules"}),
            json!({"role": "user", "content": "hello"}),
        ];
        let memory = vec![json!({"role": "user", "content": "## Your role\nextract"})];
        let dream = vec![json!({
            "role": "user",
            "content": "## Current memory graph\nnodes: []"
        })];
        assert_eq!(request_area(&chat), "chat");
        assert_eq!(request_area(&memory), "mem");
        assert_eq!(request_area(&dream), "dream");
    }
