    #[test]
    fn classify_prompt_carries_active_task_hint_only_when_set() {
        let biased = super::build_classify_prompt("Nc3", "Nf6", true);
        assert!(biased.contains("ongoing task with moves already recorded"));
        let unbiased = super::build_classify_prompt("Nc3", "Nf6", false);
        assert!(!unbiased.contains("ongoing task"));
    }

    use super::{
        AgentPrompt, CHAT_SYSTEM_PROMPT, CONSOLIDATE_EXPLANATION, CONTINUE_PASS, DAILY_REPORT,
        DREAM_BIOLOGIST, EXTRACT_EXPLANATION, JSON_OUTPUT_RULES, MEMORY_RULES,
        SCRIBE_EXPLANATION, SCRIBE_TOOL_DEFINITIONS, TOOL_DEFINITIONS, build_agent_prompt,
        build_chat_system_prompt, build_classify_prompt, build_continue_prompt, build_dream_prompt,
        build_scribe_continue_prompt, build_scribe_prompt,
    };

    const INJECTED_HEADING: &str = "## Beliefs injected into this turn";

    fn prompt_with(injected_toon: Option<&str>) -> String {
        build_agent_prompt(&AgentPrompt {
            explanation: EXTRACT_EXPLANATION,
            toon: "nodes: []",
            user_msg: "user",
            assistant_msg: "assistant",
            diff_text: None,
            injected_toon,
            active_game: None,
        })
    }

    fn position(prompt: &str, needle: &str) -> usize {
        prompt.find(needle).unwrap_or(usize::MAX)
    }

    fn common_prefix_len(left: &str, right: &str) -> usize {
        let left_bytes = left.as_bytes();
        let right_bytes = right.as_bytes();
        let limit = left_bytes.len().min(right_bytes.len());
        for index in 0..limit {
            if left_bytes[index] != right_bytes[index] {
                return index;
            }
        }
        limit
    }

    #[test]
    fn heavy_prompt_orders_shared_graph_before_role_and_exchange() {
        let prompt = prompt_with(Some("injected edge"));
        assert!(
            position(&prompt, JSON_OUTPUT_RULES) < position(&prompt, "## Current memory state")
        );
        assert!(
            position(&prompt, "## Current memory state")
                < position(&prompt, "\n\n## Your role\n\n")
        );
        assert!(
            position(&prompt, "\n\n## Your role\n\n") < position(&prompt, "## Exchange to process")
        );
    }

    #[test]
    fn extract_and_consolidate_share_prefix_through_graph_toon() {
        let toon = "nodes: [shared graph]\nedges: [a -> b]";
        let extract = build_agent_prompt(&AgentPrompt {
            explanation: EXTRACT_EXPLANATION,
            toon,
            user_msg: "user",
            assistant_msg: "assistant",
            diff_text: None,
            injected_toon: None,
            active_game: None,
        });
        let consolidate = build_agent_prompt(&AgentPrompt {
            explanation: CONSOLIDATE_EXPLANATION,
            toon,
            user_msg: "user",
            assistant_msg: "assistant",
            diff_text: Some("+ a -> b"),
            injected_toon: None,
            active_game: None,
        });
        let toon_end = position(&extract, toon) + toon.len();
        assert!(common_prefix_len(&extract, &consolidate) >= toon_end);
    }

    #[test]
    fn heavy_roles_warn_that_memory_state_is_bounded() {
        let warning = "relevance-selected subset, not the whole graph";
        assert!(EXTRACT_EXPLANATION.contains(warning));
        assert!(CONSOLIDATE_EXPLANATION.contains(warning));
    }

    #[test]
    fn dream_prompt_orders_shared_graph_before_role_and_context() {
        let prompt = build_dream_prompt(DREAM_BIOLOGIST, "shared toon", Some("prior proposals"));
        assert!(
            position(&prompt, JSON_OUTPUT_RULES) < position(&prompt, "## Current memory graph")
        );
        assert!(
            position(&prompt, "## Current memory graph")
                < position(&prompt, "\n\n## Your role\n\n")
        );
        assert!(
            position(&prompt, "\n\n## Your role\n\n")
                < position(&prompt, "## Proposals from previous agents")
        );
    }

    #[test]
    fn scribe_and_classifier_put_role_after_the_shared_framing() {
        let scribe = build_scribe_prompt("user", "assistant");
        assert!(position(&scribe, JSON_OUTPUT_RULES) < position(&scribe, "\n\n## Your role\n\n"));
        assert!(
            position(&scribe, "\n\n## Your role\n\n") < position(&scribe, "## Exchange to record")
        );
        let classify = build_classify_prompt("user", "assistant", false);
        assert!(classify.starts_with(super::AGENT_SYSTEM_FRAMING));
        assert!(position(&classify, "\n\n## Your role\n\n") < position(&classify, "## Exchange"));
    }

    #[test]
    fn scribe_continue_extends_first_prompt_after_the_exchange() {
        let initial = build_scribe_prompt("user", "assistant");
        let applied = vec!["record_move: e4".to_string()];
        let continuation =
            build_scribe_continue_prompt(SCRIBE_EXPLANATION, "user", "assistant", &applied);
        assert!(continuation.starts_with(&initial));
        assert!(
            position(&continuation, "## Exchange to record")
                < position(&continuation, CONTINUE_PASS)
        );
        assert!(
            position(&continuation, CONTINUE_PASS) < position(&continuation, "## Already recorded")
        );
    }

    #[test]
    fn heavy_continue_keeps_applied_calls_at_the_tail() {
        let applied = vec!["upsert_edge: a -> b".to_string()];
        let prompt = build_continue_prompt(EXTRACT_EXPLANATION, "user", "assistant", &applied);
        assert!(
            position(&prompt, "## Exchange being processed") < position(&prompt, CONTINUE_PASS)
        );
        assert!(position(&prompt, CONTINUE_PASS) < position(&prompt, "## Already applied"));
        assert!(prompt.trim_end().ends_with("upsert_edge: a -> b"));
    }

    #[test]
    fn chat_style_hint_is_static_head_content() {
        let hint = "Keep replies narrow enough for a phone.";
        let with = build_chat_system_prompt("remembered fact", &[], None, None, Some(hint));
        assert!(position(&with, hint) < position(&with, "Today's date is"));
        assert!(position(&with, hint) < position(&with, "\n\n<memory>\n"));
        let without = build_chat_system_prompt("remembered fact", &[], None, None, None);
        assert!(!without.contains(hint));
    }

    #[test]
    fn injected_section_is_present_when_edges_were_injected() {
        let prompt = prompt_with(Some("edges: alice -> berlin"));
        assert!(prompt.contains(INJECTED_HEADING));
        assert!(prompt.contains("alice -> berlin"));
    }

    #[test]
    fn injected_section_is_omitted_when_nothing_was_injected() {
        assert!(!prompt_with(None).contains(INJECTED_HEADING));
        // An all-whitespace render is an empty prediction, not a heading.
        assert!(!prompt_with(Some("  \n ")).contains(INJECTED_HEADING));
    }

    #[test]
    fn continuations_never_carry_the_injected_section() {
        let applied = vec!["upsert_edge: alice → berlin".to_string()];
        let continuation =
            build_continue_prompt(EXTRACT_EXPLANATION, "user", "assistant", &applied);
        assert!(!continuation.contains(INJECTED_HEADING));
        assert!(!continuation.contains("## Current memory state"));
        assert!(continuation.contains("upsert_edge: alice → berlin"));
    }

    #[test]
    fn chat_prompt_carries_the_pressure_line_only_when_present() {
        let with = build_chat_system_prompt(
            "",
            &[],
            Some(
                "Internal state: 30 unconsolidated experience events since the last dream (a dream is forced at 50).",
            ),
            None,
            None,
        );
        assert!(with.contains("Internal state: 30 unconsolidated"));
        let without = build_chat_system_prompt("", &[], None, None, None);
        assert!(!without.contains("Internal state:"));
    }

    #[test]
    fn chat_prompt_carries_the_sequence_line_only_when_present() {
        let line = "Sequence memory: after rust, this user has previously moved on to sqlite (unprompted continuations: 2.0).";
        let with = build_chat_system_prompt("", &[], None, Some(line), None);
        assert!(with.contains(line));
        let without = build_chat_system_prompt("", &[], None, None, None);
        assert!(!without.contains("Sequence memory:"));
    }

    #[test]
    fn chat_defaults_to_blunt_length_matched_answers() {
        assert!(CHAT_SYSTEM_PROMPT.contains("Answer like an Estonian would"));
        assert!(CHAT_SYSTEM_PROMPT.contains("if one word is enough, use one word"));
        assert!(CHAT_SYSTEM_PROMPT.contains("only when the user explicitly asks"));
        assert!(CHAT_SYSTEM_PROMPT.contains("Reply in English unless"));
    }

    #[test]
    fn chat_prompt_requires_recall_before_user_specific_claims() {
        assert!(CHAT_SYSTEM_PROMPT.contains("call retrieve_memory"));
        assert!(CHAT_SYSTEM_PROMPT.contains("specific fact about the user"));
        assert!(CHAT_SYSTEM_PROMPT.contains("Do not retrieve general knowledge"));
    }

    #[test]
    fn heavy_and_scribe_prompts_both_document_loops() {
        let heavy = prompt_with(None);
        assert!(heavy.contains("## Prospective memory — open loops"));
        assert!(heavy.contains("open_loop(intention!"));
        assert!(heavy.contains("close_loop(loop!"));
        let scribe = build_scribe_prompt("user", "assistant");
        assert!(scribe.contains("open_loop(intention!"));
        assert!(scribe.contains("close_loop(loop!"));
    }

    #[test]
    fn node_tools_are_fact_only() {
        for definitions in [SCRIBE_TOOL_DEFINITIONS, TOOL_DEFINITIONS] {
            assert!(!definitions.contains(r#""type":"entity"#));
            assert!(!definitions.contains("entity | fact"));
            assert!(definitions.contains("type!=\"fact\""));
            assert!(definitions.contains("auto-create entities"));
        }
    }

    #[test]
    fn temporal_guidance_separates_learned_and_occurred_dates() {
        assert!(TOOL_DEFINITIONS.contains("occurred_at?"));
        assert!(!TOOL_DEFINITIONS.contains("subject!, date!"));
        assert!(SCRIBE_TOOL_DEFINITIONS.contains("subject!=false, date!"));
        assert!(!SCRIBE_TOOL_DEFINITIONS.contains("occurred_at"));
        assert!(MEMORY_RULES.contains("Node date is automatic"));
        assert!(MEMORY_RULES.contains("Edge date is automatic"));
        assert!(MEMORY_RULES.contains("never use today's date as occurred_at"));
    }

    #[test]
    fn loop_anchors_are_documented_as_automatic() {
        let heavy = prompt_with(None);
        let scribe = build_scribe_prompt("user", "assistant");
        for prompt in [heavy, scribe] {
            assert!(!prompt.contains("emit its upsert_node before open_loop"));
            assert!(!prompt.contains("emit its upsert_node first if needed"));
            assert!(prompt.contains("anchor is created automatically if missing"));
        }
    }

    #[test]
    fn heavy_and_scribe_prompts_share_the_conservative_reaction_tool() {
        let heavy = prompt_with(None);
        let scribe = build_scribe_prompt("user", "assistant");
        for prompt in [heavy, scribe] {
            assert!(prompt.contains("record_reaction(reaction!"));
            assert!(prompt.contains("plain thanks"));
            assert!(prompt.contains("be conservative"));
        }
    }

    #[test]
    fn heavy_and_scribe_prompts_share_the_standing_interest_tool() {
        let heavy = prompt_with(None);
        let scribe = build_scribe_prompt("user", "assistant");
        for prompt in [heavy, scribe] {
            assert!(prompt.contains("register_interest(topic!)"));
            assert!(prompt.contains("not a one-off question"));
        }
        assert!(CHAT_SYSTEM_PROMPT.contains("building your understanding"));
    }

    #[test]
    fn heavy_and_scribe_prompts_translate_notification_policy_to_iso() {
        let heavy = prompt_with(None);
        let scribe = build_scribe_prompt("be quiet this week", "okay");
        for prompt in [heavy, scribe] {
            assert!(prompt.contains("set_ping_policy(action!"));
            assert!(prompt.contains("resolve fuzzy durations"));
            assert!(prompt.contains("ISO-8601"));
        }
        assert!(CHAT_SYSTEM_PROMPT.contains("your ping policy"));
    }

    #[test]
    fn daily_report_prompt_mentions_due_loops_and_confirmed_absence() {
        assert!(DAILY_REPORT.contains("Mention Due loops in one sentence"));
        assert!(DAILY_REPORT.contains("did it happen?"));
        assert!(DAILY_REPORT.contains("Never treat the deadline alone as evidence of absence"));
        assert!(DAILY_REPORT.contains("Report the Autonomy line honestly"));
        assert!(DAILY_REPORT.contains("Held for you"));
        assert!(DAILY_REPORT.contains("Report a failed or skipped pass as failed or skipped"));
        assert!(DAILY_REPORT.contains("Never write \"we\", \"discussed\""));
        assert!(DAILY_REPORT.contains("drawn only from the Thread lines"));
    }
