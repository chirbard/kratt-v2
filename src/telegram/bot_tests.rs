    use super::{TELEGRAM_MAX_CHARS, parse_updates, split_message};
    use serde_json::json;

    #[test]
    fn parses_user_chat_and_text_from_an_update() {
        let value = json!({"result": [{
            "update_id": 10,
            "message": {"from": {"id": 42}, "chat": {"id": 7}, "text": "hello"}
        }]});
        let updates = parse_updates(&value);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 10);
        let Some(message) = updates[0].message.as_ref() else {
            unreachable!("message parsed")
        };
        assert_eq!(message.user_id, 42);
        assert_eq!(message.chat_id, 7);
        assert_eq!(message.text, "hello");
    }

    #[test]
    fn a_malformed_update_is_skipped_but_still_advances_the_offset() {
        // No text field — nothing to answer, but update_id is intact.
        let value = json!({"result": [{
            "update_id": 5,
            "message": {"from": {"id": 42}, "chat": {"id": 7}}
        }]});
        let updates = parse_updates(&value);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 5);
        assert!(updates[0].message.is_none());
    }

    #[test]
    fn an_entry_without_an_update_id_is_dropped_entirely() {
        let value = json!({"result": [{"message": {"text": "x"}}]});
        assert!(parse_updates(&value).is_empty());
    }

    #[test]
    fn a_short_answer_stays_one_message() {
        let chunks = split_message("just a short reply", TELEGRAM_MAX_CHARS);
        assert_eq!(chunks, vec!["just a short reply".to_string()]);
    }

    #[test]
    fn a_long_answer_splits_on_paragraph_boundaries_in_order() {
        let para = "x".repeat(3000);
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = split_message(&text, TELEGRAM_MAX_CHARS);
        assert!(chunks.len() >= 2, "must split");
        for chunk in &chunks {
            assert!(chunk.chars().count() <= TELEGRAM_MAX_CHARS);
        }
        // Order preserved: concatenating back yields the paragraphs in sequence.
        let rejoined = chunks.join("\n\n");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn a_single_oversized_paragraph_is_hard_split() {
        let text = "y".repeat(TELEGRAM_MAX_CHARS * 2 + 50);
        let chunks = split_message(&text, TELEGRAM_MAX_CHARS);
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= TELEGRAM_MAX_CHARS);
        }
        assert_eq!(chunks.concat(), text);
    }
