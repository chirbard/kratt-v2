    use super::{INGEST_MAX_FULL_CHUNKS, chunk_document, curiosity_budget, parse_full_chunk_mask};

    #[test]
    fn chunking_packs_paragraphs_without_splitting() {
        let document = "a\n\na\n\na\n\na";
        let chunks = chunk_document(document, 4);
        assert_eq!(chunks, vec!["a\n\na".to_string(), "a\n\na".to_string()]);
    }

    #[test]
    fn parsed_full_chunk_mask_uses_one_based_indices() {
        let mask = parse_full_chunk_mask(r#"{"full_chunks":[2,4]}"#, 4);
        assert_eq!(mask, Some(vec![false, true, false, true]));
    }

    #[test]
    fn curiosity_budget_is_clamped() {
        assert_eq!(curiosity_budget(1), 1);
        assert_eq!(curiosity_budget(16), 4);
        assert_eq!(curiosity_budget(200), 6);
    }

    #[test]
    fn full_chunk_limit_constant_matches_expectation() {
        assert_eq!(INGEST_MAX_FULL_CHUNKS, 25);
    }
