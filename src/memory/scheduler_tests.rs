    use super::*;

    #[test]
    fn test_decide_ingest_when_notes_above_threshold() {
        // Create a temporary file that exists for the test
        let temp_path = std::env::temp_dir().join("test_file.md");
        let _ = std::fs::File::create(&temp_path);

        let inputs = SchedulerInputs {
            queued_inbox: 0,
            re_read_candidates: vec![(
                temp_path.to_string_lossy().into_owned(),
                3,
                "node1".to_string(),
            )],
            actor_positions: vec![],
        };
        let action = decide(&inputs);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);

        match action {
            SchedulerAction::Ingest { path, reason } => {
                assert_eq!(path, temp_path);
                assert!(reason.contains("3"));
            }
            _ => panic!("Expected Ingest action, got {:?}", action),
        }
    }

    #[test]
    fn inbox_study_precedes_re_read() {
        let inputs = SchedulerInputs {
            queued_inbox: 2,
            re_read_candidates: vec![("missing".to_string(), 9, "node".to_string())],
            ..SchedulerInputs::default()
        };
        assert_eq!(
            decide(&inputs),
            SchedulerAction::InboxStudy {
                reason: "2 inbox items queued".to_string()
            }
        );
    }

    #[test]
    fn pressure_line_renders_only_real_backlog() {
        let none: Vec<String> = Vec::new();
        assert_eq!(super::render_pressure(&none), None);
        let queued = vec!["okf.md".to_string(), "paper.html".to_string()];
        let Some(queued_only) = super::render_pressure(&queued) else {
            unreachable!("queued renders")
        };
        assert!(queued_only.starts_with("Internal state: queued study items"));
    }
