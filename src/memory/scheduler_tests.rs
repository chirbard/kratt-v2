    use super::*;

    #[test]
    fn test_decide_dream_when_debt_high() {
        let inputs = SchedulerInputs {
            acquisition_debt: 60,
            queued_inbox: 0,
            re_read_candidates: vec![],
            actor_positions: vec![],
            news_scan_due: false,
        };
        let action = decide(&inputs);
        match action {
            SchedulerAction::Dream { reason } => {
                assert!(reason.contains("60"));
                assert!(reason.contains("50"));
            }
            _ => panic!("Expected Dream action, got {:?}", action),
        }
    }

    #[test]
    fn test_decide_ingest_when_notes_above_threshold() {
        // Create a temporary file that exists for the test
        let temp_path = std::env::temp_dir().join("test_file.md");
        let _ = std::fs::File::create(&temp_path);

        let inputs = SchedulerInputs {
            acquisition_debt: 10,
            queued_inbox: 0,
            re_read_candidates: vec![(
                temp_path.to_string_lossy().into_owned(),
                3,
                "node1".to_string(),
            )],
            actor_positions: vec![],
            news_scan_due: false,
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
    fn news_scan_runs_below_maintenance_and_above_rest() {
        let due = SchedulerInputs {
            news_scan_due: true,
            ..SchedulerInputs::default()
        };
        assert!(matches!(decide(&due), SchedulerAction::NewsScan { .. }));

        let debt = SchedulerInputs {
            acquisition_debt: ACQUISITION_DEBT_THRESHOLD,
            news_scan_due: true,
            ..SchedulerInputs::default()
        };
        assert!(matches!(decide(&debt), SchedulerAction::Dream { .. }));

        let scanned = SchedulerInputs {
            news_scan_due: false,
            ..SchedulerInputs::default()
        };
        assert!(matches!(decide(&scanned), SchedulerAction::Rest { .. }));
    }

    #[test]
    fn test_rule_priority_dream_over_ingest() {
        // Dream should win over ingest when debt is high
        let inputs = SchedulerInputs {
            acquisition_debt: 60, // >= 50
            queued_inbox: 0,
            re_read_candidates: vec![("path/to/file.md".to_string(), 5, "node1".to_string())],
            actor_positions: vec![],
            news_scan_due: false,
        };
        let action = decide(&inputs);
        match action {
            SchedulerAction::Dream { .. } => {}
            _ => panic!("Expected Dream action to win over Ingest, got {:?}", action),
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
        assert_eq!(super::render_pressure(0, &none), None);
        assert_eq!(super::render_pressure(24, &none), None);
        let Some(debt_line) = super::render_pressure(30, &none) else {
            unreachable!("debt renders")
        };
        assert!(debt_line.contains("30 unconsolidated"));
        let queued = vec!["okf.md".to_string(), "paper.html".to_string()];
        let Some(both) = super::render_pressure(30, &queued) else {
            unreachable!("both render")
        };
        assert!(both.contains("queued study items not yet read: okf.md, paper.html"));
        let Some(queued_only) = super::render_pressure(0, &queued) else {
            unreachable!("queued renders")
        };
        assert!(queued_only.starts_with("Internal state: queued study items"));
    }

    #[test]
    fn consolidation_debt_precedes_inbox() {
        let inputs = SchedulerInputs {
            acquisition_debt: ACQUISITION_DEBT_THRESHOLD,
            queued_inbox: 1,
            ..SchedulerInputs::default()
        };
        assert!(matches!(decide(&inputs), SchedulerAction::Dream { .. }));
    }
