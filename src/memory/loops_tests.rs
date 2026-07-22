    use super::{OpenLoop, render_open_loops};

    fn open_loop(intention: &str, due: &str) -> OpenLoop {
        OpenLoop {
            source: "loop-check-server".to_string(),
            target: "home-server".to_string(),
            intention: intention.to_string(),
            due: due.to_string(),
        }
    }

    #[test]
    fn empty_loop_list_renders_no_block() {
        assert!(render_open_loops(&[]).is_empty());
    }

    #[test]
    fn loop_block_renders_contextual_and_due_lines() {
        let loops = vec![
            open_loop("ask whether the service survived", ""),
            open_loop("check whether the file arrived", "2026-07-15"),
        ];
        assert_eq!(
            render_open_loops(&loops),
            "## Open loops\n- ask whether the service survived\n- check whether the file arrived (due 2026-07-15)\n"
        );
    }
