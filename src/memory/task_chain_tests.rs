    use super::{game_display_name, render_game_knowledge, render_task_chain};
    use crate::memory::graph::Edge;

    fn game_edge(content: &str, confidence: f64) -> Edge {
        Edge {
            source: "game-chess".to_string(),
            target: "the-king-moves-one-square".to_string(),
            content: content.to_string(),
            stability: 0.9,
            confidence,
            importance: 1.0,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
            memory_type: "semantic".to_string(),
            origin_session: String::new(),
            provenance: "legacy".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    #[test]
    fn game_display_name_strips_the_prefix() {
        assert_eq!(game_display_name("game-chess"), "chess");
        assert_eq!(game_display_name("chess"), "chess");
    }

    #[test]
    fn game_block_renders_definition_body_then_one_line_per_edge() {
        let edges = vec![
            game_edge("the king moves one square", 0.9),
            game_edge("develop knights before bishops", 0.6),
        ];
        let block = render_game_knowledge("chess", "user plays White, reply with the move only", &edges);
        assert_eq!(
            block,
            "## Game: chess\nuser plays White, reply with the move only\nthe king moves one square\ndevelop knights before bishops\n"
        );
    }

    #[test]
    fn game_block_omits_an_empty_definition_body() {
        let block = render_game_knowledge("chess", "", &[game_edge("a rule", 0.9)]);
        assert_eq!(block, "## Game: chess\na rule\n");
    }

    fn ply_edge(content: &str) -> Edge {
        Edge {
            source: "task-game-1".to_string(),
            target: "task-game-1-ply-001".to_string(),
            content: content.to_string(),
            stability: 0.3,
            confidence: 1.0,
            importance: 1.0,
            subject: false,
            date: String::new(),
            occurred_at: None,
            expired_at: None,
            memory_type: "episodic".to_string(),
            origin_session: "game-1".to_string(),
            provenance: "user-confirmed".to_string(),
            ceiling: None,
            last_decayed_at: String::new(),
        }
    }

    #[test]
    fn empty_chain_renders_no_block() {
        assert!(render_task_chain(&[]).is_empty());
    }

    #[test]
    fn chain_renders_header_and_one_line_per_ply_in_order() {
        let edges = vec![
            ply_edge("ply 1 (assistant): e4"),
            ply_edge("ply 2 (user): e5"),
        ];
        let block = render_task_chain(&edges);
        assert_eq!(
            block,
            "## Current task (in order)\nply 1 (assistant): e4\nply 2 (user): e5\n"
        );
    }
