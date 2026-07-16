use super::db;
use super::graph::{Edge, normalize_id};
use super::ledger::EdgeRef;

// A game you are currently playing is not optional context, so the chain
// bypasses TOP_K ranking entirely. This cap is only a guard against a
// pathological chain blowing the context — a chess game never reaches it.
pub const MAX_CHAIN_INJECT: usize = 240; // replay-tunable

// Rules + strategies loaded onto a game block. Like the chain, this is outside
// TOP_K — the rules of the game you are playing are mandatory context.
pub const MAX_GAME_KNOWLEDGE: usize = 60; // replay-tunable

// The definition body, rules and strategies of the game this session is
// playing, ready for injection. Empty when the session has no game link.
pub struct GameKnowledge {
    pub block: String,
    pub refs: Vec<EdgeRef>,
}

// The display name of a game node — its id minus the "game-" prefix.
pub(crate) fn game_display_name(game_id: &str) -> String {
    game_id.strip_prefix("game-").unwrap_or(game_id).to_string()
}

// Loads the game-definition body plus all live semantic rule/strategy edges on
// the game node this session's hub links to. No game link → empty block and no
// refs, so a normal chat turn pays nothing.
pub fn load_game_knowledge(session_id: &str) -> GameKnowledge {
    let Ok(conn) = db::open_connection() else {
        return empty_knowledge();
    };
    let hub_id = normalize_id(&format!("task-{session_id}"));
    let Ok(Some((game_id, node))) = db::load_session_game(&conn, &hub_id) else {
        return empty_knowledge();
    };
    let edges = match db::load_game_edges(&conn, &game_id, MAX_GAME_KNOWLEDGE) {
        Ok(edges) => edges,
        Err(_) => return empty_knowledge(),
    };
    let name = game_display_name(&game_id);
    let block = render_game_knowledge(&name, &node.body, &edges);
    let mut refs = Vec::new();
    for edge in &edges {
        refs.push(EdgeRef::new(&edge.source, &edge.target));
    }
    GameKnowledge { block, refs }
}

fn empty_knowledge() -> GameKnowledge {
    GameKnowledge {
        block: String::new(),
        refs: Vec::new(),
    }
}

// Definition body first (presentation rules live there — injected every turn),
// then one line per rule/strategy edge.
fn render_game_knowledge(name: &str, body: &str, edges: &[Edge]) -> String {
    let mut block = format!("## Game: {name}\n");
    if !body.is_empty() {
        block.push_str(body);
        block.push('\n');
    }
    for edge in edges {
        block.push_str(&edge.content);
        block.push('\n');
    }
    block
}

// The ordered same-session move chain, ready for injection: the rendered block
// and the edge refs whose `injected` events retrieval must record.
pub struct TaskChain {
    pub block: String,
    pub refs: Vec<EdgeRef>,
}

// Loads the current session's task-hub chain (complete and in ply order) and
// renders it. Empty chain → empty block and no refs, so a normal chat turn
// pays nothing.
pub fn load_task_chain(session_id: &str) -> TaskChain {
    let Ok(conn) = db::open_connection() else {
        return empty_chain();
    };
    let edges = match db::load_task_chain(&conn, session_id, MAX_CHAIN_INJECT) {
        Ok(edges) => edges,
        Err(_) => return empty_chain(),
    };
    // Only a full-cap load can be a truncation — the common (short) chain and the
    // empty chat turn never pay for the extra count query.
    if edges.len() == MAX_CHAIN_INJECT {
        log_if_truncated(&conn, session_id, edges.len());
    }
    let block = render_task_chain(&edges);
    let mut refs = Vec::new();
    for edge in &edges {
        refs.push(EdgeRef::new(&edge.source, &edge.target));
    }
    TaskChain { block, refs }
}

fn log_if_truncated(conn: &turso::Connection, session_id: &str, kept: usize) {
    let hub_id = normalize_id(&format!("task-{session_id}"));
    if db::count_task_chain(conn, &hub_id).is_ok_and(|total| total > kept) {
        crate::runlog::log(
            "chat",
            &format!("task chain truncated to most recent {kept} plies"),
        );
    }
}

fn empty_chain() -> TaskChain {
    TaskChain {
        block: String::new(),
        refs: Vec::new(),
    }
}

// Pure render: one line per ply, in order. The edge content already carries
// "ply <N> (<actor>): <move>", so rendering is a straight join under a header.
fn render_task_chain(edges: &[Edge]) -> String {
    if edges.is_empty() {
        return String::new();
    }
    let mut block = String::from("## Current task (in order)\n");
    for edge in edges {
        block.push_str(&edge.content);
        block.push('\n');
    }
    block
}

#[cfg(test)]
mod tests {
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
}
