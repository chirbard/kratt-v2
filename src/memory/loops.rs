use super::db;
use super::ledger::EdgeRef;

pub const MAX_LOOP_INJECT: usize = 5; // replay-tunable

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpenLoop {
    pub(crate) source: String,
    pub(crate) target: String,
    pub(crate) intention: String,
    pub(crate) due: String,
}

impl OpenLoop {
    pub(crate) fn display(&self) -> String {
        if self.due.is_empty() {
            return self.intention.clone();
        }
        format!("{} (due {})", self.intention, self.due)
    }

    fn edge_ref(&self) -> EdgeRef {
        EdgeRef::new(&self.source, &self.target)
    }
}

pub(crate) struct OpenLoops {
    pub(crate) block: String,
    pub(crate) refs: Vec<EdgeRef>,
}

// Open loops are mandatory context once their anchor appears, and overdue
// loops surface even without an anchor match. Query failures stay best-effort:
// an ordinary chat turn must still proceed with an empty section.
pub(crate) fn load_open_loops(anchor_ids: &[String], today: &str) -> OpenLoops {
    let Ok(conn) = db::open_connection() else {
        return empty_loops();
    };
    let loops = match db::load_loops_for_anchors(&conn, anchor_ids, today, MAX_LOOP_INJECT) {
        Ok(loops) => loops,
        Err(_) => return empty_loops(),
    };
    let block = render_open_loops(&loops);
    let mut refs = Vec::new();
    for open_loop in &loops {
        refs.push(open_loop.edge_ref());
    }
    OpenLoops { block, refs }
}

fn empty_loops() -> OpenLoops {
    OpenLoops {
        block: String::new(),
        refs: Vec::new(),
    }
}

fn render_open_loops(loops: &[OpenLoop]) -> String {
    if loops.is_empty() {
        return String::new();
    }
    let mut block = String::from("## Open loops\n");
    for open_loop in loops {
        block.push_str("- ");
        block.push_str(&open_loop.display());
        block.push('\n');
    }
    block
}

#[cfg(test)]
mod tests {
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
}
