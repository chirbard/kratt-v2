use super::db;
use super::ledger::EdgeRef;

pub const MAX_LOOP_INJECT: usize = 5; // replay-tunable
// Max cosine distance for the similarity trigger: a loop this close to the
// query surfaces even when its anchor never entered the ranked TOP_K (a fresh
// anchor's only edge is the loop edge itself, which ranked retrieval excludes).
const LOOP_SIMILARITY_MAX_DISTANCE: f64 = 0.55; // replay-tunable

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
pub(crate) fn load_open_loops(
    anchor_ids: &[String],
    today: &str,
    query_vec: &[f64],
) -> OpenLoops {
    let Ok(conn) = db::open_connection() else {
        return empty_loops();
    };
    let mut loops = match db::load_loops_for_anchors(&conn, anchor_ids, today, MAX_LOOP_INJECT) {
        Ok(loops) => loops,
        Err(_) => return empty_loops(),
    };
    merge_similar_loops(&conn, query_vec, &mut loops);
    let block = render_open_loops(&loops);
    let mut refs = Vec::new();
    for open_loop in &loops {
        refs.push(open_loop.edge_ref());
    }
    OpenLoops { block, refs }
}

// Similarity trigger: best-effort, deduplicated against anchor/due hits,
// respecting the overall cap.
fn merge_similar_loops(conn: &turso::Connection, query_vec: &[f64], loops: &mut Vec<OpenLoop>) {
    if query_vec.is_empty() || loops.len() >= MAX_LOOP_INJECT {
        return;
    }
    let similar = match db::load_loops_by_similarity(
        conn,
        query_vec,
        LOOP_SIMILARITY_MAX_DISTANCE,
        MAX_LOOP_INJECT,
    ) {
        Ok(similar) => similar,
        Err(_) => return,
    };
    for candidate in similar {
        if loops.len() >= MAX_LOOP_INJECT {
            return;
        }
        if !loops.contains(&candidate) {
            loops.push(candidate);
        }
    }
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
#[path = "loops_tests.rs"]
mod tests;
