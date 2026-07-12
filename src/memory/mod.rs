mod agent;
mod db;
mod decay;
mod dream;
mod embed;
mod explore;
mod graph;
mod hygiene;
mod ingest;
mod ledger;
mod loops;
mod reinforce;
mod routing;
pub mod scheduler;
pub use scheduler::pressure_line;
mod synthesis;
mod task_chain;

pub use agent::{MemoryEvent, MemorySeats, run_memory_agent};
pub(crate) use agent::{classify_reply_label, describe_tool_call, parse_tool_calls};
pub use db::{
    db_init, insert_message, last_sync_error, latest_session_id, read_message_context,
    register_session, search_messages_fts, sync_now, sync_pull_best_effort, sync_push_best_effort,
};
pub(crate) use db::{InboxItem, oldest_queued_inbox, open_connection, resolve_inbox_item};
pub use dream::{DreamEvent, run_dream_pass};
pub use embed::{SemanticEdge, edges_to_toon, run_embed_pass, select_semantic_edges_routed};
pub use explore::{ExploreEvent, run_exploration_pass};
pub use ingest::{IngestEvent, run_focused_ingestion_pass, run_ingestion_pass};
pub use ledger::{
    Actor, Attribution, DayDigest, EdgeRef, Event, EventContext, NightWindow, day_digest,
    newest_dream_session_id, recompute_positions, record_events, turn_attribution,
};
pub(crate) use loops::{OpenLoops, load_open_loops};
#[allow(unused_imports)]
pub use reinforce::{record_move_outcome, record_outcome};
pub use routing::Routing;
pub use task_chain::{GameKnowledge, TaskChain, load_game_knowledge, load_task_chain};
