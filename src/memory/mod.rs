mod agent;
mod db;
mod decay;
pub mod decay_curve;
mod decay_simulator;
mod dream;
mod embed;
mod explore;
mod graph;
mod heavy;
mod hygiene;
mod ingest;
mod ledger;
mod loops;
mod news;
mod reinforce;
mod routing;
mod sequence;
pub mod scheduler;
pub use scheduler::pressure_line;
pub use sequence::sequence_line;
mod synthesis;
mod task_chain;
mod toon_display;

pub use agent::{MemoryEvent, MemorySeats, run_memory_agent};
pub(crate) use agent::{classify_reply_label, describe_tool_call, parse_tool_calls};
#[allow(unused_imports)]
pub use db::{
    PassRun, db_init, finish_pass_run, insert_message, last_sync_error, latest_session_id,
    read_message_context, register_session, search_messages_fts, start_pass_run, startup_database,
    sync_now, sync_push_best_effort,
};
pub(crate) use db::{Finding, InboxItem, acquire_write_lock, oldest_queued_inbox, open_connection, resolve_inbox_item};
pub(crate) use db::{
    Interest, PingPolicy, RecentLearning, block_on, conversation_boundary_async,
    latest_assistant_message_timestamp_async, load_inbox_snapshot_async,
    load_open_curiosity_notes_async, load_open_intentions_async, load_ping_policy,
    load_recent_learning_async, mark_finding_surfaced, top_interests, unsurfaced_findings,
};
#[cfg(test)]
pub(crate) use db::{insert_interest, top_interests_with};
pub use dream::{DreamEvent, run_dream_pass};
pub use decay_simulator::run as run_decay_simulator;
pub use embed::{SemanticEdge, edges_to_toon, run_embed_pass, select_semantic_edges, select_semantic_edges_routed};
pub use explore::{ExploreEvent, run_exploration_pass};
pub use ingest::{IngestEvent, run_focused_ingestion_pass, run_ingestion_pass};
pub(crate) use news::run_news_scan;
pub use ledger::{
    Actor, Attribution, DayDigest, EdgeRef, Event, EventContext, NightWindow, day_digest,
    newest_dream_session_id, recompute_positions, record_events, turn_attribution,
};
pub(crate) use ledger::catch_up_digest_async;
pub(crate) use loops::{OpenLoops, load_open_loops};
#[allow(unused_imports)]
pub use reinforce::record_outcome;
pub use routing::Routing;
pub use task_chain::{GameKnowledge, TaskChain, load_game_knowledge, load_task_chain};
