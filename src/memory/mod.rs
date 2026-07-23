mod db;
mod embed;
mod graph;
mod toon_display;

#[allow(unused_imports)]
pub use db::{
    PassRun, db_init, insert_message, last_sync_error, read_message_context, register_session,
    search_messages_fts, startup_database, sync_now, sync_push_best_effort,
};
pub(crate) use db::{Finding, acquire_write_lock, open_connection};
pub(crate) use db::{
    Interest, PingPolicy, RecentLearning, block_on, conversation_boundary_async,
    latest_assistant_message_timestamp_async, load_inbox_snapshot_async,
    load_open_curiosity_notes_async, load_ping_policy,
    load_recent_learning_async, mark_finding_surfaced, top_interests, unsurfaced_findings,
};
#[cfg(test)]
pub(crate) use db::{insert_interest, top_interests_with};
pub use embed::{
    SemanticEdge, edges_to_toon, run_embed_pass, select_semantic_edges,
    select_semantic_edges_for_chat,
};
