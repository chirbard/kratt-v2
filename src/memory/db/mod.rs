mod core;
mod findings;
mod graph_store;
#[allow(dead_code)]
mod inbox;
mod migrations;
mod pass_runs;
mod retrieval;
mod schema;
mod sessions;
mod task_store;
mod values;

#[cfg(test)]
mod test_support;

pub(crate) use core::{acquire_write_lock, block_on};
pub use core::{
    last_sync_error, open_connection, startup_database, sync_now, sync_push_best_effort,
    timestamp_now,
};
pub(crate) use findings::{
    Finding, PingPolicy, count_findings_in_windows, divert_ping_topic, load_ping_policy,
    mark_finding_surfaced, resume_pings, set_ping_mute, unsurfaced_findings,
    unsurfaced_findings_in_windows,
};
pub use graph_store::{
    begin_transaction, commit_transaction, db_init, erase_edge, erase_node, graph_diff_toon,
    load_edge, load_edges_for_node, load_graph, load_node, load_nodes_by_ids, write_edge,
    write_node,
};
use graph_store::{clamped_confidence, node_count};
pub(crate) use graph_store::{
    ensure_entity_endpoints, parse_edge_row, touch_edge_dates, update_edge_reinforcement,
};
pub use inbox::{inbox_source_is_queued, insert_inbox_item, load_inbox_outcomes};
#[cfg(test)]
pub(crate) use inbox::{count_queued_inbox, oldest_queued_inbox, resolve_inbox_item};
use migrations::{
    add_edge_columns, add_interest_node_columns, backfill_edge_epistemics, clamp_stability_once,
    drop_node_agent_views_table, migrate_entity_bodies_matching_ids, migrate_legacy_if_empty,
};
pub use pass_runs::PassRun;
pub(crate) use pass_runs::{
    consecutive_failures_async, last_successful_pass_async, pass_runs_in_window_async,
};
use retrieval::delete_edge_actor_views_async;
pub(crate) use retrieval::{
    Interest, RecentLearning, RegionTransition, insert_interest, last_suggestion_for_session,
    load_inbox_snapshot_async, load_open_curiosity_notes_async,
    load_previous_turn_region, load_recent_learning_async, load_region_transitions, load_turn_region,
    load_unembedded_interests, save_interest_embedding, top_interests,
};
#[cfg(test)]
pub(crate) use retrieval::top_interests_with;
pub use retrieval::{
    SemanticEdgeRow, edge_count, load_all_edges_semantic, load_unembedded_edges,
    parse_vector32_json, query_edges_by_vector,
    save_edge_embedding, vector32_json,
};
#[cfg(test)]
use schema::create_pass_run_table;
use schema::init_schema;
pub(crate) use sessions::{conversation_boundary_async, latest_assistant_message_timestamp_async};
pub use sessions::{
    insert_message, read_message_context, register_ingestion_session, register_session,
    search_messages_fts, source_session_id,
};
pub use task_store::{
    count_task_chain, ensure_task_hub, load_game_edges, load_session_game, load_task_chain, next_ply,
};
pub(crate) use values::{int, real, round2, text};
use values::{optional_real, optional_text, text_or};

use findings::DEFAULT_PING_MUTED_UNTIL;
