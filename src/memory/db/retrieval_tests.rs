use super::super::{
    block_on,
    test_support::{EVENTS_DDL, NODES_DDL, harness},
};
use super::{
    insert_interest, load_all_edges_semantic_async, load_unembedded_edges_async, top_interests_with,
};

#[test]
fn top_interests_unifies_registered_and_derived_regions_and_caps() {
    let conn = block_on(harness());
    block_on(async {
        let _ = conn.execute(NODES_DDL, ()).await;
        let _ = conn.execute(EVENTS_DDL, ()).await;
        let _ = conn
            .execute(
                "INSERT INTO nodes (id, node_type) VALUES ('rust', 'entity')",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "INSERT INTO events (ts, actor_type, actor_id, event_type, source) VALUES
                ('2026-07-14T01:00:00Z', 'agent', 'local', 'marked', 'rust'),
                ('2026-07-14T02:00:00Z', 'agent', 'local', 'reinforced', 'rust')",
                (),
            )
            .await;
    });
    assert!(insert_interest(&conn, "libsql internals", "user").is_ok());
    let interests = block_on(top_interests_with(&conn, 10)).unwrap_or_default();
    let mut found_registered = false;
    let mut found_derived = false;
    for item in interests {
        if item.topic == "libsql internals" {
            found_registered = true;
        }
        if item.topic == "rust" {
            found_derived = true;
        }
    }
    assert!(found_registered);
    assert!(found_derived);
    let capped = block_on(top_interests_with(&conn, 1)).unwrap_or_default();
    assert_eq!(capped.len(), 1);
    assert_eq!(capped[0].topic, "rust");
}

#[test]
fn ordinary_retrieval_excludes_loop_anchor_edges() {
    let conn = block_on(async {
        let conn = harness().await;
        let _ = conn.execute(NODES_DDL, ()).await;
        let _ = conn.execute("INSERT INTO nodes (id, node_type) VALUES ('loop-reminder', 'loop'), ('topic', 'entity')", ()).await;
        let _ = conn.execute("INSERT INTO edges (source, target, content) VALUES ('loop-reminder', 'topic', 'open loop: ask about the service')", ()).await;
        conn
    });
    assert!(insert_interest(&conn, "libsql internals", "user").is_ok());
    block_on(async {
        let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content) VALUES
                     ('ordinary', 'topic', 'ordinary belief'),
                     ('libsql-internals', 'topic', 'standing interest edge')",
                (),
            )
            .await;
    });
    // Standing-node edges ARE embedded for their dedicated similarity
    // paths…
    let pending = block_on(load_unembedded_edges_async(&conn));
    assert_eq!(pending.len(), 3);
    // …but stay excluded from ranked retrieval.
    let semantic = block_on(load_all_edges_semantic_async(&conn));
    assert_eq!(semantic.len(), 1);
    assert_eq!(semantic[0].0, "ordinary");
}
