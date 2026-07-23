use super::super::{
    block_on,
    test_support::{EVENTS_DDL, NODES_DDL, harness},
};
use super::{
    insert_interest, load_all_edges_semantic_async, load_due_loops_async, load_loops_for_anchors,
    load_unembedded_edges_async, top_interests_with,
};
use turso::{Connection, params};

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

async fn loop_harness() -> Connection {
    let conn = harness().await;
    let _ = conn.execute(NODES_DDL, ()).await;
    conn
}

async fn insert_loop(
    conn: &Connection,
    loop_id: &str,
    anchor_id: &str,
    details: (&str, &str, &str, Option<&str>),
) {
    let (intention, due, edge_date, expired_at) = details;
    let _ = conn
        .execute(
            "INSERT OR IGNORE INTO nodes (id, node_type) VALUES (?1, 'entity')",
            params![anchor_id],
        )
        .await;
    let _ = conn
        .execute(
            "INSERT INTO nodes (id, node_type, body, date) VALUES (?1, 'loop', ?2, ?3)",
            params![loop_id, intention, due],
        )
        .await;
    let content = format!("open loop: {intention}");
    let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, stability, confidence, importance, memory_type, date, expired_at) VALUES (?1, ?2, ?3, 0.9, 0.9, 1.0, 'semantic', ?4, ?5)",
                params![loop_id, anchor_id, content.as_str(), edge_date, expired_at],
            )
            .await;
}

async fn seed_anchor_loop_cases(conn: &Connection) {
    let cases = [
        (
            "loop-context",
            "home-server",
            ("ask about the service", "", "2026-07-02", None),
        ),
        (
            "loop-due",
            "file-transfer",
            (
                "check whether the file arrived",
                "2026-07-10",
                "2026-07-01",
                None,
            ),
        ),
        (
            "loop-future",
            "other-topic",
            ("ask about the future", "2026-07-12", "2026-07-03", None),
        ),
        (
            "loop-closed",
            "home-server",
            (
                "closed intention",
                "",
                "2026-06-30",
                Some("2026-07-09T12:00:00Z"),
            ),
        ),
    ];
    for (loop_id, anchor_id, details) in cases {
        insert_loop(conn, loop_id, anchor_id, details).await;
    }
}

async fn seed_due_loop_cases(conn: &Connection) {
    let cases = [
        (
            "loop-past",
            "past-topic",
            ("past intention", "2026-07-10", "2026-07-01", None),
        ),
        (
            "loop-today",
            "today-topic",
            ("today intention", "2026-07-11", "2026-07-02", None),
        ),
        (
            "loop-context",
            "context-topic",
            ("context only", "", "2026-07-03", None),
        ),
        (
            "loop-future",
            "future-topic",
            ("future intention", "2026-07-12", "2026-07-04", None),
        ),
        (
            "loop-closed",
            "closed-topic",
            (
                "closed intention",
                "2026-07-09",
                "2026-07-05",
                Some("2026-07-10T00:00:00Z"),
            ),
        ),
    ];
    for (loop_id, anchor_id, details) in cases {
        insert_loop(conn, loop_id, anchor_id, details).await;
    }
}

#[test]
fn loop_loader_matches_anchors_includes_due_and_excludes_closed() {
    let conn = block_on(loop_harness());
    block_on(seed_anchor_loop_cases(&conn));

    let anchors = vec!["home-server".to_string()];
    let Ok(loops) = load_loops_for_anchors(&conn, &anchors, "2026-07-11", 5) else {
        unreachable!("loops load")
    };
    assert_eq!(loops.len(), 2);
    assert_eq!(loops[0].source, "loop-due", "oldest loop comes first");
    assert_eq!(loops[1].source, "loop-context");
}

#[test]
fn loop_loader_cap_keeps_oldest_edge_dates_first() {
    let conn = block_on(loop_harness());
    block_on(insert_loop(
        &conn,
        "loop-third",
        "topic-third",
        ("third", "2026-07-08", "2026-07-10", None),
    ));
    block_on(insert_loop(
        &conn,
        "loop-first",
        "topic-first",
        ("first", "2026-07-10", "2026-07-08", None),
    ));
    block_on(insert_loop(
        &conn,
        "loop-second",
        "topic-second",
        ("second", "2026-07-09", "2026-07-09", None),
    ));

    let Ok(loops) = load_loops_for_anchors(&conn, &[], "2026-07-11", 2) else {
        unreachable!("loops load")
    };
    assert_eq!(loops.len(), 2);
    assert_eq!(loops[0].source, "loop-first");
    assert_eq!(loops[1].source, "loop-second");
}

#[test]
fn loop_loader_preserves_insertion_order_for_same_day() {
    let conn = block_on(loop_harness());
    block_on(insert_loop(
        &conn,
        "z-first",
        "topic-first",
        ("first", "2026-07-10", "2026-07-10", None),
    ));
    block_on(insert_loop(
        &conn,
        "a-second",
        "topic-second",
        ("second", "2026-07-10", "2026-07-10", None),
    ));
    let Ok(loops) = load_loops_for_anchors(&conn, &[], "2026-07-11", 1) else {
        unreachable!("loops load")
    };
    assert_eq!(loops[0].source, "z-first");
}

#[test]
fn due_loop_loader_formats_only_live_due_loops() {
    let conn = block_on(loop_harness());
    block_on(seed_due_loop_cases(&conn));

    let Ok(loops) = block_on(load_due_loops_async(&conn, "2026-07-11")) else {
        unreachable!("due loops load")
    };
    let mut displays = Vec::new();
    for open_loop in &loops {
        displays.push(open_loop.display());
    }
    assert_eq!(
        displays,
        vec![
            "past intention (due 2026-07-10)",
            "today intention (due 2026-07-11)"
        ]
    );
}

#[test]
fn ordinary_retrieval_excludes_loop_anchor_edges() {
    let conn = block_on(loop_harness());
    block_on(insert_loop(
        &conn,
        "loop-reminder",
        "topic",
        ("ask about the service", "", "2026-07-10", None),
    ));
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
