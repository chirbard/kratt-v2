use super::super::{
    block_on,
    test_support::{NODES_DDL, harness},
};
use super::{load_game_edges, load_session_game, load_task_chain};
use turso::{Connection, params};

async fn game_harness() -> Connection {
    let conn = harness().await;
    let _ = conn.execute(NODES_DDL, ()).await;
    let _ = conn
        .execute(
            "INSERT INTO nodes (id, node_type, body) VALUES ('game-chess', 'game', 'the rules')",
            (),
        )
        .await;
    // Episodic hub → game link (excluded from the semantic knowledge load).
    let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, memory_type, origin_session) VALUES ('task-game-1', 'game-chess', 'this session plays chess', 'episodic', 'game-1')",
                (),
            )
            .await;
    conn
}

async fn insert_game_edge(conn: &Connection, target: &str, confidence: f64, expired: Option<&str>) {
    let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, confidence, memory_type, expired_at) VALUES ('game-chess', ?1, ?1, ?2, 'semantic', ?3)",
                params![target, confidence, expired],
            )
            .await;
}

#[test]
fn session_game_returns_the_linked_definition_and_none_without_a_link() {
    let conn = block_on(game_harness());
    let Ok(Some((game_id, node))) = load_session_game(&conn, "task-game-1") else {
        unreachable!("game is linked")
    };
    assert_eq!(game_id, "game-chess");
    assert_eq!(node.body, "the rules");
    let Ok(missing) = load_session_game(&conn, "task-game-2") else {
        unreachable!("load succeeds")
    };
    assert!(missing.is_none());
}

#[test]
fn game_edges_load_highest_confidence_first_and_exclude_expired() {
    let conn = block_on(game_harness());
    block_on(insert_game_edge(&conn, "rule-low", 0.4, None));
    block_on(insert_game_edge(&conn, "rule-high", 0.9, None));
    block_on(insert_game_edge(
        &conn,
        "rule-gone",
        0.95,
        Some("2026-01-01"),
    ));
    let Ok(edges) = load_game_edges(&conn, "game-chess", 60) else {
        unreachable!("edges load")
    };
    assert_eq!(edges.len(), 2, "expired excluded, episodic link excluded");
    assert_eq!(edges[0].target, "rule-high");
    assert_eq!(edges[1].target, "rule-low");
}

#[test]
fn game_edges_cap_keeps_the_highest_confidence() {
    let conn = block_on(game_harness());
    block_on(insert_game_edge(&conn, "rule-a", 0.3, None));
    block_on(insert_game_edge(&conn, "rule-b", 0.9, None));
    let Ok(edges) = load_game_edges(&conn, "game-chess", 1) else {
        unreachable!("edges load")
    };
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].target, "rule-b");
}

async fn insert_ply(
    conn: &Connection,
    hub: &str,
    ply: usize,
    session: &str,
    expired: Option<&str>,
) {
    let target = format!("{hub}-ply-{ply:03}");
    let content = format!("ply {ply} (assistant): move{ply}");
    let _ = conn
            .execute(
                "INSERT INTO edges (source, target, content, memory_type, origin_session, expired_at) VALUES (?1, ?2, ?3, 'episodic', ?4, ?5)",
                params![hub, target.as_str(), content.as_str(), session, expired],
            )
            .await;
}

#[test]
fn chain_returns_plies_in_order_for_this_session_only() {
    let conn = block_on(harness());
    block_on(insert_ply(&conn, "task-game-1", 2, "game-1", None));
    block_on(insert_ply(&conn, "task-game-1", 1, "game-1", None));
    // A different session's hub must never leak in.
    block_on(insert_ply(&conn, "task-game-2", 1, "game-2", None));

    let Ok(chain) = load_task_chain(&conn, "game-1", 240) else {
        unreachable!("chain loads")
    };
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].target, "task-game-1-ply-001");
    assert_eq!(chain[1].target, "task-game-1-ply-002");
}

#[test]
fn chain_excludes_expired_plies() {
    let conn = block_on(harness());
    block_on(insert_ply(&conn, "task-game-1", 1, "game-1", None));
    block_on(insert_ply(
        &conn,
        "task-game-1",
        2,
        "game-1",
        Some("2026-01-01"),
    ));

    let Ok(chain) = load_task_chain(&conn, "game-1", 240) else {
        unreachable!("chain loads")
    };
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].target, "task-game-1-ply-001");
}

#[test]
fn cap_keeps_the_most_recent_plies_in_order() {
    let conn = block_on(harness());
    for ply in 1..=5 {
        block_on(insert_ply(&conn, "task-game-1", ply, "game-1", None));
    }
    let Ok(chain) = load_task_chain(&conn, "game-1", 2) else {
        unreachable!("chain loads")
    };
    assert_eq!(chain.len(), 2);
    // Newest two plies, still in ascending order.
    assert_eq!(chain[0].target, "task-game-1-ply-004");
    assert_eq!(chain[1].target, "task-game-1-ply-005");
}
