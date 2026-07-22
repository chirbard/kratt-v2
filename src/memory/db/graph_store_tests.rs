use super::super::{block_on, test_support::harness};
use super::{update_edge_confidence_async, write_edge};
use crate::memory::graph::Edge;

fn epistemic_edge(source: &str, provenance: &str, confidence: f64) -> Edge {
    Edge {
        source: source.to_string(),
        target: "target".to_string(),
        content: "belief".to_string(),
        stability: 0.5,
        confidence,
        importance: 0.5,
        subject: false,
        date: "2026-07-16".to_string(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: provenance.to_string(),
        ceiling: None,
        last_decayed_at: String::new(),
    }
}

#[test]
fn writes_and_confidence_updates_obey_provenance_ceiling() {
    let conn = block_on(harness());
    let dreamed = epistemic_edge("dream", "dreamed", 0.9);
    let confirmed = epistemic_edge("user", "user-confirmed", 0.9);
    assert!(write_edge(&conn, &dreamed, "").is_ok());
    assert!(write_edge(&conn, &confirmed, "").is_ok());
    let Ok(Some(dreamed_stored)) = super::load_edge(&conn, "dream", "target") else {
        unreachable!("dreamed edge loads")
    };
    let Ok(Some(confirmed_stored)) = super::load_edge(&conn, "user", "target") else {
        unreachable!("confirmed edge loads")
    };
    assert_eq!(dreamed_stored.confidence, 0.25);
    assert_eq!(confirmed_stored.confidence, 0.9);

    assert!(block_on(update_edge_confidence_async(&conn, "dream", "target", 0.8)).unwrap_or(false));
    let Ok(Some(updated)) = super::load_edge(&conn, "dream", "target") else {
        unreachable!("updated edge loads")
    };
    assert_eq!(updated.confidence, 0.25);
}

#[test]
fn writes_obey_a_derived_edges_own_ceiling() {
    let conn = block_on(harness());
    let mut edge = epistemic_edge("synthesis", "synthesized", 0.9);
    edge.ceiling = Some(0.72);
    assert!(write_edge(&conn, &edge, "").is_ok());
    let Ok(Some(stored)) = super::load_edge(&conn, "synthesis", "target") else {
        unreachable!("derived edge loads")
    };
    assert_eq!(stored.confidence, 0.72);
    assert_eq!(stored.ceiling, Some(0.72));
}
