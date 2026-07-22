    use super::{
        CONFIDENCE_MAX, CONFIDENCE_MIN, REINFORCE_RATE, SliceEvent, TEMPORAL_FLOOR,
        clamp_confidence, compute_edge_deltas, reinforced_stability, temporal_weight,
    };

    fn edge_event(activity_weight: f64) -> SliceEvent {
        named_edge_event("a", "b", activity_weight)
    }

    fn named_edge_event(source: &str, target: &str, activity_weight: f64) -> SliceEvent {
        SliceEvent {
            source: source.to_string(),
            target: target.to_string(),
            activity_weight,
        }
    }

    const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";

    async fn insert_event(
        conn: &turso::Connection,
        turn_id: &str,
        event_type: &str,
        actor_id: &str,
    ) {
        let _ = conn
            .execute(
                "INSERT INTO events (session_id, turn_id, actor_id, event_type, source, target) VALUES ('s1', ?1, ?2, ?3, 'a', 'b')",
                turso::params![turn_id, actor_id, event_type],
            )
            .await;
    }

    // A game outcome must credit what the game used (game-tagged injections)
    // but never the ranked beliefs of every chat turn (local-tagged ones).
    #[test]
    fn outcome_slice_includes_game_injections_but_not_ranked_ones() {
        let slice = crate::memory::db::block_on(async {
            let Ok(database) = turso::Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(EVENTS_DDL, ()).await;
            insert_event(&conn, "turn-1", "injected", "game").await;
            insert_event(&conn, "turn-1", "injected", "local").await;
            insert_event(&conn, "turn-1", "retrieved", "explorer").await;
            let Ok(session_slice) = super::load_session_slice_async(&conn, "s1", "").await else {
                unreachable!("slice loads")
            };
            let Ok(turn_slice) = super::load_session_slice_async(&conn, "s1", "turn-1").await
            else {
                unreachable!("turn slice loads")
            };
            (session_slice, turn_slice)
        });
        assert_eq!(slice.0.len(), 2, "session excludes the ranked injection");
        assert_eq!(
            slice.1.len(),
            3,
            "turn includes local injection and retrieval"
        );
    }

    #[test]
    fn temporal_weight_single_event_is_full_weight() {
        assert!((temporal_weight(0, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn temporal_weight_first_event_is_floor() {
        assert!((temporal_weight(0, 5) - TEMPORAL_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn temporal_weight_last_event_is_full_weight() {
        assert!((temporal_weight(4, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_edge_sums_activity_weight() {
        let slice = vec![edge_event(1.0), edge_event(2.0)];
        let deltas = compute_edge_deltas(&slice, 1.0);
        assert_eq!(deltas.len(), 1);
        assert!((deltas[0].1.activity_weight - 3.0).abs() < 1e-9);
    }

    #[test]
    fn delta_sign_follows_outcome_sign() {
        let slice = vec![edge_event(1.0)];
        let positive = compute_edge_deltas(&slice, 1.0);
        let negative = compute_edge_deltas(&slice, -1.0);
        assert!(positive[0].1.delta > 0.0);
        assert!(negative[0].1.delta < 0.0);
    }

    #[test]
    fn clamp_confidence_bounds_both_ends() {
        assert!((clamp_confidence(0.9, 10.0) - CONFIDENCE_MAX).abs() < 1e-9);
        assert!((clamp_confidence(0.1, -10.0) - CONFIDENCE_MIN).abs() < 1e-9);
    }

    #[test]
    fn positive_delta_never_lowers_over_ceiling_edge() {
        // An edge already above the ceiling must stay put under a positive nudge.
        assert!((clamp_confidence(0.98, 0.05) - 0.98).abs() < 1e-9);
        assert!((clamp_confidence(0.02, -0.05) - 0.02).abs() < 1e-9);
    }

    #[test]
    fn only_positive_reinforcement_grows_stability() {
        assert_eq!(reinforced_stability(0.4, 0.05), 0.45);
        assert_eq!(reinforced_stability(0.4, -0.05), 0.4);
        assert_eq!(reinforced_stability(1.0, 0.05), 1.0);
    }

    #[test]
    fn normalization_caps_max_delta_at_rate() {
        // Regardless of slice size, the most-active edge moves by REINFORCE_RATE.
        let slice = vec![
            named_edge_event("a", "b", 2.0),
            named_edge_event("a", "b", 2.0),
            named_edge_event("c", "d", 3.0),
        ];
        let deltas = compute_edge_deltas(&slice, 1.0);
        let mut max_delta = 0.0f64;
        for (_, edge_delta) in &deltas {
            if edge_delta.delta > max_delta {
                max_delta = edge_delta.delta;
            }
        }
        assert!((max_delta - REINFORCE_RATE).abs() < 1e-9);
    }

    #[test]
    fn normalization_scales_less_active_edge_proportionally() {
        let slice = vec![
            named_edge_event("a", "b", 1.0),
            named_edge_event("c", "d", 2.0),
        ];
        // Single-event positions both weight 1.0 temporally (n=2: pos0=floor, pos1=1.0),
        // so the ratio is driven by activity weight — the weaker edge stays under RATE.
        let deltas = compute_edge_deltas(&slice, 1.0);
        for (edge, edge_delta) in &deltas {
            assert!(edge_delta.delta <= REINFORCE_RATE + 1e-9);
            if edge.source == "a" {
                assert!(edge_delta.delta < REINFORCE_RATE);
            }
        }
    }
