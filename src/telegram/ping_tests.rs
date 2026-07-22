    use super::{PING_NOVELTY_MIN, PingGate, select_finding};
    use crate::memory::{Finding, PingPolicy};

    fn finding(topic: &str, novelty: f64) -> Finding {
        Finding {
            finding_id: 1,
            interest_topic: topic.to_string(),
            title: "A finding".to_string(),
            url: "https://example.com".to_string(),
            novelty,
        }
    }

    #[test]
    fn default_mute_and_active_policy_suppress_candidates() {
        let muted = PingPolicy {
            muted_until: "9999-12-31T23:59:59".to_string(),
            diverts: Vec::new(),
        };
        assert!(select_finding(&[finding("Rust", 0.9)], &muted, "2026-07-14T12:00:00").is_none());

        let diverted = PingPolicy {
            muted_until: String::new(),
            diverts: vec![("rust".to_string(), "2026-07-15T00:00:00".to_string())],
        };
        assert!(
            select_finding(&[finding("Rust", 0.9)], &diverted, "2026-07-14T12:00:00").is_none()
        );
    }

    #[test]
    fn resumed_policy_selects_only_above_the_ping_bar() {
        let resumed = PingPolicy::default();
        assert!(
            select_finding(
                &[finding("Rust", PING_NOVELTY_MIN - 0.01)],
                &resumed,
                "2026-07-14T12:00:00"
            )
            .is_none()
        );
        assert!(
            select_finding(
                &[finding("Rust", PING_NOVELTY_MIN)],
                &resumed,
                "2026-07-14T12:00:00"
            )
            .is_some()
        );
    }

    #[test]
    fn rate_gate_enforces_cooldown_cap_and_day_rollover() {
        let mut gate = PingGate::new();
        assert!(gate.allows("2026-07-14", 100));
        gate.record(100);
        assert!(!gate.allows("2026-07-14", 101));
        assert!(gate.allows("2026-07-14", 10_900));
        gate.record(10_900);
        assert!(!gate.allows("2026-07-14", 99_999));
        assert!(gate.allows("2026-07-15", 100_000));
    }
