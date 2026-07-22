    use super::{
        EPISODIC_MAX_HALF_LIFE_DAYS, decay_factor, decay_factor_with, episodic_half_life_days,
        semantic_half_life_days,
    };

    #[test]
    fn semantic_curve_matches_spec_table() {
        let cases = [(0.15, 19.8), (0.30, 56.0), (0.50, 224.0), (1.0, 7168.0)];
        for (stability, expected) in cases {
            let actual = semantic_half_life_days(stability);
            assert!(
                (actual - expected).abs() < 1.0,
                "stability {stability}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn episodic_curve_never_exceeds_cap() {
        let mut step = 0usize;
        while step <= 100 {
            let stability = step as f64 / 100.0;
            assert!(episodic_half_life_days(stability) <= EPISODIC_MAX_HALF_LIFE_DAYS);
            step += 1;
        }
    }

    #[test]
    fn one_half_life_halves_confidence() {
        for (memory_type, stability) in [("semantic", 0.3), ("episodic", 1.0)] {
            let half_life = if memory_type == "episodic" {
                episodic_half_life_days(stability)
            } else {
                semantic_half_life_days(stability)
            };
            let factor = decay_factor(memory_type, stability, half_life);
            assert!((factor - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn nightly_charging_equals_closed_form() {
        let nights = 90usize;
        let stability = 0.3;
        let mut nightly = 1.0;
        let mut night = 0usize;
        while night < nights {
            nightly *= decay_factor_with("semantic", stability, 1.0, 0.15);
            night += 1;
        }
        let closed = decay_factor_with("semantic", stability, nights as f64, 0.15);
        assert!((nightly - closed).abs() < 1e-12);
    }
