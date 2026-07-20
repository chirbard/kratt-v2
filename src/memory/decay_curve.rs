pub const SEMANTIC_BASE_HALF_LIFE: f64 = 7.0; // replay-tunable.
#[allow(dead_code)] // Staged for the production-decay step; simulator sweeps the `_with` variant.
pub const STABILITY_DOUBLING: f64 = 0.15; // replay-tunable.
pub const EPISODIC_HALF_LIFE_DAYS: f64 = 1.0; // replay-tunable.
pub const EPISODIC_MAX_HALF_LIFE_DAYS: f64 = 3.0; // replay-tunable.

#[allow(dead_code)] // Pure public curve required before production wiring.
pub fn semantic_half_life_days(stability: f64) -> f64 {
    semantic_half_life_days_with(stability, STABILITY_DOUBLING)
}

pub fn semantic_half_life_days_with(stability: f64, stability_doubling: f64) -> f64 {
    let safe_doubling = stability_doubling.max(f64::EPSILON);
    SEMANTIC_BASE_HALF_LIFE * 2.0_f64.powf(stability.max(0.0) / safe_doubling)
}

pub fn episodic_half_life_days(stability: f64) -> f64 {
    let _ = stability;
    EPISODIC_HALF_LIFE_DAYS.min(EPISODIC_MAX_HALF_LIFE_DAYS)
}

#[allow(dead_code)] // Pure public curve required before production wiring.
pub fn decay_factor(memory_type: &str, stability: f64, elapsed_days: f64) -> f64 {
    decay_factor_with(memory_type, stability, elapsed_days, STABILITY_DOUBLING)
}

pub fn decay_factor_with(
    memory_type: &str,
    stability: f64,
    elapsed_days: f64,
    stability_doubling: f64,
) -> f64 {
    let half_life = if memory_type == "episodic" {
        episodic_half_life_days(stability)
    } else {
        semantic_half_life_days_with(stability, stability_doubling)
    };
    0.5_f64.powf(elapsed_days.max(0.0) / half_life)
}

#[cfg(test)]
mod tests {
    use super::{
        EPISODIC_MAX_HALF_LIFE_DAYS, decay_factor, decay_factor_with, episodic_half_life_days,
        semantic_half_life_days,
    };

    #[test]
    fn semantic_curve_matches_spec_table() {
        let cases = [(0.15, 14.0), (0.30, 28.0), (0.50, 71.0), (1.0, 711.0)];
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
}
