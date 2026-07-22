pub const SEMANTIC_BASE_HALF_LIFE: f64 = 7.0; // replay-tunable.
pub const STABILITY_DOUBLING: f64 = 0.10; // replay-tunable.
pub const EPISODIC_HALF_LIFE_DAYS: f64 = 1.0; // replay-tunable.
pub const EPISODIC_MAX_HALF_LIFE_DAYS: f64 = 3.0; // replay-tunable.

#[allow(dead_code)] // Public curve probe retained for simulator and regression tests.
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
#[path = "decay_curve_tests.rs"]
mod tests;
