use super::dream::DOMAIN_AGENT_NAMES;
use super::ledger;

// Query dominates until interest tiers (phase 19+) make the user position
// reflect salience instead of bulk exposure — at ×1-only weighting the
// position is close to the graph centroid and blending it dilutes recall.
const W_QUERY: f64 = 0.8; // replay-tunable.
const W_USER_POS: f64 = 0.15; // replay-tunable.
const W_USER_DIR: f64 = 0.05; // replay-tunable.
const DIRECTION_WEIGHT: f64 = 0.3; // replay-tunable.

#[derive(Clone)]
pub struct Routing {
    pub routing_vector: Vec<f64>,
    pub expert: Option<String>,
    pub expert_score: f64,
    pub expert_with_direction: bool,
}

pub fn route(query_embedding: &[f64]) -> Routing {
    let user_position = filter_vector_dim(
        ledger::load_actor_position("user", "local"),
        query_embedding.len(),
    );
    let user_direction = filter_vector_dim(
        ledger::load_actor_direction("user", "local"),
        query_embedding.len(),
    );
    let routing_vector = blend_routing_vector(
        query_embedding,
        user_position.as_deref(),
        user_direction.as_deref(),
    );
    let (expert, score, with_direction) = choose_expert(&routing_vector);
    Routing {
        routing_vector,
        expert,
        expert_score: score,
        expert_with_direction: with_direction,
    }
}

fn filter_vector_dim(vector: Option<Vec<f64>>, dim: usize) -> Option<Vec<f64>> {
    let vector = vector?;
    if vector.len() != dim {
        return None;
    }
    Some(vector)
}

fn blend_routing_vector(
    query_embedding: &[f64],
    user_position: Option<&[f64]>,
    user_direction: Option<&[f64]>,
) -> Vec<f64> {
    let mut terms: Vec<(&[f64], f64)> = vec![(query_embedding, W_QUERY)];
    if let Some(position) = user_position {
        terms.push((position, W_USER_POS));
    }
    if let Some(direction) = user_direction {
        terms.push((direction, W_USER_DIR));
    }
    let mut weight_sum = 0.0;
    let mut term_index = 0usize;
    while term_index < terms.len() {
        weight_sum += terms[term_index].1;
        term_index += 1;
    }
    if weight_sum <= 0.0 {
        return query_embedding.to_vec();
    }
    let mut routing = vec![0.0; query_embedding.len()];
    let mut index = 0usize;
    while index < terms.len() {
        let (vector, weight) = terms[index];
        let normalized_weight = weight / weight_sum;
        let mut vector_index = 0usize;
        while vector_index < routing.len() {
            routing[vector_index] += vector[vector_index] * normalized_weight;
            vector_index += 1;
        }
        index += 1;
    }
    routing
}

fn choose_expert(routing_vector: &[f64]) -> (Option<String>, f64, bool) {
    let mut best_expert: Option<String> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_with_direction = false;
    for agent_name in DOMAIN_AGENT_NAMES {
        let Some(position) = ledger::load_actor_position("agent", agent_name) else {
            continue;
        };
        if position.len() != routing_vector.len() {
            continue;
        }
        let position_score = ledger::cosine_similarity(routing_vector, &position).unwrap_or(0.0);
        let direction = ledger::load_actor_direction("agent", agent_name);
        let (score, with_direction) =
            score_agent(routing_vector, position_score, direction.as_deref());
        if score > best_score {
            best_score = score;
            best_expert = Some(agent_name.to_string());
            best_with_direction = with_direction;
        }
    }
    if best_expert.is_none() {
        return (None, 0.0, false);
    }
    (best_expert, best_score, best_with_direction)
}

fn score_agent(
    routing_vector: &[f64],
    position_score: f64,
    direction: Option<&[f64]>,
) -> (f64, bool) {
    let Some(direction_vector) = direction else {
        return (position_score, false);
    };
    if direction_vector.len() != routing_vector.len() {
        return (position_score, false);
    }
    let direction_score =
        ledger::cosine_similarity(routing_vector, direction_vector).unwrap_or(0.0);
    (
        (1.0 - DIRECTION_WEIGHT) * position_score + DIRECTION_WEIGHT * direction_score,
        true,
    )
}
