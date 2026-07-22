use std::collections::{HashMap, HashSet};

use crate::api::EmbeddingClient;

use super::embed::{SemanticEdge, nearest_edges, packed_edges_to_toon, semantic_edge};
use super::graph::{Edge, Graph, Node, node_carries_value, normalize_id};

pub(crate) const HEAVY_CONTEXT_BUDGET_TOKENS: usize = 16_000; // replay-tunable.
const CONTRADICTION_IMPORTANCE: f64 = 0.7; // replay-tunable.
const CONTRADICTION_CANDIDATES: usize = 20; // replay-tunable.

type EdgeKey = (String, String);

pub(crate) struct HeavySelection {
    pub(crate) toon: String,
    pub(crate) edge_count: usize,
    pub(crate) node_count: usize,
    pub(crate) estimated_tokens: usize,
    pub(crate) tier_fill: [usize; 4],
}

pub(crate) struct HeavyInputs<'scope> {
    pub(crate) graph: &'scope Graph,
    pub(crate) injected: &'scope [Edge],
    pub(crate) exchange: &'scope str,
    pub(crate) session_id: Option<&'scope str>,
    pub(crate) active_ids: &'scope [String],
    pub(crate) reserved_text: &'scope str,
}

pub(crate) fn select<'scope>(
    embed: &dyn EmbeddingClient,
    input: &HeavyInputs<'scope>,
) -> HeavySelection {
    let nearest = nearest_edges(
        embed,
        input.exchange,
        input.session_id,
        super::embed::TOP_K * 2,
    );
    select_with_nearest(input, &nearest, HEAVY_CONTEXT_BUDGET_TOKENS)
}

fn select_with_nearest<'scope>(
    input: &HeavyInputs<'scope>,
    nearest: &[SemanticEdge],
    budget: usize,
) -> HeavySelection {
    let live = sorted_live_edges(input.graph);
    let tiers = build_tiers(input, &live, nearest);
    pack_tiers(&tiers, &input.graph.nodes, input.reserved_text, budget)
}

fn build_tiers<'scope>(
    input: &HeavyInputs<'scope>,
    live: &[Edge],
    nearest: &[SemanticEdge],
) -> [Vec<Edge>; 4] {
    [
        mandatory_edges(input, live),
        lexical_edges(input.exchange, live),
        semantic_and_hop_edges(nearest, live),
        contradiction_edges(nearest, live),
    ]
}

fn mandatory_edges<'scope>(input: &HeavyInputs<'scope>, live: &[Edge]) -> Vec<Edge> {
    let injected = edge_key_set(input.injected);
    let mut active = HashSet::new();
    for active_id in input.active_ids {
        active.insert(active_id.as_str());
    }
    let mut edges = Vec::new();
    for edge in live {
        let key = edge_key(edge);
        let loop_edge = endpoint_has_type(input.graph, edge, "loop");
        let active_edge =
            active.contains(edge.source.as_str()) || active.contains(edge.target.as_str());
        if injected.contains(&key) || loop_edge || active_edge {
            edges.push(edge.clone());
        }
    }
    edges
}

fn lexical_edges(exchange: &str, live: &[Edge]) -> Vec<Edge> {
    let terms = exchange_terms(exchange);
    let mut edges = Vec::new();
    for edge in live {
        let searchable = normalize_id(&format!("{} {} {}", edge.source, edge.content, edge.target));
        if contains_term(&searchable, &terms) {
            edges.push(edge.clone());
        }
    }
    edges
}

fn contains_term(searchable: &str, terms: &[String]) -> bool {
    for term in terms {
        if searchable.contains(term) {
            return true;
        }
    }
    false
}

// Current-scale implementation: a deterministic in-memory contains pass.
// Upgrade point: move this tier to indexed SQL when live-edge scans become material.
fn exchange_terms(exchange: &str) -> Vec<String> {
    let mut terms = HashSet::new();
    for raw in exchange.split(|character: char| !character.is_alphanumeric()) {
        let term = normalize_id(raw);
        if term.chars().count() >= 3 {
            terms.insert(term);
        }
    }
    let mut sorted: Vec<String> = terms.into_iter().collect();
    sorted.sort();
    sorted
}

fn semantic_and_hop_edges(nearest: &[SemanticEdge], live: &[Edge]) -> Vec<Edge> {
    let lookup = edge_lookup(live);
    let mut edges = Vec::new();
    for anchor in nearest.iter().take(super::embed::TOP_K) {
        push_lookup_edge(&mut edges, &lookup, &semantic_key(anchor));
        for edge in live {
            if touches_anchor(edge, anchor) {
                edges.push(edge.clone());
            }
        }
    }
    deduplicate(edges)
}

fn contradiction_edges(nearest: &[SemanticEdge], live: &[Edge]) -> Vec<Edge> {
    let lookup = edge_lookup(live);
    let mut candidates = Vec::new();
    for edge in nearest.iter().skip(super::embed::TOP_K) {
        if edge.importance >= CONTRADICTION_IMPORTANCE
            && let Some(stored) = lookup.get(&semantic_key(edge))
        {
            candidates.push((*stored).clone());
        }
    }
    candidates.sort_by(|left, right| {
        right
            .importance
            .total_cmp(&left.importance)
            .then_with(|| edge_key(left).cmp(&edge_key(right)))
    });
    candidates.truncate(CONTRADICTION_CANDIDATES);
    candidates
}

fn pack_tiers(
    tiers: &[Vec<Edge>; 4],
    nodes: &HashMap<String, Node>,
    reserved_text: &str,
    budget: usize,
) -> HeavySelection {
    let empty_toon = packed_edges_to_toon(&[], nodes);
    let mut state = PackState {
        nodes,
        budget,
        used_tokens: estimate_tokens(&empty_toon) + estimate_tokens(reserved_text),
        packed: Vec::new(),
        selected: HashSet::new(),
        introduced_nodes: HashSet::new(),
    };
    let mut fills = [0usize; 4];
    for (tier_index, tier) in tiers.iter().enumerate() {
        fills[tier_index] = pack_tier(tier, tier_index, &mut state);
    }
    finish_selection(state.packed, nodes, reserved_text, fills, budget)
}

struct PackState<'scope> {
    nodes: &'scope HashMap<String, Node>,
    budget: usize,
    used_tokens: usize,
    packed: Vec<PackedItem>,
    selected: HashSet<EdgeKey>,
    introduced_nodes: HashSet<String>,
}

struct PackedItem {
    edge: SemanticEdge,
    tier_index: usize,
}

fn pack_tier<'scope>(tier: &[Edge], tier_index: usize, state: &mut PackState<'scope>) -> usize {
    let mut fill = 0usize;
    for edge in tier {
        let key = edge_key(edge);
        if state.selected.contains(&key) {
            continue;
        }
        let (unit_tokens, new_nodes) = edge_unit_cost(edge, state);
        if state.used_tokens + unit_tokens > state.budget {
            continue;
        }
        state.used_tokens += unit_tokens;
        state.selected.insert(key);
        for node_id in new_nodes {
            state.introduced_nodes.insert(node_id);
        }
        state.packed.push(PackedItem {
            edge: semantic_edge(edge),
            tier_index,
        });
        fill += 1;
    }
    fill
}

fn edge_unit_cost<'scope>(edge: &Edge, state: &PackState<'scope>) -> (usize, Vec<String>) {
    let mut node_ids = Vec::new();
    collect_new_node(&edge.source, state, &mut node_ids);
    collect_new_node(&edge.target, state, &mut node_ids);
    let mut characters = edge_value_characters(edge);
    for node_id in &node_ids {
        if let Some(node) = state.nodes.get(node_id) {
            characters += node_value_characters(node_id, node);
        }
    }
    (characters.div_ceil(4), node_ids)
}

fn collect_new_node<'scope>(node_id: &str, state: &PackState<'scope>, node_ids: &mut Vec<String>) {
    if state.introduced_nodes.contains(node_id) || node_id_seen(node_id, node_ids) {
        return;
    }
    let Some(node) = state.nodes.get(node_id) else {
        return;
    };
    if node_carries_value(node) {
        node_ids.push(node_id.to_string());
    }
}

fn node_id_seen(node_id: &str, node_ids: &[String]) -> bool {
    for seen in node_ids {
        if seen == node_id {
            return true;
        }
    }
    false
}

fn edge_value_characters(edge: &Edge) -> usize {
    edge.source.chars().count()
        + edge.target.chars().count()
        + edge.content.chars().count()
        + edge.date.chars().count()
        + edge.provenance.chars().count()
        + edge.occurred_at.as_deref().unwrap_or("").chars().count()
        + edge.expired_at.as_deref().unwrap_or("").chars().count()
        + 80
}

fn node_value_characters(node_id: &str, node: &Node) -> usize {
    node_id.chars().count()
        + node.node_type.chars().count()
        + node.body.chars().count()
        + node.date.chars().count()
        + 24
}

fn finish_selection(
    mut items: Vec<PackedItem>,
    nodes: &HashMap<String, Node>,
    reserved_text: &str,
    mut tier_fill: [usize; 4],
    budget: usize,
) -> HeavySelection {
    let mut edges = packed_edges(&items);
    let mut toon = packed_edges_to_toon(&edges, nodes);
    while estimate_tokens(&toon) + estimate_tokens(reserved_text) > budget {
        let Some(item) = items.pop() else {
            break;
        };
        tier_fill[item.tier_index] = tier_fill[item.tier_index].saturating_sub(1);
        edges = packed_edges(&items);
        toon = packed_edges_to_toon(&edges, nodes);
    }
    let node_count = selected_node_count(&edges, nodes);
    let estimated_tokens = estimate_tokens(&toon) + estimate_tokens(reserved_text);
    HeavySelection {
        toon,
        edge_count: edges.len(),
        node_count,
        estimated_tokens,
        tier_fill,
    }
}

fn packed_edges(items: &[PackedItem]) -> Vec<SemanticEdge> {
    let mut edges = Vec::new();
    for item in items {
        edges.push(item.edge.clone());
    }
    edges
}

pub(crate) fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn selected_node_count(edges: &[SemanticEdge], nodes: &HashMap<String, Node>) -> usize {
    let mut selected = HashSet::new();
    for edge in edges {
        count_value_node(&edge.source, nodes, &mut selected);
        count_value_node(&edge.target, nodes, &mut selected);
    }
    selected.len()
}

fn count_value_node(node_id: &str, nodes: &HashMap<String, Node>, selected: &mut HashSet<String>) {
    if nodes.get(node_id).is_some_and(node_carries_value) {
        selected.insert(node_id.to_string());
    }
}

fn sorted_live_edges(graph: &Graph) -> Vec<Edge> {
    let mut edges = Vec::new();
    for edge in &graph.edges {
        if edge.expired_at.is_none() {
            edges.push(edge.clone());
        }
    }
    edges.sort_by_key(edge_key);
    edges
}

fn endpoint_has_type(graph: &Graph, edge: &Edge, node_type: &str) -> bool {
    graph
        .nodes
        .get(&edge.source)
        .is_some_and(|node| node.node_type == node_type)
        || graph
            .nodes
            .get(&edge.target)
            .is_some_and(|node| node.node_type == node_type)
}

fn edge_key_set(edges: &[Edge]) -> HashSet<EdgeKey> {
    let mut keys = HashSet::new();
    for edge in edges {
        keys.insert(edge_key(edge));
    }
    keys
}

fn edge_lookup(edges: &[Edge]) -> HashMap<EdgeKey, &Edge> {
    let mut lookup = HashMap::new();
    for edge in edges {
        lookup.insert(edge_key(edge), edge);
    }
    lookup
}

fn push_lookup_edge(edges: &mut Vec<Edge>, lookup: &HashMap<EdgeKey, &Edge>, key: &EdgeKey) {
    if let Some(edge) = lookup.get(key) {
        edges.push((*edge).clone());
    }
}

fn touches_anchor(edge: &Edge, anchor: &SemanticEdge) -> bool {
    edge.source == anchor.source
        || edge.source == anchor.target
        || edge.target == anchor.source
        || edge.target == anchor.target
}

fn deduplicate(edges: Vec<Edge>) -> Vec<Edge> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for edge in edges {
        if seen.insert(edge_key(&edge)) {
            unique.push(edge);
        }
    }
    unique
}

fn edge_key(edge: &Edge) -> EdgeKey {
    (edge.source.clone(), edge.target.clone())
}

fn semantic_key(edge: &SemanticEdge) -> EdgeKey {
    (edge.source.clone(), edge.target.clone())
}

#[cfg(test)]
#[path = "heavy_tests.rs"]
mod tests;
