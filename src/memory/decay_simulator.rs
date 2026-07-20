use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::path::Path;

use turso::{Builder, Connection, Value};

use super::decay_curve::decay_factor_with;
use super::embed::{SemanticEdge, TOP_K, replay_rank_score};
use super::graph::{Edge, PROVENANCE_LEGACY, PROVENANCE_USER_CONFIRMED, stability_seed_max};

const SWEEP_VALUES: [f64; 3] = [0.10, 0.15, 0.20]; // replay-tunable.
const HORIZONS: [usize; 4] = [1, 7, 30, 90]; // replay-tunable.
const QUERY_ANCHORS: usize = 20; // replay-tunable.
const CONFIDENCE_FLOOR: f64 = 0.05; // replay-tunable.
const USER_RETENTION_THRESHOLD: f64 = 0.10; // replay-tunable.
const JUNK_PREFIXES: [&str; 2] = ["rust-ownership-immune-response", "immune-response"]; // replay-tunable.

type EdgeKey = (String, String);
type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Clone)]
struct SimulationEdge {
    edge: Edge,
    embedding: Option<Vec<f64>>,
    source_node_type: String,
    target_node_type: String,
}

#[derive(Clone)]
struct Anchor {
    key: EdgeKey,
    embedding: Vec<f64>,
}

struct Snapshot {
    edges: Vec<SimulationEdge>,
    injections: Vec<(EdgeKey, i64)>,
    today: i64,
    today_text: String,
}

#[derive(Clone, Copy)]
enum Scenario {
    Untouched,
    ReplayedRefresh,
}

impl Scenario {
    fn label(self) -> &'static str {
        match self {
            Self::Untouched => "untouched",
            Self::ReplayedRefresh => "replayed refresh",
        }
    }
}

struct Churn {
    anchors: usize,
    jaccard: f64,
    displacement: f64,
}

#[derive(Clone, Copy)]
enum Cohort {
    Junk,
    UserConfirmed,
}

impl Cohort {
    fn label(self) -> &'static str {
        match self {
            Self::Junk => "rust/immune",
            Self::UserConfirmed => "user-confirmed",
        }
    }

    fn contains(self, edge: &SimulationEdge) -> bool {
        match self {
            Self::Junk => is_junk(edge),
            Self::UserConfirmed => is_user_confirmed(edge),
        }
    }
}

pub fn run(path: &Path, today_override: Option<&str>) -> Result<(), Box<dyn Error>> {
    match run_to_string(path, today_override) {
        Ok(report) => {
            print!("{report}");
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn run_to_string(path: &Path, today_override: Option<&str>) -> Result<String, BoxError> {
    if !path.is_file() {
        return Err(format!("snapshot does not exist: {}", path.display()).into());
    }
    let runtime = tokio::runtime::Runtime::new()?;
    let snapshot = runtime.block_on(load_snapshot(path, today_override))?;
    Ok(build_report(&snapshot))
}

async fn load_snapshot(path: &Path, today_override: Option<&str>) -> Result<Snapshot, BoxError> {
    let path_text = path
        .to_str()
        .ok_or_else(|| format!("snapshot path is not UTF-8: {}", path.display()))?;
    let database = Builder::new_local(path_text)
        .experimental_index_method(true)
        .build()
        .await?;
    let conn = database.connect()?;
    conn.execute("PRAGMA query_only=1", ()).await?;
    verify_query_only(&conn).await?;
    let edges = load_edges(&conn).await?;
    let raw_injections = load_injections(&conn).await?;
    let newest_event = load_newest_event_timestamp(&conn).await?;
    let (today, today_text) = snapshot_today(
        &edges,
        &raw_injections,
        newest_event.as_deref(),
        today_override,
    )?;
    let injections = dated_injections(raw_injections);
    Ok(Snapshot {
        edges,
        injections,
        today,
        today_text,
    })
}

async fn verify_query_only(conn: &Connection) -> Result<(), BoxError> {
    let mut rows = conn.query("PRAGMA query_only", ()).await?;
    let Some(row) = rows.next().await? else {
        return Err("snapshot did not report PRAGMA query_only".into());
    };
    if integer(row.get_value(0)?) != 1 {
        return Err("snapshot connection is not read-only".into());
    }
    Ok(())
}

async fn load_edges(conn: &Connection) -> Result<Vec<SimulationEdge>, BoxError> {
    let columns = table_columns(conn, "edges").await?;
    let query = edge_select_query(&columns);
    let mut rows = conn.query(&query, ()).await?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next().await? {
        if let Some(edge) = parse_simulation_edge(&row) {
            edges.push(edge);
        }
    }
    Ok(edges)
}

async fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, BoxError> {
    let mut rows = conn
        .query(&format!("PRAGMA table_info({table})"), ())
        .await?;
    let mut columns = HashSet::new();
    while let Some(row) = rows.next().await? {
        columns.insert(text(row.get_value(1)?));
    }
    Ok(columns)
}

fn edge_select_query(columns: &HashSet<String>) -> String {
    let fields = [
        edge_column(columns, "source", "''"),
        edge_column(columns, "target", "''"),
        edge_column(columns, "content", "''"),
        edge_column(columns, "stability", "0.0"),
        edge_column(columns, "confidence", "0.0"),
        edge_column(columns, "importance", "0.0"),
        edge_column(columns, "subject", "0"),
        edge_column(columns, "date", "''"),
        edge_column(columns, "occurred_at", "NULL"),
        edge_column(columns, "expired_at", "NULL"),
        edge_column(columns, "memory_type", "'semantic'"),
        edge_column(columns, "origin_session", "''"),
        edge_column(columns, "provenance", "'legacy'"),
        edge_column(columns, "ceiling", "NULL"),
        edge_column(columns, "last_decayed_at", "''"),
        embedding_column(columns),
    ];
    format!(
        "SELECT {}, COALESCE(source_node.node_type, ''), COALESCE(target_node.node_type, '') FROM edges e LEFT JOIN nodes source_node ON source_node.id = e.source LEFT JOIN nodes target_node ON target_node.id = e.target",
        fields.join(", ")
    )
}

fn edge_column(columns: &HashSet<String>, name: &str, fallback: &str) -> String {
    if columns.contains(name) {
        return format!("e.{name}");
    }
    fallback.to_string()
}

fn embedding_column(columns: &HashSet<String>) -> String {
    if columns.contains("embedding") {
        return "CASE WHEN e.embedding IS NULL THEN NULL ELSE vector_extract(e.embedding) END"
            .to_string();
    }
    "NULL".to_string()
}

fn parse_simulation_edge(row: &turso::Row) -> Option<SimulationEdge> {
    let mut edge = super::db::parse_edge_row(row)?;
    if edge.provenance.trim().is_empty() {
        edge.provenance = PROVENANCE_LEGACY.to_string();
    }
    Some(SimulationEdge {
        edge,
        embedding: parse_embedding(row.get_value(15).ok()),
        source_node_type: text(row.get_value(16).ok()?),
        target_node_type: text(row.get_value(17).ok()?),
    })
}

fn parse_embedding(value: Option<Value>) -> Option<Vec<f64>> {
    let Value::Text(json) = value? else {
        return None;
    };
    super::db::parse_vector32_json(&json).ok()
}

async fn load_injections(conn: &Connection) -> Result<Vec<(EdgeKey, String)>, BoxError> {
    let columns = table_columns(conn, "events").await?;
    if !has_injection_columns(&columns) {
        return Ok(Vec::new());
    }
    let mut rows = conn
        .query(
            "SELECT source, target, ts FROM events WHERE event_type = 'injected' AND source != '' AND target != '' ORDER BY ts DESC, rowid DESC",
            (),
        )
        .await?;
    let mut events = Vec::new();
    while let Some(row) = rows.next().await? {
        let key = (text(row.get_value(0)?), text(row.get_value(1)?));
        events.push((key, text(row.get_value(2)?)));
    }
    Ok(events)
}

async fn load_newest_event_timestamp(conn: &Connection) -> Result<Option<String>, BoxError> {
    let columns = table_columns(conn, "events").await?;
    if !columns.contains("ts") {
        return Ok(None);
    }
    let mut rows = conn.query("SELECT MAX(ts) FROM events", ()).await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let timestamp = text(row.get_value(0)?);
    if timestamp.is_empty() {
        return Ok(None);
    }
    Ok(Some(timestamp))
}

fn has_injection_columns(columns: &HashSet<String>) -> bool {
    for column in ["source", "target", "ts", "event_type"] {
        if !columns.contains(column) {
            return false;
        }
    }
    true
}

fn snapshot_today(
    edges: &[SimulationEdge],
    injections: &[(EdgeKey, String)],
    newest_event: Option<&str>,
    today_override: Option<&str>,
) -> Result<(i64, String), BoxError> {
    if let Some(raw) = today_override {
        return parsed_day(raw);
    }
    let mut newest: Option<(i64, String)> = None;
    for edge in edges {
        consider_date(&mut newest, &edge.edge.date);
        consider_date(&mut newest, &edge.edge.last_decayed_at);
        if let Some(occurred_at) = &edge.edge.occurred_at {
            consider_date(&mut newest, occurred_at);
        }
        if let Some(expired_at) = &edge.edge.expired_at {
            consider_date(&mut newest, expired_at);
        }
    }
    for (_, timestamp) in injections {
        consider_date(&mut newest, timestamp);
    }
    if let Some(timestamp) = newest_event {
        consider_date(&mut newest, timestamp);
    }
    newest.ok_or_else(|| "snapshot has no valid edge or injected-event timestamp".into())
}

fn parsed_day(raw: &str) -> Result<(i64, String), BoxError> {
    let date = date_part(raw).ok_or_else(|| format!("invalid simulation date: {raw}"))?;
    let day = date_to_days(&date).ok_or_else(|| format!("invalid simulation date: {raw}"))?;
    Ok((day, date))
}

fn consider_date(newest: &mut Option<(i64, String)>, raw: &str) {
    let Some(date) = date_part(raw) else {
        return;
    };
    let Some(day) = date_to_days(&date) else {
        return;
    };
    if newest.as_ref().is_none_or(|(known, _)| day > *known) {
        *newest = Some((day, date));
    }
}

fn dated_injections(events: Vec<(EdgeKey, String)>) -> Vec<(EdgeKey, i64)> {
    let mut dated = Vec::new();
    for (key, timestamp) in events {
        let Some(date) = date_part(&timestamp) else {
            continue;
        };
        let Some(day) = date_to_days(&date) else {
            continue;
        };
        dated.push((key, day));
    }
    dated
}

fn build_report(snapshot: &Snapshot) -> String {
    let anchors = query_anchors(snapshot);
    let mut output = report_header(snapshot, &anchors);
    for stability_doubling in SWEEP_VALUES {
        output.push_str(&format!(
            "\n== STABILITY_DOUBLING {stability_doubling:.2} ==\n"
        ));
        for scenario in [Scenario::Untouched, Scenario::ReplayedRefresh] {
            append_scenario(
                &mut output,
                snapshot,
                &anchors,
                stability_doubling,
                scenario,
            );
        }
    }
    output
}

fn report_header(snapshot: &Snapshot, anchors: &[Anchor]) -> String {
    let mut output = format!(
        "decay simulation\nsnapshot today: {}\nread-only: PRAGMA query_only=1\nreplayed refresh: trailing 90 ledger days translated to simulated nights 1..90\n",
        snapshot.today_text
    );
    output.push_str("query anchors (newest distinct injected edges with embeddings):\n");
    for anchor in anchors {
        output.push_str(&format!("  {} -> {}\n", anchor.key.0, anchor.key.1));
    }
    output.push_str("rust/immune-response cluster node ids:\n");
    for node_id in junk_node_ids(&snapshot.edges) {
        output.push_str(&format!("  {node_id}\n"));
    }
    output
}

fn append_scenario(
    output: &mut String,
    snapshot: &Snapshot,
    anchors: &[Anchor],
    stability_doubling: f64,
    scenario: Scenario,
) {
    let states = simulate(snapshot, stability_doubling, scenario);
    output.push_str(&format!("\n-- scenario: {} --\n", scenario.label()));
    append_trajectory_table(output, &states);
    append_prune_table(output, &states);
    append_count_table(output, &states);
    append_churn_table(output, snapshot, anchors, &states);
    append_acceptance(output, snapshot, anchors, &states);
}

fn simulate(
    snapshot: &Snapshot,
    stability_doubling: f64,
    scenario: Scenario,
) -> Vec<(usize, Vec<SimulationEdge>)> {
    let mut edges = clamped_edges(&snapshot.edges);
    let schedule = replay_schedule(snapshot, scenario);
    let mut states = Vec::new();
    let mut night = 1usize;
    while night <= 90 {
        let current_day = snapshot.today + night as i64;
        charge_night(
            &mut edges,
            stability_doubling,
            schedule.get(&night),
            current_day,
        );
        if HORIZONS.contains(&night) {
            states.push((night, edges.clone()));
        }
        night += 1;
    }
    states
}

fn clamped_edges(edges: &[SimulationEdge]) -> Vec<SimulationEdge> {
    let mut clamped = edges.to_vec();
    for edge in &mut clamped {
        edge.edge.stability = edge
            .edge
            .stability
            .min(stability_seed_max(&edge.edge.provenance));
    }
    clamped
}

fn replay_schedule(snapshot: &Snapshot, scenario: Scenario) -> HashMap<usize, HashSet<EdgeKey>> {
    let mut schedule: HashMap<usize, HashSet<EdgeKey>> = HashMap::new();
    if matches!(scenario, Scenario::Untouched) {
        return schedule;
    }
    let window_start = snapshot.today - 89;
    for (key, event_day) in &snapshot.injections {
        if *event_day < window_start || *event_day > snapshot.today {
            continue;
        }
        let night = (*event_day - window_start + 1) as usize;
        schedule.entry(night).or_default().insert(key.clone());
    }
    schedule
}

fn charge_night(
    edges: &mut [SimulationEdge],
    stability_doubling: f64,
    refreshed: Option<&HashSet<EdgeKey>>,
    current_day: i64,
) {
    for edge in edges {
        if edge.edge.expired_at.is_some() {
            continue;
        }
        let key = edge_key(edge);
        if refreshed.is_some_and(|keys| keys.contains(&key)) {
            let date = days_to_date(current_day);
            edge.edge.date.clone_from(&date);
            edge.edge.last_decayed_at = date;
            continue;
        }
        let factor = decay_factor_with(
            &edge.edge.memory_type,
            edge.edge.stability,
            1.0,
            stability_doubling,
        );
        edge.edge.confidence = (edge.edge.confidence * factor).clamp(CONFIDENCE_FLOOR, 1.0);
    }
}

fn append_trajectory_table(output: &mut String, states: &[(usize, Vec<SimulationEdge>)]) {
    output
        .push_str("confidence trajectory\nnight  type      cohort          n    q1    median q3\n");
    for (night, edges) in states {
        for memory_type in ["semantic", "episodic"] {
            append_quartile_row(output, *night, memory_type, Cohort::Junk, edges);
            append_quartile_row(output, *night, memory_type, Cohort::UserConfirmed, edges);
        }
    }
}

fn append_quartile_row(
    output: &mut String,
    night: usize,
    memory_type: &str,
    cohort: Cohort,
    edges: &[SimulationEdge],
) {
    let values = cohort_confidences(edges, memory_type, cohort);
    if let Some((first, median, third)) = quartiles(&values) {
        output.push_str(&format!(
            "{night:>5}  {memory_type:<9} {cohort:<15} {:>4}  {first:.3} {median:.3}  {third:.3}\n",
            values.len(),
            cohort = cohort.label()
        ));
    } else {
        output.push_str(&format!(
            "{night:>5}  {memory_type:<9} {cohort:<15}    0  n/a   n/a    n/a\n",
            cohort = cohort.label()
        ));
    }
}

fn cohort_confidences(edges: &[SimulationEdge], memory_type: &str, cohort: Cohort) -> Vec<f64> {
    let mut values = Vec::new();
    for edge in edges {
        if is_live_type(edge, memory_type) && cohort.contains(edge) {
            values.push(edge.edge.confidence);
        }
    }
    values
}

fn append_prune_table(output: &mut String, states: &[(usize, Vec<SimulationEdge>)]) {
    output.push_str("prune-eligible counts by provenance\nnight  type      provenance        today replacement\n");
    for (night, edges) in states {
        for memory_type in ["semantic", "episodic"] {
            let counts = prune_counts(edges, memory_type);
            for (provenance, (today, replacement)) in counts {
                output.push_str(&format!(
                    "{night:>5}  {memory_type:<9} {provenance:<17} {today:>5} {replacement:>11}\n"
                ));
            }
        }
    }
}

fn prune_counts(edges: &[SimulationEdge], memory_type: &str) -> BTreeMap<String, (usize, usize)> {
    let mut counts = BTreeMap::new();
    for edge in edges {
        if !is_live_type(edge, memory_type) {
            continue;
        }
        let entry = counts
            .entry(edge.edge.provenance.clone())
            .or_insert((0usize, 0usize));
        if today_prune_eligible(edge) {
            entry.0 += 1;
        }
        if replacement_prune_eligible(edge) {
            entry.1 += 1;
        }
    }
    counts
}

fn today_prune_eligible(edge: &SimulationEdge) -> bool {
    super::dream::prune_block_reason(&edge.edge, 0).is_none()
}

fn replacement_prune_eligible(edge: &SimulationEdge) -> bool {
    !edge_is_pinned(edge)
        && !edge.edge.subject
        && edge.edge.confidence < super::dream::PRUNE_GUARD_THRESHOLD
}

fn edge_is_pinned(edge: &SimulationEdge) -> bool {
    edge.edge.expired_at.is_none()
        && edge.edge.memory_type == "semantic"
        && edge.source_node_type == "loop"
        && edge.edge.content.starts_with("open loop:")
}

fn append_count_table(output: &mut String, states: &[(usize, Vec<SimulationEdge>)]) {
    output.push_str("confidence bands\nnight  type      at-floor above-0.5\n");
    for (night, edges) in states {
        for memory_type in ["semantic", "episodic"] {
            let (floor, above) = confidence_counts(edges, memory_type);
            output.push_str(&format!(
                "{night:>5}  {memory_type:<9} {floor:>8} {above:>9}\n"
            ));
        }
    }
}

fn confidence_counts(edges: &[SimulationEdge], memory_type: &str) -> (usize, usize) {
    let mut floor = 0usize;
    let mut above = 0usize;
    for edge in edges {
        if !is_live_type(edge, memory_type) {
            continue;
        }
        if edge.edge.confidence <= CONFIDENCE_FLOOR + f64::EPSILON {
            floor += 1;
        }
        if edge.edge.confidence > 0.5 {
            above += 1;
        }
    }
    (floor, above)
}

fn append_churn_table(
    output: &mut String,
    snapshot: &Snapshot,
    anchors: &[Anchor],
    states: &[(usize, Vec<SimulationEdge>)],
) {
    output.push_str("retrieval churn (top-k vs snapshot confidence)\nnight  type      anchors jaccard mean-rank-displacement\n");
    for (night, edges) in states {
        let reference_date = days_to_date(snapshot.today + *night as i64);
        for memory_type in ["semantic", "episodic"] {
            let churn = retrieval_churn(
                &snapshot.edges,
                edges,
                anchors,
                memory_type,
                (&snapshot.today_text, &reference_date),
            );
            output.push_str(&format!(
                "{night:>5}  {memory_type:<9} {:>7} {:.3} {:>22.3}\n",
                churn.anchors, churn.jaccard, churn.displacement
            ));
        }
    }
}

fn retrieval_churn(
    baseline: &[SimulationEdge],
    simulated: &[SimulationEdge],
    anchors: &[Anchor],
    memory_type: &str,
    dates: (&str, &str),
) -> Churn {
    let mut jaccard_sum = 0.0;
    let mut displacement_sum = 0.0;
    let mut used = 0usize;
    for anchor in anchors {
        let before = ranked_keys(baseline, &anchor.embedding, memory_type, dates.0);
        let after = ranked_keys(simulated, &anchor.embedding, memory_type, dates.1);
        if before.is_empty() && after.is_empty() {
            continue;
        }
        let (jaccard, displacement) = compare_rankings(&before, &after);
        jaccard_sum += jaccard;
        displacement_sum += displacement;
        used += 1;
    }
    Churn {
        anchors: used,
        jaccard: safe_mean(jaccard_sum, used),
        displacement: safe_mean(displacement_sum, used),
    }
}

fn ranked_keys(
    edges: &[SimulationEdge],
    query: &[f64],
    memory_type: &str,
    reference_date: &str,
) -> Vec<EdgeKey> {
    let mut scores = Vec::new();
    for edge in edges {
        if !retrieval_candidate(edge, memory_type) {
            continue;
        }
        let semantic = semantic_edge(edge);
        let score = replay_rank_score(&semantic, query, reference_date);
        scores.push((score, edge_key(edge)));
    }
    scores.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    scores.truncate(TOP_K);
    let mut keys = Vec::new();
    for (_, key) in scores {
        keys.push(key);
    }
    keys
}

fn retrieval_candidate(edge: &SimulationEdge, memory_type: &str) -> bool {
    is_live_type(edge, memory_type)
        && edge.embedding.is_some()
        && !matches!(edge.source_node_type.as_str(), "loop" | "interest")
        && !matches!(edge.target_node_type.as_str(), "loop" | "interest")
}

fn semantic_edge(edge: &SimulationEdge) -> SemanticEdge {
    SemanticEdge {
        source: edge.edge.source.clone(),
        target: edge.edge.target.clone(),
        content: edge.edge.content.clone(),
        confidence: edge.edge.confidence,
        stability: edge.edge.stability,
        importance: edge.edge.importance,
        subject: edge.edge.subject,
        date: edge.edge.date.clone(),
        occurred_at: edge.edge.occurred_at.clone(),
        expired_at: edge.edge.expired_at.clone(),
        memory_type: edge.edge.memory_type.clone(),
        origin_session: edge.edge.origin_session.clone(),
        provenance: edge.edge.provenance.clone(),
        embedding: edge.embedding.clone(),
    }
}

fn compare_rankings(before: &[EdgeKey], after: &[EdgeKey]) -> (f64, f64) {
    let before_set: HashSet<&EdgeKey> = before.iter().collect();
    let after_set: HashSet<&EdgeKey> = after.iter().collect();
    let union = before_set.union(&after_set).count();
    let intersection = before_set.intersection(&after_set).count();
    let jaccard = if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    };
    (jaccard, mean_rank_displacement(before, after))
}

fn mean_rank_displacement(before: &[EdgeKey], after: &[EdgeKey]) -> f64 {
    let mut positions: HashMap<&EdgeKey, usize> = HashMap::new();
    for (index, key) in after.iter().enumerate() {
        positions.insert(key, index);
    }
    let mut total = 0usize;
    let mut common = 0usize;
    for (before_index, key) in before.iter().enumerate() {
        if let Some(after_index) = positions.get(key) {
            total += before_index.abs_diff(*after_index);
            common += 1;
        }
    }
    safe_mean(total as f64, common)
}

fn append_acceptance(
    output: &mut String,
    snapshot: &Snapshot,
    anchors: &[Anchor],
    states: &[(usize, Vec<SimulationEdge>)],
) {
    let junk_pass = state_at(states, 30).is_some_and(junk_is_prune_eligible);
    let user_pass = state_at(states, 90).is_some_and(user_retained);
    let churn_reported = !anchors.is_empty() && !snapshot.edges.is_empty();
    output.push_str("acceptance\n");
    output.push_str(&format!(
        "  junk prune-eligible at 30 nights: {}\n",
        pass_fail(junk_pass)
    ));
    output.push_str(&format!(
        "  user-confirmed q1 > {USER_RETENTION_THRESHOLD:.2} at 90 nights: {}\n",
        pass_fail(user_pass)
    ));
    output.push_str(&format!(
        "  retrieval churn reported (no threshold chosen): {}\n",
        pass_fail(churn_reported)
    ));
}

fn user_retained(edges: &[SimulationEdge]) -> bool {
    let values = cohort_confidences(edges, "semantic", Cohort::UserConfirmed);
    quartiles(&values).is_some_and(|(first, _, _)| first > USER_RETENTION_THRESHOLD)
}

fn junk_is_prune_eligible(edges: &[SimulationEdge]) -> bool {
    for edge in edges {
        if is_junk(edge) && replacement_prune_eligible(edge) {
            return true;
        }
    }
    false
}

fn state_at(states: &[(usize, Vec<SimulationEdge>)], horizon: usize) -> Option<&[SimulationEdge]> {
    for (night, edges) in states {
        if *night == horizon {
            return Some(edges);
        }
    }
    None
}

fn query_anchors(snapshot: &Snapshot) -> Vec<Anchor> {
    let mut edge_map: HashMap<EdgeKey, &SimulationEdge> = HashMap::new();
    for edge in &snapshot.edges {
        edge_map.insert(edge_key(edge), edge);
    }
    let mut seen = HashSet::new();
    let mut anchors = Vec::new();
    for (key, _) in &snapshot.injections {
        if !seen.insert(key.clone()) {
            continue;
        }
        let Some(edge) = edge_map.get(key) else {
            continue;
        };
        let Some(embedding) = &edge.embedding else {
            continue;
        };
        anchors.push(Anchor {
            key: key.clone(),
            embedding: embedding.clone(),
        });
        if anchors.len() == QUERY_ANCHORS {
            break;
        }
    }
    anchors
}

fn junk_node_ids(edges: &[SimulationEdge]) -> Vec<String> {
    let mut ids = HashSet::new();
    for edge in edges {
        for node_id in [&edge.edge.source, &edge.edge.target] {
            if node_has_junk_prefix(node_id) {
                ids.insert(node_id.clone());
            }
        }
    }
    let mut sorted: Vec<String> = ids.into_iter().collect();
    sorted.sort();
    sorted
}

fn is_junk(edge: &SimulationEdge) -> bool {
    node_has_junk_prefix(&edge.edge.source) || node_has_junk_prefix(&edge.edge.target)
}

fn node_has_junk_prefix(node_id: &str) -> bool {
    for prefix in JUNK_PREFIXES {
        if node_id.starts_with(prefix) {
            return true;
        }
    }
    false
}

fn is_user_confirmed(edge: &SimulationEdge) -> bool {
    edge.edge.provenance == PROVENANCE_USER_CONFIRMED
}

fn is_live_type(edge: &SimulationEdge, memory_type: &str) -> bool {
    edge.edge.expired_at.is_none() && edge.edge.memory_type == memory_type
}

fn edge_key(edge: &SimulationEdge) -> EdgeKey {
    (edge.edge.source.clone(), edge.edge.target.clone())
}

fn quartiles(values: &[f64]) -> Option<(f64, f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some((
        percentile(&sorted, 0.25),
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.75),
    ))
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn safe_mean(total: f64, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    total / count as f64
}

fn pass_fail(passed: bool) -> &'static str {
    if passed { "PASS" } else { "FAIL" }
}

fn text(value: Value) -> String {
    match value {
        Value::Text(value) => value,
        _ => String::new(),
    }
}

fn integer(value: Value) -> i64 {
    match value {
        Value::Integer(value) => value,
        _ => 0,
    }
}

fn date_part(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.len() < 10 {
        return None;
    }
    let date = &trimmed[..10];
    date_to_days(date)?;
    Some(date.to_string())
}

fn date_to_days(date: &str) -> Option<i64> {
    let mut parts = date.splitn(3, '-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    let (adjusted_year, adjusted_month) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

fn days_to_date(days: i64) -> String {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use turso::Builder;

    use super::{SimulationEdge, date_to_days, days_to_date, run_to_string, today_prune_eligible};
    use crate::memory::graph::Edge;

    #[test]
    fn clamp_uses_legacy_seed_maximum() {
        let edge = test_simulation_edge(1.0);
        let clamped = super::clamped_edges(&[edge]);
        assert_eq!(clamped[0].edge.stability, 0.3);
    }

    #[test]
    fn today_prune_count_matches_current_guard() {
        let mut edge = test_simulation_edge(0.6);
        edge.edge.importance = 0.5;
        edge.edge.confidence = 0.6;
        assert!(today_prune_eligible(&edge));
        edge.edge.stability = 0.7;
        assert!(!today_prune_eligible(&edge));
        edge.edge.stability = 0.6;
        edge.edge.confidence = 0.7;
        assert!(!today_prune_eligible(&edge));
        edge.edge.confidence = 0.6;
        edge.edge.importance = 1.0;
        assert!(!today_prune_eligible(&edge));
    }

    #[test]
    fn date_round_trip_is_stable() {
        for date in ["2025-01-01", "2026-07-20", "2030-12-31"] {
            let Some(days) = date_to_days(date) else {
                panic!("valid date");
            };
            assert_eq!(days_to_date(days), date);
        }
    }

    #[test]
    fn simulator_is_read_only_and_produces_report() {
        let path = fixture_path();
        create_fixture(&path);
        let before = std::fs::read(&path).unwrap_or_default();
        let report = run_to_string(&path, None).unwrap_or_default();
        let after = std::fs::read(&path).unwrap_or_default();
        assert_eq!(before, after, "snapshot bytes changed");
        assert!(report.contains("STABILITY_DOUBLING 0.15"));
        assert!(report.contains("prune-eligible"));
        assert!(report.contains("rust-fact -> immune-response-node"));
        cleanup_fixture(&path);
    }

    fn fixture_path() -> PathBuf {
        std::env::temp_dir().join(format!("kratt-decay-simulator-{}.db", std::process::id()))
    }

    fn create_fixture(path: &std::path::Path) {
        cleanup_fixture(path);
        let Ok(runtime) = tokio::runtime::Runtime::new() else {
            panic!("runtime");
        };
        runtime.block_on(async {
            let Some(path_text) = path.to_str() else {
                panic!("path");
            };
            let Ok(database) = Builder::new_local(path_text)
                .experimental_index_method(true)
                .build()
                .await
            else {
                panic!("database");
            };
            let Ok(conn) = database.connect() else {
                panic!("connection");
            };
            create_fixture_schema(&conn).await;
            insert_fixture_rows(&conn).await;
        });
    }

    async fn create_fixture_schema(conn: &turso::Connection) {
        conn.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, last_decayed_at TEXT NOT NULL DEFAULT '', embedding BLOB, PRIMARY KEY (source, target))",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '')",
            (),
        )
        .await
        .unwrap_or(0);
    }

    async fn insert_fixture_rows(conn: &turso::Connection) {
        conn.execute(
            "INSERT INTO nodes (id, node_type) VALUES ('rust-fact', 'fact'), ('immune-response-node', 'entity')",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "INSERT INTO edges (source, target, content, stability, confidence, importance, date, last_decayed_at, embedding) VALUES ('rust-fact', 'immune-response-node', 'bad analogy', 1.0, 0.6, 0.4, '2026-07-20', '2026-07-20', vector32('[1.0,0.0]'))",
            (),
        )
        .await
        .unwrap_or(0);
        conn.execute(
            "INSERT INTO events (ts, event_type, source, target) VALUES ('2026-07-20T12:00:00', 'injected', 'rust-fact', 'immune-response-node')",
            (),
        )
        .await
        .unwrap_or(0);
    }

    fn cleanup_fixture(path: &std::path::Path) {
        for suffix in ["", "-wal", "-shm"] {
            let candidate = PathBuf::from(format!("{}{suffix}", path.display()));
            let _ = std::fs::remove_file(candidate);
        }
    }

    fn test_simulation_edge(stability: f64) -> SimulationEdge {
        SimulationEdge {
            edge: Edge {
                source: "rust-fact".to_string(),
                target: "immune-response-node".to_string(),
                content: "bad analogy".to_string(),
                stability,
                confidence: 0.6,
                importance: 0.4,
                date: "2026-07-20".to_string(),
                subject: false,
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
                ceiling: None,
                last_decayed_at: "2026-07-20".to_string(),
            },
            embedding: Some(vec![1.0, 0.0]),
            source_node_type: "fact".to_string(),
            target_node_type: "entity".to_string(),
        }
    }
}
