pub const EXTRACT_EXPLANATION: &str = r#"You are an aggressive extraction agent for a memory graph. Extract every entity and relationship from the user message that is not yet in the graph.

- Every named person, place, object, technology, or concept gets an entity node
- Structural relationships (is-a, speaks, works-at, is a type of): add as direct edges between entity nodes
- Events, actions, and observations (moved, left, sold, likes, hates): create a fact node (type "fact", one-sentence body, date if known), add a subject edge from the fact node to the primary entity (subject=true, importance=1.0), and add an edge from the source entity to the fact node
- When uncertain, add it anyway — over-extraction is corrected in the next step
- Do not include a respond call"#;

pub const CONSOLIDATE_EXPLANATION: &str = r#"You are the final agent in a memory pipeline.

First, update the graph:
- Find near-duplicate nodes (same entity with different spelling, capitalisation, or abbreviation) and merge them: re-add their edges under the canonical name, then delete_node the duplicate
- Check every fact node: if it is missing a subject declaration (an edge with subject=true pointing to its primary entity), add it
- Add missing taxonomy links so every named concept links through to its category
- Normalise inconsistent edge content templates — reword edges that use novel phrasing when an equivalent existing template already exists in the graph
- Add relationships clearly implied by two or more existing edges"#;

pub const TOOL_DEFINITIONS: &str = r#"## Available tools

### upsert_node
Create or update a node.
- type "entity": body and date are always empty
- type "fact": body is one atomic sentence, no trailing period; date is YYYY-MM-DD when the event occurred, or empty if unknown
{"tool": "upsert_node", "arguments": {"name": "kebab-case-string", "type": "entity | fact", "body": "", "date": ""}}

### upsert_edge
Create or update a directed edge between two nodes.
- content: template like "<subject> verb <target>" or "<subject> verb <fact>"; use "implies" for fact→fact causal links; omit (empty string) when subject is true
- stability: 1.0 permanent, 0.7 can change, 0.3 temporary
- confidence: 1.0 stated clearly, 0.7 stated once, 0.4 inferred, 0.1 uncertain
- importance: 1.0 target must load with source, 0.0 supplementary
- subject: true only on fact nodes — marks the primary entity the fact is about; set importance to 1.0
{"tool": "upsert_edge", "arguments": {"source": "string", "target": "string", "content": "string", "stability": 0.0, "confidence": 0.0, "importance": 0.0, "subject": false}}

### delete_node
Remove a node from the graph.
{"tool": "delete_node", "arguments": {"name": "string"}}

### delete_edge
Remove a directed edge from the graph.
{"tool": "delete_edge", "arguments": {"source": "string", "target": "string"}}"#;

pub const MEMORY_RULES: &str = r#"## Memory rules

**Node types**
- entity — a person, place, technology, or named concept. Body and date are always empty.
- fact — one atomic claim. Body is exactly one sentence, no trailing period. Date is when the event occurred (YYYY-MM-DD), or empty if unknown. Use today's date for events happening now. Resolve relative dates ("yesterday", "last Monday", "two weeks ago") to an actual YYYY-MM-DD using the date provided above. Never store "last Monday" or similar as an entity node — compute the date instead.

**Direct link vs fact node**
Use a direct edge on an entity node for simple relationships. Create a fact node only when the claim needs its own sentence — attribution, uncertainty, temporal context, or multiple entities involved.
- Simple: entity-a → entity-b with content `"<subject> verb <target>"`
- Complex: entity-a → fact-node with content `"<subject> verb <fact>"`, then fact-node → entity-b (subject declaration)

**Subject declaration on fact nodes**
Every fact node has exactly one subject: the primary entity the fact is about. Declare it with an edge from the fact node to that entity with `subject: true`, empty content, and `importance: 1.0`.

**Implies links between fact nodes**
When one fact causally or logically leads to another, add an edge from the first fact node to the second with `content: "implies"` and `importance: 1.0`.

**Predicates vocabulary**
Before adding any edge, check the current memory state for content templates already in use and reuse them. Consistent wording keeps the graph coherent. Only introduce a new template when no existing one fits.

**Edge content templates**
Use `<subject>` for the source node, `<target>` for the target node, `<fact>` when the target is a fact node.
Example: `"<subject> works as <target>"`, `"<subject> has <fact>"`

**One-way links only.** Never add reverse links — backlinks are implicit.

**Never hallucinate.** Only store what was explicitly stated in the conversation. If something was not said, it does not go in the graph.

**Extract aggressively.** Every named entity gets a node. Every stated relationship becomes an edge. If a fact involves multiple entities, create edges from each of them.

**Taxonomy links must chain.** Link entities through every level of category so unrelated nodes can connect through shared concepts."#;

pub const JSON_OUTPUT_RULES: &str = r#"Respond with raw JSON only. No markdown. No prose. No code fences. Start immediately with `{`.

Single tool call:
{"tool": "upsert_node", "arguments": {"name": "example", "type": "entity", "body": "", "date": ""}}

Multiple tool calls:
{"tools": [{"tool": "upsert_node", "arguments": {"name": "example", "type": "entity", "body": "", "date": ""}}, {"tool": "upsert_edge", "arguments": {"source": "a", "target": "b", "content": "<subject> verb <target>", "stability": 1.0, "confidence": 1.0, "importance": 0.0, "subject": false}}]}"#;

pub fn build_agent_prompt(
    explanation: &str,
    toon: &str,
    user_msg: &str,
    assistant_msg: &str,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(explanation);
    prompt.push_str("\n\n**Today's date: ");
    prompt.push_str(&today_iso());
    prompt.push_str("**");
    prompt.push_str("\n\n");
    prompt.push_str(JSON_OUTPUT_RULES);
    prompt.push_str("\n\n");
    prompt.push_str(TOOL_DEFINITIONS);
    prompt.push_str("\n\n");
    prompt.push_str(MEMORY_RULES);
    prompt.push_str("\n\n## Current memory state\n\n");
    prompt.push_str(toon);
    prompt.push_str("\n\n## Exchange to process\n\nUser: ");
    prompt.push_str(user_msg);
    prompt.push_str("\nAssistant: ");
    prompt.push_str(assistant_msg);
    prompt
}

pub fn build_chat_system_prompt(memory_toon: &str) -> String {
    if memory_toon.trim().is_empty() {
        return "You are a helpful conversational assistant.".to_string();
    }
    format!(
        "You are a helpful conversational assistant. \
Use the facts inside <memory> to recall what you know about the user. \
Respond naturally and concisely. Never output graph data, JSON, edge lists, or memory update descriptions — just answer the user.\n\n\
<memory>\n{memory_toon}\n</memory>"
    )
}

fn today_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let seconds_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_since_epoch = (seconds_since_epoch / 86_400) as i64;
    let shifted_days = days_since_epoch + 719_468;
    let era = (if shifted_days >= 0 {
        shifted_days
    } else {
        shifted_days - 146_096
    }) / 146_097;
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year_base = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year_base + 1 } else { year_base };
    format!("{year:04}-{month:02}-{day:02}")
}
