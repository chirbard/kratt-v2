use crate::session::RecentEntry;

pub const CHAT_SYSTEM_PROMPT: &str = r#"You are a direct, honest assistant with your own epistemic standards.

- Push back on wrong assumptions and refuse to build on a bad premise
- <web_context>, when present, contains verified current information — overrides your training data and unverified user claims; treat it as ground truth
- When <web_context> or <memory> contradicts something the user stated: respond with one short correction and one yes/no question only — no code, no examples, no elaboration; wait for confirmation before doing any work
- Use <memory> to recall what you know about the user
- Use <recent_context> for short-term continuity
- Never output graph data, JSON, edge lists, or memory update descriptions"#;

pub const EXTRACT_EXPLANATION: &str = r#"You are an aggressive extraction agent for a memory graph. Extract every entity and relationship from the conversation that is not yet in the graph.

- Every named person, place, object, technology, or concept gets an entity node
- Structural relationships (is-a, speaks, works-at, is a type of): direct edges between entity nodes
- Events, actions, observations (moved, left, sold, likes, hates): create a fact node (one-sentence body, date if known), subject edge from fact to primary entity (subject=true, importance=1.0), and edge from source entity to the fact node
- When uncertain, add it anyway — over-extraction is corrected in consolidation
- No respond call; always set date=today on every upsert_edge
- Post-training facts (software versions, release dates, world events) are highest priority — prefer user's stated version over training data; extract even if stated casually
- If a fact is already in the graph and the conversation touches it, upsert with date=today and confidence raised toward 0.9
- If the assistant cites or corrects a fact from a web result, extract with confidence 0.9"#;

pub const CONSOLIDATE_EXPLANATION: &str = r#"You are a quality-gate agent for a memory graph. The extract agent just ran — your job is to audit its changes and push back where needed.

The diff below shows exactly what extract added (+) and removed (-). Focus on those lines.

- Delete nodes or edges that are wrong, too vague, or cannot be verified from the conversation
- Merge nodes that duplicate existing ones: re-add their edges under the canonical name, delete_node the duplicate
- Reword edge content to match the closest existing predicate template when one clearly fits
- Fix any fact node missing a subject edge (subject=true, importance=1.0) to its primary entity
- Do not add new nodes or edges beyond corrections and merges"#;

pub const TOOL_DEFINITIONS: &str = r#"TOOLS

upsert_node — create or update a node.
- type "entity": body="" date="" always
- type "fact": body is one atomic sentence, no period; date YYYY-MM-DD when event occurred, empty if unknown
{"tool":"upsert_node","args":{"name":"kebab-case-string","type":"entity | fact","body":"","date":""}}

upsert_edge — create or update a directed edge.
- content: "<subject> verb <target>" or "<subject> verb <fact>"; "implies" for fact→fact; empty when subject=true
- stability: 1.0 permanent, 0.7 can change, 0.3 temporary
- confidence: 1.0 stated clearly, 0.7 once, 0.4 inferred, 0.1 uncertain
- importance: 1.0 must load with source, 0.0 supplementary
- subject: true only on fact nodes; set importance=1.0
- date: always today's date
{"tool":"upsert_edge","args":{"source":"string","target":"string","content":"string","stability":0.0,"confidence":0.0,"importance":0.0,"subject":false,"date":"YYYY-MM-DD"}}

delete_node — remove a node.
{"tool":"delete_node","args":{"name":"string"}}

delete_edge — remove a directed edge.
{"tool":"delete_edge","args":{"source":"string","target":"string"}}"#;

pub const MEMORY_RULES: &str = r#"Node types: entity (person, place, technology, named concept; body and date always empty) or fact (one atomic claim; body is one sentence, no period; date YYYY-MM-DD when event occurred, empty if unknown; use today for current events; resolve relative dates to YYYY-MM-DD).

Direct edges vs fact nodes: use a direct edge for simple relationships. Create a fact node only when the claim needs a sentence — attribution, uncertainty, temporal context, or multiple entities.
- Simple: entity-a → entity-b, content "<subject> verb <target>"
- Complex: entity-a → fact-node, content "<subject> verb <fact>"; then fact-node → entity-b (subject declaration)

Subject declaration: every fact node has exactly one subject — the primary entity it's about. Edge from fact node to that entity: subject=true, content empty, importance=1.0.

Implies links: when one fact causally leads to another, edge from first to second: content="implies", importance=1.0.

Predicates: reuse existing content templates from current memory. Only introduce a new template when none fits. Use <subject> for source, <target> for target, <fact> when target is a fact node.

One-way only. Never add reverse links.

Never hallucinate. Only store what was explicitly stated.

Extract aggressively. Every named entity gets a node. Every stated relationship becomes an edge. If a fact involves multiple entities, add edges from each.

Taxonomy links must chain through every category level so unrelated nodes can connect through shared concepts."#;

pub const JSON_OUTPUT_RULES: &str = r#"Respond with raw JSON only. No markdown. No prose. No code fences. Start with `{`.

Single call: {"tool":"upsert_node","args":{"name":"example","type":"entity","body":"","date":""}}
Multiple calls: {"tools":[{"tool":"upsert_node","args":{"name":"example","type":"entity","body":"","date":""}},{"tool":"upsert_edge","args":{"source":"a","target":"b","content":"<subject> verb <target>","stability":1.0,"confidence":1.0,"importance":0.0,"subject":false,"date":"YYYY-MM-DD"}}]}"#;

pub fn build_agent_prompt(
    explanation: &str,
    toon: &str,
    user_msg: &str,
    assistant_msg: &str,
    diff_text: Option<&str>,
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
    if let Some(diff) = diff_text {
        prompt.push_str("\n\n## What extract added\n\n");
        prompt.push_str(diff);
    }
    prompt.push_str("\n\n## Exchange to process\n\nUser: ");
    prompt.push_str(user_msg);
    prompt.push_str("\nAssistant: ");
    prompt.push_str(assistant_msg);
    prompt
}

pub fn build_chat_system_prompt(memory_toon: &str, recent_entries: &[RecentEntry]) -> String {
    let mut prompt = String::from(CHAT_SYSTEM_PROMPT);
    prompt.push_str("\nToday's date is ");
    prompt.push_str(&today_iso());
    prompt.push('.');
    if !memory_toon.trim().is_empty() {
        prompt.push_str("\n\n<memory>\n");
        prompt.push_str(memory_toon);
        prompt.push_str("\n</memory>");
    }
    if !recent_entries.is_empty() {
        prompt.push_str("\n\n<recent_context>\n");
        for recent_entry in recent_entries {
            prompt.push_str(recent_role_label(recent_entry));
            prompt.push_str(": ");
            prompt.push_str(&recent_entry.content);
            prompt.push('\n');
        }
        prompt.push_str("</recent_context>");
    }
    prompt
}

fn recent_role_label(recent_entry: &RecentEntry) -> &str {
    match recent_entry.role.as_str() {
        "user" => "User",
        "assistant" => "Assistant",
        _ => "Message",
    }
}

pub fn today_iso() -> String {
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
