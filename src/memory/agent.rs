use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient, ToolCallRequest};
use crate::prompts::{
    AgentPrompt, CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, SCRIBE_EXPLANATION,
    build_agent_prompt, build_classify_prompt, build_continue_prompt, build_scribe_continue_prompt,
    build_scribe_prompt,
};
use crate::runlog;
use turso::Connection;

use super::db;
use super::decay;
use super::embed::SemanticEdge;
use super::graph::{Edge, Node, PROVENANCE_USER_CONFIRMED, graph_to_json, normalize_id};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

const POSITIVE_REACTION_SCORE: f64 = 0.5; // replay-tunable.
const NEGATIVE_REACTION_SCORE: f64 = -0.5; // replay-tunable.

pub enum MemoryEvent {
    Log { phase: &'static str, text: String },
    Append { phase: &'static str, text: String },
    Done,
    MaintenanceDone,
}

// The three model seats the memory pass runs on. classifier picks task-vs-chat;
// scribe records episodic moves; heavy runs extract/consolidate. They are often
// clones of the same client — default config wires all three to MEMORY_MODEL.
#[derive(Clone)]
pub struct MemorySeats {
    pub classifier: Arc<dyn LlmClient>,
    pub scribe: Arc<dyn LlmClient>,
    pub heavy: Arc<dyn LlmClient>,
}

impl MemorySeats {
    // All three seats on one client — the pre-split default.
    #[cfg(test)]
    pub fn uniform(llm: Arc<dyn LlmClient>) -> Self {
        Self {
            classifier: Arc::clone(&llm),
            scribe: Arc::clone(&llm),
            heavy: llm,
        }
    }
}

struct AgentRun<'run> {
    seats: MemorySeats,
    user_msg: &'run str,
    assistant_msg: &'run str,
    sender: &'run std::sync::mpsc::Sender<MemoryEvent>,
    user_message_id: &'run str,
}

// A heavy pass emits a few tool calls per response and is re-prompted until it
// says DONE, so no single response has to carry the whole pass.
const MAX_PASS_ITERATIONS: usize = 4; // replay-tunable
// The scribe runs post-turn (not in the chat critical path) and one continuation
// is enough to catch an under-emitted setup turn (define_task + both opening
// plies), so its own cap is tighter than the heavy passes'.
const MAX_SCRIBE_ITERATIONS: usize = 2; // replay-tunable

struct PassSpec {
    phase: &'static str,
    default_memory_type: &'static str,
    // Re-sent on every continuation so the pass keeps its role; a generic
    // continuation prompt turned consolidate into a second extractor.
    explanation: &'static str,
    // Per-pass iteration cap — the heavy passes get MAX_PASS_ITERATIONS, the
    // scribe the tighter MAX_SCRIBE_ITERATIONS.
    max_iterations: usize,
    // Builds the continuation prompt for this pass — heavy passes reuse the
    // heavy tool docs, the scribe keeps its own tool set.
    continue_prompt: fn(&str, &str, &str, &[String]) -> String,
}

struct IterationOutcome {
    descriptions: Vec<String>,
    done: bool,
}

// The client and connection a pass runs against — kept together so the
// per-iteration signatures stay under the argument cap.
struct PassCtx<'ctx> {
    llm: &'ctx Arc<dyn LlmClient>,
    conn: &'ctx Connection,
}

pub(crate) enum ToolCall {
    UpsertNode {
        name: String,
        node_type: String,
        body: String,
        date: String,
    },
    UpsertEdge {
        source: String,
        target: String,
        content: String,
        stability: f64,
        confidence: f64,
        importance: f64,
        subject: bool,
        date: String,
        // None means the caller did not specify memory_type — the writer's
        // default (ApplyToolOptions::default_memory_type) applies.
        memory_type: Option<String>,
    },
    DeleteNode {
        name: String,
    },
    DeleteEdge {
        source: String,
        target: String,
    },
    ExpireEdge {
        source: String,
        target: String,
        rewritten_content: String,
    },
    PromoteEdge {
        source: String,
        target: String,
    },
    // Lossy consolidation: a new semantic generalization induced from several
    // episodic edges. supported_by holds the (source, target) refs of those
    // episodes, already normalized. Dream-side only — the chat apply path
    // denies it.
    GistEdge {
        source: String,
        target: String,
        content: String,
        confidence: f64,
        stability: f64,
        supported_by: Vec<(String, String)>,
    },
    // Scribe-only: the model supplies only the move text and whose move it was;
    // code derives the task hub, ply number, and move-node id (apply_record_move).
    // Denied on every non-scribe apply path.
    RecordMove {
        move_text: String,
        actor: String,
    },
    // Scribe-only: define (once) the game/task being set up. Upserts the durable
    // game-definition node and links this session's hub to it.
    DefineTask {
        name: String,
        description: String,
    },
    // Scribe or heavy: anchor a stated rule of the current game as durable
    // knowledge on the game-definition node. Dream denies it.
    RecordRule {
        rule: String,
    },
    // Scribe-only: the game ended — score the session from the assistant's
    // perspective (win/loss/draw). Denied on every non-scribe apply path.
    RecordOutcome {
        result: String,
        comment: String,
    },
    RecordReaction {
        reaction: String,
        comment: String,
    },
    // Heavy-only prospective memory: open an intention against a contextual
    // anchor. Code derives the loop id and fixed durability fields.
    OpenLoop {
        intention: String,
        anchor: String,
        due: String,
    },
    // Heavy-only lifecycle transition: expire the loop's live anchor edge and
    // retain the resolution in rewritten_content.
    CloseLoop {
        loop_id: String,
        resolution: String,
        note: String,
    },
    InboxAdd {
        source: String,
        instruction: String,
    },
    RegisterInterest {
        topic: String,
    },
    SetPingPolicy {
        action: String,
        topic: String,
        until: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnMode {
    TaskMove,
    Conversation,
}

impl TurnMode {
    fn label(self) -> &'static str {
        match self {
            TurnMode::TaskMove => "task",
            TurnMode::Conversation => "chat",
        }
    }
}

fn classify_turn(
    llm: &Arc<dyn LlmClient>,
    user_msg: &str,
    assistant_msg: &str,
    active_task: bool,
) -> TurnMode {
    let _timer = runlog::time_step("mem", "classify");
    let prompt = build_classify_prompt(user_msg, assistant_msg, active_task);
    match llm.collect(&prompt) {
        Ok(reply) => parse_turn_mode(&reply),
        Err(_) => TurnMode::Conversation,
    }
}

// Read-only classifier verdict label ("task"/"chat") for a raw model reply —
// used by the seat-replay harness, which never touches the DB.
pub(crate) fn classify_reply_label(reply: &str) -> &'static str {
    parse_turn_mode(reply).label()
}

// One-line human description of a parsed tool call for eyeball inspection in
// the replay report.
pub(crate) fn describe_tool_call(call: &ToolCall) -> String {
    match call {
        ToolCall::UpsertNode {
            name, node_type, ..
        } => {
            format!("upsert_node {name} ({node_type})")
        }
        ToolCall::UpsertEdge { source, target, .. } => format!("upsert_edge {source} → {target}"),
        ToolCall::DeleteNode { name } => format!("delete_node {name}"),
        ToolCall::DeleteEdge { source, target } => format!("delete_edge {source} → {target}"),
        ToolCall::ExpireEdge { source, target, .. } => format!("expire_edge {source} → {target}"),
        ToolCall::PromoteEdge { source, target } => format!("promote_edge {source} → {target}"),
        ToolCall::GistEdge { source, target, .. } => format!("gist_edge {source} → {target}"),
        ToolCall::RecordMove { move_text, actor } => format!("record_move {move_text} ({actor})"),
        ToolCall::DefineTask { name, .. } => format!("define_task {name}"),
        ToolCall::RecordRule { rule } => format!("record_rule {rule}"),
        ToolCall::RecordOutcome { result, .. } => format!("record_outcome {result}"),
        ToolCall::RecordReaction { reaction, .. } => format!("record_reaction {reaction}"),
        ToolCall::SetPingPolicy { action, topic, .. } => {
            format!("set_ping_policy {action} {topic}")
        }
        ToolCall::OpenLoop { intention, .. } => format!("open_loop {intention}"),
        ToolCall::CloseLoop {
            loop_id,
            resolution,
            ..
        } => {
            format!("close_loop {loop_id} ({resolution})")
        }
        ToolCall::InboxAdd { source, .. } => format!("inbox_add {source}"),
        ToolCall::RegisterInterest { topic } => format!("register_interest {topic}"),
    }
}

fn parse_turn_mode(reply: &str) -> TurnMode {
    // Bias toward CHAT (the heavy semantic path): a conversation misclassified
    // as a task would record durable facts as fast-decaying episodic edges —
    // worse than the reverse — so any mention of CHAT wins the tie.
    let upper = reply.to_uppercase();
    if upper.contains("TASK") && !upper.contains("CHAT") {
        TurnMode::TaskMove
    } else {
        TurnMode::Conversation
    }
}

pub fn run_memory_agent(
    seats: MemorySeats,
    user_msg: &str,
    assistant_msg: &str,
    user_message_id: &str,
    sender: std::sync::mpsc::Sender<MemoryEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    super::sequence::record_feedback(user_message_id);
    let agent_run = AgentRun {
        seats,
        user_msg,
        assistant_msg,
        sender: &sender,
        user_message_id,
    };
    let result = agent_run.run();
    db::sync_push_best_effort("memory agent");
    let _ = sender.send(MemoryEvent::Done);
    result
}

impl<'run> AgentRun<'run> {
    fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let active_task = self.session_has_chain();
        let mode = classify_turn(
            &self.seats.classifier,
            self.user_msg,
            self.assistant_msg,
            active_task,
        );
        self.send_log("mem", format!("turn classified: {}", mode.label()));
        match mode {
            TurnMode::TaskMove => self.run_scribe(),
            TurnMode::Conversation => self.run_heavy(),
        }
    }

    fn consolidate_prompt(
        &self,
        pre_extract_graph: &super::graph::Graph,
        pre_extract_toon: &str,
        active_game: Option<&str>,
    ) -> String {
        let post_extract_graph = db::load_graph();
        let diff = db::graph_diff_toon(pre_extract_graph, &post_extract_graph);
        build_agent_prompt(&AgentPrompt {
            explanation: CONSOLIDATE_EXPLANATION,
            toon: pre_extract_toon,
            user_msg: self.user_msg,
            assistant_msg: self.assistant_msg,
            diff_text: if diff.is_empty() { None } else { Some(&diff) },
            injected_toon: None,
            active_game,
        })
    }

    // The display name of the game this session plays, if any — deterministic
    // context for the heavy prompt so rules/outcomes route to the game tools.
    fn active_game_name(&self, conn: &Connection) -> Option<String> {
        let session_id = db::source_session_id(self.user_message_id)?;
        let hub_id = normalize_id(&format!("task-{session_id}"));
        let (game_id, _) = db::load_session_game(conn, &hub_id).ok()??;
        Some(super::task_chain::game_display_name(&game_id))
    }

    // Does this turn's session already have a live move chain? Best-effort:
    // any failure reads as "no chain" and classification proceeds unbiased.
    fn session_has_chain(&self) -> bool {
        let Some(session_id) = db::source_session_id(self.user_message_id) else {
            return false;
        };
        let Ok(conn) = db::open_connection() else {
            return false;
        };
        let hub_id = normalize_id(&format!("task-{session_id}"));
        db::count_task_chain(&conn, &hub_id).is_ok_and(|count| count > 0)
    }

    fn run_scribe(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _write_guard = db::acquire_write_lock();
        let conn = db::open_connection()?;
        self.refresh_injected_clock(&conn)?;
        let prompt = build_scribe_prompt(self.user_msg, self.assistant_msg);
        // Looped (cap MAX_SCRIBE_ITERATIONS): the scribe runs post-turn, so a
        // second iteration only costs latency when the first response
        // under-emitted an opening turn — and then it recovers the lost plies.
        let ctx = PassCtx {
            llm: &self.seats.scribe,
            conn: &conn,
        };
        self.run_pass(&ctx, &scribe_pass(), prompt)?;
        Ok(())
    }

    fn run_heavy(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _write_guard = db::acquire_write_lock();
        let conn = db::open_connection()?;
        // Before decay, so beliefs used this turn are not aged in the same pass.
        let injected = self.refresh_injected_clock(&conn)?;
        let graph = self.decayed_graph()?;
        let ctx = PassCtx {
            llm: &self.seats.heavy,
            conn: &conn,
        };
        let pre_extract_graph = graph.clone();
        let pre_extract_toon = toon::encode(&graph_to_json(&pre_extract_graph), None);
        let injected_toon = injected_toon(&injected);
        let active_game = self.active_game_name(&conn);
        let extract_prompt = build_agent_prompt(&AgentPrompt {
            explanation: EXTRACT_EXPLANATION,
            toon: &pre_extract_toon,
            user_msg: self.user_msg,
            assistant_msg: self.assistant_msg,
            diff_text: None,
            injected_toon: injected_toon.as_deref(),
            active_game: active_game.as_deref(),
        });
        self.run_pass(&ctx, &heavy_pass("extract"), extract_prompt)?;

        let consolidate_prompt = self.consolidate_prompt(
            &pre_extract_graph,
            &pre_extract_toon,
            active_game.as_deref(),
        );
        self.run_pass(&ctx, &heavy_pass("consolidate"), consolidate_prompt)?;

        let final_graph = db::load_graph();
        self.send_log(
            "memory",
            format!(
                "saved — {} nodes, {} edges",
                final_graph.nodes.len(),
                final_graph.edges.len()
            ),
        );
        Ok(())
    }

    fn decayed_graph(&self) -> Result<super::graph::Graph, Box<dyn Error + Send + Sync>> {
        let mut graph = db::load_graph();
        let decay_ran = decay::apply_decay_if_needed(&mut graph);
        if decay_ran && let Err(err) = db::save_graph(&graph) {
            self.send_log("memory", format!("error: {err}"));
            return Err(err);
        }
        Ok(graph)
    }

    // Spacing effect: the beliefs retrieval used this turn get their decay
    // clock reset to today. The `injected` ledger event already records the
    // use, so no new event type is written here — this is only the clock touch.
    fn refresh_injected_clock(
        &self,
        conn: &Connection,
    ) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
        let injected = db::block_on(ledger::load_injected_edges(conn, self.user_message_id))?;
        let today = crate::prompts::today_iso();
        let refreshed = db::block_on(db::touch_edge_dates(conn, &injected, &today))?;
        self.send_log(
            "mem",
            format!("refreshed decay clock on {refreshed} injected edges"),
        );
        Ok(injected)
    }

    fn run_pass(
        &self,
        ctx: &PassCtx<'_>,
        pass: &PassSpec,
        prompt: String,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.drive_pass(pass, prompt, |iteration, iteration_prompt| {
            self.run_iteration(ctx, pass, iteration, iteration_prompt)
        })
    }

    // Emit → apply → re-prompt. Continuations carry only the exchange and a
    // one-line summary of what was applied, never the full graph again.
    fn drive_pass<Iterate>(
        &self,
        pass: &PassSpec,
        first_prompt: String,
        mut iterate: Iterate,
    ) -> Result<(), Box<dyn Error + Send + Sync>>
    where
        Iterate: FnMut(usize, String) -> Result<IterationOutcome, Box<dyn Error + Send + Sync>>,
    {
        let mut prompt = first_prompt;
        let mut applied: Vec<String> = Vec::new();
        for iteration in 1..=pass.max_iterations {
            let outcome = iterate(iteration, prompt)?;
            if outcome.done || iteration == pass.max_iterations {
                break;
            }
            // Continuations must list everything applied so far, not just the
            // latest iteration, or the model re-emits earlier calls.
            applied.extend(outcome.descriptions);
            prompt = (pass.continue_prompt)(
                pass.explanation,
                self.user_msg,
                self.assistant_msg,
                &applied,
            );
        }
        self.send_log(pass.phase, "done".to_string());
        Ok(())
    }

    fn run_iteration(
        &self,
        ctx: &PassCtx<'_>,
        pass: &PassSpec,
        iteration: usize,
        prompt: String,
    ) -> Result<IterationOutcome, Box<dyn Error + Send + Sync>> {
        let _timer = runlog::time_step(pass.phase, format!("iteration {iteration}"));
        self.send_log(pass.phase, format!("iteration {iteration}: calling LLM…"));
        let call_text = match stream_collect(
            Arc::clone(ctx.llm),
            prompt,
            |text| self.send_append("think", text),
            |text| self.send_append("debug", text),
        ) {
            Ok(call_text) => call_text,
            Err(err) => {
                self.send_log(pass.phase, format!("error: {err}"));
                return Err(err);
            }
        };
        let tool_calls = parse_tool_calls(&call_text);
        self.send_log(
            pass.phase,
            format!("iteration {iteration}: {} tool calls", tool_calls.len()),
        );
        // A response may both apply calls and declare the pass finished.
        let done = tool_calls.is_empty() || says_done(&call_text);
        match apply_tool_calls(
            self,
            ctx.conn,
            pass.phase,
            tool_calls,
            pass.default_memory_type,
        ) {
            Ok(descriptions) => Ok(IterationOutcome { descriptions, done }),
            Err(err) => {
                self.send_log(pass.phase, format!("error: {err}"));
                Err(err)
            }
        }
    }

    fn send_log(&self, phase: &'static str, text: String) {
        runlog::log(phase, &text);
        let _ = self.sender.send(MemoryEvent::Log { phase, text });
    }

    fn send_append(&self, phase: &'static str, text: String) {
        let _ = self.sender.send(MemoryEvent::Append { phase, text });
    }
}

// None when nothing was injected — the prompt then omits the section entirely
// rather than showing the model an empty prediction.
fn injected_toon(edges: &[Edge]) -> Option<String> {
    if edges.is_empty() {
        return None;
    }
    let mut semantic = Vec::new();
    for edge in edges {
        semantic.push(to_semantic_edge(edge));
    }
    Some(super::embed::edges_to_toon(&semantic))
}

fn to_semantic_edge(edge: &Edge) -> SemanticEdge {
    SemanticEdge {
        source: edge.source.clone(),
        target: edge.target.clone(),
        content: edge.content.clone(),
        confidence: edge.confidence,
        stability: edge.stability,
        importance: edge.importance,
        subject: edge.subject,
        date: edge.date.clone(),
        occurred_at: edge.occurred_at.clone(),
        expired_at: edge.expired_at.clone(),
        memory_type: edge.memory_type.clone(),
        origin_session: edge.origin_session.clone(),
        provenance: edge.provenance.clone(),
        embedding: None,
    }
}

fn heavy_pass(phase: &'static str) -> PassSpec {
    let explanation = if phase == "extract" {
        EXTRACT_EXPLANATION
    } else {
        CONSOLIDATE_EXPLANATION
    };
    PassSpec {
        phase,
        default_memory_type: "semantic",
        explanation,
        max_iterations: MAX_PASS_ITERATIONS,
        continue_prompt: build_continue_prompt,
    }
}

fn scribe_pass() -> PassSpec {
    PassSpec {
        phase: "scribe",
        default_memory_type: "episodic",
        // Re-sent on each continuation so the scribe keeps its role and its own
        // tool set (record_move/define_task/record_rule/record_outcome).
        explanation: SCRIBE_EXPLANATION,
        max_iterations: MAX_SCRIBE_ITERATIONS,
        continue_prompt: build_scribe_continue_prompt,
    }
}

// Whole-word match: substring matching would fire on "abandoned"/"undone"
// inside prose or edge content and kill the pass early.
fn says_done(text: &str) -> bool {
    for token in text.split(|letter: char| !letter.is_ascii_alphanumeric()) {
        if token.eq_ignore_ascii_case("done") {
            return true;
        }
    }
    false
}

pub(super) fn stream_collect(
    llm: Arc<dyn LlmClient>,
    prompt: String,
    on_reasoning: impl Fn(String),
    on_content: impl Fn(String),
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let (chunk_tx, chunk_rx) = std::sync::mpsc::channel::<Chunk>();
    std::thread::spawn(move || llm.stream(prompt, None, chunk_tx));
    let mut reasoning_full = String::new();
    let mut content = String::new();
    let mut err: Option<Box<dyn Error + Send + Sync>> = None;
    for chunk in chunk_rx {
        match chunk {
            Chunk::Reasoning(text) => {
                reasoning_full.push_str(&text);
                on_reasoning(text);
            }
            Chunk::Content(text) => {
                content.push_str(&text);
                on_content(text);
            }
            Chunk::Done(_) => break,
            Chunk::Err(error) => {
                err = Some(error.into());
                break;
            }
            Chunk::ToolCall(_) | Chunk::MemoryLog { .. } | Chunk::Attribution(_) => {}
        }
    }
    if let Some(error) = err {
        return Err(error);
    }
    if !content.is_empty() {
        return Ok(content);
    }
    Ok(reasoning_full)
}

pub(crate) fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let stripped = strip_markdown_fences(text);
    let json_value = match serde_json::from_str::<serde_json::Value>(stripped) {
        Ok(json_value) => json_value,
        // Models sometimes wrap the JSON in prose or channel markers — retry
        // on the outermost brace span. If that also fails the response is
        // likely truncated mid-array: salvage the complete tool objects
        // rather than discarding the whole batch.
        Err(_) => match parse_embedded_json(stripped) {
            Some(json_value) => json_value,
            None => return salvage_tool_calls(stripped),
        },
    };
    if let Some(tools) = tool_array(&json_value) {
        let mut calls = Vec::new();
        for tool in tools {
            if let Some(call) = parse_single_tool(tool) {
                calls.push(call);
            }
        }
        return calls;
    }
    match parse_single_tool(&json_value) {
        Some(call) => vec![call],
        None => Vec::new(),
    }
}

// Walks the "tools" array of a truncated response and parses each complete
// {...} object individually; an object cut off by truncation is dropped.
fn salvage_tool_calls(text: &str) -> Vec<ToolCall> {
    let Some(tools_key) = text.find("\"tools\"") else {
        return Vec::new();
    };
    let after_key = &text[tools_key..];
    let Some(array_offset) = after_key.find('[') else {
        return Vec::new();
    };
    let array_body = &after_key[array_offset + 1..];
    let mut calls = Vec::new();
    for span in complete_object_spans(array_body) {
        let Ok(json_value) = serde_json::from_str::<serde_json::Value>(span) else {
            continue;
        };
        if let Some(call) = parse_single_tool(&json_value) {
            calls.push(call);
        }
    }
    calls
}

// Returns the byte spans of balanced top-level {...} objects, string-aware.
fn complete_object_spans(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut strings = StringScanner::default();
    for (index, byte) in text.bytes().enumerate() {
        if strings.consumes(byte) {
            continue;
        }
        match byte {
            b'{' => {
                if depth == 0 {
                    start = index;
                }
                depth += 1;
            }
            b'}' => {
                if depth == 1 {
                    spans.push(&text[start..=index]);
                }
                depth = depth.saturating_sub(1);
            }
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    spans
}

// Tracks whether the scan is inside a JSON string literal (escape-aware).
#[derive(Default)]
struct StringScanner {
    in_string: bool,
    escaped: bool,
}

impl StringScanner {
    fn consumes(&mut self, byte: u8) -> bool {
        if self.in_string {
            if self.escaped {
                self.escaped = false;
            } else if byte == b'\\' {
                self.escaped = true;
            } else if byte == b'"' {
                self.in_string = false;
            }
            return true;
        }
        if byte == b'"' {
            self.in_string = true;
            return true;
        }
        false
    }
}

fn parse_embedded_json(text: &str) -> Option<serde_json::Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(&text[start..=end]).ok()
}

fn strip_markdown_fences(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(fenced) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let Some((_, body)) = fenced.split_once('\n') else {
        return trimmed;
    };
    let Some(body) = body.strip_suffix("```") else {
        return trimmed;
    };
    body.trim()
}

fn parse_single_tool(obj: &serde_json::Value) -> Option<ToolCall> {
    let tool_name = obj.get("tool")?.as_str()?;
    let args = match obj.get("args") {
        Some(val) => val,
        None => obj.get("arguments")?,
    };
    match tool_name {
        "upsert_node" => parse_upsert_node(args),
        "upsert_edge" => parse_upsert_edge(args),
        "delete_node" => parse_delete_node(args),
        "delete_edge" => parse_delete_edge(args),
        "expire_edge" => parse_expire_edge(args),
        "promote_edge" => parse_promote_edge(args),
        "gist_edge" => parse_gist_edge(args),
        "record_move" => parse_record_move(args),
        "define_task" => parse_define_task(args),
        "record_rule" => parse_record_rule(args),
        "record_outcome" => parse_record_outcome(args),
        "record_reaction" => parse_record_reaction(args),
        "open_loop" => parse_open_loop(args),
        "close_loop" => parse_close_loop(args),
        "inbox_add" => parse_inbox_add(args),
        "register_interest" => parse_register_interest(args),
        "set_ping_policy" => parse_set_ping_policy(args),
        _ => None,
    }
}

// gpt-oss models leak harmony channel markers into the tool name
// ("upsert_node<|channel|>json") — strip everything from the first "<|" so
// those calls are not silently dropped.
pub(crate) fn sanitized_tool_name(name: &str) -> &str {
    match name.find("<|") {
        Some(index) => &name[..index],
        None => name,
    }
}

pub(crate) fn parse_tool_call_request(call: &ToolCallRequest) -> Option<ToolCall> {
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.args) else {
        return None;
    };
    match sanitized_tool_name(&call.name) {
        "upsert_node" => parse_upsert_node(&args),
        "upsert_edge" => parse_upsert_edge(&args),
        "delete_node" => parse_delete_node(&args),
        "delete_edge" => parse_delete_edge(&args),
        "expire_edge" => parse_expire_edge(&args),
        "promote_edge" => parse_promote_edge(&args),
        "inbox_add" => parse_inbox_add(&args),
        "record_reaction" => parse_record_reaction(&args),
        "register_interest" => parse_register_interest(&args),
        "set_ping_policy" => parse_set_ping_policy(&args),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn memory_mutation_tools() -> serde_json::Value {
    serde_json::Value::Array(vec![
        memory_tool(
            "upsert_node",
            "Create or update a graph node.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["entity", "fact", "note"]},
                    "body": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["name", "type", "body", "date"]
            }),
        ),
        memory_tool(
            "register_interest",
            "Adopt a topic as a durable standing interest.",
            serde_json::json!({
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            }),
        ),
        memory_tool(
            "upsert_edge",
            "Create or update a directed graph edge.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                    "stability": {"type": "number"},
                    "confidence": {"type": "number"},
                    "importance": {"type": "number"},
                    "subject": {"type": "boolean"},
                    "date": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic"]}
                },
                "required": ["source", "target", "content", "stability", "confidence", "importance", "subject", "date"]
            }),
        ),
        memory_tool(
            "delete_node",
            "Delete a node and its incident edges.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }),
        ),
        memory_tool(
            "delete_edge",
            "Delete a directed edge.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["source", "target"]
            }),
        ),
        memory_tool(
            "expire_edge",
            "Mark a belief historical instead of deleting it.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "rewritten_content": {"type": "string"}
                },
                "required": ["source", "target", "rewritten_content"]
            }),
        ),
    ])
}

fn memory_tool(name: &str, description: &str, parameters: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    })
}

fn tool_array(obj: &serde_json::Value) -> Option<&[serde_json::Value]> {
    let tools = obj.get("tools")?;
    match tools.as_array() {
        Some(array) => Some(array.as_slice()),
        None => None,
    }
}

fn parse_upsert_node(args: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(args, "name");
    let node_type = json_str(args, "type");
    if name.is_empty() || node_type.is_empty() {
        return None;
    }
    Some(ToolCall::UpsertNode {
        name,
        node_type,
        body: json_str(args, "body"),
        date: json_str(args, "date"),
    })
}

fn parse_upsert_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    let date = {
        let date_val = json_str(args, "date");
        if date_val.is_empty() {
            json_str(args, "reinforced_at")
        } else {
            date_val
        }
    };
    Some(ToolCall::UpsertEdge {
        source,
        target,
        content: json_str(args, "content"),
        stability: json_f64(args, "stability"),
        confidence: json_f64(args, "confidence"),
        importance: json_f64(args, "importance"),
        subject: json_bool(args, "subject"),
        date,
        memory_type: json_opt_str(args, "memory_type"),
    })
}

fn parse_promote_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some(ToolCall::PromoteEdge { source, target })
}

// supported_by is mandatory and must be an array — a gist without provenance
// is a dream hypothesis, not a consolidation, so a missing/malformed field
// rejects the whole call rather than degrading it silently.
fn parse_gist_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    let refs = args.get("supported_by")?.as_array()?;
    let mut supported_by = Vec::new();
    for value in refs {
        if let Some(text) = value.as_str()
            && let Some(edge_ref) = parse_support_ref(text)
        {
            supported_by.push(edge_ref);
        }
    }
    let stability = match args.get("stability") {
        Some(value) => value.as_f64().unwrap_or(0.5),
        None => 0.5,
    };
    Some(ToolCall::GistEdge {
        source,
        target,
        content: json_str(args, "content"),
        confidence: json_f64(args, "confidence"),
        stability,
        supported_by,
    })
}

// The move text is the only required field: without it there is nothing to
// record. actor defaults to empty and is mapped to a valid seat at apply time.
fn parse_record_move(args: &serde_json::Value) -> Option<ToolCall> {
    let move_text = json_str(args, "move");
    if move_text.is_empty() {
        return None;
    }
    Some(ToolCall::RecordMove {
        move_text,
        actor: json_str(args, "actor"),
    })
}

// The game name is required — without it there is no node to define. The
// description (rules + presentation instructions) may be empty on a re-mention.
fn parse_define_task(args: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(args, "name");
    if name.is_empty() {
        return None;
    }
    Some(ToolCall::DefineTask {
        name,
        description: json_str(args, "description"),
    })
}

fn parse_record_rule(args: &serde_json::Value) -> Option<ToolCall> {
    let rule = json_str(args, "rule");
    if rule.is_empty() {
        return None;
    }
    Some(ToolCall::RecordRule { rule })
}

// result must be one of the three discrete verdicts — an unrecognized value is
// a malformed call, not a partial score, so the whole call is rejected.
fn parse_record_outcome(args: &serde_json::Value) -> Option<ToolCall> {
    let result = json_str(args, "result");
    if !matches!(result.as_str(), "win" | "loss" | "draw") {
        return None;
    }
    Some(ToolCall::RecordOutcome {
        result,
        comment: json_str(args, "comment"),
    })
}

fn parse_record_reaction(args: &serde_json::Value) -> Option<ToolCall> {
    let reaction = json_str(args, "reaction");
    if !matches!(reaction.as_str(), "positive" | "negative") {
        return None;
    }
    Some(ToolCall::RecordReaction {
        reaction,
        comment: json_str(args, "comment"),
    })
}

fn parse_set_ping_policy(args: &serde_json::Value) -> Option<ToolCall> {
    let action = json_str(args, "action");
    let topic = json_str(args, "topic");
    let until = json_str(args, "until");
    let valid = match action.as_str() {
        "resume" => true,
        "mute" => !until.is_empty(),
        "divert" => !topic.is_empty() && !until.is_empty(),
        _ => false,
    };
    if !valid {
        return None;
    }
    Some(ToolCall::SetPingPolicy {
        action,
        topic,
        until,
    })
}

fn parse_open_loop(args: &serde_json::Value) -> Option<ToolCall> {
    let intention = json_str(args, "intention");
    let anchor = json_str(args, "anchor");
    if intention.is_empty() || anchor.is_empty() {
        return None;
    }
    Some(ToolCall::OpenLoop {
        intention,
        anchor,
        due: json_str(args, "due"),
    })
}

fn parse_inbox_add(args: &serde_json::Value) -> Option<ToolCall> {
    Some(ToolCall::InboxAdd {
        source: json_str(args, "source"),
        instruction: json_str(args, "instruction"),
    })
}

fn parse_register_interest(args: &serde_json::Value) -> Option<ToolCall> {
    let topic = json_str(args, "topic");
    if topic.trim().is_empty() {
        return None;
    }
    Some(ToolCall::RegisterInterest { topic })
}

fn parse_close_loop(args: &serde_json::Value) -> Option<ToolCall> {
    let loop_id = json_str(args, "loop");
    let resolution = json_str(args, "resolution");
    if loop_id.is_empty() || !matches!(resolution.as_str(), "done" | "abandoned" | "absence") {
        return None;
    }
    Some(ToolCall::CloseLoop {
        loop_id,
        resolution,
        note: json_str(args, "note"),
    })
}

// Accepts both "a → b" and "a->b" spellings and normalizes both endpoints.
fn parse_support_ref(text: &str) -> Option<(String, String)> {
    let (raw_source, raw_target) = split_support_ref(text)?;
    let source = normalize_id(raw_source);
    let target = normalize_id(raw_target);
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some((source, target))
}

fn split_support_ref(text: &str) -> Option<(&str, &str)> {
    if let Some(parts) = text.split_once('→') {
        return Some(parts);
    }
    text.split_once("->")
}

fn parse_delete_node(args: &serde_json::Value) -> Option<ToolCall> {
    let name = json_str(args, "name");
    if name.is_empty() {
        return None;
    }
    Some(ToolCall::DeleteNode { name })
}

fn parse_delete_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some(ToolCall::DeleteEdge { source, target })
}

fn parse_expire_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    let rewritten_content = json_str(args, "rewritten_content");
    if source.is_empty() || target.is_empty() || rewritten_content.is_empty() {
        return None;
    }
    Some(ToolCall::ExpireEdge {
        source,
        target,
        rewritten_content,
    })
}

fn json_str(obj: &serde_json::Value, key: &str) -> String {
    let Some(value) = obj.get(key) else {
        return String::new();
    };
    match value.as_str() {
        Some(text) => text.to_string(),
        None => String::new(),
    }
}

fn json_opt_str(obj: &serde_json::Value, key: &str) -> Option<String> {
    let text = obj.get(key)?.as_str()?;
    Some(text.to_string())
}

fn json_bool(obj: &serde_json::Value, key: &str) -> bool {
    let Some(value) = obj.get(key) else {
        return false;
    };
    value.as_bool().unwrap_or(false)
}

fn json_f64(obj: &serde_json::Value, key: &str) -> f64 {
    let Some(value) = obj.get(key) else {
        return 0.0;
    };
    value.as_f64().unwrap_or(0.0)
}

pub(crate) struct ApplyToolOptions<'scope> {
    pub actor: Actor,
    pub context: EventContext,
    pub source_message_id: &'scope str,
    pub event_payload: &'scope str,
    pub confidence_cap: Option<f64>,
    pub provenance: &'static str,
    // memory_type an upsert_edge call gets when it doesn't specify one
    // itself. Chat and ingestion write durable belief ("semantic"— lived
    // experience is not yet knowledge); exploration/task sessions write
    // lived experience by default ("episodic").
    pub default_memory_type: &'static str,
}

fn apply_tool_calls<'agent_run>(
    agent_run: &AgentRun<'agent_run>,
    conn: &Connection,
    phase: &'static str,
    tool_calls: Vec<ToolCall>,
    default_memory_type: &'static str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let mut events: Vec<Event> = Vec::new();
    let mut descriptions: Vec<String> = Vec::new();
    let result = (|| {
        let options = ApplyToolOptions {
            actor: Actor::new("system", phase),
            context: mutation_context(agent_run.user_message_id),
            source_message_id: agent_run.user_message_id,
            event_payload: "",
            confidence_cap: None,
            provenance: PROVENANCE_USER_CONFIRMED,
            default_memory_type,
        };
        for call in tool_calls {
            let description = apply_tool_call(call, conn, &options, &mut events)?;
            agent_run.send_log(phase, description.clone());
            descriptions.push(description);
        }
        db::block_on(ledger::record_events_in_tx(conn, &events))
    })();
    if let Err(err) = result {
        if let Err(rollback_err) = db::block_on(async { conn.execute("ROLLBACK", ()).await }) {
            runlog::log("ledger", &format!("rollback failed: {rollback_err}"));
        }
        return Err(err);
    }
    db::commit_transaction(conn)?;
    Ok(descriptions)
}

#[rustfmt::skip]
pub(crate) fn apply_tool_call<'scope>(
    call: ToolCall,
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if tool_call_touches_loop(conn, &call)? {
        return Ok("mutation denied: loop lifecycle requires open_loop or close_loop".to_string());
    }
    match call {
        ToolCall::UpsertNode { name, node_type, body, date } =>
            apply_upsert_node(conn, name, node_type, body, date),
        ToolCall::UpsertEdge { source, target, content, stability, confidence, importance, subject, date, memory_type } =>
            apply_upsert_edge_call(conn, options, events, UpsertEdgeCall { source, target, content, stability, confidence, importance, subject, date, memory_type }),
        ToolCall::DeleteNode { name } =>
            apply_delete_node(conn, name, options, events),
        ToolCall::DeleteEdge { source, target } =>
            apply_delete_edge(conn, EdgeRef::new(&source, &target), options, events),
        ToolCall::ExpireEdge { source, target, rewritten_content } =>
            apply_expire_edge(conn, ExpireEdgeCall { source, target, rewritten_content }, options, events),
        ToolCall::PromoteEdge { source, target } =>
            apply_promote_edge(conn, EdgeRef::new(&source, &target), options.actor.clone(), options.context.clone(), events),
        ToolCall::GistEdge { .. } =>
            Ok("gist_edge denied: gist_edge is a dreaming tool".to_string()),
        ToolCall::RecordMove { move_text, actor } =>
            apply_record_move_call(conn, options, events, RecordMoveCall { move_text, actor }),
        ToolCall::DefineTask { name, description } =>
            apply_define_task_call(conn, options, events, DefineTaskCall { name, description }),
        ToolCall::RecordRule { rule } =>
            apply_record_rule_call(conn, options, events, rule),
        ToolCall::RecordOutcome { result, comment } =>
            apply_record_outcome_call(conn, options, RecordOutcomeCall { result, comment }),
        ToolCall::RecordReaction { reaction, comment } =>
            apply_record_reaction_call(conn, options, &reaction, &comment),
        ToolCall::OpenLoop { intention, anchor, due } =>
            apply_open_loop_call(conn, options, events, OpenLoopCall { intention, anchor, due }),
        ToolCall::CloseLoop { loop_id, resolution, note } =>
            apply_close_loop_call(conn, options, events, CloseLoopCall { loop_id, resolution, note }),
        ToolCall::InboxAdd { source, instruction } =>
            apply_inbox_add_call(conn, options, &source, &instruction),
        ToolCall::RegisterInterest { topic } =>
            apply_register_interest_call(conn, options, &topic),
        ToolCall::SetPingPolicy { action, topic, until } =>
            apply_set_ping_policy_call(conn, options, &action, &topic, &until),
    }
}

fn apply_set_ping_policy_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    action: &str,
    topic: &str,
    until: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !interest_seat_allowed(&options.actor.actor_id) {
        return Ok("set_ping_policy denied: set_ping_policy is a scribe or heavy tool".to_string());
    }
    match action {
        "mute" => db::set_ping_mute(conn, until)?,
        "resume" => db::resume_pings(conn)?,
        "divert" => db::divert_ping_topic(conn, topic, until)?,
        _ => return Ok("set_ping_policy denied: unknown action".to_string()),
    }
    Ok(format!("set_ping_policy: {action}"))
}

fn apply_register_interest_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    topic: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !interest_seat_allowed(&options.actor.actor_id) {
        return Ok(
            "register_interest denied: register_interest is a scribe or heavy tool".to_string(),
        );
    }
    let interest_id = db::insert_interest(conn, topic, "user")?;
    runlog::log("interest", &format!("registered: {} (user)", topic.trim()));
    Ok(format!("register_interest: {interest_id}"))
}

fn interest_seat_allowed(actor_id: &str) -> bool {
    matches!(actor_id, "scribe" | "extract" | "consolidate")
}

fn apply_inbox_add_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    source: &str,
    instruction: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !matches!(
        options.actor.actor_id.as_str(),
        "scribe" | "extract" | "consolidate"
    ) {
        return Ok("inbox_add denied: inbox_add is a chat memory tool".to_string());
    }
    let is_url = source.starts_with("http://") || source.starts_with("https://");
    if source.is_empty() || (!is_url && !std::path::Path::new(source).exists()) {
        return Ok("inbox_add denied: source is neither a URL nor an existing file".to_string());
    }
    // Apply-side dedup: the looped passes forget prompt-side "one per source"
    // across iterations (observed live: the scribe queued the same file twice
    // in one pass). A queued duplicate is a no-op, not an error.
    if db::inbox_source_is_queued(conn, source)? {
        return Ok(format!("inbox_add: {source} is already queued"));
    }
    let item_id = db::insert_inbox_item(conn, source, instruction)?;
    crate::runlog::log("inbox", &format!("queued: {source}"));
    Ok(format!("inbox_add: queued item {item_id}"))
}

struct RecordMoveCall {
    move_text: String,
    actor: String,
}

// record_move is a scribe tool: only the episodic scribe pass may append to a
// task chain. The heavy pass and dream deny it, mirroring gist_edge's
// dream-only apply. The phase travels in options.actor.actor_id.
fn apply_record_move_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    call: RecordMoveCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if options.actor.actor_id != "scribe" {
        return Ok("record_move denied: record_move is a scribe tool".to_string());
    }
    let actor = normalize_move_actor(&call.actor);
    let session_id = options.context.session_id.clone();
    let hub_id = normalize_id(&format!("task-{session_id}"));
    db::ensure_task_hub(conn, &hub_id)?;
    let ply = db::next_ply(conn, &hub_id)?;
    let move_id = format!("{hub_id}-ply-{ply:03}");
    let today = crate::prompts::today_iso();
    apply_upsert_node(
        conn,
        move_id.clone(),
        "move".to_string(),
        call.move_text.clone(),
        String::new(),
    )?;
    let edge = Edge {
        source: hub_id,
        target: move_id,
        content: format!("ply {ply} ({actor}): {}", call.move_text),
        stability: 0.3,
        confidence: 1.0,
        importance: 1.0,
        subject: false,
        date: today,
        occurred_at: None,
        expired_at: None,
        memory_type: "episodic".to_string(),
        origin_session: session_id,
        provenance: options.provenance.to_string(),
    };
    apply_upsert_edge(conn, edge, options, events)?;
    Ok(format!(
        "record_move: ply {ply} ({actor}) {}",
        call.move_text
    ))
}

// A game turn has exactly two sides: user = the opponent's move, assistant =
// our own. Anything else is a scribe slip — record it as the opponent's move
// rather than dropping it, and leave a trace.
fn normalize_move_actor(raw: &str) -> String {
    match raw {
        "assistant" => "assistant".to_string(),
        "user" => "user".to_string(),
        other => {
            runlog::log(
                "scribe",
                &format!("record_move: unknown actor '{other}' mapped to user"),
            );
            "user".to_string()
        }
    }
}

struct DefineTaskCall {
    name: String,
    description: String,
}

// define_task is a scribe tool: setup turns are task turns. It upserts the
// durable game-definition node (shared across games) and links this session's
// hub to it with a per-session episodic edge. Re-defining keeps the longer
// body so a terse re-mention never loses detail.
fn apply_define_task_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    call: DefineTaskCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if options.actor.actor_id != "scribe" {
        return Ok("define_task denied: define_task is a scribe tool".to_string());
    }
    let game_id = normalize_id(&format!("game-{}", call.name));
    let name = super::task_chain::game_display_name(&game_id);
    let existing = db::load_node(conn, &game_id)?;
    let body = longer_body(existing.as_ref(), &call.description);
    apply_upsert_node(
        conn,
        game_id.clone(),
        "game".to_string(),
        body,
        crate::prompts::today_iso(),
    )?;
    let session_id = options.context.session_id.clone();
    let hub_id = normalize_id(&format!("task-{session_id}"));
    db::ensure_task_hub(conn, &hub_id)?;
    let edge = Edge {
        source: hub_id,
        target: game_id,
        content: format!("this session plays {name}"),
        stability: 0.3,
        confidence: 1.0,
        importance: 1.0,
        subject: false,
        date: crate::prompts::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "episodic".to_string(),
        origin_session: session_id,
        provenance: options.provenance.to_string(),
    };
    apply_upsert_edge(conn, edge, options, events)?;
    Ok(format!("define_task: {name}"))
}

// Keep the longer of the stored and incoming descriptions — cheap idempotence
// so a repeated terse mention never overwrites a rich definition.
fn longer_body(existing: Option<&Node>, description: &str) -> String {
    match existing {
        Some(node) if node.body.chars().count() >= description.chars().count() => node.body.clone(),
        _ => description.to_string(),
    }
}

// record_rule is scribe-or-heavy: a rules turn classifies as chat by design, so
// extract must be able to anchor rules too. Writes a durable (0.9/0.9) semantic
// fact anchored to the game node. No game defined yet → deny.
fn apply_record_rule_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    rule: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !rule_seat_allowed(&options.actor.actor_id) {
        return Ok("record_rule denied: record_rule is a scribe or heavy tool".to_string());
    }
    let session_id = options.context.session_id.clone();
    let hub_id = normalize_id(&format!("task-{session_id}"));
    let Some((game_id, _)) = db::load_session_game(conn, &hub_id)? else {
        return Ok("record_rule denied: define_task first".to_string());
    };
    let rule_id = normalize_id(&rule);
    apply_upsert_node(
        conn,
        rule_id.clone(),
        "fact".to_string(),
        rule.clone(),
        String::new(),
    )?;
    let edge = Edge {
        source: game_id,
        target: rule_id,
        content: rule.clone(),
        stability: 0.9,
        confidence: 0.9,
        importance: 1.0,
        subject: false,
        date: crate::prompts::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: options.provenance.to_string(),
    };
    apply_upsert_edge(conn, edge, options, events)?;
    Ok(format!("record_rule: {rule}"))
}

// scribe records moves and rules; the heavy pass (extract/consolidate) may
// anchor rules stated in chat. Ingest and dream may not.
fn rule_seat_allowed(actor_id: &str) -> bool {
    matches!(actor_id, "scribe" | "extract" | "consolidate")
}

struct RecordOutcomeCall {
    result: String,
    comment: String,
}

// record_outcome is a scribe tool: the game ended this turn. Scores the session
// from the assistant's perspective and runs reinforcement inside the scribe
// pass's open transaction (see record_outcome_in_tx for why not the public path).
fn apply_record_outcome_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    call: RecordOutcomeCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !rule_seat_allowed(&options.actor.actor_id) {
        return Ok("record_outcome denied: record_outcome is a scribe or heavy tool".to_string());
    }
    let session_id = options.context.session_id.clone();
    let hub_id = normalize_id(&format!("task-{session_id}"));
    let game = match db::load_session_game(conn, &hub_id)? {
        Some((game_id, _)) => super::task_chain::game_display_name(&game_id),
        None => String::new(),
    };
    let score = outcome_score(&call.result);
    let payload = serde_json::json!({
        "kind": "game",
        "game": game,
        "comment": call.comment,
    })
    .to_string();
    let summary = super::reinforce::record_outcome_in_tx(conn, &session_id, score, &payload)?;
    runlog::log("scribe", &summary);
    Ok(format!("record_outcome: {} ({game})", call.result))
}

// From the assistant's (learner's) perspective: win +1, draw 0, loss -1.
fn outcome_score(result: &str) -> f64 {
    match result {
        "win" => 1.0,
        "loss" => -1.0,
        _ => 0.0,
    }
}

fn apply_record_reaction_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    reaction: &str,
    comment: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !rule_seat_allowed(&options.actor.actor_id) {
        return Ok("record_reaction denied: record_reaction is a scribe or heavy tool".to_string());
    }
    let session_id = &options.context.session_id;
    let turn_id = &options.context.turn_id;
    let Some(target_turn) = previous_turn(conn, session_id, turn_id)? else {
        return Ok("record_reaction denied: no previous turn in this session".to_string());
    };
    let score = reaction_score(reaction);
    let surfaced = turn_was_surfaced(conn, session_id, &target_turn)?;
    let payload = serde_json::json!({
        "kind":"reaction",
        "reaction":reaction,
        "comment":comment,
        "surfaced":surfaced
    })
    .to_string();
    let summary = super::reinforce::record_turn_outcome_in_tx(
        conn,
        session_id,
        &target_turn,
        score,
        &payload,
    )?;
    runlog::log("reaction", &summary);
    Ok(format!("record_reaction: {reaction} ({target_turn})"))
}

fn turn_was_surfaced(
    conn: &Connection,
    session_id: &str,
    turn_id: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    db::block_on(async {
        let mut rows = conn
            .query(
                "SELECT 1 FROM events WHERE session_id = ?1 AND turn_id = ?2
                 AND event_type = 'surfaced' LIMIT 1",
                turso::params![session_id, turn_id],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    })
}

fn previous_turn(
    conn: &Connection,
    session_id: &str,
    current_turn: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    db::block_on(async {
        let mut rows = conn
            .query(
                "SELECT turn_id FROM events WHERE session_id = ?1 AND turn_id != '' AND turn_id != ?2 ORDER BY id DESC LIMIT 1",
                turso::params![session_id, current_turn],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        Ok(Some(db::text(row.get_value(0)?)))
    })
}

fn reaction_score(reaction: &str) -> f64 {
    match reaction {
        "positive" => POSITIVE_REACTION_SCORE,
        _ => NEGATIVE_REACTION_SCORE,
    }
}

#[allow(clippy::too_many_lines)]
pub(super) fn tool_call_touches_loop(
    conn: &Connection,
    call: &ToolCall,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    match call {
        ToolCall::UpsertNode {
            name, node_type, ..
        } => Ok(node_type == "loop" || node_is_loop(conn, name)?),
        ToolCall::DeleteNode { name } => node_or_neighbor_is_loop(conn, name),
        ToolCall::UpsertEdge { source, target, .. }
        | ToolCall::DeleteEdge { source, target }
        | ToolCall::ExpireEdge { source, target, .. }
        | ToolCall::PromoteEdge { source, target }
        | ToolCall::GistEdge { source, target, .. } => edge_touches_loop(conn, source, target),
        ToolCall::RecordMove { .. }
        | ToolCall::DefineTask { .. }
        | ToolCall::RecordRule { .. }
        | ToolCall::RecordOutcome { .. }
        | ToolCall::RecordReaction { .. }
        | ToolCall::OpenLoop { .. }
        | ToolCall::CloseLoop { .. }
        | ToolCall::InboxAdd { .. }
        | ToolCall::RegisterInterest { .. }
        | ToolCall::SetPingPolicy { .. } => Ok(false),
    }
}

fn edge_touches_loop(
    conn: &Connection,
    source: &str,
    target: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    Ok(node_is_loop(conn, source)? || node_is_loop(conn, target)?)
}

fn node_is_loop(conn: &Connection, node_id: &str) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(node_id);
    let Some(node) = db::load_node(conn, &normalized)? else {
        return Ok(false);
    };
    Ok(node.node_type == "loop")
}

fn node_or_neighbor_is_loop(
    conn: &Connection,
    node_id: &str,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    if node_is_loop(conn, node_id)? {
        return Ok(true);
    }
    let normalized = normalize_id(node_id);
    let edges = db::load_edges_for_node(conn, &normalized)?;
    for edge in edges {
        if edge_touches_loop(conn, &edge.source, &edge.target)? {
            return Ok(true);
        }
    }
    Ok(false)
}

struct OpenLoopCall {
    intention: String,
    anchor: String,
    due: String,
}

fn apply_open_loop_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    call: OpenLoopCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !loop_seat_allowed(&options.actor.actor_id) {
        return Ok("open_loop denied: open_loop is a heavy tool".to_string());
    }
    let anchor_id = normalize_id(&call.anchor);
    let Some(anchor_node) = db::load_node(conn, &anchor_id)? else {
        return Ok(format!("open_loop denied: anchor {anchor_id} not found"));
    };
    if anchor_node.node_type == "loop" {
        return Ok(format!("open_loop denied: anchor {anchor_id} is a loop"));
    }
    let loop_id = normalize_id(&call.intention);
    if let Some(reason) = open_loop_block_reason(conn, &loop_id, &anchor_id)? {
        return Ok(format!("open_loop denied: {reason}"));
    }
    let due = validated_loop_due(call.due);
    apply_upsert_node(
        conn,
        loop_id.clone(),
        "loop".to_string(),
        call.intention.clone(),
        due,
    )?;
    let edge = open_loop_edge(&loop_id, &anchor_id, &call.intention);
    apply_upsert_edge(conn, edge, options, events)?;
    Ok(format!("open_loop: {loop_id} → {anchor_id}"))
}

fn open_loop_block_reason(
    conn: &Connection,
    loop_id: &str,
    anchor_id: &str,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    if loop_id.is_empty() {
        return Ok(Some("intention has an empty id".to_string()));
    }
    if loop_id == anchor_id {
        return Ok(Some("loop and anchor resolve to the same id".to_string()));
    }
    if let Some(node) = db::load_node(conn, loop_id)?
        && node.node_type != "loop"
    {
        return Ok(Some(format!("id {loop_id} belongs to a non-loop node")));
    }
    for edge in live_loop_anchors(conn, loop_id)? {
        if edge.target != anchor_id {
            return Ok(Some(format!(
                "loop {loop_id} already has anchor {}",
                edge.target
            )));
        }
    }
    Ok(None)
}

fn open_loop_edge(loop_id: &str, anchor_id: &str, intention: &str) -> Edge {
    Edge {
        source: loop_id.to_string(),
        target: anchor_id.to_string(),
        content: format!("open loop: {intention}"),
        stability: 0.9,
        confidence: 0.9,
        importance: 1.0,
        subject: false,
        date: crate::prompts::today_iso(),
        occurred_at: None,
        expired_at: None,
        memory_type: "semantic".to_string(),
        origin_session: String::new(),
        provenance: PROVENANCE_USER_CONFIRMED.to_string(),
    }
}

fn validated_loop_due(due: String) -> String {
    if due.is_empty() || is_iso_date(&due) {
        return due;
    }
    runlog::log(
        "mem",
        &format!("open_loop: ignored invalid due date '{due}'"),
    );
    String::new()
}

fn is_iso_date(date: &str) -> bool {
    let bytes = date.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let mut idx = 0usize;
    while idx < bytes.len() {
        if !matches!(idx, 4 | 7) && !bytes[idx].is_ascii_digit() {
            return false;
        }
        idx += 1;
    }
    let Ok(year) = date[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = date[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = date[8..10].parse::<u32>() else {
        return false;
    };
    year > 0 && month > 0 && month <= 12 && day > 0 && day <= days_in_month(year, month)
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    }
}

fn is_leap_year(year: u32) -> bool {
    year.is_multiple_of(400) || (year.is_multiple_of(4) && !year.is_multiple_of(100))
}

struct CloseLoopCall {
    loop_id: String,
    resolution: String,
    note: String,
}

fn apply_close_loop_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    call: CloseLoopCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !loop_seat_allowed(&options.actor.actor_id) {
        return Ok("close_loop denied: close_loop is a heavy tool".to_string());
    }
    let loop_id = normalize_id(&call.loop_id);
    let Some(node) = db::load_node(conn, &loop_id)? else {
        return Ok(format!("close_loop denied: loop {loop_id} not found"));
    };
    if node.node_type != "loop" {
        return Ok(format!("close_loop denied: {loop_id} is not a loop"));
    }
    let anchors = live_loop_anchors(conn, &loop_id)?;
    if anchors.is_empty() {
        return Ok(format!("close_loop denied: loop {loop_id} is not open"));
    }
    let rewritten_content = format!(
        "{}: {} (closed {})",
        call.resolution,
        call.note,
        crate::prompts::today_iso()
    );
    for edge in anchors {
        apply_expire_edge(
            conn,
            ExpireEdgeCall {
                source: edge.source,
                target: edge.target,
                rewritten_content: rewritten_content.clone(),
            },
            options,
            events,
        )?;
    }
    Ok(format!("close_loop: {loop_id} ({})", call.resolution))
}

fn live_loop_anchors(
    conn: &Connection,
    loop_id: &str,
) -> Result<Vec<Edge>, Box<dyn Error + Send + Sync>> {
    let edges = db::load_edges_for_node(conn, loop_id)?;
    let mut anchors = Vec::new();
    for edge in edges {
        if edge.source == loop_id
            && edge.expired_at.is_none()
            && edge.memory_type == "semantic"
            && edge.content.starts_with("open loop:")
        {
            anchors.push(edge);
        }
    }
    Ok(anchors)
}

// Scribe included deliberately: intention statements ("remind me if I
// don't…") legitimately classify as task turns — same seat-symmetry lesson as
// record_outcome. Dream and ingest stay denied: they cannot own intentions.
fn loop_seat_allowed(actor_id: &str) -> bool {
    matches!(actor_id, "scribe" | "extract" | "consolidate")
}

struct ExpireEdgeCall {
    source: String,
    target: String,
    rewritten_content: String,
}

fn apply_expire_edge<'scope>(
    conn: &Connection,
    call: ExpireEdgeCall,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let ExpireEdgeCall {
        source,
        target,
        rewritten_content,
    } = call;
    let normalized_source = normalize_id(&source);
    let normalized_target = normalize_id(&target);
    let Some(mut edge) = db::load_edge(conn, &normalized_source, &normalized_target)? else {
        return Ok(format!(
            "expire_edge skipped: {source} → {target} (not found)"
        ));
    };
    edge.content = rewritten_content;
    edge.expired_at = Some(crate::timeutil::timestamp_now());
    db::write_edge(conn, &edge, options.source_message_id)?;
    let payload = if options.event_payload.is_empty() {
        serde_json::json!({"expired": true}).to_string()
    } else {
        options.event_payload.to_string()
    };
    events.push(Event::updated(
        options.actor.clone(),
        EdgeRef::new(&edge.source, &edge.target),
        options.context.clone(),
        payload,
    ));
    Ok(format!("expire_edge: {source} → {target}"))
}

struct UpsertEdgeCall {
    source: String,
    target: String,
    content: String,
    stability: f64,
    confidence: f64,
    importance: f64,
    subject: bool,
    date: String,
    memory_type: Option<String>,
}

fn apply_upsert_edge_call<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    mut call: UpsertEdgeCall,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(confidence_cap) = options.confidence_cap {
        call.confidence = call.confidence.min(confidence_cap);
    }
    let memory_type = match call.memory_type {
        Some(requested) => super::graph::normalize_memory_type(&requested),
        None => options.default_memory_type.to_string(),
    };
    let origin_session = if memory_type == "episodic" {
        options.context.session_id.clone()
    } else {
        String::new()
    };
    let edge = Edge {
        source: normalize_id(&call.source),
        target: normalize_id(&call.target),
        content: call.content,
        stability: call.stability,
        confidence: call.confidence,
        importance: call.importance,
        subject: call.subject,
        date: call.date,
        occurred_at: None,
        expired_at: None,
        memory_type,
        origin_session,
        provenance: options.provenance.to_string(),
    };
    if edge.memory_type == "episodic" {
        runlog::log(
            "mem",
            &format!(
                "episodic edge {} → {} (session {})",
                edge.source, edge.target, edge.origin_session
            ),
        );
    }
    apply_upsert_edge(conn, edge, options, events)
}

fn apply_upsert_node(
    conn: &Connection,
    name: String,
    node_type: String,
    body: String,
    date: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let node = Node {
        node_type: node_type.clone(),
        body,
        date,
    };
    db::write_node(conn, &normalized, &node)?;
    Ok(format!("upsert_node: {name} ({node_type})"))
}

fn apply_upsert_edge<'scope>(
    conn: &Connection,
    mut edge: Edge,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let existing = db::load_edge(conn, &edge.source, &edge.target)?;
    edge.provenance = match &existing {
        Some(stored) => stored.provenance.clone(),
        None => options.provenance.to_string(),
    };
    db::write_edge(conn, &edge, options.source_message_id)?;
    let event = if options.event_payload.is_empty() {
        Event::upsert(
            options.actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            options.context.clone(),
            existing.is_some(),
        )
    } else {
        Event::upsert_with_payload(
            options.actor.clone(),
            EdgeRef::new(&edge.source, &edge.target),
            options.context.clone(),
            existing.is_some(),
            options.event_payload.to_string(),
        )
    };
    events.push(event);
    Ok(format!("upsert_edge: {} → {}", edge.source, edge.target))
}

// Dreaming's promotion path: an episodic edge that kept being reinforced or
// that synthesis built on becomes a durable belief. Flips memory_type,
// clears origin_session (it is no longer session-scoped), and refreshes
// date so semantic decay restarts from the promotion point. Reused by both
// the chat/explore/ingest tool-call path and dream's consolidation agent.
pub(crate) fn apply_promote_edge(
    conn: &Connection,
    edge_ref: EdgeRef,
    actor: Actor,
    context: EventContext,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let EdgeRef { source, target } = edge_ref;
    let normalized_source = normalize_id(&source);
    let normalized_target = normalize_id(&target);
    let Some(mut edge) = db::load_edge(conn, &normalized_source, &normalized_target)? else {
        return Ok(format!(
            "promote_edge skipped: {source} → {target} (not found)"
        ));
    };
    edge.memory_type = "semantic".to_string();
    edge.origin_session = String::new();
    edge.date = crate::prompts::today_iso();
    db::write_edge(conn, &edge, "")?;
    events.push(Event::updated(
        actor,
        EdgeRef::new(&edge.source, &edge.target),
        context,
        serde_json::json!({"promoted_from": "episodic"}).to_string(),
    ));
    runlog::log(
        "mem",
        &format!("promoted {} → {} to semantic", edge.source, edge.target),
    );
    Ok(format!("promote_edge: {} → {}", edge.source, edge.target))
}

fn apply_delete_node<'scope>(
    conn: &Connection,
    name: String,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let deleted_edges = db::load_edges_for_node(conn, &normalized)?;
    db::erase_node(conn, &normalized)?;
    let pruned = ledger::pruned_events(
        options.actor.clone(),
        options.context.clone(),
        &deleted_edges,
    )?;
    events.extend(pruned);
    Ok(format!("delete_node: {name}"))
}

fn apply_delete_edge<'scope>(
    conn: &Connection,
    edge_ref: EdgeRef,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized_source = normalize_id(&edge_ref.source);
    let normalized_target = normalize_id(&edge_ref.target);
    let deleted_edge = db::load_edge(conn, &normalized_source, &normalized_target)?;
    db::erase_edge(conn, &normalized_source, &normalized_target)?;
    if let Some(edge) = deleted_edge {
        let pruned =
            ledger::pruned_events(options.actor.clone(), options.context.clone(), &[edge])?;
        events.extend(pruned);
    }
    Ok(format!(
        "delete_edge: {} → {}",
        edge_ref.source, edge_ref.target
    ))
}

fn mutation_context(source_message_id: &str) -> EventContext {
    let session_id = db::source_session_id(source_message_id).unwrap_or_default();
    EventContext::new(session_id.as_str(), source_message_id)
}

#[cfg(test)]
mod tests {
    use super::{
        AgentRun, IterationOutcome, MAX_PASS_ITERATIONS, MemoryEvent, ToolCall, TurnMode,
        parse_tool_calls, parse_turn_mode, says_done, stream_collect,
    };
    use crate::api::{Chunk, LlmClient, Stats, TurnOutcome};
    use std::error::Error;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender};

    // Replays one canned response per stream() call; the loop under test never
    // reaches the DB, so only the LLM half of an iteration is exercised.
    struct ScriptedClient {
        responses: Mutex<Vec<String>>,
    }

    impl ScriptedClient {
        fn scripted(responses: &[&str]) -> Arc<dyn LlmClient> {
            let mut queued: Vec<String> = Vec::new();
            for text in responses.iter().rev() {
                queued.push((*text).to_string());
            }
            Arc::new(ScriptedClient {
                responses: Mutex::new(queued),
            })
        }
    }

    impl LlmClient for ScriptedClient {
        fn collect(&self, _prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }

        fn stream(&self, _prompt: String, _system: Option<String>, sender: Sender<Chunk>) {
            let Ok(mut queued) = self.responses.lock() else {
                return;
            };
            // Running dry means the loop asked for more turns than scripted;
            // an empty reply reads as "no tool calls" and ends the pass.
            let text = queued.pop().unwrap_or_default();
            let _ = sender.send(Chunk::Content(text));
            let _ = sender.send(Chunk::Done(Stats {
                tokens_in: 0,
                tokens_out: 0,
                cost_usd: None,
                ttft_ms: 0,
            }));
        }

        fn stream_turn(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _sender: &Sender<Chunk>,
        ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
            unimplemented!("scripted client streams only")
        }
    }

    const ONE_CALL: &str =
        r#"{"tools":[{"tool":"delete_edge","args":{"source":"a","target":"b"}}]}"#;

    // Counts iterations by driving drive_pass with an iterate closure that does
    // the real LLM + parse work but stubs out the DB write.
    fn count_iterations(responses: &[&str]) -> usize {
        let (sender, receiver): (Sender<MemoryEvent>, Receiver<MemoryEvent>) =
            std::sync::mpsc::channel();
        let llm = ScriptedClient::scripted(responses);
        let agent_run = AgentRun {
            seats: super::MemorySeats::uniform(llm),
            user_msg: "user",
            assistant_msg: "assistant",
            sender: &sender,
            user_message_id: "msg-1",
        };
        let mut iterations = 0usize;
        let pass = super::heavy_pass("extract");
        let result = agent_run.drive_pass(&pass, "first prompt".to_string(), |_index, prompt| {
            iterations += 1;
            let call_text = stream_collect(Arc::clone(&agent_run.seats.heavy), prompt, drop, drop)?;
            let tool_calls = parse_tool_calls(&call_text);
            let done = tool_calls.is_empty() || says_done(&call_text);
            let mut descriptions = Vec::new();
            for _call in &tool_calls {
                descriptions.push("applied".to_string());
            }
            Ok(IterationOutcome { descriptions, done })
        });
        assert!(result.is_ok());
        drop(receiver);
        iterations
    }

    // Same as count_iterations but drives the scribe pass, whose tighter cap
    // (MAX_SCRIBE_ITERATIONS) is what these tests exercise.
    fn count_scribe_iterations(responses: &[&str]) -> usize {
        let (sender, receiver): (Sender<MemoryEvent>, Receiver<MemoryEvent>) =
            std::sync::mpsc::channel();
        let llm = ScriptedClient::scripted(responses);
        let agent_run = AgentRun {
            seats: super::MemorySeats::uniform(llm),
            user_msg: "user",
            assistant_msg: "assistant",
            sender: &sender,
            user_message_id: "msg-1",
        };
        let mut iterations = 0usize;
        let pass = super::scribe_pass();
        let result = agent_run.drive_pass(&pass, "first prompt".to_string(), |_index, prompt| {
            iterations += 1;
            let call_text =
                stream_collect(Arc::clone(&agent_run.seats.scribe), prompt, drop, drop)?;
            let tool_calls = parse_tool_calls(&call_text);
            let done = tool_calls.is_empty() || says_done(&call_text);
            let mut descriptions = Vec::new();
            for _call in &tool_calls {
                descriptions.push("recorded".to_string());
            }
            Ok(IterationOutcome { descriptions, done })
        });
        assert!(result.is_ok());
        drop(receiver);
        iterations
    }

    #[test]
    fn scribe_continues_once_to_catch_an_under_emitted_turn() {
        // First response records something, the continuation says DONE: two
        // iterations, exactly the recovery the cap is meant to allow.
        assert_eq!(count_scribe_iterations(&[ONE_CALL, "DONE"]), 2);
    }

    #[test]
    fn scribe_stops_at_its_two_iteration_cap() {
        // The model keeps emitting calls and never says DONE — the tighter
        // scribe cap halts it at two, not the heavy MAX_PASS_ITERATIONS.
        let responses = vec![ONE_CALL; MAX_PASS_ITERATIONS];
        assert_eq!(count_scribe_iterations(&responses), 2);
    }

    #[test]
    fn scribe_single_shot_stops_at_one_iteration() {
        // A setup turn the first response fully captures: DONE on the first
        // pass ends it immediately, no continuation latency.
        let done = format!("{ONE_CALL}\nDONE");
        assert_eq!(count_scribe_iterations(&[&done]), 1);
        // Zero tool calls also reads as finished on the first iteration.
        assert_eq!(count_scribe_iterations(&["nothing to record"]), 1);
    }

    #[test]
    fn pass_stops_when_the_model_reports_done() {
        assert_eq!(count_iterations(&[ONE_CALL, "DONE"]), 2);
    }

    #[test]
    fn pass_stops_immediately_when_no_tool_calls_are_emitted() {
        assert_eq!(count_iterations(&["nothing to record here"]), 1);
    }

    #[test]
    fn pass_stops_at_the_iteration_cap_when_the_model_never_finishes() {
        let responses = vec![ONE_CALL; MAX_PASS_ITERATIONS + 2];
        assert_eq!(count_iterations(&responses), MAX_PASS_ITERATIONS);
    }

    #[test]
    fn done_detection_tolerates_prose_and_casing() {
        assert!(says_done("done"));
        assert!(says_done("  Done.\n"));
        assert!(says_done("Nothing else remains — DONE"));
        assert!(!says_done(ONE_CALL));
        assert!(!says_done("the plan was abandoned, work undone"));
    }

    #[test]
    fn a_response_with_both_tool_calls_and_done_applies_then_stops() {
        let mixed = format!("{ONE_CALL}\nDONE");
        let calls = parse_tool_calls(&mixed);
        assert_eq!(calls.len(), 1);
        assert!(says_done(&mixed));
        assert_eq!(count_iterations(&[&mixed]), 1);
    }

    #[test]
    fn turn_mode_parses_task_and_chat_and_defaults_to_conversation() {
        assert_eq!(parse_turn_mode("TASK"), TurnMode::TaskMove);
        assert_eq!(parse_turn_mode("task"), TurnMode::TaskMove);
        assert_eq!(parse_turn_mode("CHAT"), TurnMode::Conversation);
        assert_eq!(parse_turn_mode("garbage nonsense"), TurnMode::Conversation);
        assert_eq!(parse_turn_mode(""), TurnMode::Conversation);
        // Ambiguous replies mentioning both bias to the safe heavy path.
        assert_eq!(
            parse_turn_mode("this is chat, not a task"),
            TurnMode::Conversation
        );
    }

    #[test]
    fn truncated_tool_array_salvages_complete_objects() {
        let truncated = r#"{"tools":[{"tool":"upsert_node","args":{"name":"a","type":"entity","body":"","date":""}},{"tool":"upsert_edge","args":{"source":"a","target":"b","content":"a relates to b","stability":0.5,"confidence":0.5,"importance":0.5,"subject":false,"date":"2026-07-06"}},{"tool":"upsert_node","args":{"name":"trunc"#;
        let calls = parse_tool_calls(truncated);
        assert_eq!(calls.len(), 2);
        assert!(matches!(calls[0], ToolCall::UpsertNode { .. }));
        assert!(matches!(calls[1], ToolCall::UpsertEdge { .. }));
    }

    #[test]
    fn gist_edge_parses_full_args_and_both_ref_spellings() {
        let arrow = r#"{"tools":[{"tool":"gist_edge","args":{"source":"early-queen-sorties","target":"chess-openings","content":"queen sorties get punished","confidence":0.6,"stability":0.4,"supported_by":["game3-hub → move-qh5","game4-hub->move-qf3"]}}]}"#;
        let calls = parse_tool_calls(arrow);
        assert_eq!(calls.len(), 1);
        let ToolCall::GistEdge {
            source,
            supported_by,
            confidence,
            stability,
            ..
        } = &calls[0]
        else {
            unreachable!("parsed as gist_edge")
        };
        assert_eq!(source, "early-queen-sorties");
        assert!((*confidence - 0.6).abs() < 1e-9);
        assert!((*stability - 0.4).abs() < 1e-9);
        // Both "a → b" and "a->b" normalize to the same (source, target) refs.
        assert_eq!(
            supported_by,
            &vec![
                ("game3-hub".to_string(), "move-qh5".to_string()),
                ("game4-hub".to_string(), "move-qf3".to_string()),
            ]
        );
    }

    #[test]
    fn gist_edge_without_supported_by_is_rejected() {
        let missing = r#"{"tool":"gist_edge","args":{"source":"a","target":"b","content":"c"}}"#;
        assert!(parse_tool_calls(missing).is_empty());
        let not_array = r#"{"tool":"gist_edge","args":{"source":"a","target":"b","content":"c","supported_by":"x → y"}}"#;
        assert!(parse_tool_calls(not_array).is_empty());
    }

    #[test]
    fn gist_edge_stability_defaults_to_half_when_absent() {
        let no_stability = r#"{"tool":"gist_edge","args":{"source":"a","target":"b","content":"c","supported_by":["x → y","p → q"]}}"#;
        let calls = parse_tool_calls(no_stability);
        assert_eq!(calls.len(), 1);
        let ToolCall::GistEdge { stability, .. } = &calls[0] else {
            unreachable!("parsed as gist_edge")
        };
        assert!((*stability - 0.5).abs() < 1e-9);
    }

    #[test]
    fn prose_wrapped_json_is_recovered() {
        let wrapped = r#"Here are the extracted facts: {"tools":[{"tool":"delete_edge","args":{"source":"a","target":"b"}}]} done."#;
        let calls = parse_tool_calls(wrapped);
        assert_eq!(calls.len(), 1);
    }

    mod record_move {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', PRIMARY KEY (source, target))";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            conn
        }

        fn options_for_phase(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("game-1", "game-1:user:ts"),
                source_message_id: "game-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn record(move_text: &str, actor: &str) -> ToolCall {
            ToolCall::RecordMove {
                move_text: move_text.to_string(),
                actor: actor.to_string(),
            }
        }

        fn apply<'opt>(
            conn: &turso::Connection,
            options: &ApplyToolOptions<'opt>,
            call: ToolCall,
        ) -> String {
            let mut events: Vec<Event> = Vec::new();
            match apply_tool_call(call, conn, options, &mut events) {
                Ok(description) => description,
                Err(err) => panic!("apply succeeds: {err}"),
            }
        }

        fn chain(conn: &turso::Connection) -> Vec<crate::memory::graph::Edge> {
            let Ok(chain) = db::load_task_chain(conn, "game-1", 240) else {
                unreachable!("chain loads")
            };
            chain
        }

        #[test]
        fn valid_args_parse_and_missing_move_is_rejected() {
            let calls = parse_tool_calls(
                r#"{"tool":"record_move","args":{"move":"Nc3","actor":"assistant"}}"#,
            );
            assert_eq!(calls.len(), 1);
            assert!(matches!(calls[0], ToolCall::RecordMove { .. }));
            assert!(
                parse_tool_calls(r#"{"tool":"record_move","args":{"actor":"user"}}"#).is_empty()
            );
        }

        // Two moves applied as two separate calls: the second call reads the ply
        // counter back from the DB (simulated restart), so it appends rather than
        // restarting at 1.
        fn two_move_chain(conn: &turso::Connection) -> Vec<crate::memory::graph::Edge> {
            let options = options_for_phase("scribe");
            let first = apply(conn, &options, record("e4", "assistant"));
            assert!(first.contains("ply 1"), "{first}");
            let second = apply(conn, &options, record("e5", "user"));
            assert!(second.contains("ply 2"), "{second}");
            chain(conn)
        }

        fn targets(chain: &[crate::memory::graph::Edge]) -> Vec<String> {
            let mut ids = Vec::new();
            for edge in chain {
                ids.push(edge.target.clone());
            }
            ids
        }

        fn contents(chain: &[crate::memory::graph::Edge]) -> Vec<String> {
            let mut lines = Vec::new();
            for edge in chain {
                lines.push(edge.content.clone());
            }
            lines
        }

        #[test]
        fn scribe_numbers_plies_in_order_from_the_db() {
            let conn = db::block_on(harness());
            let chain = two_move_chain(&conn);
            assert_eq!(
                targets(&chain),
                vec!["task-game-1-ply-001", "task-game-1-ply-002"]
            );
        }

        #[test]
        fn scribe_writes_episodic_edges_with_actor_content() {
            let conn = db::block_on(harness());
            let chain = two_move_chain(&conn);
            assert_eq!(
                contents(&chain),
                vec!["ply 1 (assistant): e4", "ply 2 (user): e5"]
            );
            assert_eq!(chain[0].memory_type, "episodic");
            assert_eq!(chain[0].origin_session, "game-1");
        }

        #[test]
        fn unknown_actor_maps_to_user_without_panicking() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(&conn, &options, record("Qh5", "white"));
            assert!(description.contains("(user)"), "{description}");
        }

        #[test]
        fn heavy_and_dream_paths_deny_the_tool() {
            let conn = db::block_on(harness());
            for phase in ["extract", "consolidate", "consolidation"] {
                let options = options_for_phase(phase);
                let description = apply(&conn, &options, record("e4", "assistant"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            assert!(chain(&conn).is_empty(), "denied calls write nothing");
        }
    }

    mod game {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', embedding BLOB, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', PRIMARY KEY (source, target))";
        const EVENTS_DDL: &str = "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')";
        const VIEWS_DDL: &str = "CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))";
        const POSITIONS_DDL: &str = "CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))";
        const POSITION_HISTORY_DDL: &str = "CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            let _ = conn.execute(EVENTS_DDL, ()).await;
            let _ = conn.execute(VIEWS_DDL, ()).await;
            let _ = conn.execute(POSITIONS_DDL, ()).await;
            let _ = conn.execute(POSITION_HISTORY_DDL, ()).await;
            conn
        }

        fn options_for_phase(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("game-1", "game-1:user:ts"),
                source_message_id: "game-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn apply<'opt>(
            conn: &turso::Connection,
            options: &ApplyToolOptions<'opt>,
            call: ToolCall,
        ) -> String {
            let mut events: Vec<Event> = Vec::new();
            match apply_tool_call(call, conn, options, &mut events) {
                Ok(description) => description,
                Err(err) => panic!("apply succeeds: {err}"),
            }
        }

        fn define(name: &str, description: &str) -> ToolCall {
            ToolCall::DefineTask {
                name: name.to_string(),
                description: description.to_string(),
            }
        }

        fn edge(
            conn: &turso::Connection,
            source: &str,
            target: &str,
        ) -> crate::memory::graph::Edge {
            let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
                unreachable!("edge {source} → {target} exists")
            };
            edge
        }

        #[test]
        fn define_task_creates_game_node_and_session_hub_link() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(
                &conn,
                &options,
                define("Chess", "user White, replies move only"),
            );
            assert!(description.contains("chess"), "{description}");
            let Ok(Some(node)) = db::load_node(&conn, "game-chess") else {
                unreachable!("game node exists")
            };
            assert_eq!(node.node_type, "game");
            assert_eq!(node.body, "user White, replies move only");
            let link = edge(&conn, "task-game-1", "game-chess");
            assert_eq!(link.content, "this session plays chess");
            assert_eq!(link.memory_type, "episodic");
            assert_eq!(link.origin_session, "game-1");
        }

        #[test]
        fn redefining_with_a_shorter_description_keeps_the_longer_body() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            apply(
                &conn,
                &options,
                define("chess", "the long, detailed rules of the game"),
            );
            apply(&conn, &options, define("chess", "chess"));
            let Ok(Some(node)) = db::load_node(&conn, "game-chess") else {
                unreachable!("game node exists")
            };
            assert_eq!(node.body, "the long, detailed rules of the game");
        }

        #[test]
        fn define_task_is_scribe_only() {
            let conn = db::block_on(harness());
            for phase in ["extract", "consolidate", "consolidation"] {
                let options = options_for_phase(phase);
                let description = apply(&conn, &options, define("chess", "rules"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            let Ok(missing) = db::load_node(&conn, "game-chess") else {
                unreachable!("load_node succeeds")
            };
            assert!(missing.is_none());
        }

        fn rule(text: &str) -> ToolCall {
            ToolCall::RecordRule {
                rule: text.to_string(),
            }
        }

        #[test]
        fn record_rule_writes_a_durable_semantic_fact_anchored_to_the_game() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            apply(&conn, &options, define("chess", "rules"));
            let description = apply(&conn, &options, rule("the king moves one square"));
            assert!(description.contains("record_rule"), "{description}");
            let rule_edge = edge(&conn, "game-chess", "the-king-moves-one-square");
            assert!((rule_edge.confidence - 0.9).abs() < 1e-9);
            assert!((rule_edge.stability - 0.9).abs() < 1e-9);
            assert_eq!(rule_edge.memory_type, "semantic");
        }

        #[test]
        fn record_rule_is_denied_without_a_defined_game() {
            let conn = db::block_on(harness());
            let options = options_for_phase("scribe");
            let description = apply(&conn, &options, rule("some rule"));
            assert!(description.contains("define_task first"), "{description}");
        }

        #[test]
        fn record_rule_is_allowed_from_scribe_and_heavy_but_denied_from_dream() {
            let conn = db::block_on(harness());
            apply(
                &conn,
                &options_for_phase("scribe"),
                define("chess", "rules"),
            );
            for phase in ["scribe", "extract", "consolidate"] {
                let description = apply(&conn, &options_for_phase(phase), rule("shared rule"));
                assert!(
                    description.contains("record_rule"),
                    "{phase}: {description}"
                );
            }
            let denied = apply(
                &conn,
                &options_for_phase("consolidation"),
                rule("shared rule"),
            );
            assert!(denied.contains("denied"), "{denied}");
        }

        fn outcome(result: &str) -> ToolCall {
            ToolCall::RecordOutcome {
                result: result.to_string(),
                comment: "the game ended".to_string(),
            }
        }

        fn reaction(value: &str) -> ToolCall {
            ToolCall::RecordReaction {
                reaction: value.to_string(),
                comment: "evaluated the previous answer".to_string(),
            }
        }

        async fn seed_reaction_edge(conn: &turso::Connection) {
            let _ = conn
                .execute("INSERT INTO nodes (id) VALUES ('a'), ('b')", ())
                .await;
            let _ = conn
                .execute(
                    "INSERT INTO edges (source, target, confidence, provenance)
                     VALUES ('a', 'b', 0.5, 'user-confirmed')",
                    (),
                )
                .await;
            let _ = conn.execute("INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, source, target) VALUES ('', 'game-1', 'turn-old', 'system', 'local', 'injected', 'a', 'b'), ('', 'game-1', 'turn-latest', 'system', 'local', 'injected', 'a', 'b')", ()).await;
        }

        fn outcome_turns(conn: &turso::Connection) -> Vec<String> {
            db::block_on(async {
                let Ok(mut rows) = conn
                    .query(
                        "SELECT turn_id FROM events WHERE event_type = 'outcome' ORDER BY id",
                        (),
                    )
                    .await
                else {
                    unreachable!("outcomes query")
                };
                let mut turns = Vec::new();
                while let Ok(Some(row)) = rows.next().await {
                    let Ok(value) = row.get_value(0) else {
                        unreachable!("turn value")
                    };
                    turns.push(db::text(value));
                }
                turns
            })
        }

        fn outcome_payloads(conn: &turso::Connection) -> Vec<String> {
            db::block_on(async {
                let Ok(mut rows) = conn
                    .query(
                        "SELECT payload FROM events WHERE event_type = 'outcome' ORDER BY id",
                        (),
                    )
                    .await
                else {
                    unreachable!("outcomes query")
                };
                let mut payloads = Vec::new();
                while let Ok(Some(row)) = rows.next().await {
                    let Ok(value) = row.get_value(0) else {
                        unreachable!("payload value")
                    };
                    payloads.push(db::text(value));
                }
                payloads
            })
        }

        #[test]
        fn reaction_parse_accepts_binary_values_only() {
            for value in ["positive", "negative"] {
                let input = format!(
                    r#"{{"tool":"record_reaction","args":{{"reaction":"{value}","comment":"clear evaluation"}}}}"#
                );
                assert!(matches!(
                    parse_tool_calls(&input)[0],
                    ToolCall::RecordReaction { .. }
                ));
            }
            assert!(
                parse_tool_calls(r#"{"tool":"record_reaction","args":{"reaction":"meh"}}"#)
                    .is_empty()
            );
        }

        #[test]
        fn reaction_targets_latest_prior_turn_and_is_seat_symmetric() {
            let conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&conn));
            for phase in ["scribe", "extract"] {
                let description = apply(&conn, &options_for_phase(phase), reaction("positive"));
                assert!(
                    description.contains("turn-latest"),
                    "{phase}: {description}"
                );
            }
            for phase in ["consolidation", "ingest"] {
                let description = apply(&conn, &options_for_phase(phase), reaction("positive"));
                assert!(description.contains("denied"), "{phase}: {description}");
            }
            assert_eq!(outcome_turns(&conn), vec!["turn-latest"]);
        }

        #[test]
        fn reaction_to_ping_is_stamped_surfaced_but_normal_reaction_is_not() {
            let ping_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&ping_conn));
            db::block_on(async {
                let _ = ping_conn
                    .execute(
                        r#"INSERT INTO events (ts, session_id, turn_id, actor_type, actor_id, event_type, payload)
                         VALUES ('', 'game-1', 'ping-1', 'system', 'ping', 'surfaced', '{"surfaced":true}')"#,
                        (),
                    )
                    .await;
            });
            apply(
                &ping_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(outcome_payloads(&ping_conn)[0].contains(r#""surfaced":true"#));

            let normal_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&normal_conn));
            apply(
                &normal_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(outcome_payloads(&normal_conn)[0].contains(r#""surfaced":false"#));
        }

        #[test]
        fn reaction_denies_without_prior_turn() {
            let conn = db::block_on(harness());
            let description = apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            assert_eq!(
                description,
                "record_reaction denied: no previous turn in this session"
            );
            assert!(outcome_turns(&conn).is_empty());
        }

        #[test]
        fn reaction_score_moves_confidence_and_session_outcome_coexists() {
            let positive_conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&positive_conn));
            let positive_before = edge(&positive_conn, "a", "b").confidence;
            apply(
                &positive_conn,
                &options_for_phase("scribe"),
                reaction("positive"),
            );
            assert!(edge(&positive_conn, "a", "b").confidence > positive_before);

            let conn = db::block_on(harness());
            db::block_on(seed_reaction_edge(&conn));
            let before = edge(&conn, "a", "b").confidence;
            apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            assert!(edge(&conn, "a", "b").confidence < before);
            apply(&conn, &options_for_phase("scribe"), reaction("negative"));
            apply(&conn, &options_for_phase("scribe"), outcome("draw"));
            assert_eq!(outcome_turns(&conn), vec!["turn-latest", ""]);
        }

        #[test]
        fn record_outcome_works_from_scribe_and_heavy_but_not_dream_or_ingest() {
            let conn = db::block_on(harness());
            apply(
                &conn,
                &options_for_phase("scribe"),
                define("chess", "rules"),
            );
            let description = apply(&conn, &options_for_phase("scribe"), outcome("loss"));
            assert!(
                description.contains("loss") && description.contains("chess"),
                "{description}"
            );
            // Game-end turns legitimately classify chat ("I resign — good
            // game" has no new move), so the heavy pass must be able to score.
            let from_extract = apply(&conn, &options_for_phase("extract"), outcome("win"));
            assert!(from_extract.contains("win"), "{from_extract}");
            let denied = apply(&conn, &options_for_phase("ingest"), outcome("win"));
            assert!(denied.contains("denied"), "{denied}");
        }

        #[test]
        fn invalid_outcome_result_is_parse_rejected() {
            assert!(
                parse_tool_calls(r#"{"tool":"record_outcome","args":{"result":"forfeit"}}"#)
                    .is_empty()
            );
            let calls = parse_tool_calls(r#"{"tool":"record_outcome","args":{"result":"draw"}}"#);
            assert_eq!(calls.len(), 1);
            assert!(matches!(calls[0], ToolCall::RecordOutcome { .. }));
        }

        #[test]
        fn outcome_score_maps_the_three_verdicts() {
            assert!((super::super::outcome_score("win") - 1.0).abs() < 1e-9);
            assert!((super::super::outcome_score("draw")).abs() < 1e-9);
            assert!((super::super::outcome_score("loss") + 1.0).abs() < 1e-9);
        }
    }

    mod inbox {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const INBOX_DDL: &str = "CREATE TABLE inbox (id INTEGER PRIMARY KEY AUTOINCREMENT, accepted_at TEXT NOT NULL, source TEXT NOT NULL, instruction TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'queued', resolved_at TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '')";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(INBOX_DDL, ()).await;
            conn
        }

        fn options(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "chat-1:user:ts"),
                source_message_id: "chat-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn add(conn: &turso::Connection, phase: &'static str, source: &str) -> String {
            let call = ToolCall::InboxAdd {
                source: source.to_string(),
                instruction: "focus on X".to_string(),
            };
            let mut events: Vec<Event> = Vec::new();
            let Ok(description) = apply_tool_call(call, conn, &options(phase), &mut events) else {
                unreachable!("apply succeeds")
            };
            description
        }

        #[test]
        fn inbox_add_queues_urls_from_scribe_and_heavy_seats() {
            let conn = db::block_on(harness());
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued"));
            assert!(add(&conn, "extract", "http://example.com/b").contains("queued"));
            let count = db::block_on(db::count_queued_inbox(&conn));
            assert_eq!(count, 2);
        }

        // Looped passes re-emit across iterations; the apply layer holds the line.
        #[test]
        fn inbox_add_dedups_a_source_already_queued() {
            let conn = db::block_on(harness());
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued item"));
            let second = add(&conn, "scribe", "https://example.com/a");
            assert!(second.contains("already queued"), "{second}");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 1);
            // A resolved source may be queued again (re-study is legitimate).
            let Ok(Some(item)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("item exists")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, item.item_id, "done", "ingest-1") else {
                unreachable!("resolve succeeds")
            };
            assert!(add(&conn, "scribe", "https://example.com/a").contains("queued item"));
        }

        #[test]
        fn inbox_add_denied_from_dream_and_ingest_seats() {
            let conn = db::block_on(harness());
            for phase in ["consolidation", "ingest"] {
                let result = add(&conn, phase, "https://example.com/a");
                assert!(result.contains("denied"), "{phase}: {result}");
            }
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 0);
        }

        #[test]
        fn inbox_add_rejects_a_source_that_is_neither_url_nor_file() {
            let conn = db::block_on(harness());
            let result = add(&conn, "extract", "definitely/not/a/real/path.txt");
            assert!(result.contains("denied"), "{result}");
            let empty = add(&conn, "extract", "");
            assert!(empty.contains("denied"), "{empty}");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 0);
        }

        #[test]
        fn queue_is_fifo_and_resolution_updates_status() {
            let conn = db::block_on(harness());
            let Ok(first) = db::insert_inbox_item(&conn, "https://a", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(_second) = db::insert_inbox_item(&conn, "https://b", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(Some(oldest)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("oldest exists")
            };
            assert_eq!(oldest.item_id, first);
            assert_eq!(oldest.source, "https://a");
            let Ok(()) = db::resolve_inbox_item(&conn, first, "done", "ingest-1") else {
                unreachable!("resolve succeeds")
            };
            let Ok(Some(next)) = db::oldest_queued_inbox(&conn) else {
                unreachable!("next exists")
            };
            assert_eq!(next.source, "https://b");
            assert_eq!(db::block_on(db::count_queued_inbox(&conn)), 1);
        }

        #[test]
        #[allow(clippy::too_many_lines)]
        fn outcomes_cover_window_resolutions_and_all_queued_items() {
            let conn = db::block_on(harness());
            let Ok(done_id) = db::insert_inbox_item(&conn, "https://done", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(failed_id) = db::insert_inbox_item(&conn, "https://failed", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(old_id) = db::insert_inbox_item(&conn, "https://old", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(_queued) = db::insert_inbox_item(&conn, "https://queued", "") else {
                unreachable!("insert succeeds")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, done_id, "done", "ingest-9") else {
                unreachable!("resolve succeeds")
            };
            let Ok(()) = db::resolve_inbox_item(&conn, failed_id, "failed", "HTTP 404") else {
                unreachable!("resolve succeeds")
            };
            // Resolved long before the window: must not re-report.
            let _ = db::block_on(conn.execute(
                "UPDATE inbox SET status = 'done', resolved_at = '2020-01-01T00:00:00', result = 'x' WHERE id = ?1",
                turso::params![old_id],
            ));
            let Ok(outcomes) =
                db::block_on(db::load_inbox_outcomes(&conn, "2026-01-01", "2099-01-01"))
            else {
                unreachable!("outcomes load")
            };
            assert!(
                outcomes.contains(&"https://done: done".to_string()),
                "{outcomes:?}"
            );
            assert!(
                outcomes.contains(&"https://failed: failed (HTTP 404)".to_string()),
                "{outcomes:?}"
            );
            assert!(
                outcomes.contains(&"https://queued: queued".to_string()),
                "{outcomes:?}"
            );
            for line in &outcomes {
                assert!(!line.starts_with("https://old"), "{outcomes:?}");
            }
        }
    }

    mod interests {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            conn
        }

        fn register(conn: &turso::Connection, phase: &'static str, topic: &str) -> String {
            let options = ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "turn-1"),
                source_message_id: "message-1",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "semantic",
            };
            let mut events: Vec<Event> = Vec::new();
            let call = ToolCall::RegisterInterest {
                topic: topic.to_string(),
            };
            apply_tool_call(call, conn, &options, &mut events).unwrap_or_default()
        }

        #[test]
        fn registration_is_seat_symmetric_and_idempotent() {
            let conn = db::block_on(harness());
            for phase in ["scribe", "extract", "consolidate"] {
                assert!(register(&conn, phase, "libsql internals").contains("libsql-internals"));
            }
            let interests = db::block_on(db::top_interests_with(&conn, 10)).unwrap_or_default();
            assert_eq!(interests.len(), 1);
            assert_eq!(interests[0].topic, "libsql internals");
            assert_eq!(interests[0].origin, "user");
        }

        #[test]
        fn registration_is_denied_from_dream_and_ingest() {
            let conn = db::block_on(harness());
            for phase in ["consolidation", "ingest"] {
                assert!(register(&conn, phase, "libsql internals").contains("denied"));
            }
            let interests = db::block_on(db::top_interests_with(&conn, 10)).unwrap_or_default();
            assert!(interests.is_empty());
        }

        #[test]
        fn register_interest_parses_only_non_empty_topics() {
            let calls = parse_tool_calls(
                r#"{"tool":"register_interest","args":{"topic":"libsql internals"}}"#,
            );
            assert!(matches!(
                calls.first(),
                Some(ToolCall::RegisterInterest { .. })
            ));
            assert!(
                parse_tool_calls(r#"{"tool":"register_interest","args":{"topic":""}}"#).is_empty()
            );
        }
    }

    mod ping_policy {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute("CREATE TABLE ping_policy (topic TEXT PRIMARY KEY, muted_until TEXT NOT NULL DEFAULT '', divert_until TEXT NOT NULL DEFAULT '')", ()).await;
            conn
        }

        fn apply_policy(conn: &turso::Connection, phase: &str, call: ToolCall) -> String {
            let options = ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "turn-1"),
                source_message_id: "message-1",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "semantic",
            };
            let mut events: Vec<Event> = Vec::new();
            apply_tool_call(call, conn, &options, &mut events).unwrap_or_default()
        }

        fn resume() -> ToolCall {
            ToolCall::SetPingPolicy {
                action: "resume".to_string(),
                topic: String::new(),
                until: String::new(),
            }
        }

        #[test]
        fn policy_parser_rejects_unknown_and_incomplete_actions() {
            assert!(matches!(
                parse_tool_calls(r#"{"tool":"set_ping_policy","args":{"action":"resume"}}"#)
                    .first(),
                Some(ToolCall::SetPingPolicy { .. })
            ));
            for input in [
                r#"{"tool":"set_ping_policy","args":{"action":"pause"}}"#,
                r#"{"tool":"set_ping_policy","args":{"action":"mute"}}"#,
                r#"{"tool":"set_ping_policy","args":{"action":"divert","until":"2026-07-15T00:00:00"}}"#,
            ] {
                assert!(parse_tool_calls(input).is_empty(), "{input}");
            }
        }

        #[test]
        fn policy_tool_is_seat_symmetric_and_denied_to_autonomous_seats() {
            let conn = db::block_on(harness());
            for phase in ["scribe", "extract", "consolidate"] {
                assert!(apply_policy(&conn, phase, resume()).contains("set_ping_policy"));
            }
            for phase in ["consolidation", "ingest"] {
                assert!(apply_policy(&conn, phase, resume()).contains("denied"));
            }
        }
    }

    mod loops {
        use super::super::{ApplyToolOptions, ToolCall, apply_tool_call, parse_tool_calls};
        use crate::memory::db;
        use crate::memory::graph::{Edge, Node};
        use crate::memory::ledger::{Actor, Event, EventContext};
        use turso::Builder;

        const NODES_DDL: &str = "CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')";
        const EDGES_DDL: &str = "CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, source_message_id TEXT NOT NULL DEFAULT '', memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', PRIMARY KEY (source, target))";

        async fn harness() -> turso::Connection {
            let Ok(database) = Builder::new_local(":memory:").build().await else {
                unreachable!("in-memory database builds")
            };
            let Ok(conn) = database.connect() else {
                unreachable!("in-memory database connects")
            };
            let _ = conn.execute(NODES_DDL, ()).await;
            let _ = conn.execute(EDGES_DDL, ()).await;
            conn
        }

        fn options(phase: &'static str) -> ApplyToolOptions<'static> {
            ApplyToolOptions {
                actor: Actor::new("system", phase),
                context: EventContext::new("chat-1", "chat-1:user:ts"),
                source_message_id: "chat-1:user:ts",
                event_payload: "",
                confidence_cap: None,
                provenance: "user-confirmed",
                default_memory_type: "episodic",
            }
        }

        fn apply(conn: &turso::Connection, phase: &'static str, call: ToolCall) -> String {
            let mut events: Vec<Event> = Vec::new();
            let Ok(description) = apply_tool_call(call, conn, &options(phase), &mut events) else {
                unreachable!("apply succeeds")
            };
            description
        }

        fn open(intention: &str, anchor: &str, due: &str) -> ToolCall {
            ToolCall::OpenLoop {
                intention: intention.to_string(),
                anchor: anchor.to_string(),
                due: due.to_string(),
            }
        }

        fn close(loop_id: &str, resolution: &str, note: &str) -> ToolCall {
            ToolCall::CloseLoop {
                loop_id: loop_id.to_string(),
                resolution: resolution.to_string(),
                note: note.to_string(),
            }
        }

        fn write_anchor(conn: &turso::Connection, anchor: &str) {
            let node = Node {
                node_type: "entity".to_string(),
                body: String::new(),
                date: String::new(),
            };
            let Ok(()) = db::write_node(conn, anchor, &node) else {
                unreachable!("anchor writes")
            };
        }

        fn edge(conn: &turso::Connection, source: &str, target: &str) -> Edge {
            let Ok(Some(edge)) = db::load_edge(conn, source, target) else {
                unreachable!("edge exists")
            };
            edge
        }

        fn write_raw_edge(conn: &turso::Connection, source: &str, target: &str, content: &str) {
            let raw_edge = Edge {
                source: source.to_string(),
                target: target.to_string(),
                content: content.to_string(),
                stability: 0.9,
                confidence: 0.9,
                importance: 1.0,
                subject: false,
                date: crate::prompts::today_iso(),
                occurred_at: None,
                expired_at: None,
                memory_type: "semantic".to_string(),
                origin_session: String::new(),
                provenance: "legacy".to_string(),
            };
            let Ok(()) = db::write_edge(conn, &raw_edge, "test") else {
                unreachable!("raw edge writes")
            };
        }

        #[test]
        fn loop_calls_parse_and_unknown_resolution_is_rejected() {
            let text = r#"{"tools":[{"tool":"open_loop","args":{"intention":"ask Markus","anchor":"home-server","due":"2026-07-15"}},{"tool":"close_loop","args":{"loop":"ask-markus","resolution":"done","note":"asked"}}]}"#;
            let calls = parse_tool_calls(text);
            assert_eq!(calls.len(), 2);
            assert!(matches!(calls[0], ToolCall::OpenLoop { .. }));
            assert!(matches!(calls[1], ToolCall::CloseLoop { .. }));
            let invalid = r#"{"tool":"close_loop","args":{"loop":"ask-markus","resolution":"forfeit","note":"no"}}"#;
            assert!(parse_tool_calls(invalid).is_empty());
        }

        #[test]
        fn open_loop_writes_fixed_semantic_node_and_edge() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            let intention = "ask Markus whether the systemd unit is running";
            let description = apply(
                &conn,
                "extract",
                open(intention, "home-server", "2026-07-15"),
            );
            assert!(description.contains("open_loop"), "{description}");
            let Ok(Some(node)) =
                db::load_node(&conn, "ask-markus-whether-the-systemd-unit-is-running")
            else {
                unreachable!("loop node exists")
            };
            assert_eq!(
                (
                    node.node_type.as_str(),
                    node.body.as_str(),
                    node.date.as_str()
                ),
                ("loop", intention, "2026-07-15")
            );
            let edge = edge(
                &conn,
                "ask-markus-whether-the-systemd-unit-is-running",
                "home-server",
            );
            assert_eq!(edge.content, format!("open loop: {intention}"));
            assert!((edge.confidence - 0.9).abs() < 1e-9);
            assert!((edge.stability - 0.9).abs() < 1e-9);
            assert_eq!(
                (edge.importance, edge.memory_type.as_str()),
                (1.0, "semantic")
            );
            assert_eq!(edge.provenance, "user-confirmed");
        }

        #[test]
        fn invalid_due_date_is_dropped_but_loop_still_opens() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(
                &conn,
                "extract",
                open("check the unit", "home-server", "Friday"),
            );
            let Ok(Some(node)) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("loop node exists")
            };
            assert!(node.date.is_empty());
            let loop_edge = edge(&conn, "check-the-unit", "home-server");
            assert!(loop_edge.expired_at.is_none());
            apply(
                &conn,
                "extract",
                open("check the calendar", "home-server", "2026-02-29"),
            );
            let Ok(Some(calendar)) = db::load_node(&conn, "check-the-calendar") else {
                unreachable!("calendar loop exists")
            };
            assert!(calendar.date.is_empty(), "non-leap February 29 is invalid");
        }

        #[test]
        fn close_loop_expires_anchor_with_code_built_resolution() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let description = apply(
                &conn,
                "consolidate",
                close("check-the-unit", "done", "confirmed running"),
            );
            assert!(description.contains("close_loop"), "{description}");
            let loop_edge = edge(&conn, "check-the-unit", "home-server");
            let expected = format!(
                "done: confirmed running (closed {})",
                crate::prompts::today_iso()
            );
            assert_eq!(loop_edge.content, expected);
            assert!(loop_edge.expired_at.is_some());
        }

        #[test]
        fn unknown_loop_returns_a_friendly_denial() {
            let conn = db::block_on(harness());
            let description = apply(&conn, "extract", close("missing-loop", "done", "finished"));
            assert!(description.contains("denied"));
            assert!(description.contains("not found"));
        }

        #[test]
        // Scribe is allowed: intention statements legitimately classify as
        // task turns (live evidence 2026-07-12), so loop tools are
        // seat-symmetric like record_outcome. Dream/ingest stay denied.
        fn dream_and_ingest_seats_cannot_mutate_loops() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            for phase in ["consolidation", "ingest"] {
                let opened = apply(&conn, phase, open("check the unit", "home-server", ""));
                let closed = apply(&conn, phase, close("check-the-unit", "done", "finished"));
                assert!(opened.contains("denied"), "{phase}: {opened}");
                assert!(closed.contains("denied"), "{phase}: {closed}");
            }
            let Ok(node) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("node lookup succeeds")
            };
            assert!(node.is_none());

            let opened = apply(&conn, "scribe", open("check the unit", "home-server", ""));
            assert!(opened.contains("open_loop:"), "{opened}");
            let closed = apply(&conn, "scribe", close("check-the-unit", "done", "finished"));
            assert!(closed.contains("close_loop:"), "{closed}");
        }

        #[test]
        fn open_loop_requires_an_existing_anchor() {
            let conn = db::block_on(harness());
            let description = apply(&conn, "extract", open("check the unit", "missing", ""));
            assert!(
                description.contains("anchor missing not found"),
                "{description}"
            );
            let Ok(node) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("node lookup succeeds")
            };
            assert!(node.is_none());
        }

        #[test]
        fn open_loop_denies_node_collisions_and_self_anchors() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "check-the-unit");
            let collision = apply(&conn, "extract", open("check the unit", "home-server", ""));
            assert!(collision.contains("non-loop node"), "{collision}");
            let self_anchor = apply(&conn, "extract", open("home server", "home-server", ""));
            assert!(self_anchor.contains("same id"), "{self_anchor}");
            let Ok(Some(node)) = db::load_node(&conn, "check-the-unit") else {
                unreachable!("colliding entity remains")
            };
            assert_eq!(node.node_type, "entity");
        }

        #[test]
        fn reopening_cannot_add_a_second_live_anchor() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "other-topic");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let denied = apply(&conn, "extract", open("check the unit", "other-topic", ""));
            assert!(
                denied.contains("already has anchor home-server"),
                "{denied}"
            );
            let Ok(missing) = db::load_edge(&conn, "check-the-unit", "other-topic") else {
                unreachable!("edge lookup succeeds")
            };
            assert!(missing.is_none());
        }

        #[test]
        fn close_loop_expires_only_actual_anchor_edges() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            write_anchor(&conn, "other-topic");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            write_raw_edge(&conn, "check-the-unit", "other-topic", "related fact");
            apply(
                &conn,
                "extract",
                close("check-the-unit", "done", "finished"),
            );
            assert!(
                edge(&conn, "check-the-unit", "home-server")
                    .expired_at
                    .is_some()
            );
            assert!(
                edge(&conn, "check-the-unit", "other-topic")
                    .expired_at
                    .is_none()
            );
        }

        #[test]
        fn generic_mutations_cannot_bypass_loop_lifecycle() {
            let conn = db::block_on(harness());
            write_anchor(&conn, "home-server");
            apply(&conn, "extract", open("check the unit", "home-server", ""));
            let expire = ToolCall::ExpireEdge {
                source: "check-the-unit".to_string(),
                target: "home-server".to_string(),
                rewritten_content: "gone".to_string(),
            };
            let denied = apply(&conn, "ingest", expire);
            assert!(denied.contains("loop lifecycle"), "{denied}");
            assert!(
                edge(&conn, "check-the-unit", "home-server")
                    .expired_at
                    .is_none()
            );
        }
    }
}
