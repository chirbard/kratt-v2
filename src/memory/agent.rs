use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;

use crate::api::{Chunk, EmbeddingClient, LlmClient, ToolCallRequest};
use crate::prompts::{
    AgentPrompt, CONSOLIDATE_EXPLANATION, EXTRACT_EXPLANATION, SCRIBE_EXPLANATION,
    active_game_context, build_agent_prompt, build_classify_prompt, build_continue_prompt,
    build_scribe_continue_prompt, build_scribe_prompt,
};
use crate::runlog;
use turso::Connection;

use super::db;
use super::decay;
use super::graph::{
    Edge, Node, PROVENANCE_INFERRED, PROVENANCE_USER_CONFIRMED, edge_is_pinned, inferred_ceiling,
    node_type_is_loop, normalize_id, stability_seed_max,
};
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
    pub embed: Arc<dyn EmbeddingClient>,
}

impl MemorySeats {
    // All three seats on one client — the pre-split default.
    #[cfg(test)]
    pub fn uniform(llm: Arc<dyn LlmClient>) -> Self {
        Self {
            classifier: Arc::clone(&llm),
            scribe: Arc::clone(&llm),
            heavy: llm,
            embed: Arc::new(NoopEmbedding),
        }
    }
}

#[cfg(test)]
struct NoopEmbedding;

#[cfg(test)]
impl EmbeddingClient for NoopEmbedding {
    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
        Ok(Vec::new())
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
    parsed: usize,
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
    },
    UpsertEdge {
        source: String,
        target: String,
        content: String,
        stability: f64,
        confidence: f64,
        importance: f64,
        subject: bool,
        occurred_at: Option<String>,
        // None means the caller did not specify memory_type — the writer's
        // default (ApplyToolOptions::default_memory_type) applies.
        memory_type: Option<String>,
        inferred_from: Option<Vec<(String, String)>>,
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
    // knowledge on the game-definition node.
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
    mut seats: MemorySeats,
    user_msg: &str,
    assistant_msg: &str,
    user_message_id: &str,
    sender: std::sync::mpsc::Sender<MemoryEvent>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let trace_context = crate::trace::TraceContext::new("memory");
    seats.classifier =
        crate::trace::TracingClient::wrap(seats.classifier, trace_context.with_seat("classifier"));
    seats.scribe =
        crate::trace::TracingClient::wrap(seats.scribe, trace_context.with_seat("scribe"));
    seats.heavy = crate::trace::TracingClient::wrap(seats.heavy, trace_context.with_seat("heavy"));
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
        let active_game = self.active_game_name(&conn);
        let selection =
            self.heavy_selection(&pre_extract_graph, &injected, &conn, active_game.as_deref());
        self.log_heavy_selection(&selection);
        let pre_extract_toon = selection.toon;
        let extract_prompt = build_agent_prompt(&AgentPrompt {
            explanation: EXTRACT_EXPLANATION,
            toon: &pre_extract_toon,
            user_msg: self.user_msg,
            assistant_msg: self.assistant_msg,
            diff_text: None,
            injected_toon: None,
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

    fn heavy_selection(
        &self,
        graph: &super::graph::Graph,
        injected: &[Edge],
        conn: &Connection,
        active_game: Option<&str>,
    ) -> super::heavy::HeavySelection {
        let active_ids = self.active_context_ids(conn);
        let reserved = match active_game {
            Some(game) => active_game_context(game),
            None => String::new(),
        };
        let exchange = format!("{}\n{}", self.user_msg, self.assistant_msg);
        let session_id = db::source_session_id(self.user_message_id);
        super::heavy::select(
            &*self.seats.embed,
            &super::heavy::HeavyInputs {
                graph,
                injected,
                exchange: &exchange,
                session_id: session_id.as_deref(),
                active_ids: &active_ids,
                reserved_text: &reserved,
            },
        )
    }

    fn active_context_ids(&self, conn: &Connection) -> Vec<String> {
        let Some(session_id) = db::source_session_id(self.user_message_id) else {
            return Vec::new();
        };
        let hub_id = normalize_id(&format!("task-{session_id}"));
        let mut ids = Vec::new();
        if db::count_task_chain(conn, &hub_id).is_ok_and(|count| count > 0) {
            ids.push(hub_id.clone());
        }
        if let Ok(Some((game_id, _))) = db::load_session_game(conn, &hub_id) {
            ids.push(game_id);
        }
        ids
    }

    fn log_heavy_selection(&self, selection: &super::heavy::HeavySelection) {
        let fill = selection.tier_fill;
        self.send_log(
            "heavy",
            format!(
                "selection: {} edges + {} nodes (~{} tokens, tier fill 1:{} 2:{} 3:{} 4:{})",
                selection.edge_count,
                selection.node_count,
                selection.estimated_tokens,
                fill[0],
                fill[1],
                fill[2],
                fill[3]
            ),
        );
    }

    fn decayed_graph(&self) -> Result<super::graph::Graph, Box<dyn Error + Send + Sync>> {
        let mut graph = db::load_graph();
        // Decay persists itself with targeted updates; the whole-graph rewrite
        // that used to follow here destroyed embeddings on every run.
        decay::apply_decay_if_needed(&mut graph);
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
        let mut parsed_count = 0usize;
        let mut applied_count = 0usize;
        let mut denied_count = 0usize;
        for iteration in 1..=pass.max_iterations {
            let outcome = iterate(iteration, prompt)?;
            parsed_count += outcome.parsed;
            for description in &outcome.descriptions {
                if description.contains("denied") {
                    denied_count += 1;
                } else {
                    applied_count += 1;
                }
            }
            applied.extend(outcome.descriptions);
            if outcome.done || iteration == pass.max_iterations {
                break;
            }
            // Continuations must list everything applied so far, not just the
            // latest iteration, or the model re-emits earlier calls.
            prompt = (pass.continue_prompt)(
                pass.explanation,
                self.user_msg,
                self.assistant_msg,
                &applied,
            );
        }
        let seat = if pass.phase == "scribe" {
            "scribe"
        } else {
            "heavy"
        };
        self.send_log(
            "mem",
            format!(
                "{seat}: {parsed_count} parsed, {applied_count} applied, {denied_count} denied"
            ),
        );
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
        let parsed = tool_calls.len();
        self.send_log(
            pass.phase,
            format!("iteration {iteration}: {parsed} tool calls"),
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
            Ok(descriptions) => Ok(IterationOutcome {
                descriptions,
                parsed,
                done,
            }),
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
            None => {
                let calls = salvage_tool_calls(stripped);
                if calls.is_empty() && stripped.contains("\"tool") {
                    log_dropped_text(stripped, "bad JSON");
                }
                return calls;
            }
        },
    };
    if let Some(tools) = tool_array(&json_value) {
        let mut calls = Vec::new();
        for tool in tools {
            if let Some(call) = parse_single_tool(tool) {
                calls.push(call);
            } else {
                log_dropped_tool(tool);
            }
        }
        return calls;
    }
    match parse_single_tool(&json_value) {
        Some(call) => vec![call],
        None => {
            log_dropped_tool(&json_value);
            Vec::new()
        }
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
            log_dropped_text(span, "bad JSON");
            continue;
        };
        if let Some(call) = parse_single_tool(&json_value) {
            calls.push(call);
        } else {
            log_dropped_tool(&json_value);
        }
    }
    calls
}

fn log_dropped_tool(tool: &serde_json::Value) {
    log_dropped_text(&tool.to_string(), dropped_tool_reason(tool));
}

fn dropped_tool_reason(tool: &serde_json::Value) -> &'static str {
    let Some(name_value) = tool.get("tool") else {
        return "bad shape";
    };
    let Some(name) = name_value.as_str() else {
        return "bad shape";
    };
    if known_tool_name(name) {
        "bad shape"
    } else {
        "unknown tool"
    }
}

fn log_dropped_text(text: &str, reason: &str) {
    let snippet = truncated_snippet(text);
    runlog::log("mem", &format!("dropped tool call ({reason}): {snippet}"));
}

fn truncated_snippet(text: &str) -> String {
    let mut snippet: String = text.chars().take(120).collect();
    if text.chars().count() > 120 {
        snippet.push('…');
    }
    snippet
}

fn known_tool_name(name: &str) -> bool {
    matches!(
        name,
        "upsert_node"
            | "upsert_edge"
            | "delete_node"
            | "delete_edge"
            | "expire_edge"
            | "record_move"
            | "define_task"
            | "record_rule"
            | "record_outcome"
            | "record_reaction"
            | "open_loop"
            | "close_loop"
            | "inbox_add"
            | "register_interest"
            | "set_ping_policy"
    )
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
                    "body": {"type": "string"}
                },
                "required": ["name", "type", "body"]
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
                    "occurred_at": {"type": "string"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic"]}
                },
                "required": ["source", "target", "content", "stability", "confidence", "importance", "subject"]
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
    })
}

fn parse_upsert_edge(args: &serde_json::Value) -> Option<ToolCall> {
    let source = json_str(args, "source");
    let target = json_str(args, "target");
    if source.is_empty() || target.is_empty() {
        return None;
    }
    let occurred_at = json_str(args, "occurred_at");
    Some(ToolCall::UpsertEdge {
        source,
        target,
        content: json_str(args, "content"),
        stability: json_f64(args, "stability"),
        confidence: json_f64(args, "confidence"),
        importance: json_f64(args, "importance"),
        subject: json_bool(args, "subject"),
        occurred_at: if occurred_at.trim().is_empty() {
            None
        } else {
            Some(occurred_at)
        },
        memory_type: json_opt_str(args, "memory_type"),
        inferred_from: parse_optional_edge_refs(args, "inferred_from"),
    })
}

fn parse_optional_edge_refs(args: &serde_json::Value, key: &str) -> Option<Vec<(String, String)>> {
    let values = args.get(key)?.as_array()?;
    let mut refs = Vec::new();
    for value in values {
        let parsed = match value {
            serde_json::Value::String(text) => parse_support_ref(text),
            serde_json::Value::Array(pair) if pair.len() == 2 => parse_ref_pair(&pair[0], &pair[1]),
            _ => None,
        };
        if let Some(edge_ref) = parsed {
            refs.push(edge_ref);
        }
    }
    Some(refs)
}

fn parse_ref_pair(
    source_value: &serde_json::Value,
    target_value: &serde_json::Value,
) -> Option<(String, String)> {
    let source = normalize_id(source_value.as_str()?);
    let target = normalize_id(target_value.as_str()?);
    if source.is_empty() || target.is_empty() {
        return None;
    }
    Some((source, target))
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

pub(crate) struct BatchPreflight {
    denials: Vec<Option<String>>,
}

impl BatchPreflight {
    pub(crate) fn denial(&self, idx: usize) -> Option<&String> {
        self.denials[idx].as_ref()
    }
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
        let applied = apply_preflighted_batch(conn, tool_calls, &options, &mut events)?;
        for description in applied {
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

fn apply_preflighted_batch<'scope>(
    conn: &Connection,
    tool_calls: Vec<ToolCall>,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let synthesis_nodes = HashSet::new();
    let preflight = preflight_tool_batch(conn, &tool_calls, &synthesis_nodes)?;
    let mut batch = AgentApplyBatch::new(tool_calls);
    apply_preflight_nodes(conn, options, events, &preflight, &mut batch)?;
    apply_preflight_rest(conn, options, events, &preflight, &mut batch)?;
    Ok(batch.finish())
}

struct AgentApplyBatch {
    pending: Vec<Option<ToolCall>>,
    outcomes: Vec<Option<String>>,
}

impl AgentApplyBatch {
    fn new(tool_calls: Vec<ToolCall>) -> Self {
        let mut pending = Vec::new();
        for call in tool_calls {
            pending.push(Some(call));
        }
        let outcomes = vec![None; pending.len()];
        Self { pending, outcomes }
    }

    fn finish(self) -> Vec<String> {
        let mut finished = Vec::new();
        for outcome in self.outcomes {
            finished.push(outcome.unwrap_or_default());
        }
        finished
    }
}

fn apply_preflight_nodes<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    preflight: &BatchPreflight,
    batch: &mut AgentApplyBatch,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for idx in 0..batch.pending.len() {
        if preflight.denial(idx).is_some() || !is_upsert_node(batch.pending[idx].as_ref()) {
            continue;
        }
        let Some(call) = batch.pending[idx].take() else {
            continue;
        };
        batch.outcomes[idx] = Some(apply_tool_call(call, conn, options, events)?);
    }
    Ok(())
}

fn apply_preflight_rest<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
    preflight: &BatchPreflight,
    batch: &mut AgentApplyBatch,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for idx in 0..batch.pending.len() {
        if let Some(denial) = preflight.denial(idx) {
            batch.outcomes[idx] = Some(denial.clone());
            continue;
        }
        let Some(call) = batch.pending[idx].take() else {
            continue;
        };
        batch.outcomes[idx] = Some(apply_tool_call(call, conn, options, events)?);
    }
    Ok(())
}

fn is_upsert_node(call: Option<&ToolCall>) -> bool {
    matches!(call, Some(ToolCall::UpsertNode { .. }))
}

pub(crate) fn preflight_tool_batch(
    conn: &Connection,
    calls: &[ToolCall],
    synthesis_nodes: &HashSet<String>,
) -> Result<BatchPreflight, Box<dyn Error + Send + Sync>> {
    let mut denials = vec![None; calls.len()];
    let mut declared = HashMap::new();
    preflight_node_declarations(conn, calls, synthesis_nodes, &mut declared, &mut denials)?;
    preflight_edges(conn, calls, synthesis_nodes, &declared, &mut denials)?;
    Ok(BatchPreflight { denials })
}

fn preflight_node_declarations(
    conn: &Connection,
    calls: &[ToolCall],
    synthesis_nodes: &HashSet<String>,
    declared: &mut HashMap<String, String>,
    denials: &mut [Option<String>],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for (idx, call) in calls.iter().enumerate() {
        let ToolCall::UpsertNode {
            name, node_type, ..
        } = call
        else {
            continue;
        };
        let node_id = normalize_id(name);
        match declared_node_denial(conn, &node_id, node_type, synthesis_nodes)? {
            Some(denial) => set_node_denial(denials, idx, denial),
            None => {
                declared.insert(
                    node_id.clone(),
                    resolved_declared_type(&node_id, synthesis_nodes),
                );
            }
        }
    }
    Ok(())
}

fn preflight_edges(
    conn: &Connection,
    calls: &[ToolCall],
    synthesis_nodes: &HashSet<String>,
    declared: &HashMap<String, String>,
    denials: &mut [Option<String>],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for (idx, call) in calls.iter().enumerate() {
        if denials[idx].is_some() {
            continue;
        }
        let ToolCall::UpsertEdge {
            source,
            target,
            content,
            subject,
            ..
        } = call
        else {
            continue;
        };
        let source_id = normalize_id(source);
        let target_id = normalize_id(target);
        let edge = EdgeDeclaration {
            source: &source_id,
            target: &target_id,
            content,
            subject: *subject,
        };
        if let Some(denial) = edge_denial(conn, &edge, synthesis_nodes, declared)? {
            set_node_denial(denials, idx, denial);
        }
    }
    Ok(())
}

struct EdgeDeclaration<'scope> {
    source: &'scope str,
    target: &'scope str,
    content: &'scope str,
    subject: bool,
}

fn edge_denial<'scope>(
    conn: &Connection,
    edge: &EdgeDeclaration<'scope>,
    synthesis_nodes: &HashSet<String>,
    declared: &HashMap<String, String>,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    if synthesis_reference_missing(edge.source, edge.target, synthesis_nodes, declared) {
        return Ok(Some(
            "upsert_edge denied: synthesis endpoints must be declared in the same batch"
                .to_string(),
        ));
    }
    let source_fact = endpoint_is_fact_like(conn, edge.source, declared)?;
    let target_fact = endpoint_is_fact_like(conn, edge.target, declared)?;
    if edge.subject && !source_fact {
        return Ok(Some(format!(
            "upsert_edge denied: subject source {} must be fact or synthesis",
            edge.source
        )));
    }
    if edge.content.trim().eq_ignore_ascii_case("implies") && (!source_fact || !target_fact) {
        return Ok(Some(
            "upsert_edge denied: implies endpoints must be fact or synthesis".to_string(),
        ));
    }
    Ok(None)
}

fn synthesis_reference_missing(
    source: &str,
    target: &str,
    synthesis_nodes: &HashSet<String>,
    declared: &HashMap<String, String>,
) -> bool {
    for node_id in [source, target] {
        let declared_synthesis = match declared.get(node_id) {
            Some(node_type) => node_type == "synthesis",
            None => false,
        };
        if synthesis_nodes.contains(node_id) && !declared_synthesis {
            return true;
        }
    }
    false
}

fn endpoint_is_fact_like(
    conn: &Connection,
    node_id: &str,
    declared: &HashMap<String, String>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    if let Some(node_type) = declared.get(node_id) {
        return Ok(matches!(node_type.as_str(), "fact" | "synthesis"));
    }
    let node = db::load_node(conn, node_id)?;
    Ok(node
        .as_ref()
        .is_some_and(|stored| matches!(stored.node_type.as_str(), "fact" | "synthesis")))
}

fn set_node_denial(denials: &mut [Option<String>], idx: usize, denial: String) {
    runlog::log("nodes", &format!("denied: {denial}"));
    denials[idx] = Some(denial);
}

fn resolved_declared_type(node_id: &str, synthesis_nodes: &HashSet<String>) -> String {
    if synthesis_nodes.contains(node_id) {
        return "synthesis".to_string();
    }
    "fact".to_string()
}

pub(crate) fn declared_node_denial(
    conn: &Connection,
    node_id: &str,
    requested: &str,
    synthesis_nodes: &HashSet<String>,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let resolved = resolved_declared_type(node_id, synthesis_nodes);
    if resolved == "fact" && requested != "fact" {
        return Ok(Some(
            "upsert_node denied: only fact nodes may be declared; entities are created automatically from edge endpoints".to_string(),
        ));
    }
    let Some(stored) = db::load_node(conn, node_id)? else {
        return Ok(None);
    };
    if stored.node_type == resolved {
        return Ok(None);
    }
    if stored.node_type == "entity" {
        return Ok(Some(
            "upsert_node denied: entity collision; an entity is an identity, a fact is a claim about one — use a distinct fact id and connect them with an edge".to_string(),
        ));
    }
    Ok(Some(format!(
        "upsert_node denied: id {node_id} belongs to a {} node",
        stored.node_type
    )))
}

#[rustfmt::skip]
pub(crate) fn apply_tool_call<'scope>(
    call: ToolCall,
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if !matches!(&call, ToolCall::UpsertNode { .. }) && tool_call_touches_loop(conn, &call)? {
        return Ok("mutation denied: loop lifecycle requires open_loop or close_loop".to_string());
    }
    match call {
        ToolCall::UpsertNode { name, node_type, body } =>
            apply_declared_upsert_node(conn, name, node_type, body),
        ToolCall::UpsertEdge { source, target, content, stability, confidence, importance, subject, occurred_at, memory_type, inferred_from } =>
            apply_upsert_edge_call(conn, options, events, UpsertEdgeCall { source, target, content, stability, confidence, importance, subject, occurred_at, memory_type, inferred_from }),
        ToolCall::DeleteNode { name } =>
            apply_delete_node(conn, name, options, events),
        ToolCall::DeleteEdge { source, target } =>
            apply_delete_edge(conn, EdgeRef::new(&source, &target), options, events),
        ToolCall::ExpireEdge { source, target, rewritten_content } =>
            apply_expire_edge(conn, ExpireEdgeCall { source, target, rewritten_content }, options, events),
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
        ceiling: None,
        last_decayed_at: String::new(),
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
        ceiling: None,
        last_decayed_at: String::new(),
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
        ceiling: None,
        last_decayed_at: String::new(),
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
        | ToolCall::ExpireEdge { source, target, .. } => edge_touches_loop(conn, source, target),
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
    Ok(node_type_is_loop(&node.node_type))
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
    let loop_id = normalize_id(&call.intention);
    if let Some(reason) = open_loop_block_reason(conn, &loop_id, &anchor_id)? {
        return Ok(format!("open_loop denied: {reason}"));
    }
    db::ensure_entity_endpoints(conn, &anchor_id, &anchor_id)?;
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
        ceiling: None,
        last_decayed_at: String::new(),
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
    occurred_at: Option<String>,
    memory_type: Option<String>,
    inferred_from: Option<Vec<(String, String)>>,
}

fn upsert_edge_to_edge(
    call: UpsertEdgeCall,
    memory_type: String,
    origin_session: String,
    provenance: &str,
) -> Edge {
    Edge {
        source: normalize_id(&call.source),
        target: normalize_id(&call.target),
        content: call.content,
        stability: call.stability,
        confidence: call.confidence,
        importance: call.importance,
        subject: call.subject,
        date: crate::prompts::today_iso(),
        occurred_at: call.occurred_at,
        expired_at: None,
        memory_type,
        origin_session,
        provenance: provenance.to_string(),
        ceiling: None,
        last_decayed_at: String::new(),
    }
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
    let inferred_from = call.inferred_from.take();
    let memory_type = match call.memory_type.take() {
        Some(requested) => super::graph::normalize_memory_type(&requested),
        None => options.default_memory_type.to_string(),
    };
    if memory_type == "episodic" {
        call.occurred_at = None;
    }
    let origin_session = if memory_type == "episodic" {
        options.context.session_id.clone()
    } else {
        String::new()
    };
    let mut edge = upsert_edge_to_edge(call, memory_type, origin_session, options.provenance);
    apply_inferred_declaration(conn, options, &inferred_from, &mut edge)?;
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

fn apply_inferred_declaration<'scope>(
    conn: &Connection,
    options: &ApplyToolOptions<'scope>,
    refs: &Option<Vec<(String, String)>>,
    edge: &mut Edge,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(premises) = refs else {
        return Ok(());
    };
    if !matches!(options.actor.actor_id.as_str(), "extract" | "consolidate") {
        return Ok(());
    }
    let confidences = resolve_live_premise_confidences(conn, premises)?;
    let Some(ceiling) = inferred_ceiling(&confidences) else {
        runlog::log(
            "epistemics",
            &format!(
                "[epistemics] inferred fallback: {} → {} has no resolvable premises",
                edge.source, edge.target
            ),
        );
        return Ok(());
    };
    edge.provenance = PROVENANCE_INFERRED.to_string();
    edge.ceiling = Some(ceiling);
    Ok(())
}

pub(crate) fn resolve_live_premise_confidences(
    conn: &Connection,
    refs: &[(String, String)],
) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let mut confidences = Vec::new();
    for (source, target) in refs {
        match db::load_edge(conn, source, target)? {
            Some(edge) if edge.expired_at.is_none() => confidences.push(edge.confidence),
            _ => runlog::log(
                "epistemics",
                &format!("[epistemics] premise skipped: {source} → {target}"),
            ),
        }
    }
    Ok(confidences)
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

fn apply_declared_upsert_node(
    conn: &Connection,
    name: String,
    node_type: String,
    body: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let normalized = normalize_id(&name);
    let synthesis_nodes = HashSet::new();
    if let Some(denial) = declared_node_denial(conn, &normalized, &node_type, &synthesis_nodes)? {
        runlog::log("nodes", &format!("denied: {denial}"));
        return Ok(denial);
    }
    let date = first_seen_date(conn, &normalized)?;
    apply_upsert_node(conn, name, "fact".to_string(), body, date)
}

fn first_seen_date(
    conn: &Connection,
    node_id: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(stored) = db::load_node(conn, node_id)?
        && !stored.date.is_empty()
    {
        return Ok(stored.date);
    }
    Ok(crate::prompts::today_iso())
}

fn apply_upsert_edge<'scope>(
    conn: &Connection,
    mut edge: Edge,
    options: &ApplyToolOptions<'scope>,
    events: &mut Vec<Event>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let existing = db::load_edge(conn, &edge.source, &edge.target)?;
    edge.provenance = match &existing {
        Some(_) if options.provenance == PROVENANCE_USER_CONFIRMED => {
            PROVENANCE_USER_CONFIRMED.to_string()
        }
        Some(stored) => stored.provenance.clone(),
        None => edge.provenance.clone(),
    };
    if existing.is_some() && options.provenance == PROVENANCE_USER_CONFIRMED {
        edge.ceiling = None;
    } else if let Some(stored) = &existing {
        edge.ceiling = stored.ceiling;
    }
    apply_stability_policy(conn, &existing, options, &mut edge)?;
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

const REPEAT_STABILITY_BONUS: f64 = 0.05; // replay-tunable.

fn apply_stability_policy<'scope>(
    conn: &Connection,
    existing: &Option<Edge>,
    options: &ApplyToolOptions<'scope>,
    edge: &mut Edge,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let pinned = edge_is_pinned_in_db(conn, edge)?;
    let Some(stored) = existing else {
        if !pinned {
            edge.stability = edge.stability.min(stability_seed_max(&edge.provenance));
        }
        return Ok(());
    };
    if pinned {
        edge.stability = stored.stability;
        return Ok(());
    }
    if memory_pass_actor(&options.actor.actor_id) && stored.expired_at.is_none() {
        edge.stability = (stored.stability + REPEAT_STABILITY_BONUS).min(1.0);
    } else {
        edge.stability = stored.stability;
    }
    Ok(())
}

fn memory_pass_actor(actor_id: &str) -> bool {
    matches!(actor_id, "scribe" | "extract" | "consolidate")
}

fn edge_is_pinned_in_db(
    conn: &Connection,
    edge: &Edge,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let source = db::load_node(conn, &edge.source)?;
    let source_type = match source.as_ref() {
        Some(node) => node.node_type.as_str(),
        None => "",
    };
    Ok(edge_is_pinned(edge, source_type))
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
#[path = "agent_tests.rs"]
mod tests;
