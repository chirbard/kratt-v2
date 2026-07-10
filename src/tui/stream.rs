use std::sync::Arc;

use serde_json::Value;

use crate::api::{Chunk, EmbeddingClient, LlmClient, Stats, TurnOutcome};
use crate::memory;
use crate::runlog;
use crate::session;
use crate::turn::{self, MemoryBuild, TurnContext};
use crate::web::SearchClient;

use super::app::{App, Message, Role, State};

pub(crate) struct StreamJob {
    pub(crate) llm: Arc<dyn LlmClient>,
    pub(crate) embed: Arc<dyn EmbeddingClient>,
    pub(crate) search: Option<Arc<dyn SearchClient>>,
    pub(crate) prompt: String,
    pub(crate) session_path: std::path::PathBuf,
    pub(crate) turn_id: String,
    pub(crate) sender: std::sync::mpsc::Sender<Chunk>,
}

struct StatsAgg {
    tokens_in: u32,
    tokens_out: u32,
    cost_usd: f64,
    ttft_ms: u128,
}

impl StatsAgg {
    fn new() -> Self {
        Self {
            tokens_in: 0,
            tokens_out: 0,
            cost_usd: 0.0,
            ttft_ms: 0,
        }
    }
}

pub(crate) fn start_stream(app: &mut App) {
    let prompt = std::mem::take(&mut app.input);
    app.last_user_message_id = session::append_entry(&app.session_path, "user", &prompt);
    app.history.push(Message {
        role: Role::User,
        reasoning: String::new(),
        content: prompt.clone(),
        tool_call: String::new(),
        attribution: String::new(),
    });
    app.history.push(Message {
        role: Role::Assistant,
        reasoning: String::new(),
        content: String::new(),
        tool_call: String::new(),
        attribution: String::new(),
    });
    app.stats = None;
    app.auto_scroll = true;
    app.state = State::Streaming;
    let (sender, recvr) = std::sync::mpsc::channel();
    app.stream_rx = Some(recvr);
    let job = StreamJob {
        llm: Arc::clone(&app.chat_llm),
        embed: Arc::clone(&app.embed),
        search: clone_search_client(app),
        prompt,
        session_path: app.session_path.clone(),
        turn_id: app.last_user_message_id.clone(),
        sender,
    };
    std::thread::spawn(move || run_stream_worker(job));
}

fn clone_search_client(app: &App) -> Option<Arc<dyn SearchClient>> {
    match &app.search {
        Some(arc) => Some(Arc::clone(arc)),
        None => None,
    }
}

fn run_stream_worker(job: StreamJob) {
    let ctx = TurnContext {
        embed: &*job.embed,
        search: job.search.as_deref(),
        prompt: &job.prompt,
        session_path: &job.session_path,
        turn_id: &job.turn_id,
        style_hint: None,
    };
    let (mut messages, tools, memory_build) = match turn::prepare_stream_context(&ctx) {
        Ok(result) => result,
        Err(err) => {
            let _ = job.sender.send(Chunk::Err(err.to_string()));
            return;
        }
    };
    turn::log_memory_context(&memory_build);
    stream_turns(&job, &tools, &mut messages, &memory_build);
}

fn stream_turns(
    job: &StreamJob,
    tools: &Value,
    messages: &mut Vec<Value>,
    memory_build: &MemoryBuild,
) {
    let mut stats_agg = StatsAgg::new();
    let mut first_turn = true;
    loop {
        match job.llm.stream_turn(messages, Some(tools), &job.sender) {
            Ok(TurnOutcome::Done(stats)) => {
                finish_stream(job, memory_build, stats, &mut stats_agg, first_turn);
                return;
            }
            Ok(TurnOutcome::ToolCalls(calls)) => {
                turn::execute_tool_turn(
                    job.search.as_deref(),
                    &job.session_path,
                    &job.sender,
                    &calls,
                    messages,
                );
            }
            Err(err) => {
                let _ = job.sender.send(Chunk::Err(err.to_string()));
                return;
            }
        }
        first_turn = false;
    }
}

fn finish_stream(
    job: &StreamJob,
    memory_build: &MemoryBuild,
    stats: Stats,
    stats_agg: &mut StatsAgg,
    first_turn: bool,
) {
    // Done must be sent last: the app drops the stream receiver once it
    // leaves the Streaming state, so anything sent after Done is lost.
    emit_routing_debug(&job.sender, &memory_build.debug_log);
    emit_attribution(job);
    let final_stats = send_final_stats(&job.sender, stats, stats_agg, first_turn);
    runlog::log("chat", &format_stats_line(&final_stats));
}

fn emit_routing_debug(sender: &std::sync::mpsc::Sender<Chunk>, routing_debug: &str) {
    if routing_debug.is_empty() {
        return;
    }
    let _ = sender.send(Chunk::MemoryLog {
        phase: "debug",
        text: routing_debug.to_string(),
    });
}

fn emit_attribution(job: &StreamJob) {
    let attribution = memory::turn_attribution(&job.turn_id);
    let _ = job
        .sender
        .send(Chunk::Attribution(turn::format_attribution_line(&attribution)));
}

fn send_final_stats(
    sender: &std::sync::mpsc::Sender<Chunk>,
    stats: Stats,
    stats_agg: &mut StatsAgg,
    first_turn: bool,
) -> Stats {
    stats_agg.tokens_in += stats.tokens_in;
    stats_agg.tokens_out += stats.tokens_out;
    stats_agg.cost_usd += stats.cost_usd.unwrap_or(0.0);
    if first_turn {
        stats_agg.ttft_ms = stats.ttft_ms;
    }
    let cost_usd = if stats_agg.cost_usd > 0.0 {
        Some(stats_agg.cost_usd)
    } else {
        None
    };
    let final_stats = Stats {
        tokens_in: stats_agg.tokens_in,
        tokens_out: stats_agg.tokens_out,
        cost_usd,
        ttft_ms: stats_agg.ttft_ms,
    };
    let _ = sender.send(Chunk::Done(Stats {
        tokens_in: final_stats.tokens_in,
        tokens_out: final_stats.tokens_out,
        cost_usd: final_stats.cost_usd,
        ttft_ms: final_stats.ttft_ms,
    }));
    final_stats
}

fn format_stats_line(stats: &Stats) -> String {
    let cost = match stats.cost_usd {
        Some(value) => format!("${value:.6}"),
        None => "N/A".to_string(),
    };
    format!(
        "final stats: in={} out={} cost={} ttft={}ms",
        stats.tokens_in, stats.tokens_out, cost, stats.ttft_ms
    )
}
