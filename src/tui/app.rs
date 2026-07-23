use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{Chunk, EmbeddingClient, LlmClient, Stats};
use crate::memory::{self, IngestEvent, MemoryEvent, MemorySeats};
use crate::session;
use crate::web::SearchClient;

#[derive(PartialEq)]
pub(crate) enum Mode {
    Insert,
    Normal,
}

#[derive(PartialEq)]
pub(crate) enum State {
    Idle,
    Streaming,
}

#[derive(PartialEq)]
pub(crate) enum MemoryState {
    Idle,
    Updating,
}

#[derive(PartialEq)]
pub(crate) enum IngestState {
    Idle,
    Running,
    Done,
}

pub(crate) enum Role {
    User,
    Assistant,
}

pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) reasoning: String,
    pub(crate) content: String,
    pub(crate) tool_call: String,
    pub(crate) attribution: String,
}

pub(crate) struct App {
    pub(crate) chat_llm: Arc<dyn LlmClient>,
    // The memory pass runs on per-seat clients (classifier/scribe/heavy). The
    // heavy seat is also the model for ingestion passes.
    pub(crate) seats: MemorySeats,
    pub(crate) embed: Arc<dyn EmbeddingClient>,
    pub(crate) search: Option<Arc<dyn SearchClient>>,
    pub(crate) input: String,
    pub(crate) mode: Mode,
    pub(crate) history: Vec<Message>,
    pub(crate) tick: u64,
    pub(crate) state: State,
    pub(crate) memory_state: MemoryState,
    pub(crate) stats: Option<Stats>,
    pub(crate) scroll: u16,
    pub(crate) auto_scroll: bool,
    pub(crate) memory_scroll: u16,
    pub(crate) memory_auto_scroll: bool,
    pub(crate) stream_rx: Option<std::sync::mpsc::Receiver<Chunk>>,
    pub(crate) memory_rx: Option<std::sync::mpsc::Receiver<MemoryEvent>>,
    pub(crate) embed_rx: Option<std::sync::mpsc::Receiver<MemoryEvent>>,
    pub(crate) memory_log: Vec<(String, String)>,
    pub(crate) show_memory_panel: bool,
    pub(crate) show_reasoning: bool,
    pub(crate) show_debug: bool,
    pub(crate) show_tools: bool,
    pub(crate) show_attribution: bool,
    pub(crate) session_path: PathBuf,
    pub(crate) last_user_message_id: String,
    pub(crate) ingest_state: IngestState,
    pub(crate) ingest_rx: Option<std::sync::mpsc::Receiver<IngestEvent>>,
    pub(crate) ingest_log: Vec<(String, String)>,
    pub(crate) ingest_summary: String,
}

impl App {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(
        chat_llm: Arc<dyn LlmClient>,
        seats: MemorySeats,
        embed: Arc<dyn EmbeddingClient>,
        search: Option<Arc<dyn SearchClient>>,
        session_path: PathBuf,
    ) -> Self {
        Self {
            chat_llm,
            seats,
            embed,
            search,
            input: String::new(),
            mode: Mode::Insert,
            history: Vec::new(),
            tick: 0,
            state: State::Idle,
            memory_state: MemoryState::Idle,
            stats: None,
            scroll: 0,
            auto_scroll: true,
            memory_scroll: 0,
            memory_auto_scroll: true,
            stream_rx: None,
            memory_rx: None,
            embed_rx: None,
            memory_log: Vec::new(),
            show_memory_panel: false,
            show_reasoning: true,
            show_debug: false,
            show_tools: false,
            show_attribution: true,
            session_path,
            last_user_message_id: String::new(),
            ingest_state: IngestState::Idle,
            ingest_rx: None,
            ingest_log: Vec::new(),
            ingest_summary: String::new(),
        }
    }
}

pub(crate) fn drain_stream(app: &mut App) {
    let recv = match app.stream_rx.take() {
        Some(recv) => recv,
        None => return,
    };
    while let Ok(chunk) = recv.try_recv() {
        apply_stream_chunk(app, chunk);
    }
    if app.state == State::Streaming {
        app.stream_rx = Some(recv);
    }
}

fn apply_stream_chunk(app: &mut App, chunk: Chunk) {
    match chunk {
        Chunk::Reasoning(text) => append_reasoning(app, &text),
        Chunk::Content(text) => append_content(app, &text),
        Chunk::ToolCall(desc) => set_tool_call(app, desc),
        Chunk::Attribution(text) => set_attribution(app, text),
        Chunk::MemoryLog { phase, text } => app.memory_log.push((phase.to_string(), text)),
        Chunk::Done(stats) => complete_stream(app, stats),
        Chunk::Err(err) => set_stream_error(app, &err),
    }
}

fn append_reasoning(app: &mut App, text: &str) {
    if let Some(msg) = app.history.last_mut() {
        msg.reasoning.push_str(text);
    }
}

fn append_content(app: &mut App, text: &str) {
    if let Some(msg) = app.history.last_mut() {
        msg.content.push_str(text);
    }
}

fn set_tool_call(app: &mut App, desc: String) {
    if let Some(msg) = app.history.last_mut() {
        msg.tool_call = desc;
    }
}

fn set_attribution(app: &mut App, text: String) {
    if let Some(msg) = app.history.last_mut() {
        msg.attribution = text;
    }
}

fn complete_stream(app: &mut App, stats: Stats) {
    app.stats = Some(stats);
    app.state = State::Idle;
    start_memory_update(app);
}

fn set_stream_error(app: &mut App, err: &str) {
    if let Some(msg) = app.history.last_mut() {
        msg.content = format!("Error: {err}");
    }
    app.state = State::Idle;
}

pub(crate) fn drain_memory(app: &mut App) {
    let memory_receiver = match app.memory_rx.take() {
        Some(memory_receiver) => memory_receiver,
        None => return,
    };
    loop {
        match memory_receiver.try_recv() {
            Ok(MemoryEvent::Log { phase, text }) => {
                app.memory_log.push((phase.to_string(), text));
            }
            Ok(MemoryEvent::Append { phase, text }) => match app.memory_log.last_mut() {
                Some(last) if last.0 == phase => last.1.push_str(&text),
                _ => app.memory_log.push((phase.to_string(), text)),
            },
            Ok(MemoryEvent::Done) => {
                app.memory_state = MemoryState::Idle;
                let embed = Arc::clone(&app.embed);
                let (embed_tx, embed_rx) = std::sync::mpsc::channel::<MemoryEvent>();
                std::thread::spawn(move || memory::run_embed_pass(embed, &embed_tx));
                app.embed_rx = Some(embed_rx);
                return;
            }
            Ok(MemoryEvent::MaintenanceDone) => {
                app.memory_state = MemoryState::Idle;
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.memory_rx = Some(memory_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.memory_state = MemoryState::Idle;
                return;
            }
        }
    }
}

pub(crate) fn drain_embed(app: &mut App) {
    let embed_receiver = match app.embed_rx.take() {
        Some(embed_receiver) => embed_receiver,
        None => return,
    };
    loop {
        match embed_receiver.try_recv() {
            Ok(MemoryEvent::Log { phase, text }) => {
                app.memory_log.push((phase.to_string(), text));
            }
            Ok(MemoryEvent::Append { .. })
            | Ok(MemoryEvent::Done)
            | Ok(MemoryEvent::MaintenanceDone) => {}
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.embed_rx = Some(embed_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
        }
    }
}

pub(crate) fn drain_ingest(app: &mut App) {
    let ingest_receiver = match app.ingest_rx.take() {
        Some(ingest_receiver) => ingest_receiver,
        None => return,
    };
    loop {
        match ingest_receiver.try_recv() {
            Ok(IngestEvent::Log { phase, text }) => {
                app.ingest_log.push((phase.to_string(), text));
            }
            Ok(IngestEvent::Done { chunks_processed }) => {
                app.ingest_summary = format!("ingest done — {chunks_processed} chunks");
                app.ingest_state = IngestState::Done;
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.ingest_rx = Some(ingest_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.ingest_state = IngestState::Idle;
                return;
            }
        }
    }
}

fn start_memory_update(app: &mut App) {
    let assistant_content = last_assistant_content(app);
    let user_content = second_to_last_content(app);
    let _ = session::append_entry(&app.session_path, "assistant", &assistant_content);
    let user_message_id = app.last_user_message_id.clone();
    app.memory_log.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let seats = app.seats.clone();
    std::thread::spawn(move || {
        let _ = memory::run_memory_agent(
            seats,
            &user_content,
            &assistant_content,
            &user_message_id,
            sender,
        );
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

fn clone_search_client(app: &App) -> Option<Arc<dyn SearchClient>> {
    match &app.search {
        Some(search) => Some(Arc::clone(search)),
        None => None,
    }
}

pub(crate) fn start_ingestion(app: &mut App, path: PathBuf) {
    app.ingest_log.clear();
    app.ingest_summary.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<IngestEvent>();
    let llm = Arc::clone(&app.seats.heavy);
    let embed = Arc::clone(&app.embed);
    let search = clone_search_client(app);
    std::thread::spawn(move || {
        let _ = memory::run_ingestion_pass(llm, embed, search, path, sender);
    });
    app.ingest_rx = Some(receiver);
    app.ingest_state = IngestState::Running;
}

pub(crate) fn start_recompute_positions(app: &mut App) {
    app.memory_log.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let _ = sender.send(MemoryEvent::Log {
        phase: "memory",
        text: "recomputing positions…".to_string(),
    });
    std::thread::spawn(move || {
        match memory::recompute_positions() {
            Ok(logs) => {
                for log in logs {
                    let _ = sender.send(MemoryEvent::Log {
                        phase: "memory",
                        text: log,
                    });
                }
            }
            Err(err) => {
                let _ = sender.send(MemoryEvent::Log {
                    phase: "memory",
                    text: format!("error: {err}"),
                });
            }
        }
        let _ = sender.send(MemoryEvent::MaintenanceDone);
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

pub(crate) fn start_sync(app: &mut App) {
    app.memory_log.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let _ = sender.send(MemoryEvent::Log {
        phase: "sync",
        text: "syncing with Turso Cloud…".to_string(),
    });
    std::thread::spawn(move || {
        let summary = memory::sync_now();
        let _ = sender.send(MemoryEvent::Log {
            phase: "sync",
            text: summary,
        });
        let _ = sender.send(MemoryEvent::MaintenanceDone);
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

pub(crate) fn last_assistant_content(app: &App) -> String {
    match app.history.last() {
        Some(message) => message.content.clone(),
        None => String::new(),
    }
}

pub(crate) fn second_to_last_content(app: &App) -> String {
    match app.history.iter().rev().nth(1) {
        Some(message) => message.content.clone(),
        None => String::new(),
    }
}
