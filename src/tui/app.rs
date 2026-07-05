use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{Chunk, EmbeddingClient, LlmClient, Stats};
use crate::memory::{self, DreamEvent, ExploreEvent, MemoryEvent};
use crate::session;
use crate::web::SearchClient;

pub(crate) const RECENT_TURNS: usize = 2;

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
pub(crate) enum DreamState {
    Idle,
    Running,
    Done,
}

#[derive(PartialEq)]
pub(crate) enum ExploreState {
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
    pub(crate) memory_llm: Arc<dyn LlmClient>,
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
    pub(crate) dream_state: DreamState,
    pub(crate) dream_rx: Option<std::sync::mpsc::Receiver<DreamEvent>>,
    pub(crate) dream_log: Vec<(String, String)>,
    pub(crate) dream_summary: String,
    pub(crate) explore_state: ExploreState,
    pub(crate) explore_rx: Option<std::sync::mpsc::Receiver<ExploreEvent>>,
    pub(crate) explore_log: Vec<(String, String)>,
    pub(crate) explore_summary: String,
}

impl App {
    pub(crate) fn new(
        chat_llm: Arc<dyn LlmClient>,
        memory_llm: Arc<dyn LlmClient>,
        embed: Arc<dyn EmbeddingClient>,
        search: Option<Arc<dyn SearchClient>>,
        session_path: PathBuf,
    ) -> Self {
        Self {
            chat_llm,
            memory_llm,
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
            dream_state: DreamState::Idle,
            dream_rx: None,
            dream_log: Vec::new(),
            dream_summary: String::new(),
            explore_state: ExploreState::Idle,
            explore_rx: None,
            explore_log: Vec::new(),
            explore_summary: String::new(),
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

pub(crate) fn drain_dream(app: &mut App) {
    let dream_receiver = match app.dream_rx.take() {
        Some(dream_receiver) => dream_receiver,
        None => return,
    };
    loop {
        match dream_receiver.try_recv() {
            Ok(DreamEvent::Log { phase, text }) => {
                app.dream_log.push((phase.to_string(), text));
            }
            Ok(DreamEvent::Append { phase, text }) => match app.dream_log.last_mut() {
                Some(last) if last.0 == phase => last.1.push_str(&text),
                _ => app.dream_log.push((phase.to_string(), text)),
            },
            Ok(DreamEvent::Done { edges_added }) => {
                app.dream_summary = format!("dream done — {edges_added} edges written");
                app.dream_state = DreamState::Done;
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.dream_rx = Some(dream_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.dream_state = DreamState::Idle;
                return;
            }
        }
    }
}

pub(crate) fn drain_explore(app: &mut App) {
    let explore_receiver = match app.explore_rx.take() {
        Some(explore_receiver) => explore_receiver,
        None => return,
    };
    loop {
        match explore_receiver.try_recv() {
            Ok(ExploreEvent::Log { phase, text }) => {
                app.explore_log.push((phase.to_string(), text));
            }
            Ok(ExploreEvent::Done { events_recorded }) => {
                app.explore_summary = format!("explore done — {events_recorded} events");
                app.explore_state = ExploreState::Done;
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                app.explore_rx = Some(explore_receiver);
                return;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.explore_state = ExploreState::Idle;
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
    let llm = Arc::clone(&app.memory_llm);
    std::thread::spawn(move || {
        let _ = memory::run_memory_agent(
            llm,
            &user_content,
            &assistant_content,
            &user_message_id,
            sender,
        );
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

pub(crate) fn start_dream(app: &mut App) {
    app.dream_log.clear();
    app.dream_summary.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<DreamEvent>();
    let llm = Arc::clone(&app.memory_llm);
    let embed = Arc::clone(&app.embed);
    std::thread::spawn(move || {
        let _ = memory::run_dream_pass(llm, embed, sender);
    });
    app.dream_rx = Some(receiver);
    app.dream_state = DreamState::Running;
}

pub(crate) fn start_exploration(app: &mut App) {
    app.explore_log.clear();
    app.explore_summary.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<ExploreEvent>();
    let llm = Arc::clone(&app.memory_llm);
    let embed = Arc::clone(&app.embed);
    std::thread::spawn(move || {
        let _ = memory::run_exploration_pass(llm, embed, sender);
    });
    app.explore_rx = Some(receiver);
    app.explore_state = ExploreState::Running;
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
