use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{Chunk, LlmClient, Stats};
use crate::memory::{self, MemoryEvent};
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

pub(crate) enum Role {
    User,
    Assistant,
}

pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) reasoning: String,
    pub(crate) content: String,
    pub(crate) tool_call: String,
}

pub(crate) struct App {
    pub(crate) chat_llm: Arc<dyn LlmClient>,
    pub(crate) memory_llm: Arc<dyn LlmClient>,
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
    pub(crate) memory_log: Vec<(String, String)>,
    pub(crate) show_memory_panel: bool,
    pub(crate) show_reasoning: bool,
    pub(crate) show_debug: bool,
    pub(crate) show_tools: bool,
    pub(crate) session_path: PathBuf,
    pub(crate) memory_toon: String,
}

impl App {
    pub(crate) fn new(
        chat_llm: Arc<dyn LlmClient>,
        memory_llm: Arc<dyn LlmClient>,
        search: Option<Arc<dyn SearchClient>>,
        memory_toon: String,
        session_path: PathBuf,
    ) -> Self {
        Self {
            chat_llm,
            memory_llm,
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
            memory_log: Vec::new(),
            show_memory_panel: false,
            show_reasoning: true,
            show_debug: false,
            show_tools: false,
            session_path,
            memory_toon,
        }
    }
}

pub(crate) fn drain_stream(app: &mut App) {
    let recv = match app.stream_rx.take() {
        Some(recv) => recv,
        None => return,
    };
    while let Ok(chunk) = recv.try_recv() {
        match chunk {
            Chunk::Reasoning(text) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.reasoning.push_str(&text);
                }
            }
            Chunk::Content(text) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.content.push_str(&text);
                }
            }
            Chunk::ToolCall(desc) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.tool_call = desc;
                }
            }
            Chunk::Done(stats) => {
                app.stats = Some(stats);
                app.state = State::Idle;
                start_memory_update(app);
            }
            Chunk::Err(err) => {
                if let Some(msg) = app.history.last_mut() {
                    msg.content = format!("Error: {err}");
                }
                app.state = State::Idle;
            }
        }
    }
    if app.state == State::Streaming {
        app.stream_rx = Some(recv);
    }
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
            Ok(MemoryEvent::Append { phase, text }) => {
                match app.memory_log.last_mut() {
                    Some(last) if last.0 == phase => last.1.push_str(&text),
                    _ => app.memory_log.push((phase.to_string(), text)),
                }
            }
            Ok(MemoryEvent::Done) => {
                let graph = memory::load_graph();
                app.memory_toon = toon::encode(&memory::graph_to_json(&graph), None);
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

fn start_memory_update(app: &mut App) {
    let assistant_content = last_assistant_content(app);
    let user_content = second_to_last_content(app);
    let session_label = session_label(&app.session_path);
    session::append_entry(&app.session_path, "assistant", &assistant_content);
    app.memory_log.clear();
    let (sender, receiver) = std::sync::mpsc::channel::<MemoryEvent>();
    let llm = Arc::clone(&app.memory_llm);
    std::thread::spawn(move || {
        let _ = memory::run_memory_agent(llm, &user_content, &assistant_content, session_label, sender);
    });
    app.memory_rx = Some(receiver);
    app.memory_state = MemoryState::Updating;
}

fn session_label(session_path: &std::path::Path) -> String {
    let stem_os = match session_path.file_stem() {
        Some(stem_os) => stem_os,
        None => return "unknown".to_string(),
    };
    match stem_os.to_str() {
        Some(stem_text) => stem_text.to_string(),
        None => "unknown".to_string(),
    }
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
