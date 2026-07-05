mod client;
mod embed;

use std::error::Error;
use std::sync::mpsc::Sender;

pub struct Stats {
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub cost_usd: Option<f64>,
    pub ttft_ms: u128,
}

pub enum Chunk {
    Reasoning(String),
    Content(String),
    ToolCall(String),
    MemoryLog { phase: &'static str, text: String },
    Attribution(String),
    Done(Stats),
    Err(String),
}

#[allow(clippy::min_ident_chars)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub args: String,
}

pub enum TurnOutcome {
    Done(Stats),
    ToolCalls(Vec<ToolCallRequest>),
}

pub trait LlmClient: Send + Sync {
    #[allow(dead_code)]
    fn collect(&self, prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>>;
    fn stream(&self, prompt: String, system: Option<String>, sender: Sender<Chunk>);
    fn stream_turn(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        sender: &std::sync::mpsc::Sender<Chunk>,
    ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>>;
}

pub trait EmbeddingClient: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>>;
}

pub(crate) fn new_client(api_key: String, model: String) -> impl LlmClient {
    client::OpenRouterClient::new(api_key, model)
}

pub(crate) fn new_embed_client(api_key: String, model: String) -> impl EmbeddingClient {
    embed::OpenRouterEmbedClient::new(api_key, model)
}
