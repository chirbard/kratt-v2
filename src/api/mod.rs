mod client;

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
    Done(Stats),
    Err(String),
}

pub trait LlmClient: Send + Sync {
    fn collect(&self, prompt: &str) -> Result<String, Box<dyn Error + Send + Sync>>;
    fn stream(&self, prompt: String, system: Option<String>, sender: Sender<Chunk>);
}

pub(crate) fn new_client(api_key: String, model: String) -> impl LlmClient {
    client::OpenRouterClient::new(api_key, model)
}
