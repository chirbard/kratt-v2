mod client;
mod embed;

use std::error::Error;
use std::sync::Arc;
use std::sync::mpsc::Sender;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
// OpenAI-compatible base of a local Ollama server; `/chat/completions` is
// appended to reach the endpoint. Override the base with OLLAMA_URL.
const DEFAULT_OLLAMA_BASE: &str = "http://localhost:11434/v1";

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

    fn stream_traced(
        &self,
        prompt: String,
        system: Option<String>,
        sender: Sender<Chunk>,
        _span: &crate::trace::TraceSpan,
    ) {
        self.stream(prompt, system, sender);
    }

    fn stream_turn_traced(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        sender: &Sender<Chunk>,
        _span: &crate::trace::TraceSpan,
    ) -> Result<TurnOutcome, Box<dyn Error + Send + Sync>> {
        self.stream_turn(messages, tools, sender)
    }
}

pub trait EmbeddingClient: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>>;
}

pub(crate) fn new_client(api_key: String, model: String) -> impl LlmClient {
    client::OpenRouterClient::new(api_key, model, OPENROUTER_URL.to_string())
}

// Resolves a model spec into the right client. A plain spec
// ("openai/gpt-oss-20b:nitro") targets OpenRouter; an "ollama:" prefix
// ("ollama:minicpm5-1b") targets a local Ollama server's OpenAI-compatible
// endpoint, base URL from OLLAMA_URL or the default localhost port.
pub(crate) fn resolve_model_client(spec: &str, api_key: &str) -> Arc<dyn LlmClient> {
    let ollama_base = match std::env::var("OLLAMA_URL") {
        Ok(base) => base,
        Err(_) => DEFAULT_OLLAMA_BASE.to_string(),
    };
    let (model, url) = resolve_model_spec(spec, &ollama_base);
    Arc::new(client::OpenRouterClient::new(api_key.to_string(), model, url))
}

// Pure spec → (model, url) split, no env access, so the routing is unit-testable.
fn resolve_model_spec(spec: &str, ollama_base: &str) -> (String, String) {
    if let Some(model) = spec.strip_prefix("ollama:") {
        return (model.to_string(), format!("{ollama_base}/chat/completions"));
    }
    (spec.to_string(), OPENROUTER_URL.to_string())
}

pub(crate) fn new_embed_client(api_key: String, model: String) -> impl EmbeddingClient {
    embed::OpenRouterEmbedClient::new(api_key, model)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
