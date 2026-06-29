use std::error::Error;

use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

use super::EmbeddingClient;

const OPENROUTER_EMBED_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const EMBED_DIM: usize = 1024;

#[derive(Deserialize)]
struct EmbedData {
    index: usize,
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

pub(super) struct OpenRouterEmbedClient {
    api_key: String,
    model: String,
}

impl OpenRouterEmbedClient {
    pub(super) fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }
}

impl EmbeddingClient for OpenRouterEmbedClient {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
        let response = post_embed_request(&self.api_key, &self.model, texts)?;
        collect_embeddings(response, texts.len())
    }
}

fn post_embed_request(
    api_key: &str,
    model: &str,
    texts: &[String],
) -> Result<reqwest::blocking::Response, Box<dyn Error + Send + Sync>> {
    let response = Client::new()
        .post(OPENROUTER_EMBED_URL)
        .bearer_auth(api_key)
        .json(&json!({
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }))
        .send()?;
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().unwrap_or_default();
    Err(format!("HTTP {status}: {body}").into())
}

fn collect_embeddings(
    response: reqwest::blocking::Response,
    text_count: usize,
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let parsed: EmbedResponse = response.json()?;
    let mut by_index: Vec<Option<Vec<f64>>> = Vec::new();
    let mut index = 0usize;
    while index < text_count {
        by_index.push(None);
        index += 1;
    }
    for item in parsed.data {
        validate_embedding(&item.embedding)?;
        if item.index >= by_index.len() {
            return Err("embedding index out of range".into());
        }
        by_index[item.index] = Some(item.embedding);
    }
    finalize_embeddings(by_index)
}

fn validate_embedding(embedding: &[f64]) -> Result<(), Box<dyn Error + Send + Sync>> {
    if embedding.len() == EMBED_DIM {
        return Ok(());
    }
    Err(format!("wrong embedding length: {}", embedding.len()).into())
}

fn finalize_embeddings(
    by_index: Vec<Option<Vec<f64>>>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let mut embeddings = Vec::new();
    for maybe_embedding in by_index {
        let embedding = match maybe_embedding {
            Some(embedding) => embedding,
            None => return Err("missing embedding".into()),
        };
        embeddings.push(embedding);
    }
    Ok(embeddings)
}
