mod check;
mod client;

pub use check::decide_queries;

pub trait SearchClient: Send + Sync {
    fn search_many(&self, queries: &[String]) -> Option<String>;
}

pub(crate) fn new_search_client(api_key: String) -> impl SearchClient {
    client::TavilyClient::new(api_key)
}
