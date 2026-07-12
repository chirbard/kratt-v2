mod client;

pub trait SearchClient: Send + Sync {
    fn search_many(&self, queries: &[String]) -> Option<String>;
    fn extract_url(&self, _url: &str) -> Result<String, String> {
        Err("URL extraction unavailable".to_string())
    }
}

pub(crate) fn new_search_client(api_key: String) -> impl SearchClient {
    client::TavilyClient::new(api_key)
}
