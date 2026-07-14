mod client;

#[derive(Clone, Debug, PartialEq)]
pub struct NewsItem {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub trait SearchClient: Send + Sync {
    fn search_many(&self, queries: &[String]) -> Option<String>;
    fn search_news(&self, _topic: &str, _days: usize) -> Result<Vec<NewsItem>, String> {
        Err("news search unavailable".to_string())
    }
    fn extract_url(&self, _url: &str) -> Result<String, String> {
        Err("URL extraction unavailable".to_string())
    }
}

pub(crate) fn new_search_client(api_key: String) -> impl SearchClient {
    client::TavilyClient::new(api_key)
}
