mod client;

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub struct NewsItem {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub trait SearchClient: Send + Sync {
    fn search_many(&self, queries: &[String]) -> Option<String>;
    #[allow(dead_code)]
    fn search_news(&self, _topic: &str, _days: usize) -> Result<Vec<NewsItem>, String> {
        Err("news search unavailable".to_string())
    }
}

pub(crate) fn new_search_client(api_key: String) -> impl SearchClient {
    client::TavilyClient::new(api_key)
}
