use std::error::Error;

use turso::Connection;

use crate::api::EmbeddingClient;
use crate::runlog;
use crate::web::{NewsItem, SearchClient};

use super::db::{self, Interest};
use super::embed;

const NEWS_INTEREST_COUNT: usize = 3; // replay-tunable.
const NEWS_RECENCY_DAYS: usize = 7; // replay-tunable.
const NEWS_NOVELTY_MIN: f64 = 0.4; // replay-tunable.

#[derive(Default)]
pub(crate) struct NewsScanSummary {
    pub interests: usize,
    pub findings: usize,
}

pub(crate) fn run_news_scan(
    embed_client: &dyn EmbeddingClient,
    search: Option<&dyn SearchClient>,
) -> Result<NewsScanSummary, Box<dyn Error + Send + Sync>> {
    let interests = db::top_interests(NEWS_INTEREST_COUNT);
    register_scan_session();
    let mut summary = NewsScanSummary {
        interests: interests.len(),
        findings: 0,
    };
    if interests.is_empty() {
        return Ok(summary);
    }
    let Some(search_client) = search else {
        return Err("news search unavailable".into());
    };
    let conn = db::open_connection()?;
    let mut successful_searches = 0usize;
    let mut last_error = None;
    for interest in interests {
        match scan_interest(&conn, embed_client, search_client, &interest) {
            Ok(findings) => {
                successful_searches += 1;
                summary.findings += findings;
            }
            Err(err) => {
                runlog::log("news", &err.to_string());
                last_error = Some(err);
            }
        }
    }
    if successful_searches == 0
        && let Some(err) = last_error
    {
        return Err(err);
    }
    if summary.interests > 0 {
        db::sync_push_best_effort("news");
    }
    Ok(summary)
}

fn register_scan_session() {
    let created_at = db::timestamp_now();
    let session_id = format!("news:{}", crate::timeutil::today_iso());
    db::register_news_session(&session_id, &created_at);
}

fn scan_interest(
    conn: &Connection,
    embed_client: &dyn EmbeddingClient,
    search_client: &dyn SearchClient,
    interest: &Interest,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let results = match search_client.search_news(&interest.topic, NEWS_RECENCY_DAYS) {
        Ok(results) => results,
        Err(reason) => {
            return Err(format!("search failed for {}: {reason}", interest.topic).into());
        }
    };
    Ok(process_interest_results(
        conn,
        embed_client,
        interest,
        &results,
    ))
}

pub(crate) fn process_interest_results(
    conn: &Connection,
    embed_client: &dyn EmbeddingClient,
    interest: &Interest,
    results: &[NewsItem],
) -> usize {
    let mut recorded = 0usize;
    for item in results {
        recorded += process_interest_result(conn, embed_client, interest, item);
    }
    recorded
}

fn process_interest_result(
    conn: &Connection,
    embed_client: &dyn EmbeddingClient,
    interest: &Interest,
    item: &NewsItem,
) -> usize {
    let Ok(vector) = embed_result(embed_client, item) else {
        return 0;
    };
    let _write_guard = db::acquire_write_lock();
    let Ok(nearest) = db::nearest_live_edges(conn, &vector, 1) else {
        return 0;
    };
    let closest = nearest.first();
    let novelty = match closest {
        Some(edge) => (1.0 - edge.similarity).clamp(0.0, 1.0),
        None => 1.0,
    };
    record_novel_finding(conn, interest, item, &vector, novelty)
}

fn record_novel_finding(
    conn: &Connection,
    interest: &Interest,
    item: &NewsItem,
    vector: &[f64],
    novelty: f64,
) -> usize {
    if novelty < NEWS_NOVELTY_MIN {
        return 0;
    }
    let Ok(finding_id) = db::insert_finding(conn, &interest.topic, item, novelty) else {
        return 0;
    };
    if let Err(reason) = db::record_news_graph_material(conn, finding_id, interest, item, vector) {
        runlog::log("news", &format!("graph record failed: {reason}"));
    }
    runlog::log(
        "news",
        &format!(
            "[news] finding: {} (interest {}, novelty {:.2})",
            item.title, interest.topic, novelty
        ),
    );
    1
}

fn embed_result(
    embed_client: &dyn EmbeddingClient,
    item: &NewsItem,
) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let text = format!("{} {}", item.title, item.snippet);
    embed::embed_text(embed_client, &text)
}

#[cfg(test)]
fn novelty_from_similarity(similarity: Option<f64>) -> f64 {
    (1.0 - similarity.unwrap_or(0.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
#[path = "news_tests.rs"]
mod tests;
