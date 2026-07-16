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
) -> NewsScanSummary {
    let interests = db::top_interests(NEWS_INTEREST_COUNT);
    register_scan_session();
    let mut summary = NewsScanSummary {
        interests: interests.len(),
        findings: 0,
    };
    let Some(search_client) = search else {
        runlog::log("news", "news search unavailable");
        return summary;
    };
    let Ok(conn) = db::open_connection() else {
        runlog::log("news", "database unavailable");
        return summary;
    };
    for interest in interests {
        summary.findings += scan_interest(&conn, embed_client, search_client, &interest);
    }
    if summary.findings > 0 {
        db::sync_push_best_effort("news");
    }
    summary
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
) -> usize {
    let results = match search_client.search_news(&interest.topic, NEWS_RECENCY_DAYS) {
        Ok(items) => items,
        Err(reason) => {
            runlog::log(
                "news",
                &format!("search failed for {}: {reason}", interest.topic),
            );
            return 0;
        }
    };
    process_interest_results(conn, embed_client, interest, &results)
}

pub(crate) fn process_interest_results(
    conn: &Connection,
    embed_client: &dyn EmbeddingClient,
    interest: &Interest,
    results: &[NewsItem],
) -> usize {
    let mut recorded = 0usize;
    for item in results {
        let Ok(vector) = embed_result(embed_client, item) else {
            continue;
        };
        let Ok(similarity) = db::nearest_live_edge_similarity(conn, &vector) else {
            continue;
        };
        let novelty = novelty_from_similarity(similarity);
        if novelty < NEWS_NOVELTY_MIN {
            continue;
        }
        let _write_guard = db::acquire_write_lock();
        let Ok(finding_id) = db::insert_finding(conn, &interest.topic, item, novelty) else {
            continue;
        };
        if let Err(reason) =
            db::record_news_graph_material(conn, finding_id, interest, item, &vector)
        {
            runlog::log("news", &format!("graph record failed: {reason}"));
        }
        runlog::log(
            "news",
            &format!(
                "[news] finding: {} (interest {}, novelty {:.2})",
                item.title, interest.topic, novelty
            ),
        );
        recorded += 1;
    }
    recorded
}

fn embed_result(
    embed_client: &dyn EmbeddingClient,
    item: &NewsItem,
) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let text = format!("{} {}", item.title, item.snippet);
    embed::embed_text(embed_client, &text)
}

fn novelty_from_similarity(similarity: Option<f64>) -> f64 {
    (1.0 - similarity.unwrap_or(0.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::EmbeddingClient;
    use turso::Builder;

    struct TestEmbed;

    impl EmbeddingClient for TestEmbed {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
            let mut vectors = Vec::new();
            for text in texts {
                if text.contains("near") {
                    vectors.push(vec![1.0, 0.0]);
                } else {
                    vectors.push(vec![0.0, 1.0]);
                }
            }
            Ok(vectors)
        }
    }

    #[test]
    fn threshold_boundary_is_kept() {
        assert_eq!(novelty_from_similarity(Some(0.6)), NEWS_NOVELTY_MIN);
        assert!(novelty_from_similarity(Some(0.61)) < NEWS_NOVELTY_MIN);
    }

    #[test]
    fn near_result_is_discarded_and_distant_result_is_recorded() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        let interest = Interest {
            node_id: "rust".to_string(),
            topic: "Rust".to_string(),
            origin: "user".to_string(),
            created_at: "2026-07-14T00:00:00Z".to_string(),
            strength: 1.0,
        };
        let results = vec![news_item("near update"), news_item("distant update")];
        let count = process_interest_results(&conn, &TestEmbed, &interest, &results);
        assert_eq!(count, 1);
        let findings = db::unsurfaced_findings(&conn, 10).unwrap_or_default();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].title, "distant update");
        db::block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT provenance, confidence FROM edges
                     WHERE provenance = 'web-surfaced'",
                    (),
                )
                .await
            else {
                unreachable!("news edge query succeeds")
            };
            let Ok(Some(row)) = rows.next().await else {
                unreachable!("news edge exists")
            };
            assert_eq!(
                db::text(row.get_value(0).unwrap_or(turso::Value::Null)),
                "web-surfaced"
            );
            assert_eq!(
                db::real(row.get_value(1).unwrap_or(turso::Value::Null)),
                0.5
            );
        });
    }

    async fn test_connection() -> Connection {
        let Ok(database) = Builder::new_local(":memory:").build().await else {
            unreachable!("in-memory database builds")
        };
        let Ok(conn) = database.connect() else {
            unreachable!("in-memory database connects")
        };
        conn
    }

    fn seed_test_graph(conn: &Connection) {
        db::block_on(async {
            let _ = conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '', date TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', embedding BLOB, PRIMARY KEY (source, target))", ()).await;
            let _ = conn.execute("CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn
                .execute(
                    "INSERT INTO nodes (id, node_type) VALUES ('rust', 'interest')",
                    (),
                )
                .await;
            let _ = conn.execute("INSERT INTO edges (source, target, embedding) VALUES ('known', 'fact', vector32('[1.0,0.0]'))", ()).await;
        });
    }

    fn news_item(title: &str) -> NewsItem {
        NewsItem {
            title: title.to_string(),
            url: format!("https://example.com/{title}"),
            snippet: "details".to_string(),
        }
    }
}
