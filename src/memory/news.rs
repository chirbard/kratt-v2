use std::error::Error;

use serde_json::json;
use turso::{Connection, params};

use crate::api::EmbeddingClient;
use crate::runlog;
use crate::web::{NewsItem, SearchClient};

use super::db::{self, Interest};
use super::embed;
use super::graph::{PROVENANCE_USER_CONFIRMED, PROVENANCE_WEB_CONFIRMED, edge_ceiling};
use super::ledger::{self, Actor, EdgeRef, Event, EventContext};

const NEWS_INTEREST_COUNT: usize = 3; // replay-tunable.
const NEWS_RECENCY_DAYS: usize = 7; // replay-tunable.
const NEWS_NOVELTY_MIN: f64 = 0.4; // replay-tunable.
const CORROBORATION_MIN: f64 = 0.75; // replay-tunable.
const CORROBORATION_CONFIDENCE_FLOOR: f64 = 0.6; // replay-tunable.
const CORROBORATION_STABILITY_BONUS: f64 = 0.1; // replay-tunable.

struct PromotionUpdate<'edge> {
    edge: &'edge super::graph::Edge,
    provenance: &'edge str,
    confidence: f64,
    stability: f64,
}

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
    if summary.interests > 0 {
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
    if matches!(closest, Some(edge) if edge.similarity >= CORROBORATION_MIN) {
        promote_news_match(conn, closest, item);
        return 0;
    }
    let novelty = match closest {
        Some(edge) => (1.0 - edge.similarity).clamp(0.0, 1.0),
        None => 1.0,
    };
    record_novel_finding(conn, interest, item, &vector, novelty)
}

fn promote_news_match(conn: &Connection, closest: Option<&db::NearestLiveEdge>, item: &NewsItem) {
    let Some(edge) = closest else {
        return;
    };
    if let Err(reason) = promote_corroborated_edge(conn, edge, &item.title) {
        runlog::log("news", &format!("promotion failed: {reason}"));
    }
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

fn promote_corroborated_edge(
    conn: &Connection,
    nearest: &db::NearestLiveEdge,
    title: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Some(edge) = db::load_edge(conn, &nearest.source, &nearest.target)? else {
        return Ok(());
    };
    if edge.expired_at.is_some() {
        return Ok(());
    }
    let from_tier = edge.provenance.clone();
    let to_tier = promoted_tier(&from_tier);
    let confidence = promoted_confidence(edge.confidence, &from_tier, to_tier);
    let stability = db::round2((edge.stability + CORROBORATION_STABILITY_BONUS).min(1.0));
    let payload = json!({"from": from_tier, "to": to_tier, "title": title}).to_string();
    let event = Event::promoted(
        Actor::new("system", "news"),
        EdgeRef::new(&edge.source, &edge.target),
        EventContext::new(&format!("news:{}", crate::timeutil::today_iso()), ""),
        payload,
    );
    let update = PromotionUpdate {
        edge: &edge,
        provenance: to_tier,
        confidence,
        stability,
    };
    write_promotion(conn, &update, &event)?;
    runlog::log(
        "news",
        &format!(
            "[epistemics] promoted: {} → {} ({} → {}, news \"{}\")",
            edge.source, edge.target, edge.provenance, to_tier, title
        ),
    );
    Ok(())
}

fn promoted_tier(tier: &str) -> &str {
    if matches!(tier, PROVENANCE_USER_CONFIRMED | PROVENANCE_WEB_CONFIRMED) {
        tier
    } else {
        PROVENANCE_WEB_CONFIRMED
    }
}

fn promoted_confidence(confidence: f64, from_tier: &str, to_tier: &str) -> f64 {
    let proposed = if matches!(
        from_tier,
        PROVENANCE_USER_CONFIRMED | PROVENANCE_WEB_CONFIRMED
    ) {
        confidence
    } else {
        confidence.max(CORROBORATION_CONFIDENCE_FLOOR)
    };
    db::round2(proposed.min(edge_ceiling(to_tier, None)))
}

fn write_promotion<'edge>(
    conn: &Connection,
    update: &PromotionUpdate<'edge>,
    event: &Event,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    db::begin_transaction(conn)?;
    let result = db::block_on(async {
        conn.execute(
            // Corroboration replaces a derived edge's inherited cap with the
            // web-confirmed tier cap.
            "UPDATE edges SET provenance = ?1, confidence = ?2, stability = ?3, ceiling = NULL
             WHERE source = ?4 AND target = ?5 AND expired_at IS NULL",
            params![
                update.provenance,
                update.confidence,
                update.stability,
                update.edge.source.as_str(),
                update.edge.target.as_str()
            ],
        )
        .await?;
        ledger::record_events_in_tx(conn, std::slice::from_ref(event)).await
    });
    if let Err(reason) = result {
        let _ = db::block_on(conn.execute("ROLLBACK", ()));
        return Err(reason);
    }
    db::commit_transaction(conn)
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
                } else if text.contains("mid") {
                    vectors.push(vec![0.7, 0.714_142_842_9]);
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
        assert_eq!(
            promoted_confidence(0.95, "legacy", PROVENANCE_WEB_CONFIRMED),
            0.8
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn near_result_promotes_and_distant_result_is_recorded() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        set_known_edge(&conn, "synthesized", 0.2, 0.4);
        db::block_on(async {
            let _ = conn
                .execute(
                    "UPDATE edges SET ceiling = 0.25 WHERE source = 'known' AND target = 'fact'",
                    (),
                )
                .await;
        });
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
            let Ok(mut promoted) = conn
                .query(
                    "SELECT provenance, confidence, stability, ceiling FROM edges
                     WHERE source = 'known' AND target = 'fact'",
                    (),
                )
                .await
            else {
                unreachable!("promoted edge query succeeds")
            };
            let Ok(Some(row)) = promoted.next().await else {
                unreachable!("promoted edge exists")
            };
            assert_eq!(
                db::text(row.get_value(0).unwrap_or(turso::Value::Null)),
                "web-confirmed"
            );
            assert_eq!(
                db::real(row.get_value(1).unwrap_or(turso::Value::Null)),
                0.6
            );
            assert_eq!(
                db::real(row.get_value(2).unwrap_or(turso::Value::Null)),
                0.5
            );
            assert!(matches!(
                row.get_value(3).unwrap_or(turso::Value::Real(1.0)),
                turso::Value::Null
            ));
            let Ok(mut events) = conn
                .query(
                    "SELECT payload FROM events WHERE event_type = 'promoted'",
                    (),
                )
                .await
            else {
                unreachable!("promotion event query succeeds")
            };
            let Ok(Some(event)) = events.next().await else {
                unreachable!("promotion event exists")
            };
            let payload = db::text(event.get_value(0).unwrap_or(turso::Value::Null));
            assert!(payload.contains(r#""from":"synthesized""#), "{payload}");
            assert!(payload.contains(r#""to":"web-confirmed""#), "{payload}");
        });
    }

    #[test]
    fn repeated_corroboration_only_bumps_stability_for_high_tiers() {
        for (tier, confidence) in [("user-confirmed", 0.9), ("web-confirmed", 0.7)] {
            let conn = db::block_on(test_connection());
            seed_test_graph(&conn);
            set_known_edge(&conn, tier, confidence, 0.7);
            let count = process_interest_results(
                &conn,
                &TestEmbed,
                &test_interest(),
                &[news_item("near update")],
            );
            assert_eq!(count, 0);
            let Ok(Some(edge)) = db::load_edge(&conn, "known", "fact") else {
                unreachable!("known edge exists")
            };
            assert_eq!(edge.provenance, tier);
            assert_eq!(edge.confidence, confidence);
            assert_eq!(edge.stability, 0.8);
        }
    }

    #[test]
    fn mid_similarity_is_discarded_without_promotion() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        set_known_edge(&conn, "legacy", 0.4, 0.4);
        let count = process_interest_results(
            &conn,
            &TestEmbed,
            &test_interest(),
            &[news_item("mid update")],
        );
        assert_eq!(count, 0);
        let Ok(Some(edge)) = db::load_edge(&conn, "known", "fact") else {
            unreachable!("known edge exists")
        };
        assert_eq!(edge.provenance, "legacy");
        assert_eq!(edge.confidence, 0.4);
        assert_eq!(edge.stability, 0.4);
        let events = promotion_event_count(&conn);
        assert_eq!(events, 0);
    }

    #[test]
    fn nearest_live_edges_return_identity_in_similarity_order() {
        let conn = db::block_on(test_connection());
        seed_test_graph(&conn);
        db::block_on(async {
            let _ = conn.execute("INSERT INTO edges (source, target, provenance, confidence, embedding) VALUES ('second', 'claim', 'web-surfaced', 0.4, vector32('[0.8,0.6]'))", ()).await;
        });
        let edges = db::nearest_live_edges(&conn, &[1.0, 0.0], 2).unwrap_or_default();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].source, "known");
        assert_eq!(edges[0].target, "fact");
        assert_eq!(edges[0].provenance, "legacy");
        assert_eq!(edges[0].confidence, 0.0);
        assert!(edges[0].similarity > edges[1].similarity);
        assert_eq!(edges[1].source, "second");
        assert_eq!(edges[1].target, "claim");
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
            let _ = conn.execute("CREATE TABLE edges (source TEXT NOT NULL, target TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', stability REAL NOT NULL DEFAULT 0.0, confidence REAL NOT NULL DEFAULT 0.0, importance REAL NOT NULL DEFAULT 0.0, subject INTEGER NOT NULL DEFAULT 0, date TEXT NOT NULL DEFAULT '', occurred_at TEXT, expired_at TEXT, memory_type TEXT NOT NULL DEFAULT 'semantic', origin_session TEXT NOT NULL DEFAULT '', provenance TEXT NOT NULL DEFAULT 'legacy', ceiling REAL, embedding BLOB, PRIMARY KEY (source, target))", ()).await;
            let _ = conn.execute("CREATE TABLE findings (id INTEGER PRIMARY KEY AUTOINCREMENT, interest_topic TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL DEFAULT '', novelty REAL NOT NULL, found_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate', surfaced_at TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '', actor_type TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', target TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1.0, payload TEXT NOT NULL DEFAULT '')", ()).await;
            let _ = conn.execute("CREATE TABLE edge_actor_views (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL, views REAL NOT NULL DEFAULT 0, last_visited_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id, source, target))", ()).await;
            let _ = conn.execute("CREATE TABLE actor_positions (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, sum_vector BLOB, total_weight REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (actor_type, actor_id))", ()).await;
            let _ = conn.execute("CREATE TABLE actor_position_history (actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, embedding BLOB, recorded_at TEXT NOT NULL)", ()).await;
            let _ = conn
                .execute(
                    "INSERT INTO nodes (id, node_type) VALUES ('rust', 'interest')",
                    (),
                )
                .await;
            let _ = conn.execute("INSERT INTO edges (source, target, embedding) VALUES ('known', 'fact', vector32('[1.0,0.0]'))", ()).await;
        });
    }

    fn set_known_edge(conn: &Connection, provenance: &str, confidence: f64, stability: f64) {
        db::block_on(async {
            let _ = conn
                .execute(
                    "UPDATE edges SET provenance = ?1, confidence = ?2, stability = ?3
                     WHERE source = 'known' AND target = 'fact'",
                    params![provenance, confidence, stability],
                )
                .await;
        });
    }

    fn promotion_event_count(conn: &Connection) -> usize {
        db::block_on(async {
            let Ok(mut rows) = conn
                .query(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'promoted'",
                    (),
                )
                .await
            else {
                return 0;
            };
            let Ok(Some(row)) = rows.next().await else {
                return 0;
            };
            db::real(row.get_value(0).unwrap_or(turso::Value::Null)) as usize
        })
    }

    fn test_interest() -> Interest {
        Interest {
            node_id: "rust".to_string(),
            topic: "Rust".to_string(),
            origin: "user".to_string(),
            created_at: "2026-07-14T00:00:00Z".to_string(),
            strength: 1.0,
        }
    }

    fn news_item(title: &str) -> NewsItem {
        NewsItem {
            title: title.to_string(),
            url: format!("https://example.com/{title}"),
            snippet: "details".to_string(),
        }
    }
}
