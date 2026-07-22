use std::time::Duration;

use crate::memory::{self, Finding, PingPolicy};
use crate::runlog;
use crate::session;
use crate::timeutil;

use super::bot::BotTransport;

pub(crate) const PING_NOVELTY_MIN: f64 = 0.5; // replay-tunable.
const MAX_PINGS_PER_DAY: usize = 2; // replay-tunable.
const PING_COOLDOWN: Duration = Duration::from_secs(3 * 60 * 60); // replay-tunable.
const PING_CANDIDATE_LIMIT: usize = 20;

struct PingGate {
    day: String,
    count: usize,
    last_ping: Option<u64>,
}

impl PingGate {
    fn new() -> Self {
        Self {
            day: String::new(),
            count: 0,
            last_ping: None,
        }
    }

    fn allows(&mut self, today: &str, now: u64) -> bool {
        if self.day != today {
            self.day = today.to_string();
            self.count = 0;
            self.last_ping = None;
        }
        if self.count >= MAX_PINGS_PER_DAY {
            return false;
        }
        match self.last_ping {
            Some(last) => now.saturating_sub(last) >= PING_COOLDOWN.as_secs(),
            None => true,
        }
    }

    fn record(&mut self, now: u64) {
        self.count += 1;
        self.last_ping = Some(now);
    }
}

pub struct PingCycle {
    gate: PingGate,
    chat_id: i64,
}

struct PingMoment {
    today: String,
    epoch_seconds: u64,
}

impl PingCycle {
    pub fn new(chat_id: i64) -> Self {
        Self {
            gate: PingGate::new(),
            chat_id,
        }
    }

    pub fn maybe_send(&mut self, transport: &dyn BotTransport) {
        let moment = PingMoment {
            today: timeutil::today_iso(),
            epoch_seconds: timeutil::epoch_seconds_now(),
        };
        if !self.gate.allows(&moment.today, moment.epoch_seconds) {
            return;
        }
        let now = timeutil::timestamp_now();
        let Ok(conn) = memory::open_connection() else {
            return;
        };
        let Ok(policy) = memory::load_ping_policy(&conn, &now) else {
            return;
        };
        let Ok(findings) = memory::unsurfaced_findings(&conn, PING_CANDIDATE_LIMIT) else {
            return;
        };
        let Some(finding) = select_finding(&findings, &policy, &now) else {
            return;
        };
        self.surface(transport, &conn, &finding, &moment);
    }

    fn surface(
        &mut self,
        transport: &dyn BotTransport,
        conn: &turso::Connection,
        finding: &Finding,
        moment: &PingMoment,
    ) {
        let session_id = format!("tg-{}", moment.today);
        let turn_id = format!("ping-finding-{}", finding.finding_id);
        {
            let _guard = memory::acquire_write_lock();
            if memory::mark_finding_surfaced(conn, finding.finding_id, &session_id, &turn_id)
                .is_err()
            {
                return;
            }
        }
        self.gate.record(moment.epoch_seconds);
        let message = ping_message(finding);
        super::send_answer(transport, self.chat_id, &message);
        session::append_entry(
            &super::session_path_for_id(&session_id),
            "assistant",
            &message,
        );
        runlog::log(
            "autonomy",
            &format!(
                "[autonomy] ping — {} (interest {}, novelty {:.2})",
                finding.title, finding.interest_topic, finding.novelty
            ),
        );
        memory::sync_push_best_effort("proactive ping");
    }
}

fn select_finding(findings: &[Finding], policy: &PingPolicy, now: &str) -> Option<Finding> {
    if policy.muted_until.as_str() > now {
        return None;
    }
    for finding in findings {
        if finding.novelty < PING_NOVELTY_MIN {
            continue;
        }
        if topic_is_diverted(&finding.interest_topic, &policy.diverts) {
            continue;
        }
        return Some(finding.clone());
    }
    None
}

fn topic_is_diverted(topic: &str, diverts: &[(String, String)]) -> bool {
    for (diverted, _) in diverts {
        if diverted.eq_ignore_ascii_case(topic) {
            return true;
        }
    }
    false
}

fn ping_message(finding: &Finding) -> String {
    format!(
        "On {}: {}\n{}",
        finding.interest_topic, finding.title, finding.url
    )
}

#[cfg(test)]
#[path = "ping_tests.rs"]
mod tests;
