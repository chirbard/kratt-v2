    use super::{RecentEntry, select_recent_turns};

    fn entry(role: &str, content: &str) -> RecentEntry {
        RecentEntry {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: String::new(),
        }
    }

    #[test]
    fn a_lone_assistant_before_the_oldest_pair_is_kept() {
        let entries = vec![
            entry("assistant", "morning report"),
            entry("user", "1"),
            entry("assistant", "scored yesterday +0.5 — thanks"),
        ];
        let recent = select_recent_turns(entries, 2);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "morning report");
    }

    #[test]
    fn a_trailing_report_with_no_pairs_is_still_context() {
        let entries = vec![entry("assistant", "morning report")];
        let recent = select_recent_turns(entries, 2);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].content, "morning report");
    }

    #[test]
    fn plain_pairs_are_unchanged() {
        let entries = vec![
            entry("user", "hi"),
            entry("assistant", "hello"),
            entry("user", "how"),
            entry("assistant", "fine"),
        ];
        let recent = select_recent_turns(entries, 1);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "how");
    }
