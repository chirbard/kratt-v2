    use super::*;

    fn window(raw: &str) -> DreamWindow {
        match DreamWindow::parse(raw) {
            Ok(win) => win,
            Err(err) => panic!("valid window {raw}: {err}"),
        }
    }

    fn gate(raw: &str) -> NightGate {
        NightGate::new(window(raw), Duration::from_secs(1200), 3)
    }

    #[test]
    fn window_parses_and_bounds_are_end_exclusive() {
        let win = window("3-6");
        assert!(win.contains(3));
        assert!(win.contains(4));
        assert!(win.contains(5));
        assert!(!win.contains(2));
        assert!(!win.contains(6));
    }

    #[test]
    fn window_wraps_across_midnight() {
        let win = window("23-5");
        for hour in [23, 0, 1, 2, 3, 4] {
            assert!(win.contains(hour), "hour {hour} should be inside");
        }
        assert!(!win.contains(5));
        assert!(!win.contains(22));
    }

    #[test]
    fn malformed_window_is_an_error_not_a_default() {
        assert!(DreamWindow::parse("banana").is_err());
        assert!(DreamWindow::parse("3").is_err());
        assert!(DreamWindow::parse("3-99").is_err());
        assert!(DreamWindow::parse("4-4").is_err());
    }

    #[test]
    fn dispatches_inside_window_when_cooldown_elapsed_and_under_cap() {
        let mut gate = gate("3-6");
        assert!(gate.should_evaluate(4, "2026-07-10", Instant::now()));
    }

    #[test]
    fn never_dispatches_outside_window() {
        let mut gate = gate("3-6");
        assert!(!gate.should_evaluate(2, "2026-07-10", Instant::now()));
        assert!(!gate.should_evaluate(6, "2026-07-10", Instant::now()));
    }

    #[test]
    fn does_not_dispatch_before_cooldown_elapses() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_action(base);
        let soon = base + Duration::from_secs(60);
        assert!(!gate.should_evaluate(4, "2026-07-10", soon));
        let later = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn does_not_dispatch_at_cap() {
        let mut gate = gate("3-6");
        let mut now = Instant::now();
        for _ in 0..3 {
            assert!(gate.should_evaluate(4, "2026-07-10", now));
            gate.record_action(now);
            now += Duration::from_secs(1200);
        }
        assert!(!gate.should_evaluate(4, "2026-07-10", now));
    }

    #[test]
    fn cap_resets_on_a_new_day() {
        let mut gate = gate("3-6");
        let mut now = Instant::now();
        for _ in 0..3 {
            assert!(gate.should_evaluate(4, "2026-07-10", now));
            gate.record_action(now);
            now += Duration::from_secs(1200);
        }
        assert!(!gate.should_evaluate(4, "2026-07-10", now));
        // Re-entering the window on a new day clears the counter.
        assert!(gate.should_evaluate(4, "2026-07-11", now));
    }

    #[test]
    fn rest_clears_dirty_and_evaluation_stops_until_marked_dirty() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_rest();
        // Same night, cooldown irrelevant: inputs unchanged, never re-evaluate.
        let later = base + Duration::from_secs(3600);
        assert!(!gate.should_evaluate(4, "2026-07-10", later));
        // A state change (turn handled) re-enables evaluation.
        gate.dirty = true;
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn a_new_day_sets_dirty_and_resets_the_nightly_dream_flag() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        // Establish the night first — entering the window on a new day is
        // itself a dirtying event, so the rest must come after.
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.dreamed_tonight = true;
        gate.record_rest();
        assert!(!gate.should_evaluate(4, "2026-07-10", base));
        // Window re-entry on the next day: dirty again, dream flag cleared.
        assert!(gate.should_evaluate(4, "2026-07-11", base));
        assert!(!gate.dreamed_tonight);
    }

    #[test]
    fn actions_leave_dirty_set_so_the_follow_up_evaluation_happens() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_action(base);
        // Cooldown gates the timing, but the state is still dirty.
        let later = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, "2026-07-10", later));
    }

    #[test]
    fn failed_dream_allows_one_cooldown_gated_retry() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        let day = timeutil::today_iso();
        assert!(gate.should_evaluate(4, &day, base));
        gate.record_dream(base, false);
        assert!(!gate.dreamed_tonight);
        assert!(gate.can_attempt_dream());
        assert!(!gate.should_evaluate(4, &day, base + Duration::from_secs(60)));
        let retry = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, &day, retry));
        gate.record_dream(retry, false);
        assert!(gate.dreamed_tonight);
        assert!(!gate.can_attempt_dream());
    }

    #[test]
    fn successful_dream_closes_the_nightly_gate() {
        let mut gate = gate("3-6");
        let now = Instant::now();
        gate.record_dream(now, true);
        assert!(gate.dreamed_tonight);
        assert!(!gate.can_attempt_dream());
        assert_eq!(gate.dream_attempts, 1);
    }

    #[test]
    fn pass_errors_use_stable_reason_classes() {
        assert_eq!(
            classify_pass_error("Input length exceeds maximum context length"),
            "context_length"
        );
        assert_eq!(classify_pass_error("HTTP 500"), "provider_error");
    }

    #[test]
    fn news_scan_latch_rearms_on_day_rollover() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.scanned_news_today = true;
        assert!(gate.scanned_news_today);
        let next = base + Duration::from_secs(1200);
        assert!(gate.should_evaluate(4, "2026-07-11", next));
        assert!(!gate.scanned_news_today);
    }

    #[test]
    fn empty_curiosity_consumes_a_slot_and_settles() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_curiosity(base, false);
        assert_eq!(gate.count, 1);
        let later = base + Duration::from_secs(1200);
        assert!(!gate.should_evaluate(4, "2026-07-10", later));
        assert_eq!(gate.last_curiosity_had_yield, Some(false));
    }

    #[test]
    fn skipped_action_consumes_cooldown_but_not_cap() {
        let mut gate = gate("3-6");
        let base = Instant::now();
        assert!(gate.should_evaluate(4, "2026-07-10", base));
        gate.record_skipped(base);
        assert_eq!(gate.count, 0, "cap not consumed");
        let soon = base + Duration::from_secs(60);
        assert!(
            !gate.should_evaluate(4, "2026-07-10", soon),
            "cooldown consumed"
        );
    }
