    #[test]
    fn timing_format_includes_step_and_zero_milliseconds() {
        assert_eq!(super::format_timing("test step", 0), "test step took 0ms");
    }
