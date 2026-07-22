    use super::markdown_to_telegram_html;

    #[test]
    fn headers_become_bold_lines() {
        assert_eq!(
            markdown_to_telegram_html("# Big title\ntext"),
            "<b>Big title</b>\ntext"
        );
        assert_eq!(markdown_to_telegram_html("### Small"), "<b>Small</b>");
    }

    #[test]
    fn bullets_become_dots_and_bold_and_code_map_to_tags() {
        assert_eq!(
            markdown_to_telegram_html("- item with **bold** and `code`"),
            "• item with <b>bold</b> and <code>code</code>"
        );
        assert_eq!(markdown_to_telegram_html("  - nested"), "  • nested");
    }

    #[test]
    fn unpaired_markers_stay_literal() {
        assert_eq!(markdown_to_telegram_html("a ** b"), "a ** b");
        assert_eq!(markdown_to_telegram_html("2 ** 3 ** 4 ** 5"), "2 <b> 3 </b> 4 ** 5");
    }

    #[test]
    fn html_specials_are_escaped() {
        assert_eq!(
            markdown_to_telegram_html("if a < b && b > c"),
            "if a &lt; b &amp;&amp; b &gt; c"
        );
    }

    #[test]
    fn code_fences_become_pre_blocks_without_the_fence_lines() {
        let text = "before\n```rust\nlet x = 1;\n```\nafter";
        assert_eq!(
            markdown_to_telegram_html(text),
            "before\n<pre>let x = 1;</pre>\nafter"
        );
    }

    #[test]
    fn tables_are_wrapped_in_pre_for_monospace_alignment() {
        let text = "| a | b |\n|---|---|\n| 1 | 2 |\ntail";
        assert_eq!(
            markdown_to_telegram_html(text),
            "<pre>| a | b |\n|---|---|\n| 1 | 2 |</pre>\ntail"
        );
    }
