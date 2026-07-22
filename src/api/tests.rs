    use super::{OPENROUTER_URL, resolve_model_spec};

    #[test]
    fn plain_spec_targets_openrouter() {
        let (model, url) = resolve_model_spec("openai/gpt-oss-20b:nitro", "http://ignored/v1");
        assert_eq!(model, "openai/gpt-oss-20b:nitro");
        assert_eq!(url, OPENROUTER_URL);
    }

    #[test]
    fn ollama_prefix_targets_local_endpoint() {
        let (model, url) = resolve_model_spec("ollama:minicpm5-1b", "http://localhost:11434/v1");
        assert_eq!(model, "minicpm5-1b");
        assert_eq!(url, "http://localhost:11434/v1/chat/completions");
    }

    #[test]
    fn ollama_base_override_is_respected() {
        let (model, url) = resolve_model_spec("ollama:gemma-e2b", "http://box:9999/v1");
        assert_eq!(model, "gemma-e2b");
        assert_eq!(url, "http://box:9999/v1/chat/completions");
    }
