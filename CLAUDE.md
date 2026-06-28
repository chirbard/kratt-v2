# kratt-v2

## Vision

This is not just a personal assistant. If the memory system proves itself, it is a state-of-the-art in-context continual learning system — the kind of thing that does not exist yet.

The core idea: a weighted knowledge graph where every edge is a natural-language belief with confidence, provenance, and decay. A snapshot of that graph is a **knowledge crystal** — human-readable, versionable, portable, loadable by any base LLM without fine-tuning. The knowledge is separable from the model. That is a fundamentally different artifact from fine-tuned weights, RAG corpora, or prompt templates.

Design for the endgame: **millions of edges**, semantic subgraph selection at retrieval time, agent specialization through repeated exploration, inference-time routing to the best-positioned expert for each query. The current graph size is a temporary convenience. Every architectural decision should hold at scale.

## Code rules

- `cargo clippy` must pass clean — no warnings, no exceptions
- Feature-based directory layout: each feature owns its module, no cross-contamination between features; dependencies flow inward (hexagonal-lite: core has no knowledge of TUI, TUI has no knowledge of memory internals)
