# kratt-v2

## Vision

Every AI provider is racing to build the perfect system — more parameters, more compute, closer to flawless. This is not that race.

This system is trying to mimic a human learner: imperfect, but extraordinarily sample-efficient. A person who decides to learn Go never plays a million games. They play once, remember what happened, dream about the positions overnight, read a book, play again — and gradually develop *wisdom* that no cold model possesses, because it was earned. The knowledge graph is long-term memory; dreaming is the overnight consolidation that prevents overfitting to recent experience; exploration is curiosity-driven acquisition; ingestion is focused study of a specific source. The core artifact is a **knowledge crystal** — a weighted graph of natural-language beliefs with confidence, provenance, and decay, portable and loadable by any base LLM without fine-tuning.

At scale, many independent copies of this system run in parallel, each spending its time in a different domain — one learning an unknown game, one reading biology textbooks, one processing case law. No copy is the best in the world at anything. But together they resemble a large community of people with different experiences and hard-won wisdom rather than just accumulated knowledge. That community can put lightning into any stone: a knowledge crystal snapshot from any copy can be loaded into even a modest base LLM and make it reason with the depth of someone who has genuinely spent time in the domain.

Design for the endgame: **millions of edges**, semantic subgraph selection at retrieval time, agent specialisation through repeated exploration, inference-time routing to the best-positioned expert for each query. The current graph size is a temporary convenience. Every architectural decision should hold at scale.

## Code rules

- `cargo clippy` must pass clean — no warnings, no exceptions
- Feature-based directory layout: each feature owns its module, no cross-contamination between features; dependencies flow inward (hexagonal-lite: core has no knowledge of TUI, TUI has no knowledge of memory internals)
- New phases add unit tests for the logic they touch; DB-touching logic should get an in-memory test harness rather than staying untested
- No agent step may require an LLM output longer than ~2k tokens — decompose into a loop of small tool calls (see kratt-roadmap concepts/cognitive-core.md)
