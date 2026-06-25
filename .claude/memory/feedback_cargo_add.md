---
name: feedback-cargo-add
description: User prefers cargo add over direct Cargo.toml edits for adding dependencies
metadata:
  type: feedback
---

Use `cargo add` to add dependencies, not direct Cargo.toml edits.

**Why:** User preference — cargo add is the idiomatic tool for this.

**How to apply:** Any time a new crate needs to be added to a Rust project, run `cargo add <crate>` via Bash instead of editing Cargo.toml manually.
