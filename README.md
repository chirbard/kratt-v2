## using cargo-modules to see boundary violations

for LLM (90% of the time codebase-memory-mcp does the job good enough)

```sh
cargo modules structure
```

DOT graph for human

```sh
cargo modules dependencies --no-externs --no-fns --no-sysroot --no-traits --no-types --no-uses > mods.dot
```
