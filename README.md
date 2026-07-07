# kratt-v2

## Turso Cloud sync (backup + multi-machine)

The database (graph, ledger, positions, history) lives in `kratt-data/`. With
Turso Cloud credentials set, the app runs on a synced local file
(`kratt-data/kratt-synced.db`): reads and writes stay local, changes push to
the cloud after every background pass (memory agent, dream, explore, embed)
and on app exit, and the app pulls on startup. Without credentials it runs
fully local on `kratt-data/kratt.db`, unchanged.

One-time setup:

```sh
# install the Turso CLI and sign up (free plan: 5GB, 3GB sync/month)
curl -sSfL https://get.tur.so/install.sh | bash
turso auth signup   # or: turso auth login

turso db create kratt
turso db show kratt --url          # -> TURSO_DATABASE_URL
turso db tokens create kratt       # -> TURSO_AUTH_TOKEN
```

Put both in `.env` (loaded via dotenvy at startup):

```sh
TURSO_DATABASE_URL=libsql://kratt-<org>.turso.io
TURSO_AUTH_TOKEN=<token>
```

First run on this machine: the synced database starts empty, the legacy
`kratt.db` is detected and copied over automatically (check the run log for
`[sync] legacy migration done`), then pushed to the cloud. On a second
machine, just set the same two variables — the local file bootstraps from the
cloud on first start. The legacy `kratt.db` is never modified; it stays as a
local fallback.

Notes:

- First synced start needs network (bootstrap); after that the app works
  offline and syncs on the next opportunity.
- Conflict resolution is "last push wins" per statement — fine for one person
  alternating between machines; don't run heavy write passes on both
  machines simultaneously.
- Human review: the Turso Cloud dashboard (app.turso.tech) has a per-database
  SQL shell and data browser; the local inspector
  (`cargo run --bin inspector`, http://127.0.0.1:7070) still works and reads
  whichever file the app uses.
- Message full-text search degrades in synced mode: the cloud server cannot
  parse the `CREATE INDEX … USING fts` DDL (pushes carrying it fail, and pulls
  can drop the local index record), so the FTS index is created in local mode
  only and synced-mode history search falls back to LIKE.
- Sync failures show as a red `⚠ sync` warning in the status line; press S to
  retry. The warning clears on the next successful sync.

## using cargo-modules to see boundary violations

for LLM (90% of the time codebase-memory-mcp does the job good enough)

```sh
cargo modules structure
```

DOT graph for human

```sh
cargo modules dependencies --no-externs --no-fns --no-sysroot --no-traits --no-types --no-uses > mods.dot
```
