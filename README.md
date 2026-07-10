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

## Telegram gateway (headless daily driver)

The same binary runs headless behind Telegram: `cargo run --release -- --telegram`
long-polls the Bot API and answers through the exact chat-turn pipeline the TUI
uses (memory injection, tool loop, memory pass after every turn). No inbound
port needed — works behind NAT.

Setup:

1. Create a bot with [@BotFather](https://t.me/BotFather) → token.
2. Get your numeric user id (message [@userinfobot](https://t.me/userinfobot)).
3. Put both in `.env` (see `.env.example`):

```sh
TELEGRAM_BOT_TOKEN=123456:ABC-...
TELEGRAM_USER_ID=123456789
```

4. Run `cargo run --release -- --telegram` — the console prints
   `kratt telegram gateway ready` when polling starts.

Behavior and rules:

- **Allowlist**: only `TELEGRAM_USER_ID` is answered; everyone else is dropped
  with a log line. This is the entire security model — set it correctly.
- **Sessions are per calendar day** (`tg-YYYY-MM-DD`), unlike the TUI's
  session-per-launch.
- Turns are strictly sequential: answer → memory pass → next message. Messages
  sent while a turn (or the daemon) is busy queue on Telegram's side and are
  picked up in order; you'll see "typing…" the whole time a turn is running.
- Answers are rendered to Telegram HTML (headers→bold, bullets, code blocks;
  tables land as monospace blocks). The system prompt tells the model it is on
  a phone, so it avoids tables in the first place.
- Ctrl-C / SIGTERM shuts down gracefully (finishes the in-flight turn and
  memory pass, syncs, exits — the console says so). A second Ctrl-C force-quits.
- Observability = the same run logs: `kratt-data/logs/kratt-YYYY-MM-DD.log`
  (gateway lines are tagged `[tg]`).

Development without touching the live brain — two axes, both required:

```sh
# separate data dir AND a separate dev bot token
KRATT_DATA_DIR=kratt-data-dev TELEGRAM_BOT_TOKEN=<dev bot> TELEGRAM_USER_ID=<you> \
  cargo run -- --telegram
```

Never run two pollers on one token (they fight over `getUpdates`), and never
run the TUI and the daemon against the same data dir at the same time (the
write lock is in-process only).

Home-server deploy: build with `--release`, run under systemd with
`Restart=on-failure`, `.env` in the working directory, `KRATT_DATA_DIR` unset
(live data root + Turso sync as backup). Deploys lose nothing: Telegram queues
messages while the daemon is down and long polling picks them up on restart.

## using cargo-modules to see boundary violations

for LLM (90% of the time codebase-memory-mcp does the job good enough)

```sh
cargo modules structure
```

DOT graph for human

```sh
cargo modules dependencies --no-externs --no-fns --no-sysroot --no-traits --no-types --no-uses > mods.dot
```
