# Cut idle ingestion cost: 30-min default + skip walk on unchanged HEAD

## Why this came up

Looked at `docker compose logs documentation-server` and saw a wall of `Found N files / All N file(s) unchanged / Ingestion complete: skipped=N` blocks repeating every 5 minutes for all 17 sources. No errors, just very chatty for what is — almost always — a zero-change cycle.

Initial guess from logs was "the cycle is re-initializing Config + KnowledgeBase every tick." Investigation by an engineering-team subagent corrected that: the `Initializing knowledge base` / `SQLite initialized` / `ChromaDB connected` lines come from the **ingestion subprocess** that the supervisor spawns each cycle (the deliberate isolation pattern recorded in `ingestion_supervisor.py:1-22`), not from the long-lived server process. The server's `_kb` and `_config` globals are set once in `init_app()` and reused. So Fix 2 from my initial diagnosis was moot — the right lever for that noise is just running cycles less often.

That left two real changes plus docs.

## What changed

1. **`config.py`** — `poll_interval_seconds` default raised 300 → 1800 (5 min → 30 min). The env var `DOCSERVER_POLL_INTERVAL` and yaml key `poll_interval` already existed, so this was a one-line tweak in `load_config()` plus the dataclass default. `docker-compose.yml` and `config/sources.example.yaml` were updated to match (the env var in compose was the actual production value of 300; without updating it the code default would never apply).

2. **`ingestion.py:1295`** — added a single `continue` in `Ingester.run_once()` that fires when `source.is_remote and sync_results.get(source.name) is False`. This skips:
   - `manager.get_files()` (glob walk)
   - `_bulk_git_created_at()` (one `git log` subprocess)
   - `get_indexed_content_hashes()` (SQLite read)
   - `get_all_doc_ids_for_source()` (SQLite read)

   Logs a `skip_unchanged` event with the source name so the operational story is still visible.

3. **The `is_remote` carve-out is load-bearing.** `_sync_local()` always returns `False`. If the short-circuit fired on local sources, their index would freeze — the file walk + content-hash comparison is the *only* change-detection mechanism for them. The `is False` (not `not changed`) check is also intentional: it must not fire on `None` (sync error, already handled by the prior guard) or `True` (HEAD advanced).

## Tests

Updated one existing test that was asserting the old "all skipped" semantics, added two new ones:

- `test_no_changes_short_circuits_file_walk` — stats dict for the source has `files=0, skipped=0, new=0` (not `skipped >= 1`) on the second cycle, proving the walk was bypassed entirely.
- `test_short_circuit_emits_skip_unchanged_log` — captures log records via `caplog`, asserts exactly one `skip_unchanged` event for the remote source and zero `files_found` events.
- `test_local_source_not_short_circuited` — creates a local source, mutates a file in-place between two `run_once()` calls, asserts `modified >= 1` on the second cycle. This is the regression guard that protects the local-source path.

Full suite: 408 passed. Coverage on `ingestion.py` jumped from 64% to 89% (the new tests exercise paths that weren't hit before).

## Out of scope (and why)

1. **Eliminating the worker subprocess "init" log spam.** Would require collapsing ingestion back into the server process, undoing the WU6 isolation work that fixed the production memory leak. 6× cadence reduction is the cheap win; the rest is fine.
2. **Hot-reload of `sources.yaml` in the server process.** `_config` is built once at startup. Adding a new source still works (the worker subprocess re-reads config on every spawn, so indexing picks it up immediately) but `/health` and the `ingestion_status` MCP tool show stale source counts until restart. Cosmetic; not worth the complexity.
3. **ChromaDB / SQLite reconnect logic.** Verified neither needs it: the HttpClient is stateless between requests, and `KnowledgeBase` opens a fresh SQLite connection per operation.

## Operational impact

After deploying:
- Remote sources whose HEAD didn't advance produce one `sync_unchanged` + one `skip_unchanged` log line per cycle, instead of the full `Found N files / All N unchanged / Ingestion complete` block.
- Idle cycles drop from ~200 log lines to ~40, and the wasteful file-walk + git-log subprocess + SQLite reads per source are gone.
- 6× fewer subprocess spawns = 6× less of the "Initializing knowledge base..." noise.

To pick this up on the deployed server: `docker compose pull && docker compose up -d`. The new `DOCSERVER_POLL_INTERVAL=1800` is now baked into `docker-compose.yml`.

## Files touched

- `src/docserver/config.py` (default 300 → 1800, dataclass default aligned)
- `src/docserver/ingestion.py` (short-circuit added in `Ingester.run_once`)
- `tests/test_config.py` (assertion updated to 1800)
- `tests/test_ingestion.py` (1 test updated, 2 added)
- `docker-compose.yml` (env override updated to 1800 with explanatory comment)
- `config/sources.example.yaml` (example updated to 1800)
- `README.md` (env var table, example yaml, sample health JSON)
- `docs/architecture.md` (cycle-step list updated, new step 3 documents the short-circuit)
- `docs/operations.md` (`skip_unchanged` event added to log table, env-var table updated, troubleshooting reference to "5 minutes" → "30 minutes")

Shipped on main as `4195e63`. CI green on first push (test + Docker build/push to ghcr.io both passed).
