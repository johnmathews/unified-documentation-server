# Per-source health status tracking

## What changed

The `/health` endpoint previously returned a binary status: `"ok"` if the DB query succeeded,
`"error"` if it threw an exception. Individual source scan failures were invisible to the UI.

### New `source_status` table

Added SQLite table to persist per-source scan results:

- `source` (PK), `last_checked`, `last_error`, `last_error_at`, `consecutive_failures`
- `KnowledgeBase.update_source_check(source, error=None)` — upsert on success or failure
- `KnowledgeBase.get_source_statuses()` — returns all status records keyed by source name
- Table is created in `_SCHEMA` (new DBs) — no migration needed since it uses `CREATE TABLE IF NOT EXISTS`

### Ingester integration

- `_sync_source()` calls `kb.update_source_check()` on both success and failure paths
- `get_last_check_times()` now merges DB-persisted values with in-memory cache (covers restarts)

### Health endpoint changes

- Per-source status: `healthy` / `warning` / `error` / `unknown`
  - 1 consecutive failure → warning
  - 2+ consecutive failures → error
  - `last_checked` stale by >2x poll interval → warning, >5x → error
  - No status record → unknown
- Overall status: `healthy` / `degraded` / `error`
  - All sources OK → healthy
  - Any source warning/error → degraded
  - All sources error/unknown → error
- New response fields: `poll_interval_seconds`, `source_status`, `last_error`, `last_error_at`,
  `consecutive_failures`

## Tests added

- 7 unit tests for `update_source_check()` and `get_source_statuses()`
- 5 integration tests for health endpoint status computation
- All 254 tests pass, 85% coverage
