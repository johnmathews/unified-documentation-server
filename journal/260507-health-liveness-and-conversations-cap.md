# 2026-05-07 — `/health` Chroma liveness + `conversations.db` row cap

Two small follow-ups to the OOM-mitigation work earlier today (commit
`fd6afcf`). Both were Phase 2 items in the engineering-team plan that
fell out of scope of the OOM containment fix itself.

## Unit 8 — Live Chroma ping in `/health`

`/health` derived its `status` from SQLite source-status counters and
the supervisor's last-cycle metrics. If Chroma went down between
ingestion cycles, the endpoint kept returning `healthy` until the next
worker tried to write and recorded a failure.

Added `KnowledgeBase.ping_chroma()` — a synchronous wrapper around
`ClientAPI.heartbeat()` that converts any exception into
`(False, error_message)`. The `/health` handler now dispatches it via
`asyncio.to_thread(...)` so the call doesn't block the request loop,
and surfaces two new fields:

- `chroma_alive: bool`
- `chroma_error: str | None`

If the ping fails, the overall `status` is downgraded to `degraded`.
The HTTP status code stays 200 — the response is well-formed and
informative; only catastrophic handler errors return 503.

I considered adding an explicit `asyncio.wait_for` wall-clock bound (so
a hung Chroma client couldn't stall the handler indefinitely) but
dropped it for two reasons:

1. `chromadb.HttpClient` already enforces a sensible httpx-level
   timeout (~5s default), so `heartbeat()` returns within seconds even
   when Chroma is unresponsive.
2. Testing the wall-clock bound under starlette's `TestClient` is
   fragile because each test request runs on a one-shot event loop
   that waits for the executor to drain on shutdown — measuring
   `elapsed` doesn't reflect what the production loop sees. If we ever
   observe handler hangs in practice, we can add the timeout then.

Tests:

1. `test_health_includes_chroma_alive_true` — happy path, fields
   present.
2. `test_health_degrades_when_chroma_down` — mocked failure, overall
   status flips to `degraded`, error string surfaced.
3. `test_health_chroma_ping_runs_off_event_loop` — verifies the call
   is dispatched to a worker thread, not the main thread.

## Unit 9 — Cap on `conversations.db` row count

`ConversationStore` had no row-count or age-based cap; every chat
created a row that was never pruned. Single-user homelab use means
this would have taken months or years to become a real problem, but
it's a known unbounded growth path.

Added a configurable cap (default 1000 rows, override via
`max_conversations=...` constructor arg or `DOCSERVER_MAX_CONVERSATIONS`
env var). On every `create()` we run a delete-everything-beyond-the-cap
query ordered by `updated_at DESC` (with `created_at` as tiebreaker),
so the LRU conversation is dropped first. A conversation that's been
created but recently updated will be retained ahead of one that hasn't
been touched.

Implementation note: SQLite's `LIMIT -1 OFFSET N` idiom is used to
identify rows past the first N — a clean way to express "everything
beyond the cap" without loading the row count first.

Tests in `TestRowCap`:

1. `test_default_cap_used_when_unspecified` — sanity.
2. `test_explicit_cap_overrides_default` — constructor arg wins.
3. `test_env_var_overrides_default` — env-var path.
4. `test_invalid_env_var_falls_back_to_default` — non-integer env var
   logs a warning and defaults; doesn't crash startup.
5. `test_cap_under_one_clamped` — pathological cap of 0 clamps to 1.
6. `test_insert_under_cap_does_not_prune` — no false-positive deletes.
7. `test_insert_at_cap_prunes_oldest` — basic cap behaviour.
8. `test_repeated_inserts_keep_count_stable` — 50 inserts with cap=5
   leaves exactly 5 rows.
9. `test_pruning_uses_updated_at_not_created_at` — LRU semantic, not
   FIFO. A recently-updated old conversation survives ahead of an
   untouched newer one.

## Tests + lint

424 tests pass (was 412), ruff clean, coverage 87%. Net add: 12 tests
(3 health, 9 conversations).

## Out of scope

- Unit 4 (profile worker peak RSS) — needs the new image deployed on
  infra to be useful.
- Unit 10 (move stack to dedicated LXC) — long-term architectural;
  deferred until OOM containment is verified in production.

## Reminder

The production `/srv/infra/docker-compose.yml` on infra is still
running the old values from before commit `fd6afcf` (mem_limit: 2g,
POLL_INTERVAL=300, no oom_score_adj). Re-run the Ansible playbook
(or hand-edit) to render the new template
(`johnmathews/home-server@2ff2ce3` in `proxmox-setup`) and `docker
compose pull && docker compose up -d` to take effect.
