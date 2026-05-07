# 2026-05-07 — OOM mitigation and subprocess-isolation hardening

Background: the infra VM (6 GB max, balloon floor 1.95 GB) ran out of
memory around 22:48 UTC on 2026-05-06 and OOM-killed services. The doc
stack runs on that VM alongside the observability platform (Loki,
Grafana, Alloy, exporters). Investigation showed that the failure mode
was *recurring* (kernel ring buffer has 9 OOM kills in 7 days) and the
victim every time was a `python` process with anon-rss 1.2-1.4 GB and
total-vm ~3.2 GB — i.e. the APScheduler-spawned ingestion worker
subprocess loading the ONNX embedding model.

Two systemic problems explained why the cgroup never contained the kill:

1. The *deployed* compose on infra had drifted from the repo source. It
   set `mem_limit: 2g` on a host with ~1.8 GB usable memory, so the
   cgroup limit could never trigger; the kernel global OOM fired first,
   degrading the whole host. It also set `DOCSERVER_POLL_INTERVAL=300`
   (5 minutes) instead of the repo's `1800` (30 minutes), so ingestion
   ran 6× more often and the chance of a real-content cycle coinciding
   with peak memory usage of other tenants was much higher.
2. The worker subprocess applied `RLIMIT_AS=400m` via
   `DOCSERVER_INGEST_MEM_LIMIT_MB=400`. ONNX Runtime mmaps its model
   weights, which counts against virtual address space well before
   pages are resident. Observed total-vm at kill time was ~3.2 GB —
   8× the rlimit. The rlimit was either silently failing or firing at
   unpredictable allocation points; either way, it was not the
   containment we thought it was.

## What changed

### Compose — repo source of truth (`docker-compose.yml`)

- `docserver`: `mem_limit: 1024m` → `768m`; `mem_reservation: 1024m` →
  `256m`. The new `mem_limit` is below the VM's available memory so the
  cgroup OOM can actually fire. `mem_reservation` reflects the
  steady-state floor, not the embedding spike.
- `documentation-webapp`: added `mem_limit: 192m`,
  `mem_reservation: 64m` (was unlimited).
- Removed `DOCSERVER_INGEST_MEM_LIMIT_MB=400` from the docserver
  service env block (the worker no longer applies `RLIMIT_AS`).
- Updated the inline comment to explain the host's memory constraints
  and the trade-off between cgroup-limit-fires-cleanly vs
  global-OOM-fires-and-degrades-the-host.

### Worker — drop RLIMIT_AS (`src/docserver/ingestion_worker.py`)

Removed the `resource.setrlimit(RLIMIT_AS, ...)` block entirely. The
container cgroup is now the worker's only memory boundary. When it
fires, Docker's OOM killer picks the highest-RSS process in the cgroup
— the worker — and kills it cleanly, leaving the server parent
process alive. Updated module docstring to explain the rationale.

### `reindex` MCP tool routes through the supervisor (`src/docserver/server.py`)

The `reindex` MCP tool used to call `Ingester.run_once()` directly
inside the server process. That loaded the ONNX embedding model into
the *server's* RSS, defeating the subprocess isolation. Tool now
dispatches to `IngesterSupervisor.run_subprocess_cycle()`, the same
path the scheduler and `/rescan` use. Added handling for
`IngestionAlreadyRunning`, `IngestionTimeout`, and the
`run_subprocess_cycle()` returning `None` failure path.

The supervisor's return type changed: `run_subprocess_cycle()` now
returns the per-source stats dict (matching `Ingester.run_once`'s
shape) on success, with cycle-level metrics still available via the
`last_ingestion` property. New `last_stats` property exposes the
per-source dict for callers that need it.

### Drop dead `unload_embedding_model` call (`src/docserver/ingestion.py`)

`_run_once_safe` was calling `kb.unload_embedding_model()` on the
server's KB after each scheduled cycle. In the current subprocess
architecture the server's KB is used only for queries — unloading it
adds latency to the next search with no memory benefit. The unload
was a leftover from when ingestion ran in-process. Removed the call;
left `reclaim_memory()` since gc + malloc_trim are still useful.

### Cache the chat inventory context (`src/docserver/server.py`)

Every `/api/chat` and `/api/chat/stream` invocation called
`kb.get_document_tree()` (full table scan of `documents WHERE is_chunk
= FALSE`) and `kb.get_sources_summary()` to build the system prompt.
Cheap per call, but pointless re-allocation under chat load.

Added a module-level `(rendered_inventory, expires_at)` cache with a
60s TTL guarded by a lock. The supervisor's new `on_cycle_success`
callback (wired in `init_app`) clears the cache after every successful
ingest, so a freshly-indexed document is visible to chat within a
single chat request rather than waiting for the TTL.

## Tests

- Updated `test_ingestion.py` — three tests that asserted
  `_run_once_safe` calls `unload_embedding_model` collapsed into one
  test asserting the *opposite* (that the new code does NOT unload).
  Added a separate test that `reclaim_memory` still runs even on
  exception.
- Updated `test_ingestion_supervisor.py` — adjusted
  `test_run_subprocess_cycle_parses_metrics` for the new return-value
  semantics (per-source stats instead of cycle metrics).
- Rewrote `test_server.py::TestReindex` — old tests assumed in-process
  ingestion against a tmp_path config. New tests mock the supervisor
  at the boundary; integration of the worker against a tmp config is
  covered by the supervisor's own tests. Added coverage for
  `IngestionAlreadyRunning` and the `run_subprocess_cycle`-returns-None
  failure path.
- Added `test_server.py::TestInventoryCache` — three tests covering
  cache hits avoiding KB calls, manual invalidation forcing rebuild,
  and the supervisor's `on_cycle_success` callback clearing the cache.

Final: 412 tests pass, ruff clean, coverage 87%.

## Documentation

- `README.md` — removed `DOCSERVER_INGEST_MEM_LIMIT_MB` from the env
  var table.
- `docs/operations.md` — removed the env var, updated the resource
  usage table to show 768 MB, added the webapp service row, and
  updated worker memory description to reflect cgroup-bounded
  containment.
- `docs/adr/0002-ingestion-subprocess.md` — appended a paragraph
  recording why `RLIMIT_AS` was removed and what containment now looks
  like. (ADR-0001 and ADR-0002's historical preamble kept as-is —
  those describe the state at the time of decision.)

## What still needs to happen on infra (NOT done in this commit)

This commit only touches the repo. The deployed `/srv/infra/docker-compose.yml`
on the infra VM still has the drifted values. To actually take effect,
the operator needs to:

1. Pull the new image (`ghcr.io/johnmathews/unified-documentation-server:latest`)
   once GitHub Actions has built it from this commit.
2. Update `/srv/infra/docker-compose.yml` to match the repo:
   - `docserver`: `mem_limit: 768m`, `mem_reservation: 256m`
   - `docserver`: `DOCSERVER_POLL_INTERVAL=1800` (was 300)
   - `docserver`: remove `DOCSERVER_INGEST_MEM_LIMIT_MB`
   - `documentation-webapp`: add `mem_limit: 192m`, `mem_reservation: 64m`
3. `docker compose pull docserver documentation-webapp && docker compose up -d`

Until step 2 runs, the cgroup OOM still cannot fire on the deployed
container — the code change is necessary but not sufficient.

## Out of scope for this round

- `oom_score_adj` on Loki/Grafana/Alloy/Prometheus/cadvisor (Unit 3 in
  the engineering-team plan) — those compose files live elsewhere on
  infra; will need a separate pass once their location is established.
- Profiling the worker's actual peak RSS to identify the source of the
  ~1.3 GB anon-rss (Unit 4) — the test suite doesn't reach that
  pathway and infra is not yet on the new image.
- Moving the documentation stack to its own LXC (Unit 10) — long-term
  architectural change; deferred until the immediate stability fixes
  prove out.
- Bounding `conversations.db` growth (Unit 9) — not urgent for
  single-user homelab use.
- Adding a Chroma liveness probe to `/health` (Unit 8) — separate small
  PR.
