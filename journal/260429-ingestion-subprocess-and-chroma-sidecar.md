# 260429 — Ingestion subprocess + ChromaDB sidecar (WU6)

## Context

This is the architectural fix for the webapp 502s the user reported during
ingestion windows on 2026-04-29. The diagnostic instrumentation from
WU1 (committed earlier in `fb01502`) had not yet produced production
data, but the user opted to ship the architectural fix anyway rather
than wait for it.

The original WU6 plan in `.engineering-team/wu6-ingestion-subprocess-plan.md`
proposed running ingestion as a per-cycle subprocess. Mid-planning, we
verified via Chroma's GitHub issues and cookbook that two
`PersistentClient` instances on the same on-disk path corrupt the
SQLite store backing Chroma 1.5.x (chroma-core/chroma#5868), with the
fix only landing post-1.5.5 in PR #6373. So the subprocess plan was
amended to require Chroma client/server mode as a prerequisite (WU6.0).

## Implementation

Six commits on `eng-ingestion-subprocess`:

1. `2e63971` — WU6.0a: SQLite WAL mode on `documents.db`. Without WAL,
   the default DELETE journal serialises every reader behind every
   writer; multi-process access requires WAL.
2. `92ed605` — WU6.0b: Chroma runs as a sidecar service in
   `docker-compose.yml`. The docserver and the ingestion worker both
   connect via `chromadb.HttpClient`. `KnowledgeBase` accepts
   `chroma_host`/`chroma_port` via `Config`; tests fall back to
   `PersistentClient` when host is unset.
3. `9fc01a3` — WU6.1: New `src/docserver/ingestion_worker.py`. CLI:
   `python -m docserver.ingestion_worker [--source X --force]`. Reuses
   `Ingester.run_once` verbatim; emits the cycle stats and metrics as a
   single JSON line on stdout immediately before exit.
4. `ba86cc7` — WU6.2: New `IngesterSupervisor`. Owns the APScheduler
   timer in the docserver process. Per tick, spawns the worker
   subprocess, streams its stdout to the container log stream, parses
   the final `ingestion_cycle_complete` line into `_last_ingestion`. A
   watchdog thread enforces the configured timeout. Concurrent calls
   raise `IngestionAlreadyRunning`. `/health` exposes `last_ingestion`
   and a new `last_ingestion_failure` field for operator visibility.
5. `80e9dca` — WU6.3: `/rescan` endpoint now calls
   `supervisor.run_subprocess_cycle()`. The old `_rescan_lock`,
   `_rescan_running` state and ad-hoc `threading.Thread` are gone;
   concurrency is owned by the supervisor.
6. `6496662` — WU6.4: Two OS-level knobs on the worker.
   `DOCSERVER_INGEST_NICE` (default 10 via supervisor) lowers CPU
   priority. `DOCSERVER_INGEST_MEM_LIMIT_MB` (set to 400 in compose)
   caps `RLIMIT_AS` so a runaway worker is killed by the kernel before
   it can pull the whole container down.

## Test changes

387 → 406 (+19 new tests):

- 1 for SQLite WAL mode persistence
- 3 for `chroma_host` / `chroma_port` config loading and HttpClient branch
- 6 for the ingestion worker entry point (success, three failure modes,
  argv flags, nice env var)
- 9 for the supervisor (happy path, log pass-through, non-zero exit,
  missing-metrics-line, hard timeout, concurrent rejection, argv
  construction with/without flags, stop-with-running-worker)
- 2 rewrites of existing /rescan tests to mock the supervisor

Coverage holds at 87%. Ruff clean.

## Decisions worth remembering

1. **Spawn-per-cycle, not a persistent worker.** A persistent worker
   would keep the ~350 MB embedding model resident in a second process
   continuously — the very memory headroom we are trying to recover.
   Per-cycle spawn pays a ~5 s cold-start (ONNX model mmap from disk
   cache + KB open) at a 300 s poll interval, ~1.7% overhead. The
   worker's full RSS is released to the OS on exit.
2. **SQLite WAL mode is non-negotiable for multi-process access.**
   Even though the docserver is mostly read-only against
   `documents.db`, a worker writing during a cycle would block every
   reader without WAL. WAL also lets `synchronous=NORMAL` be safe,
   which trades fsync-per-write for fsync-per-checkpoint.
3. **Soft fall-back to `PersistentClient` for tests.** Production sets
   `DOCSERVER_CHROMA_HOST=chroma` so `KnowledgeBase` uses `HttpClient`;
   tests omit it and get `PersistentClient` against a tmp dir. This
   keeps the existing 387 unit tests on a fast in-process path with no
   need to spin up a Chroma server in CI. Multi-process safety only
   matters when the worker is in the picture — tests do not exercise
   that path simultaneously.
4. **Stdout pass-through, not log re-emission.** The supervisor reads
   the worker's stdout line-by-line and writes each line to its own
   stdout (`sys.stdout.write(line)`) rather than re-logging via Python.
   Both processes use `setup_logging` to emit JSON to stdout, so the
   container log stream stays homogeneous; the worker's `pid` field
   distinguishes its lines from the docserver's.

## Empirical risks not yet verified

These are not blocking, but the user should know they are unverified
before merging:

1. **On-disk format compatibility between `PersistentClient` and
   `chroma run`.** The `chromadb` package internals appear to use the
   same Rust storage layer regardless of which client is in front, but
   we did not dump-and-reload an existing store to confirm. Worst case
   on first deploy: the chroma sidecar can't open the existing
   `/data/chroma/`, the operator wipes it, and ingestion re-populates
   on the next cycle (a few minutes of CPU time).
2. **Chroma image tag.** The `chromadb` Python package is pinned at
   1.5.5; Docker Hub tags jump from 1.5.6 to 1.5.7 to 1.5.8 (no 1.5.5
   image was published). We pin to `chromadb/chroma:1.5.8`. Chroma's
   wire protocol is meant to be stable across patch versions, but not
   tested.
3. **Healthcheck command.** `chroma`'s container uses `</dev/tcp/...`
   which requires bash. If the upstream image switches to a
   distroless/alpine base, the healthcheck breaks. Worth replacing
   with a Python `urllib.request` one-liner if it shows up as a
   problem.

## Follow-ups

1. Once deployed, watch `/health.last_ingestion_failure` for the first
   week. Any non-null value is a worker that timed out, crashed, or
   exited non-zero — investigate via the worker's stdout in container
   logs.
2. Watch `last_ingestion.rss_at_end_mb` from WU1's existing
   instrumentation. Should now report the **worker's** peak RSS, not
   the docserver's. If it is consistently approaching 400 MiB
   (`DOCSERVER_INGEST_MEM_LIMIT_MB`), consider lowering the embedding
   batch size or raising the limit.
3. The Chroma sidecar's own RSS over time. 256 MB `mem_limit` was a
   guess; if it OOMs once steady-state is reached, raise.
