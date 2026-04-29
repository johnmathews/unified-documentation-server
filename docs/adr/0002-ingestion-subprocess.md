# ADR 0002 — Run ingestion as a per-cycle subprocess

**Status:** Accepted (2026-04-29)

**Depends on:** ADR 0001 (the Chroma sidecar is the prerequisite that
makes a second process touching the database safe).

## Context

The docserver had been doing ingestion in-process via
`APScheduler.BackgroundScheduler`, which runs jobs on a thread pool
inside the same process as the FastMCP server. That meant the request
loop and the ingestion pipeline shared GIL, heap, and a single Docker
`mem_limit` of 512 MB.

The user reported intermittent 502s on the webapp during ingestion
windows. Three plausible root causes:

1. **OOM kill** — embedding a batch loads tokenisation buffers, numpy
   intermediates, and ChromaDB-side write state on top of the ~350 MB
   ONNX model and the docserver's own RSS, peaks past 512 MB, Docker
   kills the container, in-flight requests 502, `restart: unless-stopped`
   brings it back.
2. **GIL + SQLite-writer contention.** ONNX inference releases the
   GIL during the native call but the surrounding tokenisation, numpy
   ops, and SQLite/Chroma writes do not. The ingestion thread holds
   the GIL long enough that the asyncio event loop pauses past the
   webapp's upstream timeout.
3. **Slow `/health`.** Docker marks the container unhealthy and the
   webapp's `depends_on` reaction yanks it.

WU1's diagnostic instrumentation (`/health.last_ingestion`,
`ingestion_cycle_metrics` log lines) had been merged but had not yet
produced production data. The user opted to ship the architectural
fix anyway rather than wait, on the basis that the architecture is
defensible against all three causes simultaneously.

## Decision

The docserver process keeps the APScheduler timer and a thin
`IngesterSupervisor` class. On each tick (and on each `POST /rescan`),
the supervisor spawns `python -m docserver.ingestion_worker` as a
subprocess, streams its stdout into the container log stream, parses
the worker's final `ingestion_cycle_complete` JSON line into
`_last_ingestion`, and waits for the process to exit. A watchdog
thread enforces a hard 600 s timeout (configurable). At most one
worker runs per supervisor — concurrent calls raise
`IngestionAlreadyRunning`.

The worker reuses `Ingester.run_once` verbatim. The indexing logic
itself is unchanged.

OS-level guard rails (`DOCSERVER_INGEST_NICE=10` and
`DOCSERVER_INGEST_MEM_LIMIT_MB=400`) lower the worker's CPU priority
and clamp its address space below the container's `mem_limit`, so a
runaway worker is killed by the kernel before Docker has to OOM-kill
the whole container.

## Alternatives considered

1. **Persistent worker subprocess.** Spawn the worker once at server
   startup; signal it (or queue jobs to it) per cycle; worker keeps
   the embedding model resident continuously. Rejected: keeps a ~350
   MB working set live in a second process forever, eating the very
   memory headroom we are trying to recover. Per-cycle spawn pays
   ~5 s cold-start (1.7% overhead at the default 300 s poll interval)
   and wins everywhere else.
2. **`multiprocessing.Process` instead of `subprocess.Popen`.** Less
   isolation (shared start state via fork on POSIX), more entanglement
   with Python's process-management primitives, and harder to give
   the worker a different log/env shape. We chose the heavier
   subprocess boundary on purpose.
3. **Just optimise the in-process path** — smaller embedding batches,
   `time.sleep(0)` yield points, `os.nice(10)` on the ingestion
   thread. Cheaper but partial: doesn't help if the dominant cause is
   OOM, which the metrics did not yet exclude.
4. **Run ingestion on a separate host.** Out of scope for a home
   server with a single VM.

## Consequences

### Positive

1. **OOM isolation.** Worker peaks live in their own address space.
   When the worker exits, the OS reclaims everything; the docserver's
   RSS does not see the spike. If the worker is OOM-killed, the
   docserver keeps serving requests — only the cycle is lost.
2. **GIL isolation.** The docserver's asyncio event loop never sees
   the worker's interpreter at all. Request handlers stay responsive
   regardless of what the worker is doing.
3. **Simpler /rescan.** The endpoint's old `_rescan_lock` +
   `_rescan_running` state and ad-hoc `threading.Thread` are gone.
   Concurrency is owned authoritatively by the supervisor.
4. **Operator visibility.** `/health.last_ingestion_failure` is set
   when the most recent worker timed out, crashed, or exited
   non-zero. Silent ingestion stalls become visible without scraping
   logs.

### Negative

1. **Cold start per cycle.** ~5 s of ONNX model mmap + KB open + chroma
   handshake on every cycle. Negligible at 300 s poll, would be
   meaningful (~8%) if the interval ever drops to 60 s.
2. **The docserver still loads the embedding model.** Subtle — the
   `OnnxEmbeddingFunction` is registered on the Chroma collection
   client-side, so the docserver mmaps the model lazily on the first
   `/api/search` query and keeps it resident. This is ~110 MB, not the
   full ~350 MB peak that ingestion incurs, but the subprocess pattern
   is *not* "the docserver becomes model-free". It buys peak
   decoupling, not steady-state shrinking.
3. **Two paths to maintain.** The in-process `Ingester.start()` path
   still exists for tests that drive `Ingester.run_once` directly.
   Production uses the supervisor; tests use the in-process path.
   Different than what runs in production, but the alternative
   (forcing every test to spin up a real subprocess) costs ~5 s per
   test run and would have blown out CI time.
4. **Stdout-protocol coupling between worker and supervisor.** The
   supervisor parses the worker's final stdout line as JSON to extract
   metrics. Any other code that prints `"ingestion_cycle_complete"`
   into the worker's stdout would break the parse. Mitigation: the
   sentinel is in a `"event": "ingestion_cycle_complete"` payload that
   is unlikely to appear by accident, and a malformed JSON line is
   suppressed without crashing.

## References

1. Implementation commits: `9fc01a3` (worker entry point), `ba86cc7`
   (supervisor), `80e9dca` (`/rescan` migration), `6496662` (resource
   limits).
2. Plan archive: `.engineering-team/wu6-ingestion-subprocess-plan.md`.
3. Memory metrics that should ride along with this work:
   `/health.last_ingestion` (introduced in WU1, commit `fb01502`).
