# 2026-05-09 — Live scan progress events

## Why

Clicking "Scan Now" in the webapp showed a static "Scanning…" banner that stayed put for the entire run, with no signal that anything was happening. For a typical re-index that takes 30–60 seconds, the user had no way to tell whether it was stuck or just busy. We now surface live progress so the banner can show "Found N documents from M sources to update" and then "Processing X/N: <doc>" as work proceeds.

## What changed

Three layers, smallest changes that get us there:

1. **`Ingester.run_once()` gains a `progress_callback` parameter.** Default is a no-op so existing callers (including the in-process scheduler path) are unaffected. The callback is invoked at three points:
   - `{"phase": "syncing"}` once at the top, before git sync.
   - `{"phase": "discovery_done", "total_docs": N, "sources_changed": M, "sources_total": K}` after a new `_count_pending_files` helper walks every target's files, parses + hashes them, and counts how many would actually need re-embedding. This pre-walk is necessary to give the user an accurate "Found N documents" message before any embedding work starts.
   - `{"phase": "processing", "current": X, "total": N, "source": "...", "doc": "..."}` inside the existing per-file processing loop, just before each file is parsed-for-real and queued for upsert. `current` is global across all sources.

2. **`ingestion_worker.py` registers a default callback** that writes each event as a JSON line on stdout: `{"event": "scan_progress", ...}`. Dedicated IPC channel — separate from the structured logger — so it's robust to log-format changes. The supervisor was already iterating worker stdout for the existing `ingestion_cycle_complete` sentinel, so adding a second sentinel was natural.

3. **`IngesterSupervisor` parses the events into `_current_progress`,** exposed via a `current_progress` property under a fresh `_progress_lock`. Reset to `{"phase": "starting"}` at the top of `run_subprocess_cycle` and cleared to `None` in a `finally` block so a worker crash, timeout, or non-zero exit all leave `current_progress = None` (the `/health` consumer treats null as "not running").

4. **`/health` exposes `current_progress`** as a top-level field. The webapp's existing 2 s poll loop picks it up automatically — no new endpoint, no SSE.

## What we didn't do

- **Two-phase ingestion with cached parses.** Discovery re-parses each file on its own, then the processing loop re-parses again. Markdown parsing is cheap (file read + regex for the title); caching parsed content between phases would meaningfully add to the worker's RSS, which we already work hard to keep bounded. The duplicate I/O is a worthwhile trade.
- **SSE / WebSocket transport.** Considered briefly. The user pain is *slow* scans where the 2 s poll cadence is fine. SSE would add streaming machinery on both halves for sub-second latency we don't actually need here.
- **Banner cancel button / SIGTERM plumbing.** Nice-to-have, but a separate concern.

## Discovery cost

The `_count_pending_files` pre-pass parses every file in every source. For an unchanged source (most cycles), this is the same work as the old single-pass loop, just done up front. For a changed source, every file is parsed twice — once in discovery, once in processing. Worst case is the first cold start of a new clone, where every file is "new"; the parse cost is dominated by embedding/upsert anyway. No measurable cycle-time regression in local testing.

## Tests

- `tests/test_ingestion.py::TestIngester` — three new tests covering callback sequencing on a fresh ingest, no callbacks for unchanged files on re-runs, and graceful no-op when `progress_callback=None`.
- `tests/test_ingestion_supervisor.py` — five new tests covering live updates from a fake worker, cleanup on success / failure / timeout, copy semantics on the `current_progress` getter, and silent recovery from malformed JSON lines.
- `tests/test_server.py::TestHealthEndpoint` — `/health` test exercising both null (idle) and populated (mid-scan) `current_progress`.
- Existing `test_ingestion_worker.py::test_main_passes_source_and_force_flags` updated to assert the new `progress_callback` kwarg.

446 tests pass (was 437 before this change).

## Webapp counterpart

The webapp side of this change lives in `documentation-webapp` — see its journal entry for the same date. The two halves are deployed independently; the backend is forward-compatible (`current_progress` is just an extra field that older webapps will ignore).
