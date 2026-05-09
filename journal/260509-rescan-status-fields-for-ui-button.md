# /health additions and /rescan no-op log for the webapp "Scan now" button

## Context

The webapp added a manual "Scan now" button on the homepage and `/status` page so the
user can pick up changes immediately rather than waiting up to 30 min for the scheduled
poll. The button kicks off an async scan and then polls `/health` until the scan
completes, at which point it shows a small inline summary (added/updated/removed).

The webapp side of this lives in `documentation-webapp` on
`feature/scan-now-button`. This entry covers the small backend changes that made
that polling-based UI possible.

## What changed

1. `IngesterSupervisor.ingestion_running` — new read-only property that returns
   whether a worker subprocess is currently executing a cycle. Holds `_proc_lock`
   so it's safe to call concurrently with cycle starts/finishes.
2. `/health` JSON response now includes:
   - `last_stats: dict[str, ScanStats] | None` — per-source counts from the most
     recent successful cycle (already captured by the supervisor; previously not
     surfaced over HTTP).
   - `ingestion_running: bool` — drives the UI's polling loop. The webapp
     considers a scan finished when `ingestion_running == false` AND
     `last_ingestion.completed_at` advances past the click time.
3. `POST /rescan` 409 branch now logs an INFO event
   (`"Rescan trigger ignored: ingestion already running."`,
   `event=rescan_noop_already_running`) recording the requested sources and
   `force` flag. This was an explicit ask from the operator: a manual click that
   no-ops should leave a trail explaining why.
4. The pre-check for the 409 branch now uses the new `ingestion_running` property
   instead of poking `_current_proc` directly.

## Tests

- `TestHealthEndpoint::test_health_exposes_last_stats` — sets `_last_stats` on
  the supervisor, asserts the per-source counts come through.
- `TestHealthEndpoint::test_health_exposes_ingestion_running` — flips the
  supervisor's `_current_proc` between unset and a fake-running mock, asserts
  the bool toggles.
- `TestRescanEndpoint::test_rescan_rejects_concurrent` — extended to assert
  the no-op log also fires (uses `caplog` against the `docserver.server` logger).

Full suite: 437 passed (was 434 before these changes).

## What this does NOT change

- No new endpoint. `POST /rescan` was already there and already had the
  concurrency guarantees we wanted (single in-flight cycle, returns 409 on
  conflict). Adding a second endpoint would have been a duplicate.
- No change to scan semantics, scheduling, or worker subprocess handling.
- No change to the 30-min default poll cadence.
