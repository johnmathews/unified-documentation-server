# 260519 — Fast local-dev mode: skip startup/poll ingestion

## Problem

Running the backend locally for webapp testing was computationally heavy —
fans spinning, laptop hot. Investigation (engineering-team triage) showed it
was *not* "re-embed from scratch every time": `local-data/` persists across
runs and the skip-if-unchanged logic (HEAD short-circuit + per-file SHA256) is
correct.

The real cost: `IngesterSupervisor.start()` scheduled the interval job with
`next_run_time=datetime.now(UTC)`, so **every** `python -m docserver` start ran
a full cycle immediately, and again every `poll_interval` (600s in
`sources.local.yaml`). Each cycle spawns a fresh subprocess that loads the ONNX
mpnet model and git-fetches all 12 configured remote GitHub repos, then walks +
hashes every doc and re-embeds anything whose upstream HEAD moved. Since the
user authors those 12 repos, "HEAD moved" is true most sessions. There was no
knob to disable this for local dev — `DOCSERVER_POLL_INTERVAL` only changed the
interval and never suppressed the boot cycle.

## Change

Explicit-flag approach (chosen over an auto-skip-when-populated heuristic):

- `config.py`: new `Config.ingest_on_start: bool = True`, parsed from
  `DOCSERVER_INGEST_ON_START` (falsey set `{0,false,no,off,""}`) with YAML
  `ingest_on_start` fallback. Env overrides YAML, matching the existing
  precedence pattern.
- `ingestion_supervisor.py`: extracted `_ingestion_mode() -> (mode, immediate)`
  returning one of `interval` / `once` / `disabled`. `start()` branches on it:
  - `interval` (poll>0): recurring job; `next_run_time=now` only if
    `ingest_on_start`.
  - `once` (poll≤0, ingest_on_start): single `trigger="date"` boot cycle, no
    recurring job.
  - `disabled` (poll≤0, not ingest_on_start): log `supervisor_disabled` and
    return without starting the scheduler — `stop()` stays safe (guarded by
    `_scheduler.running`).
- `server.py`: `DOCSERVER_INGEST_ON_START` added to the startup env log.

`/rescan` is unchanged and still runs an on-demand cycle regardless of mode,
so the fast path loses nothing.

## Fast local-dev recipe

```bash
DOCSERVER_DATA_DIR=./local-data DOCSERVER_CONFIG=./config/sources.local.yaml \
DOCSERVER_POLL_INTERVAL=0 DOCSERVER_INGEST_ON_START=0 \
uv run python -m docserver
```

## Tests / verification

- `test_config.py`: defaults, env falsey/truthy parametrised, YAML, env-over-
  YAML.
- `test_ingestion_supervisor.py`: full `_ingestion_mode()` matrix +
  disabled-mode never starts the scheduler and `stop()` is safe.
- Full suite green: 524 passed, 86% coverage.

Default/production behaviour is unchanged (interval + immediate). Docker
compose already uses a 30-min poll; not touched.

## Docs

`docs/operations.md` (env-var table + new *Fast local development* section),
`CLAUDE.md` (APScheduler line). Webapp repo updated in parallel
(`260519-fast-local-dev-backend-flags.md`).
