# 260429 — Engineering team: post-incident operational hardening

## Context

The 2026-04-25 outage (commit `c061f60` set `CHAT_MODEL` to a non-existent
`claude-opus-4-latest` alias) was hotfixed in `7b34119` on 2026-04-26. This
session ran a focused engineering-team Phase 1–3 cycle targeted at:

1. Hardening the operational posture so this class of bug fails fast next
   time instead of in front of a real user.
2. Closing the three open Dependabot advisories.
3. Diagnosing webapp 502s the user reported during ingestion windows.

## Phase 1 evaluation

`.engineering-team/evaluation-report.md` captures the full findings. The
three highest-impact gaps:

1. No startup validation of `CHAT_MODEL` and the chat tests mock the
   Anthropic client wholesale, so the bad model string was unobservable in
   CI. (Findings 1, 2)
2. CI workflow only built and pushed Docker images on push to `main`. No
   `pull_request` trigger, no `pytest` job, no `ruff` job — a broken PR
   could land with zero automated gating. (Finding 7)
3. Ingestion runs in-process via `APScheduler.BackgroundScheduler` on a
   thread pool. Under the 512 MB Docker `mem_limit`, OOM and GIL contention
   during embedding plausibly cause the webapp 502s the user reported.
   (Finding 11)

Three Dependabot advisories were also open: two GitPython issues (High;
non-exploitable here — all call sites consume operator-controlled config,
no user input), one `python-multipart` (Medium; transitive via `mcp`, never
exercised — zero multipart endpoints in the server).

## Phase 2 plan

`.engineering-team/improvement-plan.md` sequenced the work as five WUs:

1. WU1 — ingestion diagnostics (no behaviour change, just observability)
2. WU2 — chat path hardening
3. WU3 — CI hardening
4. WU4 — Dependabot upgrades (gated on WU3)
5. WU5 — compose & runtime polish
6. WU6 — fix the ingestion bottleneck (deferred until WU1 produces real data)

## Phase 3 implementation

Five commits on a single worktree (`worktree-eng-ops-hardening`):

- `fb01502` WU1 — `Ingester.run_once` now records `duration_s`,
  `rss_at_start_mb`, `rss_at_end_mb`, `rss_growth_mb`, `flush_count`,
  `flush_total_s`, `flush_max_s` to a structured log line and to
  `Ingester._last_ingestion`. `/health` exposes this as `last_ingestion`.
  Pure observability — zero production behaviour change.
- `e6977c2` WU2 — `_probe_chat_model()` runs at startup. On
  `NotFoundError`, `_chat_model_valid` is set to `False`, logged at
  ERROR, surfaced via `/health` as `chat_model_valid` /
  `chat_model_error`, and the chat handlers short-circuit with HTTP 503.
  The probe is skipped (assumed valid) when `ANTHROPIC_API_KEY` is unset.
  Tests gain an assertion that the `model` argument passed to the
  Anthropic mock matches `CHAT_MODEL` — the one-line guard that would
  have caught the original bug. Error→HTTP mapping narrowed:
  `NotFoundError`/`BadRequestError` → 500 (server config),
  `APIConnectionError`/`InternalServerError` → 502 (true upstream),
  `RateLimitError` stays 429.
- `bbda708` WU3 — new `.github/workflows/ci.yml` runs `ruff` + `pytest`
  on `pull_request` and `push: main` with concurrency cancellation.
  `docker-publish.yml` gains `workflow_dispatch:` and bumps to
  `actions/checkout@v6`, `docker/login-action@v4`,
  `docker/metadata-action@v6`, `docker/build-push-action@v7`. New
  `.github/dependabot.yml` configures weekly upgrade PRs for the `uv`
  ecosystem and `github-actions`, with minor+patch updates grouped.
- `76678ec` WU4 — `gitpython 3.1.46 → 3.1.49`, `python-multipart 0.0.22 →
  0.0.27`. All 386 tests still pass.
- `daf5dc5` WU5 — webapp `depends_on: docserver` switched to long-form
  with `condition: service_healthy` so it waits for healthcheck rather
  than mere process start. Dockerfile `start-period` reduced from 120s to
  60s now that the embedding model is image-baked.

## Test suite

380 → 386 tests (+4 chat hardening, +2 ingestion observability). Coverage
holds at 87%. `ruff check` clean throughout.

## Decisions worth remembering

1. **Soft startup probe over hard exit.** A misconfigured chat model does
   not stop the server from booting — search, document browsing, and the
   MCP tool surface keep working, only chat goes 503. The argument was
   that silent degradation was the actual harm in the original incident,
   and a hard-exit probe over-corrects by also taking down the
   non-Anthropic functionality that has nothing to do with the misconfig.
2. **Status-code narrowing matters for diagnostic clarity.** Returning
   502 for an Anthropic 404 falsely framed a server config bug as a
   gateway issue and made the original incident harder to root-cause.
   The new mapping makes the failure mode legible to anyone reading the
   webapp's network tab.
3. **Diagnose before fixing the ingestion bottleneck.** The user reported
   webapp 502s during ingestion — the eval listed three plausible root
   causes (memory pressure, GIL contention, slow `/health`) — but with
   no metrics, picking a fix would have been guessing. WU1 ships the
   instrumentation; WU6 is deferred until at least one cycle's data has
   landed in production logs.

## Follow-ups (next session)

1. Watch the `last_ingestion` field on `/health` and the
   `ingestion_cycle_metrics` log lines for one or more real cycles.
   Identify whether peak RSS is approaching the 512 MB cap, which flush
   step dominates the duration, and whether `/health` is staying
   responsive throughout. Then plan WU6.
2. The skill's evaluation also flagged that the `/rescan` endpoint
   spawns its own `threading.Thread` with the same in-process pattern —
   WU6 will need to address both call paths together.
