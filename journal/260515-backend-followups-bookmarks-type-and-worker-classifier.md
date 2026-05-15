# 2026-05-15 — Backend follow-ups: bookmarks `type` field + worker classifier

Two independent backend bugs flagged across the webapp's round-2 PRs and
deferred as out-of-scope each time. Both shipped together on
`eng-backend-followups`.

## 1. `/api/bookmarks` was missing the `type` field

`api_list_bookmarks` enriched each bookmark from `kb.get_document(doc_id)`
but the result dict omitted `type`. The webapp's `BookmarkEntry`
(`webapp/src/lib/api.ts:441`) already typed `type?: DocType` and the
`/bookmarks` page groups bookmarks by it — so every bookmarked doc was
arriving with `type` undefined and the page degenerated to a single
"Documentation" subheading per source.

Fix: one line in `server.py:1466` — `"type": doc.get("type") if doc else
None,` in the enrichment dict. No API shape change, no webapp follow-up
needed.

### Decision: join in the endpoint, not separate fetches

The webapp's bookmark page renders the doc-type badge inline against
every bookmark row, and the endpoint already fetches each doc via
`kb.get_document` for the existing enrichment fields. Returning `type`
in that same response keeps the page to a single round-trip and adds zero
new fetches. Separate `/api/docs/{id}` lookups would be N+1 with no
upside — the doc rows are already in scope at enrichment time. The
webapp side stays untouched.

### Tests added (`test_server.py::TestBookmarkEndpoints`)

1. `test_list_enriched_with_doc_metadata` — extended to assert
   `type == "documentation"` (the SQLite default for unclassified docs).
2. `test_list_carries_non_default_type` — seeds a doc with
   `type="journal"`, bookmarks it, asserts the response carries `journal`.
   Reproduces the round-2 regression: failed before the fix, passes after.
3. `test_list_type_is_none_for_orphan_bookmark` — a bookmark whose doc
   was deleted from KB still lists with every enriched field (including
   `type`) reported as `None`. Pins the missing-doc fallback path so the
   endpoint can't accidentally start raising on orphan rows.

## 2. Doc-type classifier never ran in the worker subprocess

The Stage 2 classifier (`260513-doc-types-and-exclude.md`) was wired into
`server.init_app`, but the actual ingestion runs in a separate process
spawned by `IngesterSupervisor` (`260429-ingestion-subprocess-and-chroma-
sidecar.md`). The worker entry point `ingestion_worker.py:120` built an
`Ingester(config, kb)` without `doc_types_config`, so every doc ingested
in Docker fell to `fallback_type="documentation"` regardless of what
`doc_types.yaml` said. Local dev (in-process ingestion via `init_app`)
classified correctly, so the bug only showed up in production.

The in-process path's one-shot backfill at startup is unaffected — the
`meta`-table hash gate fires once on server boot, against the same
`doc_types.yaml` the worker now reads. The two paths now agree.

### Fix: re-load the config in the worker, don't try to serialise it

The handoff posed it as "serialize config and pass to subprocess vs
re-load from disk in the subprocess." Re-loading wins because:

1. `subprocess.Popen` already gets the env, including
   `DOCSERVER_DOC_TYPES_CONFIG` — the file path is the natural IPC
   channel. No new flag, no pickle, no schema-skew risk.
2. Mid-cycle config edits propagate on the next cycle without restart.
3. `DocTypesConfig` is frozen so it would have to be re-validated on the
   other side anyway — re-loading just centralises that step.

The fix is six lines in `ingestion_worker.main()`:
`load_doc_types_config(known_source_names={s.name for s in config.sources})`
between the KB open and `Ingester(...)`, with `kb.close()` on the
classifier-load failure path so a malformed YAML doesn't leak a connection.

### Tests added

Two layers, because the bug crosses a process boundary:

1. `test_ingestion_worker.py::test_main_classifies_doc_types` — calls
   `worker_module.main(argv=[])` directly with a real KB and a real
   `doc_types.yaml`, then re-opens the KB and asserts the doc's `type` is
   the classified value (`journal`), not the fallback. Catches the
   regression at the worker's entry point.
2. `test_ingestion_supervisor.py::test_real_worker_via_supervisor_classifies_doc_types`
   — constructs an `IngesterSupervisor` with
   `worker_module="docserver.ingestion_worker"` (the real one, not the
   tmp-path fake-worker used by every other supervisor test), points
   `DOCSERVER_*` env vars at a tmp source repo + tmp `doc_types.yaml`,
   and calls `run_subprocess_cycle()`. After the subprocess exits the
   test opens the KB and asserts the doc has `type="journal"`. This is
   the only test in the suite that exercises the supervisor's real
   `Popen` against the real worker module — every other supervisor test
   uses a tiny fake-worker module under `_fake_workers/` for hermetic
   speed. The end-to-end test takes ~2s and is worth the cost: a future
   refactor that breaks env propagation (e.g. someone changes
   `_worker_env` to a sanitised allowlist) would slip past every direct
   `main()` test but would be caught here.

Also added `test_main_returns_one_when_doc_types_config_invalid` —
malformed `doc_types.yaml` must fail the worker loudly rather than
silently classify everything as fallback. The new
`load_doc_types_config` call sits inside a `try`/`except`; the test pins
that contract.

## Verification

```
uv run pytest tests/ -q            # 489 passed (was 484), 86% coverage
uv run ruff check src/ tests/      # All checks passed!
```

Sanity check: stashed only the two production fixes, kept the tests,
re-ran the four new tests — all four failed. Restored fixes — all pass.
The tests are correctly checking the bug surface, not a tautology.

## What this does *not* fix

1. **Production data already in SQLite with stale `type='documentation'`.**
   The conditional backfill in `KnowledgeBase.backfill_types_if_needed`
   gates on `SHA256(doc_types.yaml)` stored in the `meta` table. On the
   first deploy of this fix, if the yaml itself hasn't changed, the hash
   will match and the backfill won't run — existing rows stay as
   `documentation`. There are two ways to force a reclass:
   - Touch `doc_types.yaml` (whitespace edit changes the hash). The
     next server start will reclassify every parent doc and sync Chroma
     metadata in place.
   - Or `DELETE FROM meta WHERE key = 'doc_types_config_hash'` and
     restart — same effect.
   - Either way: a one-shot operator step after deployment. Not
     automated by this PR.
2. **Webapp-side handling of `type=null`.** The endpoint now returns
   `null` for orphan bookmarks. The bookmarks page already treats
   missing `type` as falling through to `"documentation"` for the
   category grouping — confirmed by reading the round-2 journal entry —
   so no webapp change is needed.

## Cross-references

- Round-2 webapp prompt that flagged both bugs:
  `webapp/journal/260515-bookmarks-route-ui-round2.md`, "Backend
  follow-ups required" section.
- Original classifier work that missed the subprocess path:
  `server/journal/260513-doc-types-and-exclude.md`.
- Supervisor / worker architecture (why the worker is a subprocess at
  all): `server/journal/260429-ingestion-subprocess-and-chroma-
  sidecar.md`.
