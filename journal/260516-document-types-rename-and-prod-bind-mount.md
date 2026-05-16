# 2026-05-16 — Rename `doc_types.yaml` → `document-types.yml` + prod bind mount

Two cleanups bundled because they both need to ship before the doc-type
classifier actually does anything in production:

1. **Rename the file from `doc_types.yaml` to `document-types.yml`.**
   Prod was already standardised on the hyphenated, `.yml` spelling
   (`/srv/infra/docserver/config/document-types.yml`); the codebase
   defaulted to the underscored, `.yaml` version. Rename the codebase
   side to match prod.
2. **Add the bind mount to `docker-compose.yml`.** The classifier work
   shipped on 260513 added the env-var resolution and the
   `backfill_types_if_needed` gate, but the committed compose file only
   bind-mounted `sources.yaml`. Production has been running with no
   `doc_types.yaml` visible inside the container at all, so the W2
   subprocess fix on 260515 was necessary but not sufficient — there was
   no config for the worker to read.

## Scope of the rename

Filename-only, with the env var name updated for consistency. Python
identifiers and log-event tags stay as they were, because hyphens are
not valid in either and renaming them is pure churn:

| Renamed                             | From → To                                                              |
|-------------------------------------|------------------------------------------------------------------------|
| Example config                      | `config/doc_types.example.yaml` → `config/document-types.example.yml`  |
| Default path in `config.py`         | `/config/doc_types.yaml` → `/config/document-types.yml`                |
| Default path in `server.py`         | same                                                                   |
| Env var                             | `DOCSERVER_DOC_TYPES_CONFIG` → `DOCSERVER_DOCUMENT_TYPES_CONFIG`        |
| `.gitignore`                        | added `config/document-types.yml` + `.local.yml`                       |
| `docker-compose.yml`                | added bind mount `./config/document-types.yml:/config/document-types.yml:ro` |
| `docs/architecture.md`              | filename + env-var references                                          |
| Docstrings + comments in code       | filename references                                                    |
| Test fixture filenames + env writes | `tmp_path / "doc_types.yaml"` → `tmp_path / "document-types.yml"`      |

| Deliberately not renamed            | Why                                                                    |
|-------------------------------------|------------------------------------------------------------------------|
| Python `DocTypesConfig` class       | identifier; hyphens not valid; pure-stylistic churn                    |
| Python `load_doc_types_config`      | same                                                                   |
| Python `classify_doc_type`          | same                                                                   |
| Python `_doc_types_config` attr     | same                                                                   |
| Log event tags `doc_types_*`        | stable IDs for log aggregation / future dashboards                     |
| SQLite `meta` key `doc_types_hash`  | persists across restarts; renaming would orphan existing prod data     |
| Constant `_DEFAULT_DOC_TYPES`       | identifier                                                             |

The env-var rename is safe because the var was first published in W2's
ship yesterday (`a4d3967`) — no existing operator scripts or compose
overrides could reference the old name yet.

## How reclassification fires on prod

`KnowledgeBase.backfill_types_if_needed` gates on
`SHA256(doc_types_path)` stored under the `doc_types_hash` key of the
`meta` table. Production's stored hash is currently `""` — the old
default path `/config/doc_types.yaml` never existed inside the
container, so `_hash_file` returned the empty sentinel. After deploy:

1. `server.init_app` resolves `doc_types_path` to
   `/config/document-types.yml` (the new default).
2. Compose bind-mounts the host's
   `/srv/infra/docserver/config/document-types.yml` to that path.
3. `_hash_file` returns the real SHA → mismatch with the stored `""` →
   `backfill_types_if_needed` walks every doc, reclassifies, syncs
   ChromaDB chunk metadata, writes the new hash to `meta`.
4. Subsequent restarts short-circuit (hash matches).

No manual `DELETE FROM meta` step is needed for this deploy. The
operator only needs `docker compose pull && docker compose up -d` on
the prod host.

## Operator steps for prod

1. Confirm host file exists:
   ```
   ls -la /srv/infra/docserver/config/document-types.yml
   ```
   It already does (per the user — that's what kicked this off).
2. Pull and restart on the prod host:
   ```
   cd /srv/infra/docserver
   docker compose pull
   docker compose up -d
   ```
3. Verify in the docserver logs:
   ```
   docker logs unified-documentation-server 2>&1 | grep doc_types_backfill
   ```
   Expect a `doc_types_backfill_start` followed by
   `doc_types_backfill_complete updated=<N>` where N is roughly the
   number of docs whose path matches one of the global_rules patterns
   (anything under `journal/`, `prompts/`, plus the `not-docs` ones).
4. Spot-check via the bookmarks API: pick a bookmarked doc that should
   now classify as `journal`, hit `/api/bookmarks`, confirm `type`
   reads as `"journal"` rather than `"documentation"`.

## What this does *not* change

1. **Stage 2 internals.** The classifier logic (`classify_doc_type`,
   `_pattern_matches`, `DocTypeRule`, the SHA256 cache key) is
   unchanged. The two paths that consume the loaded config — `Ingester`
   for new docs, `backfill_types_if_needed` for existing docs — are
   both unchanged.
2. **Public Python API.** Anyone with `from docserver.config import
   load_doc_types_config, DocTypesConfig, classify_doc_type` is
   unaffected; only the env-var name and the default filename
   changed. There are no consumers outside this repo today, but the
   convention matters once webapp / tests / scripts start importing.
3. **Existing data.** No SQLite migration. The `meta` key
   `doc_types_hash` stays under its existing name (renaming it would
   orphan the prod hash and trigger an unnecessary reclassification on
   every deploy until reseeded).

## Verification

```
uv run pytest tests/ -q   # 489 passed, 86% coverage (unchanged from 260515)
uv run ruff check src/ tests/  # All checks passed!
grep -rn 'doc_types\.yaml\|DOCSERVER_DOC_TYPES\|/config/doc_types' \
  src/ docs/ tests/ config/ docker-compose.yml .gitignore
# (no results — only historical journal entries still reference the old
#  name, which is by design)
```

Historical journal entries on 260513 and 260515 still mention
`doc_types.yaml` — left alone per the project convention that journals
are point-in-time records, not living documents.
