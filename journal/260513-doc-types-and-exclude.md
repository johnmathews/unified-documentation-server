# Stage 2 backend — per-document types and `exclude_types`

Stage 2 W2.1–W2.5 land in a single backend commit on
`feature/repo-structure-stage2`. The pipeline now classifies every doc and
chunk with one of four types (`documentation`, `journal`, `prompt`,
`not-docs`), exposes that type on every response payload, and accepts an
`exclude_types` filter on search, query, and chat.

## What changed

1. **Schema (W2.1).** New `type TEXT DEFAULT 'documentation'` column on
   `documents`, idempotent index `idx_documents_type`, and a sibling
   `meta (key, value)` table for the backfill hash. The `meta` table lives
   in `_SCHEMA` next to `documents` and `source_status` rather than
   `_MIGRATIONS` — `CREATE TABLE IF NOT EXISTS` is more discoverable beside
   the other table declarations. All upserts now write `type` (defaulting
   to `'documentation'`) so `INSERT OR REPLACE` doesn't strip the value back
   to default on every re-ingest.
2. **Classifier config (W2.2).** New `config/doc_types.example.yaml` plus
   `DocTypesConfig`, `load_doc_types_config()`, and `classify_doc_type()` in
   `config.py`. Resolution: `DOCSERVER_DOC_TYPES_CONFIG` env var (default
   `/config/doc_types.yaml`). Missing config → fallback type for everything.
   Rules are first-match-wins with per-source before global. Pattern
   matching is `fnmatch.fnmatch` with a small gitignore-style shim for
   leading `**/`: stripping the prefix on a fnmatch miss lets a single
   `**/journal/**` rule match both top-level and nested journals.
3. **Ingestion + conditional backfill (W2.3).** `Ingester` accepts a
   `doc_types_config` and classifies each doc at the call site after
   `parse_markdown` / `parse_binary` — the parser stays
   single-responsibility (raw → ParsedDocument), classification is the
   Ingester's job. `KnowledgeBase.backfill_types_if_needed()` compares a
   SHA256 of `doc_types.yaml` against the value stored in `meta`. When the
   hashes match the call is O(1); on a mismatch (including first-ever run),
   every parent row and chunk is reclassified, ChromaDB metadata for chunks
   is updated in place (preserving other keys), and the new hash is
   written. `server.init_app()` is the wiring point — it loads the config
   after `KnowledgeBase(...)` is constructed and triggers the conditional
   backfill before instantiating the Ingester, so the first scheduled
   ingestion cycle sees correct types.
4. **`exclude_types` plumbing (W2.4).**
   - `KnowledgeBase.search()`, `_dense_search()`, `_bm25_search_chunks()`,
     and `query_documents()` take an optional `exclude_types: list[str]`.
   - BM25 always joins `documents` so the score column survives and `type`
     is returned in metadata. The plan asked for a JOIN only when filtering;
     we unified the two SQL paths because the JOIN cost is negligible
     against the FTS scan and a single SQL path is cheaper to maintain.
   - Dense leg composes `{"$and": [...]}` when both `source` and `type`
     filters are active so Chroma's `where` operator stays valid.
   - MCP `search_docs` and `query_docs` tools accept `exclude_types` as a
     comma-separated string.
   - REST `POST /api/chat` and `POST /api/chat/stream` accept
     `exclude_types: list[str]` in the request body. The chat dispatch
     unions the request-level policy with whatever the model passes — the
     UI toggle (W2.8 / W3.6) is policy, not a model suggestion.
   - Chat agent's tool schema also exposes `exclude_types` (array<string>)
     so the model *can* self-exclude when relevant. Per the open question
     the user resolved: the agent has no system-prompt nudge to do so —
     UI controls policy.
5. **`type` exposed everywhere (W2.5).** `get_source_files`,
   `query_documents`, `get_document` (via `SELECT *`), and BM25 result
   metadata all carry `type`. Sets up W2.6 (TypeBadge) and W2.7
   (typeFilters store) on the webapp side.

## Deviations from the plan

- **Classification call site.** Plan said thread `doc_type` through
  `DocumentParser.parse_markdown` / `parse_binary`. We classify at the
  call site instead. Cleaner separation; one fewer parser argument; no
  test fallout because no caller of `DocumentParser` outside the Ingester.
- **Meta table location.** Plan put `CREATE TABLE meta` in `_MIGRATIONS`;
  we put it in `_SCHEMA` next to the other `CREATE TABLE IF NOT EXISTS`
  statements. `_MIGRATIONS` should be reserved for non-idempotent ALTER.
- **Single BM25 SQL path.** Plan branched on `exclude_types` to avoid the
  JOIN when not filtering. We always JOIN so the result metadata can carry
  `type`. Confirmed no measurable regression on the BM25 test set.

## FTS5 JOIN check (replaces the W2.4 spike task)

The plan flagged an FTS5 JOIN spike — verifying that `bm25(chunks_fts, …)`
still produces correct ordering under
`FROM chunks_fts JOIN documents ON …`. We validated this directly in
`test_search_exclude_types_bm25`: a query that matches both a `journal`
chunk and a `documentation` chunk returns only the docs chunk when
`exclude_types=["journal"]`, ordering preserved. `bm25()` works with the
table name (not the alias) even under the JOIN, matching the SQLite docs.

## Gotchas to remember

1. `INSERT OR REPLACE` strips unlisted columns back to default. Every
   future column addition on `documents` needs to be threaded through
   both upsert sites or upserts will silently overwrite the new value.
2. ChromaDB metadata sync runs *inside* the backfill. If someone changes
   the chunk metadata shape, both `upsert_document` and `_sync_chroma_types`
   need to agree on the key set — otherwise drift surfaces when a backfill
   runs.
3. The gitignore-style `**/` shim is fnmatch + a one-line escape hatch,
   not a full gitignore implementation. If we need stricter semantics
   (e.g., trailing `/` to match directories only) the right move is to
   pull in `pathspec`. For now the existing tests cover the patterns
   anyone is likely to write.

## Test summary

- Full suite: 483 passed, 86% coverage (up from 467 / 86%).
- 12 new tests across `test_knowledge_base.py` and the new
  `test_doc_types_config.py`. Covers: default type, meta table,
  backfill conditional logic, classifier precedence/validation/env
  var resolution, `exclude_types` on both search legs and
  `query_documents`, and `type` in `get_source_files`.

## Open follow-ups

- Smoke test against the local dev stack still pending — plan calls for
  this after W2.7+W2.8 land on the webapp. Stage 2 backend is dark-launched
  with no consumer until then.
- `journal/**` etc. rules in the shipped `doc_types.example.yaml` cover
  the obvious cases; production `doc_types.yaml` should be tuned to
  whatever sources are configured at the time of switchover.
