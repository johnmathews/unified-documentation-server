# Folder-tree endpoints for the webapp sidebar

Stage 1 of a cross-repo plan (`.engineering-team/improvement-plan-260513-repo-structure.md`
in the workspace root) to make per-source structure legible in the webapp.
Stage 2 will add per-document types (`documentation`, `journal`, `prompt`,
`not-docs`) driven by a new `config/doc_types.yaml`; Stage 3 will promote the
chat panel to a `/chat` route.

## What this change ships

Two new REST endpoints expose per-source file lists. The webapp builds the
folder tree client-side from `file_path` — no server-side nesting.

- `GET /api/sources/{name}/tree` returns `{"source": name, "files": [...]}`
  for one source. 404 for unknown source names.
- `GET /api/sources/tree` returns `{"sources": [...]}` for every **configured**
  source. The bulk endpoint iterates `config.sources` rather than only those
  with rows in the KB, so a freshly-added but not-yet-indexed source still
  appears in the sidebar (empty `files: []`).
- New `KnowledgeBase.get_source_files(source)` powers both — SQL filtered to
  parents (`is_chunk = FALSE OR chunk_index IS NULL`) ordered by `file_path`.

## Decisions

- **Path prefix `/api/`** to match the existing convention (`/api/tree`,
  `/api/documents`, `/api/search`).
- **No MCP tool** for the source tree. The chat agent uses `search_docs` /
  `query_docs`; a raw file listing would add context noise without helping
  retrieval.
- **Bulk endpoint iterates configured sources, not just KB-populated ones.**
  Caught while booting the dev stack against a config where one source had
  no rows yet — the sidebar would have silently dropped it. The 404 check
  on the single-source endpoint stays strict (must be configured), so
  unknown names still fail loudly.

## Tests

- Backend went 456 → 464 passes (+8). Coverage stable at 86%.
- All tests live in `tests/test_server.py::TestSourcesTreeEndpoints` and
  `tests/test_knowledge_base.py::test_get_source_files_*`.

## Smoke test

Booted against the local-data DB with two configured sources (one indexed,
one empty). The webapp sidebar rendered the pi-harness folder structure
correctly: `.engineering-team/`, `journal/`, then root-level files at the
top. Tech Blog appeared with `(0)` and collapsed.
