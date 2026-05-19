# 2026-05-19 — get_source_files surfaces line_count

## Why

The webapp's redesigned per-source view (flat table per directory) needs a
"Lines" column. `get_source_files` only returned `doc_id, file_path, title,
modified_at, type` — no length metric usable for a list view (`size_bytes`
lives elsewhere and bytes aren't what the UI wants).

## What

Added a derived `line_count` to the `get_source_files` query, computed from
the already-stored parent-doc `content`:

```sql
CASE WHEN content IS NULL OR content = '' THEN 0
     ELSE LENGTH(content) - LENGTH(REPLACE(content, CHAR(10), '')) + 1 END
AS line_count
```

i.e. newline count + 1; NULL/empty content (PDFs, contentless rows) → 0. No
schema change — parent docs already store `content` in the `documents` table.
Both tree endpoints (`/api/sources/tree`, `/api/sources/{name}/tree`) return
`get_source_files` rows verbatim, so the field flows through with no
server.py change.

## Tests

Added `test_get_source_files_includes_line_count` (3-line doc → 3, empty doc →
0). Full `test_knowledge_base.py` + `test_server.py` green (131 passed).

## Cross-repo

Paired with webapp branch `eng-ui-fixes`, which adds `line_count` to the
`TreeDocument` TS type and renders the column. Commit/push the two repos
separately.
