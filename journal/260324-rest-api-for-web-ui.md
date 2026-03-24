# REST API Endpoints for Web UI

**Date:** 2026-03-24

## What was done

Added REST API endpoints to the MCP server to support a new web UI frontend:

### New endpoints
- `GET /api/tree` — Document tree organized by source and category (docs vs journal)
- `GET /api/documents/{doc_id}` — Full document content by ID
- `GET /api/search?q=&source=&limit=` — Semantic search returning deduplicated parent docs
- `POST /api/chat` — RAG-powered chat (searches docs for context, calls Claude API)

### New KnowledgeBase methods
- `get_document_tree()` — Groups all parent documents by source and categorizes them as "docs" or "journal" based on file path
- `search_documents()` — Wraps ChromaDB chunk search and deduplicates results back to parent documents with metadata

### Dependencies
- Added `anthropic` SDK for the chat endpoint

## Key decisions

- **Manual CORS headers** via `_cors_json()` helper rather than Starlette middleware, since only the `/api/*` routes need CORS (the MCP transport routes should not have it)
- **Chat uses Claude Sonnet** (`claude-sonnet-4-20250514`) for cost efficiency — the chat endpoint is user-facing and may get frequent use
- **RAG context**: Chat endpoint includes both the currently viewed document (if any) and the top 5 search results as context for the LLM
- **History capped at 10 messages** to keep token usage reasonable
- **`ANTHROPIC_API_KEY`** must be set as an env var on the server for chat to work; endpoint returns 503 if missing

## Tests

Added 2 new tests to `test_knowledge_base.py`:
- `test_get_document_tree` — Verifies tree structure, category assignment, and chunk exclusion
- `test_search_documents_deduplicates` — Verifies chunk results are deduplicated to parent docs

All 140 tests pass. KB coverage improved from 80% to 95%.

## Follow-up fixes (same session)

- **Parent docs now store full content** — Previously parent docs had empty content (only chunks had text). Changed ingestion to pass the full markdown content to parent doc upserts. This means the UI serves the exact original document instead of a reconstruction from chunks.
- **Force rescan** — Added `?force=true` parameter to `/rescan` endpoint, bypassing the content hash check. Needed when the storage format changes but file content hasn't.
- **`get_full_document()`** — Added as a fallback method that reassembles content from chunks for any parent docs that were indexed before the content storage change. The API endpoint uses `get_document()` directly since new ingestion stores content on parents.
