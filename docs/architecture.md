# Architecture

## Overview

The Documentation MCP Server has three layers:

```
Git Repos (local/remote)
        |
        v
  [Ingestion Layer]     Polls repos, parses markdown, chunks text
        |
        v
  [Knowledge Base]      SQLite (metadata) + ChromaDB (vector embeddings)
        |
        v
  [MCP Server]          FastMCP with streamable HTTP transport
```

## Ingestion Layer

**Module:** `src/docserver/ingestion.py`

The ingestion layer manages git repositories and converts markdown files into searchable documents.

### Components

- **RepoManager**: Handles git operations for a single source. Remote repos are cloned into `/data/clones/<source_name>/` on first run, then pulled on subsequent cycles. Local repos (mounted as Docker volumes) are pulled if they have a git remote, or treated as static directories otherwise.

- **DocumentParser**: Reads files and extracts metadata. Supports two parsing modes:
  - **Markdown** (`parse_markdown`): Reads text content, extracts title from first `#` heading (or filename as fallback), creation date (from `git log --follow --diff-filter=A`), modification time (filesystem mtime), and file size. Content is stored for chunking and search.
  - **Binary** (`parse_binary`): For non-text files like PDFs (extensions in `BINARY_EXTENSIONS`). Stores metadata only (title from filename stem, dates, size) with empty content. Binary files are not chunked or embedded — they appear in the sidebar tree but are served raw via the `/api/files/` endpoint rather than rendered as text.

- **Chunking** (section-aware): Documents are split into ~400-character chunks using a multi-step strategy:
  1. Parse markdown headings into a section tree
  2. Within each section, group blocks (paragraphs, lists, code fences) into chunks at ~400 chars
  3. Prepend each chunk with its section heading path (e.g. `[Setup > Networking > Ports]`) so chunks are self-describing in isolation
  4. Add ~100 chars of overlap from the previous chunk (prefixed with `[...]`) to preserve context across boundaries

  Special handling:
  - **Lists**: Contiguous list items (`-`, `*`, `1.`) are kept together as a single block
  - **Code fences**: Everything between ` ``` ` markers stays in one block; headings inside code fences are ignored
  - **Oversized blocks**: Blocks larger than the target size are emitted whole rather than split mid-content

- **Ingester**: Orchestrates the full cycle via APScheduler. On each tick:
  1. Clean up orphaned sources — detects renames via URL matching before deleting (see below)
  2. Sync all repos in parallel using a thread pool (clone on first run, then fetch + hard reset to match remote)
  3. Enumerate files matching glob patterns (default: `**/*.md`). Add `**/*.pdf` to include PDFs. Root-level `README.md` files are always included even when custom patterns are specified.
  4. Bulk-fetch git creation dates for all files in a single `git log` call (with per-file fallback for renamed files)
  5. Compare SHA-256 content hash against stored hash — skip unchanged files
  6. Parse and chunk changed files
  7. Batch-upsert into the knowledge base (SQLite via `executemany` in one transaction, ChromaDB in capped batches of 64 to leverage the embedding model's batch_size=32)
  8. Delete stale documents that no longer exist in the repo

  **Performance optimizations:**
  - **Parallel source sync**: Git fetch/pull for multiple sources runs concurrently via `ThreadPoolExecutor` (up to 4 workers). Ingestion (KB writes) remains sequential since SQLite and ChromaDB are not thread-safe.
  - **Bulk git dates**: Instead of spawning one `git log` subprocess per file, a single `git log --diff-filter=A --name-only` call retrieves creation dates for all files. Files not found (e.g. renamed) fall back to per-file lookup.
  - **Batch ChromaDB upserts**: Chunks are collected and upserted in batches (flushed every 64 items), allowing the ONNX embedding model to process 32 documents at a time instead of one-by-one.
  - **Batch SQLite upserts**: Uses `executemany` within a single connection/transaction instead of one connection per document.
  - **Content hash skipping**: Uses SHA-256 content hashing (not filesystem mtime), so files are correctly skipped even after a fresh clone where all mtimes are reset.

  Typical poll cycles (no changes) finish in seconds. Each file being indexed is logged with a progress counter, change type (`new` or `modified`), file path, and chunk count. Batch flushes are logged with item counts and running totals. Completion stats break down files into new/modified/skipped/deleted/error counts.

  Orphan cleanup only runs during full ingestion cycles (not when specific sources are targeted via the `sources` parameter).

- **Rename detection**: When a source name changes in `sources.yaml` but points to the same repository, the ingester detects this and migrates data in-place instead of deleting and re-indexing from scratch. Detection works by comparing the git remote URL of orphaned clone directories against configured source URLs. URLs are normalised (stripped of credentials, `.git` suffix, trailing slashes, case-insensitive) so that `https://token@github.com/User/Repo.git` and `https://github.com/user/repo` match correctly. When a rename is detected:
  1. SQLite doc_ids and source columns are updated to the new name
  2. ChromaDB entries are migrated with their existing embeddings preserved (no re-computation)
  3. The clone directory is renamed

  This avoids expensive re-cloning and re-embedding. A typical rename migration completes in under a second versus minutes for a full re-index. If the new name already has data in the KB, the rename is skipped and the orphan is deleted normally (prevents false matches).

### Document ID Scheme

Each file produces two types of records:

| Type | Doc ID Format | Purpose |
|------|--------------|---------|
| Parent doc | `source:relative/path.md` | Metadata-only record for structured queries |
| Chunk      | `source:relative/path.md#chunk0` | Text content for semantic search            |

Parent docs have `is_chunk = False` and are stored in SQLite only. Chunks have `is_chunk = True` and are stored in both SQLite and ChromaDB.

Binary files (e.g. PDFs) are stored as parent docs with empty content and zero chunks. They appear in the document tree but have no vector embeddings and are not searchable. The raw file is served via the `/api/files/` endpoint.

## Knowledge Base

**Module:** `src/docserver/knowledge_base.py`

Dual-store design:

### SQLite (`/data/documents.db`)

Stores all document metadata in a single `documents` table:

```sql
doc_id        TEXT PRIMARY KEY
source        TEXT NOT NULL
file_path     TEXT NOT NULL
title         TEXT
content       TEXT
chunk_index   INTEGER
total_chunks  INTEGER
created_at    TEXT           -- ISO timestamp from git history
modified_at   TEXT           -- ISO timestamp from file mtime
indexed_at    TEXT           -- ISO timestamp when we indexed it
size_bytes    INTEGER
is_chunk      BOOLEAN
section_path  TEXT           -- heading hierarchy, e.g. "Setup > Ports"
content_hash  TEXT           -- SHA-256 of file content for change detection
```

Supports structured queries like:
- "When was documentation about X created?"
- "List all files in source Y"
- "What was indexed after date Z?"

Parent docs (non-chunks) are returned by `query_documents()` so results represent whole files, not fragments.

### ChromaDB (`/data/chroma/`)

Stores chunk text with vector embeddings for semantic similarity search. Uses the **all-mpnet-base-v2** embedding model via ONNX Runtime (768 dimensions, ~500MB RAM, runs locally with no external API calls). ONNX Runtime was chosen over PyTorch-based sentence-transformers to keep Docker images small (~1GB vs ~8GB) and build times fast.

Only chunks are stored in ChromaDB. Parent docs are excluded since they have no content body. Each chunk's metadata in ChromaDB includes `source`, `file_path`, `title`, `chunk_index`, `total_chunks`, and `section_path` for filtering.

## MCP Server

**Module:** `src/docserver/server.py`

Built with FastMCP, exposes five tools over streamable HTTP:

| Tool               | Use Case                                                          |
|--------------------|-------------------------------------------------------------------|
| `search_docs`      | Natural language questions ("what ports does VM X use?")          |
| `query_docs`       | Structured filters (source, path, title, date range)              |
| `get_document`     | Retrieve a specific document by its ID                            |
| `list_sources`     | Show all sources with file/chunk counts and last indexed time     |
| `ingestion_status` | Full indexing status: configured vs indexed sources, missing gaps |
| `reindex`          | Trigger an immediate ingestion cycle                              |

### Endpoints

- `/mcp` — MCP protocol endpoint (streamable HTTP transport)
- `/health` — Health check returning status, total source/chunk counts, and per-source breakdown (file count, chunk count, last indexed time)
- `/rescan` (POST) — Trigger an immediate ingestion cycle. Optional `?source=name` query param to rescan a single source. Returns stats with duration.
- `/api/tree` (GET) — Document tree organized by source and category. Each source has `root_docs` (root-level files like README.md), `docs` (files in subdirectories), and `journal` (files under journal/).
- `/api/documents/:doc_id` (GET) — Full document content reassembled from chunks.
- `/api/files/:doc_id` (GET) — Raw file served from disk with correct MIME type and `Content-Disposition: inline`. Used by the UI to embed PDFs in an iframe. Includes path traversal protection.
- `/api/search?q=&source=&limit=` (GET) — Semantic search via ChromaDB.
- `/api/chat` (POST) — RAG-powered chat (searches docs, sends context to Claude).

### Logging

**Module:** `src/docserver/logging_config.py`

Structured JSON logging to stdout for Docker log collection. Each log line is a JSON object with `timestamp`, `level`, `logger`, `message`, and any extra structured fields (all extra fields are included automatically). Configurable via `DOCSERVER_LOG_FORMAT` (json/text) and `DOCSERVER_LOG_LEVEL`.

Logging covers every phase: config loading, KB initialization, ingestion cycle start/end, per-source sync/clone/file-enumeration/upsert/cleanup, embedding model status, and search queries. Credentials are redacted in all log output. Each log line includes an `event` field for easy filtering (e.g. `sync_start`, `ingestion_done`, `clone_start`).

## Deployment

Single Docker container (Python 3.13) running all three layers. The ONNX embedding model files are pre-downloaded during the Docker build into `/app/models-cache`. On first startup, this is copied to `/data/models/` (the persistent volume) so subsequent restarts load the model instantly without re-downloading.

Source paths in `sources.yaml` support `${VAR}` environment variable expansion for authenticating with private repositories (e.g. `https://${GITHUB_TOKEN}@github.com/...`).

Data persists in a named Docker volume mounted at `/data`. Local repos are mounted read-only into `/repos/`.

```yaml
# docker-compose.yml volumes
volumes:
  - docserver-data:/data                    # Persistent storage
  - ./config/sources.yaml:/config/sources.yaml:ro  # Config
  - /path/to/repo:/repos/repo-name:ro      # Local repos
```
