# Unified Documentation Server

## Project Structure

- `src/docserver/` - Main package
  - `config.py` - YAML + env var configuration
  - `ingestion.py` - Git repo syncing, markdown parsing, chunking
  - `knowledge_base.py` - SQLite (incl. FTS5) + ChromaDB storage and hybrid search
  - `embedding.py` - ONNX cross-encoder for the dense leg of L1
  - `reranker.py` - ONNX cross-encoder for L2 reranking
  - `server.py` - FastMCP server with tool definitions + agentic chat endpoint
  - `conversations.py` - Server-side conversation persistence (SQLite)
  - `bookmarks.py` - Document bookmarks/favourites persistence (SQLite)
  - `logging_config.py` - Structured JSON logging for Docker
  - `__main__.py` - Entry point
- `tests/` - pytest test suite
- `config/` - Configuration files (sources.yaml)
- `docs/` - Project documentation
- `journal/` - Development journal

## Key Commands

- `uv sync --group dev` - Install dependencies
- `uv run pytest tests/ -v` - Run tests
- `uv run python -m docserver` - Run server locally
- `docker compose up -d` - Run containerized

## Architecture Decisions

- **uv** for dependency and environment management
- **Hybrid search pipeline (L1 + L2)**:
  - L1 retrieval: SQLite FTS5 BM25 (over chunk content + title, title-weighted 2x)
    + ChromaDB cosine (all-mpnet-base-v2, 768 dims), fused with Reciprocal Rank
    Fusion (k=60). Top 50 chunks pass to L2.
  - L2 reranking: cross-encoder/ms-marco-MiniLM-L6-v2 (ONNX int8, ~23MB on disk,
    ~85MB resident). Reorders L1 candidates with full query–passage attention,
    then dedup-to-parent picks the best chunk per parent.
  - Both models lazy-loaded on first search; pre-baked into the Docker image so
    production never downloads on cold start.
  - The legacy synthetic-0.5 title-keyword fallback was deleted — FTS5 with the
    title column subsumes it without the score-scale hack. Score field
    semantics are "higher = better" (was: lower=better cosine distance).
- **SQLite** for structured metadata queries (dates, paths, sources)
- Documents are chunked at ~400 chars on section/paragraph boundaries with 100-char overlap; parent doc metadata stored separately for structured queries
- Doc IDs follow pattern: `{source}:{path}` (parent) and `{source}:{path}#chunk{N}` (chunks)
- Chunks go into ChromaDB (vectors), the FTS5 `chunks_fts` virtual table (BM25), and the SQLite `documents` table (raw content). Parent docs live only in `documents`.
- APScheduler runs ingestion on a background thread
- MCP transport: streamable HTTP on port 8080
- Chat agent uses Claude API tool-use loop (not single-shot RAG) with search_docs, query_docs, get_document, list_sources, get_bookmarks
- Chat conversations persisted in `conversations.db` (separate from docserver.db) for review and resumption
- Bookmarks persisted in `bookmarks.db` with user_id column for future multi-user support
