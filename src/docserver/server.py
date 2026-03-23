"""MCP server exposing documentation search and query tools via FastMCP."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from docserver.config import Config, load_config

if TYPE_CHECKING:
    from starlette.requests import Request
from docserver.ingestion import Ingester
from docserver.knowledge_base import KnowledgeBase
from docserver.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Module-level references, initialized by init_app() or run_server().
_kb: KnowledgeBase | None = None
_ingester: Ingester | None = None
_config: Config | None = None


def _get_kb() -> KnowledgeBase:
    assert _kb is not None, "Server not initialized — call init_app() first"
    return _kb


def _get_ingester() -> Ingester:
    assert _ingester is not None, "Server not initialized — call init_app() first"
    return _ingester


def create_mcp(config: Config) -> FastMCP:
    """Create and return the FastMCP server instance with all tools registered."""
    server = FastMCP(
        "Documentation Server",
        instructions="Search and query documentation from indexed git repositories",
        host=config.server_host,
        port=config.server_port,
    )

    # ---- Health endpoint ------------------------------------------------

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        try:
            kb = _get_kb()
            summary = kb.get_sources_summary()
            return JSONResponse(
                {
                    "status": "ok",
                    "total_sources": len(summary),
                    "total_chunks": sum(s.get("chunk_count", 0) for s in summary),
                    "sources": summary,
                }
            )
        except Exception:
            logger.exception("Health check failed.")
            return JSONResponse({"status": "error"}, status_code=503)

    # ---- Tools ----------------------------------------------------------

    @server.tool()
    def search_docs(query: str, num_results: int = 10, source: str = "") -> str:
        """Semantic search across all indexed documentation.

        Use this to find documentation relevant to a natural language question,
        e.g. "how does service X communicate with service Y" or "what ports
        are used on the foo VM".

        Args:
            query: Natural language search query.
            num_results: Maximum number of results to return (default 10).
            source: Optional source name to filter results to a specific repo.
        """
        kb = _get_kb()
        num_results = max(1, min(num_results, 100))
        t0 = time.monotonic()
        results = kb.search(
            query=query,
            n_results=num_results,
            source_filter=source or None,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "search_docs query=%r results=%d duration_ms=%d",
            query,
            len(results),
            duration_ms,
            extra={"event": "search", "duration_ms": duration_ms},
        )

        if not results:
            return "No matching documents found."

        output_parts = []
        for r in results:
            meta = r.get("metadata", {})
            output_parts.append(
                f"--- {meta.get('title', 'Untitled')} ---\n"
                f"Source: {meta.get('source', '?')} | "
                f"File: {meta.get('file_path', '?')} | "
                f"Score: {r.get('score', '?'):.4f}\n\n"
                f"{r.get('content', '')}\n"
            )

        return "\n".join(output_parts)

    @server.tool()
    def query_docs(
        source: str = "",
        file_path_contains: str = "",
        title_contains: str = "",
        created_after: str = "",
        created_before: str = "",
        limit: int = 20,
    ) -> str:
        """Structured query for document metadata.

        Use this to answer questions like "when was documentation about X created",
        "list all docs in source Y", or "what files were added after date Z".

        Args:
            source: Filter by source name.
            file_path_contains: Filter by substring in file path.
            title_contains: Filter by substring in title.
            created_after: ISO date string, e.g. "2024-01-01".
            created_before: ISO date string.
            limit: Max results (default 20).
        """
        kb = _get_kb()
        limit = max(1, min(limit, 100))
        docs = kb.query_documents(
            source=source or None,
            file_path_contains=file_path_contains or None,
            title_contains=title_contains or None,
            created_after=created_after or None,
            created_before=created_before or None,
            limit=limit,
        )

        if not docs:
            return "No matching documents found."

        return json.dumps(docs, indent=2, default=str)

    @server.tool()
    def get_document(doc_id: str) -> str:
        """Retrieve a specific document by its ID.

        Document IDs have the format "source_name:relative/path" for parent
        documents, or "source_name:relative/path#chunkN" for chunks.

        Args:
            doc_id: The document ID to retrieve.
        """
        kb = _get_kb()
        doc = kb.get_document(doc_id)
        if doc is None:
            return f"Document '{doc_id}' not found."
        return json.dumps(doc, indent=2, default=str)

    @server.tool()
    def list_sources() -> str:
        """List all configured documentation sources and their indexing status.

        Returns source names, file counts, chunk counts, and last indexed time.
        """
        kb = _get_kb()
        summary = kb.get_sources_summary()

        if not summary:
            return "No sources have been indexed yet."

        return json.dumps(summary, indent=2, default=str)

    @server.tool()
    def reindex(source: str = "") -> str:
        """Trigger an immediate re-indexing of documentation sources.

        Args:
            source: Optional source name to re-index only that source.
                   If empty, re-indexes all sources.
        """
        ingester = _get_ingester()
        t0 = time.monotonic()
        stats = ingester.run_once(sources=[source] if source else None)
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "reindex completed duration_ms=%d stats=%s",
            duration_ms,
            stats,
            extra={"event": "reindex", "duration_ms": duration_ms, "stats": stats},
        )

        if source and source not in stats:
            return f"Source '{source}' not found."

        return json.dumps(stats, indent=2)

    return server


def init_app(config: Config | None = None) -> FastMCP:
    """Initialize the application: KB, ingester, and MCP server.

    Useful for testing — pass a custom Config to avoid touching real state.
    """
    global _kb, _ingester, _config

    if config is None:
        config = load_config()
    _config = config

    _kb = KnowledgeBase(config.data_dir)
    _ingester = Ingester(config, _kb)

    return create_mcp(config)


def run_server() -> None:
    """Start the ingestion scheduler and run the MCP server."""
    json_output = os.environ.get("DOCSERVER_LOG_FORMAT", "json") == "json"
    log_level = os.environ.get("DOCSERVER_LOG_LEVEL", "INFO")
    setup_logging(level=log_level, json_output=json_output)

    mcp = init_app()

    logger.info(
        "Starting documentation MCP server on %s:%d",
        _config.server_host,
        _config.server_port,
        extra={"event": "startup"},
    )
    logger.info(
        "Configured sources: %s",
        [s.name for s in _config.sources],
        extra={"event": "startup"},
    )

    _ingester.start()

    try:
        mcp.run(transport="streamable-http")
    finally:
        _ingester.stop()
        _kb.close()
