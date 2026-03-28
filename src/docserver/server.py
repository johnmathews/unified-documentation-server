"""MCP server exposing documentation search and query tools via FastMCP."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import anthropic
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


def _cors_json(data: Any, status_code: int = 200) -> JSONResponse:
    """Return a JSONResponse with CORS headers for the UI."""
    return JSONResponse(
        data,
        status_code=status_code,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


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

    # ---- Rescan endpoint ------------------------------------------------

    _rescan_lock = threading.Lock()
    _rescan_running = False

    @server.custom_route("/rescan", methods=["POST"])
    async def rescan(request: Request) -> JSONResponse:
        """Trigger a background ingestion cycle. Optionally pass ?source=name&force=true."""
        nonlocal _rescan_running
        try:
            if _rescan_running:
                return JSONResponse({"status": "already_running"}, status_code=409)

            ingester = _get_ingester()
            source = request.query_params.get("source", "")
            force = request.query_params.get("force", "").lower() == "true"
            sources = [source] if source else None

            def _run_rescan() -> None:
                nonlocal _rescan_running
                try:
                    t0 = time.monotonic()
                    stats = ingester.run_once(sources=sources, force=force)
                    duration_ms = int((time.monotonic() - t0) * 1000)
                    logger.info(
                        "Rescan completed in %dms: %s",
                        duration_ms,
                        stats,
                        extra={"event": "rescan", "duration_ms": duration_ms, "stats": stats},
                    )
                except Exception:
                    logger.exception("Rescan failed.")
                finally:
                    _rescan_running = False

            with _rescan_lock:
                if _rescan_running:
                    return JSONResponse({"status": "already_running"}, status_code=409)
                _rescan_running = True

            thread = threading.Thread(target=_run_rescan, daemon=True)
            thread.start()

            return JSONResponse({"status": "started", "force": force, "sources": sources or "all"})
        except Exception:
            logger.exception("Rescan failed to start.")
            return JSONResponse({"status": "error"}, status_code=500)

    # ---- REST API for Web UI -------------------------------------------

    @server.custom_route("/api/tree", methods=["GET"])
    async def api_tree(request: Request) -> JSONResponse:
        """Return document tree: sources → categories → documents."""
        try:
            kb = _get_kb()
            tree = kb.get_document_tree()
            return _cors_json(tree)
        except Exception:
            logger.exception("API tree failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/documents/{doc_id:path}", methods=["GET"])
    async def api_get_document(request: Request) -> JSONResponse:
        """Return a single document by ID (URL-encoded)."""
        try:
            kb = _get_kb()
            doc_id = unquote(request.path_params["doc_id"])
            doc = kb.get_document(doc_id)
            if doc is None:
                return _cors_json({"error": "Not found"}, 404)
            return _cors_json(doc)
        except Exception:
            logger.exception("API get_document failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/search", methods=["GET"])
    async def api_search(request: Request) -> JSONResponse:
        """Search documents. Query params: q (required), source (optional), limit (optional)."""
        try:
            kb = _get_kb()
            query = request.query_params.get("q", "")
            if not query:
                return _cors_json({"error": "Missing 'q' parameter"}, 400)
            source = request.query_params.get("source", "") or None
            limit = min(int(request.query_params.get("limit", "20")), 100)
            results = kb.search_documents(query, n_results=limit, source_filter=source)
            return _cors_json(results)
        except Exception:
            logger.exception("API search failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/chat", methods=["POST", "OPTIONS"])
    async def api_chat(request: Request) -> JSONResponse:
        """Chat endpoint: RAG search + Claude API. Body: {message, doc_id?, history?}."""
        if request.method == "OPTIONS":
            return _cors_json({})

        try:
            kb = _get_kb()
            body = await request.json()
            message = body.get("message", "")
            if not message:
                return _cors_json({"error": "Missing 'message'"}, 400)

            current_doc_id = body.get("doc_id")
            history: list[dict[str, str]] = body.get("history", [])

            # Build context from current page + RAG search
            context_parts: list[str] = []

            if current_doc_id:
                current_doc = kb.get_document(current_doc_id)
                if current_doc:
                    context_parts.append(
                        f"The user is currently viewing this document:\n"
                        f"Title: {current_doc.get('title', 'Untitled')}\n"
                        f"Source: {current_doc.get('source', '?')}\n"
                        f"Path: {current_doc.get('file_path', '?')}\n\n"
                        f"{current_doc.get('content', '')}"
                    )

            # Add document inventory with indexing stats
            doc_tree = kb.get_document_tree()
            source_stats = {s["source"]: s for s in kb.get_sources_summary()}
            inventory_lines: list[str] = []
            total_files = 0
            total_chunks = 0
            for src in doc_tree:
                src_name = src["source"]
                stats = source_stats.get(src_name, {})
                file_count = stats.get("file_count", 0)
                chunk_count = stats.get("chunk_count", 0)
                last_indexed = stats.get("last_indexed", "never")
                total_files += file_count
                total_chunks += chunk_count

                root_docs = [d.get("title") or d.get("file_path", "?") for d in src.get("root_docs", [])]
                doc_titles = [d.get("title") or d.get("file_path", "?") for d in src["docs"]]
                journal_titles = [d.get("title") or d.get("file_path", "?") for d in src["journal"]]
                eng_titles = [d.get("title") or d.get("file_path", "?") for d in src.get("engineering_team", [])]

                inventory_lines.append(
                    f"**{src_name}** ({file_count} files, {chunk_count} chunks, last indexed: {last_indexed}):"
                )
                if root_docs:
                    inventory_lines.append(f"  Root docs ({len(root_docs)}): {', '.join(root_docs)}")
                if doc_titles:
                    inventory_lines.append(f"  Documentation ({len(doc_titles)}): {', '.join(doc_titles)}")
                if journal_titles:
                    inventory_lines.append(f"  Journal ({len(journal_titles)}): {', '.join(journal_titles)}")
                if eng_titles:
                    inventory_lines.append(f"  Engineering team ({len(eng_titles)}): {', '.join(eng_titles)}")

            inventory_header = (
                f"Documentation inventory: {len(doc_tree)} sources, "
                f"{total_files} files, {total_chunks} vector chunks.\n\n"
            )
            context_parts.insert(0, inventory_header + "\n".join(inventory_lines))

            search_results = kb.search(query=message, n_results=8)
            if search_results:
                rag_context = "\n\n---\n\n".join(
                    f"**{r['metadata'].get('title', 'Untitled')}** "
                    f"(source: {r['metadata'].get('source', '?')}, "
                    f"file: {r['metadata'].get('file_path', '?')})\n\n"
                    f"{r['content']}"
                    for r in search_results
                )
                context_parts.append(f"Relevant documentation excerpts:\n\n{rag_context}")

            system_prompt = (
                "You are a documentation assistant for a home server infrastructure project. "
                "You have full access to the indexed documentation inventory below, including "
                "per-source file counts, chunk counts, and last-indexed timestamps.\n\n"
                "Guidelines:\n"
                "- For questions about what's indexed, source status, or document counts, "
                "use the inventory stats provided below — you have complete information.\n"
                "- For questions about document content, use the search results below.\n"
                "- For structural questions ('what journal entries exist', 'which sources are "
                "indexed'), use the document inventory.\n"
                "- When asked about the most recent journal entry, look at the created_at dates "
                "in the inventory and search results.\n"
                "- Answer confidently from the data you have. Do not say 'I would need to use "
                "tools' or 'I cannot confirm' when the answer is in the inventory.\n"
                "- Be concise and direct."
            )
            if context_parts:
                system_prompt += "\n\n" + "\n\n---\n\n".join(context_parts)

            # Build messages from history + current
            messages: list[dict[str, str]] = []
            for h in history[-10:]:  # Keep last 10 exchanges
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": message})

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return _cors_json(
                    {"error": "ANTHROPIC_API_KEY not configured on server"}, 503
                )

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            )

            reply = response.content[0].text if response.content else ""
            return _cors_json({"reply": reply})

        except anthropic.APIError as exc:
            logger.exception("Anthropic API error in chat.")
            return _cors_json({"error": f"AI service error: {exc.message}"}, 502)
        except Exception:
            logger.exception("API chat failed.")
            return _cors_json({"error": "Internal error"}, 500)

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
    def ingestion_status() -> str:
        """Get detailed indexing status for all documentation sources.

        Returns per-source stats including file counts, chunk counts,
        last indexed time, and configured source names. Use this to
        answer questions about whether sources are fully indexed.
        """
        kb = _get_kb()
        summary = kb.get_sources_summary()
        configured = [s.name for s in config.sources]

        indexed_names = {s["source"] for s in summary}
        missing = [n for n in configured if n not in indexed_names]

        result = {
            "configured_sources": configured,
            "indexed_sources": summary,
            "total_files": sum(s.get("file_count", 0) for s in summary),
            "total_chunks": sum(s.get("chunk_count", 0) for s in summary),
            "missing_sources": missing,
            "fully_indexed": len(missing) == 0 and all(
                s.get("file_count", 0) > 0 for s in summary
            ),
        }
        return json.dumps(result, indent=2, default=str)

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
        "Starting documentation MCP server on %s:%d (poll_interval=%ds, data_dir=%s)",
        _config.server_host,
        _config.server_port,
        _config.poll_interval_seconds,
        _config.data_dir,
        extra={"event": "startup"},
    )
    logger.info(
        "Configured %d source(s): %s",
        len(_config.sources),
        [s.name for s in _config.sources],
        extra={"event": "startup"},
    )

    _ingester.start()

    try:
        mcp.run(transport="streamable-http")
    finally:
        _ingester.stop()
        _kb.close()
