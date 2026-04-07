"""MCP server exposing documentation search and query tools via FastMCP."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import anthropic
from anthropic.types import MessageParam, TextBlock
from mcp.server.fastmcp import FastMCP
from starlette.responses import FileResponse, JSONResponse

from docserver.config import Config, load_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from starlette.requests import Request
from docserver.ingestion import Ingester
from docserver.knowledge_base import KnowledgeBase, SourceStatus, SourceSummary
from docserver.logging_config import setup_logging

# Scalar type alias matching knowledge_base.Scalar
Scalar = str | int | float | bool | None

logger = logging.getLogger(__name__)


CHAT_MODEL = "claude-sonnet-4-20250514"
CHAT_MAX_TOKENS = 2048

# ---- Chat prompt building (pure functions, testable) -------------------------

CHAT_SYSTEM_INSTRUCTIONS = (
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


def _format_doc(d: dict[str, Scalar]) -> str:
    """Format a single document as a compact metadata line."""
    title = d.get("title") or d.get("file_path", "?")
    parts: list[str] = [str(title)]
    if d.get("file_path") and d.get("title"):
        parts.append(f"path={d['file_path']}")
    if d.get("created_at"):
        parts.append(f"created={d['created_at']}")
    if d.get("modified_at"):
        parts.append(f"modified={d['modified_at']}")
    if d.get("size_bytes"):
        parts.append(f"size={d['size_bytes']}b")
    return " | ".join(parts)


# Type alias for the nested tree structure returned by KnowledgeBase.get_document_tree()
_TreeNode = dict[str, "Scalar | list[dict[str, Scalar]]"]


def build_inventory_context(
    doc_tree: list[_TreeNode],
    source_stats: Mapping[str, SourceSummary],
) -> str:
    """Build the document inventory section of the system prompt.

    Args:
        doc_tree: Output of kb.get_document_tree().
        source_stats: Dict keyed by source name from kb.get_sources_summary().

    Returns:
        Formatted inventory string with per-source stats and document lists.
    """
    inventory_lines: list[str] = []
    total_files = 0
    total_chunks = 0
    for src in doc_tree:
        src_name = str(src["source"])
        stats = source_stats.get(src_name)
        file_count = stats["file_count"] if stats else 0
        chunk_count = stats["chunk_count"] if stats else 0
        last_indexed = stats["last_indexed"] if stats else "never"
        total_files += file_count
        total_chunks += chunk_count

        inventory_lines.append(
            f"**{src_name}** ({file_count} files, {chunk_count} chunks, last indexed: {last_indexed}):"
        )

        for category, key in [
            ("Root docs", "root_docs"),
            ("Documentation", "docs"),
            ("Journal", "journal"),
            ("Engineering team", "engineering_team"),
            ("Skills", "skills"),
            ("Runbooks", "runbooks"),
        ]:
            raw_docs = src.get(key, [])
            docs = raw_docs if isinstance(raw_docs, list) else []
            if docs:
                inventory_lines.append(f"  {category} ({len(docs)}):")
                for d in docs:
                    inventory_lines.append(f"    - {_format_doc(d)}")

    header = (
        f"Documentation inventory: {len(doc_tree)} sources, "
        f"{total_files} files, {total_chunks} vector chunks.\n\n"
    )
    return header + "\n".join(inventory_lines)


def build_system_prompt(context_parts: list[str]) -> str:
    """Combine system instructions with context parts into the full system prompt."""
    prompt = CHAT_SYSTEM_INSTRUCTIONS
    if context_parts:
        prompt += "\n\n" + "\n\n---\n\n".join(context_parts)
    return prompt


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


def _cors_json(data: object, status_code: int = 200) -> JSONResponse:
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
    source_names = [s.name for s in config.sources]
    source_list = ", ".join(source_names) if source_names else "none configured yet"

    server = FastMCP(
        "Documentation Server",
        instructions=(
            "Personal documentation server for John's home server infrastructure "
            "and software projects. Contains indexed markdown documentation, dev "
            "journals, learning notes, and engineering team reports from multiple "
            "git repositories.\n\n"
            "Use these tools when you need information about:\n"
            "- Home server setup (Proxmox, Docker, VMs, networking, storage)\n"
            "- Monitoring and observability (Prometheus, Grafana, Loki, exporters)\n"
            "- Self-hosted services and their configuration\n"
            "- Project architecture, design decisions, and development history\n"
            "- Past debugging sessions, incident notes, and operational runbooks\n\n"
            f"Currently indexed sources: {source_list}\n\n"
            "Start with search_docs for natural language questions, or list_sources "
            "to see what documentation is available."
        ),
        host=config.server_host,
        port=config.server_port,
    )

    # ---- Health endpoint ------------------------------------------------

    def _compute_source_status(
        status_row: SourceStatus | None,
        poll_interval: int,
    ) -> str:
        """Derive a per-source health label from its status record.

        Returns one of: ``"healthy"``, ``"warning"``, ``"error"``, ``"unknown"``.
        """
        if status_row is None:
            return "unknown"

        # Any consecutive failures -> escalate based on count.
        failures = status_row["consecutive_failures"]
        if failures >= 2:
            return "error"
        if failures == 1:
            return "warning"

        last_checked = status_row["last_checked"]
        if last_checked is None:
            return "unknown"

        # Check staleness relative to poll interval.
        from datetime import UTC, datetime

        try:
            checked_dt = datetime.fromisoformat(last_checked)
            age_seconds = (datetime.now(UTC) - checked_dt).total_seconds()
        except (ValueError, TypeError):
            return "unknown"

        if age_seconds > poll_interval * 5:
            return "error"
        if age_seconds > poll_interval * 2:
            return "warning"
        return "healthy"

    @server.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        try:
            kb = _get_kb()
            ingester = _get_ingester()
            summary = kb.get_sources_summary()
            last_check_times = ingester.get_last_check_times()
            source_statuses = kb.get_source_statuses()
            poll_interval = config.poll_interval_seconds

            sources_out: list[dict[str, str | int | None]] = []
            per_source_labels: list[str] = []

            for src in summary:
                name = src["source"]
                st = source_statuses.get(name)
                label = _compute_source_status(st, poll_interval)
                per_source_labels.append(label)

                sources_out.append(
                    {
                        "source": name,
                        "source_status": label,
                        "file_count": src["file_count"],
                        "chunk_count": src["chunk_count"],
                        "last_indexed": src["last_indexed"],
                        "last_checked": last_check_times.get(name),
                        "last_error": st["last_error"] if st else None,
                        "last_error_at": st["last_error_at"] if st else None,
                        "consecutive_failures": st["consecutive_failures"] if st else 0,
                    }
                )

            # Overall status: error if ALL sources are error/unknown,
            # degraded if ANY source is warning/error, else healthy.
            if not per_source_labels:
                overall = "healthy"
            elif all(s in ("error", "unknown") for s in per_source_labels):
                overall = "error"
            elif any(s in ("warning", "error") for s in per_source_labels):
                overall = "degraded"
            else:
                overall = "healthy"

            return JSONResponse(
                {
                    "status": overall,
                    "total_sources": len(summary),
                    "total_chunks": sum(s.get("chunk_count", 0) for s in summary),
                    "poll_interval_seconds": poll_interval,
                    "sources": sources_out,
                }
            )
        except Exception:
            logger.exception("Health check failed.")
            return JSONResponse({"status": "error"}, status_code=503)

    # ---- Rescan endpoint ------------------------------------------------

    _rescan_lock = threading.Lock()
    _rescan_running = False

    @server.custom_route("/rescan", methods=["POST"])
    async def rescan(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
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
    async def api_tree(_request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Return document tree: sources → categories → documents."""
        try:
            kb = _get_kb()
            tree = kb.get_document_tree()
            return _cors_json(tree)
        except Exception:
            logger.exception("API tree failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/documents/{doc_id:path}", methods=["GET"])
    async def api_get_document(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Return a single document by ID (URL-encoded)."""
        try:
            kb = _get_kb()
            raw_doc_id: str = str(request.path_params["doc_id"])  # pyright: ignore[reportAny]
            doc_id = unquote(raw_doc_id)
            doc = kb.get_document(doc_id)
            if doc is None:
                return _cors_json({"error": "Not found"}, 404)
            return _cors_json(doc)
        except Exception:
            logger.exception("API get_document failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/files/{doc_id:path}", methods=["GET"])
    async def api_get_file(request: Request) -> JSONResponse | FileResponse:  # pyright: ignore[reportUnusedFunction]
        """Serve a raw file from disk by doc_id (e.g. for PDFs)."""
        try:
            kb = _get_kb()
            raw_file_doc_id: str = str(request.path_params["doc_id"])  # pyright: ignore[reportAny]
            doc_id = unquote(raw_file_doc_id)
            doc = kb.get_document(doc_id)
            if doc is None:
                return _cors_json({"error": "Not found"}, 404)

            source_name = str(doc["source"])
            file_path = str(doc["file_path"])

            # Find the matching source config to resolve the repo root.
            repo_source = None
            for src in config.sources:
                if src.name == source_name:
                    repo_source = src
                    break
            if repo_source is None:
                return _cors_json({"error": "Source not found"}, 404)

            # Resolve the absolute path on disk.
            if repo_source.is_remote:
                repo_root = Path(config.data_dir) / "clones" / source_name
            else:
                repo_root = Path(repo_source.path)

            absolute_path = repo_root / file_path

            # Security: ensure the resolved path is within the repo root.
            try:
                _ = absolute_path.resolve().relative_to(repo_root.resolve())
            except ValueError:
                return _cors_json({"error": "Invalid path"}, 400)

            if not absolute_path.is_file():
                return _cors_json({"error": "File not found on disk"}, 404)

            media_type = mimetypes.guess_type(str(absolute_path))[0] or "application/octet-stream"
            return FileResponse(
                str(absolute_path),
                media_type=media_type,
                filename=absolute_path.name,
                content_disposition_type="inline",
            )
        except Exception:
            logger.exception("API get_file failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/search", methods=["GET"])
    async def api_search(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
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
    async def api_chat(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Chat endpoint: RAG search + Claude API. Body: {message, doc_id?, history?}."""
        if request.method == "OPTIONS":
            return _cors_json({})

        try:
            kb = _get_kb()
            body: dict[str, Any] = await request.json()  # pyright: ignore[reportExplicitAny, reportAny]
            message: str = body.get("message", "")  # pyright: ignore[reportAny]
            if not message:
                return _cors_json({"error": "Missing 'message'"}, 400)

            current_doc_id: str | None = body.get("doc_id")  # pyright: ignore[reportAny]
            history: list[dict[str, str]] = body.get("history", [])  # pyright: ignore[reportAny]

            # Build context from current page + RAG search
            context_parts: list[str] = []

            if current_doc_id:
                current_doc = kb.get_document(current_doc_id)
                if current_doc:
                    context_parts.append(
                        "The user is currently viewing this document:\n"
                        + f"Title: {current_doc.get('title', 'Untitled')}\n"
                        + f"Source: {current_doc.get('source', '?')}\n"
                        + f"Path: {current_doc.get('file_path', '?')}\n\n"
                        + f"{current_doc.get('content', '')}"
                    )

            # Add document inventory with indexing stats
            doc_tree = kb.get_document_tree()
            source_stats = {s["source"]: s for s in kb.get_sources_summary()}
            context_parts.insert(0, build_inventory_context(doc_tree, source_stats))

            search_results = kb.search(query=message, n_results=8)
            if search_results:
                rag_context = "\n\n---\n\n".join(
                    f"**{r['metadata'].get('title', 'Untitled')}** "
                    + f"(source: {r['metadata'].get('source', '?')}, "
                    + f"file: {r['metadata'].get('file_path', '?')})\n\n"
                    + f"{r['content']}"
                    for r in search_results
                )
                context_parts.append(f"Relevant documentation excerpts:\n\n{rag_context}")

            system_prompt = build_system_prompt(context_parts)

            # Build messages from history + current
            messages: list[MessageParam] = []
            for h in history[-10:]:  # Keep last 10 exchanges
                role = h["role"]
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": h["content"]})
            messages.append({"role": "user", "content": message})

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return _cors_json({"error": "ANTHROPIC_API_KEY not configured on server"}, 503)

            model = os.environ.get("DOCSERVER_CHAT_MODEL", CHAT_MODEL)
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=CHAT_MAX_TOKENS,
                system=system_prompt,
                messages=messages,
            )

            first_block = response.content[0] if response.content else None
            reply = first_block.text if isinstance(first_block, TextBlock) else ""
            return _cors_json({"reply": reply})

        except anthropic.APIError as exc:
            logger.exception("Anthropic API error in chat.")
            return _cors_json({"error": f"AI service error: {exc.message}"}, 502)
        except Exception:
            logger.exception("API chat failed.")
            return _cors_json({"error": "Internal error"}, 500)

    # ---- Tools ----------------------------------------------------------

    @server.tool()
    def search_docs(query: str, num_results: int = 10, source: str = "") -> str:  # pyright: ignore[reportUnusedFunction]
        """Semantic search across all indexed documentation from John's home
        server infrastructure and software projects.

        Use this to find documentation relevant to a natural language question,
        e.g. "how does service X communicate with service Y", "what ports
        are used on the foo VM", or "how is Prometheus configured".

        Covers: project docs, dev journals, learning notes, engineering team
        reports, runbooks, and architecture decision records.

        Args:
            query: Natural language search query.
            num_results: Maximum number of results to return (default 10).
            source: Optional source name to filter results to a specific
                repository. Use list_sources to see available source names.
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

        output_parts: list[str] = []
        for r in results:
            meta = r.get("metadata", {})
            output_parts.append(
                f"--- {meta.get('title', 'Untitled')} ---\n"
                + f"Source: {meta.get('source', '?')} | "
                + f"File: {meta.get('file_path', '?')} | "
                + f"Score: {r.get('score', '?'):.4f}\n\n"
                + f"{r.get('content', '')}\n"
            )

        return "\n".join(output_parts)

    @server.tool()
    def query_docs(  # pyright: ignore[reportUnusedFunction]
        source: str = "",
        file_path_contains: str = "",
        title_contains: str = "",
        created_after: str = "",
        created_before: str = "",
        limit: int = 20,
    ) -> str:
        """Structured query for document metadata across indexed sources.

        Use this to answer questions like "when was documentation about X created",
        "list all docs in source Y", or "what files were added after date Z".

        Args:
            source: Filter by source name. Use list_sources to see available names.
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
    def get_document(doc_id: str) -> str:  # pyright: ignore[reportUnusedFunction]
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
    def list_sources() -> str:  # pyright: ignore[reportUnusedFunction]
        """List all configured documentation sources and their indexing status.

        Returns source names, file counts, chunk counts, and last indexed time.
        """
        kb = _get_kb()
        summary = kb.get_sources_summary()

        if not summary:
            return "No sources have been indexed yet."

        return json.dumps(summary, indent=2, default=str)

    @server.tool()
    def ingestion_status() -> str:  # pyright: ignore[reportUnusedFunction]
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
            "fully_indexed": len(missing) == 0
            and all(s.get("file_count", 0) > 0 for s in summary),
        }
        return json.dumps(result, indent=2, default=str)

    @server.tool()
    def reindex(source: str = "") -> str:  # pyright: ignore[reportUnusedFunction]
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


def _check_port(host: str, port: int) -> None:
    """Raise OSError if *host*:*port* cannot be bound.

    Opens and immediately closes a TCP socket to detect conflicts before
    uvicorn tries — giving us the chance to log a clear error message.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    finally:
        sock.close()


def run_server() -> None:
    """Start the ingestion scheduler and run the MCP server."""
    json_output = os.environ.get("DOCSERVER_LOG_FORMAT", "json") == "json"
    log_level = os.environ.get("DOCSERVER_LOG_LEVEL", "INFO")
    setup_logging(level=log_level, json_output=json_output)

    mcp = init_app()
    assert _config is not None
    assert _ingester is not None
    assert _kb is not None
    cfg = _config
    ingester = _ingester
    kb = _kb

    # Log server configuration
    logger.info(
        "Starting documentation MCP server on %s:%d",
        cfg.server_host,
        cfg.server_port,
        extra={"event": "startup"},
    )
    logger.info(
        "Configured %d source(s): %s",
        len(cfg.sources),
        [s.name for s in cfg.sources],
        extra={"event": "startup"},
    )

    # Log LLM and embedding configuration
    chat_model = os.environ.get("DOCSERVER_CHAT_MODEL", CHAT_MODEL)
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    logger.info(
        "Chat LLM: provider=anthropic, model=%s, api_key_set=%s",
        chat_model,
        has_api_key,
        extra={"event": "startup", "provider": "anthropic", "model": chat_model},
    )
    logger.info(
        "Embedding model: sentence-transformers/all-mpnet-base-v2 (ONNX Runtime, local inference)",
        extra={"event": "startup"},
    )

    # Log non-secret environment variables
    env_vars = {
        "DOCSERVER_DATA_DIR": cfg.data_dir,
        "DOCSERVER_POLL_INTERVAL": str(cfg.poll_interval_seconds),
        "DOCSERVER_HOST": cfg.server_host,
        "DOCSERVER_PORT": str(cfg.server_port),
        "DOCSERVER_LOG_FORMAT": os.environ.get("DOCSERVER_LOG_FORMAT", "json"),
        "DOCSERVER_LOG_LEVEL": os.environ.get("DOCSERVER_LOG_LEVEL", "INFO"),
        "DOCSERVER_MODEL_DIR": os.environ.get("DOCSERVER_MODEL_DIR", "(default)"),
        "DOCSERVER_CHAT_MODEL": chat_model,
    }
    logger.info(
        "Environment: %s",
        env_vars,
        extra={"event": "startup", "env": env_vars},
    )

    # Pre-flight check: verify the port is available before starting background
    # work.  Without this, uvicorn silently fails to bind and the process looks
    # like it exits after indexing finishes (because ingester.stop(wait=True)
    # keeps the process alive until the current job completes).
    try:
        _check_port(cfg.server_host, cfg.server_port)
    except OSError as exc:
        logger.critical(
            "Cannot bind to %s:%d — %s. "
            "Is another instance already running? "
            "(Use DOCSERVER_PORT to choose a different port.)",
            cfg.server_host,
            cfg.server_port,
            exc,
            extra={"event": "port_unavailable"},
        )
        sys.exit(1)

    ingester.start()

    try:
        mcp.run(transport="streamable-http")
    finally:
        ingester.stop()
        kb.close()
