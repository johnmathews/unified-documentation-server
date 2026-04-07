"""MCP server exposing documentation search and query tools via FastMCP."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUseBlock
from mcp.server.fastmcp import FastMCP
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.responses import FileResponse, JSONResponse

from docserver.config import Config, load_config
from docserver.conversations import ConversationStore

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
    "You have access to tools to search, query, and retrieve documents across all "
    "indexed sources.\n\n"
    "Guidelines:\n"
    "- The inventory summary below shows source names and document counts. "
    "Use list_sources for detailed indexing status.\n"
    "- For structural questions ('what journal entries exist', 'list documents'), "
    "use query_docs to list documents by source, path, title, or date range.\n"
    "- For content questions, use search_docs to find relevant chunks, then "
    "get_document to read full documents. Search proactively — don't guess.\n"
    "- You can search across ALL indexed sources. Every source is part of the "
    "same unified documentation server.\n"
    "- Be concise and direct."
)

CHAT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Semantic search across indexed documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Max results (default 3, max 10).",
                },
                "source": {"type": "string", "description": "Source name filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_docs",
        "description": "List documents by source, path, title, or date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source name filter."},
                "file_path_contains": {"type": "string", "description": "Path substring."},
                "title_contains": {"type": "string", "description": "Title substring."},
                "created_after": {"type": "string", "description": "ISO date."},
                "created_before": {"type": "string", "description": "ISO date."},
                "limit": {"type": "integer", "description": "Max results (default 20)."},
            },
        },
    },
    {
        "name": "get_document",
        "description": "Retrieve a document by ID (format: 'source:path').",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document ID."},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "list_sources",
        "description": "List indexed sources with file counts and status.",
        "input_schema": {"type": "object", "properties": {}},
        "cache_control": {"type": "ephemeral"},
    },
]

CHAT_MAX_TOOL_ITERATIONS = 10


# Type alias for the nested tree structure returned by KnowledgeBase.get_document_tree()
_TreeNode = dict[str, "Scalar | list[dict[str, Scalar]]"]

# Category key → display label mapping for inventory context
_CATEGORY_LABELS: list[tuple[str, str]] = [
    ("root_docs", "root"),
    ("docs", "docs"),
    ("journal", "journal"),
    ("engineering_team", "engineering"),
    ("skills", "skills"),
    ("runbooks", "runbooks"),
]


def build_inventory_context(
    doc_tree: list[_TreeNode],
    source_stats: Mapping[str, SourceSummary],
) -> str:
    """Build a compact inventory summary for the system prompt.

    Returns only per-source category counts — no per-document listings.
    The model should use query_docs/search_docs for document details.
    """
    total_files = 0
    total_chunks = 0
    source_lines: list[str] = []

    for src in doc_tree:
        src_name = str(src["source"])
        stats = source_stats.get(src_name)
        file_count = stats["file_count"] if stats else 0
        chunk_count = stats["chunk_count"] if stats else 0
        total_files += file_count
        total_chunks += chunk_count

        # Count documents per category
        categories: list[str] = []
        for key, label in _CATEGORY_LABELS:
            raw = src.get(key, [])
            count = len(raw) if isinstance(raw, list) else 0
            if count:
                categories.append(f"{count} {label}")

        cat_str = ", ".join(categories) if categories else "empty"
        source_lines.append(f"  {src_name}: {file_count} files ({cat_str})")

    header = (
        f"Documentation inventory: {len(doc_tree)} sources, "
        f"{total_files} files, {total_chunks} vector chunks."
    )
    if not source_lines:
        return header
    return header + "\n" + "\n".join(source_lines)


def build_system_prompt(context_parts: list[str]) -> str:
    """Combine system instructions with context parts into the full system prompt."""
    prompt = CHAT_SYSTEM_INSTRUCTIONS
    if context_parts:
        prompt += "\n\n" + "\n\n---\n\n".join(context_parts)
    return prompt


def _safe_int(value: Any, *, default: int, lo: int, hi: int) -> int:  # pyright: ignore[reportExplicitAny]
    """Coerce a model-supplied value to int within [lo, hi], falling back to default."""
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


_SEARCH_CONTENT_MAX = 300  # Max chars per search result chunk
_GET_DOC_MAX = 6000  # Max chars for get_document response
_TOOL_RESULT_COMPACT_THRESHOLD = 200  # Compact old tool results above this size


def _compact_old_tool_results(messages: list[MessageParam]) -> None:
    """Replace tool_result content from all but the latest tool-use round.

    After the model has consumed tool results and produced new tool calls,
    the verbose raw results from earlier rounds are no longer needed.
    Replacing them with short summaries prevents linear token accumulation.
    Mutates the messages list in-place.
    """
    # Find user messages containing tool_result lists (not plain text user messages)
    tool_msg_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
    ]
    # Compact all except the most recent tool_result message
    for idx in tool_msg_indices[:-1]:
        content = messages[idx]["content"]
        for item in content:  # type: ignore[union-attr]
            if isinstance(item, dict) and item.get("type") == "tool_result":
                orig = str(item.get("content", ""))
                if len(orig) > _TOOL_RESULT_COMPACT_THRESHOLD:
                    item["content"] = f"[Prior result: {len(orig)} chars]"


def _execute_chat_tool(kb: KnowledgeBase, tool_name: str, tool_input: dict[str, Any]) -> str:  # pyright: ignore[reportExplicitAny]
    """Execute a chat tool call and return the result as a string."""
    if tool_name == "search_docs":
        query = str(tool_input.get("query", ""))
        num_results = _safe_int(tool_input.get("num_results"), default=3, lo=1, hi=10)
        source_filter = str(tool_input.get("source", "")) or None
        results = kb.search(query=query, n_results=num_results, source_filter=source_filter)
        if not results:
            return "No matching documents found."
        parts: list[str] = []
        for r in results:
            meta = r.get("metadata", {})
            content = str(r.get("content", ""))
            if len(content) > _SEARCH_CONTENT_MAX:
                content = content[:_SEARCH_CONTENT_MAX] + "..."
            parts.append(
                f"[{meta.get('source', '?')}:{meta.get('file_path', '?')}] "
                f"{meta.get('title', 'Untitled')} "
                f"(score:{r.get('score', 0):.2f})\n{content}"
            )
        return "\n\n".join(parts)

    if tool_name == "query_docs":
        docs = kb.query_documents(
            source=str(tool_input.get("source", "")) or None,
            file_path_contains=str(tool_input.get("file_path_contains", "")) or None,
            title_contains=str(tool_input.get("title_contains", "")) or None,
            created_after=str(tool_input.get("created_after", "")) or None,
            created_before=str(tool_input.get("created_before", "")) or None,
            limit=_safe_int(tool_input.get("limit"), default=20, lo=1, hi=20),
        )
        if not docs:
            return "No matching documents found."
        # Return only key fields to save tokens
        compact = [
            {
                "doc_id": d.get("doc_id"),
                "title": d.get("title"),
                "source": d.get("source"),
                "file_path": d.get("file_path"),
            }
            for d in docs
        ]
        return json.dumps(compact, default=str)

    if tool_name == "get_document":
        doc_id = str(tool_input.get("doc_id", ""))
        doc = kb.get_document(doc_id)
        if doc is None:
            return f"Document '{doc_id}' not found."
        result = json.dumps(doc, indent=2, default=str)
        if len(result) > _GET_DOC_MAX:
            result = result[:_GET_DOC_MAX] + "\n\n... [truncated — use search_docs for specific sections]"
        return result

    if tool_name == "list_sources":
        summary = kb.get_sources_summary()
        if not summary:
            return "No sources have been indexed yet."
        return json.dumps(summary, default=str)

    return f"Unknown tool: {tool_name}"


def _log_token_usage(response: Any, *, iteration: int) -> None:  # pyright: ignore[reportExplicitAny]
    """Log token usage from an Anthropic API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    logger.info(
        "Chat API tokens: input=%d output=%d cache_read=%d cache_create=%d (iteration %d)",
        getattr(usage, "input_tokens", 0),
        getattr(usage, "output_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        getattr(usage, "cache_creation_input_tokens", 0),
        iteration,
        extra={"event": "chat_token_usage", "iteration": iteration},
    )


# Module-level references, initialized by init_app() or run_server().
_kb: KnowledgeBase | None = None
_ingester: Ingester | None = None
_config: Config | None = None
_conversations: ConversationStore | None = None


def _get_kb() -> KnowledgeBase:
    assert _kb is not None, "Server not initialized — call init_app() first"
    return _kb


def _get_ingester() -> Ingester:
    assert _ingester is not None, "Server not initialized — call init_app() first"
    return _ingester


def _get_conversations() -> ConversationStore:
    assert _conversations is not None, "Server not initialized — call init_app() first"
    return _conversations


def _cors_json(data: object, status_code: int = 200) -> JSONResponse:
    """Return a JSONResponse with CORS headers for the UI."""
    return JSONResponse(data, status_code=status_code, headers=_CORS_HEADERS)


_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


@dataclass
class _ChatRequest:
    """Parsed and validated chat request shared by /api/chat and /api/chat/stream."""

    system_blocks: list[dict[str, Any]]  # pyright: ignore[reportExplicitAny]
    messages: list[MessageParam]
    conversation_id: str | None
    page_context: dict[str, str] | None
    history: list[dict[str, str]]
    user_message: str
    model: str
    client: anthropic.Anthropic


async def _prepare_chat_request(request: Request) -> _ChatRequest | JSONResponse:
    """Parse and validate a chat request body, build system blocks and messages.

    Returns a _ChatRequest on success, or a JSONResponse error to return directly.
    """
    kb = _get_kb()
    body: dict[str, Any] = await request.json()  # pyright: ignore[reportExplicitAny, reportAny]
    message: str = body.get("message", "")  # pyright: ignore[reportAny]
    if not message:
        return _cors_json({"error": "Missing 'message'"}, 400)

    current_doc_id: str | None = body.get("doc_id")  # pyright: ignore[reportAny]
    page_context: dict[str, str] | None = body.get("page_context")  # pyright: ignore[reportAny]
    history: list[dict[str, str]] = body.get("history", [])  # pyright: ignore[reportAny]
    conversation_id: str | None = body.get("conversation_id")  # pyright: ignore[reportAny]

    # Build system prompt as content blocks for caching
    system_blocks: list[dict[str, Any]] = [  # pyright: ignore[reportExplicitAny]
        {
            "type": "text",
            "text": CHAT_SYSTEM_INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    doc_tree = kb.get_document_tree()
    source_stats = {s["source"]: s for s in kb.get_sources_summary()}
    inventory = build_inventory_context(doc_tree, source_stats)

    page_hint = ""
    if current_doc_id:
        page_hint = (
            f"The user is viewing document '{current_doc_id}'. "
            "Use get_document to read it if relevant."
        )
    elif page_context:
        ctx_source = page_context.get("source", "")
        ctx_category = page_context.get("category", "")
        if ctx_source and ctx_category:
            page_hint = f"The user is browsing '{ctx_category}' in '{ctx_source}'."
        elif ctx_source:
            page_hint = f"The user is browsing source '{ctx_source}'."

    dynamic_parts = [inventory]
    if page_hint:
        dynamic_parts.append(page_hint)
    system_blocks.append({"type": "text", "text": "\n\n".join(dynamic_parts)})

    messages: list[MessageParam] = []
    for h in history[-10:]:
        role = h["role"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": h["content"]})
    messages.append({"role": "user", "content": message})

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _cors_json({"error": "ANTHROPIC_API_KEY not configured on server"}, 503)

    model = os.environ.get("DOCSERVER_CHAT_MODEL", CHAT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    return _ChatRequest(
        system_blocks=system_blocks,
        messages=messages,
        conversation_id=conversation_id,
        page_context=page_context,
        history=history,
        user_message=message,
        model=model,
        client=client,
    )


def _tool_result_summary(tool_name: str, result_text: str) -> str:
    """Return a concise human-readable summary of a tool result."""
    if "No matching documents found" in result_text or "not found" in result_text:
        return "No results"
    if tool_name == "search_docs":
        count = result_text.count("\n\n") + 1 if result_text.strip() else 0
        return f"{count} result{'s' if count != 1 else ''} found"
    if tool_name == "query_docs":
        try:
            docs = json.loads(result_text)
            return f"Found {len(docs)} document{'s' if len(docs) != 1 else ''}"
        except (json.JSONDecodeError, TypeError):
            return "Results returned"
    if tool_name == "get_document":
        return f"Document retrieved ({len(result_text):,} chars)"
    if tool_name == "list_sources":
        try:
            sources = json.loads(result_text)
            return f"Listed {len(sources)} source{'s' if len(sources) != 1 else ''}"
        except (json.JSONDecodeError, TypeError):
            return "Sources listed"
    return f"{len(result_text):,} chars"


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
        """Agentic chat endpoint with tool use and prompt caching.

        Body: {message, doc_id?, page_context?, history?, conversation_id?}
        """
        if request.method == "OPTIONS":
            return _cors_json({})

        try:
            req = await _prepare_chat_request(request)
            if isinstance(req, JSONResponse):
                return req

            kb = _get_kb()
            response = req.client.messages.create(
                model=req.model,
                max_tokens=CHAT_MAX_TOKENS,
                system=req.system_blocks,  # type: ignore[arg-type]
                messages=req.messages,
                tools=CHAT_TOOLS,  # type: ignore[arg-type]
            )
            _log_token_usage(response, iteration=0)

            # Agentic loop: keep going while the model wants to use tools
            iterations = 0
            while response.stop_reason == "tool_use" and iterations < CHAT_MAX_TOOL_ITERATIONS:
                iterations += 1
                req.messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

                tool_results: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
                for block in response.content:
                    if isinstance(block, ToolUseBlock):
                        tool_input = block.input if isinstance(block.input, dict) else {}
                        result_text = _execute_chat_tool(kb, block.name, tool_input)
                        logger.info(
                            "Chat tool call: %s(%r) -> %d chars",
                            block.name, tool_input, len(result_text),
                            extra={"event": "chat_tool_call", "tool": block.name},
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                if not tool_results:
                    break

                req.messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]
                _compact_old_tool_results(req.messages)

                response = req.client.messages.create(
                    model=req.model,
                    max_tokens=CHAT_MAX_TOKENS,
                    system=req.system_blocks,  # type: ignore[arg-type]
                    messages=req.messages,
                    tools=CHAT_TOOLS,  # type: ignore[arg-type]
                )
                _log_token_usage(response, iteration=iterations)

            # Extract final text response
            reply = "\n".join(
                block.text for block in response.content if isinstance(block, TextBlock)
            )

            # Persist conversation
            conversations = _get_conversations()
            chat_messages = [
                h for h in req.history if h.get("role") in ("user", "assistant")
            ]
            chat_messages.append({"role": "user", "content": req.user_message})
            chat_messages.append({"role": "assistant", "content": reply})

            if req.conversation_id:
                conversations.update(req.conversation_id, chat_messages, req.page_context)
            else:
                req.conversation_id = conversations.create(chat_messages, req.page_context)

            return _cors_json({"reply": reply, "conversation_id": req.conversation_id})

        except anthropic.RateLimitError:
            logger.warning("Anthropic rate limit hit in chat.", extra={"event": "chat_rate_limit"})
            return _cors_json({"error": "Rate limit reached — please wait a moment and try again."}, 429)
        except anthropic.APIError as exc:
            logger.exception("Anthropic API error in chat.")
            return _cors_json({"error": f"AI service error: {exc.message}"}, 502)
        except Exception:
            logger.exception("API chat failed.")
            return _cors_json({"error": "Internal error"}, 500)

    # ---- Streaming Chat (SSE) --------------------------------------------

    @server.custom_route("/api/chat/stream", methods=["POST", "OPTIONS"])
    async def api_chat_stream(request: Request) -> JSONResponse | EventSourceResponse:  # pyright: ignore[reportUnusedFunction]
        """Streaming chat endpoint using Server-Sent Events.

        Sends tool-use progress events during the agentic loop, then the
        final reply. Same request body as /api/chat.
        """
        if request.method == "OPTIONS":
            return _cors_json({})

        req = await _prepare_chat_request(request)
        if isinstance(req, JSONResponse):
            return req

        kb = _get_kb()

        async def event_generator():  # noqa: C901
            call_index = 0
            try:
                yield ServerSentEvent(
                    data=json.dumps({"status": "thinking"}),
                    event="status",
                )

                response = await asyncio.to_thread(
                    req.client.messages.create,
                    model=req.model,
                    max_tokens=CHAT_MAX_TOKENS,
                    system=req.system_blocks,  # type: ignore[arg-type]
                    messages=req.messages,
                    tools=CHAT_TOOLS,  # type: ignore[arg-type]
                )
                _log_token_usage(response, iteration=0)

                iterations = 0
                while response.stop_reason == "tool_use" and iterations < CHAT_MAX_TOOL_ITERATIONS:
                    iterations += 1
                    req.messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

                    tool_results: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
                    for block in response.content:
                        if isinstance(block, ToolUseBlock):
                            tool_input = block.input if isinstance(block.input, dict) else {}

                            yield ServerSentEvent(
                                data=json.dumps({"index": call_index, "tool": block.name, "input": tool_input}),
                                event="tool_call",
                            )

                            result_text = await asyncio.to_thread(
                                _execute_chat_tool, kb, block.name, tool_input,
                            )
                            logger.info(
                                "Chat tool call: %s(%r) -> %d chars",
                                block.name, tool_input, len(result_text),
                                extra={"event": "chat_tool_call", "tool": block.name},
                            )
                            summary = _tool_result_summary(block.name, result_text)

                            yield ServerSentEvent(
                                data=json.dumps({"index": call_index, "tool": block.name, "summary": summary}),
                                event="tool_result",
                            )
                            call_index += 1

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            })

                    if not tool_results:
                        break

                    req.messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]
                    _compact_old_tool_results(req.messages)

                    yield ServerSentEvent(
                        data=json.dumps({"status": "thinking", "iteration": iterations}),
                        event="status",
                    )

                    response = await asyncio.to_thread(
                        req.client.messages.create,
                        model=req.model,
                        max_tokens=CHAT_MAX_TOKENS,
                        system=req.system_blocks,  # type: ignore[arg-type]
                        messages=req.messages,
                        tools=CHAT_TOOLS,  # type: ignore[arg-type]
                    )
                    _log_token_usage(response, iteration=iterations)

                # Extract final reply
                reply = "\n".join(
                    block.text for block in response.content if isinstance(block, TextBlock)
                )

                # Persist conversation
                conversations = _get_conversations()
                chat_messages = [
                    h for h in req.history if h.get("role") in ("user", "assistant")
                ]
                chat_messages.append({"role": "user", "content": req.user_message})
                chat_messages.append({"role": "assistant", "content": reply})

                if req.conversation_id:
                    conversations.update(req.conversation_id, chat_messages, req.page_context)
                else:
                    req.conversation_id = conversations.create(chat_messages, req.page_context)

                yield ServerSentEvent(
                    data=json.dumps({"reply": reply, "conversation_id": req.conversation_id}),
                    event="reply",
                )

            except anthropic.RateLimitError:
                logger.warning("Anthropic rate limit hit in chat stream.", extra={"event": "chat_rate_limit"})
                yield ServerSentEvent(
                    data=json.dumps({"error": "Rate limit reached — please wait a moment and try again."}),
                    event="error",
                )
            except anthropic.APIError as exc:
                logger.exception("Anthropic API error in chat stream.")
                yield ServerSentEvent(
                    data=json.dumps({"error": f"AI service error: {exc.message}"}),
                    event="error",
                )
            except Exception:
                logger.exception("Chat stream failed.")
                yield ServerSentEvent(
                    data=json.dumps({"error": "Internal error"}),
                    event="error",
                )

        return EventSourceResponse(
            content=event_generator(),
            headers=_CORS_HEADERS,
            ping=15,
        )

    # ---- Conversation API ------------------------------------------------

    @server.custom_route("/api/conversations", methods=["GET"])
    async def api_list_conversations(_request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """List saved conversations, most recent first."""
        try:
            conversations = _get_conversations()
            return _cors_json(conversations.list_all())
        except Exception:
            logger.exception("API list conversations failed.")
            return _cors_json({"error": "Internal error"}, 500)

    @server.custom_route("/api/conversations/{conv_id}", methods=["GET", "DELETE", "OPTIONS"])
    async def api_conversation(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Get or delete a conversation."""
        if request.method == "OPTIONS":
            return _cors_json({})

        try:
            conversations = _get_conversations()
            raw_conv_id: str = str(request.path_params["conv_id"])  # pyright: ignore[reportAny]
            conv_id = unquote(raw_conv_id)

            if request.method == "DELETE":
                if conversations.delete(conv_id):
                    return _cors_json({"deleted": True})
                return _cors_json({"error": "Not found"}, 404)

            conv = conversations.get(conv_id)
            if conv is None:
                return _cors_json({"error": "Not found"}, 404)
            return _cors_json(conv)
        except Exception:
            logger.exception("API conversation failed.")
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
    """Initialize the application: KB, ingester, conversation store, and MCP server.

    Useful for testing — pass a custom Config to avoid touching real state.
    """
    global _kb, _ingester, _config, _conversations

    if config is None:
        config = load_config()
    _config = config

    _kb = KnowledgeBase(config.data_dir)
    _ingester = Ingester(config, _kb)
    _conversations = ConversationStore(config.data_dir)

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
