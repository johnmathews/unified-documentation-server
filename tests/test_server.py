"""Tests for the MCP server tools."""

import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import docserver.server as server_module
from docserver.config import Config, RepoSource
from docserver.server import _check_port


@pytest.fixture
def app(tmp_path):
    """Initialize the server with a temp data dir and seed some test data."""
    config = Config(
        sources=[],
        data_dir=str(tmp_path / "data"),
        poll_interval_seconds=9999,
    )
    mcp = server_module.init_app(config)

    kb = server_module._get_kb()

    # Seed a parent doc
    kb.upsert_document(
        "docs:setup.md",
        "",
        {
            "source": "docs",
            "file_path": "setup.md",
            "title": "Setup Guide",
            "created_at": "2025-06-01T00:00:00+00:00",
            "modified_at": "2025-07-15T00:00:00+00:00",
            "size_bytes": 1234,
            "is_chunk": False,
            "total_chunks": 2,
        },
    )

    # Seed chunks
    kb.upsert_document(
        "docs:setup.md#chunk0",
        "The web server listens on port 8080. Nginx proxies HTTPS on port 443.",
        {
            "source": "docs",
            "file_path": "setup.md",
            "title": "Setup Guide",
            "chunk_index": 0,
            "total_chunks": 2,
            "is_chunk": True,
            "section_path": "Setup > Ports",
        },
    )
    kb.upsert_document(
        "docs:setup.md#chunk1",
        "SSH access is available on port 22. Only key-based auth is allowed.",
        {
            "source": "docs",
            "file_path": "setup.md",
            "title": "Setup Guide",
            "chunk_index": 1,
            "total_chunks": 2,
            "is_chunk": True,
            "section_path": "Setup > SSH",
        },
    )

    yield mcp

    kb.close()


def _call_tool(mcp, name: str, **kwargs) -> str:
    """Call a tool function by name from the MCP server."""
    tool = mcp._tool_manager.get_tool(name)
    assert tool is not None, f"Tool '{name}' not found"
    return tool.fn(**kwargs)


class TestSearchDocs:
    def test_finds_relevant_content(self, app):
        result = _call_tool(app, "search_docs", query="what port does the web server use")
        assert "8080" in result

    def test_no_results(self, app):
        result = _call_tool(app, "search_docs", query="quantum computing algorithms")
        # May still return results due to vector similarity, but at minimum shouldn't crash
        assert isinstance(result, str)

    def test_source_filter(self, app):
        result = _call_tool(app, "search_docs", query="port", source="docs")
        assert "8080" in result or "443" in result or "22" in result

    def test_source_filter_nonexistent(self, app):
        result = _call_tool(app, "search_docs", query="port", source="nonexistent")
        assert "No matching" in result


class TestQueryDocs:
    def test_query_by_source(self, app):
        result = _call_tool(app, "query_docs", source="docs")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["doc_id"] == "docs:setup.md"

    def test_query_by_title(self, app):
        result = _call_tool(app, "query_docs", title_contains="Setup")
        parsed = json.loads(result)
        assert len(parsed) >= 1
        assert parsed[0]["title"] == "Setup Guide"

    def test_query_no_results(self, app):
        result = _call_tool(app, "query_docs", title_contains="nonexistent")
        assert "No matching" in result

    def test_query_by_date(self, app):
        result = _call_tool(app, "query_docs", created_after="2025-05-01")
        parsed = json.loads(result)
        assert len(parsed) >= 1


class TestGetDocument:
    def test_get_parent(self, app):
        result = _call_tool(app, "get_document", doc_id="docs:setup.md")
        parsed = json.loads(result)
        assert parsed["title"] == "Setup Guide"

    def test_get_chunk(self, app):
        result = _call_tool(app, "get_document", doc_id="docs:setup.md#chunk0")
        parsed = json.loads(result)
        assert "8080" in parsed["content"]

    def test_not_found(self, app):
        result = _call_tool(app, "get_document", doc_id="nope:nope.md")
        assert "not found" in result.lower()


class TestListSources:
    def test_lists_sources(self, app):
        result = _call_tool(app, "list_sources")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["source"] == "docs"
        assert parsed[0]["file_count"] == 1
        assert parsed[0]["chunk_count"] == 2


class TestReindex:
    """The reindex MCP tool dispatches to the ingestion supervisor (which
    spawns a subprocess in production) and formats its result. We mock the
    supervisor at the boundary because integration of the worker subprocess
    against a tmp test config is covered by the supervisor's own tests.
    """

    def test_reindex_empty(self, app):
        sup = server_module._get_supervisor()
        with patch.object(sup, "run_subprocess_cycle", return_value={}) as mock_run:
            result = _call_tool(app, "reindex")
        mock_run.assert_called_once_with(sources=None)
        parsed = json.loads(result)
        assert parsed == {}

    def test_reindex_unknown_source(self, app):
        sup = server_module._get_supervisor()
        with patch.object(
            sup, "run_subprocess_cycle", return_value={"docs": {"upserted": 1}}
        ):
            result = _call_tool(app, "reindex", source="nonexistent")
        assert "not found" in result.lower()

    def test_reindex_single_source(self, app):
        """reindex(source='x') should pass [x] as the sources filter to the
        supervisor and round-trip the supervisor's per-source stats back to
        the caller."""
        sup = server_module._get_supervisor()
        with patch.object(
            sup,
            "run_subprocess_cycle",
            return_value={"x": {"upserted": 3}},
        ) as mock_run:
            result = _call_tool(app, "reindex", source="x")
        mock_run.assert_called_once_with(sources=["x"])
        parsed = json.loads(result)
        assert parsed == {"x": {"upserted": 3}}

    def test_reindex_already_running(self, app):
        from docserver.ingestion_supervisor import IngestionAlreadyRunning

        sup = server_module._get_supervisor()
        with patch.object(
            sup, "run_subprocess_cycle", side_effect=IngestionAlreadyRunning("busy")
        ):
            result = _call_tool(app, "reindex")
        parsed = json.loads(result)
        assert parsed == {"status": "already_running"}

    def test_reindex_failed(self, app):
        sup = server_module._get_supervisor()
        with patch.object(sup, "run_subprocess_cycle", return_value=None):
            sup._last_failure = {  # set what the supervisor would have stored
                "completed_at": "2026-05-07T00:00:00+00:00",
                "exit_code": "1",
                "reason": "worker exited non-zero",
            }
            result = _call_tool(app, "reindex")
        parsed = json.loads(result)
        assert parsed["status"] == "failed"
        assert parsed["exit_code"] == "1"


class TestHealthEndpoint:
    def test_health_returns_ok(self, app) -> None:
        """GET /health should return JSON with overall and per-source status."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("healthy", "degraded", "error")
        assert "total_sources" in body
        assert "total_chunks" in body
        assert "poll_interval_seconds" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)
        assert "last_ingestion" in body
        # Initially None until the ingester completes its first cycle.
        assert body["last_ingestion"] is None or isinstance(body["last_ingestion"], dict)
        assert "chat_model_valid" in body
        assert isinstance(body["chat_model_valid"], bool)
        assert "chat_model_error" in body
        # Verify per-source structure if sources exist
        if body["sources"]:
            src = body["sources"][0]
            assert "source" in src
            assert "source_status" in src
            assert src["source_status"] in ("healthy", "warning", "error", "unknown")
            assert "file_count" in src
            assert "chunk_count" in src
            assert "last_indexed" in src
            assert "last_error" in src
            assert "consecutive_failures" in src

    def test_health_last_ingestion_after_cycle(self, app) -> None:
        """/health should expose the most recent ingestion cycle metrics
        from the supervisor."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        supervisor = server_module._get_supervisor()
        supervisor._last_ingestion = {
            "completed_at": "2026-04-29T12:00:00+00:00",
            "duration_s": 4.2,
            "rss_at_start_mb": 180.0,
            "rss_at_end_mb": 240.0,
            "rss_growth_mb": 60.0,
            "flush_count": 3,
            "flush_total_s": 3.5,
            "flush_max_s": 1.6,
        }

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["last_ingestion"] is not None
        assert body["last_ingestion"]["duration_s"] == 4.2
        assert body["last_ingestion"]["rss_at_end_mb"] == 240.0
        assert body["last_ingestion"]["flush_count"] == 3
        # last_ingestion_failure should be None on success.
        assert body["last_ingestion_failure"] is None

    def test_health_last_ingestion_failure(self, app) -> None:
        """/health surfaces last_ingestion_failure when the supervisor records one."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        supervisor = server_module._get_supervisor()
        supervisor._last_failure = {
            "completed_at": "2026-04-29T12:00:00+00:00",
            "exit_code": "1",
            "reason": "worker exited non-zero",
        }

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["last_ingestion_failure"]["exit_code"] == "1"

    def test_health_exposes_last_stats(self, app) -> None:
        """/health surfaces per-source last_stats so the UI can render a scan summary."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        supervisor = server_module._get_supervisor()
        original = supervisor._last_stats
        supervisor._last_stats = {
            "tech-blog": {
                "upserted": 4,
                "deleted": 1,
                "skipped": 12,
                "new": 3,
                "modified": 1,
                "files": 16,
                "errors": 0,
            }
        }
        try:
            response = client.get("/health")
            assert response.status_code == 200
            body = response.json()
            assert body["last_stats"] is not None
            assert body["last_stats"]["tech-blog"]["new"] == 3
            assert body["last_stats"]["tech-blog"]["modified"] == 1
            assert body["last_stats"]["tech-blog"]["deleted"] == 1
        finally:
            supervisor._last_stats = original

    def test_health_exposes_ingestion_running(self, app) -> None:
        """/health surfaces ingestion_running so the UI can poll for completion."""
        from unittest.mock import MagicMock

        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        supervisor = server_module._get_supervisor()
        original = supervisor._current_proc
        try:
            response = client.get("/health")
            assert response.json()["ingestion_running"] is False

            fake_proc = MagicMock()
            fake_proc.poll.return_value = None
            supervisor._current_proc = fake_proc
            response = client.get("/health")
            assert response.json()["ingestion_running"] is True
        finally:
            supervisor._current_proc = original

    def test_health_exposes_current_progress(self, app) -> None:
        """/health surfaces the supervisor's current_progress so the webapp
        can render live scan progress."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        supervisor = server_module._get_supervisor()
        original = supervisor._current_progress
        try:
            # When idle, current_progress is null.
            response = client.get("/health")
            assert response.json()["current_progress"] is None

            # When a scan is in flight, current_progress mirrors the latest
            # scan_progress event from the worker.
            with supervisor._progress_lock:
                supervisor._current_progress = {
                    "phase": "processing",
                    "current": 7,
                    "total": 42,
                    "source": "demo",
                    "doc": "guide.md",
                }
            response = client.get("/health")
            body = response.json()
            assert body["current_progress"] == {
                "phase": "processing",
                "current": 7,
                "total": 42,
                "source": "demo",
                "doc": "guide.md",
            }
        finally:
            with supervisor._progress_lock:
                supervisor._current_progress = original

    def test_health_reflects_chat_model_invalid(self, app) -> None:
        """/health should expose chat_model_valid=False when probe failed."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        original_valid = server_module._chat_model_valid
        original_error = server_module._chat_model_error
        try:
            server_module._chat_model_valid = False
            server_module._chat_model_error = "model: claude-bogus-1 not found"
            response = client.get("/health")
            assert response.status_code == 200
            body = response.json()
            assert body["chat_model_valid"] is False
            assert "claude-bogus-1" in body["chat_model_error"]
        finally:
            server_module._chat_model_valid = original_valid
            server_module._chat_model_error = original_error

    def test_health_includes_last_checked(self, app) -> None:
        """GET /health should include last_checked from ingester check times."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        # Simulate the ingester having recorded a check time for the "docs" source.
        ingester = server_module._get_ingester()
        ingester._last_check_times["docs"] = "2025-07-15T12:00:00+00:00"

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["sources"]
        src = body["sources"][0]
        assert src["source"] == "docs"
        assert src["last_checked"] == "2025-07-15T12:00:00+00:00"

    def test_health_last_checked_null_when_not_synced(self, app) -> None:
        """last_checked should be null for sources that haven't been synced yet."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        # Ensure no check times are set.
        ingester = server_module._get_ingester()
        ingester._last_check_times.clear()

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["sources"]
        src = body["sources"][0]
        assert "last_checked" in src
        assert src["last_checked"] is None

    def test_health_returns_503_on_error(self, app) -> None:
        """GET /health should return 503 when KB raises."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        with patch.object(
            server_module._get_kb(), "get_sources_summary", side_effect=RuntimeError("db broken")
        ):
            response = client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"

    def test_health_source_status_healthy_after_check(self, app) -> None:
        """A source with a recent successful check should be 'healthy'."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        # Record a successful check for the source.
        kb = server_module._get_kb()
        kb.update_source_check("docs")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        src = next(s for s in body["sources"] if s["source"] == "docs")
        assert src["source_status"] == "healthy"
        assert src["last_error"] is None
        assert src["consecutive_failures"] == 0

    def test_health_source_status_warning_on_one_failure(self, app) -> None:
        """A source with 1 consecutive failure should be 'warning'."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        kb.update_source_check("docs", error="git fetch failed")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        src = next(s for s in body["sources"] if s["source"] == "docs")
        assert src["source_status"] == "warning"
        assert src["last_error"] == "git fetch failed"
        assert src["consecutive_failures"] == 1

    def test_health_source_status_error_on_two_failures(self, app) -> None:
        """A source with 2+ consecutive failures should be 'error'."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        kb.update_source_check("docs", error="fail 1")
        kb.update_source_check("docs", error="fail 2")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        src = next(s for s in body["sources"] if s["source"] == "docs")
        assert src["source_status"] == "error"
        assert src["consecutive_failures"] == 2

    def test_health_overall_degraded(self, app) -> None:
        """Overall status should be 'degraded' when some sources have issues."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        # Mark the source as having one failure.
        kb.update_source_check("docs", error="network error")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        # With a single source that has a warning, overall should be degraded.
        assert body["status"] == "degraded"

    def test_health_source_recovers_after_success(self, app) -> None:
        """A source should recover to 'healthy' after a successful check."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        # First, create a failure.
        kb.update_source_check("docs", error="fail")
        kb.update_source_check("docs", error="fail again")
        # Then, a successful check should reset.
        kb.update_source_check("docs")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        src = next(s for s in body["sources"] if s["source"] == "docs")
        assert src["source_status"] == "healthy"
        assert src["last_error"] is None
        assert src["consecutive_failures"] == 0

    def test_health_includes_chroma_alive_true(self, app) -> None:
        """/health should ping Chroma on every call and surface the result."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.get("/health")
        body = response.json()
        # Tests use the in-process PersistentClient fallback whose
        # heartbeat() works, so chroma_alive should be True.
        assert body["chroma_alive"] is True
        assert body["chroma_error"] is None

    def test_health_degrades_when_chroma_down(self, app) -> None:
        """A failing Chroma heartbeat must downgrade overall status to
        'degraded' and surface chroma_error."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        with patch.object(
            kb, "ping_chroma", return_value=(False, "ConnectionError: refused")
        ):
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["chroma_alive"] is False
        assert body["chroma_error"] == "ConnectionError: refused"

    def test_health_chroma_ping_runs_off_event_loop(self, app) -> None:
        """The chroma heartbeat must be dispatched via asyncio.to_thread so
        a slow ping doesn't block the request loop. We can't test the
        wall-clock bound directly under TestClient (its per-request loop
        waits for the executor to drain on shutdown), so we instead verify
        the call is dispatched to a worker thread, not the event loop."""
        import threading

        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        kb = server_module._get_kb()
        captured_thread_name: list[str] = []

        def _capture(*_a, **_kw):
            captured_thread_name.append(threading.current_thread().name)
            return True, None

        with patch.object(kb, "ping_chroma", side_effect=_capture):
            response = client.get("/health")
        assert response.status_code == 200
        assert captured_thread_name, "ping_chroma was not invoked"
        # asyncio.to_thread runs in the default ThreadPoolExecutor, whose
        # threads are named 'asyncio_<n>'. The main / event-loop thread is
        # not. The exact name varies by Python version, but it must NOT be
        # the main thread.
        assert captured_thread_name[0] != "MainThread"


class TestRescanEndpoint:
    def test_rescan_starts_background(self, app) -> None:
        """POST /rescan should hand the cycle to the supervisor and return immediately.

        We patch ``run_subprocess_cycle`` so the test does not actually spawn
        a subprocess (which would shell out to ``python -m docserver.ingestion_worker``
        and add ~5s to the test).
        """
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        with patch.object(
            server_module._get_supervisor(),
            "run_subprocess_cycle",
            return_value={"duration_s": 0.1},
        ):
            response = client.post("/rescan")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "started"

    def test_rescan_rejects_concurrent(self, app, caplog) -> None:
        """POST /rescan should return 409 (and log the no-op) when a cycle is in flight."""
        import logging
        from unittest.mock import MagicMock

        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        # Pretend the supervisor already has a running worker — the
        # endpoint's cheap pre-check inspects _current_proc.
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None  # still running
        supervisor = server_module._get_supervisor()
        original = supervisor._current_proc
        supervisor._current_proc = fake_proc
        try:
            with caplog.at_level(logging.INFO, logger="docserver.server"):
                response = client.post("/rescan")
            assert response.status_code == 409
            assert response.json()["status"] == "already_running"
            assert any(
                "ignored" in r.getMessage() and "already running" in r.getMessage()
                for r in caplog.records
            ), "expected an INFO log explaining the no-op"
        finally:
            supervisor._current_proc = original


class TestFilesEndpoint:
    def test_serves_raw_file(self, tmp_path: Path) -> None:
        """GET /api/files/{doc_id} should serve the raw file with correct MIME type."""
        # Create a source directory with a PDF
        source_dir = tmp_path / "my-docs"
        source_dir.mkdir()
        pdf_file = source_dir / "report.pdf"
        pdf_content = b"%PDF-1.4 fake pdf content for testing"
        pdf_file.write_bytes(pdf_content)

        config = Config(
            sources=[RepoSource(name="my-docs", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
            poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()

        # Seed the doc in the KB (as ingestion would)
        kb.upsert_document(
            "my-docs:report.pdf",
            "",
            {
                "source": "my-docs",
                "file_path": "report.pdf",
                "title": "report",
                "is_chunk": False,
                "total_chunks": 0,
            },
        )

        try:
            starlette_app = mcp.streamable_http_app()
            client = TestClient(starlette_app)
            response = client.get("/api/files/my-docs:report.pdf")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == pdf_content
        finally:
            kb.close()

    def test_not_found_doc(self, tmp_path: Path) -> None:
        """GET /api/files/{doc_id} should return 404 for unknown doc_id."""
        config = Config(
            sources=[],
            data_dir=str(tmp_path / "data"),
            poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()

        try:
            starlette_app = mcp.streamable_http_app()
            client = TestClient(starlette_app)
            response = client.get("/api/files/nope:nope.pdf")
            assert response.status_code == 404
        finally:
            kb.close()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """GET /api/files/ should block path traversal attempts."""
        source_dir = tmp_path / "docs"
        source_dir.mkdir()

        config = Config(
            sources=[RepoSource(name="docs", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
            poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()

        # Seed a doc with a traversal path
        kb.upsert_document(
            "docs:../../etc/passwd",
            "",
            {
                "source": "docs",
                "file_path": "../../etc/passwd",
                "title": "evil",
                "is_chunk": False,
            },
        )

        try:
            starlette_app = mcp.streamable_http_app()
            client = TestClient(starlette_app)
            response = client.get("/api/files/docs:../../etc/passwd")
            assert response.status_code in (400, 404)
        finally:
            kb.close()


class TestInventoryCache:
    """The chat-system-prompt inventory is cached with a short TTL because
    it reruns a full SQLite tree scan on every chat call. The cache must
    return on cache hits, refresh on misses, and clear when the supervisor
    reports a successful ingest cycle.
    """

    def setup_method(self) -> None:
        # Each test starts with an empty cache.
        server_module._invalidate_inventory_cache()

    def test_cache_hits_avoid_kb_calls(self, app) -> None:
        kb = server_module._get_kb()
        with patch.object(
            kb, "get_document_tree", wraps=kb.get_document_tree
        ) as mock_tree:
            _ = server_module._get_cached_inventory(kb)
            _ = server_module._get_cached_inventory(kb)
            _ = server_module._get_cached_inventory(kb)
        # First call builds, subsequent calls hit the cache.
        assert mock_tree.call_count == 1

    def test_invalidation_forces_rebuild(self, app) -> None:
        kb = server_module._get_kb()
        with patch.object(
            kb, "get_document_tree", wraps=kb.get_document_tree
        ) as mock_tree:
            _ = server_module._get_cached_inventory(kb)
            server_module._invalidate_inventory_cache()
            _ = server_module._get_cached_inventory(kb)
        assert mock_tree.call_count == 2

    def test_supervisor_invalidates_on_successful_cycle(self, app) -> None:
        """The supervisor's on_cycle_success callback must clear the cache
        so the next chat sees a freshly-rendered inventory."""
        kb = server_module._get_kb()
        sup = server_module._get_supervisor()

        # Prime the cache.
        _ = server_module._get_cached_inventory(kb)
        # Cache is now populated.
        assert server_module._inventory_cache is not None

        # Trigger the success callback the supervisor would fire after a
        # worker cycle that returned exit 0 with a parseable payload.
        assert sup._on_cycle_success is not None
        sup._on_cycle_success()

        assert server_module._inventory_cache is None


class TestInputValidation:
    def test_search_clamps_num_results_low(self, app) -> None:
        """num_results=0 should be clamped to 1 (no error)."""
        result = _call_tool(app, "search_docs", query="port", num_results=0)
        assert isinstance(result, str)

    def test_search_clamps_num_results_high(self, app) -> None:
        """num_results=1000 should be clamped to 100 (no error)."""
        result = _call_tool(app, "search_docs", query="port", num_results=1000)
        assert isinstance(result, str)

    def test_query_clamps_limit_low(self, app) -> None:
        """limit=0 should be clamped to 1."""
        result = _call_tool(app, "query_docs", limit=0)
        # Should not crash; returns either results or "No matching"
        assert isinstance(result, str)

    def test_query_clamps_limit_high(self, app) -> None:
        """limit=9999 should be clamped to 100."""
        result = _call_tool(app, "query_docs", limit=9999)
        assert isinstance(result, str)


class TestCheckPort:
    """Tests for the _check_port pre-flight helper."""

    def test_free_port_succeeds(self) -> None:
        """_check_port should not raise when the port is available."""
        # Bind to port 0 to let the OS pick a free port, then check that port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, free_port = s.getsockname()
        # Port is now released — _check_port should succeed.
        _check_port("127.0.0.1", free_port)

    def test_occupied_port_raises(self) -> None:
        """_check_port should raise OSError when the port is already bound."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            _, occupied_port = s.getsockname()
            with pytest.raises(OSError):
                _check_port("127.0.0.1", occupied_port)


class TestBookmarkEndpoints:
    """Tests for the /api/bookmarks REST endpoints."""

    def test_list_empty(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.get("/api/bookmarks")
        assert response.status_code == 200
        assert response.json() == []

    def test_add_bookmark(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.post("/api/bookmarks", json={"doc_id": "docs:setup.md"})
        assert response.status_code == 201
        body = response.json()
        assert body["doc_id"] == "docs:setup.md"
        assert body["user_id"] == "default"
        assert "created_at" in body

    def test_add_bookmark_missing_doc_id(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.post("/api/bookmarks", json={})
        assert response.status_code == 400

    def test_list_after_add(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        client.post("/api/bookmarks", json={"doc_id": "docs:setup.md"})
        response = client.get("/api/bookmarks")
        assert response.status_code == 200
        bookmarks = response.json()
        assert len(bookmarks) == 1
        bm = bookmarks[0]
        assert bm["doc_id"] == "docs:setup.md"
        assert bm["title"] == "Setup Guide"
        assert bm["source"] == "docs"
        assert bm["file_path"] == "setup.md"

    def test_delete_bookmark(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        client.post("/api/bookmarks", json={"doc_id": "docs:setup.md"})
        response = client.delete("/api/bookmarks/docs:setup.md")
        assert response.status_code == 200
        assert response.json()["deleted"] is True
        # Verify it's gone
        assert client.get("/api/bookmarks").json() == []

    def test_delete_nonexistent(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.delete("/api/bookmarks/nope:nope.md")
        assert response.status_code == 404

    def test_bulk_check(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        client.post("/api/bookmarks", json={"doc_id": "docs:setup.md"})
        response = client.post(
            "/api/bookmarks/check",
            json={"doc_ids": ["docs:setup.md", "docs:other.md"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["docs:setup.md"] is True
        assert result["docs:other.md"] is False

    def test_list_enriched_with_doc_metadata(self, app) -> None:
        """Bookmarks list should include document metadata from KB."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        client.post("/api/bookmarks", json={"doc_id": "docs:setup.md"})
        response = client.get("/api/bookmarks")
        bm = response.json()[0]
        assert bm["title"] == "Setup Guide"
        assert bm["source"] == "docs"
        assert bm["size_bytes"] == 1234

    def test_bookmark_with_custom_user_id(self, app) -> None:
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        client.post("/api/bookmarks", json={"doc_id": "docs:setup.md", "user_id": "alice"})
        # Default user should see nothing
        assert client.get("/api/bookmarks").json() == []
        # Alice should see the bookmark
        response = client.get("/api/bookmarks?user_id=alice")
        assert len(response.json()) == 1


class TestGetBookmarksTool:
    """Tests for the get_bookmarks chat tool."""

    def test_no_bookmarks(self, app) -> None:
        result = _call_tool(app, "get_bookmarks")
        assert "No bookmarked" in result

    def test_with_bookmarks(self, app) -> None:
        bookmarks_store = server_module._get_bookmarks()
        bookmarks_store.add("docs:setup.md")
        result = _call_tool(app, "get_bookmarks")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["doc_id"] == "docs:setup.md"
        assert parsed[0]["title"] == "Setup Guide"


class TestSourcesTreeEndpoints:
    """Tests for /api/sources/tree and /api/sources/{name}/tree."""

    def _make_app(self, tmp_path: Path, source_name: str = "docs"):
        source_dir = tmp_path / source_name
        source_dir.mkdir()
        config = Config(
            sources=[RepoSource(name=source_name, path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
            poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()
        return mcp, kb

    def test_single_source_tree_returns_files(self, tmp_path: Path) -> None:
        mcp, kb = self._make_app(tmp_path)
        try:
            kb.upsert_document(
                "docs:a.md",
                "",
                {"source": "docs", "file_path": "a.md", "title": "A", "is_chunk": False},
            )
            kb.upsert_document(
                "docs:sub/b.md",
                "",
                {
                    "source": "docs",
                    "file_path": "sub/b.md",
                    "title": "B",
                    "is_chunk": False,
                },
            )

            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/docs/tree")
            assert response.status_code == 200
            body = response.json()
            assert body["source"] == "docs"
            paths = [f["file_path"] for f in body["files"]]
            assert paths == ["a.md", "sub/b.md"]
            assert body["files"][0]["doc_id"] == "docs:a.md"
            assert body["files"][0]["title"] == "A"
        finally:
            kb.close()

    def test_single_source_tree_excludes_chunks(self, tmp_path: Path) -> None:
        mcp, kb = self._make_app(tmp_path)
        try:
            kb.upsert_document(
                "docs:a.md",
                "",
                {"source": "docs", "file_path": "a.md", "title": "A", "is_chunk": False},
            )
            kb.upsert_document(
                "docs:a.md#chunk0",
                "content",
                {
                    "source": "docs",
                    "file_path": "a.md",
                    "title": "A",
                    "chunk_index": 0,
                    "is_chunk": True,
                },
            )

            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/docs/tree")
            assert response.status_code == 200
            assert len(response.json()["files"]) == 1
        finally:
            kb.close()

    def test_single_source_tree_unknown_source(self, tmp_path: Path) -> None:
        mcp, kb = self._make_app(tmp_path)
        try:
            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/nonexistent/tree")
            assert response.status_code == 404
        finally:
            kb.close()

    def test_bulk_tree_returns_all_sources(self, tmp_path: Path) -> None:
        # Two configured sources, with docs in each.
        source_a = tmp_path / "a"
        source_a.mkdir()
        source_b = tmp_path / "b"
        source_b.mkdir()
        config = Config(
            sources=[
                RepoSource(name="a", path=str(source_a)),
                RepoSource(name="b", path=str(source_b)),
            ],
            data_dir=str(tmp_path / "data"),
            poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()
        try:
            kb.upsert_document(
                "a:one.md", "", {"source": "a", "file_path": "one.md", "is_chunk": False},
            )
            kb.upsert_document(
                "b:two.md", "", {"source": "b", "file_path": "two.md", "is_chunk": False},
            )

            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/tree")
            assert response.status_code == 200
            body = response.json()
            assert "sources" in body
            names = {s["source"] for s in body["sources"]}
            assert names == {"a", "b"}
            by_name = {s["source"]: s for s in body["sources"]}
            assert [f["file_path"] for f in by_name["a"]["files"]] == ["one.md"]
            assert [f["file_path"] for f in by_name["b"]["files"]] == ["two.md"]
        finally:
            kb.close()

    def test_bulk_tree_empty_when_no_sources(self, tmp_path: Path) -> None:
        config = Config(
            sources=[], data_dir=str(tmp_path / "data"), poll_interval_seconds=9999,
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()
        try:
            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/tree")
            assert response.status_code == 200
            assert response.json() == {"sources": []}
        finally:
            kb.close()

    def test_bulk_tree_includes_configured_but_unindexed_source(
        self, tmp_path: Path,
    ) -> None:
        # A configured source that has no documents indexed yet should still
        # appear in the bulk response so the sidebar can show it.
        mcp, kb = self._make_app(tmp_path, source_name="freshly-added")
        try:
            client = TestClient(mcp.streamable_http_app())
            response = client.get("/api/sources/tree")
            assert response.status_code == 200
            body = response.json()
            assert len(body["sources"]) == 1
            assert body["sources"][0]["source"] == "freshly-added"
            assert body["sources"][0]["files"] == []
        finally:
            kb.close()
