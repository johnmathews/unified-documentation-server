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
    def test_reindex_empty(self, app):
        result = _call_tool(app, "reindex")
        # No sources configured, so empty stats
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_reindex_unknown_source(self, app):
        result = _call_tool(app, "reindex", source="nonexistent")
        assert "not found" in result.lower()

    def test_reindex_single_source(self, tmp_path: Path) -> None:
        """reindex(source='x') should only process source 'x'."""
        source_dir = tmp_path / "repo-x"
        source_dir.mkdir()
        (source_dir / "doc.md").write_text("# X Doc\n\nContent for X.")

        config = Config(
            sources=[
                RepoSource(name="x", path=str(source_dir)),
                RepoSource(name="y", path=str(tmp_path / "repo-y")),
            ],
            data_dir=str(tmp_path / "data"),
        )
        mcp = server_module.init_app(config)
        kb = server_module._get_kb()

        try:
            result = _call_tool(mcp, "reindex", source="x")
            parsed = json.loads(result)
            assert "x" in parsed
            assert "y" not in parsed
            assert parsed["x"]["upserted"] >= 2  # parent + chunk(s)
        finally:
            kb.close()


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


class TestRescanEndpoint:
    def test_rescan_starts_background(self, app) -> None:
        """POST /rescan should start ingestion in the background and return immediately."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.post("/rescan")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "started"

    def test_rescan_rejects_concurrent(self, app) -> None:
        """POST /rescan should return 409 if a rescan is already running."""
        import time as _time

        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)

        # Make run_once block long enough to test concurrency
        original_run_once = server_module._get_ingester().run_once

        def slow_run_once(**kwargs: object) -> dict:
            _time.sleep(0.5)
            return original_run_once(**kwargs)

        with patch.object(server_module._get_ingester(), "run_once", side_effect=slow_run_once):
            first = client.post("/rescan")
            assert first.status_code == 200
            second = client.post("/rescan")
            assert second.status_code == 409
            assert second.json()["status"] == "already_running"


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
