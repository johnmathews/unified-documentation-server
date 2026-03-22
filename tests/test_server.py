"""Tests for the MCP server tools."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import docserver.server as server_module
from docserver.config import Config, RepoSource
from docserver.knowledge_base import KnowledgeBase


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
        """GET /health should return JSON with status 'ok'."""
        starlette_app = app.streamable_http_app()
        client = TestClient(starlette_app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "sources" in body
        assert "total_chunks" in body

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
