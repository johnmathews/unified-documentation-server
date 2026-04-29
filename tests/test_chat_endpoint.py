"""Tests for the /api/chat and /api/chat/stream HTTP endpoints.

Covers request validation, the agentic tool-use loop, SSE event formatting,
error handling (rate limits, API errors), and structured logging.
"""

import json
import os
import re
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock, ToolUseBlock
from starlette.testclient import TestClient

import docserver.server as server_module
from docserver.config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_anthropic_client_cache():
    """Clear the module-level Anthropic client singleton between tests.

    Chat endpoint tests patch ``docserver.server.anthropic.Anthropic`` with
    mocks. The server caches a single client for the lifetime of the
    process, so without this reset the first test's cached mock would be
    reused by subsequent tests, bypassing their patches.

    Also resets the chat-model probe state so a test that flips it to
    invalid does not bleed into later tests.
    """
    server_module._anthropic_client = None
    server_module._anthropic_client_class = None
    server_module._chat_model_valid = True
    server_module._chat_model_error = None
    yield
    server_module._anthropic_client = None
    server_module._anthropic_client_class = None
    server_module._chat_model_valid = True
    server_module._chat_model_error = None


@pytest.fixture
def app(tmp_path):
    """Initialize server with a temp data dir and seed test data."""
    config = Config(
        sources=[],
        data_dir=str(tmp_path / "data"),
        poll_interval_seconds=9999,
    )
    mcp = server_module.init_app(config)
    kb = server_module._get_kb()

    kb.upsert_document(
        "docs:readme.md",
        "",
        {
            "source": "docs",
            "file_path": "readme.md",
            "title": "Readme",
            "is_chunk": False,
            "total_chunks": 1,
        },
    )
    kb.upsert_document(
        "docs:readme.md#chunk0",
        "Welcome to the documentation server. It indexes markdown files.",
        {
            "source": "docs",
            "file_path": "readme.md",
            "title": "Readme",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )

    yield mcp
    kb.close()


@pytest.fixture
def client(app):
    """Starlette TestClient for HTTP endpoint tests."""
    return TestClient(app.streamable_http_app())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def _make_text_block(text: str = "This is the answer.") -> MagicMock:
    block = MagicMock(spec=TextBlock)
    block.type = "text"
    block.text = text
    return block


def _make_tool_block(
    tool_id: str = "tool_abc",
    name: str = "search_docs",
    tool_input: dict | None = None,
) -> MagicMock:
    block = MagicMock(spec=ToolUseBlock)
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input or {"query": "test"}
    return block


@dataclass
class _FakeResponse:
    stop_reason: str = "end_turn"
    content: list = None
    usage: _FakeUsage = None

    def __post_init__(self):
        if self.content is None:
            self.content = [_make_text_block()]
        if self.usage is None:
            self.usage = _FakeUsage()


def _patch_anthropic(responses):
    """Patch the Anthropic client to return canned responses.

    Returns the mock so callers can inspect calls.
    """
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = list(responses)

    mock_class = MagicMock(return_value=mock_client)
    return patch("docserver.server.anthropic.Anthropic", mock_class), mock_client


# ---------------------------------------------------------------------------
# /api/chat — sync endpoint
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_missing_message(self, client):
        response = client.post(
            "/api/chat",
            json={},
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "Missing" in response.json()["error"]

    def test_empty_message(self, client):
        response = client.post(
            "/api/chat",
            json={"message": ""},
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_no_api_key(self, client):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            response = client.post(
                "/api/chat",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 503
            assert "API_KEY" in response.json()["error"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_simple_reply(self, client):
        """Chat returns a reply without tool use."""
        patcher, mock_client = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("Hello!")]),
        ])
        with patcher:
            response = client.post(
                "/api/chat",
                json={"message": "hi"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "Hello!"
        assert "conversation_id" in body
        # The model passed to the Anthropic SDK must match the configured
        # default. This guard would have caught the 2026-04-25 outage where
        # CHAT_MODEL was set to a non-existent alias.
        assert (
            mock_client.messages.create.call_args.kwargs["model"]
            == server_module.CHAT_MODEL
        )

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_with_tool_use(self, client):
        """Chat makes a tool call then returns final reply."""
        tool_block = _make_tool_block("tool_123", "search_docs", {"query": "test"})

        tool_response = _FakeResponse(
            stop_reason="tool_use",
            content=[tool_block],
        )
        text_response = _FakeResponse(
            stop_reason="end_turn",
            content=[_make_text_block("Found results.")],
        )

        patcher, mock_client = _patch_anthropic([tool_response, text_response])
        with patcher:
            response = client.post(
                "/api/chat",
                json={"message": "search for ports"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "Found results."
        assert mock_client.messages.create.call_count == 2

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_rate_limit_error(self, client):
        """Rate limit errors return 429."""
        import anthropic as anthropic_mod

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_client.messages.create.side_effect = anthropic_mod.RateLimitError(
            message="Rate limited",
            response=mock_resp,
            body=None,
        )

        with patch("docserver.server.anthropic.Anthropic", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 429
        assert "Rate limit" in response.json()["error"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_chat_returns_503_when_model_invalid(self, client):
        """When the startup probe marked the chat model invalid, /api/chat short-circuits with 503."""
        server_module._chat_model_valid = False
        server_module._chat_model_error = "model: claude-bogus-1 not found"

        response = client.post(
            "/api/chat",
            json={"message": "hi"},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 503
        body = response.json()
        assert "misconfigured" in body["error"].lower()
        assert "claude-bogus-1" in body["error"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_not_found_error_returns_500(self, client):
        """Anthropic NotFoundError on chat call returns 500 (server config error), not 502."""
        import anthropic as anthropic_mod

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {}
        mock_client.messages.create.side_effect = anthropic_mod.NotFoundError(
            message="model: claude-bogus-1 not found",
            response=mock_resp,
            body=None,
        )

        with patch("docserver.server.anthropic.Anthropic", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 500
        assert "config error" in response.json()["error"].lower()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_connection_error_returns_502(self, client):
        """Anthropic APIConnectionError returns 502 (true gateway failure)."""
        import anthropic as anthropic_mod

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic_mod.APIConnectionError(
            message="connection refused",
            request=MagicMock(),
        )

        with patch("docserver.server.anthropic.Anthropic", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 502
        assert "unreachable" in response.json()["error"].lower()

    def test_options_returns_cors(self, client):
        response = client.options("/api/chat")
        assert response.status_code == 200

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_with_conversation_id(self, client):
        """Chat with an existing conversation_id updates it."""
        patcher, _ = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("First reply.")]),
        ])
        with patcher:
            resp1 = client.post(
                "/api/chat",
                json={"message": "hi"},
                headers={"Content-Type": "application/json"},
            )
        conv_id = resp1.json()["conversation_id"]

        patcher2, _ = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("Second reply.")]),
        ])
        with patcher2:
            resp2 = client.post(
                "/api/chat",
                json={
                    "message": "follow up",
                    "conversation_id": conv_id,
                    "history": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "First reply."},
                    ],
                },
                headers={"Content-Type": "application/json"},
            )
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id


# ---------------------------------------------------------------------------
# /api/chat/stream — SSE endpoint
# ---------------------------------------------------------------------------

def _parse_sse_events(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    for block in raw.split("\r\n\r\n"):
        block = block.strip()
        if not block:
            continue
        event_type = "message"
        data_lines = []
        for line in block.replace("\r\n", "\n").split("\n"):
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
            # skip id:, retry:, comments (:)
        if data_lines:
            events.append({"event": event_type, "data": "\n".join(data_lines)})
    return events


class TestChatStreamEndpoint:
    def test_missing_message(self, client):
        response = client.post(
            "/api/chat/stream",
            json={},
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_options_returns_cors(self, client):
        response = client.options("/api/chat/stream")
        assert response.status_code == 200

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_simple_stream(self, client):
        """Stream returns status and reply events."""
        patcher, mock_client = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("Streamed answer.")]),
        ])
        with patcher:
            response = client.post(
                "/api/chat/stream",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "reply" in event_types
        # Same guard as in test_simple_reply.
        assert (
            mock_client.messages.create.call_args.kwargs["model"]
            == server_module.CHAT_MODEL
        )

        reply_event = next(e for e in events if e["event"] == "reply")
        reply_data = json.loads(reply_event["data"])
        assert reply_data["reply"] == "Streamed answer."
        assert "conversation_id" in reply_data

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_stream_with_tool_use(self, client):
        """Stream emits tool_call and tool_result events during agentic loop."""
        tool_block = _make_tool_block("tool_456", "search_docs", {"query": "ports"})

        tool_response = _FakeResponse(
            stop_reason="tool_use",
            content=[tool_block],
        )
        text_response = _FakeResponse(
            stop_reason="end_turn",
            content=[_make_text_block("Port 8080.")],
        )

        patcher, _ = _patch_anthropic([tool_response, text_response])
        with patcher:
            response = client.post(
                "/api/chat/stream",
                json={"message": "what port"},
                headers={"Content-Type": "application/json"},
            )

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "reply" in event_types

        tool_call = next(e for e in events if e["event"] == "tool_call")
        tc_data = json.loads(tool_call["data"])
        assert tc_data["tool"] == "search_docs"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_stream_rate_limit(self, client):
        """Rate limit errors produce an SSE error event."""
        import anthropic as anthropic_mod

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_client.messages.create.side_effect = anthropic_mod.RateLimitError(
            message="Rate limited",
            response=mock_resp,
            body=None,
        )

        with patch("docserver.server.anthropic.Anthropic", return_value=mock_client):
            response = client.post(
                "/api/chat/stream",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        # SSE endpoint returns 200 with error event in stream
        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert "Rate limit" in json.loads(error_events[0]["data"])["error"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_stream_generic_error(self, client):
        """Unexpected exceptions produce an SSE error event."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("boom")

        with patch("docserver.server.anthropic.Anthropic", return_value=mock_client):
            response = client.post(
                "/api/chat/stream",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert "Internal error" in json.loads(error_events[0]["data"])["error"]


# ---------------------------------------------------------------------------
# /api/conversations — CRUD
# ---------------------------------------------------------------------------

class TestConversationEndpoints:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_list_empty(self, client):
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert response.json() == []

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_crud_lifecycle(self, client):
        """Create a conversation via chat, list it, get it, delete it."""
        patcher, _ = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("Hello!")]),
        ])
        with patcher:
            chat_resp = client.post(
                "/api/chat",
                json={"message": "hello there"},
                headers={"Content-Type": "application/json"},
            )
        conv_id = chat_resp.json()["conversation_id"]

        # List
        list_resp = client.get("/api/conversations")
        assert list_resp.status_code == 200
        convs = list_resp.json()
        assert len(convs) == 1
        assert convs[0]["id"] == conv_id

        # Get
        get_resp = client.get(f"/api/conversations/{conv_id}")
        assert get_resp.status_code == 200
        conv = get_resp.json()
        assert conv["id"] == conv_id
        assert len(conv["messages"]) == 2  # user + assistant

        # Delete
        del_resp = client.request("DELETE", f"/api/conversations/{conv_id}")
        assert del_resp.status_code == 200

        # Verify deleted
        list_resp2 = client.get("/api/conversations")
        assert list_resp2.json() == []


class TestSystemPromptDate:
    """Verify the system prompt includes the current UTC date."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_system_prompt_contains_utc_date(self, client):
        """The first system block should start with today's UTC date."""
        patcher, mock_client = _patch_anthropic([
            _FakeResponse(content=[_make_text_block("Hi")]),
        ])
        with patcher:
            client.post(
                "/api/chat",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )

        call_kwargs = mock_client.messages.create.call_args
        system_blocks = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        first_text = system_blocks[0]["text"]
        assert re.match(r"Today's date is \d{4}-\d{2}-\d{2} \(UTC\)\.", first_text)
