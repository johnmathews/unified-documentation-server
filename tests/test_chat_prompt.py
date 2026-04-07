"""Tests for chat prompt building functions and token-efficient tool execution.

Tests the pure functions that construct the system prompt, execute chat tools,
and compact tool results for the agentic chat endpoint.
"""

import json

from docserver.server import (
    CHAT_SYSTEM_INSTRUCTIONS,
    CHAT_TOOLS,
    _compact_old_tool_results,
    _execute_chat_tool,
    _safe_int,
    _tool_result_summary,
    build_inventory_context,
    build_system_prompt,
)

# ---- Fixtures ---------------------------------------------------------------


def _make_tree(
    *,
    sources: list[str] | None = None,
    with_root_docs: bool = True,
    with_engineering: bool = True,
    with_skills: bool = False,
    with_runbooks: bool = False,
) -> list[dict]:
    """Build a minimal doc_tree fixture."""
    if sources is None:
        sources = ["home-server", "timer-app"]

    tree = []
    for src in sources:
        entry: dict = {
            "source": src,
            "root_docs": [],
            "docs": [
                {
                    "title": "Setup Guide",
                    "file_path": "docs/setup.md",
                    "created_at": "2025-06-15T10:00:00+00:00",
                    "modified_at": "2026-03-01T12:00:00+00:00",
                    "size_bytes": 4200,
                },
                {
                    "title": "Architecture",
                    "file_path": "docs/architecture.md",
                    "created_at": "2025-08-20T14:30:00+00:00",
                    "modified_at": "2026-03-15T09:00:00+00:00",
                    "size_bytes": 8500,
                },
            ],
            "journal": [
                {
                    "title": "Initial commit",
                    "file_path": "journal/260101-init.md",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "modified_at": "2026-01-01T00:00:00+00:00",
                    "size_bytes": 1200,
                },
            ],
        }
        if with_root_docs:
            entry["root_docs"] = [
                {"title": None, "file_path": "README.md", "created_at": "2025-01-01T00:00:00+00:00", "modified_at": "2026-03-28T10:00:00+00:00", "size_bytes": 3000},
                {"title": None, "file_path": "CLAUDE.md", "created_at": "2025-11-01T00:00:00+00:00", "modified_at": "2026-03-20T10:00:00+00:00", "size_bytes": 1500},
            ]
        if with_engineering:
            entry["engineering_team"] = [
                {"title": "Eval Report", "file_path": ".engineering-team/evaluation-report.md", "created_at": "2026-03-25T00:00:00+00:00", "modified_at": "2026-03-25T22:00:00+00:00", "size_bytes": 13000},
            ]
        if with_skills:
            entry["skills"] = [
                {"title": "Weather Skill", "file_path": "container/skills/weather/skill.md", "created_at": "2026-03-20T00:00:00+00:00", "modified_at": "2026-03-20T00:00:00+00:00", "size_bytes": 2000},
                {"title": "Calendar Skill", "file_path": "container/skills/calendar/skill.md", "created_at": "2026-03-21T00:00:00+00:00", "modified_at": "2026-03-21T00:00:00+00:00", "size_bytes": 1800},
            ]
        if with_runbooks:
            entry["runbooks"] = [
                {"title": "Deploy Guide", "file_path": "runbooks/deploy-guide.md", "created_at": "2026-03-18T00:00:00+00:00", "modified_at": "2026-03-18T00:00:00+00:00", "size_bytes": 3500},
            ]
        tree.append(entry)
    return tree


def _make_stats(sources: list[str] | None = None) -> dict[str, dict]:
    """Build a source_stats fixture keyed by source name."""
    if sources is None:
        sources = ["home-server", "timer-app"]
    return {
        src: {
            "source": src,
            "file_count": 10 + i * 5,
            "chunk_count": 30 + i * 15,
            "last_indexed": f"2026-03-28T12:00:0{i}+00:00",
        }
        for i, src in enumerate(sources)
    }


# ---- CHAT_SYSTEM_INSTRUCTIONS ------------------------------------------------


class TestSystemInstructions:
    """Verify the system instructions contain critical phrases."""

    def test_identifies_as_documentation_assistant(self):
        assert "documentation assistant" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_inventory(self):
        assert "inventory" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_tools(self):
        assert "search_docs" in CHAT_SYSTEM_INSTRUCTIONS
        assert "query_docs" in CHAT_SYSTEM_INSTRUCTIONS
        assert "get_document" in CHAT_SYSTEM_INSTRUCTIONS

    def test_encourages_proactive_search(self):
        assert "Search proactively" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_unified_server(self):
        assert "unified documentation server" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_structural_queries(self):
        """Model should use query_docs for structural questions."""
        assert "query_docs" in CHAT_SYSTEM_INSTRUCTIONS

    def test_concise_directive(self):
        assert "concise and direct" in CHAT_SYSTEM_INSTRUCTIONS


# ---- build_inventory_context -------------------------------------------------


class TestBuildInventoryContext:
    """Test the compact inventory context builder."""

    def test_includes_source_count(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "2 sources" in result

    def test_includes_total_file_count(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        # 10 + 15 = 25
        assert "25 files" in result

    def test_includes_total_chunk_count(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        # 30 + 45 = 75
        assert "75 vector chunks" in result

    def test_includes_per_source_file_count(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "home-server: 10 files" in result
        assert "timer-app: 15 files" in result

    def test_includes_category_counts(self):
        """Compact format shows category counts, not individual documents."""
        tree = _make_tree(with_root_docs=True, with_engineering=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "2 root" in result
        assert "2 docs" in result
        assert "1 journal" in result
        assert "1 engineering" in result

    def test_skills_category_count(self):
        tree = _make_tree(with_skills=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "2 skills" in result

    def test_runbooks_category_count(self):
        tree = _make_tree(with_runbooks=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "1 runbooks" in result

    def test_no_per_document_listings(self):
        """Compact format must NOT list individual document titles or paths."""
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        # These are individual doc details that should NOT appear
        assert "Setup Guide" not in result
        assert "Architecture" not in result
        assert "Initial commit" not in result
        assert "README.md" not in result
        assert "created=" not in result
        assert "modified=" not in result
        assert "size=" not in result

    def test_omits_empty_categories(self):
        tree = _make_tree(with_root_docs=False, with_engineering=False)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "root" not in result
        assert "engineering" not in result

    def test_missing_engineering_team_key(self):
        """Tree entries without engineering_team key should not crash."""
        tree = _make_tree()
        for src in tree:
            del src["engineering_team"]
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "engineering" not in result

    def test_source_not_in_stats_shows_zero_files(self):
        tree = _make_tree(sources=["unknown-source"])
        stats = {}
        result = build_inventory_context(tree, stats)
        assert "unknown-source: 0 files" in result

    def test_empty_tree(self):
        result = build_inventory_context([], {})
        assert "0 sources" in result
        assert "0 files" in result
        assert "0 vector chunks" in result

    def test_single_source(self):
        tree = _make_tree(sources=["solo"])
        stats = _make_stats(sources=["solo"])
        result = build_inventory_context(tree, stats)
        assert "1 sources" in result
        assert "solo:" in result

    def test_compact_size(self):
        """Inventory should be much smaller than the old per-document format."""
        tree = _make_tree(with_root_docs=True, with_engineering=True, with_skills=True, with_runbooks=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        # With 2 sources and all categories, should be well under 500 chars
        assert len(result) < 500


# ---- build_system_prompt -----------------------------------------------------


class TestBuildSystemPrompt:
    """Test the full system prompt assembly."""

    def test_starts_with_instructions(self):
        prompt = build_system_prompt([])
        assert prompt.startswith(CHAT_SYSTEM_INSTRUCTIONS)

    def test_no_context_parts_returns_instructions_only(self):
        prompt = build_system_prompt([])
        assert prompt == CHAT_SYSTEM_INSTRUCTIONS

    def test_context_parts_appended_after_separator(self):
        prompt = build_system_prompt(["Part one", "Part two"])
        assert CHAT_SYSTEM_INSTRUCTIONS in prompt
        assert "Part one" in prompt
        assert "Part two" in prompt

    def test_context_parts_separated_by_hr(self):
        prompt = build_system_prompt(["AAA", "BBB"])
        assert "AAA\n\n---\n\nBBB" in prompt

    def test_instructions_separated_from_context(self):
        prompt = build_system_prompt(["Context here"])
        idx = prompt.index("Context here")
        before = prompt[:idx]
        assert before.endswith("\n\n")

    def test_full_prompt_with_inventory(self):
        """Integration: build inventory, then full prompt."""
        tree = _make_tree()
        stats = _make_stats()
        inventory = build_inventory_context(tree, stats)
        prompt = build_system_prompt([inventory])

        assert "documentation assistant" in prompt
        assert "2 sources" in prompt
        assert "home-server:" in prompt
        assert "10 files" in prompt

    def test_full_prompt_with_inventory_and_extra_context(self):
        """Inventory + extra context both appear in final prompt."""
        tree = _make_tree(sources=["myrepo"])
        stats = _make_stats(sources=["myrepo"])
        inventory = build_inventory_context(tree, stats)
        extra = "The user is browsing source 'myrepo'."
        prompt = build_system_prompt([inventory, extra])

        assert "myrepo" in prompt
        assert "browsing" in prompt
        assert "---" in prompt

    def test_page_context_source_only(self):
        inventory = build_inventory_context([], {})
        ctx = "The user is browsing source 'm3'."
        prompt = build_system_prompt([inventory, ctx])
        assert "'m3'" in prompt

    def test_page_context_source_and_category(self):
        inventory = build_inventory_context([], {})
        ctx = "The user is browsing 'journal' in 'm3'."
        prompt = build_system_prompt([inventory, ctx])
        assert "'journal'" in prompt
        assert "'m3'" in prompt


# ---- CHAT_TOOLS -------------------------------------------------------------


class TestChatTools:
    """Verify the chat tool definitions are well-formed."""

    def test_has_four_tools(self):
        assert len(CHAT_TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in CHAT_TOOLS}
        assert names == {"search_docs", "query_docs", "get_document", "list_sources"}

    def test_all_tools_have_required_fields(self):
        for tool in CHAT_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_search_docs_requires_query(self):
        search = next(t for t in CHAT_TOOLS if t["name"] == "search_docs")
        assert "query" in search["input_schema"]["required"]

    def test_get_document_requires_doc_id(self):
        get_doc = next(t for t in CHAT_TOOLS if t["name"] == "get_document")
        assert "doc_id" in get_doc["input_schema"]["required"]

    def test_last_tool_has_cache_control(self):
        """Last tool should have cache_control for prompt caching."""
        last_tool = CHAT_TOOLS[-1]
        assert "cache_control" in last_tool
        assert last_tool["cache_control"]["type"] == "ephemeral"


# ---- _execute_chat_tool -----------------------------------------------------


class TestExecuteChatTool:
    """Test the chat tool executor with a mock KnowledgeBase."""

    class MockKB:
        """Minimal mock for KnowledgeBase methods used by _execute_chat_tool."""

        def search(self, *, query: str, n_results: int, source_filter: str | None = None):
            if query == "empty":
                return []
            return [
                {
                    "content": f"Result for: {query}",
                    "score": 0.95,
                    "metadata": {
                        "title": "Test Doc",
                        "source": "test-source",
                        "file_path": "docs/test.md",
                    },
                }
            ]

        def query_documents(self, **kwargs):
            if kwargs.get("source") == "empty":
                return []
            return [
                {
                    "doc_id": "test:docs/test.md",
                    "title": "Test Doc",
                    "source": "test-source",
                    "file_path": "docs/test.md",
                    "chunk_index": 0,
                    "total_chunks": 3,
                    "created_at": "2025-06-15",
                    "size_bytes": 4200,
                }
            ]

        def get_document(self, doc_id: str):
            if doc_id == "missing:doc":
                return None
            if doc_id == "large:doc":
                return {"doc_id": doc_id, "title": "Large", "content": "x" * 20000}
            return {"doc_id": doc_id, "title": "Found", "content": "Content here"}

        def get_sources_summary(self):
            return [{"source": "test-source", "file_count": 5, "chunk_count": 20}]

    def test_search_docs_returns_results(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "search_docs", {"query": "hello"})  # type: ignore[arg-type]
        assert "Test Doc" in result
        assert "Result for: hello" in result

    def test_search_docs_compact_format(self):
        """Search results should use compact single-line headers."""
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "search_docs", {"query": "hello"})  # type: ignore[arg-type]
        # Compact format: [source:path] title (score:X.XX)
        assert "[test-source:docs/test.md]" in result
        assert "score:0.95" in result

    def test_search_docs_truncates_long_content(self):
        """Search results with long content should be truncated."""

        class LongContentKB(self.MockKB):
            def search(self, *, query: str, n_results: int, source_filter: str | None = None):
                return [
                    {
                        "content": "x" * 500,
                        "score": 0.9,
                        "metadata": {"title": "Long", "source": "s", "file_path": "f.md"},
                    }
                ]

        kb = LongContentKB()
        result = _execute_chat_tool(kb, "search_docs", {"query": "test"})  # type: ignore[arg-type]
        assert len(result) < 500
        assert "..." in result

    def test_search_docs_empty(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "search_docs", {"query": "empty"})  # type: ignore[arg-type]
        assert "No matching documents found" in result

    def test_query_docs_returns_compact_fields(self):
        """query_docs should return only key fields (doc_id, title, source, file_path)."""
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "query_docs", {"source": "test"})  # type: ignore[arg-type]
        parsed = json.loads(result)
        assert len(parsed) == 1
        doc = parsed[0]
        assert set(doc.keys()) == {"doc_id", "title", "source", "file_path"}
        assert doc["title"] == "Test Doc"

    def test_query_docs_excludes_extra_metadata(self):
        """query_docs should NOT include chunk_index, size_bytes, dates, etc."""
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "query_docs", {"source": "test"})  # type: ignore[arg-type]
        assert "chunk_index" not in result
        assert "size_bytes" not in result
        assert "created_at" not in result

    def test_query_docs_empty(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "query_docs", {"source": "empty"})  # type: ignore[arg-type]
        assert "No matching documents found" in result

    def test_get_document_found(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "get_document", {"doc_id": "test:docs/test.md"})  # type: ignore[arg-type]
        assert "Found" in result
        assert "Content here" in result

    def test_get_document_not_found(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "get_document", {"doc_id": "missing:doc"})  # type: ignore[arg-type]
        assert "not found" in result

    def test_get_document_truncates_large_docs(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "get_document", {"doc_id": "large:doc"})  # type: ignore[arg-type]
        assert len(result) < 10000
        assert "truncated" in result

    def test_list_sources(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "list_sources", {})  # type: ignore[arg-type]
        assert "test-source" in result
        assert "5" in result

    def test_unknown_tool(self):
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "nonexistent", {})  # type: ignore[arg-type]
        assert "Unknown tool" in result

    def test_search_docs_invalid_num_results(self):
        """Non-numeric num_results falls back to default without crashing."""
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "search_docs", {"query": "hello", "num_results": "five"})  # type: ignore[arg-type]
        assert "Test Doc" in result

    def test_query_docs_invalid_limit(self):
        """Non-numeric limit falls back to default without crashing."""
        kb = self.MockKB()
        result = _execute_chat_tool(kb, "query_docs", {"source": "test", "limit": None})  # type: ignore[arg-type]
        assert "Test Doc" in result


# ---- _compact_old_tool_results -----------------------------------------------


class TestCompactOldToolResults:
    """Test the tool result compaction for the agentic loop."""

    def test_no_tool_results_is_noop(self):
        messages: list[dict] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        _compact_old_tool_results(messages)  # type: ignore[arg-type]
        assert messages[0]["content"] == "Hello"

    def test_single_tool_result_not_compacted(self):
        """The most recent (only) tool result should not be compacted."""
        messages: list[dict] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "tool_use response"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "A" * 500},
            ]},
        ]
        _compact_old_tool_results(messages)  # type: ignore[arg-type]
        assert messages[2]["content"][0]["content"] == "A" * 500

    def test_older_tool_results_compacted(self):
        """Older tool results should be replaced with short summaries."""
        messages: list[dict] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "tool_use 1"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "B" * 500},
            ]},
            {"role": "assistant", "content": "tool_use 2"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "C" * 500},
            ]},
        ]
        _compact_old_tool_results(messages)  # type: ignore[arg-type]
        # First tool result (older) should be compacted
        first_result = messages[2]["content"][0]["content"]
        assert "Prior result" in first_result
        assert len(first_result) < 100
        # Latest tool result should remain intact
        latest_result = messages[4]["content"][0]["content"]
        assert latest_result == "C" * 500

    def test_short_results_not_compacted(self):
        """Short tool results below the threshold should remain intact."""
        messages: list[dict] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "tool_use 1"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "short"},
            ]},
            {"role": "assistant", "content": "tool_use 2"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": "also short"},
            ]},
        ]
        _compact_old_tool_results(messages)  # type: ignore[arg-type]
        # Short result should stay
        assert messages[2]["content"][0]["content"] == "short"

    def test_multiple_tool_results_in_one_message(self):
        """Multiple tool results in a single message should all be compacted if old."""
        messages: list[dict] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "tool_use batch"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "D" * 500},
                {"type": "tool_result", "tool_use_id": "2", "content": "E" * 300},
            ]},
            {"role": "assistant", "content": "tool_use 3"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "3", "content": "F" * 500},
            ]},
        ]
        _compact_old_tool_results(messages)  # type: ignore[arg-type]
        # Both older results compacted
        assert "Prior result" in messages[2]["content"][0]["content"]
        assert "Prior result" in messages[2]["content"][1]["content"]
        # Latest intact
        assert messages[4]["content"][0]["content"] == "F" * 500


# ---- _tool_result_summary ----------------------------------------------------


class TestToolResultSummary:
    """Test the human-readable tool result summary for SSE events."""

    def test_search_docs_with_results(self):
        # search_docs results are separated by double newlines
        result = "chunk1\n\nchunk2\n\nchunk3"
        summary = _tool_result_summary("search_docs", result)
        assert "3 results found" in summary

    def test_search_docs_single_result(self):
        summary = _tool_result_summary("search_docs", "one result only")
        assert "1 result found" in summary

    def test_search_docs_no_results(self):
        summary = _tool_result_summary("search_docs", "No matching documents found.")
        assert "No results" in summary

    def test_query_docs_with_results(self):
        result = json.dumps([{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}])
        summary = _tool_result_summary("query_docs", result)
        assert "Found 3 documents" in summary

    def test_query_docs_single_result(self):
        result = json.dumps([{"doc_id": "a"}])
        summary = _tool_result_summary("query_docs", result)
        assert "Found 1 document" in summary

    def test_query_docs_no_results(self):
        summary = _tool_result_summary("query_docs", "No matching documents found.")
        assert "No results" in summary

    def test_get_document_found(self):
        result = "x" * 2500
        summary = _tool_result_summary("get_document", result)
        assert "Document retrieved" in summary
        assert "2,500" in summary

    def test_get_document_not_found(self):
        summary = _tool_result_summary("get_document", "Document 'x:y' not found.")
        assert "No results" in summary

    def test_list_sources(self):
        result = json.dumps([{"source": "a"}, {"source": "b"}])
        summary = _tool_result_summary("list_sources", result)
        assert "Listed 2 sources" in summary

    def test_unknown_tool(self):
        summary = _tool_result_summary("something_else", "x" * 100)
        assert "100" in summary


# ---- _safe_int ---------------------------------------------------------------


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(10, default=5, lo=1, hi=20) == 10

    def test_string_number(self):
        assert _safe_int("10", default=5, lo=1, hi=20) == 10

    def test_none_returns_default(self):
        assert _safe_int(None, default=5, lo=1, hi=20) == 5

    def test_invalid_string_returns_default(self):
        assert _safe_int("five", default=5, lo=1, hi=20) == 5

    def test_clamped_low(self):
        assert _safe_int(0, default=5, lo=1, hi=20) == 1

    def test_clamped_high(self):
        assert _safe_int(100, default=5, lo=1, hi=20) == 20
