"""Tests for chat prompt building functions.

These test the pure functions that construct the system prompt for the chat
endpoint, ensuring the agent receives the right context to answer meta
questions about indexing status, document counts, and source inventory.
"""


from docserver.server import (
    CHAT_SYSTEM_INSTRUCTIONS,
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

    def test_mentions_file_counts(self):
        assert "file counts" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_chunk_counts(self):
        assert "chunk counts" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_timestamps(self):
        assert "last-indexed timestamps" in CHAT_SYSTEM_INSTRUCTIONS

    def test_instructs_confident_answers(self):
        assert "Answer confidently" in CHAT_SYSTEM_INSTRUCTIONS

    def test_discourages_hedging(self):
        assert "I would need to use" in CHAT_SYSTEM_INSTRUCTIONS
        assert "I cannot confirm" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_structural_questions(self):
        assert "structural questions" in CHAT_SYSTEM_INSTRUCTIONS

    def test_mentions_journal_dates(self):
        assert "most recent journal entry" in CHAT_SYSTEM_INSTRUCTIONS


# ---- build_inventory_context -------------------------------------------------


class TestBuildInventoryContext:
    """Test the inventory context builder."""

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

    def test_includes_per_source_stats(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "**home-server** (10 files, 30 chunks" in result
        assert "**timer-app** (15 files, 45 chunks" in result

    def test_includes_last_indexed_timestamp(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "2026-03-28T12:00:00+00:00" in result

    def test_includes_root_docs_category(self):
        tree = _make_tree(with_root_docs=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Root docs (2)" in result
        assert "README.md" in result
        assert "CLAUDE.md" in result

    def test_includes_documentation_category(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Documentation (2)" in result
        assert "Setup Guide" in result
        assert "Architecture" in result

    def test_includes_journal_category(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Journal (1)" in result
        assert "Initial commit" in result

    def test_includes_engineering_team_category(self):
        tree = _make_tree(with_engineering=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Engineering team (1)" in result
        assert "Eval Report" in result

    def test_includes_skills_category(self):
        tree = _make_tree(with_skills=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Skills (2)" in result
        assert "Weather Skill" in result
        assert "Calendar Skill" in result

    def test_includes_runbooks_category(self):
        tree = _make_tree(with_runbooks=True)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Runbooks (1)" in result
        assert "Deploy Guide" in result

    def test_includes_created_at_dates(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "created=2025-06-15T10:00:00+00:00" in result  # Setup Guide
        assert "created=2026-01-01T00:00:00+00:00" in result  # Initial commit

    def test_includes_modified_at_dates(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "modified=2026-03-01T12:00:00+00:00" in result  # Setup Guide

    def test_includes_size_bytes(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "size=4200b" in result  # Setup Guide
        assert "size=8500b" in result  # Architecture

    def test_includes_file_path_when_title_present(self):
        tree = _make_tree()
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "path=docs/setup.md" in result
        assert "path=docs/architecture.md" in result

    def test_doc_without_dates_omits_date_fields(self):
        tree = [{
            "source": "test",
            "root_docs": [],
            "docs": [{"title": "No Dates", "file_path": "docs/nodates.md"}],
            "journal": [],
        }]
        stats = _make_stats(sources=["test"])
        result = build_inventory_context(tree, stats)
        assert "No Dates" in result
        assert "created=" not in result
        assert "modified=" not in result

    def test_omits_empty_root_docs(self):
        tree = _make_tree(with_root_docs=False)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Root docs" not in result

    def test_omits_empty_engineering_team(self):
        tree = _make_tree(with_engineering=False)
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Engineering team" not in result

    def test_missing_engineering_team_key(self):
        """Tree entries without engineering_team key should not crash."""
        tree = _make_tree()
        for src in tree:
            del src["engineering_team"]
        stats = _make_stats()
        result = build_inventory_context(tree, stats)
        assert "Engineering team" not in result

    def test_source_not_in_stats_shows_zeros(self):
        tree = _make_tree(sources=["unknown-source"])
        stats = {}  # No stats at all
        result = build_inventory_context(tree, stats)
        assert "**unknown-source** (0 files, 0 chunks, last indexed: never)" in result

    def test_fallback_to_file_path_when_title_is_none(self):
        tree = [{
            "source": "test",
            "root_docs": [],
            "docs": [{"title": None, "file_path": "docs/no-title.md"}],
            "journal": [],
        }]
        stats = _make_stats(sources=["test"])
        result = build_inventory_context(tree, stats)
        assert "docs/no-title.md" in result

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
        assert "**solo**" in result


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
        # Parts should be separated by ---
        assert "AAA\n\n---\n\nBBB" in prompt

    def test_instructions_separated_from_context(self):
        prompt = build_system_prompt(["Context here"])
        # There should be a double newline between instructions and context
        idx = prompt.index("Context here")
        before = prompt[:idx]
        assert before.endswith("\n\n")

    def test_full_prompt_with_inventory(self):
        """Integration: build inventory, then full prompt, verify key data present."""
        tree = _make_tree()
        stats = _make_stats()
        inventory = build_inventory_context(tree, stats)
        prompt = build_system_prompt([inventory])

        # Instructions present
        assert "documentation assistant" in prompt
        assert "Answer confidently" in prompt

        # Inventory data present
        assert "2 sources" in prompt
        assert "**home-server**" in prompt
        assert "10 files" in prompt
        assert "30 chunks" in prompt
        assert "Root docs (2)" in prompt
        assert "Journal (1)" in prompt

    def test_full_prompt_with_inventory_and_rag(self):
        """Inventory + RAG context both appear in final prompt."""
        tree = _make_tree(sources=["myrepo"])
        stats = _make_stats(sources=["myrepo"])
        inventory = build_inventory_context(tree, stats)
        rag = "Relevant documentation excerpts:\n\nSome search result content here."
        prompt = build_system_prompt([inventory, rag])

        assert "**myrepo**" in prompt
        assert "Some search result content here" in prompt
        # Both separated by ---
        assert "---" in prompt
