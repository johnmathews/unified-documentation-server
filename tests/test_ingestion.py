"""Tests for the ingestion module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docserver.config import Config, RepoSource
from docserver.ingestion import (
    MAX_FILE_SIZE,
    DocumentParser,
    Ingester,
    RepoManager,
    _chunk_content,
    _normalise_repo_url,
    _parse_sections,
)
from docserver.knowledge_base import KnowledgeBase


class TestSectionParsing:
    def test_flat_content_no_headings(self):
        sections = _parse_sections("Hello world.\n\nSecond paragraph.")
        assert len(sections) == 1
        assert sections[0]["heading_path"] == ""
        assert len(sections[0]["blocks"]) == 2

    def test_single_heading(self):
        content = "# Title\n\nParagraph one.\n\nParagraph two."
        sections = _parse_sections(content)
        assert len(sections) == 1
        assert sections[0]["heading_path"] == "Title"
        assert len(sections[0]["blocks"]) == 2

    def test_nested_headings(self):
        content = "# Top\n\nIntro.\n\n## Sub A\n\nContent A.\n\n## Sub B\n\nContent B."
        sections = _parse_sections(content)
        assert len(sections) == 3
        assert sections[0]["heading_path"] == "Top"
        assert sections[1]["heading_path"] == "Top > Sub A"
        assert sections[2]["heading_path"] == "Top > Sub B"

    def test_deep_nesting(self):
        content = "# H1\n\n## H2\n\n### H3\n\nDeep content."
        sections = _parse_sections(content)
        assert sections[-1]["heading_path"] == "H1 > H2 > H3"

    def test_heading_level_reset(self):
        content = "# First\n\n## Sub\n\nA.\n\n# Second\n\nB."
        sections = _parse_sections(content)
        paths = [s["heading_path"] for s in sections]
        assert "First > Sub" in paths
        assert "Second" in paths

    def test_list_kept_together(self):
        content = "# Lists\n\n- item 1\n- item 2\n- item 3\n\nAfter list."
        sections = _parse_sections(content)
        blocks = sections[0]["blocks"]
        # List items should be in one block
        list_block = next(b for b in blocks if "item 1" in b)
        assert "item 2" in list_block
        assert "item 3" in list_block

    def test_code_fence_kept_together(self):
        content = "# Code\n\n```python\ndef foo():\n    pass\n```\n\nAfter code."
        sections = _parse_sections(content)
        blocks = sections[0]["blocks"]
        code_block = next(b for b in blocks if "def foo" in b)
        assert "```python" in code_block
        assert "```" in code_block.split("\n")[-1]

    def test_code_fence_with_headings_inside(self):
        content = "# Real\n\n```\n# Not a heading\n## Also not\n```\n\nAfter."
        sections = _parse_sections(content)
        # Should be one section — headings inside code fence are ignored
        assert len(sections) == 1
        assert sections[0]["heading_path"] == "Real"


class TestChunking:
    def test_small_content_single_chunk(self):
        chunks = _chunk_content("Short text.")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].section_path == ""

    def test_splits_on_paragraph_boundaries(self):
        para1 = "A" * 600
        para2 = "B" * 600
        content = f"{para1}\n\n{para2}"
        chunks = _chunk_content(content, target_size=1000, overlap_size=0)
        assert len(chunks) == 2
        assert para1 in chunks[0].text
        assert para2 in chunks[1].text

    def test_groups_small_paragraphs(self):
        content = "One.\n\nTwo.\n\nThree."
        chunks = _chunk_content(content, target_size=1000)
        assert len(chunks) == 1
        assert "One." in chunks[0].text
        assert "Three." in chunks[0].text

    def test_oversized_paragraph_emitted_as_is(self):
        big = "X" * 2000
        chunks = _chunk_content(big, target_size=1000)
        assert len(chunks) == 1
        assert big in chunks[0].text

    def test_empty_content(self):
        chunks = _chunk_content("")
        assert len(chunks) == 1

    def test_section_context_in_chunks(self):
        content = "# Setup\n\n## Ports\n\nPort 8080 is used for the web server."
        chunks = _chunk_content(content, target_size=1000)
        assert len(chunks) >= 1
        # The chunk for "Ports" section should have the heading path
        port_chunk = next(c for c in chunks if "8080" in c.text)
        assert port_chunk.section_path == "Setup > Ports"
        assert "[Setup > Ports]" in port_chunk.text

    def test_overlap_between_chunks(self):
        section_content = "\n\n".join([f"Paragraph {i} " + "x" * 200 for i in range(10)])
        content = f"# Doc\n\n{section_content}"
        chunks = _chunk_content(content, target_size=500, overlap_size=50)
        assert len(chunks) >= 2
        # Second chunk should start with [...] overlap marker
        assert chunks[1].text.count("[...]") >= 1

    def test_list_not_split_from_intro(self):
        content = (
            "# Config\n\n"
            "The following ports are used:\n\n"
            "- 8080: web\n- 443: https\n- 22: ssh\n\n"
            "End of list."
        )
        chunks = _chunk_content(content, target_size=5000)
        # With a large target, everything should be in one chunk
        assert len(chunks) == 1
        assert "ports are used" in chunks[0].text
        assert "8080: web" in chunks[0].text

    def test_multiple_sections_produce_separate_chunks(self):
        content = "# Architecture\n\n" + "A" * 800 + "\n\n# Deployment\n\n" + "B" * 800
        chunks = _chunk_content(content, target_size=1000, overlap_size=0)
        assert len(chunks) == 2
        assert chunks[0].section_path == "Architecture"
        assert chunks[1].section_path == "Deployment"


class TestDocumentParser:
    def test_parse_markdown(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# My Title\n\nSome content here.")

        parser = DocumentParser()
        doc = parser.parse_markdown(md_file, "test-source", tmp_path)

        assert doc["doc_id"] == "test-source:test.md"
        assert "Some content here." in doc["content"]
        assert doc["metadata"]["title"] == "My Title"
        assert doc["metadata"]["source"] == "test-source"
        assert doc["metadata"]["file_path"] == "test.md"

    def test_title_fallback_to_filename(self, tmp_path):
        md_file = tmp_path / "no-heading.md"
        md_file.write_text("Just some text without a heading.")

        parser = DocumentParser()
        doc = parser.parse_markdown(md_file, "src", tmp_path)

        assert doc["metadata"]["title"] == "no-heading"

    def test_nested_path(self, tmp_path):
        nested = tmp_path / "docs" / "sub"
        nested.mkdir(parents=True)
        md_file = nested / "deep.md"
        md_file.write_text("# Deep Doc\n\nNested content.")

        parser = DocumentParser()
        doc = parser.parse_markdown(md_file, "src", tmp_path)

        assert doc["doc_id"] == "src:docs/sub/deep.md"
        assert doc["metadata"]["file_path"] == "docs/sub/deep.md"

    def test_file_size_guard(self, tmp_path: Path) -> None:
        """Files larger than MAX_FILE_SIZE should raise ValueError."""
        big_file = tmp_path / "huge.md"
        big_file.write_bytes(b"x" * (MAX_FILE_SIZE + 1))

        parser = DocumentParser()
        with pytest.raises(ValueError, match="exceeds"):
            parser.parse_markdown(big_file, "src", tmp_path)


class TestRepoManager:
    def test_get_repo_path_local(self, tmp_path: Path) -> None:
        """Local source should return source.path directly."""
        source = RepoSource(name="local", path=str(tmp_path / "myrepo"), is_remote=False)
        manager = RepoManager(source, str(tmp_path / "clones"))
        assert manager.get_repo_path() == Path(source.path)

    def test_get_repo_path_remote(self, tmp_path: Path) -> None:
        """Remote source should return clone_dir / source.name."""
        source = RepoSource(
            name="remote-repo", path="https://example.com/repo.git", is_remote=True
        )
        clone_dir = str(tmp_path / "clones")
        manager = RepoManager(source, clone_dir)
        assert manager.get_repo_path() == Path(clone_dir) / "remote-repo"

    def test_get_files_glob_patterns(self, tmp_path: Path) -> None:
        """get_files should return only files matching configured glob patterns."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "readme.md").write_text("# README")
        (repo_dir / "notes.md").write_text("# Notes")
        (repo_dir / "data.txt").write_text("plain text")
        (repo_dir / "image.png").write_bytes(b"\x89PNG")

        source = RepoSource(name="local", path=str(repo_dir), glob_patterns=["**/*.md"])
        manager = RepoManager(source, str(tmp_path / "clones"))
        files = manager.get_files()

        filenames = {f.name for f in files}
        assert filenames == {"readme.md", "notes.md"}

    def test_get_files_includes_readme_with_custom_patterns(self, tmp_path: Path) -> None:
        """get_files should always include root README.md even with custom patterns."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Project")
        docs_dir = repo_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide")

        source = RepoSource(
            name="custom", path=str(repo_dir), glob_patterns=["docs/**/*.md"]
        )
        manager = RepoManager(source, str(tmp_path / "clones"))
        files = manager.get_files()

        filenames = {f.name for f in files}
        assert "README.md" in filenames
        assert "guide.md" in filenames

    def test_get_files_missing_dir(self, tmp_path: Path) -> None:
        """get_files should return an empty list when path doesn't exist."""
        source = RepoSource(name="missing", path=str(tmp_path / "nonexistent"))
        manager = RepoManager(source, str(tmp_path / "clones"))
        assert manager.get_files() == []

    def test_sync_local_plain_directory(self, tmp_path: Path) -> None:
        """A non-git local directory should return False (no error)."""
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        source = RepoSource(name="plain", path=str(plain_dir), is_remote=False)
        manager = RepoManager(source, str(tmp_path / "clones"))
        assert manager.sync() is False

    @patch("docserver.ingestion.Repo")
    def test_sync_remote_clone(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Remote repo that doesn't exist locally should trigger clone_from."""
        clone_dir = tmp_path / "clones"
        source = RepoSource(
            name="new-remote",
            path="https://example.com/repo.git",
            branch="main",
            is_remote=True,
        )
        manager = RepoManager(source, str(clone_dir))

        # The repo path should not exist yet
        result = manager.sync()

        mock_repo_cls.clone_from.assert_called_once_with(
            source.path,
            clone_dir / "new-remote",
            branch="main",
        )
        assert result is True

    @patch("docserver.ingestion.Repo")
    def test_sync_remote_uses_fetch_and_reset(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Remote sync should use fetch+reset instead of pull to force-overwrite local state."""
        clone_dir = tmp_path / "clones"
        repo_path = clone_dir / "my-remote"
        repo_path.mkdir(parents=True)

        source = RepoSource(
            name="my-remote",
            path="https://example.com/repo.git",
            branch="main",
            is_remote=True,
        )
        manager = RepoManager(source, str(clone_dir))

        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "aaa"
        mock_repo_cls.return_value = mock_repo

        # After reset, simulate a new commit
        def update_head(*args, **kwargs):
            mock_repo.head.commit.hexsha = "bbb"

        mock_repo.head.reset.side_effect = update_head

        result = manager.sync()

        mock_repo.remotes.origin.fetch.assert_called_once()
        mock_repo.head.reset.assert_called_once_with("origin/main", index=True, working_tree=True)
        # pull should NOT be called
        mock_repo.remotes.origin.pull.assert_not_called()
        assert result is True

    @patch("docserver.ingestion.Repo")
    def test_sync_remote_no_changes(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Remote sync should return False when HEAD is unchanged after fetch+reset."""
        clone_dir = tmp_path / "clones"
        repo_path = clone_dir / "my-remote"
        repo_path.mkdir(parents=True)

        source = RepoSource(
            name="my-remote",
            path="https://example.com/repo.git",
            branch="main",
            is_remote=True,
        )
        manager = RepoManager(source, str(clone_dir))

        mock_repo = MagicMock()
        # Same commit before and after
        mock_repo.head.commit.hexsha = "aaa"
        mock_repo_cls.return_value = mock_repo

        result = manager.sync()

        assert result is False

    @patch("docserver.ingestion.Repo")
    def test_sync_remote_fetch_error_returns_false(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Remote sync should return False and not crash when fetch fails."""
        clone_dir = tmp_path / "clones"
        repo_path = clone_dir / "my-remote"
        repo_path.mkdir(parents=True)

        source = RepoSource(
            name="my-remote",
            path="https://example.com/repo.git",
            branch="main",
            is_remote=True,
        )
        manager = RepoManager(source, str(clone_dir))

        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "aaa"
        mock_repo.remotes.origin.fetch.side_effect = RuntimeError("network error")
        mock_repo_cls.return_value = mock_repo

        result = manager.sync()

        assert result is False
        mock_repo.close.assert_called_once()

    @patch("docserver.ingestion.Repo")
    def test_sync_remote_corrupt_head_reclones(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Remote sync should delete clone and re-clone when HEAD has corrupt refs."""
        clone_dir = tmp_path / "clones"
        repo_path = clone_dir / "corrupt-remote"
        repo_path.mkdir(parents=True)
        # Put a marker file so we can verify the directory was deleted
        (repo_path / ".git").mkdir()
        (repo_path / ".git" / "corrupt-ref").write_text("bad")

        source = RepoSource(
            name="corrupt-remote",
            path="https://example.com/repo.git",
            branch="main",
            is_remote=True,
        )
        manager = RepoManager(source, str(clone_dir))

        mock_repo = MagicMock()
        type(mock_repo.head).commit = property(
            lambda self: (_ for _ in ()).throw(ValueError("Invalid reference"))
        )
        mock_repo_cls.return_value = mock_repo

        result = manager.sync()

        # Should have closed the corrupt repo (once in handler, once in finally)
        assert mock_repo.close.call_count >= 1
        mock_repo_cls.clone_from.assert_called_once_with(
            source.path,
            repo_path,
            branch="main",
        )
        assert result is True
        # The corrupt marker file should be gone (directory was nuked)
        assert not (repo_path / ".git" / "corrupt-ref").exists()

    @patch("docserver.ingestion.Repo")
    def test_sync_local_uses_fetch_and_reset(self, mock_repo_cls: MagicMock, tmp_path: Path) -> None:
        """Local git repo sync should use fetch+reset instead of pull."""
        repo_dir = tmp_path / "local-repo"
        repo_dir.mkdir()
        # Make it look like a git repo by letting Repo() succeed
        source = RepoSource(name="local-git", path=str(repo_dir), branch="main", is_remote=False)
        manager = RepoManager(source, str(tmp_path / "clones"))

        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "aaa"
        mock_repo.remotes.__bool__ = lambda self: True
        mock_repo_cls.return_value = mock_repo

        def update_head(*args, **kwargs):
            mock_repo.head.commit.hexsha = "bbb"

        mock_repo.head.reset.side_effect = update_head

        result = manager.sync()

        mock_repo.remotes.origin.fetch.assert_called_once()
        mock_repo.head.reset.assert_called_once_with("origin/main", index=True, working_tree=True)
        mock_repo.remotes.origin.pull.assert_not_called()
        assert result is True


class TestIngester:
    @pytest.fixture
    def kb(self, tmp_path: Path):
        """Create a real KnowledgeBase in a temp directory."""
        from docserver.knowledge_base import KnowledgeBase

        _kb = KnowledgeBase(str(tmp_path / "data"))
        yield _kb
        _kb.close()

    def _make_source_dir(self, tmp_path: Path, name: str, files: dict[str, str]) -> Path:
        """Create a temp directory with markdown files and return its path."""
        source_dir = tmp_path / name
        source_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            filepath = source_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
        return source_dir

    def test_run_once_with_files(self, tmp_path: Path, kb) -> None:
        """run_once should parse markdown files and upsert docs into KB."""
        source_dir = self._make_source_dir(
            tmp_path,
            "repo-a",
            {
                "readme.md": "# Hello\n\nWorld.",
                "guide.md": "# Guide\n\nSome guide content.",
            },
        )
        config = Config(
            sources=[RepoSource(name="repo-a", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once()

        assert "repo-a" in stats
        # 2 parent docs + at least 2 chunks = at least 4 upserts
        assert stats["repo-a"]["upserted"] >= 4
        assert stats["repo-a"]["deleted"] == 0

        # Verify docs exist in KB
        ids = kb.get_all_doc_ids_for_source("repo-a")
        assert "repo-a:readme.md" in ids
        assert "repo-a:guide.md" in ids

    def test_run_once_source_filter(self, tmp_path: Path, kb) -> None:
        """run_once(sources=["source-a"]) should only process source-a."""
        dir_a = self._make_source_dir(tmp_path, "source-a", {"a.md": "# A\n\nContent A."})
        dir_b = self._make_source_dir(tmp_path, "source-b", {"b.md": "# B\n\nContent B."})

        config = Config(
            sources=[
                RepoSource(name="source-a", path=str(dir_a)),
                RepoSource(name="source-b", path=str(dir_b)),
            ],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once(sources=["source-a"])

        assert "source-a" in stats
        assert "source-b" not in stats
        assert stats["source-a"]["upserted"] >= 2  # parent + at least 1 chunk

        # source-b should have nothing in KB
        assert kb.get_all_doc_ids_for_source("source-b") == set()

    def test_run_once_stale_doc_cleanup(self, tmp_path: Path, kb) -> None:
        """Docs no longer present in the source should be deleted on next run."""
        source_dir = self._make_source_dir(
            tmp_path,
            "cleanup-src",
            {
                "keep.md": "# Keep\n\nStays.",
                "remove.md": "# Remove\n\nGoes away.",
            },
        )
        config = Config(
            sources=[RepoSource(name="cleanup-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)

        # First run: both files ingested
        stats1 = ingester.run_once()
        assert stats1["cleanup-src"]["upserted"] >= 4

        ids_before = kb.get_all_doc_ids_for_source("cleanup-src")
        assert "cleanup-src:remove.md" in ids_before

        # Delete the file from disk
        (source_dir / "remove.md").unlink()

        # Second run: stale docs should be cleaned up
        stats2 = ingester.run_once()
        assert stats2["cleanup-src"]["deleted"] >= 1  # parent + chunks

        ids_after = kb.get_all_doc_ids_for_source("cleanup-src")
        assert "cleanup-src:remove.md" not in ids_after
        assert "cleanup-src:keep.md" in ids_after

    def test_run_once_skips_unchanged_files(self, tmp_path: Path, kb) -> None:
        """Unchanged files should be skipped on the second run."""
        source_dir = self._make_source_dir(
            tmp_path,
            "skip-src",
            {
                "a.md": "# File A\n\nContent A.",
                "b.md": "# File B\n\nContent B.",
            },
        )
        config = Config(
            sources=[RepoSource(name="skip-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)

        # First run: everything ingested
        stats1 = ingester.run_once()
        assert stats1["skip-src"]["upserted"] >= 4
        assert stats1["skip-src"]["skipped"] == 0

        # Second run without changes: everything skipped
        stats2 = ingester.run_once()
        assert stats2["skip-src"]["upserted"] == 0
        assert stats2["skip-src"]["skipped"] == 2
        assert stats2["skip-src"]["deleted"] == 0

        # Verify docs still exist
        ids = kb.get_all_doc_ids_for_source("skip-src")
        assert "skip-src:a.md" in ids
        assert "skip-src:b.md" in ids

    def test_run_once_reindexes_modified_files(self, tmp_path: Path, kb) -> None:
        """Modified files should be re-indexed, unchanged files skipped."""
        import time

        source_dir = self._make_source_dir(
            tmp_path,
            "mod-src",
            {
                "stable.md": "# Stable\n\nUnchanged.",
                "changing.md": "# Changing\n\nOriginal.",
            },
        )
        config = Config(
            sources=[RepoSource(name="mod-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)

        # First run
        ingester.run_once()

        # Modify one file (need a different mtime)
        time.sleep(0.05)
        (source_dir / "changing.md").write_text("# Changing\n\nUpdated content.")

        # Second run: only changing.md should be re-indexed
        stats2 = ingester.run_once()
        assert stats2["mod-src"]["skipped"] == 1  # stable.md
        assert stats2["mod-src"]["upserted"] >= 2  # changing.md parent + chunk(s)

    def test_run_once_first_run_all_new(self, tmp_path: Path, kb) -> None:
        """First run should report all files as 'new'."""
        source_dir = self._make_source_dir(
            tmp_path,
            "new-src",
            {
                "a.md": "# A\n\nContent.",
                "b.md": "# B\n\nContent.",
                "c.md": "# C\n\nContent.",
            },
        )
        config = Config(
            sources=[RepoSource(name="new-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once()

        assert stats["new-src"]["new"] == 3
        assert stats["new-src"]["modified"] == 0
        assert stats["new-src"]["skipped"] == 0

    def test_run_once_modified_file_counted_as_modified(self, tmp_path: Path, kb) -> None:
        """A changed file on the second run should be counted as 'modified', not 'new'."""
        import time

        source_dir = self._make_source_dir(
            tmp_path,
            "mod-count-src",
            {"a.md": "# A\n\nOriginal."},
        )
        config = Config(
            sources=[RepoSource(name="mod-count-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats1 = ingester.run_once()
        assert stats1["mod-count-src"]["new"] == 1

        time.sleep(0.05)
        (source_dir / "a.md").write_text("# A\n\nUpdated.")

        stats2 = ingester.run_once()
        assert stats2["mod-count-src"]["new"] == 0
        assert stats2["mod-count-src"]["modified"] == 1

    def test_run_once_empty_source(self, tmp_path: Path, kb) -> None:
        """A source with no matching files should complete without errors."""
        source_dir = tmp_path / "empty-src"
        source_dir.mkdir()

        config = Config(
            sources=[RepoSource(name="empty-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once()

        assert stats["empty-src"]["files"] == 0
        assert stats["empty-src"]["upserted"] == 0
        assert stats["empty-src"]["errors"] == 0

    def test_run_once_add_file_after_initial_index(self, tmp_path: Path, kb) -> None:
        """Adding a file to a previously-indexed source should index only the new file."""
        source_dir = self._make_source_dir(
            tmp_path,
            "add-src",
            {"existing.md": "# Existing\n\nAlready here."},
        )
        config = Config(
            sources=[RepoSource(name="add-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        # Add a new file
        (source_dir / "brand_new.md").write_text("# Brand New\n\nJust added.")

        stats2 = ingester.run_once()
        assert stats2["add-src"]["new"] == 1
        assert stats2["add-src"]["skipped"] == 1  # existing.md
        assert stats2["add-src"]["deleted"] == 0

        ids = kb.get_all_doc_ids_for_source("add-src")
        assert "add-src:brand_new.md" in ids
        assert "add-src:existing.md" in ids

    def test_run_once_delete_and_modify_same_cycle(self, tmp_path: Path, kb) -> None:
        """Deleting one file and modifying another in the same cycle."""
        import time

        source_dir = self._make_source_dir(
            tmp_path,
            "mixed-src",
            {
                "keep.md": "# Keep\n\nStays.",
                "remove.md": "# Remove\n\nGone.",
                "change.md": "# Change\n\nOriginal.",
            },
        )
        config = Config(
            sources=[RepoSource(name="mixed-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        # Delete one, modify another
        (source_dir / "remove.md").unlink()
        time.sleep(0.05)
        (source_dir / "change.md").write_text("# Change\n\nModified.")

        stats2 = ingester.run_once()
        assert stats2["mixed-src"]["modified"] == 1  # change.md
        assert stats2["mixed-src"]["skipped"] == 1  # keep.md
        assert stats2["mixed-src"]["deleted"] >= 1  # remove.md + its chunks

        ids = kb.get_all_doc_ids_for_source("mixed-src")
        assert "mixed-src:keep.md" in ids
        assert "mixed-src:change.md" in ids
        assert "mixed-src:remove.md" not in ids

    def test_run_once_skips_large_files(self, tmp_path: Path, kb) -> None:
        """Files exceeding MAX_FILE_SIZE should be skipped."""
        source_dir = self._make_source_dir(
            tmp_path,
            "big-src",
            {
                "small.md": "# Small\n\nOK.",
            },
        )
        # Create an oversized file
        big_file = source_dir / "huge.md"
        big_file.write_bytes(b"# Huge\n\n" + b"x" * (MAX_FILE_SIZE + 1))

        config = Config(
            sources=[RepoSource(name="big-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once()

        # Only small.md should have been upserted (parent + chunk(s))
        assert stats["big-src"]["upserted"] >= 2

        ids = kb.get_all_doc_ids_for_source("big-src")
        assert "big-src:small.md" in ids
        # The huge file should not be in KB
        assert not any("huge.md" in doc_id for doc_id in ids)

    def test_skip_uses_content_hash_not_mtime(self, tmp_path: Path, kb) -> None:
        """Files should be skipped based on content hash, not mtime.

        This simulates a fresh clone where mtimes change but content is identical.
        """
        source_dir = self._make_source_dir(
            tmp_path,
            "hash-src",
            {"a.md": "# File A\n\nContent A."},
        )
        config = Config(
            sources=[RepoSource(name="hash-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)

        # First run: indexes everything
        stats1 = ingester.run_once()
        assert stats1["hash-src"]["new"] == 1

        # Rewrite file with identical content (changes mtime)
        import time
        time.sleep(0.05)
        (source_dir / "a.md").write_text("# File A\n\nContent A.")

        # Second run: should skip because content hash is the same
        stats2 = ingester.run_once()
        assert stats2["hash-src"]["skipped"] == 1
        assert stats2["hash-src"]["upserted"] == 0

    def test_content_hash_detects_modified_content(self, tmp_path: Path, kb) -> None:
        """Changed content should be detected even if we use content hashing."""
        source_dir = self._make_source_dir(
            tmp_path,
            "hashmod-src",
            {"a.md": "# File A\n\nOriginal content."},
        )
        config = Config(
            sources=[RepoSource(name="hashmod-src", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        # Change the content
        (source_dir / "a.md").write_text("# File A\n\nModified content.")

        stats2 = ingester.run_once()
        assert stats2["hashmod-src"]["modified"] == 1
        assert stats2["hashmod-src"]["skipped"] == 0

    def test_cleanup_orphaned_sources(self, tmp_path: Path, kb) -> None:
        """Orphaned sources should be cleaned up from KB and clone dirs."""
        # Set up two sources and ingest them
        dir_a = self._make_source_dir(tmp_path, "keep-src", {"a.md": "# A\n\nKeep."})
        dir_b = self._make_source_dir(tmp_path, "remove-src", {"b.md": "# B\n\nRemove."})

        data_dir = str(tmp_path / "data")
        config = Config(
            sources=[
                RepoSource(name="keep-src", path=str(dir_a)),
                RepoSource(name="remove-src", path=str(dir_b)),
            ],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        assert kb.get_all_doc_ids_for_source("remove-src") != set()

        # Create a fake clone dir for the removed source
        clone_dir = Path(data_dir) / "clones" / "remove-src"
        clone_dir.mkdir(parents=True, exist_ok=True)
        (clone_dir / "marker.txt").write_text("exists")

        # Now create a new config without "remove-src"
        config2 = Config(
            sources=[RepoSource(name="keep-src", path=str(dir_a))],
            data_dir=data_dir,
        )
        ingester2 = Ingester(config2, kb)
        result = ingester2.cleanup_orphaned_sources()

        assert "remove-src" in result
        assert result["remove-src"] >= 1
        assert kb.get_all_doc_ids_for_source("remove-src") == set()
        assert not clone_dir.exists()

        # keep-src should be untouched
        assert kb.get_all_doc_ids_for_source("keep-src") != set()

    def test_orphan_cleanup_runs_on_full_ingestion(self, tmp_path: Path, kb) -> None:
        """Orphan cleanup should run automatically at the start of a full ingestion."""
        dir_a = self._make_source_dir(tmp_path, "alive-src", {"a.md": "# A\n\nAlive."})
        dir_b = self._make_source_dir(tmp_path, "dead-src", {"b.md": "# B\n\nDead."})

        data_dir = str(tmp_path / "data")
        config = Config(
            sources=[
                RepoSource(name="alive-src", path=str(dir_a)),
                RepoSource(name="dead-src", path=str(dir_b)),
            ],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        assert kb.get_all_doc_ids_for_source("dead-src") != set()

        # Remove dead-src from config and run full ingestion
        config2 = Config(
            sources=[RepoSource(name="alive-src", path=str(dir_a))],
            data_dir=data_dir,
        )
        ingester2 = Ingester(config2, kb)
        ingester2.run_once()  # Full run (no sources filter) triggers cleanup

        assert kb.get_all_doc_ids_for_source("dead-src") == set()

    def test_orphan_cleanup_skipped_for_filtered_run(self, tmp_path: Path, kb) -> None:
        """Orphan cleanup should NOT run when specific sources are targeted."""
        dir_a = self._make_source_dir(tmp_path, "src-a", {"a.md": "# A\n\nA."})
        dir_b = self._make_source_dir(tmp_path, "src-b", {"b.md": "# B\n\nB."})

        data_dir = str(tmp_path / "data")
        config = Config(
            sources=[
                RepoSource(name="src-a", path=str(dir_a)),
                RepoSource(name="src-b", path=str(dir_b)),
            ],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        ingester.run_once()

        # Remove src-b from config but run filtered to src-a only
        config2 = Config(
            sources=[RepoSource(name="src-a", path=str(dir_a))],
            data_dir=data_dir,
        )
        ingester2 = Ingester(config2, kb)
        ingester2.run_once(sources=["src-a"])

        # src-b should still be in KB because cleanup was skipped
        assert kb.get_all_doc_ids_for_source("src-b") != set()

    def test_rename_detection_migrates_source(self, tmp_path: Path, kb) -> None:
        """Renaming a source in config should migrate data instead of delete + re-index."""
        from git import Repo

        data_dir = str(tmp_path / "data")
        clone_dir = Path(data_dir) / "clones"
        repo_url = "https://github.com/example/repo.git"

        # Create a fake clone directory with a git remote matching the URL
        old_clone = clone_dir / "old-name"
        old_clone.mkdir(parents=True)
        repo = Repo.init(old_clone)
        repo.create_remote("origin", repo_url)
        repo.close()

        # Seed KB with docs under old name
        kb.upsert_document(
            "old-name:readme.md",
            "",
            {"source": "old-name", "file_path": "readme.md", "title": "README", "is_chunk": False,
             "content_hash": "abc123"},
        )
        kb.upsert_document(
            "old-name:readme.md#chunk0",
            "Project documentation content.",
            {
                "source": "old-name",
                "file_path": "readme.md",
                "title": "README",
                "chunk_index": 0,
                "total_chunks": 1,
                "is_chunk": True,
            },
        )

        # Create config with renamed source pointing to same URL
        config = Config(
            sources=[
                RepoSource(name="new-name", path=repo_url, is_remote=True),
            ],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        result = ingester.cleanup_orphaned_sources()

        # Should have migrated, not deleted
        assert "old-name" in result
        assert result["old-name"] == 2  # parent + chunk

        # Data should exist under new name
        assert kb.get_document("new-name:readme.md") is not None
        assert kb.get_document("new-name:readme.md#chunk0") is not None
        assert kb.get_document("new-name:readme.md")["source"] == "new-name"

        # Old name should be gone
        assert kb.get_all_doc_ids_for_source("old-name") == set()

        # Clone dir should have been renamed
        assert not old_clone.exists()
        assert (clone_dir / "new-name").exists()

    def test_rename_detection_url_normalisation(self, tmp_path: Path, kb) -> None:
        """Rename detection should work despite URL differences (.git suffix, credentials)."""
        from git import Repo

        data_dir = str(tmp_path / "data")
        clone_dir = Path(data_dir) / "clones"

        # Clone dir has URL without .git suffix
        old_clone = clone_dir / "old-src"
        old_clone.mkdir(parents=True)
        repo = Repo.init(old_clone)
        repo.create_remote("origin", "https://github.com/user/repo")
        repo.close()

        kb.upsert_document(
            "old-src:doc.md", "", {"source": "old-src", "file_path": "doc.md", "is_chunk": False}
        )

        # Config has URL with .git suffix and credentials
        config = Config(
            sources=[
                RepoSource(
                    name="new-src",
                    path="https://token@github.com/User/Repo.git",
                    is_remote=True,
                ),
            ],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        result = ingester.cleanup_orphaned_sources()

        assert "old-src" in result
        assert kb.get_document("new-src:doc.md") is not None

    def test_no_false_rename_when_new_source_has_data(self, tmp_path: Path, kb) -> None:
        """Don't rename if the new source name already has data in the KB."""
        from git import Repo

        data_dir = str(tmp_path / "data")
        clone_dir = Path(data_dir) / "clones"
        repo_url = "https://github.com/example/repo.git"

        # Old clone dir
        old_clone = clone_dir / "old-src"
        old_clone.mkdir(parents=True)
        repo = Repo.init(old_clone)
        repo.create_remote("origin", repo_url)
        repo.close()

        # Both old and new names have data
        kb.upsert_document(
            "old-src:a.md", "", {"source": "old-src", "file_path": "a.md", "is_chunk": False}
        )
        kb.upsert_document(
            "new-src:b.md", "", {"source": "new-src", "file_path": "b.md", "is_chunk": False}
        )

        config = Config(
            sources=[RepoSource(name="new-src", path=repo_url, is_remote=True)],
            data_dir=data_dir,
        )
        ingester = Ingester(config, kb)
        result = ingester.cleanup_orphaned_sources()

        # Should have deleted old-src (not renamed), since new-src already has data
        assert "old-src" in result
        assert kb.get_all_doc_ids_for_source("old-src") == set()
        # new-src's existing data should be untouched
        assert kb.get_document("new-src:b.md") is not None


class TestNormaliseRepoUrl:
    def test_strips_git_suffix(self):
        assert _normalise_repo_url("https://github.com/user/repo.git") == "https://github.com/user/repo"

    def test_strips_credentials(self):
        assert _normalise_repo_url("https://token@github.com/user/repo") == "https://github.com/user/repo"

    def test_strips_full_credentials(self):
        assert _normalise_repo_url("https://user:pass@github.com/user/repo.git") == "https://github.com/user/repo"

    def test_strips_trailing_slash(self):
        assert _normalise_repo_url("https://github.com/user/repo/") == "https://github.com/user/repo"

    def test_case_insensitive(self):
        assert _normalise_repo_url("https://GitHub.com/User/Repo") == "https://github.com/user/repo"

    def test_combined(self):
        assert (
            _normalise_repo_url("https://ghp_token@GitHub.com/User/Repo.git/")
            == "https://github.com/user/repo"
        )


# ---------------------------------------------------------------------------
# Bulk git created_at tests
# ---------------------------------------------------------------------------


class TestBulkGitCreatedAt:
    """Tests for DocumentParser._bulk_git_created_at."""

    def test_bulk_returns_dates_for_tracked_files(self, tmp_path):
        """Bulk lookup returns creation dates for files in a git repo."""
        import subprocess

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo_dir), capture_output=True, check=True)

        f1 = repo_dir / "one.md"
        f1.write_text("# One")
        subprocess.run(["git", "add", "one.md"], cwd=str(repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "add one"], cwd=str(repo_dir), capture_output=True, check=True)

        f2 = repo_dir / "two.md"
        f2.write_text("# Two")
        subprocess.run(["git", "add", "two.md"], cwd=str(repo_dir), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "add two"], cwd=str(repo_dir), capture_output=True, check=True)

        result = DocumentParser._bulk_git_created_at([f1, f2], repo_dir)

        assert f1 in result
        assert f2 in result
        assert result[f1] is not None
        assert result[f2] is not None

    def test_bulk_returns_none_for_non_git_dir(self, tmp_path):
        """Bulk lookup gracefully handles non-git directories."""
        f1 = tmp_path / "file.md"
        f1.write_text("hello")

        result = DocumentParser._bulk_git_created_at([f1], tmp_path)
        # Should fall back to per-file which also returns None for non-git
        assert f1 in result
        assert result[f1] is None

    def test_bulk_returns_empty_for_empty_list(self, tmp_path):
        result = DocumentParser._bulk_git_created_at([], tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Batch ingestion integration tests
# ---------------------------------------------------------------------------


class TestBatchIngestion:
    """Tests that batch upserts work correctly through the full ingestion pipeline."""

    def _make_source_dir(self, tmp_path, name, files):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            p = d / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return d

    def test_batch_ingestion_produces_same_results(self, tmp_path):
        """Batch ingestion indexes all files correctly."""
        kb = KnowledgeBase(str(tmp_path / "data"))

        source_dir = self._make_source_dir(tmp_path, "myrepo", {
            "docs/setup.md": "# Setup\n\nHow to set up the server.",
            "docs/deploy.md": "# Deploy\n\nDeployment instructions for production.",
        })

        config = Config(
            sources=[RepoSource(name="myrepo", path=str(source_dir))],
            data_dir=str(tmp_path / "data"),
        )
        ingester = Ingester(config, kb)
        stats = ingester.run_once()

        assert stats["myrepo"]["files"] == 2
        assert stats["myrepo"]["errors"] == 0
        assert stats["myrepo"]["new"] == 2

        # Both parent docs should exist.
        assert kb.get_document("myrepo:docs/setup.md") is not None
        assert kb.get_document("myrepo:docs/deploy.md") is not None

        # Chunks should be searchable.
        results = kb.search("deployment production")
        assert len(results) >= 1

        kb.close()
