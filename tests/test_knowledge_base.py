"""Tests for the knowledge base module."""

from unittest.mock import patch

import pytest

from docserver.knowledge_base import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    _kb = KnowledgeBase(str(tmp_path / "data"))
    yield _kb
    _kb.close()


def test_upsert_and_get_document(kb):
    kb.upsert_document(
        "src:readme.md",
        "",
        {
            "source": "src",
            "file_path": "readme.md",
            "title": "README",
            "is_chunk": False,
        },
    )

    doc = kb.get_document("src:readme.md")
    assert doc is not None
    assert doc["title"] == "README"
    assert doc["source"] == "src"


def test_upgrades_from_pre_stage2_schema(tmp_path):
    """Regression: a legacy documents.db without a type column must upgrade cleanly.

    Reproduces the production deploy bug where _SCHEMA's idx_documents_type
    CREATE INDEX ran before _MIGRATIONS' ALTER TABLE ADD COLUMN type — the
    index referenced a column that didn't exist yet, executescript aborted,
    and the migration never got a chance to fire. The fix moves the index
    creation into _MIGRATIONS so it lands after the ALTER.
    """
    import sqlite3

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "documents.db"

    # Seed a pre-Stage-2 schema by hand — same columns as before the type
    # column landed. No type column, no idx_documents_type, no meta table.
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                doc_id        TEXT PRIMARY KEY,
                source        TEXT NOT NULL,
                file_path     TEXT NOT NULL,
                title         TEXT,
                content       TEXT,
                chunk_index   INTEGER,
                total_chunks  INTEGER,
                created_at    TEXT,
                modified_at   TEXT,
                indexed_at    TEXT,
                size_bytes    INTEGER,
                is_chunk      BOOLEAN DEFAULT FALSE,
                section_path  TEXT DEFAULT ''
            );
            INSERT INTO documents (doc_id, source, file_path, title, is_chunk)
            VALUES ('src:legacy.md', 'src', 'legacy.md', 'Legacy', 0);
            """
        )

    # Constructing a fresh KB against this DB must complete without raising
    # — the migration runs first to add the column, then the index creation
    # finds it.
    _kb = KnowledgeBase(str(data_dir))
    try:
        doc = _kb.get_document("src:legacy.md")
        assert doc is not None
        # Existing row gets the column default.
        assert doc["type"] == "documentation"
    finally:
        _kb.close()


def test_default_type_is_documentation(kb):
    """Stage 2 W2.1: documents land with type='documentation' when none supplied."""
    kb.upsert_document(
        "src:readme.md",
        "",
        {"source": "src", "file_path": "readme.md", "title": "README", "is_chunk": False},
    )

    doc = kb.get_document("src:readme.md")
    assert doc is not None
    assert doc["type"] == "documentation"


def test_get_source_files_includes_type(kb):
    """Stage 2 W2.5: tree endpoint surfaces type so the webapp can render badges."""
    kb.upsert_document(
        "src:journal/today.md",
        "",
        {
            "source": "src",
            "file_path": "journal/today.md",
            "title": "Journal",
            "is_chunk": False,
            "type": "journal",
        },
    )
    rows = kb.get_source_files("src")
    assert len(rows) == 1
    assert rows[0]["type"] == "journal"


def test_query_documents_exclude_types(kb):
    """Stage 2 W2.4: query_documents filters out rows whose type is excluded."""
    kb.upsert_document(
        "src:journal/today.md",
        "",
        {
            "source": "src",
            "file_path": "journal/today.md",
            "title": "Journal",
            "is_chunk": False,
            "type": "journal",
        },
    )
    kb.upsert_document(
        "src:README.md",
        "",
        {
            "source": "src",
            "file_path": "README.md",
            "title": "README",
            "is_chunk": False,
            "type": "documentation",
        },
    )

    docs = kb.query_documents(source="src", exclude_types=["journal"])
    titles = {d["title"] for d in docs}
    assert titles == {"README"}

    # Sanity: with no exclude, both are returned.
    all_docs = kb.query_documents(source="src")
    assert {d["title"] for d in all_docs} == {"Journal", "README"}


def test_search_exclude_types_bm25(kb):
    """Stage 2 W2.4: BM25 JOIN to documents respects exclude_types."""
    kb.upsert_document(
        "src:journal/note.md#chunk0",
        "Strava activity sync notes for today's ride",
        {
            "source": "src",
            "file_path": "journal/note.md",
            "title": "Journal",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
            "type": "journal",
        },
    )
    kb.upsert_document(
        "src:docs/strava.md#chunk0",
        "Strava integration documentation and setup",
        {
            "source": "src",
            "file_path": "docs/strava.md",
            "title": "Strava Docs",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
            "type": "documentation",
        },
    )

    # Excluding 'journal' should leave only the docs chunk.
    results = kb._bm25_search_chunks("strava", exclude_types=["journal"])
    doc_ids = [r["doc_id"] for r in results]
    assert "src:docs/strava.md#chunk0" in doc_ids
    assert "src:journal/note.md#chunk0" not in doc_ids

    # Without the filter both should appear.
    all_results = kb._bm25_search_chunks("strava")
    all_ids = {r["doc_id"] for r in all_results}
    assert all_ids == {"src:docs/strava.md#chunk0", "src:journal/note.md#chunk0"}


def test_search_exclude_types_dense(kb):
    """Stage 2 W2.4: Chroma where uses $nin on type."""
    kb.upsert_document(
        "src:notes.md#chunk0",
        "Quick reminder about laundry pickup",
        {
            "source": "src",
            "file_path": "notes.md",
            "title": "Notes",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
            "type": "not-docs",
        },
    )
    kb.upsert_document(
        "src:guide.md#chunk0",
        "Quick reference guide for laundry equipment",
        {
            "source": "src",
            "file_path": "guide.md",
            "title": "Guide",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
            "type": "documentation",
        },
    )

    results = kb._dense_search("laundry", exclude_types=["not-docs"])
    doc_ids = {r["doc_id"] for r in results}
    assert "src:notes.md#chunk0" not in doc_ids
    assert "src:guide.md#chunk0" in doc_ids


def test_backfill_types_runs_when_hash_changes(kb, tmp_path):
    """Stage 2 W2.3: conditional backfill reclassifies on config change, not on restart.

    Sequence:
      1. Seed a few docs (default type 'documentation').
      2. Write a document-types.yml that promotes journal/* to 'journal'.
      3. First backfill reclassifies; second backfill no-ops; edited config
         triggers another reclassification.
    """
    from docserver.config import load_doc_types_config

    kb.upsert_document(
        "src:journal/2026-05-13.md",
        "",
        {
            "source": "src",
            "file_path": "journal/2026-05-13.md",
            "title": "Today",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "src:README.md",
        "",
        {"source": "src", "file_path": "README.md", "title": "README", "is_chunk": False},
    )

    config_path = tmp_path / "document-types.yml"
    config_path.write_text(
        """
types: [documentation, journal]
fallback_type: documentation
global_rules:
  - pattern: "**/journal/**"
    type: journal
"""
    )
    cfg = load_doc_types_config(str(config_path))

    # First run: stored hash absent, so backfill fires and reclassifies the journal doc.
    updated = kb.backfill_types_if_needed(cfg, str(config_path))
    assert updated == 1
    journal_doc = kb.get_document("src:journal/2026-05-13.md")
    readme_doc = kb.get_document("src:README.md")
    assert journal_doc is not None
    assert readme_doc is not None
    assert journal_doc["type"] == "journal"
    assert readme_doc["type"] == "documentation"

    # Second run with the same config: hash matches, no work.
    updated_again = kb.backfill_types_if_needed(cfg, str(config_path))
    assert updated_again == 0

    # Edit the config so the readme is also classified as 'journal' — new hash,
    # backfill must run again.
    config_path.write_text(
        """
types: [documentation, journal]
fallback_type: journal
global_rules:
  - pattern: "**/journal/**"
    type: journal
"""
    )
    cfg2 = load_doc_types_config(str(config_path))
    updated_after_edit = kb.backfill_types_if_needed(cfg2, str(config_path))
    assert updated_after_edit == 1
    readme_doc_after = kb.get_document("src:README.md")
    assert readme_doc_after is not None
    assert readme_doc_after["type"] == "journal"


def test_backfill_reclassifies_when_legacy_bare_sha_stored(kb, tmp_path):
    """Legacy ``meta.doc_types_hash`` values (bare SHA256, no version prefix)
    must trigger one more backfill so deployments built before the defaults
    change pick up the new baked-in rules on next startup.
    """
    import sqlite3

    from docserver.config import load_doc_types_config

    # Seed a doc whose path the (legacy) default rules wouldn't classify, but
    # which the new YAML does.
    kb.upsert_document(
        "src:journal/2026-05-18.md",
        "",
        {
            "source": "src",
            "file_path": "journal/2026-05-18.md",
            "title": "Today",
            "is_chunk": False,
        },
    )

    config_path = tmp_path / "document-types.yml"
    config_path.write_text(
        """
types: [documentation, journal]
fallback_type: documentation
global_rules:
  - pattern: "**/journal/**"
    type: journal
"""
    )
    cfg = load_doc_types_config(str(config_path))

    # Simulate a deployment that already ran a previous version of the backfill
    # and stored only the bare sha256 of the config file (no version prefix).
    import hashlib

    bare_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    with sqlite3.connect(kb._db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("doc_types_hash", bare_sha),
        )

    # First run after the upgrade: the combined hash differs from the bare
    # legacy hash, so backfill must run.
    updated = kb.backfill_types_if_needed(cfg, str(config_path))
    assert updated == 1
    journal_doc = kb.get_document("src:journal/2026-05-18.md")
    assert journal_doc is not None
    assert journal_doc["type"] == "journal"

    # Second run with the same config short-circuits: the combined hash is
    # now stored.
    updated_again = kb.backfill_types_if_needed(cfg, str(config_path))
    assert updated_again == 0


def test_backfill_reclassifies_when_no_config_after_legacy_empty_hash(kb, tmp_path):
    """The missing-config path also forces one reclassification on upgrade.

    Pre-upgrade deployments without ``document-types.yml`` stored an empty
    string as the cache key. The new combined-hash form is ``v2:`` (version
    prefix only, empty sha), which differs from ``""`` — so the backfill
    fires once and applies the baked-in defaults.
    """
    import sqlite3

    kb.upsert_document(
        "src:journal/2026-05-18.md",
        "",
        {
            "source": "src",
            "file_path": "journal/2026-05-18.md",
            "title": "Today",
            "is_chunk": False,
        },
    )

    # Seed the legacy empty-string sentinel.
    with sqlite3.connect(kb._db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("doc_types_hash", ""),
        )

    from docserver.config import DocTypesConfig

    missing_path = str(tmp_path / "absent.yml")
    cfg = DocTypesConfig()  # baked-in defaults

    updated = kb.backfill_types_if_needed(cfg, missing_path)
    assert updated == 1
    journal_doc = kb.get_document("src:journal/2026-05-18.md")
    assert journal_doc is not None
    assert journal_doc["type"] == "journal"

    # And idempotent on the second call.
    assert kb.backfill_types_if_needed(cfg, missing_path) == 0


def test_meta_table_exists(kb):
    """Stage 2 W2.1: meta key/value table is available for W2.3's hash bookkeeping."""
    import sqlite3

    with sqlite3.connect(kb._db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("doc_types_hash", "abc123"),
        )
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", ("doc_types_hash",)
        ).fetchone()
    assert row is not None
    assert row[0] == "abc123"


def test_upsert_chunk_and_search(kb):
    kb.upsert_document(
        "src:readme.md#chunk0",
        "This document describes the home server architecture and network setup.",
        {
            "source": "src",
            "file_path": "readme.md",
            "title": "README",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )

    results = kb.search("home server network")
    assert len(results) >= 1
    assert "home server" in results[0]["content"].lower()


def test_delete_document(kb):
    kb.upsert_document(
        "src:test.md",
        "",
        {"source": "src", "file_path": "test.md", "is_chunk": False},
    )
    kb.delete_document("src:test.md")
    assert kb.get_document("src:test.md") is None


def test_get_all_doc_ids_for_source(kb):
    kb.upsert_document("a:one.md", "", {"source": "a", "file_path": "one.md", "is_chunk": False})
    kb.upsert_document("a:two.md", "", {"source": "a", "file_path": "two.md", "is_chunk": False})
    kb.upsert_document(
        "b:three.md", "", {"source": "b", "file_path": "three.md", "is_chunk": False}
    )

    ids = kb.get_all_doc_ids_for_source("a")
    assert ids == {"a:one.md", "a:two.md"}


def test_query_documents_filters_chunks(kb):
    kb.upsert_document(
        "src:doc.md",
        "",
        {"source": "src", "file_path": "doc.md", "title": "Doc", "is_chunk": False},
    )
    kb.upsert_document(
        "src:doc.md#chunk0",
        "content",
        {
            "source": "src",
            "file_path": "doc.md",
            "title": "Doc",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )

    docs = kb.query_documents(source="src")
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "src:doc.md"


def test_get_sources_summary(kb):
    kb.upsert_document("src:a.md", "", {"source": "src", "file_path": "a.md", "is_chunk": False})
    kb.upsert_document(
        "src:a.md#chunk0",
        "text",
        {"source": "src", "file_path": "a.md", "chunk_index": 0, "is_chunk": True},
    )

    summary = kb.get_sources_summary()
    assert len(summary) == 1
    assert summary[0]["source"] == "src"
    assert summary[0]["file_count"] == 1
    assert summary[0]["chunk_count"] == 1


def test_get_source_files_returns_only_parents(kb):
    kb.upsert_document(
        "src:a.md",
        "",
        {"source": "src", "file_path": "a.md", "title": "A", "is_chunk": False},
    )
    kb.upsert_document(
        "src:a.md#chunk0",
        "text",
        {
            "source": "src",
            "file_path": "a.md",
            "title": "A",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )

    files = kb.get_source_files("src")
    assert [f["doc_id"] for f in files] == ["src:a.md"]
    assert files[0]["file_path"] == "a.md"
    assert files[0]["title"] == "A"


def test_get_source_files_filters_by_source(kb):
    kb.upsert_document(
        "a:one.md", "", {"source": "a", "file_path": "one.md", "is_chunk": False},
    )
    kb.upsert_document(
        "a:two.md", "", {"source": "a", "file_path": "two.md", "is_chunk": False},
    )
    kb.upsert_document(
        "b:three.md",
        "",
        {"source": "b", "file_path": "three.md", "is_chunk": False},
    )

    files = kb.get_source_files("a")
    assert [f["file_path"] for f in files] == ["one.md", "two.md"]

    files_b = kb.get_source_files("b")
    assert [f["file_path"] for f in files_b] == ["three.md"]

    files_missing = kb.get_source_files("nonexistent")
    assert files_missing == []


def test_get_source_files_ordered_by_file_path(kb):
    # Insert out of order; result should be sorted alphabetically by file_path.
    kb.upsert_document(
        "src:zeta.md", "", {"source": "src", "file_path": "zeta.md", "is_chunk": False},
    )
    kb.upsert_document(
        "src:alpha.md", "", {"source": "src", "file_path": "alpha.md", "is_chunk": False},
    )
    kb.upsert_document(
        "src:docs/nested.md",
        "",
        {"source": "src", "file_path": "docs/nested.md", "is_chunk": False},
    )

    files = kb.get_source_files("src")
    assert [f["file_path"] for f in files] == ["alpha.md", "docs/nested.md", "zeta.md"]


def test_search_with_source_filter(kb):
    kb.upsert_document(
        "a:doc.md#chunk0",
        "Python programming guide",
        {
            "source": "a",
            "file_path": "doc.md",
            "title": "Guide",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )
    kb.upsert_document(
        "b:doc.md#chunk0",
        "Python programming tutorial",
        {
            "source": "b",
            "file_path": "doc.md",
            "title": "Tutorial",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )

    results = kb.search("python programming", source_filter="a")
    assert all(r["metadata"]["source"] == "a" for r in results)


def test_query_documents_file_path_contains(kb):
    """query_documents should filter by substring in file_path."""
    kb.upsert_document(
        "src:networking/ports.md",
        "",
        {"source": "src", "file_path": "networking/ports.md", "title": "Ports", "is_chunk": False},
    )
    kb.upsert_document(
        "src:storage/disks.md",
        "",
        {"source": "src", "file_path": "storage/disks.md", "title": "Disks", "is_chunk": False},
    )

    docs = kb.query_documents(file_path_contains="networking")
    assert len(docs) == 1
    assert docs[0]["file_path"] == "networking/ports.md"


def test_query_documents_created_before(kb):
    """query_documents should filter by created_before date."""
    kb.upsert_document(
        "src:old.md",
        "",
        {
            "source": "src",
            "file_path": "old.md",
            "title": "Old",
            "is_chunk": False,
            "created_at": "2024-01-01T00:00:00",
        },
    )
    kb.upsert_document(
        "src:new.md",
        "",
        {
            "source": "src",
            "file_path": "new.md",
            "title": "New",
            "is_chunk": False,
            "created_at": "2025-06-01T00:00:00",
        },
    )

    docs = kb.query_documents(created_before="2025-01-01")
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "src:old.md"


def test_query_documents_combined_filters(kb):
    """query_documents should support combining multiple filters."""
    kb.upsert_document(
        "alpha:docs/setup.md",
        "",
        {
            "source": "alpha",
            "file_path": "docs/setup.md",
            "title": "Setup",
            "is_chunk": False,
            "created_at": "2024-06-01T00:00:00",
        },
    )
    kb.upsert_document(
        "alpha:docs/deploy.md",
        "",
        {
            "source": "alpha",
            "file_path": "docs/deploy.md",
            "title": "Deploy",
            "is_chunk": False,
            "created_at": "2025-06-01T00:00:00",
        },
    )
    kb.upsert_document(
        "beta:docs/setup.md",
        "",
        {
            "source": "beta",
            "file_path": "docs/setup.md",
            "title": "Setup",
            "is_chunk": False,
            "created_at": "2024-06-01T00:00:00",
        },
    )

    docs = kb.query_documents(
        source="alpha", file_path_contains="setup", created_before="2025-01-01"
    )
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "alpha:docs/setup.md"


def test_get_indexed_content_hashes(kb):
    """get_indexed_content_hashes should return hash for parent docs only."""
    kb.upsert_document(
        "src:a.md",
        "",
        {"source": "src", "file_path": "a.md", "is_chunk": False, "content_hash": "aaa111"},
    )
    kb.upsert_document(
        "src:a.md#chunk0",
        "content",
        {"source": "src", "file_path": "a.md", "chunk_index": 0, "is_chunk": True, "content_hash": ""},
    )
    kb.upsert_document(
        "src:b.md",
        "",
        {"source": "src", "file_path": "b.md", "is_chunk": False, "content_hash": "bbb222"},
    )

    hashes = kb.get_indexed_content_hashes("src")
    assert hashes == {
        "src:a.md": "aaa111",
        "src:b.md": "bbb222",
    }


def test_get_indexed_content_hashes_empty_source(kb):
    """Empty source should return empty dict."""
    assert kb.get_indexed_content_hashes("nonexistent") == {}


def test_get_indexed_content_hashes_ignores_other_sources(kb):
    """Should only return hashes for the requested source."""
    kb.upsert_document(
        "a:doc.md", "", {"source": "a", "file_path": "doc.md", "is_chunk": False, "content_hash": "aaa"}
    )
    kb.upsert_document(
        "b:doc.md", "", {"source": "b", "file_path": "doc.md", "is_chunk": False, "content_hash": "bbb"}
    )
    hashes = kb.get_indexed_content_hashes("a")
    assert "a:doc.md" in hashes
    assert "b:doc.md" not in hashes


def test_get_all_source_names(kb):
    """get_all_source_names should return all distinct sources in the KB."""
    assert kb.get_all_source_names() == set()

    kb.upsert_document("a:doc.md", "", {"source": "a", "file_path": "doc.md", "is_chunk": False})
    kb.upsert_document("b:doc.md", "", {"source": "b", "file_path": "doc.md", "is_chunk": False})

    assert kb.get_all_source_names() == {"a", "b"}


def test_rename_source(kb):
    """rename_source should update doc_ids, source column, and ChromaDB entries."""
    # Insert parent + chunk under old name
    kb.upsert_document(
        "old-name:readme.md",
        "",
        {"source": "old-name", "file_path": "readme.md", "title": "README", "is_chunk": False},
    )
    kb.upsert_document(
        "old-name:readme.md#chunk0",
        "Some documentation content about the project.",
        {
            "source": "old-name",
            "file_path": "readme.md",
            "title": "README",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )

    count = kb.rename_source("old-name", "new-name")
    assert count == 2

    # Old IDs should be gone
    assert kb.get_document("old-name:readme.md") is None
    assert kb.get_document("old-name:readme.md#chunk0") is None
    assert kb.get_all_doc_ids_for_source("old-name") == set()

    # New IDs should exist with updated source
    doc = kb.get_document("new-name:readme.md")
    assert doc is not None
    assert doc["source"] == "new-name"

    chunk = kb.get_document("new-name:readme.md#chunk0")
    assert chunk is not None
    assert chunk["source"] == "new-name"

    # ChromaDB search should find the chunk under the new source
    results = kb.search("documentation project", source_filter="new-name")
    assert len(results) >= 1
    assert results[0]["metadata"]["source"] == "new-name"


def test_rename_source_empty(kb):
    """rename_source on a nonexistent source should return 0."""
    assert kb.rename_source("nonexistent", "new") == 0


def test_get_document_tree(kb):
    """get_document_tree should organize docs by source and category."""
    kb.upsert_document(
        "alpha:docs/setup.md",
        "",
        {"source": "alpha", "file_path": "docs/setup.md", "title": "Setup", "is_chunk": False},
    )
    kb.upsert_document(
        "alpha:journal/250101-init.md",
        "",
        {
            "source": "alpha",
            "file_path": "journal/250101-init.md",
            "title": "Init",
            "is_chunk": False,
            "created_at": "2025-01-01T00:00:00",
        },
    )
    kb.upsert_document(
        "beta:docs/readme.md",
        "",
        {"source": "beta", "file_path": "docs/readme.md", "title": "README", "is_chunk": False},
    )
    # Chunks should be excluded
    kb.upsert_document(
        "alpha:docs/setup.md#chunk0",
        "content",
        {"source": "alpha", "file_path": "docs/setup.md", "chunk_index": 0, "is_chunk": True},
    )

    tree = kb.get_document_tree()
    assert len(tree) == 2
    assert tree[0]["source"] == "alpha"
    assert len(tree[0]["root_docs"]) == 0
    assert len(tree[0]["docs"]) == 1
    assert len(tree[0]["journal"]) == 1
    assert tree[0]["docs"][0]["title"] == "Setup"
    assert tree[0]["journal"][0]["title"] == "Init"
    assert len(tree[0]["engineering_team"]) == 0
    assert tree[1]["source"] == "beta"
    assert len(tree[1]["docs"]) == 1


def test_get_document_tree_engineering_team(kb):
    """Files under .engineering-team/ should appear in engineering_team category."""
    kb.upsert_document(
        "proj:.engineering-team/analysis.md",
        "",
        {
            "source": "proj",
            "file_path": ".engineering-team/analysis.md",
            "title": "Analysis",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "proj:docs/guide.md",
        "",
        {"source": "proj", "file_path": "docs/guide.md", "title": "Guide", "is_chunk": False},
    )

    tree = kb.get_document_tree()
    assert len(tree) == 1
    src = tree[0]
    assert len(src["engineering_team"]) == 1
    assert src["engineering_team"][0]["title"] == "Analysis"
    assert len(src["docs"]) == 1


def test_get_document_tree_root_docs(kb):
    """Root-level files (no directory) should appear in root_docs category."""
    kb.upsert_document(
        "proj:README.md",
        "",
        {"source": "proj", "file_path": "README.md", "title": "Project README", "is_chunk": False},
    )
    kb.upsert_document(
        "proj:docs/guide.md",
        "",
        {"source": "proj", "file_path": "docs/guide.md", "title": "Guide", "is_chunk": False},
    )
    kb.upsert_document(
        "proj:journal/250301-entry.md",
        "",
        {
            "source": "proj",
            "file_path": "journal/250301-entry.md",
            "title": "Entry",
            "is_chunk": False,
            "created_at": "2025-03-01T00:00:00",
        },
    )

    tree = kb.get_document_tree()
    assert len(tree) == 1
    src = tree[0]
    assert src["source"] == "proj"
    assert len(src["root_docs"]) == 1
    assert src["root_docs"][0]["title"] == "Project README"
    assert len(src["docs"]) == 1
    assert len(src["journal"]) == 1


def test_get_document_tree_pdf_category(kb):
    """PDF files should appear in the 'pdf' category regardless of directory."""
    kb.upsert_document(
        "proj:docs/report.pdf",
        "",
        {"source": "proj", "file_path": "docs/report.pdf", "title": "report", "is_chunk": False},
    )
    kb.upsert_document(
        "proj:manual.pdf",
        "",
        {"source": "proj", "file_path": "manual.pdf", "title": "manual", "is_chunk": False},
    )
    kb.upsert_document(
        "proj:docs/guide.md",
        "",
        {"source": "proj", "file_path": "docs/guide.md", "title": "Guide", "is_chunk": False},
    )

    tree = kb.get_document_tree()
    assert len(tree) == 1
    src = tree[0]
    assert len(src["pdf"]) == 2
    assert {d["title"] for d in src["pdf"]} == {"report", "manual"}
    # The markdown file should be in docs, not pdf
    assert len(src["docs"]) == 1
    assert src["docs"][0]["title"] == "Guide"


def test_get_document_tree_skills_category(kb):
    """Files under a skills/ directory should appear in the 'skills' category."""
    kb.upsert_document(
        "nanoclaw:container/skills/weather/skill.md",
        "",
        {
            "source": "nanoclaw",
            "file_path": "container/skills/weather/skill.md",
            "title": "Weather Skill",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "nanoclaw:container/skills/calendar/skill.md",
        "",
        {
            "source": "nanoclaw",
            "file_path": "container/skills/calendar/skill.md",
            "title": "Calendar Skill",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "nanoclaw:docs/guide.md",
        "",
        {"source": "nanoclaw", "file_path": "docs/guide.md", "title": "Guide", "is_chunk": False},
    )

    tree = kb.get_document_tree()
    assert len(tree) == 1
    src = tree[0]
    assert len(src["skills"]) == 2
    assert {d["title"] for d in src["skills"]} == {"Weather Skill", "Calendar Skill"}
    # Skills should be sorted alphabetically by title
    assert src["skills"][0]["title"] == "Calendar Skill"
    assert src["skills"][1]["title"] == "Weather Skill"
    # The markdown file should be in docs, not skills
    assert len(src["docs"]) == 1


def test_get_document_tree_runbooks_category(kb):
    """Files under runbooks/ should appear in the 'runbooks' category."""
    kb.upsert_document(
        "nanoclaw:runbooks/deploy-guide.md",
        "",
        {
            "source": "nanoclaw",
            "file_path": "runbooks/deploy-guide.md",
            "title": "Deploy Guide",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "nanoclaw:runbooks/incident-response.md",
        "",
        {
            "source": "nanoclaw",
            "file_path": "runbooks/incident-response.md",
            "title": "Incident Response",
            "is_chunk": False,
        },
    )
    kb.upsert_document(
        "nanoclaw:docs/readme.md",
        "",
        {"source": "nanoclaw", "file_path": "docs/readme.md", "title": "README", "is_chunk": False},
    )

    tree = kb.get_document_tree()
    assert len(tree) == 1
    src = tree[0]
    assert len(src["runbooks"]) == 2
    assert {d["title"] for d in src["runbooks"]} == {"Deploy Guide", "Incident Response"}
    # Runbooks should be sorted alphabetically by title
    assert src["runbooks"][0]["title"] == "Deploy Guide"
    # The markdown file should be in docs, not runbooks
    assert len(src["docs"]) == 1


def test_get_full_document_reassembles_chunks(kb):
    """get_full_document should reassemble content from chunks for parent docs."""
    kb.upsert_document(
        "src:guide.md",
        "",
        {"source": "src", "file_path": "guide.md", "title": "Guide", "is_chunk": False},
    )
    kb.upsert_document(
        "src:guide.md#chunk0",
        "First chunk content",
        {
            "source": "src",
            "file_path": "guide.md",
            "title": "Guide",
            "chunk_index": 0,
            "total_chunks": 2,
            "is_chunk": True,
        },
    )
    kb.upsert_document(
        "src:guide.md#chunk1",
        "Second chunk content",
        {
            "source": "src",
            "file_path": "guide.md",
            "title": "Guide",
            "chunk_index": 1,
            "total_chunks": 2,
            "is_chunk": True,
        },
    )

    # get_document returns empty content for parent
    raw = kb.get_document("src:guide.md")
    assert raw is not None
    assert raw["content"] == ""

    # get_full_document reassembles from chunks
    full = kb.get_full_document("src:guide.md")
    assert full is not None
    assert "First chunk content" in full["content"]
    assert "Second chunk content" in full["content"]
    assert full["content"].index("First") < full["content"].index("Second")


def test_get_full_document_not_found(kb):
    """get_full_document should return None for nonexistent doc."""
    assert kb.get_full_document("nonexistent:doc.md") is None


def test_search_documents_deduplicates(kb):
    """search_documents should deduplicate chunks to parent docs."""
    kb.upsert_document(
        "src:guide.md",
        "Full guide content about Python programming",
        {"source": "src", "file_path": "guide.md", "title": "Guide", "is_chunk": False},
    )
    kb.upsert_document(
        "src:guide.md#chunk0",
        "Python programming basics and fundamentals",
        {
            "source": "src",
            "file_path": "guide.md",
            "title": "Guide",
            "chunk_index": 0,
            "total_chunks": 2,
            "is_chunk": True,
        },
    )
    kb.upsert_document(
        "src:guide.md#chunk1",
        "Advanced Python programming patterns",
        {
            "source": "src",
            "file_path": "guide.md",
            "title": "Guide",
            "chunk_index": 1,
            "total_chunks": 2,
            "is_chunk": True,
        },
    )

    results = kb.search_documents("Python programming")
    # Should only get one result (the parent doc), not two chunk hits
    parent_ids = [r["doc_id"] for r in results]
    assert parent_ids.count("src:guide.md") == 1


def test_delete_source_documents(kb):
    """delete_source_documents should remove only the targeted source's docs."""
    # Upsert docs for two sources
    kb.upsert_document(
        "alpha:one.md", "", {"source": "alpha", "file_path": "one.md", "is_chunk": False}
    )
    kb.upsert_document(
        "alpha:one.md#chunk0",
        "Alpha content here",
        {"source": "alpha", "file_path": "one.md", "chunk_index": 0, "is_chunk": True},
    )
    kb.upsert_document(
        "beta:two.md", "", {"source": "beta", "file_path": "two.md", "is_chunk": False}
    )
    kb.upsert_document(
        "beta:two.md#chunk0",
        "Beta content here",
        {"source": "beta", "file_path": "two.md", "chunk_index": 0, "is_chunk": True},
    )

    # Delete alpha
    count = kb.delete_source_documents("alpha")
    assert count == 2  # parent + chunk

    # Alpha should be gone
    assert kb.get_all_doc_ids_for_source("alpha") == set()
    assert kb.get_document("alpha:one.md") is None

    # Beta should remain
    assert kb.get_all_doc_ids_for_source("beta") == {"beta:two.md", "beta:two.md#chunk0"}
    assert kb.get_document("beta:two.md") is not None


# ---------------------------------------------------------------------------
# Batch upsert tests
# ---------------------------------------------------------------------------


def test_upsert_documents_batch_sqlite_and_chroma(kb):
    """Batch upsert stores parent in SQLite and chunks in both SQLite + ChromaDB."""
    items = [
        (
            "src:doc.md",
            "Full document content",
            {
                "source": "src",
                "file_path": "doc.md",
                "title": "Doc",
                "is_chunk": False,
                "content_hash": "abc123",
            },
        ),
        (
            "src:doc.md#chunk0",
            "First chunk about home server networking and architecture.",
            {
                "source": "src",
                "file_path": "doc.md",
                "title": "Doc",
                "chunk_index": 0,
                "total_chunks": 2,
                "is_chunk": True,
                "section_path": "Setup",
            },
        ),
        (
            "src:doc.md#chunk1",
            "Second chunk about firewall configuration and port forwarding.",
            {
                "source": "src",
                "file_path": "doc.md",
                "title": "Doc",
                "chunk_index": 1,
                "total_chunks": 2,
                "is_chunk": True,
                "section_path": "Setup > Firewall",
            },
        ),
    ]

    kb.upsert_documents_batch(items)

    # All three should be in SQLite.
    assert kb.get_document("src:doc.md") is not None
    assert kb.get_document("src:doc.md#chunk0") is not None
    assert kb.get_document("src:doc.md#chunk1") is not None

    # Chunks should be searchable in ChromaDB.
    results = kb.search("firewall port")
    assert len(results) >= 1
    assert any("firewall" in r["content"].lower() for r in results)


def test_upsert_documents_batch_empty(kb):
    """Batch upsert with an empty list is a no-op."""
    kb.upsert_documents_batch([])
    assert kb.get_sources_summary() == []


def test_upsert_documents_batch_replaces_existing(kb):
    """Batch upsert replaces existing docs (INSERT OR REPLACE)."""
    kb.upsert_document(
        "src:readme.md",
        "Old content",
        {"source": "src", "file_path": "readme.md", "title": "Old", "is_chunk": False},
    )

    kb.upsert_documents_batch([
        (
            "src:readme.md",
            "New content",
            {"source": "src", "file_path": "readme.md", "title": "New", "is_chunk": False},
        ),
    ])

    doc = kb.get_document("src:readme.md")
    assert doc["title"] == "New"
    assert doc["content"] == "New content"


# ---------------------------------------------------------------------------
# Hybrid L1 search: BM25 (FTS5) + dense (ChromaDB) + RRF fusion
# ---------------------------------------------------------------------------


def test_bm25_finds_rare_token(kb):
    """BM25 leg surfaces a rare named entity that dense search would miss.

    Reproduces the strava failure mode that motivated the hybrid pipeline:
    a chunk literally containing "strava" wins over a semantically-adjacent
    chunk that never says the word.
    """
    kb.upsert_document(
        "src:strava.md#chunk0",
        "Imported a Strava activity for tracking the cycling commute.",
        {
            "source": "src",
            "file_path": "strava.md",
            "title": "Strava import",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )
    kb.upsert_document(
        "src:webapp.md#chunk0",
        "Vue 3 frontend for the journal analysis tool with OCR correction.",
        {
            "source": "src",
            "file_path": "webapp.md",
            "title": "Webapp",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )

    results = kb._bm25_search_chunks("strava")
    assert results, "BM25 should return at least one result for 'strava'"
    assert results[0]["doc_id"] == "src:strava.md#chunk0"
    assert "strava" in results[0]["content"].lower()


def test_bm25_title_match(kb):
    """BM25 weights title heavily so title-only matches still surface."""
    kb.upsert_document(
        "src:doc.md#chunk0",
        "This text is entirely about gardening and flowers.",
        {
            "source": "src",
            "file_path": "doc.md",
            "title": "Kubernetes Cluster Setup",
            "chunk_index": 0,
            "total_chunks": 1,
            "is_chunk": True,
        },
    )

    results = kb._bm25_search_chunks("Kubernetes")
    assert len(results) == 1
    assert results[0]["doc_id"] == "src:doc.md#chunk0"


def test_bm25_source_filter(kb):
    """BM25 leg respects source_filter."""
    for src in ("alpha", "beta"):
        kb.upsert_document(
            f"{src}:doc.md#chunk0",
            "Identical content about Docker networking.",
            {
                "source": src,
                "file_path": "doc.md",
                "title": "Docker",
                "chunk_index": 0,
                "is_chunk": True,
            },
        )

    results = kb._bm25_search_chunks("Docker", source_filter="alpha")
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "alpha"


def test_bm25_query_sanitization_no_syntax_error(kb):
    """Raw user queries with FTS5 operators must not raise."""
    kb.upsert_document(
        "src:doc.md#chunk0",
        "Some content about foo and bar.",
        {
            "source": "src",
            "file_path": "doc.md",
            "title": "Doc",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )

    # None of these should raise.
    _ = kb._bm25_search_chunks('"foo*bar"')
    _ = kb._bm25_search_chunks("(unclosed paren")
    _ = kb._bm25_search_chunks("-leading-minus")
    _ = kb._bm25_search_chunks("")  # empty query
    _ = kb._bm25_search_chunks("   ")  # whitespace-only


def test_delete_propagates_to_fts(kb):
    """Deleting a chunk removes it from the FTS index."""
    kb.upsert_document(
        "src:doc.md#chunk0",
        "Unique zxqwerty content for finding.",
        {
            "source": "src",
            "file_path": "doc.md",
            "title": "Doc",
            "chunk_index": 0,
            "is_chunk": True,
        },
    )
    assert kb._bm25_search_chunks("zxqwerty"), "BM25 should find the chunk before delete"

    kb.delete_document("src:doc.md#chunk0")
    assert kb._bm25_search_chunks("zxqwerty") == [], "BM25 should not find chunk after delete"


def test_delete_source_propagates_to_fts(kb):
    """Deleting a source removes its chunks from FTS."""
    kb.upsert_document(
        "alpha:doc.md#chunk0",
        "Unique zxqwerty content.",
        {"source": "alpha", "file_path": "doc.md", "title": "Doc",
         "chunk_index": 0, "is_chunk": True},
    )
    kb.upsert_document(
        "beta:doc.md#chunk0",
        "Unique zxqwerty content.",
        {"source": "beta", "file_path": "doc.md", "title": "Doc",
         "chunk_index": 0, "is_chunk": True},
    )

    kb.delete_source_documents("alpha")

    results = kb._bm25_search_chunks("zxqwerty")
    sources_returned = {str(r["metadata"]["source"]) for r in results}
    assert sources_returned == {"beta"}


def test_fts_backfill_on_existing_db(tmp_path):
    """A chunks_fts table that starts empty is backfilled from documents on init."""
    kb_dir = tmp_path / "data"
    kb1 = KnowledgeBase(str(kb_dir))
    kb1.upsert_document(
        "src:doc.md#chunk0",
        "Unique zxqwerty token in the chunk body.",
        {"source": "src", "file_path": "doc.md", "title": "Doc",
         "chunk_index": 0, "is_chunk": True},
    )
    kb1.close()

    # Wipe FTS to simulate a pre-FTS database, then re-init.
    import sqlite3
    with sqlite3.connect(str(kb_dir / "documents.db")) as conn:
        _ = conn.execute("DELETE FROM chunks_fts")

    kb2 = KnowledgeBase(str(kb_dir))
    try:
        results = kb2._bm25_search_chunks("zxqwerty")
        assert len(results) == 1
        assert results[0]["doc_id"] == "src:doc.md#chunk0"
    finally:
        kb2.close()


def test_rrf_fuse_math():
    """RRF score = sum(1/(k+rank)) across lists; sorted descending."""
    a: list = [
        {"doc_id": "x", "content": "", "metadata": {}, "score": 0.0},
        {"doc_id": "y", "content": "", "metadata": {}, "score": 0.0},
    ]
    b: list = [
        {"doc_id": "y", "content": "", "metadata": {}, "score": 0.0},
        {"doc_id": "z", "content": "", "metadata": {}, "score": 0.0},
    ]
    fused = KnowledgeBase._rrf_fuse([a, b], k=60)

    expected = {
        "x": 1 / 61,                   # rank 1 in a only
        "y": 1 / 61 + 1 / 62,          # rank 1 in b, rank 2 in a
        "z": 1 / 62,                   # rank 2 in b only
    }
    got = dict(fused)
    for doc_id, score in expected.items():
        assert got[doc_id] == pytest.approx(score)

    # y appears in both lists so should rank first; z and x are close.
    assert fused[0][0] == "y"


def test_search_documents_strava_reproduction(kb):
    """Headline acceptance test: q='strava' with a misleading semantic neighbour.

    Reproduces the prod bug where pure-vector search ranked a CLAUDE.md-like
    doc above the actual strava-mentioning chunk. After the hybrid pipeline,
    the strava doc must be top-ranked.
    """
    # Decoy: a CLAUDE.md-like parent + chunk that is semantically near
    # journal/fitness queries but never says "strava".
    kb.upsert_document(
        "journal:CLAUDE.md",
        "",
        {"source": "journal", "file_path": "CLAUDE.md",
         "title": "Journal Webapp", "is_chunk": False},
    )
    kb.upsert_document(
        "journal:CLAUDE.md#chunk0",
        "Vue 3 frontend for the Journal Analysis Tool. Displays journal "
        "entries, enables OCR correction, and will include dashboards.",
        {"source": "journal", "file_path": "CLAUDE.md",
         "title": "Journal Webapp", "chunk_index": 0, "total_chunks": 1,
         "is_chunk": True},
    )
    # Real answer: a journal entry that literally mentions Strava.
    kb.upsert_document(
        "journal:journal/strava-import.md",
        "",
        {"source": "journal", "file_path": "journal/strava-import.md",
         "title": "Strava import", "is_chunk": False},
    )
    kb.upsert_document(
        "journal:journal/strava-import.md#chunk0",
        "Imported a Strava activity for the morning cycling commute. The "
        "OAuth refresh flow is now wired up and the activity stream "
        "deduplicates against existing rides.",
        {"source": "journal", "file_path": "journal/strava-import.md",
         "title": "Strava import", "chunk_index": 0, "total_chunks": 1,
         "is_chunk": True},
    )

    results = kb.search_documents("strava", source_filter="journal")
    assert results, "search_documents should return at least one hit"
    assert results[0]["doc_id"] == "journal:journal/strava-import.md", (
        f"Top result should be the strava doc, got {results[0]['doc_id']}"
    )


# ------------------------------------------------------------------
# Source status tracking
# ------------------------------------------------------------------


def test_update_source_check_success(kb):
    """Successful check sets last_checked and clears errors."""
    kb.update_source_check("src")
    statuses = kb.get_source_statuses()
    assert "src" in statuses
    assert statuses["src"]["last_checked"] is not None
    assert statuses["src"]["last_error"] is None
    assert statuses["src"]["consecutive_failures"] == 0


def test_update_source_check_failure(kb):
    """Failed check records error and increments consecutive_failures."""
    kb.update_source_check("src", error="git fetch failed")
    statuses = kb.get_source_statuses()
    assert statuses["src"]["last_error"] == "git fetch failed"
    assert statuses["src"]["last_error_at"] is not None
    assert statuses["src"]["consecutive_failures"] == 1


def test_update_source_check_consecutive_failures(kb):
    """Multiple failures increment the counter."""
    kb.update_source_check("src", error="fail 1")
    kb.update_source_check("src", error="fail 2")
    kb.update_source_check("src", error="fail 3")
    statuses = kb.get_source_statuses()
    assert statuses["src"]["consecutive_failures"] == 3


def test_update_source_check_success_resets_failures(kb):
    """A success after failures resets the counter and clears the error."""
    kb.update_source_check("src", error="fail 1")
    kb.update_source_check("src", error="fail 2")
    kb.update_source_check("src")
    statuses = kb.get_source_statuses()
    assert statuses["src"]["consecutive_failures"] == 0
    assert statuses["src"]["last_error"] is None
    assert statuses["src"]["last_error_at"] is None
    assert statuses["src"]["last_checked"] is not None


def test_get_source_statuses_empty(kb):
    """Returns empty dict when no status records exist."""
    statuses = kb.get_source_statuses()
    assert statuses == {}


def test_get_source_statuses_multiple_sources(kb):
    """Returns status for each tracked source."""
    kb.update_source_check("src-a")
    kb.update_source_check("src-b", error="timeout")
    statuses = kb.get_source_statuses()
    assert len(statuses) == 2
    assert statuses["src-a"]["consecutive_failures"] == 0
    assert statuses["src-b"]["consecutive_failures"] == 1


def test_unload_embedding_model_delegates_to_ef(kb):
    """unload_embedding_model should delegate to the embedding function's unload()."""
    with patch.object(kb._embedding_fn, "unload", return_value=True) as mock:
        result = kb.unload_embedding_model()
    mock.assert_called_once()
    assert result is True


def test_kb_uses_http_client_when_chroma_host_set(tmp_path, monkeypatch):
    """When chroma_host is set, KnowledgeBase should construct an HttpClient
    instead of a PersistentClient. The HttpClient is mocked so the test
    does not require a running Chroma server."""
    from unittest.mock import MagicMock

    from docserver.knowledge_base import KnowledgeBase

    fake_collection = MagicMock()
    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection
    monkeypatch.setattr(
        "docserver.knowledge_base.chromadb.HttpClient",
        MagicMock(return_value=fake_client),
    )
    # PersistentClient should NOT be called in this branch.
    pclient = MagicMock()
    monkeypatch.setattr("docserver.knowledge_base.chromadb.PersistentClient", pclient)

    kb = KnowledgeBase(
        str(tmp_path),
        chroma_host="chroma-sidecar",
        chroma_port=8000,
    )

    pclient.assert_not_called()
    assert kb._chroma_client is fake_client
    assert kb._collection is fake_collection
    kb.close()


def test_sqlite_journal_mode_is_wal(kb):
    """documents.db must be in WAL mode so the ingestion worker can write while the server reads."""
    import sqlite3 as _sqlite3

    with _sqlite3.connect(kb._db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    # SQLite reports the resolved mode as a lowercase string.
    assert mode.lower() == "wal", f"expected wal, got {mode!r}"
