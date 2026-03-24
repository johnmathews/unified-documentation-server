"""Tests for the knowledge base module."""

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
    assert len(tree[0]["docs"]) == 1
    assert len(tree[0]["journal"]) == 1
    assert tree[0]["docs"][0]["title"] == "Setup"
    assert tree[0]["journal"][0]["title"] == "Init"
    assert tree[1]["source"] == "beta"
    assert len(tree[1]["docs"]) == 1


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
