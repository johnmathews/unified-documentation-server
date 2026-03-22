"""Tests for the knowledge base module."""

import tempfile

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
    kb.upsert_document(
        "a:one.md", "", {"source": "a", "file_path": "one.md", "is_chunk": False}
    )
    kb.upsert_document(
        "a:two.md", "", {"source": "a", "file_path": "two.md", "is_chunk": False}
    )
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
    kb.upsert_document(
        "src:a.md", "", {"source": "src", "file_path": "a.md", "is_chunk": False}
    )
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
        {"source": "a", "file_path": "doc.md", "title": "Guide", "chunk_index": 0, "is_chunk": True},
    )
    kb.upsert_document(
        "b:doc.md#chunk0",
        "Python programming tutorial",
        {"source": "b", "file_path": "doc.md", "title": "Tutorial", "chunk_index": 0, "is_chunk": True},
    )

    results = kb.search("python programming", source_filter="a")
    assert all(r["metadata"]["source"] == "a" for r in results)


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
