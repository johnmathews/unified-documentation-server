"""Knowledge base layer combining SQLite (structured metadata) and ChromaDB (vector search)."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

import chromadb
from chromadb.utils import embedding_functions


class SearchResult(TypedDict):
    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class SourceSummary(TypedDict):
    source: str
    file_count: int
    chunk_count: int
    last_indexed: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
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

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
CREATE INDEX IF NOT EXISTS idx_documents_is_chunk ON documents (is_chunk);
"""

_CHROMA_COLLECTION = "documents"
_EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dims, much better than default all-MiniLM-L6-v2 (384 dims)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeBase:
    """Combines SQLite for structured metadata and ChromaDB for semantic search."""

    _db_path: str
    _chroma_client: chromadb.PersistentClient
    _collection: chromadb.Collection

    def __init__(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        chroma_dir = os.path.join(data_dir, "chroma")
        os.makedirs(chroma_dir, exist_ok=True)

        self._db_path = os.path.join(data_dir, "documents.db")
        self._init_sqlite()

        self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_EMBEDDING_MODEL,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Insert or replace a document in SQLite; also index chunks in ChromaDB."""
        indexed_at = _now_iso()
        is_chunk = bool(metadata.get("is_chunk", False))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (doc_id, source, file_path, title, content, chunk_index,
                     total_chunks, created_at, modified_at, indexed_at, size_bytes, is_chunk,
                     section_path)
                VALUES
                    (:doc_id, :source, :file_path, :title, :content, :chunk_index,
                     :total_chunks, :created_at, :modified_at, :indexed_at, :size_bytes, :is_chunk,
                     :section_path)
                """,
                {
                    "doc_id": doc_id,
                    "source": metadata.get("source", ""),
                    "file_path": metadata.get("file_path", ""),
                    "title": metadata.get("title"),
                    "content": content,
                    "chunk_index": metadata.get("chunk_index"),
                    "total_chunks": metadata.get("total_chunks"),
                    "created_at": metadata.get("created_at"),
                    "modified_at": metadata.get("modified_at"),
                    "indexed_at": indexed_at,
                    "size_bytes": metadata.get("size_bytes"),
                    "is_chunk": is_chunk,
                    "section_path": metadata.get("section_path", ""),
                },
            )

        if content and is_chunk:
            chroma_meta = {
                "source": metadata.get("source", ""),
                "file_path": metadata.get("file_path", ""),
                "title": metadata.get("title") or "",
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "is_chunk": True,
                "section_path": metadata.get("section_path", ""),
            }
            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[chroma_meta],
            )

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from both SQLite and ChromaDB."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        try:
            self._collection.delete(ids=[doc_id])
        except Exception:
            # ChromaDB raises if the ID doesn't exist; that's fine.
            pass

    def delete_source_documents(self, source_name: str) -> int:
        """Delete all documents for a source. Returns the count deleted."""
        ids_to_delete = list(self.get_all_doc_ids_for_source(source_name))

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE source = ?", (source_name,)
            )
            count = cursor.rowcount

        if ids_to_delete:
            try:
                self._collection.delete(ids=ids_to_delete)
            except Exception:
                pass

        return count

    def get_all_doc_ids_for_source(self, source_name: str) -> set[str]:
        """Return all doc_ids belonging to a source."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE source = ?", (source_name,)
            ).fetchall()
        return {row["doc_id"] for row in rows}

    def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search via ChromaDB.

        Returns a list of dicts with keys: doc_id, content, metadata, score.
        score is the distance returned by ChromaDB (lower = more similar).
        """
        where: dict[str, Any] | None = None
        if source_filter:
            where = {"source": source_filter}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("ChromaDB search failed; returning empty results.")
            return []

        output: list[SearchResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            output.append(
                {
                    "doc_id": doc_id,
                    "content": doc_text,
                    "metadata": meta,
                    "score": dist,
                }
            )

        return output

    def query_documents(
        self,
        source: str | None = None,
        file_path_contains: str | None = None,
        title_contains: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Structured SQL query returning parent docs only (no chunks).

        Returns a list of metadata dicts (no content field).
        """
        conditions: list[str] = ["(is_chunk = FALSE OR chunk_index IS NULL)"]
        params: list[Any] = []

        if source:
            conditions.append("source = ?")
            params.append(source)

        if file_path_contains:
            conditions.append("file_path LIKE ?")
            params.append(f"%{file_path_contains}%")

        if title_contains:
            conditions.append("title LIKE ?")
            params.append(f"%{title_contains}%")

        if created_after:
            conditions.append("created_at >= ?")
            params.append(created_after)

        if created_before:
            conditions.append("created_at <= ?")
            params.append(created_before)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT doc_id, source, file_path, title, chunk_index, total_chunks,
                   created_at, modified_at, indexed_at, size_bytes, is_chunk
            FROM documents
            WHERE {where_clause}
            ORDER BY indexed_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single document by ID, including content."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()

        if row is None:
            return None
        return dict(row)

    def get_sources_summary(self) -> list[SourceSummary]:
        """Return per-source summary: source, file_count, chunk_count, last_indexed."""
        sql = """
            SELECT
                source,
                SUM(CASE WHEN is_chunk = FALSE OR chunk_index IS NULL THEN 1 ELSE 0 END) AS file_count,
                SUM(CASE WHEN is_chunk = TRUE THEN 1 ELSE 0 END) AS chunk_count,
                MAX(indexed_at) AS last_indexed
            FROM documents
            GROUP BY source
            ORDER BY source
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()

        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close connections. ChromaDB PersistentClient manages its own lifecycle."""
        # SQLite connections are managed per-operation via context managers.
        # ChromaDB's PersistentClient does not expose an explicit close method
        # in all versions; call it if available.
        close_fn = getattr(self._chroma_client, "close", None)
        if callable(close_fn):
            close_fn()
