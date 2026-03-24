"""Knowledge base layer combining SQLite (structured metadata) and ChromaDB (vector search)."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any, TypedDict

import chromadb

from docserver.embedding import OnnxEmbeddingFunction

logger = logging.getLogger(__name__)


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
    section_path  TEXT DEFAULT '',
    content_hash  TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
CREATE INDEX IF NOT EXISTS idx_documents_is_chunk ON documents (is_chunk);
"""

_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN content_hash TEXT DEFAULT ''",
]

_CHROMA_COLLECTION = "documents"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class KnowledgeBase:
    """Combines SQLite for structured metadata and ChromaDB for semantic search."""

    _db_path: str
    _chroma_client: chromadb.PersistentClient
    _collection: chromadb.Collection

    def __init__(self, data_dir: str) -> None:
        logger.info(
            "Initializing knowledge base in %s",
            data_dir,
            extra={"event": "kb_init"},
        )
        os.makedirs(data_dir, exist_ok=True)
        chroma_dir = os.path.join(data_dir, "chroma")
        os.makedirs(chroma_dir, exist_ok=True)

        self._db_path = os.path.join(data_dir, "documents.db")
        self._init_sqlite()
        logger.info("SQLite initialized at %s", self._db_path, extra={"event": "kb_init"})

        self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self._embedding_fn = OnnxEmbeddingFunction()
        self._collection = self._chroma_client.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            "ChromaDB initialized at %s (collection: %s)",
            chroma_dir,
            _CHROMA_COLLECTION,
            extra={"event": "kb_init"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)
            self._run_migrations(conn)

    @staticmethod
    def _run_migrations(conn: sqlite3.Connection) -> None:
        """Apply schema migrations idempotently."""
        for sql in _MIGRATIONS:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(sql)

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
                     section_path, content_hash)
                VALUES
                    (:doc_id, :source, :file_path, :title, :content, :chunk_index,
                     :total_chunks, :created_at, :modified_at, :indexed_at, :size_bytes, :is_chunk,
                     :section_path, :content_hash)
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
                    "content_hash": metadata.get("content_hash", ""),
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

        with contextlib.suppress(Exception):
            # ChromaDB raises if the ID doesn't exist; that's fine.
            self._collection.delete(ids=[doc_id])

    def delete_source_documents(self, source_name: str) -> int:
        """Delete all documents for a source. Returns the count deleted."""
        ids_to_delete = list(self.get_all_doc_ids_for_source(source_name))

        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE source = ?", (source_name,))
            count = cursor.rowcount

        if ids_to_delete:
            with contextlib.suppress(Exception):
                self._collection.delete(ids=ids_to_delete)

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

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances, strict=True):
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
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

        if row is None:
            return None
        return dict(row)

    def get_indexed_content_hashes(self, source: str) -> dict[str, str]:
        """Return {doc_id: content_hash} for all parent docs in a source."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, content_hash FROM documents WHERE source = ? AND (is_chunk = FALSE OR chunk_index IS NULL)",
                (source,),
            ).fetchall()
        return {row["doc_id"]: row["content_hash"] for row in rows}

    def rename_source(self, old_name: str, new_name: str) -> int:
        """Rename a source: update doc_ids, source column, and ChromaDB entries.

        Preserves embeddings in ChromaDB to avoid expensive re-computation.
        Returns the number of documents migrated.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, is_chunk FROM documents WHERE source = ?", (old_name,)
            ).fetchall()

        if not rows:
            return 0

        old_ids = [row["doc_id"] for row in rows]
        chunk_old_ids = [row["doc_id"] for row in rows if row["is_chunk"]]

        # Build old→new ID mapping: replace source name prefix in doc_id
        prefix_old = f"{old_name}:"
        prefix_new = f"{new_name}:"
        id_map = {}
        for old_id in old_ids:
            if old_id.startswith(prefix_old):
                id_map[old_id] = prefix_new + old_id[len(prefix_old) :]
            else:
                id_map[old_id] = old_id  # shouldn't happen, but be safe

        # Migrate ChromaDB entries (chunks only) — preserve embeddings
        if chunk_old_ids:
            batch_size = 500
            for i in range(0, len(chunk_old_ids), batch_size):
                batch = chunk_old_ids[i : i + batch_size]
                try:
                    results = self._collection.get(
                        ids=batch,
                        include=["embeddings", "documents", "metadatas"],
                    )
                except Exception:
                    logger.exception(
                        "Failed to fetch ChromaDB entries for rename '%s' → '%s'",
                        old_name,
                        new_name,
                    )
                    continue

                if not results["ids"]:
                    continue

                new_ids = [id_map[oid] for oid in results["ids"]]
                new_metadatas = [
                    {**meta, "source": new_name} for meta in results["metadatas"]
                ]

                with contextlib.suppress(Exception):
                    self._collection.delete(ids=results["ids"])

                try:
                    self._collection.add(
                        ids=new_ids,
                        embeddings=results["embeddings"],
                        documents=results["documents"],
                        metadatas=new_metadatas,
                    )
                except Exception:
                    logger.exception(
                        "Failed to re-add ChromaDB entries for rename '%s' → '%s'",
                        old_name,
                        new_name,
                    )

        # Migrate SQLite — update doc_id and source for each row
        with self._connect() as conn:
            for old_id, new_id in id_map.items():
                conn.execute(
                    "UPDATE documents SET doc_id = ?, source = ? WHERE doc_id = ?",
                    (new_id, new_name, old_id),
                )

        logger.info(
            "Renamed source '%s' → '%s': migrated %d documents",
            old_name,
            new_name,
            len(old_ids),
        )
        return len(old_ids)

    def get_all_source_names(self) -> set[str]:
        """Return the set of distinct source names in the KB."""
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT source FROM documents").fetchall()
        return {row["source"] for row in rows}

    def get_document_tree(self) -> list[dict[str, Any]]:
        """Return documents organized as a tree: source → category → documents.

        Categories are 'docs' and 'journal', determined by file_path patterns.
        Returns parent documents only (no chunks).
        """
        sql = """
            SELECT doc_id, source, file_path, title, created_at, modified_at, size_bytes
            FROM documents
            WHERE is_chunk = FALSE OR chunk_index IS NULL
            ORDER BY source, file_path
        """
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()

        sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            doc = dict(row)
            source = doc["source"]
            fp = doc.get("file_path", "")

            if "journal/" in fp or "journal\\" in fp:
                category = "journal"
            else:
                category = "docs"

            if source not in sources:
                sources[source] = {"docs": [], "journal": []}
            sources[source][category].append(doc)

        tree = []
        for source_name in sorted(sources):
            cats = sources[source_name]
            tree.append(
                {
                    "source": source_name,
                    "docs": sorted(cats["docs"], key=lambda d: d.get("title") or d.get("file_path", "")),
                    "journal": sorted(
                        cats["journal"],
                        key=lambda d: d.get("created_at") or d.get("file_path", ""),
                        reverse=True,
                    ),
                }
            )
        return tree

    def search_documents(
        self,
        query: str,
        n_results: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search and return parent document metadata (deduplicated from chunk hits).

        Uses ChromaDB for semantic search on chunks, then maps back to parent docs.
        """
        search_results = self.search(query, n_results=n_results * 2, source_filter=source_filter)

        seen_parents: set[str] = set()
        parent_docs: list[dict[str, Any]] = []

        for result in search_results:
            doc_id = result["doc_id"]
            parent_id = doc_id.split("#chunk")[0] if "#chunk" in doc_id else doc_id

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            doc = self.get_document(parent_id)
            if doc:
                parent_docs.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "file_path": doc["file_path"],
                        "title": doc["title"],
                        "created_at": doc["created_at"],
                        "modified_at": doc["modified_at"],
                        "score": result["score"],
                        "snippet": result["content"][:200],
                    }
                )

            if len(parent_docs) >= n_results:
                break

        return parent_docs

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
