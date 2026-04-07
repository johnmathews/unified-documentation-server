"""Knowledge base layer combining SQLite (structured metadata) and ChromaDB (vector search)."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

import chromadb

from docserver.embedding import OnnxEmbeddingFunction

if TYPE_CHECKING:
    from chromadb.api.types import Metadata, Where

logger = logging.getLogger(__name__)

# Scalar types stored in SQLite columns and document metadata dicts.
_Scalar = str | int | float | bool | None


class SearchResult(TypedDict):
    doc_id: str
    content: str
    metadata: dict[str, _Scalar]
    score: float


class SourceSummary(TypedDict):
    source: str
    file_count: int
    chunk_count: int
    last_indexed: str | None


class SourceStatus(TypedDict):
    source: str
    last_checked: str | None
    last_error: str | None
    last_error_at: str | None
    consecutive_failures: int


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

CREATE TABLE IF NOT EXISTS source_status (
    source               TEXT PRIMARY KEY,
    last_checked         TEXT,
    last_error           TEXT,
    last_error_at        TEXT,
    consecutive_failures INTEGER DEFAULT 0
);
"""

_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN content_hash TEXT DEFAULT ''",
]

_CHROMA_COLLECTION = "documents"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, _Scalar]]:
    """Convert sqlite3.Row list to typed dicts."""
    return [cast("dict[str, _Scalar]", dict(row)) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, _Scalar]:
    """Convert a single sqlite3.Row to a typed dict."""
    return cast("dict[str, _Scalar]", dict(row))


class KnowledgeBase:
    """Combines SQLite for structured metadata and ChromaDB for semantic search."""

    _db_path: str
    _chroma_client: chromadb.ClientAPI
    _collection: chromadb.Collection
    _embedding_fn: OnnxEmbeddingFunction

    _CHROMA_BATCH_SIZE: ClassVar[int] = 64  # max chunks per ChromaDB upsert call

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
            _ = conn.executescript(_SCHEMA)
            self._run_migrations(conn)

    @staticmethod
    def _run_migrations(conn: sqlite3.Connection) -> None:
        """Apply schema migrations idempotently."""
        for sql in _MIGRATIONS:
            with contextlib.suppress(sqlite3.OperationalError):
                _ = conn.execute(sql)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows as sqlite3.Row objects."""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return cast("list[sqlite3.Row]", rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_document(self, doc_id: str, content: str, metadata: dict[str, _Scalar]) -> None:
        """Insert or replace a document in SQLite; also index chunks in ChromaDB."""
        indexed_at = _now_iso()
        is_chunk = bool(metadata.get("is_chunk", False))

        with self._connect() as conn:
            _ = conn.execute(
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
            chroma_meta: Metadata = {
                "source": str(metadata.get("source", "")),
                "file_path": str(metadata.get("file_path", "")),
                "title": str(metadata.get("title") or ""),
                "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                "total_chunks": int(metadata.get("total_chunks", 1) or 1),
                "is_chunk": True,
                "section_path": str(metadata.get("section_path", "")),
            }
            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[chroma_meta],
            )

    def upsert_documents_batch(
        self,
        items: list[tuple[str, str, dict[str, _Scalar]]],
    ) -> None:
        """Batch-upsert documents: SQLite in one transaction, ChromaDB in batches.

        *items* is a list of ``(doc_id, content, metadata)`` tuples — the same
        arguments as :meth:`upsert_document`.  Batching ChromaDB upserts lets the
        embedding model process many documents at once (batch_size=32 internally),
        which is dramatically faster than one-at-a-time.

        ChromaDB calls are capped at :attr:`_CHROMA_BATCH_SIZE` chunks per call
        to keep memory usage bounded regardless of input size.
        """
        if not items:
            return

        indexed_at = _now_iso()

        # --- SQLite: single transaction ---
        rows: list[dict[str, _Scalar]] = []
        for doc_id, content, metadata in items:
            rows.append(
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
                    "is_chunk": bool(metadata.get("is_chunk", False)),
                    "section_path": metadata.get("section_path", ""),
                    "content_hash": metadata.get("content_hash", ""),
                }
            )

        with self._connect() as conn:
            _ = conn.executemany(
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
                rows,
            )

        # --- ChromaDB: batch upsert for chunks only ---
        chroma_ids: list[str] = []
        chroma_docs: list[str] = []
        chroma_metas: list[Metadata] = []

        for doc_id, content, metadata in items:
            if not content or not metadata.get("is_chunk", False):
                continue
            chroma_ids.append(doc_id)
            chroma_docs.append(content)
            chroma_metas.append(
                {
                    "source": str(metadata.get("source", "")),
                    "file_path": str(metadata.get("file_path", "")),
                    "title": str(metadata.get("title") or ""),
                    "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                    "total_chunks": int(metadata.get("total_chunks", 1) or 1),
                    "is_chunk": True,
                    "section_path": str(metadata.get("section_path", "")),
                }
            )

        # Send to ChromaDB in capped batches to bound memory for embeddings.
        bs = self._CHROMA_BATCH_SIZE
        for i in range(0, len(chroma_ids), bs):
            self._collection.upsert(
                ids=chroma_ids[i : i + bs],
                documents=chroma_docs[i : i + bs],
                metadatas=chroma_metas[i : i + bs],
            )

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from both SQLite and ChromaDB."""
        with self._connect() as conn:
            _ = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        with contextlib.suppress(Exception):
            # ChromaDB raises if the ID doesn't exist; that's fine.
            _ = self._collection.delete(ids=[doc_id])

    def delete_source_documents(self, source_name: str) -> int:
        """Delete all documents for a source. Returns the count deleted."""
        ids_to_delete = list(self.get_all_doc_ids_for_source(source_name))

        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE source = ?", (source_name,))
            count = cursor.rowcount

        if ids_to_delete:
            with contextlib.suppress(Exception):
                _ = self._collection.delete(ids=ids_to_delete)

        return count

    def get_all_doc_ids_for_source(self, source_name: str) -> set[str]:
        """Return all doc_ids belonging to a source."""
        rows = self._fetchall(
            "SELECT doc_id FROM documents WHERE source = ?", (source_name,)
        )
        return {str(row["doc_id"]) for row in rows}

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
        where: Where | None = None
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
        ids = results["ids"][0] if results["ids"] else []
        raw_documents = results.get("documents")
        raw_metadatas = results.get("metadatas")
        raw_distances = results.get("distances")
        documents = raw_documents[0] if raw_documents else []
        metadatas = raw_metadatas[0] if raw_metadatas else []
        distances = raw_distances[0] if raw_distances else []

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances, strict=True):
            output.append(
                SearchResult(
                    doc_id=str(doc_id),
                    content=str(doc_text),
                    metadata=cast("dict[str, _Scalar]", dict(meta)),
                    score=float(dist),
                )
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
    ) -> list[dict[str, _Scalar]]:
        """Structured SQL query returning parent docs only (no chunks).

        Returns a list of metadata dicts (no content field).
        """
        conditions: list[str] = ["(is_chunk = FALSE OR chunk_index IS NULL)"]
        params: list[str | int] = []

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

        rows = self._fetchall(sql, tuple(params))
        return _rows_to_dicts(rows)

    def get_document(self, doc_id: str) -> dict[str, _Scalar] | None:
        """Fetch a single document by ID, including content."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

        if row is None:
            return None
        return _row_to_dict(cast("sqlite3.Row", row))

    def get_full_document(self, doc_id: str) -> dict[str, _Scalar] | None:
        """Fetch a document with full content reassembled from chunks.

        Parent documents are stored with empty content — the actual text lives
        in chunk rows.  This method fetches the parent metadata and joins the
        chunk content back together in order so the UI can render the full doc.
        """
        doc = self.get_document(doc_id)
        if doc is None:
            return None

        # If the doc already has content (or is itself a chunk), return as-is.
        if doc.get("content"):
            return doc

        # Reassemble from chunks ordered by chunk_index.
        rows = self._fetchall(
            "SELECT content FROM documents "
            + "WHERE doc_id LIKE ? AND is_chunk = TRUE "
            + "ORDER BY chunk_index",
            (f"{doc_id}#chunk%",),
        )

        if rows:
            doc["content"] = "\n\n".join(
                str(row["content"]) for row in rows if row["content"]
            )

        return doc

    def get_indexed_content_hashes(self, source: str) -> dict[str, str]:
        """Return {doc_id: content_hash} for all parent docs in a source."""
        rows = self._fetchall(
            "SELECT doc_id, content_hash FROM documents WHERE source = ? AND (is_chunk = FALSE OR chunk_index IS NULL)",
            (source,),
        )
        return {str(row["doc_id"]): str(row["content_hash"]) for row in rows}

    def rename_source(self, old_name: str, new_name: str) -> int:
        """Rename a source: update doc_ids, source column, and ChromaDB entries.

        Preserves embeddings in ChromaDB to avoid expensive re-computation.
        Returns the number of documents migrated.
        """
        rows = self._fetchall(
            "SELECT doc_id, is_chunk FROM documents WHERE source = ?", (old_name,),
        )

        if not rows:
            return 0

        old_ids = [str(row["doc_id"]) for row in rows]
        chunk_old_ids = [str(row["doc_id"]) for row in rows if row["is_chunk"]]

        # Build old→new ID mapping: replace source name prefix in doc_id
        prefix_old = f"{old_name}:"
        prefix_new = f"{new_name}:"
        id_map: dict[str, str] = {}
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
                old_metadatas = results["metadatas"] or []
                new_metadatas: list[Metadata] = [
                    {**meta, "source": new_name} for meta in old_metadatas
                ]

                with contextlib.suppress(Exception):
                    _ = self._collection.delete(ids=results["ids"])

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
                _ = conn.execute(
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
        rows = self._fetchall("SELECT DISTINCT source FROM documents")
        return {str(row["source"]) for row in rows}

    def get_document_tree(self) -> list[dict[str, _Scalar | list[dict[str, _Scalar]]]]:
        """Return documents organized as a tree: source → category → documents.

        Categories are 'root_docs', 'docs', 'journal', 'engineering_team', 'pdf', 'learning_journal', 'research', 'skills', and 'runbooks', determined by file_path patterns.
        Returns parent documents only (no chunks).
        """
        sql = """
            SELECT doc_id, source, file_path, title, created_at, modified_at, size_bytes
            FROM documents
            WHERE is_chunk = FALSE OR chunk_index IS NULL
            ORDER BY source, file_path
        """
        rows = self._fetchall(sql)

        sources: dict[str, dict[str, list[dict[str, _Scalar]]]] = {}
        for row in rows:
            doc = _row_to_dict(row)
            source = str(doc["source"])
            fp = str(doc.get("file_path", ""))

            if fp.lower().endswith(".pdf"):
                category = "pdf"
            elif "journal/" in fp or "journal\\" in fp:
                category = "journal"
            elif ".engineering-team/" in fp or ".engineering-team\\" in fp:
                category = "engineering_team"
            elif "learning/" in fp or "learning\\" in fp:
                category = "learning_journal"
            elif "research/" in fp or "research\\" in fp:
                category = "research"
            elif "skills/" in fp or "skills\\" in fp:
                category = "skills"
            elif "runbooks/" in fp or "runbooks\\" in fp:
                category = "runbooks"
            elif "/" in fp or "\\" in fp:
                # File is inside a subdirectory (e.g. docs/foo.md)
                category = "docs"
            else:
                # Root-level file (e.g. README.md)
                category = "root_docs"

            if source not in sources:
                sources[source] = {"root_docs": [], "docs": [], "journal": [], "engineering_team": [], "pdf": [], "learning_journal": [], "research": [], "skills": [], "runbooks": []}
            sources[source][category].append(doc)

        def _sort_key_title(d: dict[str, _Scalar]) -> str:
            return str(d.get("title") or d.get("file_path", ""))

        def _sort_key_created(d: dict[str, _Scalar]) -> str:
            return str(d.get("created_at") or d.get("file_path", ""))

        tree: list[dict[str, _Scalar | list[dict[str, _Scalar]]]] = []
        for source_name in sorted(sources):
            cats = sources[source_name]
            tree.append(
                {
                    "source": source_name,
                    "root_docs": sorted(cats["root_docs"], key=_sort_key_title),
                    "docs": sorted(cats["docs"], key=_sort_key_title),
                    "journal": sorted(cats["journal"], key=_sort_key_created, reverse=True),
                    "engineering_team": sorted(cats["engineering_team"], key=_sort_key_title),
                    "pdf": sorted(cats["pdf"], key=_sort_key_title),
                    "learning_journal": sorted(cats["learning_journal"], key=_sort_key_created, reverse=True),
                    "research": sorted(cats["research"], key=_sort_key_created, reverse=True),
                    "skills": sorted(cats["skills"], key=_sort_key_title),
                    "runbooks": sorted(cats["runbooks"], key=_sort_key_title),
                }
            )
        return tree

    def _keyword_search_title_path(
        self,
        query: str,
        source_filter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, _Scalar]]:
        """Keyword search on title and file_path via SQLite LIKE.

        Returns parent documents whose title or file_path contain the query
        (case-insensitive). Results are given a synthetic score of 0.5 so they
        sort after strong semantic matches but before weak ones.
        """
        sql = """
            SELECT doc_id, source, file_path, title, created_at, modified_at,
                   SUBSTR(content, 1, 200) AS snippet
            FROM documents
            WHERE (is_chunk = FALSE OR chunk_index IS NULL)
              AND (title LIKE :pattern OR file_path LIKE :pattern)
        """
        params: dict[str, str | int] = {"pattern": f"%{query}%"}
        if source_filter:
            sql += " AND source = :source"
            params["source"] = source_filter
        sql += " ORDER BY modified_at DESC LIMIT :limit"
        params["limit"] = limit

        with self._connect() as conn:
            result_rows = cast(
                "list[sqlite3.Row]", conn.execute(sql, params).fetchall()
            )

        return [
            {
                "doc_id": str(row["doc_id"]),
                "source": str(row["source"]),
                "file_path": str(row["file_path"]),
                "title": str(row["title"]),
                "created_at": row["created_at"],
                "modified_at": row["modified_at"],
                "score": 0.5,
                "snippet": str(row["snippet"] or ""),
            }
            for row in result_rows
        ]

    def search_documents(
        self,
        query: str,
        n_results: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, _Scalar]]:
        """Search and return parent document metadata (deduplicated from chunk hits).

        Combines ChromaDB semantic search on chunks with keyword search on
        title and file_path, then deduplicates and limits to n_results.
        """
        search_results = self.search(query, n_results=n_results * 2, source_filter=source_filter)

        seen_parents: set[str] = set()
        parent_docs: list[dict[str, _Scalar]] = []

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
                        "snippet": str(result["content"])[:200],
                    }
                )

        # Complement with keyword matches on title and file_path
        keyword_hits = self._keyword_search_title_path(
            query, source_filter=source_filter, limit=n_results,
        )
        for hit in keyword_hits:
            if hit["doc_id"] not in seen_parents:
                seen_parents.add(str(hit["doc_id"]))
                parent_docs.append(hit)

        # Sort by score (lower = more relevant for ChromaDB distances)
        parent_docs.sort(key=lambda d: float(d.get("score", 0) or 0))

        return parent_docs[:n_results]

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
        rows = self._fetchall(sql)

        return [
            SourceSummary(
                source=str(row["source"]),
                file_count=int(row["file_count"]),
                chunk_count=int(row["chunk_count"]),
                last_indexed=cast("str | None", row["last_indexed"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Source status tracking
    # ------------------------------------------------------------------

    def update_source_check(
        self, source: str, *, error: str | None = None,
    ) -> None:
        """Record the result of a scan attempt for *source*.

        On success (``error`` is None): set ``last_checked`` to now and reset
        ``consecutive_failures`` to 0, and clear the last error.
        On failure: increment ``consecutive_failures`` and record the error.
        """
        now = _now_iso()
        with self._connect() as conn:
            if error is None:
                conn.execute(
                    """INSERT INTO source_status
                           (source, last_checked, last_error, last_error_at, consecutive_failures)
                       VALUES (?, ?, NULL, NULL, 0)
                       ON CONFLICT(source) DO UPDATE SET
                           last_checked = excluded.last_checked,
                           last_error = NULL,
                           last_error_at = NULL,
                           consecutive_failures = 0
                    """,
                    (source, now),
                )
            else:
                conn.execute(
                    """INSERT INTO source_status
                           (source, last_checked, last_error, last_error_at, consecutive_failures)
                       VALUES (?, NULL, ?, ?, 1)
                       ON CONFLICT(source) DO UPDATE SET
                           last_error = excluded.last_error,
                           last_error_at = excluded.last_error_at,
                           consecutive_failures = consecutive_failures + 1
                    """,
                    (source, error, now),
                )

    def get_source_statuses(self) -> dict[str, SourceStatus]:
        """Return status records for all sources, keyed by source name."""
        rows = self._fetchall(
            "SELECT source, last_checked, last_error, last_error_at, consecutive_failures "
            "FROM source_status"
        )
        return {
            str(row["source"]): SourceStatus(
                source=str(row["source"]),
                last_checked=cast("str | None", row["last_checked"]),
                last_error=cast("str | None", row["last_error"]),
                last_error_at=cast("str | None", row["last_error_at"]),
                consecutive_failures=int(row["consecutive_failures"]),
            )
            for row in rows
        }

    def close(self) -> None:
        """Close connections. ChromaDB PersistentClient manages its own lifecycle."""
        # SQLite connections are managed per-operation via context managers.
        # ChromaDB's PersistentClient does not expose an explicit close method
        # in all versions; call it if available.
        close_fn = getattr(self._chroma_client, "close", None)
        if callable(close_fn):
            _ = close_fn()
