"""Knowledge base layer combining SQLite (structured + BM25) and ChromaDB (dense vectors).

Search is a two-stage hybrid pipeline:

* **L1**: SQLite FTS5 BM25 over chunk content/title and ChromaDB cosine over chunk
  embeddings, fused with Reciprocal Rank Fusion (k=60). Top ``_L1_OUTPUT_SIZE``
  chunks pass to L2.
* **L2**: cross-encoder rerank (see :mod:`docserver.reranker`). Wired in by
  ``search`` and ``search_documents`` once the reranker module ships.

The legacy ``_keyword_search_title_path`` synthetic-score fallback was deleted —
FTS5 with a title-weighted BM25 query subsumes it without the score-scale hack.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

import chromadb

from docserver.embedding import OnnxEmbeddingFunction
from docserver.reranker import rerank as _rerank

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

-- Full-text BM25 index over chunk content. ``title`` is included so that
-- title-only matches still surface (BM25 weights it 2x via bm25(title,content)).
-- Tokenizer choice: ``unicode61`` (case + accent insensitive, no stemming) so
-- proper-noun queries like "strava" match the surface form rather than a
-- stemmed root.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    content,
    doc_id UNINDEXED,
    source UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

_MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN content_hash TEXT DEFAULT ''",
]

_CHROMA_COLLECTION = "documents"

# Hybrid search tuning. _RRF_CANDIDATE_K = top-K from each L1 leg before fusion.
# _L1_OUTPUT_SIZE = candidates handed to L2 rerank. _RRF_K = standard RRF
# damping (Cormack et al. 2009). _BM25_TITLE_WEIGHT > _BM25_CONTENT_WEIGHT lets
# title-only matches outrank weak content matches without crowding out strong
# content matches.
_RRF_CANDIDATE_K = 100
_L1_OUTPUT_SIZE = 50
_RRF_K = 60
_BM25_TITLE_WEIGHT = 2.0
_BM25_CONTENT_WEIGHT = 1.0

# Characters that are FTS5 query operators. Stripped from raw user queries so
# typing a literal `"foo*bar"` does not raise a syntax error.
_FTS5_OPERATOR_RE = re.compile(r'[\"\'\*\(\)\-:^]')


def _sanitize_fts_query(query: str) -> str:
    """Convert a free-text query to an FTS5-safe MATCH expression.

    Strips FTS5 operators, splits on whitespace, and rejoins as space-separated
    tokens (implicit AND between tokens in FTS5 default mode is not actually
    AND — it's "match any token in document"; that's the recall-friendly
    behaviour we want at L1 since RRF fusion does the precision work).

    Empty queries return ``'""'`` which matches nothing without raising.
    """
    cleaned = _FTS5_OPERATOR_RE.sub(" ", query)
    tokens = cleaned.split()
    return " ".join(tokens) if tokens else '""'


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

    def __init__(
        self,
        data_dir: str,
        *,
        chroma_host: str | None = None,
        chroma_port: int = 8000,
    ) -> None:
        """Initialise the knowledge base.

        SQLite always lives at ``{data_dir}/documents.db``. ChromaDB has two
        backends:
          - When ``chroma_host`` is set, uses ``chromadb.HttpClient`` against
            the sidecar service. This is the production path: the docserver
            and ingestion worker can both connect to the same Chroma server
            without corrupting the store.
          - When ``chroma_host`` is None, falls back to
            ``chromadb.PersistentClient`` at ``{data_dir}/chroma``. This is
            single-process only and used by the test suite.
        """
        logger.info(
            "Initializing knowledge base in %s",
            data_dir,
            extra={"event": "kb_init"},
        )
        os.makedirs(data_dir, exist_ok=True)

        self._db_path = os.path.join(data_dir, "documents.db")
        self._init_sqlite()
        logger.info("SQLite initialized at %s", self._db_path, extra={"event": "kb_init"})

        self._embedding_fn = OnnxEmbeddingFunction()
        if chroma_host:
            self._chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
            )
            logger.info(
                "ChromaDB connected via HttpClient to %s:%d",
                chroma_host,
                chroma_port,
                extra={
                    "event": "kb_init",
                    "chroma_host": chroma_host,
                    "chroma_port": chroma_port,
                },
            )
        else:
            chroma_dir = os.path.join(data_dir, "chroma")
            os.makedirs(chroma_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
            logger.info(
                "ChromaDB initialized as PersistentClient at %s",
                chroma_dir,
                extra={"event": "kb_init"},
            )

        self._collection = self._chroma_client.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            "ChromaDB collection ready: %s",
            _CHROMA_COLLECTION,
            extra={"event": "kb_init"},
        )

    def unload_embedding_model(self) -> bool:
        """Unload the embedding model to reclaim memory.

        The model will be transparently reloaded on the next search or upsert.
        Returns True if the model was actually unloaded.
        """
        return self._embedding_fn.unload()

    def ping_chroma(self) -> tuple[bool, str | None]:
        """Best-effort liveness check against the Chroma backend.

        Returns ``(True, None)`` on success and ``(False, error_message)``
        on failure. Wraps ``ClientAPI.heartbeat()`` and converts any
        exception into a structured result so the caller (typically the
        ``/health`` handler) can degrade gracefully without try/except
        plumbing at the call site. Synchronous: callers that need a hard
        wall-clock bound should run this via ``asyncio.to_thread`` with
        ``asyncio.wait_for``.
        """
        try:
            _ = self._chroma_client.heartbeat()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            # WAL mode lets the ingestion worker write to documents.db while
            # the server reads from it concurrently. This is a per-database
            # persistent setting (stored in the file header) so it survives
            # restarts and only needs to be set once, but setting it on every
            # init is idempotent and cheap.
            _ = conn.execute("PRAGMA journal_mode=WAL")
            _ = conn.execute("PRAGMA synchronous=NORMAL")
            _ = conn.executescript(_SCHEMA)
            self._run_migrations(conn)
            self._backfill_fts(conn)

    @staticmethod
    def _backfill_fts(conn: sqlite3.Connection) -> None:
        """Populate ``chunks_fts`` from existing chunk rows on first init.

        Idempotent: if the FTS table already has rows, this is a no-op. On a
        fresh database both tables are empty and the INSERT...SELECT yields
        zero rows, also a no-op. Only needs to do real work on the first
        deploy after the schema lands on a populated database.
        """
        row = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        if row and row[0] > 0:
            return
        _ = conn.execute(
            """
            INSERT INTO chunks_fts (title, content, doc_id, source)
            SELECT COALESCE(title, ''), COALESCE(content, ''), doc_id, source
            FROM documents
            WHERE is_chunk = TRUE AND content IS NOT NULL AND content != ''
            """
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # Re-assert busy timeout per connection — WAL still serialises writers,
        # so a 5s wait gives concurrent transactions room before we surface
        # SQLITE_BUSY to the caller.
        _ = conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _run_migrations(conn: sqlite3.Connection) -> None:
        """Apply schema migrations idempotently."""
        for sql in _MIGRATIONS:
            with contextlib.suppress(sqlite3.OperationalError):
                _ = conn.execute(sql)

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
                # Keep FTS5 in sync. Standard FTS5 (no content= option) owns
                # its data, so DELETE-then-INSERT is the upsert pattern.
                _ = conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
                _ = conn.execute(
                    "INSERT INTO chunks_fts (title, content, doc_id, source) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        str(metadata.get("title") or ""),
                        content,
                        doc_id,
                        str(metadata.get("source", "")),
                    ),
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

        # Pre-compute the FTS5 chunk rows so we can do them in the same
        # transaction as the documents table write.
        fts_chunk_rows: list[tuple[str, str, str, str]] = [
            (
                str(metadata.get("title") or ""),
                content,
                doc_id,
                str(metadata.get("source", "")),
            )
            for doc_id, content, metadata in items
            if content and metadata.get("is_chunk", False)
        ]

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
            if fts_chunk_rows:
                fts_doc_ids = [(row[2],) for row in fts_chunk_rows]
                _ = conn.executemany(
                    "DELETE FROM chunks_fts WHERE doc_id = ?", fts_doc_ids
                )
                _ = conn.executemany(
                    "INSERT INTO chunks_fts (title, content, doc_id, source) "
                    "VALUES (?, ?, ?, ?)",
                    fts_chunk_rows,
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
        """Delete a document from SQLite (incl. FTS5) and ChromaDB."""
        with self._connect() as conn:
            _ = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            _ = conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))

        with contextlib.suppress(Exception):
            # ChromaDB raises if the ID doesn't exist; that's fine.
            _ = self._collection.delete(ids=[doc_id])

    def delete_source_documents(self, source_name: str) -> int:
        """Delete all documents for a source. Returns the count deleted."""
        ids_to_delete = list(self.get_all_doc_ids_for_source(source_name))

        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE source = ?", (source_name,))
            count = cursor.rowcount
            _ = conn.execute("DELETE FROM chunks_fts WHERE source = ?", (source_name,))

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

    def _dense_search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Dense leg of L1: ChromaDB cosine over chunk embeddings.

        Returns chunk-level results ordered by cosine distance ascending
        (lower = more similar). The score field holds the raw distance.
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

    def _bm25_search_chunks(
        self,
        query: str,
        n_results: int = _RRF_CANDIDATE_K,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Lexical leg of L1: SQLite FTS5 BM25 over chunk content + title.

        Returns chunk-level results ordered by BM25 ascending (FTS5's
        ``bm25()`` returns negative floats; lower = better match). The score
        field holds the raw BM25 value.
        """
        match_expr = _sanitize_fts_query(query)
        if match_expr == '""':
            return []

        sql = (
            "SELECT doc_id, source, title, content, "
            "bm25(chunks_fts, ?, ?) AS score "
            "FROM chunks_fts "
            "WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [_BM25_TITLE_WEIGHT, _BM25_CONTENT_WEIGHT, match_expr]
        if source_filter:
            sql += " AND source = ?"
            params.append(source_filter)
        sql += " ORDER BY score LIMIT ?"
        params.append(n_results)

        try:
            with self._connect() as conn:
                rows = cast("list[sqlite3.Row]", conn.execute(sql, params).fetchall())
        except sqlite3.OperationalError:
            # Defensive: an unexpected FTS5 syntax error. Should not happen
            # after _sanitize_fts_query but fail safe rather than 500ing.
            logger.exception(
                "FTS5 BM25 query failed; returning empty results.",
                extra={"event": "bm25_search_failed", "query": query},
            )
            return []

        return [
            SearchResult(
                doc_id=str(row["doc_id"]),
                content=str(row["content"] or ""),
                metadata={
                    "source": str(row["source"] or ""),
                    "title": str(row["title"] or ""),
                },
                score=float(row["score"]),
            )
            for row in rows
        ]

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[SearchResult]], k: int = _RRF_K,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of multiple ranked candidate lists.

        Each list is assumed to be ordered best-first. RRF score for a doc_id
        is ``sum(1 / (k + rank))`` over every list it appears in (1-indexed).
        Returns ``[(doc_id, rrf_score)]`` sorted by score descending.

        k=60 is the standard default (Cormack et al. 2009). Sidesteps the
        score-scale problem between cosine distance and BM25 logits — only
        rank position matters.
        """
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, result in enumerate(ranked, start=1):
                doc_id = result["doc_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid chunk-level search: BM25 + dense, fused with RRF, then L2 rerank.

        Used by the chat agent's ``search_docs`` tool. Returns chunk-level
        results ordered by cross-encoder logit descending. The score field
        holds the rerank logit (higher = better). If the reranker is
        unavailable the L1-fused order is returned with RRF scores.
        """
        dense = self._dense_search(
            query, n_results=_RRF_CANDIDATE_K, source_filter=source_filter,
        )
        bm25 = self._bm25_search_chunks(
            query, n_results=_RRF_CANDIDATE_K, source_filter=source_filter,
        )

        # Build a lookup of doc_id → SearchResult. Prefer the dense entry for
        # content/metadata when a doc appears in both lists (its content is
        # the actual chunk text Chroma indexed; BM25 returns the same text but
        # via a different round-trip).
        by_id: dict[str, SearchResult] = {r["doc_id"]: r for r in bm25}
        by_id.update({r["doc_id"]: r for r in dense})

        fused = self._rrf_fuse([dense, bm25])
        l1_top: list[SearchResult] = []
        for doc_id, rrf_score in fused[:_L1_OUTPUT_SIZE]:
            base = by_id.get(doc_id)
            if base is None:
                continue
            l1_top.append(
                SearchResult(
                    doc_id=base["doc_id"],
                    content=base["content"],
                    metadata=base["metadata"],
                    score=rrf_score,
                )
            )

        return _rerank(query, l1_top, top_k=n_results)

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

        # Migrate SQLite — update doc_id and source for each row, including
        # the chunks_fts shadow rows.
        with self._connect() as conn:
            for old_id, new_id in id_map.items():
                _ = conn.execute(
                    "UPDATE documents SET doc_id = ?, source = ? WHERE doc_id = ?",
                    (new_id, new_name, old_id),
                )
                _ = conn.execute(
                    "UPDATE chunks_fts SET doc_id = ?, source = ? WHERE doc_id = ?",
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

    def search_documents(
        self,
        query: str,
        n_results: int = 20,
        source_filter: str | None = None,
    ) -> list[dict[str, _Scalar]]:
        """Hybrid parent-doc search: BM25 + dense + RRF, deduplicated to parents.

        L1 fuses dense and BM25 candidates with RRF; the top
        ``_L1_OUTPUT_SIZE`` chunks are passed to L2 rerank (wired in by
        ``docserver.reranker``) — until the reranker module ships, the
        post-fusion chunk order is used directly. Chunks are then deduped to
        their parent docs (best-scoring chunk wins) and the top ``n_results``
        parents are returned.

        The returned ``score`` field is the RRF score (higher = better) — a
        contract change from the legacy ChromaDB-distance scale.
        """
        dense = self._dense_search(
            query, n_results=_RRF_CANDIDATE_K, source_filter=source_filter,
        )
        bm25 = self._bm25_search_chunks(
            query, n_results=_RRF_CANDIDATE_K, source_filter=source_filter,
        )

        by_id: dict[str, SearchResult] = {r["doc_id"]: r for r in bm25}
        by_id.update({r["doc_id"]: r for r in dense})

        fused = self._rrf_fuse([dense, bm25])
        l1_top: list[SearchResult] = []
        for doc_id, rrf_score in fused[:_L1_OUTPUT_SIZE]:
            base = by_id.get(doc_id)
            if base is None:
                continue
            l1_top.append(
                SearchResult(
                    doc_id=base["doc_id"],
                    content=base["content"],
                    metadata=base["metadata"],
                    score=rrf_score,
                )
            )

        # L2 rerank reorders all L1 candidates; dedup-to-parent then takes
        # the highest-scoring chunk per parent. Reranking before dedup means
        # the cross-encoder sees full chunk content, not just the first
        # chunk per parent.
        reranked = _rerank(query, l1_top, top_k=len(l1_top))

        seen_parents: set[str] = set()
        parent_docs: list[dict[str, _Scalar]] = []
        for chunk in reranked:
            doc_id = chunk["doc_id"]
            parent_id = doc_id.split("#chunk")[0] if "#chunk" in doc_id else doc_id
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            doc = self.get_document(parent_id)
            if doc is None:
                continue
            parent_docs.append(
                {
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "file_path": doc["file_path"],
                    "title": doc["title"],
                    "created_at": doc["created_at"],
                    "modified_at": doc["modified_at"],
                    "score": chunk["score"],
                    "snippet": str(chunk["content"])[:200],
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

    def get_source_files(self, source: str) -> list[dict[str, _Scalar]]:
        """Return parent-doc rows for *source*, ordered by file_path.

        Powers the webapp's folder-tree view: callers build the nested
        structure client-side by splitting ``file_path`` on ``/``. Chunks
        are excluded — only one row per source file is returned.
        """
        rows = self._fetchall(
            """
            SELECT doc_id, file_path, title, modified_at
            FROM documents
            WHERE source = ?
              AND (is_chunk = FALSE OR chunk_index IS NULL)
            ORDER BY file_path
            """,
            (source,),
        )
        return _rows_to_dicts(rows)

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
