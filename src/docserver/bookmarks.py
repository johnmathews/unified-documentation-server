"""Server-side bookmark/favourite persistence.

Stores document bookmarks in SQLite with a user_id column to
prepare for future multi-user support. Currently defaults to
a single "default" user.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TypedDict

logger = logging.getLogger(__name__)


class Bookmark(TypedDict):
    doc_id: str
    user_id: str
    created_at: str


class BookmarkStore:
    """SQLite-backed bookmark storage."""

    def __init__(self, data_dir: str) -> None:
        import os

        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "bookmarks.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bookmarks (
                    doc_id     TEXT NOT NULL,
                    user_id    TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, doc_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bookmarks_user "
                "ON bookmarks(user_id)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, doc_id: str, user_id: str = "default") -> Bookmark:
        """Add a bookmark. Returns the bookmark dict. Idempotent."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO bookmarks (doc_id, user_id, created_at) "
                "VALUES (?, ?, ?)",
                (doc_id, user_id, now),
            )
            # Return the actual row (may have pre-existing created_at)
            row = conn.execute(
                "SELECT doc_id, user_id, created_at FROM bookmarks "
                "WHERE doc_id = ? AND user_id = ?",
                (doc_id, user_id),
            ).fetchone()
        logger.info(
            "Bookmark added: doc_id=%s user_id=%s",
            doc_id,
            user_id,
            extra={"event": "bookmark_add", "doc_id": doc_id, "user_id": user_id},
        )
        return Bookmark(
            doc_id=row["doc_id"],
            user_id=row["user_id"],
            created_at=row["created_at"],
        )

    def remove(self, doc_id: str, user_id: str = "default") -> bool:
        """Remove a bookmark. Returns False if it didn't exist."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM bookmarks WHERE doc_id = ? AND user_id = ?",
                (doc_id, user_id),
            )
        removed = cursor.rowcount > 0
        if removed:
            logger.info(
                "Bookmark removed: doc_id=%s user_id=%s",
                doc_id,
                user_id,
                extra={"event": "bookmark_remove", "doc_id": doc_id, "user_id": user_id},
            )
        return removed

    def list_all(self, user_id: str = "default") -> list[Bookmark]:
        """List all bookmarks for a user, most recently added first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, user_id, created_at FROM bookmarks "
                "WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [
            Bookmark(
                doc_id=row["doc_id"],
                user_id=row["user_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def is_bookmarked(self, doc_id: str, user_id: str = "default") -> bool:
        """Check if a single document is bookmarked."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM bookmarks WHERE doc_id = ? AND user_id = ?",
                (doc_id, user_id),
            ).fetchone()
        return row is not None

    def bulk_check(
        self, doc_ids: list[str], user_id: str = "default"
    ) -> dict[str, bool]:
        """Check bookmark status for multiple doc_ids at once."""
        if not doc_ids:
            return {}
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in doc_ids)
            rows = conn.execute(
                f"SELECT doc_id FROM bookmarks WHERE user_id = ? AND doc_id IN ({placeholders})",
                [user_id, *doc_ids],
            ).fetchall()
        bookmarked = {row["doc_id"] for row in rows}
        return {doc_id: doc_id in bookmarked for doc_id in doc_ids}

    def close(self) -> None:
        """No-op — connections are opened per-call."""
