"""Server-side conversation persistence for the chat agent.

Stores chat conversations in SQLite so they can be reviewed, resumed,
and used to improve the chat experience over time.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


# Default cap on retained conversations. Each row stores the full message
# list as JSON, so an unbounded table grows linearly with chat usage and
# eventually consumes meaningful disk on the data volume. Single-user
# homelab scale fits well under 1000; long-running deployments should
# tune via the constructor or DOCSERVER_MAX_CONVERSATIONS env var.
DEFAULT_MAX_CONVERSATIONS = 1000


class ConversationSummary(TypedDict):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str


class Conversation(TypedDict):
    id: str
    title: str
    created_at: str
    updated_at: str
    page_context: dict[str, str] | None
    messages: list[dict[str, str]]


def _generate_title(messages: list[dict[str, str]]) -> str:
    """Generate a conversation title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            text = msg["content"].strip()
            if len(text) <= 60:
                return text
            return text[:57] + "..."
    return "Untitled conversation"


class ConversationStore:
    """SQLite-backed conversation storage."""

    def __init__(
        self,
        data_dir: str,
        *,
        max_conversations: int | None = None,
    ) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "conversations.db")
        if max_conversations is None:
            env_val = os.environ.get("DOCSERVER_MAX_CONVERSATIONS")
            try:
                max_conversations = int(env_val) if env_val else DEFAULT_MAX_CONVERSATIONS
            except ValueError:
                logger.warning(
                    "Invalid DOCSERVER_MAX_CONVERSATIONS=%r; using default %d.",
                    env_val,
                    DEFAULT_MAX_CONVERSATIONS,
                    extra={"event": "conversations_max_invalid"},
                )
                max_conversations = DEFAULT_MAX_CONVERSATIONS
        self.max_conversations = max(1, max_conversations)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    page_context TEXT,
                    messages TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_updated "
                "ON conversations(updated_at DESC)"
            )

    def create(
        self,
        messages: list[dict[str, str]],
        page_context: dict[str, str] | None = None,
    ) -> str:
        """Create a new conversation. Returns the conversation ID."""
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()
        title = _generate_title(messages)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, page_context, messages) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conv_id,
                    title,
                    now,
                    now,
                    json.dumps(page_context) if page_context else None,
                    json.dumps(messages),
                ),
            )
            pruned = self._prune_to_cap(conn)
        logger.info(
            "Created conversation %s: %s%s",
            conv_id,
            title,
            f" (pruned {pruned} oldest)" if pruned else "",
            extra={
                "event": "conversation_create",
                "conversation_id": conv_id,
                "pruned": pruned,
            },
        )
        return conv_id

    def _prune_to_cap(self, conn: sqlite3.Connection) -> int:
        """Delete any rows beyond ``self.max_conversations``, dropping the
        oldest first (by ``updated_at``). Returns the number of rows
        deleted. Called from ``create()`` so the cap is enforced at the
        only path that adds rows.
        """
        # SQLite supports `LIMIT -1 OFFSET N` for "everything after the
        # first N rows" — used here to identify the rows beyond the cap
        # in the inverse-by-updated_at ordering, and delete them by id.
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id IN ("
            "  SELECT id FROM conversations "
            "  ORDER BY updated_at DESC, created_at DESC "
            "  LIMIT -1 OFFSET ?"
            ")",
            (self.max_conversations,),
        )
        return cursor.rowcount

    def update(
        self,
        conv_id: str,
        messages: list[dict[str, str]],
        page_context: dict[str, str] | None = None,
    ) -> bool:
        """Update conversation messages. Returns False if not found."""
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE conversations SET messages = ?, updated_at = ?, page_context = COALESCE(?, page_context) "
                "WHERE id = ?",
                (
                    json.dumps(messages),
                    now,
                    json.dumps(page_context) if page_context else None,
                    conv_id,
                ),
            )
            return cursor.rowcount > 0

    def get(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, page_context, messages "
                "FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if row is None:
                return None
            return Conversation(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                page_context=json.loads(row["page_context"]) if row["page_context"] else None,
                messages=json.loads(row["messages"]),
            )

    def list_all(self, limit: int = 50) -> list[ConversationSummary]:
        """List conversations, most recent first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at, messages "
                "FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result: list[ConversationSummary] = []
        for row in rows:
            messages: list[dict[str, Any]] = json.loads(row["messages"])  # pyright: ignore[reportExplicitAny]
            # Preview: last assistant message, truncated
            preview = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    text = msg["content"].strip()
                    preview = text[:100] + "..." if len(text) > 100 else text
                    break

            result.append(
                ConversationSummary(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=len(messages),
                    preview=preview,
                )
            )
        return result

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation. Returns False if not found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            return cursor.rowcount > 0

    def close(self) -> None:
        """No-op — connections are opened per-call."""
