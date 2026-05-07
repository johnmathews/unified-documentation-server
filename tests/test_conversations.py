"""Tests for server-side conversation persistence."""

import os

import pytest

from docserver.conversations import ConversationStore, _generate_title


@pytest.fixture
def store(tmp_path):
    """Create a ConversationStore backed by a temp directory."""
    return ConversationStore(str(tmp_path))


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "What is m3?"},
        {"role": "assistant", "content": "M3 is a monitoring service."},
    ]


# ---- _generate_title --------------------------------------------------------


class TestGenerateTitle:
    def test_uses_first_user_message(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        assert _generate_title(messages) == "Hello world"

    def test_truncates_long_messages(self):
        long_msg = "x" * 100
        messages = [{"role": "user", "content": long_msg}]
        title = _generate_title(messages)
        assert len(title) == 60
        assert title.endswith("...")

    def test_short_message_not_truncated(self):
        messages = [{"role": "user", "content": "Short question"}]
        assert _generate_title(messages) == "Short question"

    def test_exactly_60_chars_not_truncated(self):
        msg = "x" * 60
        messages = [{"role": "user", "content": msg}]
        assert _generate_title(messages) == msg

    def test_no_user_message(self):
        messages = [{"role": "assistant", "content": "Hello"}]
        assert _generate_title(messages) == "Untitled conversation"

    def test_empty_messages(self):
        assert _generate_title([]) == "Untitled conversation"


# ---- ConversationStore -------------------------------------------------------


class TestConversationStore:
    def test_create_returns_id(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        assert isinstance(conv_id, str)
        assert len(conv_id) == 12

    def test_get_returns_conversation(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        conv = store.get(conv_id)
        assert conv is not None
        assert conv["id"] == conv_id
        assert conv["title"] == "What is m3?"
        assert conv["messages"] == sample_messages
        assert conv["created_at"] is not None
        assert conv["updated_at"] is not None

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_create_with_page_context(self, store, sample_messages):
        ctx = {"source": "m3", "category": "journal"}
        conv_id = store.create(sample_messages, page_context=ctx)
        conv = store.get(conv_id)
        assert conv is not None
        assert conv["page_context"] == ctx

    def test_create_without_page_context(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        conv = store.get(conv_id)
        assert conv is not None
        assert conv["page_context"] is None

    def test_update_messages(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        extended = [
            *sample_messages,
            {"role": "user", "content": "Tell me more"},
            {"role": "assistant", "content": "Sure, here's more info."},
        ]
        assert store.update(conv_id, extended) is True
        conv = store.get(conv_id)
        assert conv is not None
        assert len(conv["messages"]) == 4

    def test_update_missing_returns_false(self, store):
        assert store.update("nonexistent", []) is False

    def test_update_with_page_context(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        ctx = {"source": "m3"}
        store.update(conv_id, sample_messages, page_context=ctx)
        conv = store.get(conv_id)
        assert conv is not None
        assert conv["page_context"] == ctx

    def test_list_all_returns_summaries(self, store, sample_messages):
        store.create(sample_messages)
        store.create([{"role": "user", "content": "Second conversation"}])
        convs = store.list_all()
        assert len(convs) == 2
        # Most recent first
        assert convs[0]["title"] == "Second conversation"
        assert convs[1]["title"] == "What is m3?"

    def test_list_all_includes_message_count(self, store, sample_messages):
        store.create(sample_messages)
        convs = store.list_all()
        assert convs[0]["message_count"] == 2

    def test_list_all_includes_preview(self, store, sample_messages):
        store.create(sample_messages)
        convs = store.list_all()
        assert "M3 is a monitoring service" in convs[0]["preview"]

    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_list_all_respects_limit(self, store):
        for i in range(5):
            store.create([{"role": "user", "content": f"Conv {i}"}])
        assert len(store.list_all(limit=3)) == 3

    def test_delete_existing(self, store, sample_messages):
        conv_id = store.create(sample_messages)
        assert store.delete(conv_id) is True
        assert store.get(conv_id) is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_db_file_created(self, tmp_path):
        ConversationStore(str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "conversations.db"))

    def test_multiple_stores_same_dir(self, tmp_path, sample_messages):
        """Two stores pointing to the same dir share data."""
        store1 = ConversationStore(str(tmp_path))
        conv_id = store1.create(sample_messages)
        store2 = ConversationStore(str(tmp_path))
        conv = store2.get(conv_id)
        assert conv is not None
        assert conv["title"] == "What is m3?"


class TestRowCap:
    """The store caps how many conversations are retained so the SQLite
    file doesn't grow forever. The cap is enforced on insert; oldest rows
    (by updated_at) are pruned first."""

    def test_default_cap_used_when_unspecified(self, tmp_path):
        store = ConversationStore(str(tmp_path))
        from docserver.conversations import DEFAULT_MAX_CONVERSATIONS

        assert store.max_conversations == DEFAULT_MAX_CONVERSATIONS

    def test_explicit_cap_overrides_default(self, tmp_path):
        store = ConversationStore(str(tmp_path), max_conversations=42)
        assert store.max_conversations == 42

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOCSERVER_MAX_CONVERSATIONS", "7")
        store = ConversationStore(str(tmp_path))
        assert store.max_conversations == 7

    def test_invalid_env_var_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DOCSERVER_MAX_CONVERSATIONS", "not-a-number")
        store = ConversationStore(str(tmp_path))
        from docserver.conversations import DEFAULT_MAX_CONVERSATIONS

        assert store.max_conversations == DEFAULT_MAX_CONVERSATIONS

    def test_cap_under_one_clamped(self, tmp_path):
        """A cap of 0 or negative would mean 'delete everything on insert'.
        We clamp to 1 instead — a degenerate but at least usable store."""
        store = ConversationStore(str(tmp_path), max_conversations=0)
        assert store.max_conversations == 1

    def test_insert_under_cap_does_not_prune(self, tmp_path, sample_messages):
        store = ConversationStore(str(tmp_path), max_conversations=5)
        ids = [store.create(sample_messages) for _ in range(3)]
        assert len(store.list_all()) == 3
        for cid in ids:
            assert store.get(cid) is not None

    def test_insert_at_cap_prunes_oldest(self, tmp_path, sample_messages):
        """When the cap is N and an N+1th conversation is created, the
        oldest (by updated_at) is dropped."""
        import time

        store = ConversationStore(str(tmp_path), max_conversations=3)
        first = store.create(sample_messages)
        time.sleep(0.01)
        second = store.create(sample_messages)
        time.sleep(0.01)
        third = store.create(sample_messages)
        time.sleep(0.01)
        fourth = store.create(sample_messages)

        assert store.get(first) is None
        for cid in (second, third, fourth):
            assert store.get(cid) is not None
        assert len(store.list_all(limit=100)) == 3

    def test_repeated_inserts_keep_count_stable(self, tmp_path, sample_messages):
        """Inserting many conversations with a small cap should leave the
        table at exactly cap rows."""
        store = ConversationStore(str(tmp_path), max_conversations=5)
        for _ in range(50):
            store.create(sample_messages)
        assert len(store.list_all(limit=1000)) == 5

    def test_pruning_uses_updated_at_not_created_at(self, tmp_path, sample_messages):
        """A conversation that was created early but updated recently must
        survive pruning ahead of conversations that were never touched."""
        import time

        store = ConversationStore(str(tmp_path), max_conversations=2)
        old = store.create(sample_messages)
        time.sleep(0.01)
        middle = store.create(sample_messages)
        time.sleep(0.01)
        # Touch the OLD conversation so updated_at advances.
        store.update(old, [*sample_messages, {"role": "user", "content": "again"}])
        time.sleep(0.01)
        newest = store.create(sample_messages)

        assert store.get(old) is not None, "recently-updated conv should be retained"
        assert store.get(middle) is None, "least-recently-updated conv should be pruned"
        assert store.get(newest) is not None
