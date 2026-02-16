"""Tests for MemoryStore — persistence, dedup, eviction, filtering, forget."""

import json
import time
from pathlib import Path

import pytest

from agent_framework.memory.memory_store import (
    MAX_CONTENT_LENGTH,
    MAX_MEMORIES_PER_STORE,
    MemoryEntry,
    MemoryStore,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, enabled=True)


@pytest.fixture
def repo():
    return "myorg/myrepo"


@pytest.fixture
def agent():
    return "engineer"


class TestStoreAndRecall:
    def test_remember_and_recall_roundtrip(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use pytest for testing")
        results = store.recall(repo, agent)
        assert len(results) == 1
        assert results[0].category == "conventions"
        assert results[0].content == "Use pytest for testing"

    def test_remember_multiple_categories(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use pytest")
        store.remember(repo, agent, "repo_structure", "src/ layout")
        store.remember(repo, agent, "test_commands", "pytest -v")
        results = store.recall(repo, agent)
        assert len(results) == 3

    def test_recall_filter_by_category(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use pytest")
        store.remember(repo, agent, "repo_structure", "src/ layout")
        results = store.recall(repo, agent, category="conventions")
        assert len(results) == 1
        assert results[0].category == "conventions"

    def test_recall_filter_by_tags(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use pytest", tags=["python", "testing"])
        store.remember(repo, agent, "conventions", "Use go test", tags=["go", "testing"])
        store.remember(repo, agent, "conventions", "Use jest", tags=["js"])
        results = store.recall(repo, agent, tags=["python"])
        assert len(results) == 1
        assert results[0].content == "Use pytest"

    def test_recall_tags_intersection(self, store, repo, agent):
        """Tags filter uses intersection — any overlap matches."""
        store.remember(repo, agent, "info", "item1", tags=["a", "b"])
        store.remember(repo, agent, "info", "item2", tags=["c", "d"])
        results = store.recall(repo, agent, tags=["b", "c"])
        assert len(results) == 2

    def test_recall_limit_respected(self, store, repo, agent):
        for i in range(10):
            store.remember(repo, agent, "info", f"memory {i}")
        results = store.recall(repo, agent, limit=3)
        assert len(results) == 3

    def test_recall_returns_most_recent_first(self, store, repo, agent):
        store.remember(repo, agent, "info", "old memory")
        time.sleep(0.01)
        store.remember(repo, agent, "info", "new memory")
        results = store.recall(repo, agent)
        assert results[0].content == "new memory"

    def test_recall_all_returns_everything(self, store, repo, agent):
        for i in range(15):
            store.remember(repo, agent, "info", f"memory {i}")
        results = store.recall_all(repo, agent)
        assert len(results) == 15


class TestDisabled:
    def test_remember_returns_false_when_disabled(self, tmp_path, repo, agent):
        disabled = MemoryStore(tmp_path, enabled=False)
        assert disabled.remember(repo, agent, "cat", "content") is False

    def test_recall_returns_empty_when_disabled(self, tmp_path, repo, agent):
        disabled = MemoryStore(tmp_path, enabled=False)
        assert disabled.recall(repo, agent) == []

    def test_forget_returns_false_when_disabled(self, tmp_path, repo, agent):
        disabled = MemoryStore(tmp_path, enabled=False)
        assert disabled.forget(repo, agent, "cat", "content") is False


class TestDeduplication:
    def test_duplicate_remember_increments_access_count(self, store, repo, agent):
        store.remember(repo, agent, "info", "same content")
        store.remember(repo, agent, "info", "same content")
        results = store.recall(repo, agent)
        assert len(results) == 1
        assert results[0].access_count == 1  # initial 0 + 1 increment

    def test_different_category_not_deduped(self, store, repo, agent):
        store.remember(repo, agent, "cat_a", "same content")
        store.remember(repo, agent, "cat_b", "same content")
        results = store.recall(repo, agent)
        assert len(results) == 2

    def test_different_content_not_deduped(self, store, repo, agent):
        store.remember(repo, agent, "info", "content A")
        store.remember(repo, agent, "info", "content B")
        results = store.recall(repo, agent)
        assert len(results) == 2


class TestContentTruncation:
    def test_long_content_truncated(self, store, repo, agent):
        long_content = "x" * (MAX_CONTENT_LENGTH + 500)
        store.remember(repo, agent, "info", long_content)
        results = store.recall(repo, agent)
        assert len(results[0].content) == MAX_CONTENT_LENGTH


class TestEviction:
    def test_evicts_oldest_when_over_limit(self, store, repo, agent):
        for i in range(MAX_MEMORIES_PER_STORE + 10):
            store.remember(repo, agent, "info", f"memory-{i:04d}")
        results = store.recall_all(repo, agent)
        assert len(results) == MAX_MEMORIES_PER_STORE
        # Oldest entries should have been evicted
        contents = {r.content for r in results}
        assert "memory-0000" not in contents


class TestForget:
    def test_forget_removes_matching_entry(self, store, repo, agent):
        store.remember(repo, agent, "info", "keep me")
        store.remember(repo, agent, "info", "forget me")
        removed = store.forget(repo, agent, "info", "forget me")
        assert removed is True
        results = store.recall(repo, agent)
        assert len(results) == 1
        assert results[0].content == "keep me"

    def test_forget_returns_false_if_not_found(self, store, repo, agent):
        store.remember(repo, agent, "info", "some content")
        assert store.forget(repo, agent, "info", "nonexistent") is False


class TestPersistence:
    def test_data_survives_new_store_instance(self, tmp_path, repo, agent):
        store1 = MemoryStore(tmp_path, enabled=True)
        store1.remember(repo, agent, "info", "persisted data")

        store2 = MemoryStore(tmp_path, enabled=True)
        results = store2.recall(repo, agent)
        assert len(results) == 1
        assert results[0].content == "persisted data"

    def test_corrupted_json_returns_empty(self, store, repo, agent):
        path = store._store_path(repo, agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json")
        assert store.recall(repo, agent) == []


class TestRepoPaths:
    def test_slash_in_repo_handled(self, store):
        store.remember("org/repo", "engineer", "info", "data")
        results = store.recall("org/repo", "engineer")
        assert len(results) == 1

    def test_different_repos_isolated(self, store):
        store.remember("org/repo-a", "engineer", "info", "data A")
        store.remember("org/repo-b", "engineer", "info", "data B")
        results_a = store.recall("org/repo-a", "engineer")
        results_b = store.recall("org/repo-b", "engineer")
        assert results_a[0].content == "data A"
        assert results_b[0].content == "data B"

    def test_different_agents_isolated(self, store, repo):
        store.remember(repo, "engineer", "info", "eng data")
        store.remember(repo, "qa", "info", "qa data")
        results_eng = store.recall(repo, "engineer")
        results_qa = store.recall(repo, "qa")
        assert len(results_eng) == 1
        assert results_eng[0].content == "eng data"
        assert results_qa[0].content == "qa data"


class TestMemoryEntryTouch:
    def test_touch_updates_access_metadata(self):
        entry = MemoryEntry(category="info", content="test")
        old_accessed = entry.last_accessed
        old_count = entry.access_count
        time.sleep(0.01)
        entry.touch()
        assert entry.last_accessed > old_accessed
        assert entry.access_count == old_count + 1
