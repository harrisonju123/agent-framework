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
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def repo():
    return "myorg/myrepo"


@pytest.fixture
def agent():
    return "engineer"


class TestStoreAndRetrieve:
    def test_remember_and_recall_roundtrip(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black for formatting")
        results = store.recall(repo, agent)
        assert len(results) == 1
        assert results[0].category == "conventions"
        assert results[0].content == "Use black for formatting"

    def test_remember_returns_true_on_success(self, store, repo, agent):
        assert store.remember(repo, agent, "test", "content") is True

    def test_recall_empty_store(self, store, repo, agent):
        results = store.recall(repo, agent)
        assert results == []

    def test_multiple_memories_stored(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black")
        store.remember(repo, agent, "repo_structure", "src/ layout")
        store.remember(repo, agent, "test_commands", "pytest -v")

        results = store.recall(repo, agent)
        assert len(results) == 3

    def test_persistence_across_instances(self, tmp_path, repo, agent):
        store1 = MemoryStore(workspace=tmp_path, enabled=True)
        store1.remember(repo, agent, "conventions", "Use black")

        store2 = MemoryStore(workspace=tmp_path, enabled=True)
        results = store2.recall(repo, agent)
        assert len(results) == 1
        assert results[0].content == "Use black"


class TestDisabled:
    def test_remember_returns_false_when_disabled(self, tmp_path, repo, agent):
        store = MemoryStore(workspace=tmp_path, enabled=False)
        assert store.remember(repo, agent, "test", "content") is False

    def test_recall_returns_empty_when_disabled(self, tmp_path, repo, agent):
        store = MemoryStore(workspace=tmp_path, enabled=False)
        assert store.recall(repo, agent) == []

    def test_forget_returns_false_when_disabled(self, tmp_path, repo, agent):
        store = MemoryStore(workspace=tmp_path, enabled=False)
        assert store.forget(repo, agent, "test", "content") is False

    def test_enabled_property(self, tmp_path):
        assert MemoryStore(workspace=tmp_path, enabled=True).enabled is True
        assert MemoryStore(workspace=tmp_path, enabled=False).enabled is False


class TestDeduplication:
    def test_duplicate_updates_access_count(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black")
        store.remember(repo, agent, "conventions", "Use black")

        results = store.recall(repo, agent)
        assert len(results) == 1
        assert results[0].access_count == 1  # incremented once on dedup

    def test_different_content_not_deduped(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black")
        store.remember(repo, agent, "conventions", "Use ruff")

        results = store.recall(repo, agent)
        assert len(results) == 2


class TestEviction:
    def test_evicts_oldest_beyond_limit(self, store, repo, agent):
        for i in range(MAX_MEMORIES_PER_STORE + 10):
            store.remember(repo, agent, "cat", f"memory-{i}")

        results = store.recall_all(repo, agent)
        assert len(results) == MAX_MEMORIES_PER_STORE


class TestFiltering:
    def test_filter_by_category(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black")
        store.remember(repo, agent, "repo_structure", "src/ layout")

        results = store.recall(repo, agent, category="conventions")
        assert len(results) == 1
        assert results[0].category == "conventions"

    def test_filter_by_tags(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black", tags=["python"])
        store.remember(repo, agent, "conventions", "Use eslint", tags=["js"])

        results = store.recall(repo, agent, tags=["python"])
        assert len(results) == 1
        assert results[0].content == "Use black"

    def test_recall_limit(self, store, repo, agent):
        for i in range(10):
            store.remember(repo, agent, "cat", f"memory-{i}")

        results = store.recall(repo, agent, limit=3)
        assert len(results) == 3

    def test_recall_sorted_by_recency(self, store, repo, agent):
        store.remember(repo, agent, "cat", "old-memory")
        store.remember(repo, agent, "cat", "new-memory")

        results = store.recall(repo, agent)
        assert results[0].content == "new-memory"


class TestForget:
    def test_forget_existing_memory(self, store, repo, agent):
        store.remember(repo, agent, "conventions", "Use black")
        assert store.forget(repo, agent, "conventions", "Use black") is True

        results = store.recall(repo, agent)
        assert len(results) == 0

    def test_forget_nonexistent_returns_false(self, store, repo, agent):
        assert store.forget(repo, agent, "no-cat", "no-content") is False


class TestContentTruncation:
    def test_long_content_truncated(self, store, repo, agent):
        long_content = "x" * (MAX_CONTENT_LENGTH + 500)
        store.remember(repo, agent, "test", long_content)

        results = store.recall(repo, agent)
        assert len(results[0].content) == MAX_CONTENT_LENGTH


class TestRepoPaths:
    def test_slash_in_repo_name(self, store):
        store.remember("org/repo", "engineer", "test", "content")
        results = store.recall("org/repo", "engineer")
        assert len(results) == 1

    def test_different_repos_isolated(self, store):
        store.remember("org/repo-a", "engineer", "test", "content-a")
        store.remember("org/repo-b", "engineer", "test", "content-b")

        results_a = store.recall("org/repo-a", "engineer")
        results_b = store.recall("org/repo-b", "engineer")
        assert results_a[0].content == "content-a"
        assert results_b[0].content == "content-b"

    def test_different_agents_isolated(self, store, repo):
        store.remember(repo, "engineer", "test", "eng-content")
        store.remember(repo, "architect", "test", "arch-content")

        results_eng = store.recall(repo, "engineer")
        results_arch = store.recall(repo, "architect")
        assert results_eng[0].content == "eng-content"
        assert results_arch[0].content == "arch-content"


class TestCorruptedData:
    def test_corrupted_json_returns_empty(self, store, repo, agent):
        path = store._store_path(repo, agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json {{{")

        results = store.recall(repo, agent)
        assert results == []

    def test_nonexistent_repo_returns_empty(self, store):
        results = store.recall("no/such-repo", "engineer")
        assert results == []


class TestMemoryEntry:
    def test_touch_updates_metadata(self):
        entry = MemoryEntry(category="test", content="content", access_count=0)
        before = entry.last_accessed

        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed >= before
