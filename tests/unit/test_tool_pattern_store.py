"""Tests for tool pattern store — persistence, dedup, scoring, eviction."""

import json
from pathlib import Path

import pytest

from agent_framework.memory.tool_pattern_analyzer import ToolPatternRecommendation
from agent_framework.memory.tool_pattern_store import (
    ToolPatternStore,
    MAX_PATTERNS,
    RECENCY_HALF_LIFE,
)


@pytest.fixture
def store(tmp_path):
    return ToolPatternStore(tmp_path, enabled=True)


@pytest.fixture
def repo_slug():
    return "myorg/myrepo"


def _rec(pattern_id: str, tip: str = "test tip") -> ToolPatternRecommendation:
    return ToolPatternRecommendation(pattern_id=pattern_id, tip=tip)


class TestStoreAndRetrieve:
    def test_store_and_get_roundtrip(self, store, repo_slug):
        recs = [_rec("sequential-reads", "Use Grep first")]
        count = store.store_patterns(repo_slug, recs)
        assert count == 1

        results = store.get_top_patterns(repo_slug)
        assert len(results) == 1
        assert results[0].pattern_id == "sequential-reads"
        assert results[0].tip == "Use Grep first"

    def test_multiple_patterns_stored(self, store, repo_slug):
        recs = [
            _rec("sequential-reads"),
            _rec("bash-for-search"),
            _rec("repeated-glob"),
        ]
        count = store.store_patterns(repo_slug, recs)
        assert count == 3

        results = store.get_top_patterns(repo_slug, limit=10)
        assert len(results) == 3

    def test_empty_recommendations_returns_zero(self, store, repo_slug):
        assert store.store_patterns(repo_slug, []) == 0

    def test_disabled_store_returns_empty(self, tmp_path, repo_slug):
        disabled = ToolPatternStore(tmp_path, enabled=False)
        disabled.store_patterns(repo_slug, [_rec("test")])
        assert disabled.get_top_patterns(repo_slug) == []


class TestDeduplication:
    def test_duplicate_increments_hit_count(self, store, repo_slug):
        store.store_patterns(repo_slug, [_rec("sequential-reads")])
        store.store_patterns(repo_slug, [_rec("sequential-reads")])

        results = store.get_top_patterns(repo_slug)
        assert len(results) == 1
        assert results[0].hit_count == 2  # initial 1 + 1 increment

    def test_different_patterns_not_deduped(self, store, repo_slug):
        store.store_patterns(repo_slug, [_rec("sequential-reads")])
        store.store_patterns(repo_slug, [_rec("bash-for-search")])

        results = store.get_top_patterns(repo_slug, limit=10)
        assert len(results) == 2


class TestScoring:
    def test_higher_hit_count_ranks_first(self, store, repo_slug):
        # Store pattern A with 1 hit
        store.store_patterns(repo_slug, [_rec("a-pattern", "tip A")])
        # Store pattern B with 3 hits (1 create + 2 increments)
        store.store_patterns(repo_slug, [_rec("b-pattern", "tip B")])
        store.store_patterns(repo_slug, [_rec("b-pattern", "tip B")])
        store.store_patterns(repo_slug, [_rec("b-pattern", "tip B")])

        results = store.get_top_patterns(repo_slug, limit=2)
        assert results[0].pattern_id == "b-pattern"

    def test_limit_respected(self, store, repo_slug):
        recs = [_rec(f"pattern-{i}") for i in range(10)]
        store.store_patterns(repo_slug, recs)

        results = store.get_top_patterns(repo_slug, limit=3)
        assert len(results) == 3


class TestCharBudget:
    def test_max_chars_truncates_results(self, store, repo_slug):
        # Each tip is 50 chars
        recs = [_rec(f"p-{i}", "x" * 50) for i in range(10)]
        store.store_patterns(repo_slug, recs)

        # Budget of 120 chars should fit at most 2 tips (50 each)
        results = store.get_top_patterns(repo_slug, limit=10, max_chars=120)
        assert len(results) == 2


class TestEviction:
    def test_evicts_beyond_max_patterns(self, store, repo_slug):
        # Store more than MAX_PATTERNS
        recs = [_rec(f"pattern-{i}") for i in range(MAX_PATTERNS + 10)]
        store.store_patterns(repo_slug, recs)

        # Read raw file to check count
        path = store._store_path(repo_slug)
        data = json.loads(path.read_text())
        assert len(data) == MAX_PATTERNS


class TestRepoPaths:
    def test_slash_in_repo_name_handled(self, store):
        recs = [_rec("test")]
        store.store_patterns("org/repo", recs)
        results = store.get_top_patterns("org/repo")
        assert len(results) == 1

    def test_different_repos_isolated(self, store):
        store.store_patterns("org/repo-a", [_rec("pattern-a")])
        store.store_patterns("org/repo-b", [_rec("pattern-b")])

        results_a = store.get_top_patterns("org/repo-a")
        results_b = store.get_top_patterns("org/repo-b")
        assert results_a[0].pattern_id == "pattern-a"
        assert results_b[0].pattern_id == "pattern-b"


class TestFileCorruption:
    def test_corrupted_json_returns_empty(self, store, repo_slug):
        path = store._store_path(repo_slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json")

        results = store.get_top_patterns(repo_slug)
        assert results == []

    def test_nonexistent_repo_returns_empty(self, store):
        results = store.get_top_patterns("no/such-repo")
        assert results == []
