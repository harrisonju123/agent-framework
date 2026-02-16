"""Tests for MemoryRetriever — scoring, recency decay, prompt formatting, extraction."""

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.memory.memory_retriever import (
    MAX_MEMORY_PROMPT_CHARS,
    RECENCY_HALF_LIFE_SECONDS,
    MemoryRetriever,
    _frequency_score,
    _recency_score,
    _relevance_score,
)
from agent_framework.memory.memory_store import MemoryEntry, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, enabled=True)


@pytest.fixture
def retriever(store):
    return MemoryRetriever(store)


@pytest.fixture
def repo():
    return "myorg/myrepo"


@pytest.fixture
def agent():
    return "engineer"


def _make_entry(
    category: str = "info",
    content: str = "test",
    last_accessed: float = None,
    access_count: int = 0,
    tags: list = None,
) -> MemoryEntry:
    return MemoryEntry(
        category=category,
        content=content,
        last_accessed=last_accessed or time.time(),
        access_count=access_count,
        tags=tags or [],
    )


class TestRecencyScore:
    def test_recent_entry_scores_near_one(self):
        entry = _make_entry(last_accessed=time.time())
        score = _recency_score(entry)
        assert 0.99 < score <= 1.0

    def test_one_halflife_old_scores_near_half(self):
        entry = _make_entry(last_accessed=time.time() - RECENCY_HALF_LIFE_SECONDS)
        score = _recency_score(entry)
        assert 0.45 < score < 0.55

    def test_very_old_entry_scores_near_zero(self):
        entry = _make_entry(last_accessed=time.time() - RECENCY_HALF_LIFE_SECONDS * 10)
        score = _recency_score(entry)
        assert score < 0.01


class TestFrequencyScore:
    def test_zero_access_returns_zero(self):
        entry = _make_entry(access_count=0)
        assert _frequency_score(entry) == 0.0

    def test_one_access_returns_log2(self):
        entry = _make_entry(access_count=1)
        assert _frequency_score(entry) == pytest.approx(math.log1p(1))

    def test_higher_count_returns_higher_score(self):
        low = _make_entry(access_count=2)
        high = _make_entry(access_count=20)
        assert _frequency_score(high) > _frequency_score(low)


class TestRelevanceScore:
    def test_tag_overlap_boosts_score(self):
        entry = _make_entry(tags=["python", "testing"])
        score_with_tags = _relevance_score(entry, task_tags=["python"])
        score_without_tags = _relevance_score(entry, task_tags=None)
        assert score_with_tags > score_without_tags

    def test_no_tag_overlap_no_boost(self):
        entry = _make_entry(tags=["go"])
        score_with = _relevance_score(entry, task_tags=["python"])
        score_without = _relevance_score(entry, task_tags=None)
        assert score_with == pytest.approx(score_without)

    def test_multiple_tag_overlaps_bigger_boost(self):
        entry = _make_entry(tags=["python", "testing", "backend"])
        score_one = _relevance_score(entry, task_tags=["python"])
        score_two = _relevance_score(entry, task_tags=["python", "testing"])
        assert score_two > score_one


class TestGetRelevantMemories:
    def test_returns_empty_when_no_memories(self, retriever, repo, agent):
        result = retriever.get_relevant_memories(repo, agent)
        assert result == []

    def test_returns_memories_ranked_by_relevance(self, store, retriever, repo, agent):
        # Store an old memory (low recency) with no accesses
        store.remember(repo, agent, "old", "ancient info")
        # Manually age it by patching the file
        import json
        path = store._store_path(repo, agent)
        data = json.loads(path.read_text())
        data[0]["last_accessed"] = time.time() - RECENCY_HALF_LIFE_SECONDS * 5
        path.write_text(json.dumps(data))

        # Store a fresh memory
        store.remember(repo, agent, "new", "fresh info")

        results = retriever.get_relevant_memories(repo, agent)
        assert results[0].content == "fresh info"

    def test_limit_respected(self, store, retriever, repo, agent):
        for i in range(10):
            store.remember(repo, agent, "info", f"memory {i}")
        results = retriever.get_relevant_memories(repo, agent, limit=3)
        assert len(results) == 3

    def test_task_tags_influence_ranking(self, store, retriever, repo, agent):
        store.remember(repo, agent, "info", "go stuff", tags=["go"])
        store.remember(repo, agent, "info", "python stuff", tags=["python"])
        results = retriever.get_relevant_memories(repo, agent, task_tags=["python"])
        assert results[0].content == "python stuff"


class TestFormatForPrompt:
    def test_empty_when_no_memories(self, retriever, repo, agent):
        result = retriever.format_for_prompt(repo, agent)
        assert result == ""

    def test_includes_header_and_entries(self, store, retriever, repo, agent):
        store.remember(repo, agent, "conventions", "Use PEP8 style")
        result = retriever.format_for_prompt(repo, agent)
        assert "## Memories from Previous Tasks" in result
        assert "- [conventions] Use PEP8 style" in result

    def test_respects_max_chars_budget(self, store, retriever, repo, agent):
        # Fill store with long memories
        for i in range(50):
            store.remember(repo, agent, "info", "x" * 200 + f" {i}")
        result = retriever.format_for_prompt(repo, agent)
        # Total memory lines should be under the budget
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        total = sum(len(l) for l in lines)
        assert total <= MAX_MEMORY_PROMPT_CHARS

    def test_limit_parameter_forwarded(self, store, retriever, repo, agent):
        for i in range(10):
            store.remember(repo, agent, "info", f"mem {i}")
        result = retriever.format_for_prompt(repo, agent, limit=2)
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(lines) <= 2


class TestExtractMemoriesFromResponse:
    def test_parses_learnings_section(self, store, retriever, repo, agent):
        response = """Some output here.

## Learnings
- [conventions] Always run lint before commit
- [repo_structure] Tests are in tests/unit/

## Other Section
More stuff.
"""
        count = retriever.extract_memories_from_response(response, repo, agent, "task-1")
        assert count == 2
        results = store.recall(repo, agent)
        categories = {r.category for r in results}
        assert "conventions" in categories
        assert "repo_structure" in categories

    def test_empty_response_returns_zero(self, retriever, repo, agent):
        assert retriever.extract_memories_from_response("", repo, agent, "task-1") == 0

    def test_no_learnings_section_returns_zero(self, retriever, repo, agent):
        response = "Just regular output with no learning markers."
        assert retriever.extract_memories_from_response(response, repo, agent, "task-1") == 0

    def test_malformed_bracket_skipped(self, retriever, repo, agent):
        response = """## Learnings
- [incomplete
- [] empty category
- [valid] this one works
"""
        count = retriever.extract_memories_from_response(response, repo, agent, "task-1")
        assert count == 1

    def test_stops_at_next_heading(self, store, retriever, repo, agent):
        response = """## Learnings
- [info] first learning

## Implementation Notes
- [info] not a learning
"""
        count = retriever.extract_memories_from_response(response, repo, agent, "task-1")
        assert count == 1

    def test_case_insensitive_heading_match(self, store, retriever, repo, agent):
        response = """## learnings
- [info] lowercase heading works
"""
        count = retriever.extract_memories_from_response(response, repo, agent, "task-1")
        assert count == 1
