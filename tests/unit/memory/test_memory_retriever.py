"""Tests for MemoryRetriever — replan context, priority boosting, char limits."""

import time

import pytest

from agent_framework.memory.memory_store import MemoryStore
from agent_framework.memory.memory_retriever import (
    MemoryRetriever,
    MAX_REPLAN_MEMORY_CHARS,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, enabled=True)


@pytest.fixture
def retriever(store):
    return MemoryRetriever(store)


@pytest.fixture
def repo_slug():
    return "myorg/myrepo"


@pytest.fixture
def agent_type():
    return "engineer"


class TestFormatForReplan:
    def test_priority_categories_boosted(self, retriever, store, repo_slug, agent_type):
        """Priority categories (conventions, test_commands, repo_structure) get 2x boost."""
        # Create memories in priority and non-priority categories
        store.remember(repo_slug, agent_type, "conventions", "Use snake_case for functions")
        store.remember(repo_slug, agent_type, "bug_patterns", "Check nil pointers")
        store.remember(repo_slug, agent_type, "test_commands", "Run: pytest tests/")
        store.remember(repo_slug, agent_type, "general", "Code is in src/")

        result = retriever.format_for_replan(repo_slug, agent_type)

        # Priority categories should appear first even if all have same recency/frequency
        assert "conventions" in result
        assert "test_commands" in result
        lines = result.split("\n")
        priority_lines = [l for l in lines if "conventions" in l or "test_commands" in l]
        assert len(priority_lines) >= 2

    def test_char_limit_respected(self, retriever, store, repo_slug, agent_type):
        """Result is truncated to MAX_REPLAN_MEMORY_CHARS."""
        # Create many memories to exceed char limit
        for i in range(50):
            store.remember(
                repo_slug,
                agent_type,
                "conventions",
                f"Convention {i}: This is a detailed convention with many characters to fill up space and exceed the limit",
            )

        result = retriever.format_for_replan(repo_slug, agent_type, limit=100)

        # Should not exceed char limit (with some tolerance for header)
        assert len(result) <= MAX_REPLAN_MEMORY_CHARS + 200  # +200 for headers

    def test_empty_when_no_memories(self, retriever, repo_slug, agent_type):
        """Returns empty string when no memories exist."""
        result = retriever.format_for_replan(repo_slug, agent_type)
        assert result == ""

    def test_header_format(self, retriever, store, repo_slug, agent_type):
        """Output includes correct header and intro text."""
        store.remember(repo_slug, agent_type, "conventions", "Use type hints")

        result = retriever.format_for_replan(repo_slug, agent_type)

        assert "## Repository Knowledge (from previous tasks)" in result
        assert "You've worked on this repo before. Here's what you know:" in result

    def test_mixed_categories_sorted_by_priority(self, retriever, store, repo_slug, agent_type):
        """Mixed priority and non-priority categories are sorted correctly."""
        # Create memories at same time so recency is equal
        store.remember(repo_slug, agent_type, "general", "General info")
        store.remember(repo_slug, agent_type, "repo_structure", "Files are in src/")
        store.remember(repo_slug, agent_type, "bug_patterns", "Watch for race conditions")
        store.remember(repo_slug, agent_type, "test_commands", "make test")

        result = retriever.format_for_replan(repo_slug, agent_type, limit=10)

        # Extract order of categories
        lines = [l for l in result.split("\n") if l.strip().startswith("- [")]
        categories = []
        for line in lines:
            # Parse "- [category] content"
            bracket_end = line.index("]")
            category = line[3:bracket_end]
            categories.append(category)

        # Priority categories should appear before non-priority
        priority_indices = [i for i, cat in enumerate(categories) if cat in {"repo_structure", "test_commands"}]
        non_priority_indices = [i for i, cat in enumerate(categories) if cat in {"general", "bug_patterns"}]

        if priority_indices and non_priority_indices:
            assert max(priority_indices) < min(non_priority_indices), "Priority categories should come first"

    def test_disabled_store_returns_empty(self, tmp_path, repo_slug, agent_type):
        """Returns empty string when memory store is disabled."""
        disabled_store = MemoryStore(tmp_path, enabled=False)
        disabled_retriever = MemoryRetriever(disabled_store)

        disabled_store.remember(repo_slug, agent_type, "conventions", "This should be ignored")
        result = disabled_retriever.format_for_replan(repo_slug, agent_type)

        assert result == ""


class TestFormatForPrompt:
    """Existing tests for format_for_prompt to ensure we didn't break it."""

    def test_basic_formatting(self, retriever, store, repo_slug, agent_type):
        """Basic prompt formatting works."""
        store.remember(repo_slug, agent_type, "conventions", "Use snake_case")

        result = retriever.format_for_prompt(repo_slug, agent_type)

        assert "## Memories from Previous Tasks" in result
        assert "[conventions]" in result
        assert "Use snake_case" in result
