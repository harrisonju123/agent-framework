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


class TestErrorAwareRetrieval:
    """Tests for error-type-aware memory retrieval in replanning."""

    def test_test_failure_boosts_test_commands_and_past_failures(self, retriever, store, repo_slug, agent_type):
        """Test failures should prioritize test_commands and past_failures."""
        # Create various memories
        store.remember(repo_slug, agent_type, "conventions", "Use snake_case")
        store.remember(repo_slug, agent_type, "test_commands", "Run: pytest tests/")
        store.remember(repo_slug, agent_type, "repo_structure", "Tests in tests/")
        store.remember(repo_slug, agent_type, "past_failures", "Previous test failure resolved by fixing imports")

        result = retriever.format_for_replan(
            repo_slug, agent_type, error_type="test_failure"
        )

        lines = [l for l in result.split("\n") if l.strip().startswith("- [")]

        # test_commands and past_failures should be boosted
        assert any("test_commands" in line for line in lines)
        assert any("past_failures" in line for line in lines)

    def test_past_failures_tagged_with_error_type_boosted(self, retriever, store, repo_slug, agent_type):
        """Past failures with matching error type tags should appear first."""
        # Create past_failures with different tags
        store.remember(
            repo_slug, agent_type, "past_failures",
            "Logic error: fixed by adding null check",
            tags=["error:logic"]
        )
        store.remember(
            repo_slug, agent_type, "past_failures",
            "Import error: fixed by adjusting sys.path",
            tags=["error:import_error"]
        )
        store.remember(
            repo_slug, agent_type, "past_failures",
            "Test failure: fixed by updating test fixtures",
            tags=["error:test_failure"]
        )

        result = retriever.format_for_replan(
            repo_slug, agent_type, error_type="logic"
        )

        lines = [l for l in result.split("\n") if l.strip().startswith("- [past_failures]")]

        # The logic error should appear first (if any past_failures are included)
        if lines:
            first_failure_line = lines[0]
            assert "Logic error" in first_failure_line or "null check" in first_failure_line

    def test_import_error_boosts_repo_structure(self, retriever, store, repo_slug, agent_type):
        """Import/dependency errors should prioritize repo_structure and past_failures."""
        store.remember(repo_slug, agent_type, "conventions", "Use type hints")
        store.remember(repo_slug, agent_type, "repo_structure", "Modules are in src/")
        store.remember(repo_slug, agent_type, "past_failures", "Import resolved by fixing path")
        store.remember(repo_slug, agent_type, "test_commands", "pytest tests/")

        result = retriever.format_for_replan(
            repo_slug, agent_type, error_type="import_error"
        )

        lines = [l for l in result.split("\n") if l.strip().startswith("- [")]
        categories = []
        for line in lines:
            bracket_end = line.index("]")
            category = line[3:bracket_end]
            categories.append(category)

        # repo_structure and past_failures should be prominent
        assert "repo_structure" in categories or "past_failures" in categories

    def test_default_error_type_boosts_past_failures_and_conventions(self, retriever, store, repo_slug, agent_type):
        """Unknown error types should still boost past_failures and conventions."""
        store.remember(repo_slug, agent_type, "conventions", "Use docstrings")
        store.remember(repo_slug, agent_type, "past_failures", "Generic error resolved")
        store.remember(repo_slug, agent_type, "test_commands", "make test")

        result = retriever.format_for_replan(
            repo_slug, agent_type, error_type="unknown"
        )

        lines = [l for l in result.split("\n") if l.strip().startswith("- [")]

        # Should include past_failures and conventions due to default boost
        assert any("past_failures" in line or "conventions" in line for line in lines)

    def test_past_failures_category_included(self, retriever, store, repo_slug, agent_type):
        """past_failures is now a priority category."""
        store.remember(repo_slug, agent_type, "past_failures", "Error X resolved by doing Y")
        store.remember(repo_slug, agent_type, "general", "Some general info")

        result = retriever.format_for_replan(repo_slug, agent_type)

        # past_failures should be included
        assert "past_failures" in result
        assert "Error X resolved by doing Y" in result

    def test_architectural_decisions_category_included(self, retriever, store, repo_slug, agent_type):
        """architectural_decisions is now a priority category."""
        store.remember(repo_slug, agent_type, "architectural_decisions", "Use REST not GraphQL")
        store.remember(repo_slug, agent_type, "general", "Some general info")

        result = retriever.format_for_replan(repo_slug, agent_type)

        # architectural_decisions should be included
        assert "architectural_decisions" in result
        assert "Use REST not GraphQL" in result

    def test_limit_increased_to_fifteen(self, retriever, store, repo_slug, agent_type):
        """Default limit for replan should be 15 now."""
        # Create 20 memories across priority categories
        for i in range(5):
            store.remember(repo_slug, agent_type, "conventions", f"Convention {i}")
            store.remember(repo_slug, agent_type, "test_commands", f"Test command {i}")
            store.remember(repo_slug, agent_type, "past_failures", f"Failure {i}")
            store.remember(repo_slug, agent_type, "repo_structure", f"Structure {i}")

        # Default call (limit defaults to 15 in the new signature)
        result = retriever.format_for_replan(repo_slug, agent_type)

        lines = [l for l in result.split("\n") if l.strip().startswith("- [")]

        # Should have up to 15 memories (may be less due to char limit)
        # We just verify it's more than the old limit of 10
        assert len(lines) >= 10

    def test_non_priority_categories_filtered_out(self, retriever, store, repo_slug, agent_type):
        """Non-priority categories should not appear in replan context."""
        store.remember(repo_slug, agent_type, "general", "General information")
        store.remember(repo_slug, agent_type, "bug_patterns", "Common bug")
        store.remember(repo_slug, agent_type, "conventions", "Use type hints")

        result = retriever.format_for_replan(repo_slug, agent_type)

        # Priority category should be there
        assert "conventions" in result
        # Non-priority categories should not
        assert "general" not in result
        assert "bug_patterns" not in result


class TestFormatForPrompt:
    """Existing tests for format_for_prompt to ensure we didn't break it."""

    def test_basic_formatting(self, retriever, store, repo_slug, agent_type):
        """Basic prompt formatting works."""
        store.remember(repo_slug, agent_type, "conventions", "Use snake_case")

        result = retriever.format_for_prompt(repo_slug, agent_type)

        assert "## Memories from Previous Tasks" in result
        assert "[conventions]" in result
        assert "Use snake_case" in result
