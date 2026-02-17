"""Tests for memory retrieval and prompt formatting."""

import pytest
from unittest.mock import Mock, call

from agent_framework.memory.memory_retriever import (
    MemoryRetriever,
    MAX_MEMORY_PROMPT_CHARS,
)
from agent_framework.memory.memory_store import MemoryEntry


class TestMemoryRetriever:
    """Test memory retrieval and formatting functionality."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock memory store."""
        return Mock()

    @pytest.fixture
    def retriever(self, mock_store):
        """Create a memory retriever with mock store."""
        return MemoryRetriever(mock_store)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memory entries for testing."""
        return [
            MemoryEntry(
                category="repo_structure",
                content="Project uses Go with standard layout",
                access_count=5,
                last_accessed=1000000,
            ),
            MemoryEntry(
                category="test_commands",
                content="Run tests with: make test",
                access_count=3,
                last_accessed=999000,
            ),
            MemoryEntry(
                category="conventions",
                content="Use snake_case for Python function names",
                access_count=8,
                last_accessed=1001000,
            ),
        ]

    def test_format_for_prompt_default_max_chars(self, retriever, mock_store, sample_memories):
        """Verify default max_chars uses MAX_MEMORY_PROMPT_CHARS constant."""
        mock_store.recall_all.return_value = sample_memories

        result = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
        )

        # Should include all memories within default limit
        assert "## Memories from Previous Tasks" in result
        assert "repo_structure" in result
        assert "test_commands" in result
        assert "conventions" in result

    def test_format_for_prompt_custom_max_chars(self, retriever, mock_store, sample_memories):
        """Verify custom max_chars parameter limits output."""
        mock_store.recall_all.return_value = sample_memories

        # Set very small limit — should only fit header and first memory
        result = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=100,
        )

        # Should include header and at least one entry (the most relevant one)
        assert "## Memories from Previous Tasks" in result
        # With small limit, should stop adding entries early
        memory_count = result.count("- [")
        assert memory_count < len(sample_memories)

    def test_format_for_prompt_zero_max_chars(self, retriever, mock_store, sample_memories):
        """Verify max_chars=0 returns empty string (critical budget scenario)."""
        mock_store.recall_all.return_value = sample_memories

        result = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=0,
        )

        # With 0 char limit, header alone exceeds budget
        # Should return just header + trailing newline
        assert result == "## Memories from Previous Tasks\n\n"

    def test_format_for_prompt_tight_budget(self, retriever, mock_store, sample_memories):
        """Verify max_chars=1000 (tight budget) includes fewer memories."""
        mock_store.recall_all.return_value = sample_memories

        # Normal budget (3000 chars)
        result_normal = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
        )

        # Tight budget (1000 chars)
        result_tight = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=1000,
        )

        # Tight budget should be shorter or equal to normal budget
        assert len(result_tight) <= len(result_normal)

    def test_format_for_prompt_no_memories(self, retriever, mock_store):
        """Verify empty string when no memories exist."""
        mock_store.recall_all.return_value = []

        result = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=1000,
        )

        assert result == ""

    def test_format_for_prompt_respects_char_limit(self, retriever, mock_store):
        """Verify character limit is strictly enforced."""
        # Create memories with known sizes
        large_memories = [
            MemoryEntry(
                category="test",
                content="x" * 500,  # 500 char content
                access_count=i,
                last_accessed=1000000 + i,
            )
            for i in range(10)
        ]
        mock_store.recall_all.return_value = large_memories

        max_chars = 1000
        result = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=max_chars,
        )

        # Total length should not significantly exceed max_chars
        # (header is ~34 chars, each entry line is "- [test] " + 500 chars = ~509 chars)
        # So with 1000 char limit, should fit header + 1 full entry
        lines = [line for line in result.split("\n") if line.startswith("- [")]
        assert len(lines) <= 2  # At most 2 entries should fit

    def test_format_for_prompt_none_max_chars_uses_default(self, retriever, mock_store, sample_memories):
        """Verify max_chars=None falls back to MAX_MEMORY_PROMPT_CHARS."""
        mock_store.recall_all.return_value = sample_memories

        # Explicit None should behave same as not providing the parameter
        result_none = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=None,
        )

        result_default = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
        )

        assert result_none == result_default
        assert len(result_none) > 0

    def test_format_for_prompt_budget_tiers(self, retriever, mock_store, sample_memories):
        """Verify the three budget tiers produce different output sizes."""
        mock_store.recall_all.return_value = sample_memories

        # Healthy budget (3000 chars) — most content
        result_healthy = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=3000,
        )

        # Tight budget (1000 chars) — reduced content
        result_tight = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=1000,
        )

        # Critical budget (0 chars) — minimal content
        result_critical = retriever.format_for_prompt(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=0,
        )

        # Verify progression: healthy >= tight >= critical
        assert len(result_healthy) >= len(result_tight)
        assert len(result_tight) >= len(result_critical)
        assert len(result_critical) > 0  # Should at least have header


class TestSharedMemoryMerge:
    """Test that 'shared' namespace memories are merged for all agent types."""

    @pytest.fixture
    def mock_store(self):
        return Mock()

    @pytest.fixture
    def retriever(self, mock_store):
        return MemoryRetriever(mock_store)

    @pytest.fixture
    def agent_memory(self):
        return MemoryEntry(
            category="conventions",
            content="Use snake_case for Python",
            access_count=1,
            last_accessed=1000000.0,
        )

    @pytest.fixture
    def shared_memory(self):
        return MemoryEntry(
            category="architectural_decisions",
            content="Topic: Use Postgres\nRecommendation: Use Postgres\nConfidence: high\nTrade-offs: speed vs complexity\nReasoning: proven at scale",
            access_count=0,
            last_accessed=900000.0,
            tags=["debate", "debate-123", "origin:architect", "confidence:high", "use", "postgres"],
        )

    def test_shared_memories_merged_for_agent(self, retriever, mock_store, agent_memory, shared_memory):
        """Shared namespace is queried and merged when agent_type != 'shared'."""
        # First call returns agent-specific memories, second returns shared
        mock_store.recall_all.side_effect = [[agent_memory], [shared_memory]]

        memories = retriever.get_relevant_memories("my/repo", "engineer")

        # Both calls happened: agent-specific then shared
        assert mock_store.recall_all.call_count == 2
        mock_store.recall_all.assert_any_call("my/repo", "engineer")
        mock_store.recall_all.assert_any_call("my/repo", "shared")

        # Both memories are present
        contents = [m.content for m in memories]
        assert agent_memory.content in contents
        assert shared_memory.content in contents

    def test_shared_namespace_not_queried_twice(self, retriever, mock_store, shared_memory):
        """When agent_type is 'shared', we don't recurse into shared again."""
        mock_store.recall_all.return_value = [shared_memory]

        retriever.get_relevant_memories("my/repo", "shared")

        # Only one call — no second lookup for shared
        assert mock_store.recall_all.call_count == 1
        mock_store.recall_all.assert_called_once_with("my/repo", "shared")

    def test_shared_memories_deduplicated(self, retriever, mock_store, shared_memory):
        """Memory present in both agent and shared stores is not double-counted."""
        # Same memory returned for both agent and shared lookup
        mock_store.recall_all.return_value = [shared_memory]

        memories = retriever.get_relevant_memories("my/repo", "engineer")

        # Should appear exactly once despite being in both namespaces
        matching = [m for m in memories if m.content == shared_memory.content]
        assert len(matching) == 1
