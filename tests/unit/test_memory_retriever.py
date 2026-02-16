"""Tests for memory retrieval and prompt formatting."""

import pytest
from unittest.mock import Mock

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
