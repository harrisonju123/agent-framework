"""Tests for memory integration with replanning flow."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.memory.memory_retriever import MemoryRetriever
from agent_framework.memory.memory_store import MemoryEntry, MemoryStore


def _make_memory_entry(category: str, content: str, **overrides) -> MemoryEntry:
    """Helper to create a MemoryEntry for testing."""
    defaults = dict(
        repo_slug="test/repo",
        agent_type="engineer",
        category=category,
        content=content,
        source_task_id="task-123",
        created_at=datetime.now(timezone.utc).isoformat(),
        last_accessed=datetime.now(timezone.utc).timestamp(),
        access_count=1,
        tags=[],
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


class TestFormatForReplan:
    """Tests for MemoryRetriever.format_for_replan() method."""

    def test_returns_empty_when_no_memories(self, tmp_path):
        """Should return empty string when no memories exist."""
        store = MemoryStore(workspace=tmp_path, enabled=True)
        retriever = MemoryRetriever(store)

        result = retriever.format_for_replan(
            repo_slug="test/repo",
            agent_type="engineer",
        )

        assert result == ""

    def test_prioritizes_priority_categories(self, tmp_path):
        """Should prioritize conventions, test_commands, repo_structure categories."""
        store = MemoryStore(workspace=tmp_path, enabled=True)

        # Create memories with different categories
        store.remember("test/repo", "engineer", "other", "Other info", "task-1")
        store.remember("test/repo", "engineer", "conventions", "Use pytest for tests", "task-2")
        store.remember("test/repo", "engineer", "random", "Random note", "task-3")
        store.remember("test/repo", "engineer", "test_commands", "Run: pytest tests/", "task-4")
        store.remember("test/repo", "engineer", "repo_structure", "Backend in src/", "task-5")

        retriever = MemoryRetriever(store)
        result = retriever.format_for_replan(
            repo_slug="test/repo",
            agent_type="engineer",
        )

        # Priority categories should appear before others
        lines = result.split("\n")
        assert "## Relevant Memories from This Repo" in result

        # Find the indices of priority vs non-priority categories
        conventions_idx = next(i for i, line in enumerate(lines) if "conventions" in line)
        test_cmd_idx = next(i for i, line in enumerate(lines) if "test_commands" in line)
        repo_struct_idx = next(i for i, line in enumerate(lines) if "repo_structure" in line)
        other_idx = next((i for i, line in enumerate(lines) if "other" in line), len(lines))

        # Priority categories should come before non-priority
        assert conventions_idx < other_idx
        assert test_cmd_idx < other_idx
        assert repo_struct_idx < other_idx

    def test_respects_max_chars_budget(self, tmp_path):
        """Should respect max_chars budget and truncate content."""
        store = MemoryStore(workspace=tmp_path, enabled=True)

        # Create many memories to exceed budget
        for i in range(20):
            store.remember(
                "test/repo",
                "engineer",
                "conventions",
                f"Convention {i}: This is a long convention description that takes up space",
                f"task-{i}",
            )

        retriever = MemoryRetriever(store)
        result = retriever.format_for_replan(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=500,
        )

        # Total length should not exceed max_chars by much (allowing for header)
        assert len(result) < 600  # Some overhead for header and newlines

    def test_returns_empty_when_only_header(self, tmp_path):
        """Should return empty string when no memories fit within budget."""
        store = MemoryStore(workspace=tmp_path, enabled=True)

        store.remember(
            "test/repo",
            "engineer",
            "conventions",
            "Very long content that exceeds tiny budget" * 20,
            "task-1",
        )

        retriever = MemoryRetriever(store)
        result = retriever.format_for_replan(
            repo_slug="test/repo",
            agent_type="engineer",
            max_chars=10,  # Impossibly small budget
        )

        assert result == ""


class TestRequestReplanMemoryInjection:
    """Tests for _request_replan() memory injection."""

    async def test_includes_memory_when_enabled_and_available(self, tmp_path):
        """Should include memory context when memory is enabled and memories exist."""
        # Create mock agent with memory enabled
        agent = MagicMock()
        agent._memory_enabled = True
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()
        agent._session_logger = MagicMock()
        agent._replan_model = "test-model"

        # Set up memory store with test data
        store = MemoryStore(workspace=tmp_path, enabled=True)
        store.remember("test/repo", "engineer", "conventions", "Use pytest", "task-1")
        retriever = MemoryRetriever(store)
        agent._memory_retriever = retriever

        # Mock _get_repo_slug to return a valid repo
        agent._get_repo_slug = MagicMock(return_value="test/repo")

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.success = True
        mock_llm_response.content = "Revised approach: Try a different strategy"
        agent.llm.complete = MagicMock(return_value=mock_llm_response)

        # Bind the method
        agent._request_replan = Agent._request_replan.__get__(agent)

        # Create a task with failure
        task = Task(
            id="task-123",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.FAILED,
            priority=1,
            created_by="engineer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test task",
            description="Test description",
            retry_count=2,
            last_error="Test error",
            context={"github_repo": "test/repo"},
        )

        # Execute
        await agent._request_replan(task)

        # Verify memory section was included in prompt
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories from This Repo" in prompt
        assert "conventions" in prompt
        assert "Use pytest" in prompt

        # Verify session logging
        agent._session_logger.log.assert_called_with(
            "replan_memory_injected",
            repo="test/repo",
            chars_injected=pytest.approx(50, abs=100),  # Approximate char count
        )

    async def test_works_without_memory_when_disabled(self, tmp_path):
        """Should work without memory context when memory is disabled (no regression)."""
        # Create mock agent with memory DISABLED
        agent = MagicMock()
        agent._memory_enabled = False
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()
        agent._session_logger = MagicMock()
        agent._replan_model = "test-model"

        # Mock _get_repo_slug
        agent._get_repo_slug = MagicMock(return_value="test/repo")

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.success = True
        mock_llm_response.content = "Revised approach: Try a different strategy"
        agent.llm.complete = MagicMock(return_value=mock_llm_response)

        # Bind the method
        agent._request_replan = Agent._request_replan.__get__(agent)

        # Create a task with failure
        task = Task(
            id="task-123",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.FAILED,
            priority=1,
            created_by="engineer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test task",
            description="Test description",
            retry_count=2,
            last_error="Test error",
            context={"github_repo": "test/repo"},
        )

        # Execute
        await agent._request_replan(task)

        # Verify no memory section in prompt
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories from This Repo" not in prompt

        # Verify no session logging for memory
        for call in agent._session_logger.log.call_args_list:
            assert call[0][0] != "replan_memory_injected"

    async def test_graceful_when_no_repo_slug(self, tmp_path):
        """Should work gracefully when _get_repo_slug() returns None."""
        # Create mock agent with memory enabled
        agent = MagicMock()
        agent._memory_enabled = True
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()
        agent._session_logger = MagicMock()
        agent._replan_model = "test-model"

        # Mock _get_repo_slug to return None
        agent._get_repo_slug = MagicMock(return_value=None)

        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.success = True
        mock_llm_response.content = "Revised approach: Try a different strategy"
        agent.llm.complete = MagicMock(return_value=mock_llm_response)

        # Bind the method
        agent._request_replan = Agent._request_replan.__get__(agent)

        # Create a task without repo context
        task = Task(
            id="task-123",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.FAILED,
            priority=1,
            created_by="engineer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test task",
            description="Test description",
            retry_count=2,
            last_error="Test error",
            context={},  # No github_repo
        )

        # Execute — should not crash
        await agent._request_replan(task)

        # Verify no memory section in prompt
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories from This Repo" not in prompt
