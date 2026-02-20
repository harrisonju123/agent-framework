"""Tests for preserving partial LLM output on SIGTERM/interruption."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMBackend
from agent_framework.llm.claude_cli_backend import ClaudeCLIBackend


class TestClaudeCLIBackendPartialOutput:
    """ClaudeCLIBackend.get_partial_output() returns joined chunks."""

    def test_get_partial_output_returns_joined_chunks(self):
        """Accumulated text_chunks are returned as a single string."""
        backend = MagicMock(spec=ClaudeCLIBackend)
        backend._partial_output = ["First chunk. ", "Second chunk. ", "Third."]
        backend.get_partial_output = ClaudeCLIBackend.get_partial_output.__get__(backend)

        result = backend.get_partial_output()
        assert result == "First chunk. Second chunk. Third."

    def test_get_partial_output_empty_list(self):
        """Empty partial output returns empty string."""
        backend = MagicMock(spec=ClaudeCLIBackend)
        backend._partial_output = []
        backend.get_partial_output = ClaudeCLIBackend.get_partial_output.__get__(backend)

        assert backend.get_partial_output() == ""

    def test_get_partial_output_none_safe(self):
        """get_partial_output returns empty string when _partial_output is falsy."""
        backend = MagicMock(spec=ClaudeCLIBackend)
        backend._partial_output = None
        backend.get_partial_output = ClaudeCLIBackend.get_partial_output.__get__(backend)

        assert backend.get_partial_output() == ""


class TestLLMBackendBasePartialOutput:
    """Base LLMBackend.get_partial_output() returns empty string."""

    def test_base_get_partial_output_returns_empty(self):
        backend = MagicMock(spec=LLMBackend)
        backend.get_partial_output = LLMBackend.get_partial_output.__get__(backend)
        assert backend.get_partial_output() == ""


class TestInterruptionPartialProgress:
    """Interruption path in _execute_llm_with_interruption_watch preserves partial output."""

    @pytest.fixture
    def task(self):
        return Task(
            id="test-interrupt-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Code review for auth module",
            description="Review the auth module changes",
            context={"github_repo": "org/repo"},
        )

    @pytest.fixture
    def agent(self):
        """Minimal mock agent with the real _execute_llm_with_interruption_watch bound."""
        a = MagicMock()
        a._execute_llm_with_interruption_watch = (
            Agent._execute_llm_with_interruption_watch.__get__(a)
        )
        a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
        a._extract_partial_progress = Agent._extract_partial_progress
        a._update_phase = MagicMock()
        a._session_logger = MagicMock()
        a._session_logger.log = MagicMock()
        a._session_logger.log_tool_call = MagicMock()
        a._context_window_manager = None
        a._current_specialization = None
        a._current_file_count = 0
        a.config = MagicMock()
        a.config.id = "test-agent"
        a.workspace = Path("/tmp/test-workspace")
        a.logger = MagicMock()
        a.queue = MagicMock()
        a.activity_manager = MagicMock()
        return a

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_interruption_saves_partial_progress(self, _mock_record, agent, task):
        """When interruption wins the race and LLM has partial output, it's preserved."""
        # Make the watcher finish immediately (simulating interruption)
        agent._watch_for_interruption = AsyncMock(return_value=None)

        # LLM complete hangs forever (will be cancelled)
        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(999)

        agent.llm = MagicMock()
        agent.llm.complete = slow_llm
        agent.llm.cancel = MagicMock()
        agent.llm.get_partial_output = MagicMock(
            return_value="I reviewed the auth module.\n[Tool Call: Read]\nFound 3 issues in token validation."
        )

        result = await agent._execute_llm_with_interruption_watch(
            task, "review this code", MagicMock(), None
        )

        assert result is None
        assert task.last_error == "Interrupted during LLM execution"
        assert "_previous_attempt_summary" in task.context
        assert "token validation" in task.context["_previous_attempt_summary"]

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_interruption_no_partial_output(self, _mock_record, agent, task):
        """When interruption happens but no output was generated, no summary is saved."""
        agent._watch_for_interruption = AsyncMock(return_value=None)

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(999)

        agent.llm = MagicMock()
        agent.llm.complete = slow_llm
        agent.llm.cancel = MagicMock()
        agent.llm.get_partial_output = MagicMock(return_value="")

        result = await agent._execute_llm_with_interruption_watch(
            task, "review this code", MagicMock(), None
        )

        assert result is None
        assert task.last_error == "Interrupted during LLM execution"
        assert "_previous_attempt_summary" not in task.context

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_interruption_only_tool_calls_no_summary(self, _mock_record, agent, task):
        """Partial output that's all tool-call noise produces no summary."""
        agent._watch_for_interruption = AsyncMock(return_value=None)

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(999)

        agent.llm = MagicMock()
        agent.llm.complete = slow_llm
        agent.llm.cancel = MagicMock()
        # Only tool call markers, no meaningful text
        agent.llm.get_partial_output = MagicMock(
            return_value="[Tool Call: Read][Tool Call: Glob]"
        )

        result = await agent._execute_llm_with_interruption_watch(
            task, "review this code", MagicMock(), None
        )

        assert result is None
        assert "_previous_attempt_summary" not in task.context
