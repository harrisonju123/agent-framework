"""Integration tests for retry awareness: git evidence flows from failure → prompt."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides):
    defaults = dict(
        id="retry-aware-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.now(timezone.utc),
        title="Implement feature",
        description="Add feature",
        context={"github_repo": "org/repo"},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_agent():
    a = MagicMock()
    a._handle_failed_response = Agent._handle_failed_response.__get__(a)
    a._extract_partial_progress = Agent._extract_partial_progress
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a.activity_manager = MagicMock()
    a._handle_failure = AsyncMock()
    a._error_recovery = MagicMock()
    return a


class TestHandleFailedResponseGitEvidence:
    """_handle_failed_response captures git evidence when working_dir is provided."""

    @pytest.mark.asyncio
    async def test_captures_git_evidence(self):
        agent = _make_agent()
        task = _make_task()
        response = LLMResponse(
            content="partial output here",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="error",
            latency_ms=500,
            success=False,
            error="Circuit breaker tripped",
        )

        working_dir = Path("/tmp/test-worktree")
        agent._error_recovery.gather_git_evidence.return_value = (
            "## Git Diff\n### Summary\n```\n src/auth.py | 10 ++++\n```"
        )

        with patch.object(Path, "exists", return_value=True):
            await agent._handle_failed_response(task, response, working_dir=working_dir)

        assert "_previous_attempt_git_diff" in task.context
        assert "src/auth.py" in task.context["_previous_attempt_git_diff"]

    @pytest.mark.asyncio
    async def test_no_git_evidence_without_working_dir(self):
        agent = _make_agent()
        task = _make_task()
        response = LLMResponse(
            content="",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="error",
            latency_ms=500,
            success=False,
            error="Some error",
        )

        await agent._handle_failed_response(task, response)

        assert "_previous_attempt_git_diff" not in task.context
        agent._error_recovery.gather_git_evidence.assert_not_called()

    @pytest.mark.asyncio
    async def test_git_evidence_exception_non_fatal(self):
        agent = _make_agent()
        task = _make_task()
        response = LLMResponse(
            content="",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="error",
            latency_ms=500,
            success=False,
            error="Some error",
        )

        agent._error_recovery.gather_git_evidence.side_effect = RuntimeError("git broken")

        with patch.object(Path, "exists", return_value=True):
            # Should not raise
            await agent._handle_failed_response(task, response, working_dir=Path("/tmp/bad"))

        assert "_previous_attempt_git_diff" not in task.context

    @pytest.mark.asyncio
    async def test_empty_git_evidence_not_stored(self):
        """Empty git evidence (clean worktree) is not stored in context."""
        agent = _make_agent()
        task = _make_task()
        response = LLMResponse(
            content="",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="error",
            latency_ms=500,
            success=False,
            error="Some error",
        )

        agent._error_recovery.gather_git_evidence.return_value = ""

        with patch.object(Path, "exists", return_value=True):
            await agent._handle_failed_response(task, response, working_dir=Path("/tmp/clean"))

        assert "_previous_attempt_git_diff" not in task.context

    @pytest.mark.asyncio
    async def test_both_partial_output_and_git_evidence(self):
        """Both partial output and git diff captured together."""
        agent = _make_agent()
        task = _make_task()
        response = LLMResponse(
            content="I created the auth service with JWT tokens",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="error",
            latency_ms=500,
            success=False,
            error="Circuit breaker tripped",
        )

        agent._error_recovery.gather_git_evidence.return_value = "## Git Diff\n+def auth():"

        with patch.object(Path, "exists", return_value=True):
            await agent._handle_failed_response(task, response, working_dir=Path("/tmp/work"))

        assert "_previous_attempt_summary" in task.context
        assert "_previous_attempt_git_diff" in task.context
        assert "auth" in task.context["_previous_attempt_summary"].lower()
        assert "+def auth():" in task.context["_previous_attempt_git_diff"]
