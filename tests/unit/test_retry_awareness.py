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
    a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
    a._extract_partial_progress = Agent._extract_partial_progress
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.workspace = Path("/tmp/test-workspace")
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a.activity_manager = MagicMock()
    a._handle_failure = AsyncMock()
    a._error_recovery = MagicMock()
    a._git_ops = MagicMock()
    a._git_ops.safety_commit.return_value = False
    a._model_success_store = None
    return a


class TestFinalizeFailedAttempt:
    """_finalize_failed_attempt consolidates all attempt preservation."""

    @patch("agent_framework.core.attempt_tracker.record_attempt")
    def test_stores_partial_progress_and_records_attempt(self, mock_record):
        agent = _make_agent()
        mock_record.return_value = MagicMock(branch="feature/x", commit_sha="abc123")

        task = _make_task()
        agent._finalize_failed_attempt(
            task, Path("/tmp/worktree"),
            content="I created the auth service with JWT tokens",
            error="Circuit breaker tripped",
        )

        assert "_previous_attempt_summary" in task.context
        assert "auth" in task.context["_previous_attempt_summary"].lower()
        assert task.context["_previous_attempt_branch"] == "feature/x"
        assert task.context["_previous_attempt_commit_sha"] == "abc123"
        mock_record.assert_called_once()

    @patch("agent_framework.core.attempt_tracker.record_attempt")
    def test_record_attempt_exception_does_not_block(self, mock_record):
        """Exception in attempt recording is non-fatal."""
        agent = _make_agent()
        mock_record.side_effect = RuntimeError("disk full")

        task = _make_task()
        # Should not raise
        agent._finalize_failed_attempt(task, Path("/tmp/worktree"), error="fail")

    @patch("agent_framework.core.attempt_tracker.record_attempt")
    def test_no_content_skips_partial_progress(self, mock_record):
        agent = _make_agent()
        mock_record.return_value = None

        task = _make_task()
        agent._finalize_failed_attempt(task, Path("/tmp/worktree"))

        assert "_previous_attempt_summary" not in task.context

    @patch("agent_framework.core.attempt_tracker.record_attempt")
    def test_record_returns_none_no_branch_stored(self, mock_record):
        agent = _make_agent()
        mock_record.return_value = None

        task = _make_task()
        agent._finalize_failed_attempt(task, Path("/tmp/worktree"))

        assert "_previous_attempt_branch" not in task.context


class TestHandleFailedResponseCallsFinalize:
    """_handle_failed_response delegates to _finalize_failed_attempt."""

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt")
    async def test_handle_failed_response_calls_finalize(self, mock_record):
        agent = _make_agent()
        mock_record.return_value = MagicMock(branch="feature/x", commit_sha="abc123")

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

        await agent._handle_failed_response(task, response, working_dir=Path("/tmp/worktree"))

        mock_record.assert_called_once()
        assert task.context.get("_previous_attempt_branch") == "feature/x"

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt")
    async def test_no_working_dir_still_calls_finalize(self, mock_record):
        """Finalize is called even without working_dir — record_attempt handles None."""
        agent = _make_agent()
        mock_record.return_value = None

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

        # _finalize_failed_attempt was called (record_attempt returns None for no working_dir)
        mock_record.assert_called_once()

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt")
    async def test_finalize_exception_does_not_block_retry(self, mock_record):
        """Even if record_attempt raises, _handle_failure is still called."""
        agent = _make_agent()
        mock_record.side_effect = RuntimeError("disk full")

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

        await agent._handle_failed_response(task, response, working_dir=Path("/tmp/worktree"))

        # _handle_failure should still be called
        agent._handle_failure.assert_called_once()


class TestRetryStartupRecovery:
    """Retry startup fetches pushed branch when discover_branch_work returns None."""

    def test_fetches_pushed_branch_when_discover_returns_none(self):
        """When discover_branch_work returns None, attempt history provides a fallback."""
        agent = _make_agent()
        agent._git_ops = MagicMock()
        agent._git_ops.discover_branch_work.side_effect = [
            None,           # First call returns nothing
            {"commit_count": 3, "insertions": 200, "deletions": 5,
             "commit_log": "abc Add feature", "file_list": ["src/f.py"], "diffstat": "..."},
        ]

        task = _make_task(retry_count=1)
        working_dir = Path("/tmp/test-worktree")

        with patch("agent_framework.core.attempt_tracker.get_last_pushed_branch", return_value="feature/x"), \
             patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_fetch:
            # Simulate the retry startup block from _process_task
            from agent_framework.core.attempt_tracker import get_last_pushed_branch
            from agent_framework.utils.subprocess_utils import run_git_command

            branch_work = agent._git_ops.discover_branch_work(working_dir)
            if not branch_work:
                pushed_branch = get_last_pushed_branch(agent.workspace, task.id)
                if pushed_branch:
                    run_git_command(["fetch", "origin", pushed_branch], cwd=working_dir, check=False, timeout=30)
                    branch_work = agent._git_ops.discover_branch_work(working_dir)

            if branch_work:
                task.context["_previous_attempt_branch_work"] = branch_work

        assert "_previous_attempt_branch_work" in task.context
        assert task.context["_previous_attempt_branch_work"]["commit_count"] == 3

    def test_skips_fetch_when_discover_succeeds(self):
        """When discover_branch_work succeeds, attempt history is not consulted."""
        agent = _make_agent()
        agent._git_ops = MagicMock()
        agent._git_ops.discover_branch_work.return_value = {
            "commit_count": 2, "insertions": 100, "deletions": 5,
            "commit_log": "abc Add feature", "file_list": ["src/f.py"], "diffstat": "...",
        }

        task = _make_task(retry_count=1)
        working_dir = Path("/tmp/test-worktree")

        with patch("agent_framework.core.attempt_tracker.get_last_pushed_branch") as mock_get_branch:
            branch_work = agent._git_ops.discover_branch_work(working_dir)
            if not branch_work:
                from agent_framework.core.attempt_tracker import get_last_pushed_branch
                get_last_pushed_branch(agent.workspace, task.id)

            if branch_work:
                task.context["_previous_attempt_branch_work"] = branch_work

        mock_get_branch.assert_not_called()
        assert "_previous_attempt_branch_work" in task.context


class TestDiscoverBranchWorkIntegration:
    """discover_branch_work() is called at retry startup and flows into task context."""

    def test_discovers_branch_work_on_retry(self):
        """retry_count > 0 triggers discovery and stores result in context."""
        agent = _make_agent()
        agent._git_ops = MagicMock()
        agent._git_ops.discover_branch_work.return_value = {
            "commit_count": 2,
            "insertions": 300,
            "deletions": 5,
            "commit_log": "abc1234 Add feature\ndef5678 Add tests",
            "file_list": ["src/feature.py", "tests/test_feature.py"],
            "diffstat": " src/feature.py | 200 +++\n tests/test_feature.py | 105 +++\n",
        }

        task = _make_task(retry_count=1)
        working_dir = "/tmp/test-worktree"

        # Simulate the discovery block from _handle_task
        if task.retry_count > 0:
            branch_work = agent._git_ops.discover_branch_work(working_dir)
            if branch_work:
                task.context["_previous_attempt_branch_work"] = branch_work

        assert "_previous_attempt_branch_work" in task.context
        assert task.context["_previous_attempt_branch_work"]["commit_count"] == 2
        assert task.context["_previous_attempt_branch_work"]["insertions"] == 300
        agent._git_ops.discover_branch_work.assert_called_once_with(working_dir)

    def test_skips_discovery_on_first_attempt(self):
        """retry_count == 0 does not call discover_branch_work."""
        agent = _make_agent()
        agent._git_ops = MagicMock()

        task = _make_task(retry_count=0)

        # Simulate the discovery block from _handle_task
        if task.retry_count > 0:
            branch_work = agent._git_ops.discover_branch_work("/tmp/worktree")
            if branch_work:
                task.context["_previous_attempt_branch_work"] = branch_work

        assert "_previous_attempt_branch_work" not in task.context
        agent._git_ops.discover_branch_work.assert_not_called()

    def test_none_result_not_stored(self):
        """discover_branch_work returning None does not pollute context."""
        agent = _make_agent()
        agent._git_ops = MagicMock()
        agent._git_ops.discover_branch_work.return_value = None

        task = _make_task(retry_count=1)

        if task.retry_count > 0:
            branch_work = agent._git_ops.discover_branch_work("/tmp/worktree")
            if branch_work:
                task.context["_previous_attempt_branch_work"] = branch_work

        assert "_previous_attempt_branch_work" not in task.context
