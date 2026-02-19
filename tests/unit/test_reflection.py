"""Tests for Agent._self_evaluate() — self-evaluation loop."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides):
    defaults = dict(
        id="test-task-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        acceptance_criteria=["Code compiles", "Tests pass"],
        context={},
        notes=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


class _AgentMock:
    """Custom mock that forwards llm to _error_recovery."""

    def __init__(self, error_recovery):
        self._error_recovery = error_recovery
        self._session_logger = error_recovery.session_logger
        self.logger = error_recovery.logger
        self.queue = error_recovery.queue
        self._self_evaluate = Agent._self_evaluate.__get__(self)

    def __setattr__(self, name, value):
        if name == "llm":
            # Forward to error_recovery
            self._error_recovery.llm = value
        object.__setattr__(self, name, value)


def _make_agent(**overrides):
    """Build a mock agent with _self_evaluate bound from the real class."""
    # The Agent._self_evaluate now delegates to _error_recovery.self_evaluate()
    # So we bind the ErrorRecoveryManager.self_evaluate method instead
    error_recovery = MagicMock()
    error_recovery.self_evaluate = ErrorRecoveryManager.self_evaluate.__get__(error_recovery)
    error_recovery.gather_git_evidence = ErrorRecoveryManager.gather_git_evidence.__get__(error_recovery)
    error_recovery._try_diff_strategies = ErrorRecoveryManager._try_diff_strategies.__get__(error_recovery)
    error_recovery._self_eval_max_retries = overrides.get("max_retries", 2)
    error_recovery._self_eval_model = overrides.get("model", "haiku")
    error_recovery.session_logger = MagicMock()
    error_recovery.logger = MagicMock()
    error_recovery.queue = MagicMock()
    error_recovery.llm = AsyncMock()

    agent = _AgentMock(error_recovery)
    return agent


def _llm_response(content: str, success: bool = True):
    return LLMResponse(
        content=content,
        model_used="haiku",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
        latency_ms=200,
        success=success,
    )


class TestSelfEvalPasses:
    @pytest.mark.asyncio
    async def test_pass_verdict_returns_true(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("PASS — all criteria met"))

        task = _make_task()
        result = await agent._self_evaluate(task, _llm_response("agent output"))

        assert result is True

    @pytest.mark.asyncio
    async def test_pass_case_insensitive(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("pass"))

        task = _make_task()
        result = await agent._self_evaluate(task, _llm_response("output"))

        assert result is True


class TestSelfEvalFails:
    @pytest.mark.asyncio
    async def test_fail_resets_task_to_pending(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("FAIL — missing test file")
        )

        task = _make_task()
        result = await agent._self_evaluate(task, _llm_response("output"), test_passed=True)

        assert result is False
        assert task.status == TaskStatus.PENDING
        assert task.context["_self_eval_count"] == 1
        assert "FAIL" in task.context["_self_eval_critique"]

    @pytest.mark.asyncio
    async def test_fail_appends_note(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("FAIL — tests not passing")
        )

        task = _make_task()
        await agent._self_evaluate(task, _llm_response("output"), test_passed=True)

        assert any("Self-eval failed" in n for n in task.notes)


class TestRetryLimit:
    @pytest.mark.asyncio
    async def test_skips_when_retry_limit_reached(self):
        agent = _make_agent(max_retries=2)
        agent.llm = AsyncMock()

        task = _make_task(context={"_self_eval_count": 2})
        result = await agent._self_evaluate(task, _llm_response("output"))

        # Should return True without calling LLM
        assert result is True
        agent.llm.complete.assert_not_called()


class TestNoCriteria:
    @pytest.mark.asyncio
    async def test_skips_when_no_acceptance_criteria(self):
        agent = _make_agent()
        agent.llm = AsyncMock()

        task = _make_task(acceptance_criteria=[])
        result = await agent._self_evaluate(task, _llm_response("output"))

        assert result is True
        agent.llm.complete.assert_not_called()


class TestLLMFailure:
    @pytest.mark.asyncio
    async def test_llm_failure_returns_true(self):
        """Non-fatal: if eval LLM fails, proceed without blocking."""
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("", success=False)
        )

        task = _make_task()
        result = await agent._self_evaluate(task, _llm_response("output"))

        assert result is True

    @pytest.mark.asyncio
    async def test_llm_exception_returns_true(self):
        """Non-fatal: if eval LLM raises, proceed without blocking."""
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(side_effect=RuntimeError("API down"))

        task = _make_task()
        result = await agent._self_evaluate(task, _llm_response("output"))

        assert result is True


class TestSessionLogging:
    @pytest.mark.asyncio
    async def test_logs_pass_verdict(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("PASS"))

        task = _make_task()
        await agent._self_evaluate(task, _llm_response("output"), test_passed=True)

        agent._session_logger.log.assert_called_once()
        call_kwargs = agent._session_logger.log.call_args
        assert call_kwargs[1]["verdict"] == "PASS"


class TestGatherGitEvidence:
    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_diff_summary(self, mock_git):
        """Collects stat + diff via multi-strategy approach (HEAD first)."""
        stat_out = "file.py | 10 ++++\n 1 file changed"
        diff_out = "+def foo():\n+    return 42"
        mock_git.side_effect = [
            MagicMock(stdout=stat_out),  # diff --stat HEAD
            MagicMock(stdout=diff_out),  # diff HEAD
        ]

        er = MagicMock()
        er.gather_git_evidence = ErrorRecoveryManager.gather_git_evidence.__get__(er)
        er._try_diff_strategies = ErrorRecoveryManager._try_diff_strategies.__get__(er)
        result = er.gather_git_evidence(Path("/tmp/repo"))

        assert "Git Diff" in result
        assert "file.py" in result
        assert "+def foo()" in result
        assert mock_git.call_count == 2
        # First call should be diff --stat HEAD (not HEAD~1)
        assert mock_git.call_args_list[0][0][0] == ["diff", "--stat", "HEAD"]

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_empty_on_error(self, mock_git):
        """Non-fatal: returns empty string when all strategies fail."""
        mock_git.side_effect = RuntimeError("not a git repo")

        er = MagicMock()
        er.gather_git_evidence = ErrorRecoveryManager.gather_git_evidence.__get__(er)
        er._try_diff_strategies = ErrorRecoveryManager._try_diff_strategies.__get__(er)
        result = er.gather_git_evidence(Path("/tmp/bad"))

        assert result == ""


class TestSelfEvalWithEvidence:
    @pytest.mark.asyncio
    async def test_git_diff_in_prompt(self):
        """When working_dir is provided, git diff appears in the eval prompt."""
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("PASS"))

        with patch("agent_framework.core.error_recovery.run_git_command") as mock_git:
            mock_git.side_effect = [
                MagicMock(stdout="main.py | 5 +++++"),  # diff --stat HEAD
                MagicMock(stdout="+print('hello')"),     # diff HEAD
            ]

            task = _make_task()
            result = await agent._self_evaluate(
                task, _llm_response("agent output"),
                working_dir=Path("/tmp/repo"),
            )

        assert result is True
        # Verify the eval prompt sent to LLM contains git diff
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "Git Diff" in prompt
        assert "main.py" in prompt
        # First git call should use HEAD strategy
        assert mock_git.call_args_list[0][0][0] == ["diff", "--stat", "HEAD"]

    @pytest.mark.asyncio
    async def test_test_passed_in_prompt(self):
        """When test_passed=True, 'PASSED' appears in the eval prompt."""
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("PASS"))

        task = _make_task()
        result = await agent._self_evaluate(
            task, _llm_response("output"),
            test_passed=True,
        )

        assert result is True
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Test Results\nPASSED" in prompt

    @pytest.mark.asyncio
    async def test_no_working_dir_skips_git(self):
        """Without working_dir, no git call is made but eval still works."""
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=_llm_response("PASS"))

        with patch("agent_framework.core.error_recovery.run_git_command") as mock_git:
            task = _make_task()
            result = await agent._self_evaluate(
                task, _llm_response("output"),
            )

        assert result is True
        mock_git.assert_not_called()
