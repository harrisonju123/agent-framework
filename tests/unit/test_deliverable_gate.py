"""Tests for the deliverable gate — blocks context-exhausted sessions from
being marked complete when no code was actually written."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, _IMPLEMENTATION_STEP_IDS, _NON_CODE_STEP_IDS
from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**overrides):
    defaults = dict(
        id="gate-task-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Add feature X to the system",
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_manager():
    config = MagicMock()
    config.id = "test-agent"
    config.base_id = "engineer"

    queue = MagicMock()
    llm = AsyncMock()
    logger = MagicMock()
    session_logger = MagicMock()
    retry_handler = MagicMock()
    retry_handler.max_retries = 3
    escalation_handler = MagicMock()
    escalation_handler.categorize_error = MagicMock(return_value="logic_error")
    workspace = MagicMock()

    return ErrorRecoveryManager(
        config=config,
        queue=queue,
        llm=llm,
        logger=logger,
        session_logger=session_logger,
        retry_handler=retry_handler,
        escalation_handler=escalation_handler,
        workspace=workspace,
    )


def _make_agent():
    """Build a MagicMock with real Agent methods bound for isolated testing."""
    a = MagicMock()
    a._handle_successful_response = Agent._handle_successful_response.__get__(a)
    a._is_implementation_step = Agent._is_implementation_step.__get__(a)
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.config.base_id = "engineer"
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a.activity_manager = MagicMock()
    a._handle_failure = AsyncMock()
    a._error_recovery = MagicMock()
    a._self_eval_enabled = False
    a._optimization_config = {}
    a._agent_definition = None
    a.queue = MagicMock()
    # Async methods called in _handle_successful_response before our gate
    a._run_sandbox_tests = AsyncMock(return_value=None)
    return a


def _ok_response(content="Done"):
    return LLMResponse(
        content=content,
        model_used="sonnet",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
        latency_ms=500,
        success=True,
    )


# ---------------------------------------------------------------------------
# has_deliverables()
# ---------------------------------------------------------------------------

class TestHasDeliverables:

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_true_when_diff_non_empty(self, mock_git):
        mock_git.side_effect = [
            MagicMock(stdout=" src/foo.py | 5 +++++\n 1 file changed"),
            MagicMock(stdout="+def foo(): pass"),
        ]
        manager = _make_manager()
        task = _make_task()

        assert manager.has_deliverables(task, Path("/tmp/repo")) is True

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_false_when_all_strategies_empty(self, mock_git):
        mock_git.return_value = MagicMock(stdout="")
        manager = _make_manager()
        task = _make_task()

        assert manager.has_deliverables(task, Path("/tmp/repo")) is False

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_logs_session_event_on_failure(self, mock_git):
        mock_git.return_value = MagicMock(stdout="")
        manager = _make_manager()
        task = _make_task()

        manager.has_deliverables(task, Path("/tmp/repo"))

        manager.session_logger.log.assert_called_once_with(
            "deliverable_gate",
            task_id=task.id,
            working_dir="/tmp/repo",
            result="no_changes",
        )

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_false_when_git_errors(self, mock_git):
        """Git failures → treated as no deliverables (will retry)."""
        mock_git.side_effect = RuntimeError("not a git repo")
        manager = _make_manager()
        task = _make_task()
        assert manager.has_deliverables(task, Path("/tmp/bad")) is False

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_no_session_event_when_changes_exist(self, mock_git):
        mock_git.side_effect = [
            MagicMock(stdout=" src/foo.py | 5 +++++"),
            MagicMock(stdout="+code"),
        ]
        manager = _make_manager()
        task = _make_task()

        manager.has_deliverables(task, Path("/tmp/repo"))
        manager.session_logger.log.assert_not_called()


# ---------------------------------------------------------------------------
# _is_implementation_step()
# ---------------------------------------------------------------------------

class TestIsImplementationStep:

    def test_true_for_implement_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "implement"})
        assert agent._is_implementation_step(task) is True

    def test_true_for_implementation_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "implementation"})
        assert agent._is_implementation_step(task) is True

    def test_true_for_direct_queue_engineer(self):
        """Engineer with no workflow step — direct-queue work is implementation."""
        agent = _make_agent()
        agent.config.base_id = "engineer"
        task = _make_task(context={})
        assert agent._is_implementation_step(task) is True

    def test_false_for_plan_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "plan"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_code_review_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "code_review"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_qa_review_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "qa_review"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_create_pr_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "create_pr"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_preview_review_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "preview_review"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_preview_step(self):
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "preview"})
        assert agent._is_implementation_step(task) is False

    def test_false_for_architect_with_no_step(self):
        """Architect with no explicit step produces plans, not code."""
        agent = _make_agent()
        agent.config.base_id = "architect"
        task = _make_task(context={})
        assert agent._is_implementation_step(task) is False

    def test_false_for_qa_with_no_step(self):
        agent = _make_agent()
        agent.config.base_id = "qa"
        task = _make_task(context={})
        assert agent._is_implementation_step(task) is False

    def test_frozensets_are_disjoint(self):
        """Sanity check: no step ID appears in both sets."""
        assert _IMPLEMENTATION_STEP_IDS.isdisjoint(_NON_CODE_STEP_IDS)


# ---------------------------------------------------------------------------
# Integration: deliverable gate in _handle_successful_response
# ---------------------------------------------------------------------------

class TestDeliverableGateIntegration:
    """Test the gate inside _handle_successful_response.

    We patch read_routing_signal to avoid filesystem access in downstream
    code that runs after the gate when the task proceeds to completion.
    """

    @pytest.mark.asyncio
    async def test_gate_fires_handle_failure_when_no_changes(self):
        """Implementation step + no git changes → _handle_failure, not mark_completed."""
        agent = _make_agent()
        agent._error_recovery.has_deliverables.return_value = False
        task = _make_task(context={"workflow_step": "implement"})
        response = _ok_response()

        await agent._handle_successful_response(
            task, response, datetime.now(timezone.utc), working_dir=Path("/tmp/repo")
        )

        agent._handle_failure.assert_awaited_once_with(task)
        assert "No code changes detected" in task.last_error
        agent.queue.mark_completed.assert_not_called()

    @pytest.mark.asyncio
    @patch("agent_framework.core.agent.read_routing_signal", return_value=None)
    async def test_gate_skips_when_working_dir_none(self, _mock_routing):
        """No working directory → gate cannot run, task completes normally."""
        agent = _make_agent()
        task = _make_task(context={"workflow_step": "implement"})
        response = _ok_response()

        await agent._handle_successful_response(
            task, response, datetime.now(timezone.utc), working_dir=None
        )

        agent._handle_failure.assert_not_awaited()
        task_arg = agent.queue.mark_completed.call_args[0][0]
        assert task_arg.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    @patch("agent_framework.core.agent.read_routing_signal", return_value=None)
    async def test_gate_skips_for_plan_step(self, _mock_routing):
        """Plan steps produce prose, not code — gate should not fire."""
        agent = _make_agent()
        agent.config.base_id = "architect"
        task = _make_task(context={"workflow_step": "plan"})
        response = _ok_response()

        await agent._handle_successful_response(
            task, response, datetime.now(timezone.utc), working_dir=Path("/tmp/repo")
        )

        agent._error_recovery.has_deliverables.assert_not_called()
        agent._handle_failure.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("agent_framework.core.agent.read_routing_signal", return_value=None)
    async def test_gate_passes_when_changes_exist(self, _mock_routing):
        """Implementation step with git changes → normal completion."""
        agent = _make_agent()
        agent._error_recovery.has_deliverables.return_value = True
        task = _make_task(context={"workflow_step": "implement"})
        response = _ok_response()

        await agent._handle_successful_response(
            task, response, datetime.now(timezone.utc), working_dir=Path("/tmp/repo")
        )

        agent._handle_failure.assert_not_awaited()
        task_arg = agent.queue.mark_completed.call_args[0][0]
        assert task_arg.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    @patch("agent_framework.core.agent.read_routing_signal", return_value=None)
    async def test_gate_skips_for_code_review(self, _mock_routing):
        """Code review is prose-only — exempt from deliverable gate."""
        agent = _make_agent()
        agent.config.base_id = "architect"
        task = _make_task(context={"workflow_step": "code_review"})
        response = _ok_response()

        await agent._handle_successful_response(
            task, response, datetime.now(timezone.utc), working_dir=Path("/tmp/repo")
        )

        agent._error_recovery.has_deliverables.assert_not_called()
        agent._handle_failure.assert_not_awaited()
