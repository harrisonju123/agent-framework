"""Tests for Agent._self_evaluate() — happy path, retry limit, critique injection, disabled skip."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides) -> Task:
    """Factory for minimal Task instances."""
    defaults = dict(
        id="task-eval-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A task for testing self-eval",
        acceptance_criteria=["Tests pass", "Lint clean"],
        context={},
        notes=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_response(content: str = "Some output") -> SimpleNamespace:
    """Minimal object with .content attribute to pass as response."""
    return SimpleNamespace(content=content)


def _make_llm_response(content: str, success: bool = True) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used="haiku",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
        latency_ms=200,
        success=success,
    )


def _build_agent_for_eval(tmp_path, llm_mock=None) -> Agent:
    """Construct an Agent with minimal deps, focused on self-eval config."""
    from agent_framework.core.agent import AgentConfig

    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        queue="engineer",
        prompt="You are a test agent.",
    )
    llm = llm_mock or AsyncMock()
    queue = MagicMock()

    with patch("agent_framework.core.agent.setup_rich_logging") as mock_log, \
         patch("agent_framework.workflow.executor.WorkflowExecutor"):
        mock_log.return_value = MagicMock()
        agent = Agent(
            config=config,
            llm=llm,
            queue=queue,
            workspace=tmp_path,
            self_eval_config={"enabled": True, "max_retries": 2, "model": "haiku"},
        )
    return agent


class TestSelfEvalPassed:
    async def test_passes_when_llm_says_pass(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("PASS — all criteria met"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        response = _make_response("Good output")
        result = await agent._self_evaluate(task, response)
        assert result is True

    async def test_passes_case_insensitive(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("pass - looks good"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        result = await agent._self_evaluate(task, _make_response())
        assert result is True


class TestSelfEvalFailed:
    async def test_fails_resets_task_to_pending(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("FAIL — tests not run"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        result = await agent._self_evaluate(task, _make_response())
        assert result is False
        assert task.status == TaskStatus.PENDING
        assert task.context["_self_eval_count"] == 1
        assert "FAIL" in task.context["_self_eval_critique"]

    async def test_failed_critique_stored_in_notes(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("FAIL — lint errors found"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        await agent._self_evaluate(task, _make_response())
        assert any("Self-eval failed" in n for n in task.notes)

    async def test_resets_started_at_and_started_by(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("FAIL — missing tests"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        task.started_at = datetime.now(timezone.utc)
        task.started_by = "engineer-1"
        await agent._self_evaluate(task, _make_response())
        assert task.started_at is None
        assert task.started_by is None


class TestSelfEvalRetryLimit:
    async def test_passes_when_retry_limit_reached(self, tmp_path):
        agent = _build_agent_for_eval(tmp_path)
        task = _make_task(context={"_self_eval_count": 2})
        result = await agent._self_evaluate(task, _make_response())
        assert result is True

    async def test_does_not_call_llm_at_limit(self, tmp_path):
        llm = AsyncMock()
        agent = _build_agent_for_eval(tmp_path, llm)
        task = _make_task(context={"_self_eval_count": 2})
        await agent._self_evaluate(task, _make_response())
        llm.complete.assert_not_called()


class TestSelfEvalNoCriteria:
    async def test_passes_when_no_acceptance_criteria(self, tmp_path):
        agent = _build_agent_for_eval(tmp_path)
        task = _make_task(acceptance_criteria=[])
        result = await agent._self_evaluate(task, _make_response())
        assert result is True


class TestSelfEvalLLMError:
    async def test_passes_on_llm_failure(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("", success=False))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        result = await agent._self_evaluate(task, _make_response())
        assert result is True

    async def test_passes_on_llm_exception(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("network error"))
        agent = _build_agent_for_eval(tmp_path, llm)

        task = _make_task()
        result = await agent._self_evaluate(task, _make_response())
        assert result is True
