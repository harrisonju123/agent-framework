"""Tests for Agent._self_evaluate() — self-evaluation loop."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.agent import Agent
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


def _make_agent(**overrides):
    """Build a mock agent with _self_evaluate bound from the real class."""
    agent = MagicMock()
    agent._self_evaluate = Agent._self_evaluate.__get__(agent)
    agent._self_eval_max_retries = overrides.get("max_retries", 2)
    agent._self_eval_model = overrides.get("model", "haiku")
    agent._session_logger = MagicMock()
    agent.logger = MagicMock()
    agent.queue = MagicMock()
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
        result = await agent._self_evaluate(task, _llm_response("output"))

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
        await agent._self_evaluate(task, _llm_response("output"))

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
        await agent._self_evaluate(task, _llm_response("output"))

        agent._session_logger.log.assert_called_once()
        call_kwargs = agent._session_logger.log.call_args
        assert call_kwargs[1]["verdict"] == "PASS"
