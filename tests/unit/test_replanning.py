"""Tests for Agent._request_replan() and _inject_replan_context()."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.prompt_builder import PromptBuilder
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
        title="Implement feature X",
        description="Add feature X to the system",
        last_error="TypeError: expected str got int",
        retry_count=2,
        replan_history=[],
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


class _AgentMock:
    """Custom mock that forwards llm and memory_store to _error_recovery."""

    def __init__(self, error_recovery):
        self._error_recovery = error_recovery
        self._session_logger = error_recovery.session_logger
        self.logger = error_recovery.logger
        self._request_replan = Agent._request_replan.__get__(self)
        self._inject_replan_context = Agent._inject_replan_context.__get__(self)
        self._build_replan_memory_context = lambda task: error_recovery._build_replan_memory_context(task)
        self._replan_model = "haiku"

    def __setattr__(self, name, value):
        if name == "llm":
            # Forward to error_recovery
            self._error_recovery.llm = value
        elif name == "_memory_store":
            self._error_recovery.memory_store = value
        elif name == "_memory_enabled":
            self._error_recovery.memory_store = MagicMock() if value else None
            if value:
                self._error_recovery.memory_store.enabled = True
        elif name == "config":
            self._error_recovery.config = value
        object.__setattr__(self, name, value)


def _make_agent():
    from agent_framework.core.error_recovery import ErrorRecoveryManager

    # The Agent._request_replan and _inject_replan_context now delegate to ErrorRecoveryManager
    # So we bind the ErrorRecoveryManager methods instead
    error_recovery = MagicMock()
    error_recovery.request_replan = ErrorRecoveryManager.request_replan.__get__(error_recovery)
    error_recovery.inject_replan_context = ErrorRecoveryManager.inject_replan_context.__get__(error_recovery)
    error_recovery._build_replan_memory_context = ErrorRecoveryManager._build_replan_memory_context.__get__(error_recovery)
    error_recovery._get_repo_slug = ErrorRecoveryManager._get_repo_slug.__get__(error_recovery)
    error_recovery._replan_model = "haiku"
    error_recovery.session_logger = MagicMock()
    error_recovery.logger = MagicMock()
    error_recovery.llm = AsyncMock()
    error_recovery.memory_store = None

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


class TestRequestReplan:
    @pytest.mark.asyncio
    async def test_stores_revised_plan_in_context(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("1. Try approach B\n2. Use different API")
        )

        task = _make_task()
        await agent._request_replan(task)

        assert task.context["_revised_plan"] == "1. Try approach B\n2. Use different API"
        assert task.context["_replan_attempt"] == 2

    @pytest.mark.asyncio
    async def test_appends_to_replan_history(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("New approach")
        )

        task = _make_task()
        await agent._request_replan(task)

        assert len(task.replan_history) == 1
        entry = task.replan_history[0]
        assert entry["attempt"] == 2
        assert "TypeError" in entry["error"]
        assert entry["revised_plan"] == "New approach"

    @pytest.mark.asyncio
    async def test_includes_previous_attempts_in_prompt(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("Third approach")
        )

        task = _make_task(
            retry_count=3,
            replan_history=[
                {"attempt": 2, "error": "first error", "revised_plan": "plan A"}
            ],
        )
        await agent._request_replan(task)

        # Verify the prompt included previous attempt info
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "first error" in prompt

    @pytest.mark.asyncio
    async def test_llm_failure_non_fatal(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("", success=False)
        )

        task = _make_task()
        await agent._request_replan(task)

        # Should not crash, context unchanged
        assert "_revised_plan" not in task.context

    @pytest.mark.asyncio
    async def test_llm_exception_non_fatal(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(side_effect=RuntimeError("API down"))

        task = _make_task()
        await agent._request_replan(task)

        assert "_revised_plan" not in task.context

    @pytest.mark.asyncio
    async def test_truncates_long_revised_plan(self):
        agent = _make_agent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("x" * 3000)
        )

        task = _make_task()
        await agent._request_replan(task)

        assert len(task.context["_revised_plan"]) <= 2000


class TestInjectReplanContext:
    def test_no_op_without_revised_plan(self):
        agent = _make_agent()
        task = _make_task()

        result = agent._error_recovery.inject_replan_context("original prompt", task)
        assert result == "original prompt"

    def test_appends_revised_approach(self):
        agent = _make_agent()
        task = _make_task(context={"_revised_plan": "Try approach B"})

        result = agent._error_recovery.inject_replan_context("original prompt", task)

        assert "original prompt" in result
        assert "REVISED APPROACH" in result
        assert "Try approach B" in result

    def test_includes_self_eval_critique(self):
        agent = _make_agent()
        task = _make_task(
            context={
                "_revised_plan": "Try approach B",
                "_self_eval_critique": "Missing test coverage",
            }
        )

        result = agent._error_recovery.inject_replan_context("prompt", task)

        assert "Self-Evaluation Feedback" in result
        assert "Missing test coverage" in result

    def test_includes_previous_attempt_history(self):
        agent = _make_agent()
        task = _make_task(
            context={"_revised_plan": "Latest plan"},
            replan_history=[
                {"attempt": 2, "error": "first error", "revised_plan": "plan A"},
                {"attempt": 3, "error": "second error", "revised_plan": "Latest plan"},
            ],
        )

        result = agent._error_recovery.inject_replan_context("prompt", task)

        assert "Previous Attempt History" in result
        assert "first error" in result
        # Current plan (last entry) should not appear in history section again


class TestReplanWithMemoryIntegration:
    @pytest.mark.asyncio
    async def test_memory_context_injected_in_replan_prompt(self):
        from agent_framework.memory.memory_store import MemoryEntry

        agent = _make_agent()
        agent._memory_enabled = True
        agent._memory_store = MagicMock()
        # _build_replan_memory_context is now in ErrorRecoveryManager
        # It's already bound in _make_agent()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("New approach based on memories")
        )

        # Mock memory retrieval
        agent._memory_store.recall = MagicMock(return_value=[
            MemoryEntry(category="conventions", content="Use type hints everywhere"),
            MemoryEntry(category="test_commands", content="pytest -v for verbose output"),
        ])

        task = _make_task(context={"github_repo": "owner/repo"})
        await agent._request_replan(task)

        # Verify LLM was called with memory context
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt

        assert "Relevant Context from Previous Work" in prompt
        assert "Use type hints everywhere" in prompt
        assert "pytest -v for verbose output" in prompt

    @pytest.mark.asyncio
    async def test_replan_graceful_when_memory_disabled(self):
        agent = _make_agent()
        agent._memory_enabled = False
        # _build_replan_memory_context is now in ErrorRecoveryManager
        # It's already bound in _make_agent()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(
            return_value=_llm_response("Revised approach without memories")
        )

        task = _make_task(context={"github_repo": "owner/repo"})
        await agent._request_replan(task)

        # Should complete without error, no memory context in prompt
        call_args = agent.llm.complete.call_args
        prompt = call_args[0][0].prompt

        assert "Relevant Context from Previous Work" not in prompt
        assert task.context["_revised_plan"] == "Revised approach without memories"
