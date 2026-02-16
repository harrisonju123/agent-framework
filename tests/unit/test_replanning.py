"""Tests for Agent._request_replan() and _inject_replan_context()."""

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
        title="Implement feature X",
        description="Add feature X to the system",
        last_error="TypeError: expected str got int",
        retry_count=2,
        replan_history=[],
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_agent():
    agent = MagicMock()
    agent._request_replan = Agent._request_replan.__get__(agent)
    agent._inject_replan_context = Agent._inject_replan_context.__get__(agent)
    agent._replan_model = "haiku"
    agent._session_logger = MagicMock()
    agent.logger = MagicMock()
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

        result = agent._inject_replan_context("original prompt", task)
        assert result == "original prompt"

    def test_appends_revised_approach(self):
        agent = _make_agent()
        task = _make_task(context={"_revised_plan": "Try approach B"})

        result = agent._inject_replan_context("original prompt", task)

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

        result = agent._inject_replan_context("prompt", task)

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

        result = agent._inject_replan_context("prompt", task)

        assert "Previous Attempt History" in result
        assert "first error" in result
        # Current plan (last entry) should not appear in history section again


class TestBuildReplanMemoryContext:
    def test_returns_empty_when_memory_disabled(self):
        agent = _make_agent()
        agent._memory_enabled = False
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)

        task = _make_task(context={"github_repo": "owner/repo"})
        result = agent._build_replan_memory_context(task)

        assert result == ""

    def test_returns_empty_when_no_repo_slug(self):
        agent = _make_agent()
        agent._memory_enabled = True
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)

        task = _make_task(context={})
        result = agent._build_replan_memory_context(task)

        assert result == ""

    def test_returns_empty_when_no_memories(self):
        agent = _make_agent()
        agent._memory_enabled = True
        agent._memory_store = MagicMock()
        agent._memory_store.recall = MagicMock(return_value=[])
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)
        agent.config = MagicMock()
        agent.config.base_id = "engineer"

        task = _make_task(context={"github_repo": "owner/repo"})
        result = agent._build_replan_memory_context(task)

        assert result == ""

    def test_includes_prioritized_categories(self):
        from agent_framework.memory.memory_store import MemoryEntry

        agent = _make_agent()
        agent._memory_enabled = True
        agent._memory_store = MagicMock()
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)
        agent.config = MagicMock()
        agent.config.base_id = "engineer"

        # Mock memory store to return different memories for different categories
        def recall_side_effect(repo_slug, agent_type, category, limit):
            if category == "conventions":
                return [MemoryEntry(category="conventions", content="Use black for formatting")]
            elif category == "test_commands":
                return [MemoryEntry(category="test_commands", content="Run pytest tests/ -v")]
            elif category == "repo_structure":
                return [MemoryEntry(category="repo_structure", content="Tests in tests/unit/")]
            return []

        agent._memory_store.recall = MagicMock(side_effect=recall_side_effect)

        task = _make_task(context={"github_repo": "owner/repo"})
        result = agent._build_replan_memory_context(task)

        assert "Relevant Context from Previous Work" in result
        assert "You've worked on this repo before" in result
        assert "[conventions] Use black for formatting" in result
        assert "[test_commands] Run pytest tests/ -v" in result
        assert "[repo_structure] Tests in tests/unit/" in result

    def test_caps_at_ten_memories(self):
        from agent_framework.memory.memory_store import MemoryEntry

        agent = _make_agent()
        agent._memory_enabled = True
        agent._memory_store = MagicMock()
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)
        agent.config = MagicMock()
        agent.config.base_id = "engineer"

        # Return 15 memories total (5 per category)
        def recall_side_effect(repo_slug, agent_type, category, limit):
            return [MemoryEntry(category=category, content=f"memory {i}") for i in range(5)]

        agent._memory_store.recall = MagicMock(side_effect=recall_side_effect)

        task = _make_task(context={"github_repo": "owner/repo"})
        result = agent._build_replan_memory_context(task)

        # Should cap at 10 memories
        memory_lines = [line for line in result.split("\n") if line.startswith("- [")]
        assert len(memory_lines) == 10


class TestReplanWithMemoryIntegration:
    @pytest.mark.asyncio
    async def test_memory_context_injected_in_replan_prompt(self):
        from agent_framework.memory.memory_store import MemoryEntry

        agent = _make_agent()
        agent._memory_enabled = True
        agent._memory_store = MagicMock()
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)
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
        agent._build_replan_memory_context = Agent._build_replan_memory_context.__get__(agent)
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
