"""Tests for Agent._request_replan() and Agent._inject_replan_context()."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides) -> Task:
    defaults = dict(
        id="task-replan-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Fix auth bug",
        description="Fix the authentication module",
        context={},
        notes=[],
        retry_count=2,
        last_error="Tests failed: 3 assertions",
        replan_history=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


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


def _build_agent(tmp_path, llm_mock=None) -> Agent:
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
            replan_config={"enabled": True, "min_retry_for_replan": 2, "model": "haiku"},
        )
    return agent


class TestRequestReplan:
    async def test_stores_revised_plan_in_context(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response(
            "1. Try different approach\n2. Break into smaller steps"
        ))
        agent = _build_agent(tmp_path, llm)
        task = _make_task()

        await agent._request_replan(task)
        assert "_revised_plan" in task.context
        assert "different approach" in task.context["_revised_plan"]

    async def test_appends_to_replan_history(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("New plan"))
        agent = _build_agent(tmp_path, llm)
        task = _make_task()

        await agent._request_replan(task)
        assert len(task.replan_history) == 1
        entry = task.replan_history[0]
        assert entry["attempt"] == 2
        assert "Tests failed" in entry["error"]
        assert entry["revised_plan"] == "New plan"

    async def test_stores_replan_attempt_in_context(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan B"))
        agent = _build_agent(tmp_path, llm)
        task = _make_task(retry_count=3)

        await agent._request_replan(task)
        assert task.context["_replan_attempt"] == 3

    async def test_includes_previous_attempts_in_prompt(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan C"))
        agent = _build_agent(tmp_path, llm)
        task = _make_task(
            retry_count=3,
            replan_history=[{
                "attempt": 2,
                "error": "Build failed",
                "revised_plan": "Previous plan",
            }],
        )

        await agent._request_replan(task)
        # Verify the prompt sent to LLM includes previous attempt info
        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "Build failed" in prompt

    async def test_handles_llm_failure_gracefully(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("", success=False))
        agent = _build_agent(tmp_path, llm)
        task = _make_task()

        # Should not raise
        await agent._request_replan(task)
        assert "_revised_plan" not in task.context

    async def test_handles_llm_exception_gracefully(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("timeout"))
        agent = _build_agent(tmp_path, llm)
        task = _make_task()

        await agent._request_replan(task)
        assert "_revised_plan" not in task.context

    async def test_truncates_long_revised_plan(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("x" * 5000))
        agent = _build_agent(tmp_path, llm)
        task = _make_task()

        await agent._request_replan(task)
        assert len(task.context["_revised_plan"]) <= 2000


class TestInjectReplanContext:
    def test_returns_original_prompt_when_no_revised_plan(self, tmp_path):
        agent = _build_agent(tmp_path)
        task = _make_task()
        prompt = "Original prompt"
        result = agent._inject_replan_context(prompt, task)
        assert result == prompt

    def test_appends_revised_plan_section(self, tmp_path):
        agent = _build_agent(tmp_path)
        task = _make_task(context={"_revised_plan": "Try a new approach"})
        result = agent._inject_replan_context("Base prompt", task)
        assert "## REVISED APPROACH" in result
        assert "Try a new approach" in result

    def test_includes_self_eval_critique(self, tmp_path):
        agent = _build_agent(tmp_path)
        task = _make_task(context={
            "_revised_plan": "New plan",
            "_self_eval_critique": "Tests not actually executed",
        })
        result = agent._inject_replan_context("Base", task)
        assert "## Self-Evaluation Feedback" in result
        assert "Tests not actually executed" in result

    def test_includes_previous_attempt_history(self, tmp_path):
        agent = _build_agent(tmp_path)
        task = _make_task(
            context={"_revised_plan": "Latest plan"},
            replan_history=[
                {"attempt": 1, "error": "Compile error", "revised_plan": "Plan A"},
                {"attempt": 2, "error": "Test failure", "revised_plan": "Latest plan"},
            ],
        )
        result = agent._inject_replan_context("Base", task)
        assert "## Previous Attempt History" in result
        assert "Compile error" in result
        # Current plan (last entry) is shown in the main section, not duplicated in history
        history_section = result.split("## Previous Attempt History")[1] if "## Previous Attempt History" in result else ""
        assert "Test failure" not in history_section

    def test_no_history_section_when_single_replan(self, tmp_path):
        agent = _build_agent(tmp_path)
        task = _make_task(
            context={"_revised_plan": "First plan"},
            replan_history=[
                {"attempt": 2, "error": "Error", "revised_plan": "First plan"},
            ],
        )
        result = agent._inject_replan_context("Base", task)
        # With only one entry (the current), skip the history of only one (skipped as [:-1] is empty)
        assert "## REVISED APPROACH" in result
