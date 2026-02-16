"""Tests for BudgetManager."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from agent_framework.core.budget_manager import BudgetManager, MODEL_PRICING
from agent_framework.core.task import Task, TaskType, TaskStatus
from agent_framework.llm.base import LLMResponse


def _make_task(task_type: TaskType = TaskType.IMPLEMENTATION, **overrides):
    defaults = dict(
        id="test-task-001",
        type=task_type,
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


def _make_manager(optimization_config=None):
    if optimization_config is None:
        optimization_config = {}

    logger = MagicMock()
    session_logger = MagicMock()
    llm = MagicMock()
    workspace = Path("/tmp/test-workspace")
    activity_manager = MagicMock()

    manager = BudgetManager(
        optimization_config=optimization_config,
        logger=logger,
        session_logger=session_logger,
        llm=llm,
        workspace=workspace,
        activity_manager=activity_manager,
    )

    return manager


def _llm_response(
    input_tokens: int = 1000,
    output_tokens: int = 500,
    model: str = "sonnet",
    reported_cost: float = None,
):
    return LLMResponse(
        content="Task completed successfully",
        model_used=f"claude-{model}-3-5-20250101",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason="end_turn",
        latency_ms=200,
        success=True,
        reported_cost_usd=reported_cost,
    )


class TestGetTokenBudget:
    def test_returns_default_budget_for_implementation(self):
        manager = _make_manager()

        budget = manager.get_token_budget(TaskType.IMPLEMENTATION)

        assert budget == 50000

    def test_returns_default_budget_for_review(self):
        manager = _make_manager()

        budget = manager.get_token_budget(TaskType.REVIEW)

        assert budget == 25000

    def test_returns_default_budget_for_escalation(self):
        manager = _make_manager()

        budget = manager.get_token_budget(TaskType.ESCALATION)

        assert budget == 80000

    def test_respects_configured_budget(self):
        manager = _make_manager(
            optimization_config={"token_budgets": {"implementation": 100000}}
        )

        budget = manager.get_token_budget(TaskType.IMPLEMENTATION)

        assert budget == 100000

    def test_falls_back_to_default_when_not_configured(self):
        manager = _make_manager(
            optimization_config={"token_budgets": {}}
        )

        budget = manager.get_token_budget(TaskType.IMPLEMENTATION)

        assert budget == 50000


class TestEstimateCost:
    def test_uses_reported_cost_when_available(self):
        manager = _make_manager()
        response = _llm_response(reported_cost=0.05)

        cost = manager.estimate_cost(response)

        assert cost == 0.05

    def test_calculates_cost_for_haiku(self):
        manager = _make_manager()
        response = _llm_response(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="haiku",
        )

        cost = manager.estimate_cost(response)

        # 1M input * $0.25 + 1M output * $1.25 = $1.50
        expected = MODEL_PRICING["haiku"]["input"] + MODEL_PRICING["haiku"]["output"]
        assert cost == expected

    def test_calculates_cost_for_sonnet(self):
        manager = _make_manager()
        response = _llm_response(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="sonnet",
        )

        cost = manager.estimate_cost(response)

        # 1M input * $3.0 + 1M output * $15.0 = $18.0
        expected = MODEL_PRICING["sonnet"]["input"] + MODEL_PRICING["sonnet"]["output"]
        assert cost == expected

    def test_calculates_cost_for_opus(self):
        manager = _make_manager()
        response = _llm_response(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="opus",
        )

        cost = manager.estimate_cost(response)

        # 1M input * $15.0 + 1M output * $75.0 = $90.0
        expected = MODEL_PRICING["opus"]["input"] + MODEL_PRICING["opus"]["output"]
        assert cost == expected

    def test_defaults_to_sonnet_for_unknown_model(self):
        manager = _make_manager()
        response = _llm_response(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model",
        )

        cost = manager.estimate_cost(response)

        # Should use sonnet pricing as default
        expected = MODEL_PRICING["sonnet"]["input"] + MODEL_PRICING["sonnet"]["output"]
        assert cost == expected
        # Should log warning
        manager.logger.warning.assert_called_once()


class TestLogTaskCompletionMetrics:
    def test_logs_token_usage_and_cost(self):
        manager = _make_manager()
        task = _make_task()
        response = _llm_response(input_tokens=1000, output_tokens=500)
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should log token usage
        manager.logger.token_usage.assert_called_once()
        args = manager.logger.token_usage.call_args[0]
        assert args[0] == 1000  # input tokens
        assert args[1] == 500   # output tokens
        assert args[2] > 0      # cost

    def test_logs_task_completed(self):
        manager = _make_manager()
        task = _make_task()
        response = _llm_response()
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should log task completed
        manager.logger.task_completed.assert_called_once()

    def test_warns_when_budget_exceeded(self):
        manager = _make_manager(
            optimization_config={"enable_token_budget_warnings": True}
        )
        task = _make_task(type=TaskType.IMPLEMENTATION)
        # Implementation budget is 50000, set tokens to 70000 (over threshold)
        response = _llm_response(input_tokens=50000, output_tokens=20000)
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should log warning
        manager.logger.warning.assert_called_once()
        warning_msg = manager.logger.warning.call_args[0][0]
        assert "EXCEEDED TOKEN BUDGET" in warning_msg

    def test_does_not_warn_when_under_budget(self):
        manager = _make_manager(
            optimization_config={"enable_token_budget_warnings": True}
        )
        task = _make_task(type=TaskType.IMPLEMENTATION)
        # Implementation budget is 50000, set tokens to 30000 (under threshold)
        response = _llm_response(input_tokens=20000, output_tokens=10000)
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should not log warning
        manager.logger.warning.assert_not_called()

    def test_appends_activity_event(self):
        manager = _make_manager()
        task = _make_task()
        response = _llm_response()
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should append complete event
        assert manager.activity_manager.append_event.call_count >= 1
        # Get the last call (complete event)
        last_call = manager.activity_manager.append_event.call_args_list[-1]
        event = last_call[0][0]
        assert event.type == "complete"
        assert event.task_id == task.id

    def test_logs_session_complete(self):
        manager = _make_manager()
        task = _make_task()
        response = _llm_response()
        task_start_time = datetime.now(timezone.utc)

        manager.log_task_completion_metrics(task, response, task_start_time)

        # Should log session complete
        manager.session_logger.log.assert_called_once()
        args = manager.session_logger.log.call_args[0]
        assert args[0] == "task_complete"
