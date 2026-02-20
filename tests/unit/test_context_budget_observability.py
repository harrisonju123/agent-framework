"""Tests for context budget observability: event fields, propagation, classification, analytics."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_framework.core.activity import ActivityEvent
from agent_framework.core.budget_manager import BudgetManager
from agent_framework.core.task import Task, TaskType, TaskStatus
from agent_framework.llm.base import LLMResponse
from agent_framework.safeguards.escalation import EscalationHandler
from agent_framework.analytics.agentic_metrics import AgenticMetrics


# --- helpers ---


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
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_budget_manager():
    manager = BudgetManager(
        agent_id="engineer",
        optimization_config={},
        logger=MagicMock(),
        session_logger=MagicMock(),
        llm=MagicMock(),
        workspace=Path("/tmp/test-workspace"),
        activity_manager=MagicMock(),
    )
    return manager


def _llm_response(**overrides):
    defaults = dict(
        content="Task completed successfully",
        model_used="claude-sonnet-3-5-20250101",
        input_tokens=1000,
        output_tokens=500,
        finish_reason="end_turn",
        latency_ms=200,
        success=True,
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _write_session(workspace: Path, task_id: str, events: list[dict]):
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "activity").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "debates").mkdir(parents=True)
    return tmp_path


# --- ActivityEvent field tests ---


class TestActivityEventContextFields:
    def test_accepts_context_utilization_fields(self):
        event = ActivityEvent(
            type="complete",
            agent="engineer",
            task_id="t1",
            title="Test task",
            timestamp=datetime.now(timezone.utc),
            context_utilization_percent=82.5,
            context_budget_tokens=200000,
        )
        assert event.context_utilization_percent == 82.5
        assert event.context_budget_tokens == 200000

    def test_defaults_to_none(self):
        event = ActivityEvent(
            type="complete",
            agent="engineer",
            task_id="t1",
            title="Test task",
            timestamp=datetime.now(timezone.utc),
        )
        assert event.context_utilization_percent is None
        assert event.context_budget_tokens is None

    def test_fields_survive_json_roundtrip(self):
        event = ActivityEvent(
            type="fail",
            agent="engineer",
            task_id="t1",
            title="Test task",
            timestamp=datetime.now(timezone.utc),
            context_utilization_percent=95.3,
            context_budget_tokens=200000,
        )
        data = json.loads(event.model_dump_json())
        restored = ActivityEvent(**data)
        assert restored.context_utilization_percent == 95.3
        assert restored.context_budget_tokens == 200000


# --- BudgetManager propagation tests ---


class TestBudgetManagerContextPropagation:
    def test_complete_event_includes_context_utilization(self):
        manager = _make_budget_manager()
        task = _make_task()
        response = _llm_response()
        budget_status = {
            "utilization_percent": 72.5,
            "total_budget": 200000,
            "used_so_far": 145000,
        }

        manager.log_task_completion_metrics(
            task, response, datetime.now(timezone.utc),
            context_budget_status=budget_status,
        )

        event = manager.activity_manager.append_event.call_args_list[-1][0][0]
        assert event.type == "complete"
        assert event.context_utilization_percent == 72.5
        assert event.context_budget_tokens == 200000

    def test_session_log_includes_context_utilization(self):
        manager = _make_budget_manager()
        task = _make_task()
        response = _llm_response()
        budget_status = {
            "utilization_percent": 55.0,
            "total_budget": 200000,
            "used_so_far": 110000,
        }

        manager.log_task_completion_metrics(
            task, response, datetime.now(timezone.utc),
            context_budget_status=budget_status,
        )

        call_kwargs = manager.session_logger.log.call_args[1]
        assert call_kwargs["context_utilization_percent"] == 55.0
        assert call_kwargs["context_budget_tokens"] == 200000
        assert call_kwargs["context_used_tokens"] == 110000

    def test_none_budget_status_produces_none_fields(self):
        manager = _make_budget_manager()
        task = _make_task()
        response = _llm_response()

        manager.log_task_completion_metrics(
            task, response, datetime.now(timezone.utc),
            context_budget_status=None,
        )

        event = manager.activity_manager.append_event.call_args_list[-1][0][0]
        assert event.context_utilization_percent is None
        assert event.context_budget_tokens is None

    def test_backward_compatible_without_budget_status(self):
        """Calling without the new param still works (keyword-only with default)."""
        manager = _make_budget_manager()
        task = _make_task()
        response = _llm_response()

        manager.log_task_completion_metrics(
            task, response, datetime.now(timezone.utc),
        )

        event = manager.activity_manager.append_event.call_args_list[-1][0][0]
        assert event.context_utilization_percent is None


# --- Escalation error categorization ---


class TestContextExhaustionCategorization:
    def test_categorizes_context_window_exhaustion(self):
        handler = EscalationHandler()
        result = handler.categorize_error(
            "No code changes detected — likely context window exhaustion. Retrying with a fresh context."
        )
        assert result == "context_exhaustion"

    def test_categorizes_context_exhaust_variant(self):
        handler = EscalationHandler()
        assert handler.categorize_error("context was exhausted during processing") == "context_exhaustion"

    def test_does_not_match_unrelated_context_errors(self):
        handler = EscalationHandler()
        # "context" alone shouldn't match — needs "exhaust" nearby
        assert handler.categorize_error("invalid context parameter") != "context_exhaustion"


# --- Analytics aggregation tests ---


class TestContextBudgetAnalytics:
    def test_aggregates_utilization_from_task_complete(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1",
             "context_utilization_percent": 45.0},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t2", "prompt_length": 8000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t2",
             "context_utilization_percent": 85.0},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.tasks_with_utilization == 2
        assert cb.avg_utilization_at_completion == 65.0  # (45+85)/2
        assert cb.near_limit_count == 1  # only 85% >= 80%

    def test_counts_critical_events(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
            {"ts": _now_iso(), "event": "context_budget_critical", "task_id": "t1",
             "utilization_percent": 92.0},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.context_budget.critical_count == 1

    def test_counts_exhaustion_events(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
            {"ts": _now_iso(), "event": "context_exhaustion", "task_id": "t1",
             "utilization_percent": 98.0},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.context_budget.exhaustion_count == 1

    def test_no_utilization_data_returns_zero_defaults(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.tasks_with_utilization == 0
        assert cb.avg_utilization_at_completion == 0.0
        assert cb.near_limit_count == 0

    def test_percentile_calculation_with_multiple_tasks(self, workspace):
        # 10 tasks with utilization from 10% to 100%
        for i in range(1, 11):
            util = float(i * 10)
            _write_session(workspace, f"t{i}", [
                {"ts": _now_iso(), "event": "prompt_built", "task_id": f"t{i}", "prompt_length": 5000},
                {"ts": _now_iso(), "event": "task_complete", "task_id": f"t{i}",
                 "context_utilization_percent": util},
            ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.tasks_with_utilization == 10
        assert cb.p50_utilization == 55.0  # median of 10..100
        assert cb.p90_utilization == 99.0  # quantiles(n=10)[8] on 10..100
        # 80% and 90% and 100% are >= 80
        assert cb.near_limit_count == 3

    def test_single_utilization_value(self, workspace):
        """Single sample: p90 falls back to that value."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1",
             "context_utilization_percent": 42.0},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.tasks_with_utilization == 1
        assert cb.p50_utilization == 42.0
        assert cb.p90_utilization == 42.0

    def test_empty_sessions_returns_all_zeros(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.sample_count == 0
        assert cb.tasks_with_utilization == 0
        assert cb.exhaustion_count == 0
        assert cb.critical_count == 0
