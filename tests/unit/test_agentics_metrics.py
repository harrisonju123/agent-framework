"""Tests for DashboardDataProvider.compute_agentics_metrics()."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.web.data_provider import DashboardDataProvider
from agent_framework.web.models import AgenticsMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_stream(workspace: Path, events: list[dict]) -> None:
    stream = workspace / ".agent-communication" / "activity-stream.jsonl"
    stream.parent.mkdir(parents=True, exist_ok=True)
    stream.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )


def _event(
    task_id: str,
    event_type: str = "complete",
    agent: str = "engineer",
    retry_count: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    ts: str = "2026-02-17T12:00:00+00:00",
) -> dict:
    return {
        "type": event_type,
        "task_id": task_id,
        "agent": agent,
        "title": f"Task {task_id}",
        "timestamp": ts,
        "retry_count": retry_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _write_task_file(
    workspace: Path,
    task_id: str,
    replan_history: list | None = None,
    context: dict | None = None,
    status: str = "completed",
) -> None:
    queue_dir = workspace / ".agent-communication" / "queues" / "engineer"
    queue_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "id": task_id,
        "type": "implementation",
        "status": status,
        "priority": 1,
        "title": f"Task {task_id}",
        "description": "desc",
        "created_by": "architect",
        "assigned_to": "engineer",
        "created_at": "2026-02-17T10:00:00+00:00",
        "retry_count": 0,
        "replan_history": replan_history or [],
        "context": context or {},
        "subtask_ids": [],
        "depends_on": [],
        "blocks": [],
        "notes": [],
        "acceptance_criteria": [],
        "deliverables": [],
        "replan_history": replan_history or [],
    }
    (queue_dir / f"{task_id}.json").write_text(json.dumps(task))


@pytest.fixture
def workspace(tmp_path):
    # Minimal agents.yaml so DashboardDataProvider doesn't warn
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agents.yaml").write_text("agents: []\n")
    return tmp_path


@pytest.fixture
def provider(workspace):
    return DashboardDataProvider(workspace)


# ---------------------------------------------------------------------------
# Empty / no-data cases
# ---------------------------------------------------------------------------

class TestEmptyStream:
    def test_returns_zero_rates_when_no_stream(self, provider):
        metrics = provider.compute_agentics_metrics(hours=24)
        assert isinstance(metrics, AgenticsMetrics)
        assert metrics.total_tasks_in_window == 0
        assert metrics.memory_usage_rate == 0.0
        assert metrics.self_eval_retry_rate == 0.0
        assert metrics.replan_trigger_rate == 0.0
        assert metrics.replan_success_rate == 0.0
        assert metrics.avg_context_budget_utilization == 0.0
        assert metrics.high_budget_utilization_rate == 0.0
        assert metrics.specialization_distribution == []

    def test_returns_zero_rates_when_stream_empty(self, workspace, provider):
        _write_stream(workspace, [])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.total_tasks_in_window == 0

    def test_ignores_start_events(self, workspace, provider):
        _write_stream(workspace, [_event("t1", event_type="start")])
        metrics = provider.compute_agentics_metrics(hours=24)
        # "start" events don't count — only complete/fail
        assert metrics.total_tasks_in_window == 0


# ---------------------------------------------------------------------------
# Time-window filtering
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_old_events_excluded(self, workspace, provider):
        # Event far in the past — outside a 1-hour window
        _write_stream(workspace, [_event("t1", ts="2020-01-01T00:00:00+00:00")])
        metrics = provider.compute_agentics_metrics(hours=1)
        assert metrics.total_tasks_in_window == 0

    def test_recent_events_included(self, workspace, provider):
        _write_stream(workspace, [_event("t1")])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.total_tasks_in_window == 1


# ---------------------------------------------------------------------------
# Self-eval retry rate
# ---------------------------------------------------------------------------

class TestSelfEvalRetryRate:
    def test_no_retries(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", retry_count=0),
            _event("t2", retry_count=0),
        ])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.self_eval_retry_rate == 0.0
        assert metrics.self_eval_retry_count == 0

    def test_all_retried(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", retry_count=1),
            _event("t2", retry_count=2),
        ])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.self_eval_retry_rate == 100.0
        assert metrics.self_eval_retry_count == 2

    def test_half_retried(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", retry_count=0),
            _event("t2", retry_count=1),
        ])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.self_eval_retry_rate == 50.0
        assert metrics.self_eval_retry_count == 1


# ---------------------------------------------------------------------------
# Replan trigger and success rates
# ---------------------------------------------------------------------------

class TestReplanMetrics:
    def test_no_replans(self, workspace, provider):
        _write_stream(workspace, [_event("t1")])
        _write_task_file(workspace, "t1", replan_history=[])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.replan_trigger_rate == 0.0
        assert metrics.replan_trigger_count == 0

    def test_replanned_and_completed(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", event_type="complete"),
            _event("t2", event_type="complete"),
        ])
        _write_task_file(workspace, "t1", replan_history=[{"attempt": 1}])
        _write_task_file(workspace, "t2", replan_history=[])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.replan_trigger_count == 1
        assert metrics.replan_trigger_rate == 50.0
        assert metrics.replan_success_rate == 100.0  # the replanned task completed

    def test_replanned_but_failed(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", event_type="fail"),
        ])
        _write_task_file(workspace, "t1", replan_history=[{"attempt": 1}])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.replan_trigger_count == 1
        assert metrics.replan_success_rate == 0.0


# ---------------------------------------------------------------------------
# Memory usage rate
# ---------------------------------------------------------------------------

class TestMemoryUsageRate:
    def test_no_memory_context(self, workspace, provider):
        _write_stream(workspace, [_event("t1")])
        _write_task_file(workspace, "t1", context={})
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.memory_usage_rate == 0.0

    def test_revised_plan_signals_memory_used(self, workspace, provider):
        _write_stream(workspace, [_event("t1")])
        _write_task_file(workspace, "t1", context={"_revised_plan": "Try approach B"})
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.memory_usage_rate == 100.0

    def test_self_eval_critique_signals_memory_used(self, workspace, provider):
        _write_stream(workspace, [_event("t1"), _event("t2")])
        _write_task_file(workspace, "t1", context={"_self_eval_critique": "missed coverage"})
        _write_task_file(workspace, "t2", context={})
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.memory_usage_rate == 50.0


# ---------------------------------------------------------------------------
# Specialization distribution
# ---------------------------------------------------------------------------

class TestSpecializationDistribution:
    def test_no_specialization(self, workspace, provider):
        _write_stream(workspace, [_event("t1")])
        _write_task_file(workspace, "t1", context={})
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.specialization_distribution == []

    def test_hint_in_context(self, workspace, provider):
        _write_stream(workspace, [_event("t1"), _event("t2")])
        _write_task_file(workspace, "t1", context={"specialization_hint": "backend"})
        _write_task_file(workspace, "t2", context={"specialization_hint": "frontend"})
        metrics = provider.compute_agentics_metrics(hours=24)
        profiles = {s.profile: s.count for s in metrics.specialization_distribution}
        assert profiles["backend"] == 1
        assert profiles["frontend"] == 1

    def test_sorted_by_count_descending(self, workspace, provider):
        _write_stream(workspace, [_event(f"t{i}") for i in range(5)])
        for i in range(3):
            _write_task_file(workspace, f"t{i}", context={"specialization_hint": "backend"})
        for i in range(3, 5):
            _write_task_file(workspace, f"t{i}", context={"specialization_hint": "frontend"})
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.specialization_distribution[0].profile == "backend"
        assert metrics.specialization_distribution[0].count == 3


# ---------------------------------------------------------------------------
# Context budget utilisation
# ---------------------------------------------------------------------------

class TestContextBudgetUtilization:
    def test_zero_when_no_token_data(self, workspace, provider):
        _write_stream(workspace, [_event("t1", input_tokens=0, output_tokens=0)])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.avg_context_budget_utilization == 0.0

    def test_utilization_computed_from_tokens(self, workspace, provider):
        # 25000 tokens against 50000 default budget = 50%
        _write_stream(workspace, [_event("t1", input_tokens=20000, output_tokens=5000)])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.avg_context_budget_utilization == 50.0

    def test_capped_at_100_percent(self, workspace, provider):
        # Tokens exceed budget — should cap at 100
        _write_stream(workspace, [_event("t1", input_tokens=40000, output_tokens=40000)])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.avg_context_budget_utilization == 100.0

    def test_high_utilization_rate(self, workspace, provider):
        # t1: 42000 / 50000 = 84% (≥80 → high)
        # t2: 20000 / 50000 = 40% (< 80 → not high)
        _write_stream(workspace, [
            _event("t1", input_tokens=40000, output_tokens=2000),
            _event("t2", input_tokens=10000, output_tokens=10000),
        ])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.high_budget_utilization_rate == 50.0


# ---------------------------------------------------------------------------
# Fail events counted toward totals
# ---------------------------------------------------------------------------

class TestFailEventsIncluded:
    def test_fail_events_count_toward_total(self, workspace, provider):
        _write_stream(workspace, [
            _event("t1", event_type="complete"),
            _event("t2", event_type="fail"),
        ])
        metrics = provider.compute_agentics_metrics(hours=24)
        assert metrics.total_tasks_in_window == 2
