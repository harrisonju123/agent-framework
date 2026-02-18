"""Unit tests for AgenticsMetrics computation from session JSONL logs."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.web.data_provider import DashboardDataProvider
from agent_framework.web.models import AgenticsMetrics, SpecializationCount


def _make_workspace(tmpdir: str) -> Path:
    workspace = Path(tmpdir)
    (workspace / "config").mkdir(parents=True)
    (workspace / "config" / "agents.yaml").write_text(
        """
agents:
  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "test"
"""
    )
    return workspace


def _sessions_dir(workspace: Path) -> Path:
    d = workspace / "logs" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _recent_ts(offset_seconds: int = 0) -> str:
    """Return an ISO timestamp within the 24-hour window."""
    ts = datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    return ts.isoformat()


def _old_ts() -> str:
    """Return an ISO timestamp older than 24 hours."""
    return (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()


def _write_session(sessions_dir: Path, task_id: str, events: list) -> None:
    path = sessions_dir / f"{task_id}.jsonl"
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


class TestComputeAgenticsMetrics:
    def test_empty_sessions_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            result = provider.compute_agentics_metrics(hours=24)

            assert isinstance(result, AgenticsMetrics)
            assert result.memory_recall_rate == 0.0
            assert result.memory_recalls_total == 0
            assert result.self_eval_catch_rate == 0.0
            assert result.self_eval_total == 0
            assert result.replan_trigger_rate == 0.0
            assert result.replan_total == 0
            assert result.specialization_distribution == []
            assert result.window_hours == 24

    def test_no_sessions_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            # Deliberately skip creating the sessions directory
            provider = DashboardDataProvider(workspace)
            result = provider.compute_agentics_metrics(hours=24)

            assert result.memory_recall_rate == 0.0
            assert result.replan_total == 0

    def test_memory_recall_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            # Task A: has memory recall
            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "memory_recall", "task_id": "task-A", "chars_injected": 500},
            ])
            # Task B: no memory recall
            _write_session(sessions_dir, "task-B", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-B"},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            # 1 of 2 tasks had memory recall → rate = 0.5
            assert result.memory_recall_rate == 0.5
            assert result.memory_recalls_total == 1

    def test_memory_recall_deduplicated_per_task(self):
        """Multiple recall events in one task should only count that task once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "memory_recall", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "memory_recall", "task_id": "task-A"},  # second recall
            ])

            result = provider.compute_agentics_metrics(hours=24)

            assert result.memory_recalls_total == 1  # still 1 unique task
            assert result.memory_recall_rate == 1.0

    def test_self_eval_catch_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "self_eval", "task_id": "task-A", "verdict": "FAIL"},
                {"ts": _recent_ts(), "event": "self_eval", "task_id": "task-A", "verdict": "PASS"},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            # 1 FAIL out of 2 evaluations → 0.5
            assert result.self_eval_total == 2
            assert result.self_eval_catch_rate == 0.5

    def test_self_eval_auto_pass_not_counted(self):
        """AUTO_PASS events (no objective evidence) must not inflate the totals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "self_eval", "task_id": "task-A", "verdict": "AUTO_PASS"},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            assert result.self_eval_total == 0
            assert result.self_eval_catch_rate == 0.0

    def test_replan_trigger_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "replan", "task_id": "task-A", "retry": 2},
            ])
            _write_session(sessions_dir, "task-B", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-B"},
            ])
            _write_session(sessions_dir, "task-C", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-C"},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            # 1 of 3 tasks triggered a replan
            assert result.replan_total == 1
            assert abs(result.replan_trigger_rate - 1 / 3) < 0.001

    def test_old_events_excluded(self):
        """Events older than the window must not be counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-old", [
                {"ts": _old_ts(), "event": "task_start", "task_id": "task-old"},
                {"ts": _old_ts(), "event": "memory_recall", "task_id": "task-old"},
                {"ts": _old_ts(), "event": "replan", "task_id": "task-old", "retry": 2},
                {"ts": _old_ts(), "event": "self_eval", "task_id": "task-old", "verdict": "FAIL"},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            assert result.memory_recall_rate == 0.0
            assert result.replan_total == 0
            assert result.self_eval_total == 0

    def test_malformed_json_lines_skipped(self):
        """Corrupt lines must not crash the computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            session_path = sessions_dir / "task-X.jsonl"
            session_path.write_text(
                '{"ts": "'
                + _recent_ts()
                + '", "event": "task_start", "task_id": "task-X"}\n'
                + "NOT_VALID_JSON\n"
                + '{"ts": "'
                + _recent_ts()
                + '", "event": "memory_recall", "task_id": "task-X"}\n'
            )

            result = provider.compute_agentics_metrics(hours=24)

            # Despite the corrupt line, valid events are counted
            assert result.memory_recalls_total == 1

    def test_window_hours_param_respected(self):
        """Only events in the requested window are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            # Event is 3 hours old — in a 24h window but NOT in a 1h window
            three_hours_ago = (
                datetime.now(timezone.utc) - timedelta(hours=3)
            ).isoformat()
            _write_session(sessions_dir, "task-A", [
                {"ts": three_hours_ago, "event": "task_start", "task_id": "task-A"},
                {"ts": three_hours_ago, "event": "memory_recall", "task_id": "task-A"},
            ])

            result_24h = provider.compute_agentics_metrics(hours=24)
            result_1h = provider.compute_agentics_metrics(hours=1)

            assert result_24h.memory_recalls_total == 1
            assert result_1h.memory_recalls_total == 0

    def test_zero_division_handled_gracefully(self):
        """Rates should be 0.0 when denominators are 0, not raise ZeroDivisionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _sessions_dir(workspace)  # create empty sessions dir
            provider = DashboardDataProvider(workspace)

            result = provider.compute_agentics_metrics(hours=24)

            assert result.memory_recall_rate == 0.0
            assert result.self_eval_catch_rate == 0.0
            assert result.replan_trigger_rate == 0.0

    def test_multiple_replans_single_task_deduplicated(self):
        """Multiple replan events in one task count the task only once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            sessions_dir = _sessions_dir(workspace)
            provider = DashboardDataProvider(workspace)

            _write_session(sessions_dir, "task-A", [
                {"ts": _recent_ts(), "event": "task_start", "task_id": "task-A"},
                {"ts": _recent_ts(), "event": "replan", "task_id": "task-A", "retry": 2},
                {"ts": _recent_ts(), "event": "replan", "task_id": "task-A", "retry": 3},
            ])

            result = provider.compute_agentics_metrics(hours=24)

            assert result.replan_total == 1  # 1 unique task

    def test_computed_at_is_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            before = datetime.now(timezone.utc)
            result = provider.compute_agentics_metrics(hours=24)
            after = datetime.now(timezone.utc)

            assert before <= result.computed_at <= after

    def test_model_validation(self):
        """AgenticsMetrics Pydantic model accepts all expected fields."""
        m = AgenticsMetrics(
            memory_recall_rate=0.75,
            memory_recalls_total=3,
            self_eval_catch_rate=0.5,
            self_eval_total=10,
            replan_trigger_rate=0.25,
            replan_total=1,
            replan_success_rate=0.0,
            specialization_distribution=[
                SpecializationCount(profile="backend", count=5),
                SpecializationCount(profile="frontend", count=2),
            ],
            avg_context_budget_utilization=0.0,
            context_budget_samples=0,
            window_hours=24,
        )
        assert m.memory_recall_rate == 0.75
        assert len(m.specialization_distribution) == 2
        assert m.specialization_distribution[0].profile == "backend"
