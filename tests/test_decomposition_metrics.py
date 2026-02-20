"""Tests for DecompositionMetrics session-log aggregator."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.decomposition_metrics import (
    DecompositionMetrics,
    DecompositionReport,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace with the standard directory layout."""
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "chain-state").mkdir(parents=True)
    return tmp_path


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_chain_state(workspace: Path, task_id: str, steps: list[dict]) -> None:
    path = workspace / ".agent-communication" / "chain-state" / f"{task_id}.json"
    data = {
        "root_task_id": task_id,
        "user_goal": "test",
        "workflow": "default",
        "steps": steps,
    }
    path.write_text(json.dumps(data))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_iso(hours: int = 48) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


class TestDecompositionRate:
    def test_empty_workspace_returns_zeros(self, workspace):
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.rate.tasks_evaluated == 0
        assert report.rate.tasks_decomposed == 0
        assert report.rate.decomposition_rate == 0.0

    def test_mixed_events_correct_rate(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "decomposition_evaluated", "task_id": "t1",
             "estimated_lines": 600, "file_count": 8, "requirements_count": 5,
             "should_decompose": True},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "decomposition_evaluated", "task_id": "t2",
             "estimated_lines": 100, "file_count": 2, "requirements_count": 1,
             "should_decompose": False},
        ])
        _write_session(workspace, "t3", [
            {"ts": _now_iso(), "event": "decomposition_evaluated", "task_id": "t3",
             "estimated_lines": 800, "file_count": 10, "requirements_count": 7,
             "should_decompose": True},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.rate.tasks_evaluated == 3
        assert report.rate.tasks_decomposed == 2
        assert report.rate.decomposition_rate == pytest.approx(0.667, abs=0.001)

    def test_deduplicates_by_task_id(self, workspace):
        """Multiple evaluations for the same task → only last counts."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "decomposition_evaluated", "task_id": "t1",
             "estimated_lines": 100, "file_count": 2, "requirements_count": 1,
             "should_decompose": False},
            {"ts": _now_iso(), "event": "decomposition_evaluated", "task_id": "t1",
             "estimated_lines": 600, "file_count": 8, "requirements_count": 5,
             "should_decompose": True},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.rate.tasks_evaluated == 1
        assert report.rate.tasks_decomposed == 1


class TestSubtaskDistribution:
    def test_empty_returns_zeros(self, workspace):
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.distribution.distribution == {}
        assert report.distribution.avg_subtask_count == 0.0

    def test_varying_subtask_counts(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 600, "subtask_count": 3,
             "subtask_ids": ["t1-s0", "t1-s1", "t1-s2"], "file_count": 6},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t2",
             "estimated_lines": 400, "subtask_count": 2,
             "subtask_ids": ["t2-s0", "t2-s1"], "file_count": 4},
        ])
        _write_session(workspace, "t3", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t3",
             "estimated_lines": 500, "subtask_count": 3,
             "subtask_ids": ["t3-s0", "t3-s1", "t3-s2"], "file_count": 5},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        dist = report.distribution
        assert dist.distribution == {2: 1, 3: 2}
        assert dist.avg_subtask_count == pytest.approx(2.7, abs=0.1)
        assert dist.min_subtask_count == 2
        assert dist.max_subtask_count == 3


class TestEstimationAccuracy:
    def test_no_chain_state_returns_zeros(self, workspace):
        """Decomposed tasks without chain state files → no samples."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 600, "subtask_count": 2,
             "subtask_ids": ["t1-s0", "t1-s1"], "file_count": 6},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.estimation.sample_count == 0
        assert report.estimation.samples == []

    def test_correct_ratio(self, workspace):
        """Chain state with lines_added → correct actual/estimated ratio."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 500, "subtask_count": 2,
             "subtask_ids": ["t1-s0", "t1-s1"], "file_count": 5},
        ])
        # Subtask 0: 200 lines in implement step
        _write_chain_state(workspace, "t1-s0", [
            {"step_id": "plan", "agent_id": "architect", "task_id": "t1-s0",
             "completed_at": _now_iso(), "summary": "planned", "lines_added": 0},
            {"step_id": "implement", "agent_id": "engineer", "task_id": "t1-s0",
             "completed_at": _now_iso(), "summary": "implemented", "lines_added": 200},
        ])
        # Subtask 1: 150 lines in implement step
        _write_chain_state(workspace, "t1-s1", [
            {"step_id": "implement", "agent_id": "engineer", "task_id": "t1-s1",
             "completed_at": _now_iso(), "summary": "implemented", "lines_added": 150},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        est = report.estimation
        assert est.sample_count == 1
        assert est.samples[0].estimated_lines == 500
        assert est.samples[0].actual_lines == 350
        assert est.samples[0].ratio == 0.7  # 350/500
        assert est.avg_ratio == 0.7

    def test_missing_chain_state_excluded(self, workspace):
        """Tasks with partial chain state (some subtasks missing) still counted if total > 0."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 400, "subtask_count": 2,
             "subtask_ids": ["t1-s0", "t1-s1"], "file_count": 4},
        ])
        # Only one subtask has chain state
        _write_chain_state(workspace, "t1-s0", [
            {"step_id": "implement", "agent_id": "engineer", "task_id": "t1-s0",
             "completed_at": _now_iso(), "summary": "done", "lines_added": 300},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.estimation.sample_count == 1
        assert report.estimation.samples[0].actual_lines == 300


class TestFanInMetrics:
    def test_no_decomposed_tasks(self, workspace):
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.fan_in.decomposed_tasks == 0
        assert report.fan_in.fan_in_success_rate == 0.0

    def test_partial_fan_in(self, workspace):
        """Two decomposed tasks, only one gets fan-in."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 600, "subtask_count": 3,
             "subtask_ids": ["t1-s0", "t1-s1", "t1-s2"], "file_count": 6},
            {"ts": _now_iso(), "event": "fan_in_created",
             "parent_task_id": "t1", "fan_in_task_id": "fan-t1",
             "subtask_count": 3, "completed_subtask_count": 3},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t2",
             "estimated_lines": 400, "subtask_count": 2,
             "subtask_ids": ["t2-s0", "t2-s1"], "file_count": 4},
            # No fan_in_created for t2
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        fi = report.fan_in
        assert fi.decomposed_tasks == 2
        assert fi.fan_ins_created == 1
        assert fi.fan_in_success_rate == 0.5

    def test_all_fan_in_created(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_decomposed", "task_id": "t1",
             "estimated_lines": 500, "subtask_count": 2,
             "subtask_ids": ["t1-s0", "t1-s1"], "file_count": 5},
            {"ts": _now_iso(), "event": "fan_in_created",
             "parent_task_id": "t1", "fan_in_task_id": "fan-t1",
             "subtask_count": 2, "completed_subtask_count": 2},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.fan_in.fan_in_success_rate == 1.0


class TestTimeFiltering:
    def test_old_events_excluded(self, workspace):
        """Events beyond the time window are excluded."""
        _write_session(workspace, "old", [
            {"ts": _old_iso(48), "event": "decomposition_evaluated", "task_id": "old",
             "estimated_lines": 600, "file_count": 8, "requirements_count": 5,
             "should_decompose": True},
        ])
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert report.rate.tasks_evaluated == 0


class TestReportShape:
    def test_report_is_well_formed(self, workspace):
        report = DecompositionMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, DecompositionReport)
        assert report.time_range_hours == 24
        assert report.rate.tasks_evaluated == 0
        assert report.distribution.distribution == {}
        assert report.estimation.sample_count == 0
        assert report.fan_in.decomposed_tasks == 0
