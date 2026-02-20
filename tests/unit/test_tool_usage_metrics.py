"""Tests for ToolUsageMetrics aggregation in AgenticMetrics."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import (
    AgenticMetrics,
    ToolUsageMetrics,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace with the standard directory layout."""
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "activity").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "debates").mkdir(parents=True)
    return tmp_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_session(workspace: Path, task_id: str, events: list) -> None:
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


class TestNoToolUsageData:
    def test_empty_workspace_returns_zeros(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.total_tasks_analyzed == 0
        assert report.tool_usage.avg_tool_calls_per_task == 0.0
        assert report.tool_usage.max_tool_calls == 0
        assert report.tool_usage.tool_distribution == {}
        assert report.tool_usage.top_tasks_by_calls == {}


class TestSingleTaskAggregation:
    def test_single_task_with_tool_stats(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
            {
                "ts": _now_iso(),
                "event": "tool_usage_stats",
                "task_id": "t1",
                "total_calls": 25,
                "tool_distribution": {"Read": 12, "Grep": 5, "Edit": 4, "Bash": 4},
                "duplicate_reads": {"/config.py": 3},
                "read_before_write_ratio": 0.75,
                "edit_density": 0.16,
            },
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        tu = report.tool_usage
        assert tu.total_tasks_analyzed == 1
        assert tu.avg_tool_calls_per_task == 25.0
        assert tu.max_tool_calls == 25
        assert tu.tool_distribution == {"Read": 12, "Grep": 5, "Edit": 4, "Bash": 4}
        assert tu.duplicate_read_rate == 1.0
        assert tu.avg_read_before_write_ratio == 0.75
        assert tu.avg_edit_density == 0.16
        assert tu.top_tasks_by_calls == {"t1": 25}


class TestMultiTaskAggregation:
    def test_aggregates_across_tasks(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
            {
                "ts": _now_iso(),
                "event": "tool_usage_stats",
                "task_id": "t1",
                "total_calls": 40,
                "tool_distribution": {"Read": 20, "Edit": 10, "Bash": 10},
                "duplicate_reads": {"/a.py": 2},
                "read_before_write_ratio": 1.0,
                "edit_density": 0.25,
            },
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t2"},
            {
                "ts": _now_iso(),
                "event": "tool_usage_stats",
                "task_id": "t2",
                "total_calls": 10,
                "tool_distribution": {"Read": 5, "Edit": 5},
                "duplicate_reads": {},
                "read_before_write_ratio": 0.5,
                "edit_density": 0.5,
            },
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        tu = report.tool_usage
        assert tu.total_tasks_analyzed == 2
        assert tu.avg_tool_calls_per_task == 25.0
        assert tu.max_tool_calls == 40
        # Aggregate distribution
        assert tu.tool_distribution["Read"] == 25
        assert tu.tool_distribution["Edit"] == 15
        # Only t1 had dupes
        assert tu.duplicate_read_rate == 0.5
        # Average of 1.0 and 0.5
        assert tu.avg_read_before_write_ratio == 0.75
        # Average of 0.25 and 0.5
        assert tu.avg_edit_density == 0.375

    def test_top_tasks_limited_to_five(self, workspace):
        for i in range(8):
            _write_session(workspace, f"t{i}", [
                {"ts": _now_iso(), "event": "task_start", "task_id": f"t{i}"},
                {
                    "ts": _now_iso(),
                    "event": "tool_usage_stats",
                    "task_id": f"t{i}",
                    "total_calls": (i + 1) * 10,
                    "tool_distribution": {"Read": (i + 1) * 10},
                    "duplicate_reads": {},
                    "read_before_write_ratio": 0.0,
                    "edit_density": 0.0,
                },
            ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.tool_usage.top_tasks_by_calls) == 5
        # Highest count should be in top 5
        assert 80 in report.tool_usage.top_tasks_by_calls.values()


class TestTrendBucketIntegration:
    def test_trend_buckets_include_tool_fields(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
            {
                "ts": _now_iso(),
                "event": "tool_usage_stats",
                "task_id": "t1",
                "total_calls": 30,
                "tool_distribution": {"Read": 20, "Edit": 10},
                "edit_density": 0.333,
            },
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) >= 1
        bucket = report.trends[0]
        assert bucket.avg_tool_calls == 30.0
        assert bucket.avg_edit_density == 0.333

    def test_trend_buckets_default_zero_without_stats(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        if report.trends:
            bucket = report.trends[0]
            assert bucket.avg_tool_calls == 0.0
            assert bucket.avg_edit_density == 0.0


class TestZeroTotalCallsSkipped:
    def test_zero_total_calls_not_counted(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
            {
                "ts": _now_iso(),
                "event": "tool_usage_stats",
                "task_id": "t1",
                "total_calls": 0,
                "tool_distribution": {},
                "duplicate_reads": {},
                "read_before_write_ratio": 0.0,
                "edit_density": 0.0,
            },
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.total_tasks_analyzed == 0
