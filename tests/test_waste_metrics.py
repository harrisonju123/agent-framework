"""Tests for the WasteMetrics token waste ratio collector."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.waste_metrics import (
    WasteMetrics,
    WasteMetricsReport,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace with standard directory layout."""
    (tmp_path / ".agent-communication").mkdir(parents=True)
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    return tmp_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_iso(hours: int = 48) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _write_activity(workspace: Path, events: list[dict]) -> None:
    """Write events to the activity stream JSONL."""
    path = workspace / ".agent-communication" / "activity-stream.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    """Write a session JSONL file for the given task."""
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _start_event(
    task_id: str,
    root_task_id: str | None = None,
    title: str = "Test task",
    ts: str | None = None,
) -> dict:
    event = {
        "type": "start",
        "task_id": task_id,
        "title": title,
        "timestamp": ts or _now_iso(),
    }
    if root_task_id is not None:
        event["root_task_id"] = root_task_id
    return event


def _complete_event(
    task_id: str,
    root_task_id: str | None = None,
    cost: float = 0.0,
    pr_url: str | None = None,
    ts: str | None = None,
) -> dict:
    event = {
        "type": "complete",
        "task_id": task_id,
        "cost": cost,
        "timestamp": ts or _now_iso(),
    }
    if root_task_id is not None:
        event["root_task_id"] = root_task_id
    if pr_url is not None:
        event["pr_url"] = pr_url
    return event


def _fail_event(
    task_id: str,
    root_task_id: str | None = None,
    ts: str | None = None,
) -> dict:
    event = {
        "type": "fail",
        "task_id": task_id,
        "timestamp": ts or _now_iso(),
    }
    if root_task_id is not None:
        event["root_task_id"] = root_task_id
    return event


def _llm_session_event(task_id: str, cost: float, ts: str | None = None) -> dict:
    return {
        "ts": ts or _now_iso(),
        "event": "llm_complete",
        "task_id": task_id,
        "cost": cost,
        "tokens_in": 1000,
        "tokens_out": 500,
        "duration_ms": 2000,
    }


class TestEmptyReport:
    def test_no_activity_stream(self, workspace):
        """No activity stream file → valid report with zeros."""
        report = WasteMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, WasteMetricsReport)
        assert report.roots_analyzed == 0
        assert report.total_cost == 0.0
        assert report.total_wasted_cost == 0.0
        assert report.aggregate_waste_ratio == 0.0
        assert report.avg_waste_ratio == 0.0
        assert report.max_waste_ratio == 0.0
        assert report.roots_with_zero_delivery == 0
        assert report.top_waste_roots == []

    def test_no_sessions_dir(self, tmp_path):
        """Missing directories entirely → valid report with zeros."""
        report = WasteMetrics(tmp_path).generate_report(hours=24)
        assert report.roots_analyzed == 0


class TestRootWithPRNoFailures:
    def test_waste_ratio_zero(self, workspace):
        """Root with PR and no failures → waste_ratio = 0.0."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Add feature"),
            _complete_event("t1", root_task_id="root1", cost=5.0, pr_url="https://github.com/pr/1"),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        assert report.roots_analyzed == 1
        root = report.top_waste_roots[0]
        assert root.root_task_id == "root1"
        assert root.waste_ratio == 0.0
        assert root.total_cost == 5.0
        assert root.wasted_cost == 0.0
        assert root.productive_cost == 5.0
        assert root.has_pr is True
        assert root.completed_tasks == 1
        assert root.failed_tasks == 0
        assert root.title == "Add feature"


class TestRootWithNoPR:
    def test_waste_ratio_one(self, workspace):
        """Root with no PR → waste_ratio = 1.0, all cost wasted."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Failed task"),
            _complete_event("t1", root_task_id="root1", cost=10.0),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        assert report.roots_analyzed == 1
        root = report.top_waste_roots[0]
        assert root.waste_ratio == 1.0
        assert root.wasted_cost == 10.0
        assert root.productive_cost == 0.0
        assert root.has_pr is False
        assert report.roots_with_zero_delivery == 1


class TestRootWithPRAndFailedTasks:
    def test_partial_waste(self, workspace):
        """Root with PR + failed tasks → waste = failed_cost / total_cost."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Implement X"),
            _fail_event("t1", root_task_id="root1"),
            _start_event("t2", root_task_id="root1"),
            _complete_event("t2", root_task_id="root1", cost=7.0, pr_url="https://github.com/pr/2"),
        ])
        # Failed task cost comes from session logs
        _write_session(workspace, "t1", [
            _llm_session_event("t1", cost=3.0),
        ])

        report = WasteMetrics(workspace).generate_report(hours=24)

        root = report.top_waste_roots[0]
        assert root.total_cost == 10.0
        assert root.wasted_cost == 3.0
        assert root.waste_ratio == 0.3
        assert root.productive_cost == 7.0
        assert root.has_pr is True
        assert root.failed_tasks == 1
        assert root.completed_tasks == 1


class TestMultipleRoots:
    def test_aggregate_metrics(self, workspace):
        """Multiple roots → correct aggregate waste metrics."""
        _write_activity(workspace, [
            # Root 1: has PR, no failures
            _start_event("t1", root_task_id="root1", title="Good task"),
            _complete_event("t1", root_task_id="root1", cost=4.0, pr_url="https://github.com/pr/1"),
            # Root 2: no PR, all waste
            _start_event("t2", root_task_id="root2", title="Bad task"),
            _complete_event("t2", root_task_id="root2", cost=6.0),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        assert report.roots_analyzed == 2
        assert report.total_cost == 10.0
        assert report.total_wasted_cost == 6.0
        assert report.aggregate_waste_ratio == 0.6
        # avg of 0.0 and 1.0
        assert report.avg_waste_ratio == 0.5
        assert report.max_waste_ratio == 1.0
        assert report.roots_with_zero_delivery == 1


class TestFailEventCrossReferencesSessionLogs:
    def test_cost_recovered_from_session(self, workspace):
        """Fail events with no cost → cost recovered from session llm_complete events."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Failing task"),
            _fail_event("t1", root_task_id="root1"),
        ])
        _write_session(workspace, "t1", [
            _llm_session_event("t1", cost=2.5),
            _llm_session_event("t1", cost=1.5),
        ])

        report = WasteMetrics(workspace).generate_report(hours=24)

        root = report.top_waste_roots[0]
        assert root.total_cost == 4.0
        assert root.wasted_cost == 4.0
        assert root.waste_ratio == 1.0


class TestTimeFiltering:
    def test_old_events_excluded(self, workspace):
        """Events older than the lookback window are excluded."""
        old_ts = _old_iso(hours=48)
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", ts=old_ts),
            _complete_event("t1", root_task_id="root1", cost=10.0, ts=old_ts),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)
        assert report.roots_analyzed == 0
        assert report.total_cost == 0.0


class TestStandaloneTask:
    def test_task_id_used_as_root(self, workspace):
        """Task without root_task_id → task_id used as root."""
        _write_activity(workspace, [
            _start_event("standalone1", title="Solo task"),
            _complete_event("standalone1", cost=3.0, pr_url="https://github.com/pr/5"),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        assert report.roots_analyzed == 1
        root = report.top_waste_roots[0]
        assert root.root_task_id == "standalone1"
        assert root.has_pr is True
        assert root.waste_ratio == 0.0


class TestTopWasteRootsCapped:
    def test_sorted_desc_capped_at_10(self, workspace):
        """Top waste roots sorted by wasted_cost desc, capped at 10."""
        events = []
        for i in range(15):
            root_id = f"root{i}"
            events.append(_start_event(f"t{i}", root_task_id=root_id, title=f"Task {i}"))
            # No PR → all cost is waste
            events.append(_complete_event(f"t{i}", root_task_id=root_id, cost=float(i + 1)))

        _write_activity(workspace, events)
        report = WasteMetrics(workspace).generate_report(hours=24)

        assert len(report.top_waste_roots) == 10
        # Should be sorted by wasted_cost descending
        wasted_costs = [r.wasted_cost for r in report.top_waste_roots]
        assert wasted_costs == sorted(wasted_costs, reverse=True)
        # Highest cost root should be first (root14, cost=15.0)
        assert report.top_waste_roots[0].wasted_cost == 15.0


class TestZeroCostRoot:
    def test_zero_cost_zero_ratio(self, workspace):
        """Root with zero total cost → waste_ratio = 0.0 (no division error)."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Zero cost"),
            _complete_event("t1", root_task_id="root1", cost=0.0),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        root = report.top_waste_roots[0]
        assert root.waste_ratio == 0.0
        assert root.total_cost == 0.0


class TestRetryPattern:
    def test_fail_then_succeed_same_task(self, workspace):
        """Same task_id fails then completes → counted as completed, not failed."""
        _write_activity(workspace, [
            _start_event("t1", root_task_id="root1", title="Retry task"),
            _fail_event("t1", root_task_id="root1"),
            _complete_event("t1", root_task_id="root1", cost=5.0, pr_url="https://github.com/pr/1"),
        ])
        report = WasteMetrics(workspace).generate_report(hours=24)

        root = report.top_waste_roots[0]
        assert root.completed_tasks == 1
        assert root.failed_tasks == 0
        assert root.has_pr is True
        assert root.waste_ratio == 0.0


class TestCorruptStreamLines:
    def test_corrupt_lines_skipped(self, workspace):
        """Malformed JSONL lines are silently skipped."""
        path = workspace / ".agent-communication" / "activity-stream.jsonl"
        lines = [
            "not json at all",
            json.dumps(_start_event("t1", root_task_id="r1")),
            '{"broken": true',
            json.dumps(_complete_event("t1", root_task_id="r1", cost=1.0)),
        ]
        path.write_text("\n".join(lines) + "\n")

        report = WasteMetrics(workspace).generate_report(hours=24)
        assert report.roots_analyzed == 1
        assert report.total_cost == 1.0
