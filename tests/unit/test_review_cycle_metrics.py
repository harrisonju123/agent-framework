"""Tests for review cycle enforcement analytics."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_framework.analytics.review_cycle_metrics import (
    ReviewCycleAnalyzer,
    ReviewCycleMetrics,
    ReviewCycleMetricsReport,
)


def _write_session_event(sessions_dir: Path, task_id: str, event: dict) -> None:
    """Append a session event to a task's JSONL file."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{task_id}.jsonl"
    event.setdefault("task_id", task_id)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _make_review_event(
    task_id: str = "task-1",
    workflow_step: str = "qa_review",
    target_step: str = "engineer",
    count_before: int = 0,
    count_after: int = 1,
    max_cycles: int = 2,
    enforced: bool = False,
    phase_reset: bool = False,
    halted: bool = False,
) -> dict:
    return {
        "event": "review_cycle_check",
        "task_id": task_id,
        "workflow_step": workflow_step,
        "target_step": target_step,
        "count_before": count_before,
        "count_after": count_after,
        "max": max_cycles,
        "enforced": enforced,
        "phase_reset": phase_reset,
        **({"halted": halted} if halted else {}),
    }


class TestEmptyReport:
    """Empty report when no events exist."""

    def test_empty_report_all_zeros(self, tmp_path):
        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_checks == 0
        assert report.metrics.total_enforcements == 0
        assert report.metrics.total_phase_resets == 0
        assert report.metrics.total_halts == 0
        assert report.metrics.enforcement_rate == 0.0
        assert report.metrics.cap_violations == 0
        assert report.metrics.violation_task_ids == []
        assert report.metrics.by_step == []
        assert report.raw_events == []


class TestNormalIncrement:
    """Normal review cycle increments (no enforcement)."""

    def test_single_increment_aggregated(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        event = _make_review_event(count_before=0, count_after=1)
        _write_session_event(sessions_dir, "task-1", event)

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_checks == 1
        assert report.metrics.total_enforcements == 0
        assert report.metrics.enforcement_rate == 0.0

    def test_multiple_tasks_aggregated(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", _make_review_event(task_id="task-1"))
        _write_session_event(sessions_dir, "task-2", _make_review_event(task_id="task-2"))
        _write_session_event(sessions_dir, "task-3", _make_review_event(task_id="task-3"))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_checks == 3


class TestEnforcement:
    """Enforcement detection and rate calculation."""

    def test_enforcement_counted(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", _make_review_event(
            count_before=1, count_after=2, enforced=True, target_step="create_pr",
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_enforcements == 1
        assert report.metrics.enforcement_rate == 100.0

    def test_enforcement_rate_mixed(self, tmp_path):
        """1 enforcement out of 4 checks = 25% rate."""
        sessions_dir = tmp_path / "logs" / "sessions"
        for i in range(3):
            _write_session_event(sessions_dir, f"task-{i}", _make_review_event(
                task_id=f"task-{i}", count_before=0, count_after=1,
            ))
        _write_session_event(sessions_dir, "task-3", _make_review_event(
            task_id="task-3", count_before=1, count_after=2, enforced=True,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_checks == 4
        assert report.metrics.total_enforcements == 1
        assert report.metrics.enforcement_rate == 25.0

    def test_enforcement_count_distribution(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        # Two enforcements at count_after=2
        _write_session_event(sessions_dir, "task-1", _make_review_event(
            task_id="task-1", count_after=2, enforced=True,
        ))
        _write_session_event(sessions_dir, "task-2", _make_review_event(
            task_id="task-2", count_after=2, enforced=True,
        ))
        # One enforcement at count_after=3 (if max were raised)
        _write_session_event(sessions_dir, "task-3", _make_review_event(
            task_id="task-3", count_after=3, max_cycles=3, enforced=True,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.enforcement_count_distribution == {2: 2, 3: 1}


class TestCapViolation:
    """Cap violation detection — the key bug signal."""

    def test_violation_detected(self, tmp_path):
        """count_after >= max without enforcement or phase_reset = violation."""
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-bad", _make_review_event(
            task_id="task-bad",
            count_before=1, count_after=2,
            enforced=False, phase_reset=False,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.cap_violations == 1
        assert "task-bad" in report.metrics.violation_task_ids

    def test_enforcement_not_flagged_as_violation(self, tmp_path):
        """Enforced transition at cap is not a violation."""
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-ok", _make_review_event(
            task_id="task-ok",
            count_before=1, count_after=2,
            enforced=True,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.cap_violations == 0

    def test_halted_not_flagged_as_violation(self, tmp_path):
        """Halted transition at cap is not a violation."""
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-halted", _make_review_event(
            task_id="task-halted",
            count_before=1, count_after=2,
            enforced=False, halted=True,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.cap_violations == 0

    def test_deduplicated_violation_task_ids(self, tmp_path):
        """Same task violating twice only appears once in violation_task_ids."""
        sessions_dir = tmp_path / "logs" / "sessions"
        for _ in range(2):
            _write_session_event(sessions_dir, "task-repeat", _make_review_event(
                task_id="task-repeat", count_before=1, count_after=2,
            ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.cap_violations == 1
        assert report.metrics.violation_task_ids == ["task-repeat"]


class TestPhaseReset:
    """Phase reset events counted separately."""

    def test_phase_reset_counted(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", _make_review_event(
            workflow_step="preview_review",
            target_step="implement",
            count_before=1, count_after=0,
            phase_reset=True,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        assert report.metrics.total_phase_resets == 1
        assert report.metrics.total_enforcements == 0


class TestByStepBreakdown:
    """Per-step aggregation."""

    def test_by_step_groups_correctly(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", _make_review_event(
            task_id="task-1", workflow_step="qa_review",
        ))
        _write_session_event(sessions_dir, "task-2", _make_review_event(
            task_id="task-2", workflow_step="qa_review",
        ))
        _write_session_event(sessions_dir, "task-3", _make_review_event(
            task_id="task-3", workflow_step="code_review",
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        by_step = {s.workflow_step: s for s in report.metrics.by_step}
        assert by_step["qa_review"].checks == 2
        assert by_step["code_review"].checks == 1

    def test_enforcement_in_step_breakdown(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", _make_review_event(
            workflow_step="code_review", enforced=True, count_after=2,
        ))

        analyzer = ReviewCycleAnalyzer(tmp_path)
        report = analyzer.generate_report(hours=24)

        by_step = {s.workflow_step: s for s in report.metrics.by_step}
        assert by_step["code_review"].enforcements == 1
