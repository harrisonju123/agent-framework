"""Tests for the AgenticMetrics session-log aggregator."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import AgenticMetrics, AgenticMetricsReport


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace with the standard directory layout."""
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "activity").mkdir(parents=True)
    return tmp_path


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    """Write a session JSONL file for the given task."""
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_iso(hours: int = 48) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


class TestMemoryMetrics:
    def test_no_recalls_returns_zeros(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.total_recalls == 0
        assert report.memory.recall_rate == 0.0

    def test_counts_memory_recalls(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 500},
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 300},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.total_recalls == 2
        assert report.memory.tasks_with_recall == 1
        assert report.memory.avg_chars_injected == 400.0
        assert report.memory.recall_rate == 1.0

    def test_recall_rate_partial(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 100},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.tasks_with_recall == 1
        assert report.memory.recall_rate == 0.5


class TestSelfEvalMetrics:
    def test_empty_returns_zeros(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.self_eval.total_evals == 0
        assert report.self_eval.catch_rate == 0.0

    def test_counts_verdicts(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "self_eval", "task_id": "t1", "verdict": "PASS"},
            {"ts": _now_iso(), "event": "self_eval", "task_id": "t1", "verdict": "FAIL"},
            {"ts": _now_iso(), "event": "self_eval", "task_id": "t1", "verdict": "AUTO_PASS"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        se = report.self_eval
        assert se.total_evals == 3
        assert se.pass_count == 1
        assert se.fail_count == 1
        assert se.auto_pass_count == 1
        # catch_rate = 1 fail / (1 pass + 1 fail) real evals = 0.5
        assert se.catch_rate == 0.5

    def test_all_pass_catch_rate_zero(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "self_eval", "task_id": "t1", "verdict": "PASS"},
            {"ts": _now_iso(), "event": "self_eval", "task_id": "t1", "verdict": "PASS"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.self_eval.catch_rate == 0.0


class TestReplanMetrics:
    def test_empty_returns_zeros(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.replan.total_replans == 0
        assert report.replan.trigger_rate == 0.0

    def test_replan_and_completion(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "replan", "task_id": "t1", "retry": 2},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1"},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        rp = report.replan
        assert rp.total_replans == 1
        assert rp.tasks_with_replan == 1
        assert rp.tasks_completed_after_replan == 1
        assert rp.trigger_rate == 0.5
        assert rp.success_rate_after_replan == 1.0

    def test_replan_without_completion(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "replan", "task_id": "t1", "retry": 2},
            {"ts": _now_iso(), "event": "task_failed", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.replan.tasks_completed_after_replan == 0
        assert report.replan.success_rate_after_replan == 0.0


class TestContextBudgetMetrics:
    def test_no_prompt_events(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.context_budget.sample_count == 0
        assert report.context_budget.avg_prompt_length == 0

    def test_aggregates_prompt_lengths(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 1000},
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t1", "prompt_length": 2000},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "prompt_built", "task_id": "t2", "prompt_length": 3000},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        cb = report.context_budget
        assert cb.sample_count == 3
        assert cb.min_prompt_length == 1000
        assert cb.max_prompt_length == 3000
        assert cb.avg_prompt_length == 2000


class TestTimeFiltering:
    def test_old_events_excluded(self, workspace):
        """Session files modified in the past 48h but with old event timestamps are excluded."""
        # Write a file with old timestamps — mtime will be current (tmp_path is fresh)
        # but event ts is beyond the 24h window.
        _write_session(workspace, "old", [
            {"ts": _old_iso(48), "event": "memory_recall", "task_id": "old", "chars_injected": 500},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.total_recalls == 0


class TestSpecializationDistribution:
    def test_reads_activity_files(self, workspace):
        activity_dir = workspace / ".agent-communication" / "activity"
        (activity_dir / "engineer.json").write_text(
            json.dumps({"agent_id": "engineer", "specialization": "backend", "status": "idle",
                        "last_updated": _now_iso()})
        )
        (activity_dir / "qa.json").write_text(
            json.dumps({"agent_id": "qa", "specialization": None, "status": "idle",
                        "last_updated": _now_iso()})
        )
        report = AgenticMetrics(workspace).generate_report(hours=24)
        spec = report.specialization
        assert spec.total_active_agents == 2
        assert spec.distribution.get("backend") == 1
        assert spec.distribution.get("none") == 1

    def test_empty_activity_dir(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.specialization.total_active_agents == 0
        assert report.specialization.distribution == {}


class TestReportShape:
    def test_report_is_well_formed(self, workspace):
        """Smoke test: report always returns a valid AgenticMetricsReport."""
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, AgenticMetricsReport)
        assert report.time_range_hours == 24
        assert report.total_observed_tasks == 0
        # debate is always stubbed
        assert not report.debate.available
