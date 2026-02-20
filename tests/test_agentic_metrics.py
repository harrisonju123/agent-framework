"""Tests for the AgenticMetrics session-log aggregator."""

import json
import os
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
    (tmp_path / ".agent-communication" / "debates").mkdir(parents=True)
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


def _write_debate(
    workspace: Path,
    debate_id: str,
    success: bool = True,
    confidence: str = "high",
    trade_offs: list[str] | None = None,
) -> Path:
    """Write a debate JSON file and return its path."""
    path = workspace / ".agent-communication" / "debates" / f"{debate_id}.json"
    data = {
        "topic": f"Test debate {debate_id}",
        "success": success,
        "synthesis": {
            "recommendation": "Use approach A",
            "confidence": confidence,
            "trade_offs": trade_offs or [],
            "reasoning": "Test reasoning",
        },
    }
    path.write_text(json.dumps(data))
    return path


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


class TestDebateMetrics:
    def test_no_debates_dir_returns_unavailable(self, tmp_path):
        """Missing debates directory → available=False."""
        (tmp_path / "logs" / "sessions").mkdir(parents=True)
        (tmp_path / ".agent-communication" / "activity").mkdir(parents=True)
        # No debates dir created
        report = AgenticMetrics(tmp_path).generate_report(hours=24)
        assert not report.debate.available

    def test_empty_debates_dir(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.debate.available
        assert report.debate.total_debates == 0
        assert report.debate.success_rate == 0.0

    def test_counts_and_confidence(self, workspace):
        _write_debate(workspace, "d1", success=True, confidence="high", trade_offs=["perf", "complexity"])
        _write_debate(workspace, "d2", success=True, confidence="medium", trade_offs=["readability"])
        _write_debate(workspace, "d3", success=False, confidence="low", trade_offs=[])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        d = report.debate
        assert d.total_debates == 3
        assert d.successful_debates == 2
        assert d.confidence_distribution == {"high": 1, "medium": 1, "low": 1}
        assert d.success_rate == pytest.approx(0.667, abs=0.001)
        assert d.avg_trade_offs_count == 1.0

    def test_old_debates_excluded_by_mtime(self, workspace):
        path = _write_debate(workspace, "old", success=True, confidence="high", trade_offs=["x"])
        old_time = (datetime.now(timezone.utc) - timedelta(hours=72)).timestamp()
        os.utime(path, (old_time, old_time))
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.debate.total_debates == 0


class TestMemoryUsefulness:
    def test_positive_delta(self, workspace):
        """Tasks with recall complete more often than tasks without."""
        # 2 tasks with recall: both complete
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 100},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1"},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t2", "chars_injected": 200},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t2"},
        ])
        # 2 tasks without recall: 1 completes, 1 fails
        _write_session(workspace, "t3", [
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t3"},
        ])
        _write_session(workspace, "t4", [
            {"ts": _now_iso(), "event": "task_failed", "task_id": "t4"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.completion_rate_with_recall == 1.0
        assert report.memory.completion_rate_without_recall == 0.5
        assert report.memory.recall_usefulness_delta == 0.5

    def test_zero_delta_both_complete(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 100},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1"},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.recall_usefulness_delta == 0.0

    def test_no_data_returns_zeros(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.memory.completion_rate_with_recall == 0.0
        assert report.memory.completion_rate_without_recall == 0.0
        assert report.memory.recall_usefulness_delta == 0.0


class TestTrends:
    def test_empty_returns_empty_list(self, workspace):
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.trends == []

    def test_events_bucketed_by_hour(self, workspace):
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)

        _write_session(workspace, "t1", [
            {"ts": now.isoformat(), "event": "memory_recall", "task_id": "t1", "chars_injected": 100},
            {"ts": now.isoformat(), "event": "prompt_built", "task_id": "t1", "prompt_length": 5000},
        ])
        _write_session(workspace, "t2", [
            {"ts": hour_ago.isoformat(), "event": "self_eval", "task_id": "t2", "verdict": "FAIL"},
            {"ts": hour_ago.isoformat(), "event": "self_eval", "task_id": "t2", "verdict": "PASS"},
            {"ts": hour_ago.isoformat(), "event": "prompt_built", "task_id": "t2", "prompt_length": 3000},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 2
        # Sorted by timestamp — older bucket first
        assert report.trends[0].timestamp < report.trends[1].timestamp
        # Older bucket: 1 task, self-eval catch rate = 1 fail / 2 real = 0.5
        assert report.trends[0].self_eval_catch_rate == 0.5
        assert report.trends[0].avg_prompt_length == 3000
        # Newer bucket: 1 task with memory recall
        assert report.trends[1].memory_recall_rate == 1.0
        assert report.trends[1].avg_prompt_length == 5000

    def test_rates_computed_per_bucket(self, workspace):
        now = datetime.now(timezone.utc)
        _write_session(workspace, "t1", [
            {"ts": now.isoformat(), "event": "replan", "task_id": "t1"},
        ])
        _write_session(workspace, "t2", [
            {"ts": now.isoformat(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 1
        assert report.trends[0].task_count == 2
        assert report.trends[0].replan_trigger_rate == 0.5


class TestCodebaseIndexMetrics:
    def test_no_injections_returns_zeros(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        ci = report.codebase_index
        assert ci.total_injections == 0
        assert ci.tasks_with_injection == 0
        assert ci.avg_chars_injected == 0.0
        assert ci.injection_rate == 0.0

    def test_counts_injections(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "codebase_index_injected", "task_id": "t1", "repo": "myrepo", "chars": 3000},
            {"ts": _now_iso(), "event": "codebase_index_injected", "task_id": "t1", "repo": "myrepo", "chars": 3600},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        ci = report.codebase_index
        assert ci.total_injections == 2
        assert ci.tasks_with_injection == 1
        assert ci.avg_chars_injected == 3300.0
        assert ci.injection_rate == 1.0

    def test_injection_rate_partial(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "codebase_index_injected", "task_id": "t1", "repo": "r", "chars": 500},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.codebase_index.injection_rate == 0.5

    def test_positive_usefulness_delta(self, workspace):
        """Tasks with index injection complete more often → positive delta."""
        # 2 tasks with index: both complete
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "codebase_index_injected", "task_id": "t1", "repo": "r", "chars": 3000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t1"},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "codebase_index_injected", "task_id": "t2", "repo": "r", "chars": 2000},
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t2"},
        ])
        # 2 tasks without index: 1 completes, 1 fails
        _write_session(workspace, "t3", [
            {"ts": _now_iso(), "event": "task_complete", "task_id": "t3"},
        ])
        _write_session(workspace, "t4", [
            {"ts": _now_iso(), "event": "task_failed", "task_id": "t4"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        ci = report.codebase_index
        assert ci.completion_rate_with_index == 1.0
        assert ci.completion_rate_without_index == 0.5
        assert ci.index_usefulness_delta == 0.5

    def test_trend_includes_index_rate(self, workspace):
        now = datetime.now(timezone.utc)
        _write_session(workspace, "t1", [
            {"ts": now.isoformat(), "event": "codebase_index_injected", "task_id": "t1", "repo": "r", "chars": 1000},
        ])
        _write_session(workspace, "t2", [
            {"ts": now.isoformat(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 1
        assert report.trends[0].codebase_index_rate == 0.5


class TestToolUsageMetrics:
    def test_p90_computed(self, workspace):
        """Sessions with varying tool counts → correct p90."""
        # 10 sessions with tool counts 10, 20, ..., 100
        for i in range(1, 11):
            _write_session(workspace, f"t{i}", [
                {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": f"t{i}",
                 "total_calls": i * 10, "tool_distribution": {"Read": i * 10},
                 "duplicate_reads": {}, "read_before_write_ratio": 1.0,
                 "edit_density": 0.5},
            ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        tu = report.tool_usage
        assert tu.total_tasks_analyzed == 10
        assert tu.p90_tool_calls >= 90  # P90 of [10,20,...,100] should be ~90-91

    def test_p90_single_task(self, workspace):
        """Single session → p90 equals the only value."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 42, "tool_distribution": {"Read": 42},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.p90_tool_calls == 42

    def test_sessions_exceeding_threshold(self, workspace):
        """Sessions with exploration_alert events → counted."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "exploration_alert", "task_id": "t1",
             "total_tool_calls": 55, "threshold": 50},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 55, "tool_distribution": {"Read": 55},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t2",
             "total_calls": 30, "tool_distribution": {"Read": 30},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.sessions_exceeding_threshold == 1

    def test_threshold_extracted_from_events(self, workspace):
        """exploration_alert_threshold reflects the runtime config, not hardcoded default."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "exploration_alert", "task_id": "t1",
             "total_tool_calls": 75, "threshold": 75},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 75, "tool_distribution": {"Read": 75},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.exploration_alert_threshold == 75

    def test_no_sessions_exceeding(self, workspace):
        """No exploration_alert events → count is 0."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 30, "tool_distribution": {"Read": 30},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.sessions_exceeding_threshold == 0

    def test_by_agent_breakdown(self, workspace):
        """Sessions with agent_id in tool_usage_stats → grouped correctly."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "agent_id": "architect", "total_calls": 60,
             "tool_distribution": {"Read": 60}, "duplicate_reads": {},
             "read_before_write_ratio": 1.0, "edit_density": 0.5},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t2",
             "agent_id": "architect", "total_calls": 40,
             "tool_distribution": {"Read": 40}, "duplicate_reads": {},
             "read_before_write_ratio": 1.0, "edit_density": 0.5},
        ])
        _write_session(workspace, "t3", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t3",
             "agent_id": "engineer", "total_calls": 80,
             "tool_distribution": {"Read": 80}, "duplicate_reads": {},
             "read_before_write_ratio": 1.0, "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        by_agent = report.tool_usage.by_agent
        assert by_agent["architect"] == 50.0  # avg(60, 40)
        assert by_agent["engineer"] == 80.0

    def test_by_agent_missing_id(self, workspace):
        """Sessions without agent_id are excluded from by_agent breakdown."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 30, "tool_distribution": {"Read": 30},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.by_agent == {}

    def test_trend_sessions_exceeding_threshold(self, workspace):
        """TrendBucket includes sessions_exceeding_threshold."""
        now = datetime.now(timezone.utc)
        _write_session(workspace, "t1", [
            {"ts": now.isoformat(), "event": "exploration_alert", "task_id": "t1",
             "total_tool_calls": 55, "threshold": 50},
            {"ts": now.isoformat(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 55, "tool_distribution": {"Read": 55},
             "duplicate_reads": {}},
        ])
        _write_session(workspace, "t2", [
            {"ts": now.isoformat(), "event": "task_start", "task_id": "t2"},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 1
        assert report.trends[0].sessions_exceeding_threshold == 1


class TestReportShape:
    def test_report_is_well_formed(self, workspace):
        """Smoke test: report always returns a valid AgenticMetricsReport."""
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, AgenticMetricsReport)
        assert report.time_range_hours == 24
        assert report.total_observed_tasks == 0
        # Debates dir exists in fixture, so available=True with 0 debates
        assert report.debate.available
        assert report.debate.total_debates == 0
        assert report.trends == []
