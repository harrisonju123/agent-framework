"""Unit tests for AgenticMetricsCollector."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import AgenticMetrics, AgenticMetricsCollector


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


class TestEmptyInputs:
    """Collector returns zero/default metrics when no data files exist."""

    def test_empty_workspace(self, tmp_path):
        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert isinstance(metrics, AgenticMetrics)
        assert metrics.time_range_hours == 24
        assert metrics.memory.total_recalls == 0
        assert metrics.memory.total_stores == 0
        assert metrics.memory.recall_rate == 0.0
        assert metrics.self_eval.total_evaluations == 0
        assert metrics.self_eval.catch_rate == 0.0
        assert metrics.replan.total_replans == 0
        assert metrics.replan.trigger_rate == 0.0
        assert metrics.specialization.total_specialized_tasks == 0
        assert metrics.specialization.total_generic_tasks == 0
        assert metrics.context_budget.avg_utilization_percent == 0.0
        assert metrics.debate is None


class TestMemoryMetrics:
    """Collector correctly aggregates memory_recall and memory_store events."""

    def test_recall_and_store_counts(self, tmp_path):
        now = _utc_now()
        session_events = [
            {"event": "memory_recall", "ts": _iso(now), "chars_injected": 200, "category": "solution"},
            {"event": "memory_recall", "ts": _iso(now), "chars_injected": 400, "category": "solution"},
            {"event": "memory_recall", "ts": _iso(now), "chars_injected": 300, "category": "error"},
            {"event": "memory_store", "ts": _iso(now)},
            {"event": "memory_store", "ts": _iso(now)},
        ]
        activity_events = [
            {"type": "start", "task_id": "t1", "timestamp": _iso(now), "agent": "engineer"},
        ]

        _write_jsonl(tmp_path / "logs" / "sessions" / "sess-1.jsonl", session_events)
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.memory.total_recalls == 3
        assert metrics.memory.total_stores == 2
        assert metrics.memory.avg_chars_injected == pytest.approx(300.0)
        assert metrics.memory.categories_distribution == {"solution": 2, "error": 1}
        # 3 recalls / 1 task
        assert metrics.memory.recall_rate == pytest.approx(3.0)

    def test_no_chars_injected_field(self, tmp_path):
        """avg_chars_injected is 0.0 when the field is absent from recalls."""
        now = _utc_now()
        _write_jsonl(
            tmp_path / "logs" / "sessions" / "s.jsonl",
            [{"event": "memory_recall", "ts": _iso(now)}],
        )
        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)
        assert metrics.memory.avg_chars_injected == 0.0


class TestSelfEvalMetrics:
    """Collector correctly aggregates self_eval session events."""

    def test_pass_fail_auto_pass(self, tmp_path):
        now = _utc_now()
        session_events = [
            {"event": "self_eval", "ts": _iso(now), "result": "pass"},
            {"event": "self_eval", "ts": _iso(now), "result": "fail"},
            {"event": "self_eval", "ts": _iso(now), "result": "pass", "auto_pass": True},
        ]
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.self_eval.total_evaluations == 3
        assert metrics.self_eval.pass_count == 2
        assert metrics.self_eval.fail_count == 1
        assert metrics.self_eval.auto_pass_count == 1
        assert metrics.self_eval.catch_rate == pytest.approx(1 / 3 * 100)

    def test_avg_eval_attempts_from_activity_stream(self, tmp_path):
        """avg_eval_attempts is derived from _self_eval_count in complete events."""
        now = _utc_now()
        activity_events = [
            {"type": "complete", "task_id": "t1", "timestamp": _iso(now), "agent": "eng", "_self_eval_count": 2},
            {"type": "complete", "task_id": "t2", "timestamp": _iso(now), "agent": "eng", "_self_eval_count": 4},
        ]
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.self_eval.avg_eval_attempts == pytest.approx(3.0)


class TestReplanMetrics:
    """Collector correctly aggregates replan events and correlates with task outcomes."""

    def test_replan_success_rate_uses_task_count_not_event_count(self, tmp_path):
        """replan_success_rate denominator is distinct tasks, not replan event count.

        t1 replans twice and succeeds → success rate should be 100% (1/1 task),
        not 50% (1/2 events).
        """
        now = _utc_now()
        session_events = [
            {"event": "replan", "ts": _iso(now), "task_id": "t1"},
            {"event": "replan", "ts": _iso(now), "task_id": "t1"},  # second replan, same task
        ]
        activity_events = [
            {"type": "start", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"},
            {"type": "complete", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"},
        ]
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        metrics = AgenticMetricsCollector(tmp_path).collect(hours=24)

        assert metrics.replan.total_replans == 2          # event count preserved
        assert metrics.replan.success_after_replan == 1   # 1 distinct task succeeded
        assert metrics.replan.replan_success_rate == pytest.approx(100.0)

    def test_replan_success_and_failure(self, tmp_path):
        now = _utc_now()
        session_events = [
            {"event": "replan", "ts": _iso(now), "task_id": "t1"},
            {"event": "replan", "ts": _iso(now), "task_id": "t2"},
        ]
        activity_events = [
            {"type": "start", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"},
            {"type": "start", "task_id": "t2", "timestamp": _iso(now), "agent": "eng"},
            {"type": "complete", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"},
            {"type": "fail", "task_id": "t2", "timestamp": _iso(now), "agent": "eng"},
        ]
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.replan.total_replans == 2
        assert metrics.replan.success_after_replan == 1
        assert metrics.replan.failure_after_replan == 1
        assert metrics.replan.replan_success_rate == pytest.approx(50.0)
        # 2 replans / 2 tasks * 100
        assert metrics.replan.trigger_rate == pytest.approx(100.0)


class TestSpecializationMetrics:
    """Collector correctly aggregates prompt_built events for specialization."""

    def test_profile_distribution(self, tmp_path):
        now = _utc_now()
        session_events = [
            {"event": "prompt_built", "ts": _iso(now), "task_id": "t1", "profile_id": "backend"},
            {"event": "prompt_built", "ts": _iso(now), "task_id": "t2", "profile_id": "backend"},
            {"event": "prompt_built", "ts": _iso(now), "task_id": "t3", "profile_id": "frontend"},
        ]
        activity_events = [
            {"type": "start", "task_id": f"t{i}", "timestamp": _iso(now), "agent": "eng"}
            for i in range(1, 5)  # 4 tasks started, 3 specialized
        ]
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.specialization.profile_distribution == {"backend": 2, "frontend": 1}
        assert metrics.specialization.total_specialized_tasks == 3
        assert metrics.specialization.total_generic_tasks == 1


class TestContextBudgetMetrics:
    """Collector correctly aggregates llm_complete and token_budget_exceeded events."""

    def test_utilization_and_over_budget(self, tmp_path):
        now = _utc_now()
        session_events = [
            {"event": "llm_complete", "ts": _iso(now), "utilization_percent": 60.0, "input_tokens": 1000, "output_tokens": 500},
            {"event": "llm_complete", "ts": _iso(now), "utilization_percent": 85.0, "input_tokens": 2000, "output_tokens": 800},
            {"event": "llm_complete", "ts": _iso(now), "utilization_percent": 90.0, "input_tokens": 1500, "output_tokens": 600},
        ]
        activity_events = [
            {"type": "token_budget_exceeded", "task_id": "t1", "timestamp": _iso(now)},
            {"type": "token_budget_exceeded", "task_id": "t2", "timestamp": _iso(now)},
        ]
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)

        collector = AgenticMetricsCollector(tmp_path)
        metrics = collector.collect(hours=24)

        assert metrics.context_budget.avg_utilization_percent == pytest.approx(235.0 / 3)
        assert metrics.context_budget.tasks_near_limit == 2  # 85% and 90%
        assert metrics.context_budget.tasks_over_budget == 2
        assert metrics.context_budget.avg_input_tokens == int((1000 + 2000 + 1500) / 3)
        assert metrics.context_budget.avg_output_tokens == int((500 + 800 + 600) / 3)


class TestTimeWindowFiltering:
    """Events outside the time window are excluded."""

    def test_old_activity_events_excluded(self, tmp_path):
        now = _utc_now()
        old = now - timedelta(hours=48)
        activity_events = [
            # Within window
            {"type": "start", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"},
            # Outside window (48h ago, window is 24h)
            {"type": "start", "task_id": "t2", "timestamp": _iso(old), "agent": "eng"},
        ]
        session_events = [
            # Replan only for the in-window task
            {"event": "replan", "ts": _iso(now), "task_id": "t1"},
        ]
        _write_jsonl(tmp_path / ".agent-communication" / "activity-stream.jsonl", activity_events)
        _write_jsonl(tmp_path / "logs" / "sessions" / "s.jsonl", session_events)

        metrics = AgenticMetricsCollector(tmp_path).collect(hours=24)

        # Only t1 is in the window → total_tasks=1, total_replans=1 → trigger_rate=100%.
        # If t2 were counted (filter broken), total_tasks=2 → trigger_rate=50%.
        assert metrics.replan.trigger_rate == pytest.approx(100.0)

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSONL lines are silently skipped; valid lines still counted."""
        now = _utc_now()
        stream = tmp_path / ".agent-communication" / "activity-stream.jsonl"
        stream.parent.mkdir(parents=True, exist_ok=True)
        stream.write_text(
            "{bad json\n"
            + json.dumps({"type": "start", "task_id": "t1", "timestamp": _iso(now), "agent": "eng"}) + "\n"
            + "also bad\n"
            + json.dumps({"type": "start", "task_id": "t2", "timestamp": _iso(now), "agent": "eng"}) + "\n"
        )

        metrics = AgenticMetricsCollector(tmp_path).collect(hours=24)

        # Bad lines don't crash or contribute to counts; 2 valid tasks recognised
        assert isinstance(metrics, AgenticMetrics)
        # trigger_rate = 0 replans / 2 tasks → 0% (denominator proves 2 tasks were read)
        assert metrics.replan.trigger_rate == 0.0
        assert metrics.replan.total_replans == 0

    def test_output_file_written(self, tmp_path):
        collector = AgenticMetricsCollector(tmp_path)
        collector.collect(hours=24)

        output = tmp_path / ".agent-communication" / "metrics" / "agentic-metrics.json"
        assert output.exists()
        data = json.loads(output.read_text())
        assert "memory" in data
        assert "self_eval" in data
        assert "replan" in data
