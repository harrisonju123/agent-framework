"""Tests for FeedbackLoopAnalyzer metrics aggregation."""

import json
from datetime import datetime, timezone

import pytest

from agent_framework.analytics.feedback_loop_metrics import (
    FeedbackLoopAnalyzer,
    FeedbackLoopMetrics,
    FeedbackLoopReport,
)


def _write_session_event(sessions_dir, task_id, event):
    """Append a session event to a task's JSONL file."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{task_id}.jsonl"
    event.setdefault("task_id", task_id)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def analyzer(workspace):
    return FeedbackLoopAnalyzer(workspace)


class TestEmptyReport:
    def test_empty_when_no_sessions(self, analyzer):
        report = analyzer.generate_report(hours=24)
        assert isinstance(report, FeedbackLoopReport)
        assert report.metrics.total_self_eval_stores == 0
        assert report.metrics.total_qa_pattern_stores == 0
        assert report.metrics.total_qa_warnings_injected == 0
        assert report.metrics.total_specialization_hints == 0
        assert report.raw_events == []


class TestSelfEvalMetrics:
    def test_counts_self_eval_stores(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_self_eval",
            "memories_stored": 2,
        })
        _write_session_event(sessions_dir, "task-2", {
            "event": "feedback_bus_self_eval",
            "memories_stored": 1,
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.total_self_eval_stores == 2
        assert report.metrics.self_eval_memories_stored == 3

    def test_missed_criteria_category_breakdown(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_self_eval",
            "memories_stored": 3,
        })

        report = analyzer.generate_report(hours=24)
        cats = {c.category: c for c in report.metrics.by_category}
        assert "missed_criteria" in cats
        assert cats["missed_criteria"].store_count == 1
        assert cats["missed_criteria"].total_memories == 3


class TestQaPatternMetrics:
    def test_counts_qa_pattern_stores(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_qa_pattern",
            "memories_stored": 4,
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.total_qa_pattern_stores == 1
        assert report.metrics.qa_pattern_memories_stored == 4

    def test_qa_patterns_category_breakdown(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_qa_pattern",
            "memories_stored": 2,
        })
        _write_session_event(sessions_dir, "task-2", {
            "event": "feedback_bus_qa_pattern",
            "memories_stored": 1,
        })

        report = analyzer.generate_report(hours=24)
        cats = {c.category: c for c in report.metrics.by_category}
        assert "qa_patterns" in cats
        assert cats["qa_patterns"].store_count == 2
        assert cats["qa_patterns"].total_memories == 3


class TestQaWarningsInjected:
    def test_counts_qa_warnings_injected(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "qa_warnings_injected",
            "repo": "owner/repo",
            "chars": 350,
        })
        _write_session_event(sessions_dir, "task-2", {
            "event": "qa_warnings_injected",
            "repo": "owner/repo",
            "chars": 200,
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.total_qa_warnings_injected == 2
        assert report.metrics.qa_warnings_chars_injected == 550


class TestSpecializationHints:
    def test_counts_specialization_hints(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "memory_store",
            "category": "specialization_hints",
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.total_specialization_hints == 1


class TestHintHitRate:
    def test_hit_rate_when_stores_exist(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        # 2 stores happened, 1 warning was injected -> 50% hit rate
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_self_eval",
            "memories_stored": 1,
        })
        _write_session_event(sessions_dir, "task-2", {
            "event": "feedback_bus_qa_pattern",
            "memories_stored": 2,
        })
        _write_session_event(sessions_dir, "task-3", {
            "event": "qa_warnings_injected",
            "chars": 100,
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.specialization_hint_hit_rate == pytest.approx(50.0)

    def test_hit_rate_zero_when_no_stores(self, workspace, analyzer):
        report = analyzer.generate_report(hours=24)
        assert report.metrics.specialization_hint_hit_rate == 0.0


class TestTopPatterns:
    def test_aggregates_top_patterns(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        for i in range(5):
            _write_session_event(sessions_dir, f"task-{i}", {
                "event": "feedback_bus_self_eval",
                "task_id": f"task-{i}",
                "memories_stored": 1,
            })

        report = analyzer.generate_report(hours=24)
        assert len(report.metrics.top_patterns) >= 1
        assert any(p.category == "missed_criteria" for p in report.metrics.top_patterns)


class TestReportStructure:
    def test_report_has_correct_fields(self, workspace, analyzer):
        report = analyzer.generate_report(hours=48)
        assert report.time_range_hours == 48
        assert isinstance(report.generated_at, datetime)
        assert isinstance(report.metrics, FeedbackLoopMetrics)
        assert isinstance(report.raw_events, list)


class TestMixedEvents:
    def test_handles_mixed_event_types(self, workspace, analyzer):
        sessions_dir = workspace / "logs" / "sessions"
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_self_eval",
            "memories_stored": 2,
        })
        _write_session_event(sessions_dir, "task-1", {
            "event": "feedback_bus_qa_pattern",
            "memories_stored": 1,
        })
        _write_session_event(sessions_dir, "task-1", {
            "event": "qa_warnings_injected",
            "chars": 300,
        })
        # Non-feedback event should be ignored
        _write_session_event(sessions_dir, "task-1", {
            "event": "prompt_built",
            "prompt_length": 5000,
        })

        report = analyzer.generate_report(hours=24)
        assert report.metrics.total_self_eval_stores == 1
        assert report.metrics.total_qa_pattern_stores == 1
        assert report.metrics.total_qa_warnings_injected == 1
        # Only feedback events should be in raw_events
        assert len(report.raw_events) == 3
