"""Tests for FeedbackMetrics analytics collector."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.feedback_metrics import FeedbackMetrics, FeedbackMetricsReport


@pytest.fixture
def workspace(tmp_path):
    sessions_dir = tmp_path / "logs" / "sessions"
    sessions_dir.mkdir(parents=True)
    return tmp_path


def _write_session_events(workspace: Path, task_id: str, events: list[dict]):
    """Write session events as JSONL."""
    sessions_dir = workspace / "logs" / "sessions"
    path = sessions_dir / f"{task_id}.jsonl"
    with open(path, "w") as f:
        for evt in events:
            if "ts" not in evt:
                evt["ts"] = datetime.now(timezone.utc).isoformat()
            if "task_id" not in evt:
                evt["task_id"] = task_id
            f.write(json.dumps(evt) + "\n")


class TestGenerateReport:
    def test_empty_sessions(self, workspace):
        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert isinstance(report, FeedbackMetricsReport)
        assert report.self_eval_memories_stored == 0
        assert report.qa_warnings_injected == 0
        assert report.specialization_adjustments == 0

    def test_counts_self_eval_events(self, workspace):
        _write_session_events(workspace, "task-1", [
            {"event": "feedback_emitted", "source": "self_eval", "category": "self_eval_failures", "content_preview": "Missed tests"},
            {"event": "feedback_emitted", "source": "self_eval", "category": "self_eval_failures", "content_preview": "Missed coverage"},
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.self_eval_memories_stored == 2

    def test_counts_qa_pattern_events(self, workspace):
        _write_session_events(workspace, "task-1", [
            {"event": "qa_pattern_injected", "pattern_count": 3, "top_patterns": ["sql injection", "xss"]},
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.qa_recurring_patterns_detected == 3
        assert report.qa_warnings_injected == 1
        assert "sql injection" in report.top_recurring_findings

    def test_counts_specialization_adjustments(self, workspace):
        _write_session_events(workspace, "task-1", [
            {
                "event": "specialization_adjusted",
                "original_profile": "backend",
                "adjusted_profile": "frontend",
                "debate_id": "debate-123",
            },
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.specialization_adjustments == 1
        assert len(report.recent_adjustments) == 1

    def test_counts_qa_findings_persisted(self, workspace):
        _write_session_events(workspace, "task-1", [
            {"event": "qa_findings_persisted", "count": 5, "repo": "org/repo"},
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.qa_findings_persisted == 5

    def test_aggregates_across_tasks(self, workspace):
        _write_session_events(workspace, "task-1", [
            {"event": "feedback_emitted", "source": "self_eval", "category": "self_eval_failures", "content_preview": "A"},
        ])
        _write_session_events(workspace, "task-2", [
            {"event": "feedback_emitted", "source": "self_eval", "category": "self_eval_failures", "content_preview": "B"},
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.self_eval_memories_stored == 2

    def test_events_by_category_breakdown(self, workspace):
        _write_session_events(workspace, "task-1", [
            {"event": "feedback_emitted", "category": "self_eval_failures", "content_preview": "x"},
            {"event": "feedback_emitted", "category": "qa_findings", "content_preview": "y"},
            {"event": "feedback_emitted", "category": "qa_findings", "content_preview": "z"},
        ])

        metrics = FeedbackMetrics(workspace=workspace)
        report = metrics.generate_report(hours=24)

        assert report.events_by_category["self_eval_failures"] == 1
        assert report.events_by_category["qa_findings"] == 2

    def test_no_sessions_dir(self, tmp_path):
        """Handles missing sessions directory gracefully."""
        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.self_eval_memories_stored == 0


class TestExtractTopItems:
    def test_deduplicates_and_counts(self):
        items = FeedbackMetrics._extract_top_items(
            ["foo", "bar", "foo", "foo", "baz"],
            limit=2,
        )
        assert items[0] == "foo"
        assert len(items) == 2

    def test_empty_input(self):
        assert FeedbackMetrics._extract_top_items([]) == []

    def test_strips_whitespace(self):
        items = FeedbackMetrics._extract_top_items(["  foo  ", "foo", " foo"], limit=5)
        assert len(items) == 1
