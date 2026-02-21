"""Tests for SessionLogger feedback bus event methods."""

import json
from pathlib import Path

import pytest

from agent_framework.core.session_logger import SessionLogger


@pytest.fixture
def logger(tmp_path):
    return SessionLogger(logs_dir=tmp_path, task_id="test-task", enabled=True)


def _read_events(logger) -> list[dict]:
    """Read all events from the session log."""
    logger.close()
    events = []
    if logger._path.exists():
        for line in logger._path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


class TestFeedbackEmitted:
    def test_logs_feedback_emitted_event(self, logger):
        logger.log_feedback_emitted(
            source="self_eval",
            category="self_eval_failures",
            content="Missed criteria: All tests pass",
        )

        events = _read_events(logger)
        assert len(events) == 1
        assert events[0]["event"] == "feedback_emitted"
        assert events[0]["source"] == "self_eval"
        assert events[0]["category"] == "self_eval_failures"
        assert events[0]["content_preview"] == "Missed criteria: All tests pass"

    def test_truncates_long_content(self, logger):
        long_content = "x" * 500
        logger.log_feedback_emitted("test", "cat", long_content)

        events = _read_events(logger)
        assert len(events[0]["content_preview"]) <= 200


class TestQAPatternInjected:
    def test_logs_qa_pattern_event(self, logger):
        logger.log_qa_pattern_injected(
            pattern_count=3,
            top_patterns=["sql injection", "missing validation", "xss"],
        )

        events = _read_events(logger)
        assert len(events) == 1
        assert events[0]["event"] == "qa_pattern_injected"
        assert events[0]["pattern_count"] == 3
        assert len(events[0]["top_patterns"]) == 3


class TestSpecializationAdjusted:
    def test_logs_specialization_event(self, logger):
        logger.log_specialization_adjusted(
            original_profile="backend",
            adjusted_profile="frontend",
            debate_id="debate-123",
        )

        events = _read_events(logger)
        assert len(events) == 1
        assert events[0]["event"] == "specialization_adjusted"
        assert events[0]["original_profile"] == "backend"
        assert events[0]["adjusted_profile"] == "frontend"
        assert events[0]["debate_id"] == "debate-123"


class TestDisabledLogger:
    def test_no_events_when_disabled(self, tmp_path):
        disabled = SessionLogger(logs_dir=tmp_path, task_id="noop", enabled=False)
        disabled.log_feedback_emitted("test", "cat", "content")
        disabled.log_qa_pattern_injected(3, ["a", "b"])
        disabled.log_specialization_adjusted("a", "b")
        # Should not create any files
        assert not (tmp_path / "sessions" / "noop.jsonl").exists()
