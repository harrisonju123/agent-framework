"""Tests for cross-feature feedback loop wiring.

Covers:
- Self-eval failure → memory (error_recovery.py)
- QA findings → memory (review_cycle.py)
- Debate → specialization (engineer_specialization.py)
- Feedback metrics analytics (feedback_metrics.py)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.review_cycle import ReviewCycleManager, QAFinding, ReviewOutcome
from agent_framework.core.engineer_specialization import (
    adjust_specialization_from_debate,
    BACKEND_PROFILE,
    FRONTEND_PROFILE,
)
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.analytics.feedback_metrics import FeedbackMetrics, FeedbackMetricsReport
from agent_framework.memory.memory_store import MemoryEntry, MemoryStore


# --- Helpers ---

def _make_task(**overrides) -> Task:
    defaults = dict(
        id="test-task-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test description",
        context={"github_repo": "org/repo"},
        acceptance_criteria=["Handle errors properly", "Write unit tests"],
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_error_recovery(memory_store=None, session_logger=None) -> ErrorRecoveryManager:
    config = MagicMock()
    config.id = "engineer"
    config.base_id = "engineer"
    return ErrorRecoveryManager(
        config=config,
        queue=MagicMock(),
        llm=MagicMock(),
        logger=MagicMock(),
        session_logger=session_logger or MagicMock(),
        retry_handler=MagicMock(),
        escalation_handler=MagicMock(),
        workspace=Path("/tmp"),
        memory_store=memory_store,
    )


# --- Self-eval failure → memory ---

class TestSelfEvalFailureStorage:
    def test_stores_failure_to_memory(self):
        store = MagicMock()
        store.enabled = True
        recovery = _make_error_recovery(memory_store=store)
        task = _make_task()

        recovery.store_self_eval_failure(task, "FAIL: Missing error handling in handler.go")

        store.remember.assert_called_once()
        call_kwargs = store.remember.call_args[1]
        assert call_kwargs["category"] == "self_eval_failures"
        assert "Self-eval FAIL" in call_kwargs["content"]
        assert call_kwargs["repo_slug"] == "org/repo"

    def test_skips_when_memory_disabled(self):
        store = MagicMock()
        store.enabled = False
        recovery = _make_error_recovery(memory_store=store)
        task = _make_task()

        recovery.store_self_eval_failure(task, "FAIL")
        store.remember.assert_not_called()

    def test_skips_when_no_repo(self):
        store = MagicMock()
        store.enabled = True
        recovery = _make_error_recovery(memory_store=store)
        task = _make_task(context={})

        recovery.store_self_eval_failure(task, "FAIL")
        store.remember.assert_not_called()

    def test_builds_tags_from_criteria(self):
        store = MagicMock()
        store.enabled = True
        recovery = _make_error_recovery(memory_store=store)
        task = _make_task(acceptance_criteria=["Handle errors properly", "Write tests"])

        recovery.store_self_eval_failure(task, "FAIL: missed error handling")

        call_kwargs = store.remember.call_args[1]
        tags = call_kwargs["tags"]
        assert isinstance(tags, list)
        # Should have extracted words >4 chars from criteria
        assert any("handle" in t or "errors" in t or "properly" in t for t in tags)


# --- QA findings → memory ---

class TestQAFindingsPersistence:
    def test_persists_findings_to_memory(self):
        store = MagicMock()
        store.enabled = True
        store.remember = MagicMock(return_value=True)

        config = MagicMock()
        config.base_id = "qa"

        rcm = ReviewCycleManager(
            config=config,
            queue=MagicMock(),
            logger=MagicMock(),
            agent_definition=MagicMock(),
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
            memory_store=store,
        )

        findings = [
            QAFinding(
                file="handler.go",
                line_number=42,
                severity="HIGH",
                description="Missing error check",
                suggested_fix="Add if err != nil",
                category="correctness",
            ),
            QAFinding(
                file="main.go",
                line_number=10,
                severity="MEDIUM",
                description="Unused import",
                suggested_fix=None,
                category="readability",
            ),
        ]
        task = _make_task()
        count = rcm.persist_findings_to_memory(findings, task)
        assert count == 2
        assert store.remember.call_count == 2

        # Check first call
        first_call = store.remember.call_args_list[0][1]
        assert first_call["category"] == "qa_findings"
        assert "HIGH" in first_call["content"]
        assert "handler.go" in first_call["content"]

    def test_skips_when_no_memory_store(self):
        config = MagicMock()
        config.base_id = "qa"
        rcm = ReviewCycleManager(
            config=config,
            queue=MagicMock(),
            logger=MagicMock(),
            agent_definition=MagicMock(),
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
            memory_store=None,
        )
        result = rcm.persist_findings_to_memory([], _make_task())
        assert result == 0


# --- Debate → specialization ---

class TestDebateSpecializationAdjustment:
    def test_no_adjustment_without_memories(self):
        store = MagicMock()
        store.enabled = True
        store.recall.return_value = []

        result = adjust_specialization_from_debate(
            BACKEND_PROFILE, store, "org/repo"
        )
        assert result == BACKEND_PROFILE

    def test_no_adjustment_without_store(self):
        result = adjust_specialization_from_debate(
            BACKEND_PROFILE, None, "org/repo"
        )
        assert result == BACKEND_PROFILE

    def test_adjusts_when_debate_signals_frontend(self):
        store = MagicMock()
        store.enabled = True
        store.recall.return_value = [
            MemoryEntry(
                category="architectural_decisions",
                content="Debate conclusion: use React components and CSS modules for the frontend UI layer",
            ),
            MemoryEntry(
                category="architectural_decisions",
                content="Frontend approach with Vue components recommended for this feature",
            ),
        ]

        result = adjust_specialization_from_debate(
            BACKEND_PROFILE, store, "org/repo"
        )
        # Should have adjusted to frontend because debate has strong frontend signals
        assert result.id == "frontend"

    def test_no_adjustment_when_debate_matches_profile(self):
        store = MagicMock()
        store.enabled = True
        store.recall.return_value = [
            MemoryEntry(
                category="architectural_decisions",
                content="Backend API design with database migration approach recommended",
            ),
        ]

        result = adjust_specialization_from_debate(
            BACKEND_PROFILE, store, "org/repo"
        )
        assert result == BACKEND_PROFILE

    def test_handles_exception_gracefully(self):
        store = MagicMock()
        store.enabled = True
        store.recall.side_effect = Exception("memory error")

        result = adjust_specialization_from_debate(
            BACKEND_PROFILE, store, "org/repo"
        )
        assert result == BACKEND_PROFILE


# --- Feedback metrics ---

class TestFeedbackMetrics:
    def test_empty_report_when_no_sessions(self, tmp_path):
        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert isinstance(report, FeedbackMetricsReport)
        assert report.total_sessions_scanned == 0
        assert report.self_eval_failures_stored == 0

    def test_counts_self_eval_events(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "task-1.jsonl"

        events = [
            {"event": "self_eval_failure_stored", "ts": datetime.now(timezone.utc).isoformat(),
             "content_preview": "Missing error handling"},
            {"event": "self_eval_failure_stored", "ts": datetime.now(timezone.utc).isoformat(),
             "content_preview": "Missing tests"},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in events))

        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.self_eval_failures_stored == 2
        assert report.total_sessions_scanned == 1

    def test_counts_qa_findings_events(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "task-2.jsonl"

        events = [
            {"event": "qa_findings_persisted", "ts": datetime.now(timezone.utc).isoformat(),
             "findings_count": 3},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in events))

        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.qa_findings_persisted == 3

    def test_counts_qa_pattern_injection(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "task-3.jsonl"

        events = [
            {"event": "qa_pattern_injected", "ts": datetime.now(timezone.utc).isoformat(),
             "pattern_count": 2, "top_patterns": ["missing error check", "unused import"]},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in events))

        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.qa_warnings_injected == 1
        assert len(report.qa_pattern_top_findings) == 2

    def test_counts_specialization_adjustments(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "task-4.jsonl"

        events = [
            {"event": "specialization_adjusted", "ts": datetime.now(timezone.utc).isoformat(),
             "original_profile": "backend", "adjusted_profile": "frontend",
             "signal_strength": 5},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in events))

        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.specialization_adjustments == 1
        assert report.adjustment_details[0]["original"] == "backend"

    def test_counts_feedback_bus_events(self, tmp_path):
        sessions_dir = tmp_path / "logs" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "task-5.jsonl"

        events = [
            {"event": "feedback_emitted", "ts": datetime.now(timezone.utc).isoformat(),
             "source": "self_eval", "category": "self_eval_failures"},
            {"event": "feedback_emitted", "ts": datetime.now(timezone.utc).isoformat(),
             "source": "qa", "category": "qa_findings"},
        ]
        session_file.write_text("\n".join(json.dumps(e) for e in events))

        metrics = FeedbackMetrics(workspace=tmp_path)
        report = metrics.generate_report(hours=24)
        assert report.feedback_events_emitted == 2
