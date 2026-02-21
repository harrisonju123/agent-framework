"""Tests for self-eval failure → memory storage in ErrorRecoveryManager."""

from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.feedback_bus import FeedbackBus, FeedbackEvent
from agent_framework.core.task import Task, TaskType
from agent_framework.memory.memory_store import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def task():
    t = Task(
        id="task-001",
        title="Add tests",
        description="Add unit tests for the auth module",
        type=TaskType.IMPLEMENTATION,
    )
    t.context["github_repo"] = "org/repo"
    t.acceptance_criteria = ["All tests pass", "Coverage above 80%"]
    return t


def _make_manager(memory_store, feedback_bus=None):
    """Create a minimal ErrorRecoveryManager for testing memory storage."""
    config = MagicMock()
    config.base_id = "engineer"
    config.id = "engineer-1"

    mgr = ErrorRecoveryManager(
        config=config,
        queue=MagicMock(),
        logger=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
        workspace=MagicMock(),
        memory_store=memory_store,
        feedback_bus=feedback_bus,
    )
    return mgr


class TestExtractMissedCriteria:
    def test_extracts_matching_criteria(self):
        verdict = "FAIL: Tests are not passing. Coverage is below 80%."
        criteria = ["All tests pass", "Coverage above 80%"]

        missed = ErrorRecoveryManager._extract_missed_criteria(verdict, criteria)
        assert len(missed) >= 1

    def test_returns_all_on_generic_fail(self):
        verdict = "FAIL: nothing works"
        criteria = ["Criterion A", "Criterion B"]

        missed = ErrorRecoveryManager._extract_missed_criteria(verdict, criteria)
        assert missed == criteria

    def test_empty_criteria_returns_empty(self):
        missed = ErrorRecoveryManager._extract_missed_criteria("FAIL", [])
        assert missed == []


class TestStoreSelfEvalFailure:
    def test_stores_to_memory(self, memory_store, task):
        mgr = _make_manager(memory_store)
        verdict = "FAIL: Tests are failing. Coverage is below target."
        criteria = ["All tests pass", "Coverage above 80%"]

        mgr._store_self_eval_failure(task, verdict, criteria)

        entries = memory_store.recall("org/repo", "engineer", category="self_eval_failures")
        assert len(entries) == 1
        assert "self_eval_failure" in entries[0].tags

    def test_skips_when_memory_disabled(self, tmp_path, task):
        disabled_store = MemoryStore(workspace=tmp_path, enabled=False)
        mgr = _make_manager(disabled_store)

        mgr._store_self_eval_failure(task, "FAIL: something", ["criteria"])
        # No error, just skips

    def test_skips_when_no_repo_slug(self, memory_store):
        task_no_repo = Task(
            id="task-002",
            title="Test",
            description="Test",
            type=TaskType.IMPLEMENTATION,
        )
        mgr = _make_manager(memory_store)

        mgr._store_self_eval_failure(task_no_repo, "FAIL", ["criteria"])
        entries = memory_store.recall_all("", "engineer")
        assert len(entries) == 0

    def test_emits_feedback_event(self, memory_store, task):
        bus = FeedbackBus()
        received = []
        bus.register_consumer("self_eval_failures", lambda e: received.append(e))

        mgr = _make_manager(memory_store, feedback_bus=bus)
        mgr._store_self_eval_failure(task, "FAIL: tests not passing", ["All tests pass"])

        assert len(received) == 1
        assert received[0].source == "self_eval"
        assert received[0].category == "self_eval_failures"
