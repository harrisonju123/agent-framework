"""Tests for FeedbackBus — cross-feature learning loop handlers."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.memory.feedback_bus import (
    FeedbackBus,
    _extract_tags_from_task,
    _file_to_pattern,
    _qa_finding_matches,
    _detect_domain_from_keywords,
    _QA_RECURRENCE_THRESHOLD,
)
from agent_framework.memory.memory_store import MemoryStore


def _make_task(
    task_id="task-1",
    repo="org/repo",
    acceptance_criteria=None,
    replan_history=None,
    task_type=None,
):
    """Build a minimal task-like object for testing."""
    task = SimpleNamespace()
    task.id = task_id
    task.context = {"github_repo": repo}
    task.acceptance_criteria = acceptance_criteria or []
    task.replan_history = replan_history or []
    task.type = task_type
    return task


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


class TestOnSelfEvalFail:
    def test_stores_missed_criteria(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(acceptance_criteria=["All tests pass", "No linting errors"])
        critique = "FAIL: The tests were not passing. Multiple linting errors remain."

        bus.on_self_eval_fail(task, critique)

        results = store.recall("org/repo", "engineer", category="missed_criteria")
        assert len(results) >= 1
        # At least one criterion should be stored
        contents = [r.content for r in results]
        assert any("tests" in c.lower() or "linting" in c.lower() for c in contents)

    def test_stores_raw_critique_when_no_criteria_match(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(acceptance_criteria=[])
        critique = "FAIL: Something was wrong but no specific criteria"

        bus.on_self_eval_fail(task, critique)

        results = store.recall("org/repo", "engineer", category="missed_criteria")
        assert len(results) == 1
        assert "Something was wrong" in results[0].content

    def test_noop_when_store_disabled(self, tmp_path):
        store = MemoryStore(workspace=tmp_path, enabled=False)
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()

        bus.on_self_eval_fail(task, "FAIL: something")

        results = store.recall("org/repo", "engineer", category="missed_criteria")
        assert results == []

    def test_noop_when_no_repo(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()
        task.context = {}  # No github_repo

        bus.on_self_eval_fail(task, "FAIL: something")

        # No crash, no memory stored
        results = store.recall("org/repo", "engineer", category="missed_criteria")
        assert results == []

    def test_tags_include_task_type(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(acceptance_criteria=["Complete implementation"])
        task.type = "implementation"

        bus.on_self_eval_fail(task, "FAIL: implementation was not complete")

        results = store.recall("org/repo", "engineer", category="missed_criteria")
        assert len(results) >= 1
        assert "implementation" in results[0].tags


class TestOnReplanSuccess:
    def test_stores_recovery_pattern(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(
            replan_history=[{
                "error_type": "test_failure",
                "files_involved": ["src/foo.py", "tests/test_foo.py"],
                "revised_plan": "- Use mock instead of real API calls\nDetails...",
                "approach_tried": "Direct API calls",
            }]
        )

        bus.on_replan_success(task, "org/repo")

        results = store.recall("org/repo", "engineer", category="recovery_patterns")
        assert len(results) == 1
        assert "test_failure" in results[0].content
        assert "resolved" in results[0].content
        # Should have file extension tags
        assert ".py" in results[0].tags

    def test_noop_when_no_replan_history(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(replan_history=[])

        bus.on_replan_success(task, "org/repo")

        results = store.recall("org/repo", "engineer", category="recovery_patterns")
        assert results == []

    def test_noop_when_store_disabled(self, tmp_path):
        store = MemoryStore(workspace=tmp_path, enabled=False)
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(replan_history=[{"error_type": "x", "files_involved": [], "revised_plan": "y"}])

        bus.on_replan_success(task, "org/repo")

        results = store.recall("org/repo", "engineer", category="recovery_patterns")
        assert results == []

    def test_enriches_tags_with_extensions(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task(
            replan_history=[{
                "error_type": "build_error",
                "files_involved": ["src/main.go", "go.mod"],
                "revised_plan": "Fix import paths",
            }]
        )

        bus.on_replan_success(task, "org/repo")

        results = store.recall("org/repo", "engineer", category="recovery_patterns")
        assert len(results) == 1
        assert ".go" in results[0].tags
        assert "build_error" in results[0].tags


class TestOnQAFindings:
    def test_stores_recurring_patterns(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()

        # Create findings that repeat (same category + severity on similar files)
        findings = [
            {"file": "src/handler.py", "severity": "HIGH", "category": "security", "description": "SQL injection risk"},
            {"file": "src/api.py", "severity": "HIGH", "category": "security", "description": "SQL injection in query"},
        ]

        bus.on_qa_findings(task, findings)

        results = store.recall("org/repo", "engineer", category="qa_warnings")
        assert len(results) == 1
        assert "security" in results[0].content.lower()
        assert "seen 2x" in results[0].content

    def test_ignores_non_recurring_findings(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()

        # Each finding is unique — no recurring pattern
        findings = [
            {"file": "src/handler.py", "severity": "HIGH", "category": "security", "description": "SQL injection"},
            {"file": "src/api.go", "severity": "LOW", "category": "readability", "description": "Long function"},
        ]

        bus.on_qa_findings(task, findings)

        results = store.recall("org/repo", "engineer", category="qa_warnings")
        assert results == []

    def test_handles_empty_findings(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()

        bus.on_qa_findings(task, [])

        results = store.recall("org/repo", "engineer", category="qa_warnings")
        assert results == []

    def test_handles_qa_finding_dataclass(self, store):
        """QAFinding dataclasses from review_cycle.py should work too."""
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()

        # Simulate QAFinding objects
        f1 = SimpleNamespace(file="src/a.py", severity="CRITICAL", category="correctness", description="Missing null check", suggested_fix=None, line_number=42)
        f2 = SimpleNamespace(file="src/b.py", severity="CRITICAL", category="correctness", description="Missing null check in handler", suggested_fix=None, line_number=100)

        bus.on_qa_findings(task, [f1, f2])

        results = store.recall("org/repo", "engineer", category="qa_warnings")
        assert len(results) == 1

    def test_noop_when_no_repo(self, store):
        bus = FeedbackBus(memory_store=store, agent_type="engineer")
        task = _make_task()
        task.context = {}

        findings = [
            {"file": "x.py", "severity": "HIGH", "category": "security", "description": "issue"},
            {"file": "y.py", "severity": "HIGH", "category": "security", "description": "issue"},
        ]
        bus.on_qa_findings(task, findings)

        # No crash


class TestOnDebateComplete:
    def test_records_domain_mismatch(self):
        registry = MagicMock()
        bus = FeedbackBus(memory_store=MagicMock(), agent_type="engineer", profile_registry=registry)
        task = _make_task()

        debate_result = {
            "confidence": 0.3,
            "synthesis": "The frontend React component needs careful CSS styling and DOM manipulation",
        }

        bus.on_debate_complete(task, debate_result, current_profile_id="backend-go")

        registry.record_domain_feedback.assert_called_once()
        call_kwargs = registry.record_domain_feedback.call_args
        assert call_kwargs[1]["profile_id"] == "backend-go"
        assert call_kwargs[1]["mismatch_signal"] is True
        assert "frontend" in call_kwargs[1]["domain_tags"]

    def test_ignores_high_confidence(self):
        registry = MagicMock()
        bus = FeedbackBus(memory_store=MagicMock(), agent_type="engineer", profile_registry=registry)
        task = _make_task()

        debate_result = {"confidence": 0.8, "synthesis": "Clear backend API design needed"}

        bus.on_debate_complete(task, debate_result, current_profile_id="backend-go")

        registry.record_domain_feedback.assert_not_called()

    def test_ignores_without_profile_id(self):
        registry = MagicMock()
        bus = FeedbackBus(memory_store=MagicMock(), agent_type="engineer", profile_registry=registry)
        task = _make_task()

        debate_result = {"confidence": 0.2, "synthesis": "Database schema migration needed"}

        bus.on_debate_complete(task, debate_result, current_profile_id=None)

        registry.record_domain_feedback.assert_not_called()

    def test_ignores_without_registry(self):
        bus = FeedbackBus(memory_store=MagicMock(), agent_type="engineer", profile_registry=None)
        task = _make_task()

        debate_result = {"confidence": 0.2, "synthesis": "Database schema migration needed"}

        # No crash
        bus.on_debate_complete(task, debate_result, current_profile_id="backend-go")

    def test_ignores_no_domain_keywords(self):
        registry = MagicMock()
        bus = FeedbackBus(memory_store=MagicMock(), agent_type="engineer", profile_registry=registry)
        task = _make_task()

        debate_result = {"confidence": 0.3, "synthesis": "Generic improvement needed"}

        bus.on_debate_complete(task, debate_result, current_profile_id="backend-go")

        registry.record_domain_feedback.assert_not_called()


class TestHelpers:
    def test_file_to_pattern_python(self):
        assert _file_to_pattern("src/foo/bar.py") == "**/*.py"

    def test_file_to_pattern_go(self):
        assert _file_to_pattern("pkg/handler.go") == "**/*.go"

    def test_file_to_pattern_test(self):
        assert _file_to_pattern("tests/unit/test_foo.py") == "**/tests/**/*.py"

    def test_file_to_pattern_no_ext(self):
        assert _file_to_pattern("Makefile") == "**/*"

    def test_file_to_pattern_empty(self):
        assert _file_to_pattern("") == "**/*"

    def test_extract_tags_from_task(self):
        task = SimpleNamespace()
        task.type = "implementation"
        task.context = {"github_repo": "org/repo", "jira_project": "PROJ"}
        tags = _extract_tags_from_task(task)
        assert "implementation" in tags
        assert "repo" in tags
        assert "PROJ" in tags

    def test_extract_tags_no_type(self):
        task = SimpleNamespace()
        task.type = None
        task.context = {"github_repo": "org/repo"}
        tags = _extract_tags_from_task(task)
        assert "repo" in tags

    def test_qa_finding_matches_same(self):
        a = {"category": "security", "severity": "HIGH", "file_pattern": "**/*.py"}
        b = {"category": "security", "severity": "HIGH", "file_pattern": "**/*.py"}
        assert _qa_finding_matches(a, b) is True

    def test_qa_finding_matches_different(self):
        a = {"category": "security", "severity": "HIGH", "file_pattern": "**/*.py"}
        b = {"category": "readability", "severity": "LOW", "file_pattern": "**/*.go"}
        assert _qa_finding_matches(a, b) is False

    def test_detect_domain_from_keywords(self):
        domains = _detect_domain_from_keywords("We need to fix the React component CSS")
        assert "frontend" in domains

    def test_detect_domain_multiple(self):
        domains = _detect_domain_from_keywords("The API endpoint needs SQL query optimization with JWT auth")
        assert "backend" in domains
        assert "database" in domains
        assert "security" in domains

    def test_detect_domain_empty(self):
        domains = _detect_domain_from_keywords("Nothing specific here")
        assert domains == []
