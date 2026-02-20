"""Tests for FeedbackBus — cross-feature learning loop."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.memory.feedback_bus import (
    FeedbackBus,
    _detect_domain_from_text,
    _extract_tags_from_task,
    _file_to_pattern,
    _qa_finding_matches,
)
from agent_framework.memory.memory_store import MemoryStore, MemoryEntry
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def session_logger():
    logger = MagicMock()
    return logger


@pytest.fixture
def profile_registry():
    reg = MagicMock()
    return reg


@pytest.fixture
def bus(store, session_logger, profile_registry):
    return FeedbackBus(
        memory_store=store,
        session_logger=session_logger,
        profile_registry=profile_registry,
    )


@pytest.fixture
def repo():
    return "myorg/myrepo"


def _make_task(**overrides):
    defaults = dict(
        id="test-task-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=5,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        context={"github_repo": "myorg/myrepo", "files_to_modify": ["src/main.py", "tests/test_main.py"]},
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestOnSelfEvalFail:
    def test_stores_acceptance_gap_memory(self, bus, store, repo):
        task = _make_task()
        bus.on_self_eval_fail(
            task=task,
            verdict="FAIL: Missing unit tests for the new endpoint",
            repo_slug=repo,
            agent_type="engineer",
        )

        memories = store.recall(repo, "engineer", category="acceptance_gaps")
        assert len(memories) == 1
        assert "Missing unit tests" in memories[0].content
        assert memories[0].source_task_id == "test-task-1"

    def test_extracts_tags_from_task(self, bus, store, repo):
        task = _make_task()
        bus.on_self_eval_fail(
            task=task,
            verdict="FAIL: No error handling",
            repo_slug=repo,
            agent_type="engineer",
        )

        memories = store.recall(repo, "engineer", category="acceptance_gaps")
        assert len(memories) == 1
        tags = memories[0].tags
        assert "implementation" in tags
        assert ".py" in tags

    def test_logs_session_event(self, bus, session_logger, repo):
        task = _make_task()
        bus.on_self_eval_fail(
            task=task,
            verdict="FAIL: Missing tests",
            repo_slug=repo,
            agent_type="engineer",
        )

        session_logger.log.assert_called_once()
        call_args = session_logger.log.call_args
        assert call_args[0][0] == "feedback_self_eval_stored"

    def test_noop_when_memory_disabled(self, session_logger, profile_registry):
        store = MagicMock()
        store.enabled = False
        bus = FeedbackBus(memory_store=store, session_logger=session_logger)

        task = _make_task()
        bus.on_self_eval_fail(task=task, verdict="FAIL", repo_slug="x", agent_type="e")
        store.remember.assert_not_called()

    def test_noop_when_verdict_empty_after_fail(self, bus, store, repo):
        task = _make_task()
        bus.on_self_eval_fail(
            task=task,
            verdict="FAIL",
            repo_slug=repo,
            agent_type="engineer",
        )
        # "FAIL" alone with nothing after should still produce empty gap
        memories = store.recall(repo, "engineer", category="acceptance_gaps")
        assert len(memories) == 0


class TestOnReplanSuccess:
    def test_stores_enriched_replan_history(self, bus, store, repo):
        task = _make_task(
            status=TaskStatus.COMPLETED,
            replan_history=[
                {
                    "attempt": 1,
                    "error": "ImportError: no module named foo",
                    "error_type": "import_error",
                    "approach_tried": "direct import",
                    "files_involved": ["src/main.py"],
                    "revised_plan": "- Use lazy import pattern instead",
                },
                {
                    "attempt": 2,
                    "error": "Still failing",
                    "error_type": "import_error",
                    "approach_tried": "lazy import",
                    "files_involved": ["src/main.py"],
                    "revised_plan": "- Install missing dependency first",
                },
            ],
        )

        bus.on_replan_success(task=task, repo_slug=repo, agent_type="engineer")

        memories = store.recall(repo, "engineer", category="past_failures")
        assert len(memories) == 1
        content = memories[0].content
        assert "Recovery from import_error" in content
        assert "2 attempts" in content
        assert "Attempt 1" in content
        assert "Winning approach" in content

    def test_tags_include_error_type(self, bus, store, repo):
        task = _make_task(
            replan_history=[{
                "attempt": 1,
                "error": "test error",
                "error_type": "compilation_error",
                "approach_tried": "fix syntax",
                "files_involved": [],
                "revised_plan": "- Rewrite module",
            }],
        )

        bus.on_replan_success(task=task, repo_slug=repo, agent_type="engineer")

        memories = store.recall(repo, "engineer", category="past_failures")
        assert "compilation_error" in memories[0].tags

    def test_logs_session_event(self, bus, session_logger, repo):
        task = _make_task(
            replan_history=[{
                "attempt": 1, "error": "e", "error_type": "t",
                "approach_tried": "a", "files_involved": [],
                "revised_plan": "- plan",
            }],
        )
        bus.on_replan_success(task=task, repo_slug=repo, agent_type="engineer")

        session_logger.log.assert_called_once()
        assert session_logger.log.call_args[0][0] == "feedback_replan_stored"

    def test_noop_without_replan_history(self, bus, store, repo):
        task = _make_task(replan_history=[])
        bus.on_replan_success(task=task, repo_slug=repo, agent_type="engineer")
        assert store.recall(repo, "engineer", category="past_failures") == []


class TestOnQAFindings:
    def _make_finding(self, **overrides):
        defaults = dict(
            file="src/main.py",
            line_number=42,
            severity="HIGH",
            description="Missing error handling for null input",
            suggested_fix="Add null check",
            category="correctness",
        )
        defaults.update(overrides)
        return MagicMock(**defaults)

    def test_stores_qa_findings_as_recurring(self, bus, store, repo):
        findings = [self._make_finding()]
        task = _make_task()

        bus.on_qa_findings(task=task, findings=findings, repo_slug=repo)

        memories = store.recall(repo, "shared", category="qa_recurring")
        assert len(memories) == 1
        assert "correctness" in memories[0].content
        assert "Missing error handling" in memories[0].content

    def test_tags_include_category_and_severity(self, bus, store, repo):
        findings = [self._make_finding(severity="CRITICAL", category="security")]
        task = _make_task()

        bus.on_qa_findings(task=task, findings=findings, repo_slug=repo)

        memories = store.recall(repo, "shared", category="qa_recurring")
        tags = memories[0].tags
        assert "security" in tags
        assert "critical" in tags

    def test_multiple_findings_stored(self, bus, store, repo):
        findings = [
            self._make_finding(description="Issue 1", category="security"),
            self._make_finding(description="Issue 2", category="performance"),
        ]
        task = _make_task()

        bus.on_qa_findings(task=task, findings=findings, repo_slug=repo)

        memories = store.recall(repo, "shared", category="qa_recurring")
        assert len(memories) == 2

    def test_logs_session_event(self, bus, session_logger, repo):
        findings = [self._make_finding()]
        task = _make_task()

        bus.on_qa_findings(task=task, findings=findings, repo_slug=repo)

        session_logger.log.assert_called_once()
        assert session_logger.log.call_args[0][0] == "feedback_qa_recurring_stored"

    def test_noop_with_empty_findings(self, bus, store, repo):
        task = _make_task()
        bus.on_qa_findings(task=task, findings=[], repo_slug=repo)
        assert store.recall(repo, "shared", category="qa_recurring") == []


class TestOnDebateComplete:
    def test_stores_debate_decision_in_shared_memory(self, bus, store, repo):
        debate = {
            "topic": "REST vs gRPC for internal services",
            "synthesis": {
                "recommendation": "Use gRPC for internal, REST for public APIs",
                "confidence": "high",
                "trade_offs": ["gRPC has steeper learning curve", "REST is more universal"],
            },
        }

        bus.on_debate_complete(debate_result=debate, repo_slug=repo, task_id="t1")

        memories = store.recall(repo, "shared", category="architectural_decisions")
        assert len(memories) == 1
        assert "gRPC" in memories[0].content
        assert "REST" in memories[0].content
        assert "high" in memories[0].content

    def test_detects_domain_mismatch(self, bus, profile_registry, repo):
        debate = {
            "topic": "frontend component architecture",
            "synthesis": {
                "recommendation": "Use React component composition with CSS modules",
                "confidence": "high",
                "trade_offs": ["responsive layout complexity"],
            },
        }

        bus.on_debate_complete(
            debate_result=debate,
            repo_slug=repo,
            task_id="t1",
            original_profile_id="backend",
        )

        profile_registry.record_domain_feedback.assert_called_once_with(
            task_id="t1",
            detected_domain="frontend",
            original_profile_id="backend",
        )

    def test_no_mismatch_when_domain_matches(self, bus, profile_registry, repo):
        debate = {
            "topic": "database query optimization",
            "synthesis": {
                "recommendation": "Add index on user_id for the query endpoint",
                "confidence": "high",
                "trade_offs": ["api latency"],
            },
        }

        bus.on_debate_complete(
            debate_result=debate,
            repo_slug=repo,
            task_id="t1",
            original_profile_id="backend",
        )

        # Should detect "backend" domain, same as original → no mismatch
        profile_registry.record_domain_feedback.assert_not_called()

    def test_logs_session_events(self, bus, session_logger, repo):
        debate = {
            "topic": "Test topic",
            "synthesis": {
                "recommendation": "Do X",
                "confidence": "medium",
            },
        }

        bus.on_debate_complete(debate_result=debate, repo_slug=repo, task_id="t1")

        # Should log feedback_debate_stored
        calls = [c[0][0] for c in session_logger.log.call_args_list]
        assert "feedback_debate_stored" in calls

    def test_noop_without_synthesis(self, bus, store, repo):
        debate = {"topic": "Test", "synthesis": {}}
        bus.on_debate_complete(debate_result=debate, repo_slug=repo)
        assert store.recall(repo, "shared", category="architectural_decisions") == []


class TestHelpers:
    def test_extract_tags_from_task(self):
        task = _make_task()
        tags = _extract_tags_from_task(task)
        assert "implementation" in tags
        assert ".py" in tags

    def test_file_to_pattern(self):
        assert _file_to_pattern("src/main.py") == "*.py"
        assert _file_to_pattern("Dockerfile") == "*"
        assert _file_to_pattern("") == "*"
        assert _file_to_pattern("styles/app.css") == "*.css"

    def test_detect_domain_backend(self):
        assert _detect_domain_from_text("database query optimization for the API endpoint") == "backend"

    def test_detect_domain_frontend(self):
        assert _detect_domain_from_text("react component with responsive CSS layout") == "frontend"

    def test_detect_domain_infra(self):
        assert _detect_domain_from_text("terraform docker kubernetes deployment") == "infrastructure"

    def test_detect_domain_insufficient_keywords(self):
        # Single keyword match shouldn't trigger
        assert _detect_domain_from_text("the API is slow") is None

    def test_qa_finding_matches_positive(self):
        content = "[HIGH] correctness in *.py: Missing error handling for null input"
        assert _qa_finding_matches(content, "correctness", "*.py", "Missing error handling for null input")

    def test_qa_finding_matches_negative(self):
        content = "[HIGH] security in *.js: XSS vulnerability"
        assert not _qa_finding_matches(content, "correctness", "*.py", "Missing error handling")
