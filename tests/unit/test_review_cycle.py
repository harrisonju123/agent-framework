"""Tests for ReviewCycleManager class."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock
from pathlib import Path

import pytest

from agent_framework.core.review_cycle import (
    ReviewCycleManager,
    QAFinding,
    ReviewOutcome,
    MAX_REVIEW_CYCLES,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def mock_config():
    """Create a mock AgentConfig."""
    config = Mock()
    config.id = "qa"
    config.base_id = "qa"
    return config


@pytest.fixture
def mock_queue(tmp_path):
    """Create a mock FileQueue."""
    queue = Mock()
    queue.queue_dir = tmp_path / "queue"
    queue.completed_dir = tmp_path / "completed"
    queue.queue_dir.mkdir(parents=True)
    queue.completed_dir.mkdir(parents=True)
    return queue


@pytest.fixture
def review_cycle_manager(mock_config, mock_queue):
    """Create a ReviewCycleManager instance for testing."""
    logger = MagicMock()
    return ReviewCycleManager(
        config=mock_config,
        queue=mock_queue,
        logger=logger,
        agent_definition=None,
        session_logger=None,
        activity_manager=None,
    )


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test-task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test Task",
        description="Test description",
        context={
            "jira_key": "PROJ-42",
            "workflow": "default",
        },
    )


@pytest.fixture
def sample_response():
    """Create a sample response object."""
    response = Mock()
    response.content = "PR created: https://github.com/org/repo/pull/123"
    return response


class TestExtractPrInfo:
    """Tests for PR information extraction."""

    def test_extracts_pr_from_url(self, review_cycle_manager):
        """Should extract PR info from GitHub URL."""
        content = "Created PR: https://github.com/owner/repo/pull/456"
        result = review_cycle_manager.extract_pr_info_from_response(content)

        assert result is not None
        assert result["pr_url"] == "https://github.com/owner/repo/pull/456"
        assert result["pr_number"] == 456
        assert result["owner"] == "owner"
        assert result["repo"] == "repo"
        assert result["github_repo"] == "owner/repo"

    def test_returns_none_for_no_pr(self, review_cycle_manager):
        """Should return None when no PR URL found."""
        content = "Task completed successfully"
        result = review_cycle_manager.extract_pr_info_from_response(content)
        assert result is None

    def test_get_pr_info_from_response(self, review_cycle_manager, sample_task, sample_response):
        """Should get PR info from response content."""
        result = review_cycle_manager.get_pr_info(sample_task, sample_response)

        assert result is not None
        assert result["pr_number"] == 123
        assert result["github_repo"] == "org/repo"

    def test_get_pr_info_from_task_context(self, review_cycle_manager, sample_task, sample_response):
        """Should get PR info from task context when not in response."""
        sample_response.content = "No PR URL here"
        sample_task.context["pr_url"] = "https://github.com/test/repo/pull/99"

        result = review_cycle_manager.get_pr_info(sample_task, sample_response)

        assert result is not None
        assert result["pr_number"] == 99
        assert result["github_repo"] == "test/repo"


class TestParseReviewOutcome:
    """Tests for review outcome parsing."""

    def test_parses_approved(self, review_cycle_manager):
        """Should detect APPROVE keyword."""
        content = "All looks good! APPROVE"
        outcome = review_cycle_manager.parse_review_outcome(content)

        assert outcome.approved is True
        assert outcome.has_critical_issues is False
        assert outcome.needs_fix is False

    def test_parses_critical_issues(self, review_cycle_manager):
        """Should detect CRITICAL issues."""
        content = "CRITICAL: Security vulnerability found"
        outcome = review_cycle_manager.parse_review_outcome(content)

        assert outcome.approved is False
        assert outcome.has_critical_issues is True
        assert outcome.needs_fix is True

    def test_critical_overrides_approve(self, review_cycle_manager):
        """Should treat CRITICAL as not approved even with APPROVE keyword."""
        content = "APPROVE but CRITICAL: Major issue found"
        outcome = review_cycle_manager.parse_review_outcome(content)

        assert outcome.approved is False
        assert outcome.has_critical_issues is True
        assert outcome.needs_fix is True

    def test_detects_test_failures(self, review_cycle_manager):
        """Should detect test failures."""
        content = "3 tests fail"
        outcome = review_cycle_manager.parse_review_outcome(content)

        assert outcome.has_test_failures is True
        assert outcome.needs_fix is True

    def test_handles_empty_content(self, review_cycle_manager):
        """Should handle empty content gracefully."""
        outcome = review_cycle_manager.parse_review_outcome("")

        assert outcome.approved is False
        assert outcome.has_critical_issues is False
        assert outcome.findings_summary == ""


class TestParseStructuredFindings:
    """Tests for structured findings parsing."""

    def test_parses_json_code_fence(self, review_cycle_manager):
        """Should parse structured findings from JSON code fence."""
        content = '''```json
{
  "findings": [
    {
      "file": "auth.py",
      "line_number": 42,
      "severity": "CRITICAL",
      "category": "security",
      "description": "SQL injection",
      "suggested_fix": "Use parameterized queries"
    }
  ]
}
```'''
        findings = review_cycle_manager.parse_structured_findings(content)

        assert findings is not None
        assert len(findings) == 1
        assert findings[0].severity == "CRITICAL"
        assert findings[0].file == "auth.py"
        assert findings[0].line_number == 42

    def test_handles_array_format(self, review_cycle_manager):
        """Should handle direct array format."""
        content = '''```json
[
  {
    "file": "test.py",
    "line_number": 10,
    "severity": "HIGH",
    "category": "performance",
    "description": "Slow query"
  }
]
```'''
        findings = review_cycle_manager.parse_structured_findings(content)

        assert findings is not None
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"

    def test_returns_none_for_non_json(self, review_cycle_manager):
        """Should return None for non-JSON content."""
        content = "CRITICAL: Issue found"
        findings = review_cycle_manager.parse_structured_findings(content)
        assert findings is None


class TestFormatFindingsChecklist:
    """Tests for findings checklist formatting."""

    def test_formats_single_finding(self, review_cycle_manager):
        """Should format a single finding with location."""
        findings = [
            QAFinding(
                file="test.py",
                line_number=10,
                severity="HIGH",
                category="security",
                description="Issue here",
                suggested_fix="Fix it",
            )
        ]
        result = review_cycle_manager.format_findings_checklist(findings)

        assert "1. 🟠 HIGH: Security (test.py:10)" in result
        assert "**Issue**: Issue here" in result
        assert "**Suggested Fix**: Fix it" in result

    def test_formats_multiple_findings(self, review_cycle_manager):
        """Should number multiple findings correctly."""
        findings = [
            QAFinding(
                file="a.py",
                line_number=1,
                severity="CRITICAL",
                category="security",
                description="Issue 1",
                suggested_fix=None,
            ),
            QAFinding(
                file="b.py",
                line_number=2,
                severity="HIGH",
                category="performance",
                description="Issue 2",
                suggested_fix="Fix 2",
            ),
        ]
        result = review_cycle_manager.format_findings_checklist(findings)

        assert "1. 🔴 CRITICAL" in result
        assert "2. 🟠 HIGH" in result


class TestBuildReviewTask:
    """Tests for review task building."""

    def test_builds_review_task(self, review_cycle_manager, sample_task):
        """Should build a complete review task."""
        pr_info = {
            "pr_url": "https://github.com/org/repo/pull/123",
            "pr_number": 123,
            "github_repo": "org/repo",
        }

        review_task = review_cycle_manager.build_review_task(sample_task, pr_info)

        assert review_task.type == TaskType.REVIEW
        assert review_task.assigned_to == "qa"
        assert review_task.context["pr_number"] == 123
        assert review_task.context["pr_url"] == pr_info["pr_url"]
        assert "Review PR #123" in review_task.title


class TestBuildReviewFixTask:
    """Tests for review fix task building."""

    def test_builds_fix_task_with_structured_findings(self, review_cycle_manager):
        """Should build fix task with structured findings."""
        task = Task(
            id="review-task-abc",
            type=TaskType.REVIEW,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="qa",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title="Review",
            description="",
            context={
                "jira_key": "PROJ-42",
                "pr_url": "https://github.com/org/repo/pull/99",
                "pr_number": 99,
            },
        )

        findings = [
            QAFinding(
                file="auth.py",
                line_number=42,
                severity="CRITICAL",
                category="security",
                description="SQL injection",
                suggested_fix="Use ORM",
            )
        ]

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            findings_summary="Critical issues found",
            structured_findings=findings,
        )

        fix_task = review_cycle_manager.build_review_fix_task(task, outcome, cycle_count=1)

        assert fix_task.type == TaskType.FIX
        assert fix_task.assigned_to == "engineer"
        assert fix_task.context["_review_cycle_count"] == 1
        assert "structured_findings" in fix_task.context
        assert fix_task.context["structured_findings"]["total_count"] == 1
        assert "### 1. 🔴 CRITICAL" in fix_task.description

    def test_builds_fix_task_legacy_format(self, review_cycle_manager):
        """Should build fix task in legacy format when no structured findings."""
        task = Task(
            id="review-task-abc",
            type=TaskType.REVIEW,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="qa",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title="Review",
            description="",
            context={
                "jira_key": "PROJ-42",
                "pr_url": "https://github.com/org/repo/pull/99",
                "pr_number": 99,
            },
        )

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            findings_summary="CRITICAL: Issue found",
        )

        fix_task = review_cycle_manager.build_review_fix_task(task, outcome, cycle_count=1)

        assert "## Review Findings" in fix_task.description
        assert "structured_findings" not in fix_task.context


class TestEscalateReviewToArchitect:
    """Tests for review escalation."""

    def test_escalates_after_max_cycles(self, review_cycle_manager):
        """Should create escalation task."""
        task = Task(
            id="review-task-abc",
            type=TaskType.REVIEW,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="qa",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title="Review",
            description="",
            context={
                "jira_key": "PROJ-42",
                "pr_url": "https://github.com/org/repo/pull/99",
            },
        )

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            findings_summary="Still has issues",
        )

        # Mock the queue.push method
        review_cycle_manager.queue.push = MagicMock()

        review_cycle_manager.escalate_review_to_architect(task, outcome, cycle_count=4)

        # Verify escalation task was queued
        review_cycle_manager.queue.push.assert_called_once()
        escalation_task = review_cycle_manager.queue.push.call_args[0][0]
        assert escalation_task.type == TaskType.ESCALATION
        assert escalation_task.assigned_to == "architect"
        assert "escalation_reason" in escalation_task.context


class TestQueueCodeReviewIfNeeded:
    """Tests for automatic code review queueing."""

    def test_queues_review_when_pr_created(self, review_cycle_manager, sample_task, sample_response, tmp_path):
        """Should queue review task when PR is created."""
        # Set up non-QA agent
        review_cycle_manager.config.base_id = "engineer"
        review_cycle_manager.queue.push = MagicMock()
        review_cycle_manager.queue.queue_dir = tmp_path / "queue"
        (review_cycle_manager.queue.queue_dir / "qa").mkdir(parents=True)

        review_cycle_manager.queue_code_review_if_needed(sample_task, sample_response)

        # Verify review task was queued
        review_cycle_manager.queue.push.assert_called_once()
        review_task = review_cycle_manager.queue.push.call_args[0][0]
        assert review_task.type == TaskType.REVIEW
        assert review_task.assigned_to == "qa"

    def test_skips_review_for_qa_agent(self, review_cycle_manager, sample_task, sample_response):
        """Should skip queueing review when agent is QA."""
        review_cycle_manager.config.base_id = "qa"
        review_cycle_manager.queue.push = MagicMock()

        review_cycle_manager.queue_code_review_if_needed(sample_task, sample_response)

        # Should not queue anything
        review_cycle_manager.queue.push.assert_not_called()

    def test_skips_review_when_max_cycles_reached(self, review_cycle_manager, sample_task, sample_response):
        """Should skip review when max review cycles reached."""
        review_cycle_manager.config.base_id = "engineer"
        sample_task.context["_review_cycle_count"] = MAX_REVIEW_CYCLES
        review_cycle_manager.queue.push = MagicMock()

        review_cycle_manager.queue_code_review_if_needed(sample_task, sample_response)

        # Should not queue
        review_cycle_manager.queue.push.assert_not_called()


class TestQueueReviewFixIfNeeded:
    """Tests for automatic fix task queueing."""

    def test_queues_fix_when_issues_found(self, review_cycle_manager, tmp_path):
        """Should queue fix task when QA finds issues."""
        review_cycle_manager.config.base_id = "qa"
        review_cycle_manager.queue.push = MagicMock()
        review_cycle_manager.queue.queue_dir = tmp_path / "queue"
        (review_cycle_manager.queue.queue_dir / "engineer").mkdir(parents=True)

        task = Task(
            id="review-task-abc",
            type=TaskType.REVIEW,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="engineer",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title="Review",
            description="",
            context={
                "jira_key": "PROJ-42",
                "pr_url": "https://github.com/org/repo/pull/99",
                "pr_number": 99,
                "_review_cycle_count": 0,
            },
        )

        response = Mock()
        response.content = "CRITICAL: Issues found"

        review_cycle_manager.queue_review_fix_if_needed(task, response)

        # Verify fix task was queued
        review_cycle_manager.queue.push.assert_called_once()
        fix_task = review_cycle_manager.queue.push.call_args[0][0]
        assert fix_task.type == TaskType.FIX
        assert fix_task.assigned_to == "engineer"

    def test_does_not_queue_when_approved(self, review_cycle_manager):
        """Should not queue fix task when approved."""
        review_cycle_manager.config.base_id = "qa"
        review_cycle_manager.queue.push = MagicMock()

        task = Task(
            id="review-task-abc",
            type=TaskType.REVIEW,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="engineer",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title="Review",
            description="",
            context={"jira_key": "PROJ-42"},
        )

        response = Mock()
        response.content = "APPROVE"

        sync_jira_fn = MagicMock()
        review_cycle_manager.queue_review_fix_if_needed(task, response, sync_jira_fn=sync_jira_fn)

        # Should not queue fix task
        review_cycle_manager.queue.push.assert_not_called()
        # But should sync JIRA
        sync_jira_fn.assert_called_once()
