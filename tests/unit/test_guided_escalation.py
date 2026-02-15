"""Tests for guided escalation feature with structured reports."""

from datetime import datetime

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, RetryAttempt, EscalationReport
from agent_framework.safeguards.escalation import EscalationHandler


def _make_failed_task_with_attempts(**overrides) -> Task:
    """Create a failed task with retry attempts for testing."""
    defaults = dict(
        id="task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.FAILED,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.utcnow(),
        title="Implement feature X",
        description="Some description",
        retry_count=3,
        last_error="Connection refused: API endpoint not reachable",
        retry_attempts=[
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.utcnow(),
                error_message="Connection refused: API endpoint not reachable",
                agent_id="engineer",
                error_type="network",
            ),
            RetryAttempt(
                attempt_number=2,
                timestamp=datetime.utcnow(),
                error_message="Connection timeout after 30s",
                agent_id="engineer",
                error_type="network",
            ),
            RetryAttempt(
                attempt_number=3,
                timestamp=datetime.utcnow(),
                error_message="Network unreachable",
                agent_id="engineer",
                error_type="network",
            ),
        ],
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestEscalationReportGeneration:
    def test_escalation_includes_structured_report(self):
        """Escalation should include structured report with diagnostics."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        assert escalation.escalation_report is not None
        assert escalation.escalation_report.task_id == "task-123"
        assert escalation.escalation_report.total_attempts == 3
        assert len(escalation.escalation_report.attempt_history) == 3

    def test_root_cause_hypothesis_generated(self):
        """Should generate hypothesis about root cause."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        hypothesis = escalation.escalation_report.root_cause_hypothesis
        assert hypothesis is not None
        assert len(hypothesis) > 0
        # Should mention network issues since all attempts had network errors
        assert "network" in hypothesis.lower()

    def test_suggested_interventions_provided(self):
        """Should provide actionable intervention suggestions."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        interventions = escalation.escalation_report.suggested_interventions
        assert len(interventions) > 0
        # Network errors should suggest network-related interventions
        assert any("network" in i.lower() or "connectivity" in i.lower() for i in interventions)

    def test_failure_pattern_detection(self):
        """Should detect failure patterns from retry history."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        pattern = escalation.escalation_report.failure_pattern
        assert pattern is not None
        # All network errors = consistent or intermittent_network pattern
        assert pattern in ["consistent", "intermittent_network"]

    def test_attempt_history_preserved(self):
        """All retry attempts should be preserved in report."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        attempts = escalation.escalation_report.attempt_history
        assert len(attempts) == 3
        assert attempts[0].attempt_number == 1
        assert attempts[0].error_type == "network"
        assert attempts[0].agent_id == "engineer"


class TestErrorCategorization:
    def test_network_error_categorization(self):
        """Should categorize network-related errors."""
        handler = EscalationHandler()
        assert handler._categorize_error("Connection refused") == "network"
        assert handler._categorize_error("Network timeout") == "network"
        assert handler._categorize_error("Could not resolve host") == "network"

    def test_authentication_error_categorization(self):
        """Should categorize authentication-related errors."""
        handler = EscalationHandler()
        assert handler._categorize_error("Unauthorized: 401") == "authentication"
        assert handler._categorize_error("Permission denied") == "authentication"
        assert handler._categorize_error("Invalid credentials") == "authentication"

    def test_validation_error_categorization(self):
        """Should categorize validation-related errors."""
        handler = EscalationHandler()
        assert handler._categorize_error("Validation error: missing field") == "validation"
        assert handler._categorize_error("Type error: expected int") == "validation"
        assert handler._categorize_error("Schema mismatch") == "validation"

    def test_unknown_error_fallback(self):
        """Should return unknown for uncategorized errors."""
        handler = EscalationHandler()
        assert handler._categorize_error("Something weird happened") == "unknown"


class TestFailurePatternAnalysis:
    def test_consistent_failure_pattern(self):
        """Should detect consistent failure pattern."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        # All network errors
        pattern = handler._analyze_failure_pattern(task.retry_attempts)
        assert pattern in ["consistent", "intermittent_network"]

    def test_varied_failure_pattern(self):
        """Should detect varied failure pattern."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts(
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.utcnow(),
                    error_message="Connection refused",
                    agent_id="engineer",
                    error_type="network",
                ),
                RetryAttempt(
                    attempt_number=2,
                    timestamp=datetime.utcnow(),
                    error_message="Validation error",
                    agent_id="engineer",
                    error_type="validation",
                ),
                RetryAttempt(
                    attempt_number=3,
                    timestamp=datetime.utcnow(),
                    error_message="Unauthorized",
                    agent_id="engineer",
                    error_type="authentication",
                ),
            ]
        )
        pattern = handler._analyze_failure_pattern(task.retry_attempts)
        assert pattern == "varied"


class TestHumanGuidanceInjection:
    def test_guidance_added_to_escalation_report(self):
        """Human guidance should be stored in escalation report."""
        task = _make_failed_task_with_attempts()
        report = EscalationReport(
            task_id=task.id,
            original_title=task.title,
            total_attempts=3,
            attempt_history=task.retry_attempts,
            root_cause_hypothesis="Network issues",
            suggested_interventions=["Check network", "Verify endpoints"],
            human_guidance="Use backup API endpoint: https://backup.api.com",
        )

        assert report.human_guidance == "Use backup API endpoint: https://backup.api.com"

    def test_task_supports_escalation_report(self):
        """Task model should support escalation report field."""
        task = Task(
            id="test-123",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.FAILED,
            priority=1,
            created_by="engineer",
            assigned_to="engineer",
            created_at=datetime.utcnow(),
            title="Test task",
            description="Test",
        )

        # Should be able to attach escalation report
        task.escalation_report = EscalationReport(
            task_id=task.id,
            original_title=task.title,
            total_attempts=0,
            attempt_history=[],
            root_cause_hypothesis="Test hypothesis",
            suggested_interventions=["Test intervention"],
            human_guidance="Test guidance",
        )

        assert task.escalation_report.human_guidance == "Test guidance"


class TestDescriptionFormatting:
    def test_escalation_description_includes_report(self):
        """Escalation description should include formatted report."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        description = escalation.description

        # Should include key sections
        assert "Root Cause Analysis" in description
        assert "Attempt History" in description
        assert "Suggested Interventions" in description
        assert "agent guide" in description  # Should mention guide command

    def test_description_includes_attempt_details(self):
        """Description should show attempt history details."""
        handler = EscalationHandler()
        task = _make_failed_task_with_attempts()
        escalation = handler.create_escalation(task, "engineer")

        description = escalation.description

        # Should show attempt numbers
        assert "Attempt 1" in description or "attempt 1" in description
        # Should show error types
        assert "network" in description.lower()
