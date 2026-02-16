"""Tests for escalation handler — verifies routing to architect after consolidation."""

from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.safeguards.escalation import EscalationHandler


def _make_failed_task(**overrides) -> Task:
    defaults = dict(
        id="task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.FAILED,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Some description",
        retry_count=5,
        last_error="Something went wrong",
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestEscalationRouting:
    def test_default_escalation_routes_to_architect(self):
        handler = EscalationHandler()
        assert handler.escalation_queue == "architect"

    def test_escalation_task_assigned_to_architect(self):
        handler = EscalationHandler()
        task = _make_failed_task()
        escalation = handler.create_escalation(task, "engineer")

        assert escalation.assigned_to == "architect"
        assert escalation.type == TaskType.ESCALATION
        assert escalation.priority == 0

    def test_custom_escalation_queue(self):
        handler = EscalationHandler(escalation_queue="custom-queue")
        task = _make_failed_task()
        escalation = handler.create_escalation(task, "qa")

        assert escalation.assigned_to == "custom-queue"

    def test_escalation_for_escalation_raises(self):
        """Escalation tasks must never create more escalations (infinite loop guard)."""
        handler = EscalationHandler()
        task = _make_failed_task(type=TaskType.ESCALATION)

        with pytest.raises(ValueError, match="infinite loop"):
            handler.create_escalation(task, "architect")

    def test_escalation_includes_original_context(self):
        handler = EscalationHandler()
        task = _make_failed_task(last_error="Connection refused")
        escalation = handler.create_escalation(task, "engineer")

        assert escalation.context["original_task_id"] == "task-123"
        assert escalation.context["retry_count"] == 5
        assert "Connection refused" in escalation.context["error"]


class TestErrorTruncation:
    def test_truncation_disabled_by_default(self):
        handler = EscalationHandler()
        long_error = "\n".join(f"line {i}" for i in range(100))
        task = _make_failed_task(last_error=long_error)
        escalation = handler.create_escalation(task, "engineer")

        # Full error preserved when truncation disabled
        assert escalation.context["error"] == long_error

    def test_truncation_enabled(self):
        handler = EscalationHandler(enable_error_truncation=True)
        long_error = "\n".join(f"line {i}" for i in range(100))
        task = _make_failed_task(last_error=long_error)
        escalation = handler.create_escalation(task, "engineer")

        assert "lines omitted" in escalation.context["error"]

    def test_short_error_not_truncated(self):
        handler = EscalationHandler(enable_error_truncation=True)
        task = _make_failed_task(last_error="short error")
        escalation = handler.create_escalation(task, "engineer")

        assert escalation.context["error"] == "short error"


class TestBudgetErrorCategorization:
    def test_categorize_budget_exceeded_error(self):
        handler = EscalationHandler()
        assert handler.categorize_error("budget exceeded") == "budget"

    def test_categorize_max_budget_error(self):
        handler = EscalationHandler()
        assert handler.categorize_error("max budget reached") == "budget"

    def test_categorize_quota_exceeded_error(self):
        handler = EscalationHandler()
        assert handler.categorize_error("quota exceeded") == "budget"

    def test_categorize_insufficient_credits_error(self):
        handler = EscalationHandler()
        assert handler.categorize_error("insufficient credits") == "budget"

    def test_categorize_usage_limit_exceeded_error(self):
        handler = EscalationHandler()
        assert handler.categorize_error("usage limit exceeded") == "budget"

    def test_budget_error_case_insensitive(self):
        handler = EscalationHandler()
        assert handler.categorize_error("BUDGET EXCEEDED") == "budget"
        assert handler.categorize_error("Budget Exceeded") == "budget"

    def test_escalation_includes_budget_intervention(self):
        """Test that budget errors get appropriate interventions in escalation."""
        handler = EscalationHandler()
        from agent_framework.core.task import RetryAttempt

        task = _make_failed_task(last_error="budget exceeded")
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message="budget exceeded",
                agent_id="engineer",
                error_type="budget",
                context_snapshot={},
            )
        ]

        escalation = handler.create_escalation(task, "engineer")

        # Check that budget-specific interventions are present
        interventions = escalation.escalation_report.suggested_interventions
        budget_related = any("budget" in intervention.lower() or "credits" in intervention.lower()
                           for intervention in interventions)
        assert budget_related, f"Budget interventions not found in: {interventions}"
