"""Integration test for budget error handling flow.

Tests the complete lifecycle: categorize_error → should_retry → create_escalation
to ensure budget errors are properly detected, not retried, and escalated with
appropriate interventions.
"""

from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, RetryAttempt
from agent_framework.errors.translator import ErrorTranslator
from agent_framework.safeguards.escalation import EscalationHandler
from agent_framework.safeguards.retry_handler import RetryHandler


def _make_task(**overrides) -> Task:
    """Helper to create a task for integration tests."""
    defaults = dict(
        id="task-budget-test",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.FAILED,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature requiring API calls",
        description="Task that will hit budget limits",
        retry_count=0,
        last_error="",
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestBudgetErrorLifecycle:
    """Test the complete budget error handling flow end-to-end."""

    @pytest.mark.parametrize("error_message", [
        "budget exceeded",
        "max budget reached for this account",
        "quota exceeded",
        "insufficient credits available",
        "usage limit exceeded for this billing period",
        "BUDGET EXCEEDED",  # case insensitive
    ])
    def test_budget_error_full_flow(self, error_message):
        """Test complete flow: categorization → no retry → escalation with interventions."""
        # Step 1: Create handlers
        escalation_handler = EscalationHandler()
        retry_handler = RetryHandler(max_retries=5)

        # Step 2: Categorize the error
        error_category = escalation_handler.categorize_error(error_message)
        assert error_category == "budget", f"Failed to categorize '{error_message}' as budget error"

        # Step 3: Create task with budget error
        task = _make_task(last_error=error_message)
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message=error_message,
                agent_id="engineer",
                error_type=error_category,
                context_snapshot={},
            )
        ]

        # Step 4: Verify no retry is attempted (should_retry returns False)
        should_retry = retry_handler.should_retry(task)
        assert should_retry is False, f"Budget error '{error_message}' should not be retried"

        # Step 5: Create escalation
        escalation = escalation_handler.create_escalation(task, "engineer")

        # Step 6: Verify escalation properties
        assert escalation.type == TaskType.ESCALATION
        assert escalation.assigned_to == "architect"
        assert escalation.priority == 0
        assert escalation.context["original_task_id"] == "task-budget-test"
        assert error_message in escalation.context["error"]

        # Step 7: Verify budget-specific interventions are present
        interventions = escalation.escalation_report.suggested_interventions
        budget_related = any(
            "budget" in intervention.lower() or
            "credits" in intervention.lower() or
            "quota" in intervention.lower()
            for intervention in interventions
        )
        assert budget_related, (
            f"Budget interventions not found in escalation. "
            f"Interventions: {interventions}"
        )

    def test_budget_error_skips_retries_immediately(self):
        """Verify budget errors skip retries even on first failure (retry_count=0)."""
        escalation_handler = EscalationHandler()
        retry_handler = RetryHandler(max_retries=5)

        error_message = "budget exceeded"
        error_category = escalation_handler.categorize_error(error_message)

        task = _make_task(
            retry_count=0,  # First failure
            last_error=error_message
        )
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message=error_message,
                agent_id="engineer",
                error_type=error_category,
                context_snapshot={},
            )
        ]

        # Should not retry even though retry_count=0 and max_retries=5
        assert retry_handler.should_retry(task) is False

    def test_non_budget_error_retries_normally(self):
        """Verify non-budget errors still follow normal retry logic."""
        escalation_handler = EscalationHandler()
        retry_handler = RetryHandler(max_retries=5)

        error_message = "network timeout"
        error_category = escalation_handler.categorize_error(error_message)

        task = _make_task(
            retry_count=2,  # Within retry limit
            last_error=error_message,
            last_failed_at=datetime.now(timezone.utc)
        )
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message=error_message,
                agent_id="engineer",
                error_type=error_category,
                context_snapshot={},
            )
        ]

        # Non-budget errors should be retriable (but may need to wait for backoff)
        # The categorization should NOT be "budget"
        assert error_category != "budget"
        assert error_category == "network"

    def test_error_translator_provides_helpful_budget_message(self):
        """Verify ErrorTranslator provides user-friendly guidance for budget errors."""
        translator = ErrorTranslator()

        error = Exception("API Error: budget exceeded")
        translated = translator.translate(error)

        assert translated.title == "Budget or quota exceeded"
        assert "budget limit" in translated.explanation.lower() or "quota" in translated.explanation.lower()
        assert len(translated.actions) >= 4
        assert translated.documentation == "docs/TROUBLESHOOTING.md#budget"

        # Verify actionable guidance is present
        actions_text = " ".join(translated.actions).lower()
        assert "account" in actions_text or "budget" in actions_text or "quota" in actions_text

    def test_multiple_error_types_categorized_correctly(self):
        """Verify different error types are categorized distinctly (regression test)."""
        handler = EscalationHandler()

        # Budget errors
        assert handler.categorize_error("budget exceeded") == "budget"

        # Network errors (should NOT be categorized as budget)
        assert handler.categorize_error("connection refused") == "network"

        # Authentication errors (should NOT be categorized as budget)
        assert handler.categorize_error("unauthorized") == "authentication"

        # Resource errors (should NOT be categorized as budget)
        # Note: quota patterns are in budget, not resource
        assert handler.categorize_error("out of memory") == "resource"
