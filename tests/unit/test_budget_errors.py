"""Tests for budget error handling across escalation, retry, and translation."""

from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, RetryAttempt
from agent_framework.errors.translator import ErrorTranslator
from agent_framework.safeguards.escalation import EscalationHandler
from agent_framework.safeguards.retry_handler import RetryHandler


def _make_task(**overrides) -> Task:
    """Helper to create test tasks."""
    defaults = dict(
        id="task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test description",
        retry_count=0,
        retry_attempts=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestBudgetErrorCategorization:
    """Test that budget errors are correctly categorized by EscalationHandler."""

    @pytest.fixture
    def handler(self):
        return EscalationHandler()

    def test_budget_exceeded_categorized_as_budget(self, handler):
        error = "APIError: budget_exceeded - Maximum budget of 10000 tokens exceeded"
        assert handler.categorize_error(error) == "budget"

    def test_max_budget_categorized_as_budget(self, handler):
        error = "Error: max budget reached for this task"
        assert handler.categorize_error(error) == "budget"

    def test_quota_exceeded_categorized_as_budget(self, handler):
        error = "quota exceeded for API usage"
        assert handler.categorize_error(error) == "budget"

    def test_insufficient_credits_categorized_as_budget(self, handler):
        error = "insufficient credits remaining in account"
        assert handler.categorize_error(error) == "budget"

    def test_usage_limit_exceeded_categorized_as_budget(self, handler):
        error = "usage limit exceeded - please upgrade your plan"
        assert handler.categorize_error(error) == "budget"

    def test_budget_error_case_insensitive(self, handler):
        error = "BUDGET EXCEEDED - CONTACT BILLING"
        assert handler.categorize_error(error) == "budget"

    def test_budget_error_intervention_suggestions(self, handler):
        """Verify that budget errors get appropriate intervention suggestions."""
        task = _make_task(
            status=TaskStatus.FAILED,
            retry_count=1,
            last_error="budget exceeded",
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.now(timezone.utc),
                    agent_id="engineer",
                    error_message="budget exceeded",
                    error_type="budget",
                )
            ],
        )

        suggestions = handler._generate_suggested_interventions(task)

        # Should include budget-specific suggestions
        suggestion_text = " ".join(suggestions).lower()
        assert any(word in suggestion_text for word in ["budget", "billing", "credits", "upgrade"])


class TestBudgetErrorNonRetriable:
    """Test that budget errors are not retried."""

    @pytest.fixture
    def handler(self):
        return RetryHandler(max_retries=5)

    def test_budget_error_not_retried(self, handler):
        """Budget errors should return False for should_retry regardless of retry count."""
        task = _make_task(
            retry_count=0,
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.now(timezone.utc),
                    agent_id="engineer",
                    error_message="budget exceeded",
                    error_type="budget",
                )
            ],
        )

        assert handler.should_retry(task) is False

    def test_auth_error_not_retried(self, handler):
        """Authentication errors should also be non-retriable."""
        task = _make_task(
            retry_count=0,
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.now(timezone.utc),
                    agent_id="engineer",
                    error_message="unauthorized",
                    error_type="authentication",
                )
            ],
        )

        assert handler.should_retry(task) is False

    def test_network_error_retried(self, handler):
        """Network errors should still be retriable."""
        task = _make_task(
            retry_count=0,
            last_failed_at=datetime(2020, 1, 1, tzinfo=timezone.utc),  # Far in the past
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.now(timezone.utc),
                    agent_id="engineer",
                    error_message="connection timeout",
                    error_type="network",
                )
            ],
        )

        # Network errors should be retriable (returns True when backoff elapsed)
        assert handler.should_retry(task) is True


class TestBudgetErrorTranslation:
    """Test that ErrorTranslator provides user-friendly budget error messages."""

    @pytest.fixture
    def translator(self):
        return ErrorTranslator()

    def test_budget_exceeded_translation(self, translator):
        error = Exception("budget_exceeded: Maximum budget of 10000 tokens exceeded")
        friendly = translator.translate(error)

        assert "budget" in friendly.title.lower()
        assert len(friendly.explanation) > 0
        assert len(friendly.actions) >= 4
        assert friendly.documentation is not None

    def test_budget_error_provides_actionable_steps(self, translator):
        error = Exception("quota exceeded for API usage")
        friendly = translator.translate(error)

        actions_text = " ".join(friendly.actions).lower()
        assert any(word in actions_text for word in ["billing", "dashboard", "upgrade", "credits"])

    def test_max_budget_translation(self, translator):
        error = Exception("max budget reached")
        friendly = translator.translate(error)

        assert "budget" in friendly.title.lower()

    def test_insufficient_credits_translation(self, translator):
        error = Exception("insufficient credits remaining")
        friendly = translator.translate(error)

        assert "budget" in friendly.title.lower()

    def test_usage_limit_translation(self, translator):
        error = Exception("usage limit exceeded - upgrade your plan")
        friendly = translator.translate(error)

        assert "budget" in friendly.title.lower()

    def test_budget_error_case_insensitive(self, translator):
        error = Exception("BUDGET EXCEEDED")
        friendly = translator.translate(error)

        assert "budget" in friendly.title.lower()

    def test_budget_error_preserves_original_exception(self, translator):
        original = Exception("budget_exceeded")
        friendly = translator.translate(original)

        assert friendly.original_error is original

    def test_budget_error_links_to_docs(self, translator):
        error = Exception("budget exceeded")
        friendly = translator.translate(error)

        assert friendly.documentation is not None
        assert "TROUBLESHOOTING" in friendly.documentation or "budget" in friendly.documentation.lower()


class TestBudgetErrorIntegration:
    """Integration tests for budget error handling across components."""

    def test_budget_error_full_flow(self):
        """Test complete flow: categorization -> non-retry -> escalation with interventions."""
        # 1. Categorize the error
        escalation_handler = EscalationHandler()
        error_msg = "APIError: budget_exceeded - Maximum budget exceeded"
        category = escalation_handler.categorize_error(error_msg)
        assert category == "budget"

        # 2. Create task with budget error
        task = _make_task(
            status=TaskStatus.FAILED,
            retry_count=1,
            last_error=error_msg,
            retry_attempts=[
                RetryAttempt(
                    attempt_number=1,
                    timestamp=datetime.now(timezone.utc),
                    agent_id="engineer",
                    error_message=error_msg,
                    error_type="budget",
                )
            ],
        )

        # 3. Verify retry is not attempted
        retry_handler = RetryHandler()
        assert retry_handler.should_retry(task) is False

        # 4. Create escalation with interventions
        escalation = escalation_handler.create_escalation(task, "engineer")
        assert escalation.type == TaskType.ESCALATION
        assert escalation.escalation_report is not None
        assert len(escalation.escalation_report.suggested_interventions) > 0

        # 5. Verify user-friendly translation
        translator = ErrorTranslator()
        friendly = translator.translate(Exception(error_msg))
        assert "budget" in friendly.title.lower()
        assert len(friendly.actions) >= 4
