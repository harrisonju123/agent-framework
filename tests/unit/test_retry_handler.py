"""Tests for retry handler — verifies non-retriable error handling and backoff logic."""

from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, RetryAttempt
from agent_framework.safeguards.retry_handler import RetryHandler


def _make_task(**overrides) -> Task:
    defaults = dict(
        id="task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Some description",
        retry_count=0,
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestNonRetriableErrors:
    def test_budget_error_not_retriable(self):
        """Budget errors should never be retried."""
        handler = RetryHandler()
        task = _make_task(retry_count=0)
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

        assert handler.should_retry(task) is False

    def test_authentication_error_not_retriable(self):
        """Authentication errors should never be retried."""
        handler = RetryHandler()
        task = _make_task(retry_count=0)
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message="unauthorized",
                agent_id="engineer",
                error_type="authentication",
                context_snapshot={},
            )
        ]

        assert handler.should_retry(task) is False

    def test_budget_error_not_retriable_even_with_low_retry_count(self):
        """Budget errors are not retried even on first failure."""
        handler = RetryHandler(max_retries=5)
        task = _make_task(retry_count=0)
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message="insufficient credits",
                agent_id="engineer",
                error_type="budget",
                context_snapshot={},
            )
        ]

        assert handler.should_retry(task) is False

    def test_retriable_error_respects_max_retries(self):
        """Non-budget/auth errors should respect max_retries."""
        handler = RetryHandler(max_retries=3)
        task = _make_task(retry_count=3)
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message="network timeout",
                agent_id="engineer",
                error_type="network",
                context_snapshot={},
            )
        ]

        assert handler.should_retry(task) is False

    def test_retriable_error_retries_within_limit(self):
        """Non-budget/auth errors should retry within max_retries."""
        handler = RetryHandler(max_retries=5)
        task = _make_task(retry_count=2)
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=datetime.now(timezone.utc),
                error_message="network timeout",
                agent_id="engineer",
                error_type="network",
                context_snapshot={},
            )
        ]

        assert handler.should_retry(task) is True

    def test_no_retry_attempts_falls_back_to_count(self):
        """If no retry_attempts, should use retry_count logic."""
        handler = RetryHandler(max_retries=5)
        task = _make_task(retry_count=2)
        # No retry_attempts set

        assert handler.should_retry(task) is True


class TestBackoffCalculation:
    def test_calculate_backoff_first_retry(self):
        handler = RetryHandler(initial_backoff=30, multiplier=2)
        assert handler.calculate_backoff(1) == 30

    def test_calculate_backoff_exponential(self):
        handler = RetryHandler(initial_backoff=30, multiplier=2)
        assert handler.calculate_backoff(2) == 60
        assert handler.calculate_backoff(3) == 120

    def test_calculate_backoff_respects_max(self):
        handler = RetryHandler(initial_backoff=30, multiplier=2, max_backoff=100)
        assert handler.calculate_backoff(4) == 100  # Would be 240 without max

    def test_should_retry_respects_backoff_time(self):
        """Task shouldn't retry until backoff time has passed."""
        from datetime import timedelta

        handler = RetryHandler(initial_backoff=60, multiplier=2, max_retries=5)
        now = datetime.now(timezone.utc)
        task = _make_task(
            retry_count=1,
            last_failed_at=now,
        )
        task.retry_attempts = [
            RetryAttempt(
                attempt_number=1,
                timestamp=now,
                error_message="network timeout",
                agent_id="engineer",
                error_type="network",
                context_snapshot={},
            )
        ]

        # Mock time hasn't passed enough
        assert handler.should_retry(task) is False

        # Simulate time passing
        task.last_failed_at = now - timedelta(seconds=70)
        assert handler.should_retry(task) is True
