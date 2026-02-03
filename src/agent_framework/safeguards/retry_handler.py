"""Retry handler with exponential backoff (ported from Bash system)."""

from datetime import datetime

from ..core.task import Task


class RetryHandler:
    """
    Handles task retries with exponential backoff.

    Ported from scripts/async-agent-runner.sh lines 77-96, 374-394.

    Logic:
    - Backoff: initial * multiplier^(retry_count-1), max max_backoff seconds
    - After max_retries, task is marked failed
    - Failed tasks create escalation (unless already an escalation)
    - CRITICAL: Escalations CANNOT create more escalations
    """

    def __init__(
        self,
        initial_backoff: int = 30,
        max_backoff: int = 240,
        multiplier: int = 2,
        max_retries: int = 5,
    ):
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.multiplier = multiplier
        self.max_retries = max_retries

    def calculate_backoff(self, retry_count: int) -> int:
        """
        Calculate backoff time for given retry count.

        Formula: initial * multiplier^(retry_count-1), capped at max_backoff
        """
        backoff = self.initial_backoff * (self.multiplier ** (retry_count - 1))
        return min(backoff, self.max_backoff)

    def should_retry(self, task: Task) -> bool:
        """Check if task should be retried."""
        if task.retry_count >= self.max_retries:
            return False

        if task.last_failed_at:
            backoff = self.calculate_backoff(task.retry_count)
            time_since_failure = (datetime.utcnow() - task.last_failed_at).total_seconds()
            return time_since_failure >= backoff

        return True

    def can_create_escalation(self, task: Task) -> bool:
        """
        Check if an escalation can be created for a failed task.

        CRITICAL: Escalations CANNOT create more escalations.
        This prevents infinite loops.
        """
        from ..core.task import TaskType
        return task.type != TaskType.ESCALATION
