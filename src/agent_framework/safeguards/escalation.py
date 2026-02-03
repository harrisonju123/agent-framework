"""Escalation handler for failed tasks (ported from Bash system)."""

import time
from datetime import datetime

from ..core.task import Task, TaskStatus, TaskType


def _get_type_str(task_type) -> str:
    """Get string value from task type (handles both enum and string)."""
    return task_type.value if hasattr(task_type, 'value') else str(task_type)


class EscalationHandler:
    """
    Handles task escalations for human review.

    Ported from scripts/async-agent-runner.sh lines 205-246.

    CRITICAL SAFETY:
    - Escalation tasks with type "escalation" CANNOT create more escalations
    - This is the single most important safeguard against infinite loops
    - If an escalation fails 5 times, it logs for human intervention only
    """

    def __init__(self, product_owner_queue: str = "product-owner", enable_error_truncation: bool = False):
        self.product_owner_queue = product_owner_queue
        self.enable_error_truncation = enable_error_truncation

    def _truncate_error(self, error: str, max_lines: int = 35) -> str:
        """
        Intelligently truncate error messages.

        Implements Strategy 8 (Error Truncation) from the optimization plan.

        Handles multiple error formats:
        - Stack traces (preserves error type, head, tail)
        - Single-line errors (returned as-is)
        - JSON error responses (preserved)
        - Already truncated errors (not re-truncated)

        Expected savings: 3-7KB per escalation task (~50% reduction).
        """
        if not error:
            return "No error message available"

        lines = error.split('\n')

        # Don't truncate if already small enough
        if len(lines) <= max_lines:
            return error

        # Check if already truncated
        if "lines omitted" in error or "..." in error[:100]:
            return error

        # Try to find error type - search more lines to catch context before error
        error_type = ""
        for line in lines[:10]:  # Increased from 3 to 10 lines
            if any(marker in line for marker in ["Error:", "Exception:", "Traceback", "FAILED", "ERROR"]):
                error_type = line.strip()
                break

        # Keep only meaningful lines (skip empty/whitespace-only)
        meaningful_lines = [line for line in lines if line.strip()]

        # If filtering made it small enough, return it
        if len(meaningful_lines) <= max_lines:
            return '\n'.join(meaningful_lines)

        # Truncate: keep first 20 and last 10 meaningful lines
        head_lines = 20
        tail_lines = 10

        head = meaningful_lines[:head_lines]
        tail = meaningful_lines[-tail_lines:]
        omitted = len(meaningful_lines) - (head_lines + tail_lines)

        result = []
        if error_type:
            result.append(error_type)
            result.append("")  # Blank line for readability

        result.extend(head)
        result.append("")
        result.append(f"... ({omitted} lines omitted) ...")
        result.append("")
        result.extend(tail)

        return '\n'.join(result)

    def create_escalation(self, failed_task: Task, agent_id: str) -> Task:
        """
        Create an escalation task for a failed task.

        CRITICAL: This method should NEVER be called for tasks with type="escalation"
        """
        if failed_task.type == TaskType.ESCALATION:
            raise ValueError(
                "CRITICAL: Cannot create escalation for escalation task. "
                "This would cause an infinite loop."
            )

        escalation_id = f"escalation-{int(time.time())}-{failed_task.id}"

        # Truncate error if enabled (Strategy 8: Error Truncation)
        error_msg = failed_task.last_error or "Unknown error"
        if self.enable_error_truncation:
            error_msg = self._truncate_error(error_msg)

        escalation = Task(
            id=escalation_id,
            type=TaskType.ESCALATION,
            status=TaskStatus.PENDING,
            priority=0,  # Highest priority
            created_by=agent_id,
            assigned_to=self.product_owner_queue,
            created_at=datetime.utcnow(),
            title=f"ESCALATION: Task failed after {failed_task.retry_count} retries",
            description=self._build_description(failed_task),
            failed_task_id=failed_task.id,
            needs_human_review=True,
            context={
                "original_task_id": failed_task.id,
                "original_task_type": _get_type_str(failed_task.type),
                "retry_count": failed_task.retry_count,
                "error": error_msg,
            },
        )

        return escalation

    def _build_description(self, failed_task: Task) -> str:
        """Build escalation description."""
        return (
            f"Task {failed_task.id} failed after {failed_task.retry_count} retry attempts "
            f"and has been marked as failed. This requires human intervention or product decision.\n\n"
            f"Original task: {failed_task.title}\n"
            f"Task type: {_get_type_str(failed_task.type)}\n\n"
            f"Please review the failed task and decide next steps."
        )
