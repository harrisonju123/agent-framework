"""Escalation handler for failed tasks (ported from Bash system)."""

import time
from datetime import datetime

from ..core.task import Task, TaskStatus, TaskType


class EscalationHandler:
    """
    Handles task escalations for human review.

    Ported from scripts/async-agent-runner.sh lines 205-246.

    CRITICAL SAFETY:
    - Escalation tasks with type "escalation" CANNOT create more escalations
    - This is the single most important safeguard against infinite loops
    - If an escalation fails 5 times, it logs for human intervention only
    """

    def __init__(self, product_owner_queue: str = "product-owner"):
        self.product_owner_queue = product_owner_queue

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
                "original_task_type": failed_task.type.value,
                "retry_count": failed_task.retry_count,
            },
        )

        return escalation

    def _build_description(self, failed_task: Task) -> str:
        """Build escalation description."""
        return (
            f"Task {failed_task.id} failed after {failed_task.retry_count} retry attempts "
            f"and has been marked as failed. This requires human intervention or product decision.\n\n"
            f"Original task: {failed_task.title}\n"
            f"Task type: {failed_task.type.value}\n\n"
            f"Please review the failed task and decide next steps."
        )
