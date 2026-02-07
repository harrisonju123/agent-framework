"""Tests for escalation handler — verifies routing to architect after consolidation."""

from datetime import datetime

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
        created_at=datetime.utcnow(),
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
