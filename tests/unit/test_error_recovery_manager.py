"""Tests for error recovery and replanning functionality in the Agent."""

import pytest
from datetime import datetime, timezone

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument, RetryAttempt


def test_retry_attempt_context_snapshot_includes_plan():
    """Test that RetryAttempt.context_snapshot includes plan information when available."""
    plan = PlanDocument(
        objectives=["Test objective"],
        approach=["Step 1: Do this", "Step 2: Do that"],
        files_to_modify=["file1.py", "file2.py"],
        success_criteria=["Tests pass"],
    )

    task = Task(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Test task",
        description="Test description",
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        plan=plan,
    )

    # Mark task as failed - this should capture plan info in context_snapshot
    task.mark_failed("engineer", error_message="Test error", error_type="test_error")

    # Check that retry attempt was created with plan info
    assert len(task.retry_attempts) == 1
    attempt = task.retry_attempts[0]

    assert isinstance(attempt, RetryAttempt)
    assert attempt.error_type == "test_error"
    assert "plan_approach" in attempt.context_snapshot
    assert attempt.context_snapshot["plan_approach"] == ["Step 1: Do this", "Step 2: Do that"]
    assert "plan_files_to_modify" in attempt.context_snapshot
    assert attempt.context_snapshot["plan_files_to_modify"] == ["file1.py", "file2.py"]


def test_retry_attempt_context_snapshot_without_plan():
    """Test that RetryAttempt.context_snapshot works without plan."""
    task = Task(
        id="test-task-no-plan",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Test task without plan",
        description="Test description",
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        # No plan
    )

    # Mark task as failed
    task.mark_failed("engineer", error_message="Test error", error_type="validation_error")

    # Check that retry attempt was created without plan info (shouldn't crash)
    assert len(task.retry_attempts) == 1
    attempt = task.retry_attempts[0]

    assert attempt.error_type == "validation_error"
    assert attempt.context_snapshot["task_type"] == "implementation"
    # Should not have plan fields
    assert "plan_approach" not in attempt.context_snapshot
    assert "plan_files_to_modify" not in attempt.context_snapshot


def test_retry_attempt_preserves_standard_context():
    """Test that RetryAttempt still includes standard context fields."""
    plan = PlanDocument(
        objectives=["Test"],
        approach=["Step 1"],
        success_criteria=["Done"],
    )

    task = Task(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Test task",
        description="Test",
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        plan=plan,
        depends_on=["dep-1", "dep-2"],
    )

    task.mark_failed("engineer", "Error", "error_type")

    attempt = task.retry_attempts[0]
    snapshot = attempt.context_snapshot

    # Standard fields should still be present
    assert snapshot["task_type"] == "implementation"
    assert snapshot["assigned_to"] == "engineer"
    assert snapshot["has_dependencies"] is True  # has 2 dependencies
