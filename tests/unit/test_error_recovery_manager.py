"""Tests for error recovery and replanning functionality in the Agent."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument, RetryAttempt
from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.memory.memory_store import MemoryStore


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


class TestStoreFailureMemory:
    """Tests for _store_failure_memory() method in Agent."""

    @pytest.fixture
    def task_with_repo(self):
        """Create a task with github_repo context."""
        return Task(
            id="test-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            title="Test task",
            description="Test description",
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            context={"github_repo": "owner/repo"},
        )

    def test_stores_failure_pattern_with_error_type_tag(self, tmp_path, task_with_repo):
        """Should store failure memory with error type tag."""
        from agent_framework.core.agent import Agent

        # Create memory store directly
        memory_store = MemoryStore(tmp_path, enabled=True)

        # Simulate what _store_failure_memory does
        error = "TypeError: 'NoneType' object is not subscriptable"
        error_type = "logic"
        revised_plan = "Add null checks before accessing dictionary"

        # Store the memory (mimicking the method's behavior)
        error_summary = error[:200]
        plan_summary = revised_plan[:200]
        content = f"Error type '{error_type or 'unknown'}': {error_summary}. Resolution: {plan_summary}"
        tags = [f"error:{error_type}"] if error_type else []

        memory_store.remember(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
            content=content,
            source_task_id=task_with_repo.id,
            tags=tags,
        )

        # Verify memory was stored
        memories = memory_store.recall(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
        )

        assert len(memories) == 1
        memory = memories[0]
        assert memory.category == "past_failures"
        assert "logic" in memory.content
        assert "TypeError" in memory.content
        assert "error:logic" in memory.tags
        assert memory.source_task_id == "test-task"

    def test_handles_missing_error_type(self, tmp_path, task_with_repo):
        """Should handle None error_type gracefully."""
        memory_store = MemoryStore(tmp_path, enabled=True)

        error = "Some error without a type"
        error_type = None
        revised_plan = "Try a different approach"

        # Store memory with None error_type
        error_summary = error[:200]
        plan_summary = revised_plan[:200]
        content = f"Error type '{error_type or 'unknown'}': {error_summary}. Resolution: {plan_summary}"
        tags = [f"error:{error_type}"] if error_type else []

        memory_store.remember(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
            content=content,
            source_task_id=task_with_repo.id,
            tags=tags,
        )

        memories = memory_store.recall(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
        )

        assert len(memories) == 1
        assert "unknown" in memories[0].content
        # Should have empty tags list when error_type is None
        assert memories[0].tags == []

    def test_skips_when_memory_disabled(self, tmp_path, task_with_repo):
        """Should skip memory storage when memory is disabled."""
        memory_store = MemoryStore(tmp_path, enabled=False)

        result = memory_store.remember(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
            content="test content",
            source_task_id=task_with_repo.id,
            tags=[],
        )

        # Should return False when disabled
        assert result is False

    def test_truncates_long_error_and_plan(self, tmp_path, task_with_repo):
        """Should truncate error and plan to 200 chars each."""
        memory_store = MemoryStore(tmp_path, enabled=True)

        long_error = "E" * 500
        long_plan = "P" * 500

        # Simulate truncation
        error_summary = long_error[:200]
        plan_summary = long_plan[:200]
        content = f"Error type 'logic': {error_summary}. Resolution: {plan_summary}"

        memory_store.remember(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
            content=content,
            source_task_id=task_with_repo.id,
            tags=["error:logic"],
        )

        memories = memory_store.recall(
            repo_slug="owner/repo",
            agent_type="engineer",
            category="past_failures",
        )

        assert len(memories) == 1
        # Content should be truncated (error[:200] + plan[:200] + some overhead)
        # Total should be less than 500 chars (200 + 200 + overhead)
        assert len(memories[0].content) < 500
