"""Tests for FileQueue.claim() — atomic pop + lock."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.queue.file_queue import FileQueue
from agent_framework.queue.locks import FileLock


def _make_task(task_id="task-1", status=TaskStatus.PENDING, **kwargs):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature",
        description="Build the thing.",
        context=kwargs.get("context", {}),
    )


@pytest.fixture
def queue(tmp_path):
    return FileQueue(tmp_path)


class TestClaim:
    def test_returns_task_and_lock(self, queue, tmp_path):
        """Happy path: claim returns (task, lock) for a pending task."""
        task = _make_task()
        queue.push(task, "engineer")

        result = queue.claim("engineer", "engineer-1")

        assert result is not None
        claimed_task, lock = result
        assert claimed_task.id == "task-1"
        assert claimed_task.status == TaskStatus.PENDING
        assert isinstance(lock, FileLock)
        lock.release()

    def test_skips_locked_task(self, queue, tmp_path):
        """When first task is locked, claim falls through to the next one."""
        task1 = _make_task(task_id="task-1")
        task2 = _make_task(task_id="task-2")
        queue.push(task1, "engineer")
        queue.push(task2, "engineer")

        # Pre-lock task-1
        lock1 = FileLock(queue.lock_dir, "task-1")
        assert lock1.acquire()

        result = queue.claim("engineer", "engineer-1")

        assert result is not None
        claimed_task, lock2 = result
        assert claimed_task.id == "task-2"
        lock1.release()
        lock2.release()

    def test_returns_none_on_empty_queue(self, queue):
        """Empty queue returns None."""
        result = queue.claim("engineer", "engineer-1")
        assert result is None

    def test_returns_none_when_all_locked(self, queue):
        """When all pending tasks are locked, returns None."""
        task = _make_task()
        queue.push(task, "engineer")

        lock = FileLock(queue.lock_dir, "task-1")
        assert lock.acquire()

        result = queue.claim("engineer", "engineer-1")
        assert result is None
        lock.release()

    def test_returns_none_for_nonexistent_queue(self, queue):
        """Nonexistent queue_id returns None."""
        result = queue.claim("nonexistent", "agent-1")
        assert result is None

    def test_skips_non_pending_tasks(self, queue):
        """Tasks that aren't PENDING are skipped."""
        task = _make_task(status=TaskStatus.COMPLETED)
        queue.push(task, "engineer")

        result = queue.claim("engineer", "engineer-1")
        assert result is None

    def test_respects_backoff(self, queue):
        """Tasks in backoff are skipped."""
        task = _make_task()
        task.retry_count = 3
        task.last_failed_at = datetime.now(timezone.utc)
        queue.push(task, "engineer")

        result = queue.claim("engineer", "engineer-1")
        assert result is None

    def test_recovers_orphaned_task(self, queue):
        """Orphaned in_progress tasks are recovered and can be claimed."""
        task = _make_task(status=TaskStatus.IN_PROGRESS)
        task.started_by = "dead-agent"
        queue.push(task, "engineer")

        # No heartbeat file → task is orphaned
        result = queue.claim("engineer", "engineer-1")

        assert result is not None
        claimed_task, lock = result
        assert claimed_task.id == "task-1"
        lock.release()


class TestPushCompletedGuard:
    """push() rejects tasks that already exist in completed/."""

    def test_push_rejects_already_completed_task(self, queue, caplog):
        """A task in completed/ must not be re-pushed into the queue."""
        task = _make_task(task_id="task-done")
        queue.push(task, "engineer")

        # Complete the task
        task.status = TaskStatus.COMPLETED
        queue.mark_completed(task)

        # Try pushing the same task again (simulates stale worktree sync)
        stale_copy = _make_task(task_id="task-done")
        queue.push(stale_copy, "engineer")

        # Should not appear in queue
        task_file = queue.queue_dir / "engineer" / "task-done.json"
        assert not task_file.exists()
        assert "Rejecting push for already-completed task task-done" in caplog.text

    def test_push_allows_task_after_requeue_removes_completed(self, queue):
        """requeue_task() removes from completed/ first, so push succeeds."""
        task = _make_task(task_id="task-retry")
        queue.push(task, "engineer")

        # Complete and then requeue
        task.status = TaskStatus.COMPLETED
        queue.mark_completed(task)
        queue.requeue_task(task)

        # Task should now be in the queue again
        task_file = queue.queue_dir / "engineer" / "task-retry.json"
        assert task_file.exists()
