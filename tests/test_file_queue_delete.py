"""Tests for FileQueue.delete_task() — permanent file removal."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.queue.file_queue import FileQueue


def _make_task(task_id="task-1", status=TaskStatus.PENDING, assigned_to="engineer"):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=1,
        created_by="test",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test description.",
    )


@pytest.fixture
def queue(tmp_path):
    return FileQueue(tmp_path)


class TestDeleteTask:
    def test_deletes_task_from_queue(self, queue):
        task = _make_task()
        queue.push(task, "engineer")

        task_file = queue.queue_dir / "engineer" / "task-1.json"
        assert task_file.exists()

        assert queue.delete_task("task-1") is True
        assert not task_file.exists()

    def test_deletes_task_from_completed(self, queue):
        task = _make_task(status=TaskStatus.COMPLETED)
        task_file = queue.completed_dir / "task-1.json"
        task_file.write_text(task.model_dump_json())

        assert queue.delete_task("task-1") is True
        assert not task_file.exists()

    def test_returns_false_for_nonexistent(self, queue):
        assert queue.delete_task("nonexistent") is False

    def test_cleans_up_lock_directory(self, queue):
        task = _make_task()
        queue.push(task, "engineer")

        # Create a lock directory
        lock_path = queue.lock_dir / "task-1.lock"
        lock_path.mkdir(parents=True)

        assert queue.delete_task("task-1") is True
        assert not lock_path.exists()

    def test_clears_non_pending_cache(self, queue):
        task = _make_task(status=TaskStatus.FAILED)
        queue.push(task, "engineer")

        # Warm up the cache by calling pop (which marks non-pending)
        queue.pop("engineer")
        assert "task-1.json" in queue._non_pending_files.get("engineer", set())

        queue.delete_task("task-1")
        assert "task-1.json" not in queue._non_pending_files.get("engineer", set())

    def test_clears_completed_cache(self, queue):
        """Completed dependency cache should be invalidated on delete."""
        task = _make_task(status=TaskStatus.COMPLETED)
        task_file = queue.completed_dir / "task-1.json"
        task_file.write_text(task.model_dump_json())

        # Seed the cache
        queue._completed_cache["task-1"] = (True, 999999999)

        queue.delete_task("task-1")
        assert "task-1" not in queue._completed_cache

    def test_invalidates_queue_dir_mtime(self, queue):
        task = _make_task()
        queue.push(task, "engineer")

        # Seed a stale mtime
        queue._queue_dir_mtime["engineer"] = 12345.0

        queue.delete_task("task-1")
        assert "engineer" not in queue._queue_dir_mtime

    def test_find_task_file_across_queues(self, queue):
        """_find_task_file locates tasks in any queue subdirectory."""
        task = _make_task(assigned_to="architect")
        queue.push(task, "architect")

        found = queue._find_task_file("task-1")
        assert found is not None
        assert found.name == "task-1.json"
        assert "architect" in str(found)

    def test_find_task_file_returns_none(self, queue):
        assert queue._find_task_file("nope") is None
