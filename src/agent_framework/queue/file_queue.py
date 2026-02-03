"""File-based task queue using JSON files (ported from Bash system)."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..core.task import Task, TaskStatus
from .locks import FileLock


class FileQueue:
    """
    File-based task queue using JSON files.

    Ported from scripts/async-agent-runner.sh with the following features:
    - Atomic writes using .tmp files then mv
    - Dependency checking before task retrieval
    - Exponential backoff for retries
    - mkdir-based locking
    """

    def __init__(
        self,
        workspace: Path,
        backoff_initial: int = 30,
        backoff_max: int = 240,
        backoff_multiplier: int = 2,
    ):
        self.workspace = Path(workspace)
        self.comm_dir = self.workspace / ".agent-communication"
        self.queue_dir = self.comm_dir / "queues"
        self.lock_dir = self.comm_dir / "locks"
        self.completed_dir = self.comm_dir / "completed"
        self.heartbeat_dir = self.comm_dir / "heartbeats"

        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.backoff_multiplier = backoff_multiplier

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)

    def push(self, task: Task, queue_id: str) -> None:
        """
        Add a task to a queue.

        Uses atomic write: write to .tmp then mv.
        """
        queue_path = self.queue_dir / queue_id
        queue_path.mkdir(parents=True, exist_ok=True)

        task_file = queue_path / f"{task.id}.json"
        tmp_file = queue_path / f"{task.id}.json.tmp"

        # Write to temp file
        tmp_file.write_text(task.model_dump_json(indent=2))

        # Atomic rename
        tmp_file.rename(task_file)

    def pop(self, queue_id: str) -> Optional[Task]:
        """
        Get the next available task from a queue.

        Returns None if no tasks available.
        Respects:
        - Task status (only pending)
        - Dependencies (only if all deps completed)
        - Exponential backoff (respects last_failed_at)
        """
        queue_path = self.queue_dir / queue_id
        if not queue_path.exists():
            return None

        # Sort by filename (chronological order)
        task_files = sorted(queue_path.glob("*.json"))

        for task_file in task_files:
            if not task_file.exists():
                continue

            try:
                task = self._load_task(task_file)

                # Only process pending tasks
                if task.status != TaskStatus.PENDING:
                    continue

                # Check exponential backoff
                if not self._can_retry(task):
                    continue

                # Check dependencies
                if not self._dependencies_met(task):
                    continue

                return task

            except (json.JSONDecodeError, Exception):
                continue

        return None

    def update(self, task: Task) -> None:
        """Update a task's state."""
        queue_path = self.queue_dir / task.assigned_to
        task_file = queue_path / f"{task.id}.json"

        if not task_file.exists():
            return

        tmp_file = queue_path / f"{task.id}.json.tmp"
        tmp_file.write_text(task.model_dump_json(indent=2))
        tmp_file.rename(task_file)

    def mark_completed(self, task: Task) -> None:
        """Move task to completed storage."""
        queue_path = self.queue_dir / task.assigned_to
        task_file = queue_path / f"{task.id}.json"

        completed_file = self.completed_dir / f"{task.id}.json"
        tmp_file = self.completed_dir / f"{task.id}.json.tmp"

        # Write to completed directory
        tmp_file.write_text(task.model_dump_json(indent=2))
        tmp_file.rename(completed_file)

        # Remove from queue
        if task_file.exists():
            task_file.unlink()

    def mark_failed(self, task: Task) -> None:
        """Mark task as permanently failed."""
        self.update(task)

    def acquire_lock(self, task_id: str, agent_id: str) -> Optional[FileLock]:
        """
        Acquire exclusive lock on a task.

        Returns FileLock if successful, None otherwise.
        """
        lock = FileLock(self.lock_dir, task_id)
        if lock.acquire():
            return lock
        return None

    def release_lock(self, lock: FileLock) -> None:
        """Release lock on a task."""
        lock.release()

    def get_queue_stats(self, queue_id: str) -> dict:
        """Get statistics for a queue."""
        queue_path = self.queue_dir / queue_id
        if not queue_path.exists():
            return {"count": 0, "oldest": None}

        task_files = list(queue_path.glob("*.json"))
        count = len(task_files)

        oldest = None
        if task_files:
            oldest_file = min(task_files, key=lambda f: f.stat().st_mtime)
            oldest_task = self._load_task(oldest_file)
            oldest = oldest_task.created_at

        return {"count": count, "oldest": oldest}

    def _load_task(self, task_file: Path) -> Task:
        """Load a task from a JSON file."""
        data = json.loads(task_file.read_text())
        return Task(**data)

    def _can_retry(self, task: Task) -> bool:
        """Check if task can be retried based on exponential backoff."""
        if task.retry_count == 0:
            return True

        if not task.last_failed_at:
            return True

        # Calculate backoff: initial * multiplier^(retry_count-1), max backoff_max
        backoff = self.backoff_initial * (self.backoff_multiplier ** (task.retry_count - 1))
        backoff = min(backoff, self.backoff_max)

        time_since_failure = (datetime.utcnow() - task.last_failed_at).total_seconds()
        return time_since_failure >= backoff

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in task.depends_on:
            if not dep_id:
                continue

            dep_file = self.completed_dir / f"{dep_id}.json"
            if not dep_file.exists():
                return False

        return True
