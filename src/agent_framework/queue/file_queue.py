"""File-based task queue using JSON files (ported from Bash system)."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from ..core.task import Task, TaskStatus
from .locks import FileLock

logger = logging.getLogger(__name__)


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
        self.malformed_dir = self.comm_dir / "malformed"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        self.malformed_dir.mkdir(parents=True, exist_ok=True)

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

            except FileNotFoundError:
                # File was removed between glob and read - just skip
                continue
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Malformed task file - move to malformed directory for investigation
                self._quarantine_malformed_task(task_file, e)
                continue

        return None

    def _quarantine_malformed_task(self, task_file: Path, error: Exception) -> None:
        """Move malformed task file to malformed directory for investigation."""
        try:
            # Preserve original path info in filename
            queue_name = task_file.parent.name
            dest_file = self.malformed_dir / f"{queue_name}_{task_file.name}"

            # Avoid overwriting - append timestamp if exists
            if dest_file.exists():
                timestamp = int(time.time())
                dest_file = self.malformed_dir / f"{queue_name}_{task_file.stem}_{timestamp}{task_file.suffix}"

            task_file.rename(dest_file)
            logger.warning(
                f"Quarantined malformed task file: {task_file} -> {dest_file} "
                f"(error: {error})"
            )
        except Exception as move_error:
            logger.error(f"Failed to quarantine malformed task {task_file}: {move_error}")

    def update(self, task: Task) -> None:
        """Update a task's state."""
        queue_path = self.queue_dir / task.assigned_to
        task_file = queue_path / f"{task.id}.json"

        if not task_file.exists():
            logger.warning(f"Task file not found for update: {task.id}")
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
        """Check if all dependencies are completed successfully."""
        for dep_id in task.depends_on:
            if not dep_id:
                continue

            dep_file = self.completed_dir / f"{dep_id}.json"
            if not dep_file.exists():
                return False

            # Verify task actually completed successfully (not just exists)
            try:
                dep_task = self._load_task(dep_file)
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                return False

        return True

    def get_completed(self, task_id: str) -> Optional[Task]:
        """Get a completed task by ID."""
        completed_file = self.completed_dir / f"{task_id}.json"
        if not completed_file.exists():
            return None

        try:
            return self._load_task(completed_file)
        except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
            return None

    def get_tasks_by_epic(self, epic_key: str) -> dict[str, List[Task]]:
        """Get all tasks (pending, in-progress, completed, failed) for an epic.

        Args:
            epic_key: JIRA epic key (e.g., PROJ-100)

        Returns:
            dict with keys: pending, in_progress, completed, failed
            Each containing a list of Task objects
        """
        result = {
            "pending": [],
            "in_progress": [],
            "completed": [],
            "failed": [],
        }

        # Search all queue directories for pending/in-progress tasks
        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if not queue_dir.is_dir():
                    continue
                for task_file in queue_dir.glob("*.json"):
                    try:
                        task = self._load_task(task_file)
                        if task.context.get("epic_key") == epic_key:
                            if task.status == TaskStatus.PENDING:
                                result["pending"].append(task)
                            elif task.status == TaskStatus.IN_PROGRESS:
                                result["in_progress"].append(task)
                            elif task.status == TaskStatus.FAILED:
                                result["failed"].append(task)
                    except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                        continue

        # Search completed directory
        if self.completed_dir.exists():
            for task_file in self.completed_dir.glob("*.json"):
                try:
                    task = self._load_task(task_file)
                    if task.context.get("epic_key") == epic_key:
                        if task.status == TaskStatus.COMPLETED:
                            result["completed"].append(task)
                        elif task.status == TaskStatus.FAILED:
                            result["failed"].append(task)
                except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                    continue

        return result

    def get_failed_task(self, identifier: str) -> Optional[Task]:
        """Find a failed task by task ID or JIRA key.

        Args:
            identifier: Task ID or JIRA key (e.g., PROJ-104)

        Returns:
            Task if found and failed, None otherwise
        """
        # Check completed directory for failed tasks
        if self.completed_dir.exists():
            for task_file in self.completed_dir.glob("*.json"):
                try:
                    task = self._load_task(task_file)
                    if task.status != TaskStatus.FAILED:
                        continue
                    # Match by task ID or JIRA key
                    if task.id == identifier or task.context.get("jira_key") == identifier:
                        return task
                except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                    continue

        # Also check queue directories for failed tasks
        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if not queue_dir.is_dir():
                    continue
                for task_file in queue_dir.glob("*.json"):
                    try:
                        task = self._load_task(task_file)
                        if task.status != TaskStatus.FAILED:
                            continue
                        if task.id == identifier or task.context.get("jira_key") == identifier:
                            return task
                    except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                        continue

        return None

    def requeue_task(self, task: Task) -> None:
        """Reset a failed task and re-queue it for processing.

        Args:
            task: Failed task to retry
        """
        # Remove from completed directory if present
        completed_file = self.completed_dir / f"{task.id}.json"
        if completed_file.exists():
            completed_file.unlink()

        # Also remove from queue directories if present (prevent duplicates)
        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if queue_dir.is_dir():
                    task_file = queue_dir / f"{task.id}.json"
                    if task_file.exists():
                        task_file.unlink()
                        break

        # Reset task state
        task.status = TaskStatus.PENDING
        task.last_error = None
        task.failed_at = None
        task.failed_by = None
        task.started_at = None
        task.started_by = None
        # Keep retry_count to track total attempts

        # Push to appropriate queue
        self.push(task, task.assigned_to)
