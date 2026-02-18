"""File-based task queue using JSON files (ported from Bash system)."""

import json
import logging
import shutil
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

from ..core.task import Task, TaskStatus
from .locks import FileLock
from ..utils.atomic_io import atomic_write_model

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

        # Cache of task files confirmed non-pending (avoids re-deserializing)
        self._non_pending_files: dict[str, set[str]] = {}
        self._queue_dir_mtime: dict[str, float] = {}

        # Cache of completed dependency lookups with 60s TTL
        self._completed_cache: dict[str, tuple[bool, float]] = {}

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
        Rejects tasks that already exist in completed/ to prevent
        stale worktree copies from resurrecting finished work.
        """
        completed_file = self.completed_dir / f"{task.id}.json"
        if completed_file.exists():
            logger.warning(
                f"Rejecting push for already-completed task {task.id}"
            )
            return

        queue_path = self.queue_dir / queue_id
        queue_path.mkdir(parents=True, exist_ok=True)

        task_file = queue_path / f"{task.id}.json"

        # Atomic write
        atomic_write_model(task_file, task)

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

        # Invalidate non-pending cache when directory changes (new file added/removed)
        try:
            current_mtime = queue_path.stat().st_mtime
        except OSError:
            return None
        if current_mtime != self._queue_dir_mtime.get(queue_id, 0):
            self._non_pending_files.pop(queue_id, None)
            self._queue_dir_mtime[queue_id] = current_mtime

        non_pending = self._non_pending_files.setdefault(queue_id, set())

        # Sort by filename (chronological order)
        task_files = sorted(queue_path.glob("*.json"))

        for task_file in task_files:
            if not task_file.exists():
                continue

            # Skip files we already know are non-pending
            if task_file.name in non_pending:
                continue

            try:
                task = self._load_task(task_file)

                # Recover orphaned in_progress tasks (agent crashed/restarted)
                if task.status == TaskStatus.IN_PROGRESS:
                    if self._is_orphaned(task):
                        outcome = self._recover_single_orphan(task_file, task)
                        if outcome == "auto_completed":
                            non_pending.add(task_file.name)
                            continue
                        # "reset_to_pending" — fall through to pending processing
                    else:
                        non_pending.add(task_file.name)
                        continue

                # Only process pending tasks
                if task.status != TaskStatus.PENDING:
                    non_pending.add(task_file.name)
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

    def _is_orphaned(self, task: Task) -> bool:
        """Check if an in_progress task is orphaned (no agent heartbeat).

        A task is orphaned when the agent that started it is no longer alive.
        We detect this by checking if the agent's heartbeat file is stale.
        """
        if not task.started_by:
            return True

        heartbeat_file = self.heartbeat_dir / task.started_by
        if not heartbeat_file.exists():
            return True

        try:
            stat = heartbeat_file.stat()
            age_seconds = time.time() - stat.st_mtime
            # Stale if no heartbeat for 2 minutes (agents write every poll cycle)
            return age_seconds > 120
        except OSError:
            return True

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

    def claim(self, queue_id: str, agent_id: str) -> Optional[tuple[Task, FileLock]]:
        """Atomically pop a task and acquire its lock in one step.

        Eliminates the race window between pop() and acquire_lock() where
        a second worker could pop the same task before the first locks it.

        Returns (task, lock) if a task was claimed, None if queue is empty
        or all candidates are locked by other workers.
        """
        queue_path = self.queue_dir / queue_id
        if not queue_path.exists():
            return None

        # Invalidate non-pending cache when directory changes
        try:
            current_mtime = queue_path.stat().st_mtime
        except OSError:
            return None
        if current_mtime != self._queue_dir_mtime.get(queue_id, 0):
            self._non_pending_files.pop(queue_id, None)
            self._queue_dir_mtime[queue_id] = current_mtime

        non_pending = self._non_pending_files.setdefault(queue_id, set())
        task_files = sorted(queue_path.glob("*.json"))

        for task_file in task_files:
            if not task_file.exists():
                continue

            if task_file.name in non_pending:
                continue

            try:
                task = self._load_task(task_file)

                if task.status == TaskStatus.IN_PROGRESS:
                    if self._is_orphaned(task):
                        outcome = self._recover_single_orphan(task_file, task)
                        if outcome == "auto_completed":
                            non_pending.add(task_file.name)
                            continue
                        # "reset_to_pending" — fall through to pending check
                    else:
                        non_pending.add(task_file.name)
                        continue

                if task.status != TaskStatus.PENDING:
                    non_pending.add(task_file.name)
                    continue

                if not self._can_retry(task):
                    continue

                if not self._dependencies_met(task):
                    continue

                # Atomic: try to lock before returning the task
                lock = FileLock(self.lock_dir, task.id)
                if lock.acquire():
                    return (task, lock)

                # Another worker claimed it — skip to next candidate
                continue

            except FileNotFoundError:
                continue
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self._quarantine_malformed_task(task_file, e)
                continue

        return None

    def update(self, task: Task) -> None:
        """Update a task's state."""
        queue_path = self.queue_dir / task.assigned_to
        task_file = queue_path / f"{task.id}.json"

        if not task_file.exists():
            logger.warning(f"Task file not found for update: {task.id}")
            return

        atomic_write_model(task_file, task)

        # Invalidate non-pending cache entry — status may have changed back to PENDING
        non_pending = self._non_pending_files.get(task.assigned_to)
        if non_pending:
            non_pending.discard(task_file.name)

    def mark_completed(self, task: Task) -> None:
        """Move task to completed storage."""
        self.move_to_completed(task)

    def move_to_completed(self, task: Task) -> None:
        """Move task to completed storage without changing its status.

        Use this when the task already has the correct terminal status
        (e.g. CANCELLED) and you just need to archive it.
        """
        queue_path = self.queue_dir / task.assigned_to
        task_file = queue_path / f"{task.id}.json"

        completed_file = self.completed_dir / f"{task.id}.json"

        # Write to completed directory
        atomic_write_model(completed_file, task)

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
        return self.load_task_file(task_file)

    @staticmethod
    def load_task_file(task_file: Path) -> Task:
        """Load a task from a JSON file. Public for CLI/external use."""
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

        time_since_failure = (datetime.now(timezone.utc) - task.last_failed_at).total_seconds()
        return time_since_failure >= backoff

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies are completed successfully."""
        now = time.time()
        for dep_id in task.depends_on:
            if not dep_id:
                continue

            # Check cache first (60s TTL)
            cached = self._completed_cache.get(dep_id)
            if cached is not None:
                is_completed, cached_at = cached
                if now - cached_at < 60:
                    if not is_completed:
                        return False
                    continue

            dep_file = self.completed_dir / f"{dep_id}.json"
            if not dep_file.exists():
                self._completed_cache[dep_id] = (False, now)
                return False

            # Verify task actually completed successfully (not just exists)
            try:
                dep_task = self._load_task(dep_file)
                completed = dep_task.status == TaskStatus.COMPLETED
                self._completed_cache[dep_id] = (completed, now)
                if not completed:
                    return False
            except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                self._completed_cache[dep_id] = (False, now)
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
        task = self.find_task(identifier)
        if task and task.status == TaskStatus.FAILED:
            return task
        return None

    def find_task(self, identifier: str) -> Optional[Task]:
        """Find a task by ID or JIRA key across all queues and completed, regardless of status.

        Args:
            identifier: Task ID or JIRA key (e.g., PROJ-104)

        Returns:
            Task if found, None otherwise
        """
        # Fast path: try direct file lookup by ID
        task_file = self._find_task_file(identifier)
        if task_file:
            try:
                return self._load_task(task_file)
            except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                pass

        # Slow path: scan all files for JIRA key match
        for task_file in self._iter_all_task_files():
            try:
                task = self._load_task(task_file)
                if task.context.get("jira_key") == identifier:
                    return task
            except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                continue

        return None

    def get_all_failed(self) -> List[Task]:
        """Get all failed tasks from completed and queue directories.

        Returns:
            List of failed tasks sorted by failed_at time (most recent first)
        """
        failed_tasks = []

        # Check completed directory for failed tasks
        if self.completed_dir.exists():
            for task_file in self.completed_dir.glob("*.json"):
                try:
                    task = self._load_task(task_file)
                    if task.status == TaskStatus.FAILED:
                        failed_tasks.append(task)
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
                        if task.status == TaskStatus.FAILED:
                            failed_tasks.append(task)
                    except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                        continue

        # Sort by failed_at time (most recent first)
        failed_tasks.sort(
            key=lambda t: t.failed_at or datetime.min,
            reverse=True
        )

        return failed_tasks

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

    def check_subtasks_complete(self, parent_task_id: str, subtask_ids: list[str]) -> bool:
        """Check if all subtasks of a parent are completed successfully.

        Args:
            parent_task_id: The parent task ID
            subtask_ids: List of expected subtask IDs

        Returns:
            True only when ALL subtask_ids are found in completed/ with COMPLETED status.
        """
        if not subtask_ids:
            return True

        for subtask_id in subtask_ids:
            subtask_file = self.completed_dir / f"{subtask_id}.json"
            if not subtask_file.exists():
                return False

            try:
                subtask = self._load_task(subtask_file)
                if subtask.status != TaskStatus.COMPLETED:
                    return False
            except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                return False

        return True

    def get_subtasks(self, parent_task_id: str) -> List[Task]:
        """Get all subtasks of a parent across all queues and completed directory.

        Searches by checking task's parent_task_id field.
        Returns tasks from both active queues and completed directory.
        """
        subtasks = []

        # Search all queue directories
        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if not queue_dir.is_dir():
                    continue
                for task_file in queue_dir.glob("*.json"):
                    try:
                        task = self._load_task(task_file)
                        if task.parent_task_id == parent_task_id:
                            subtasks.append(task)
                    except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                        continue

        # Search completed directory
        if self.completed_dir.exists():
            for task_file in self.completed_dir.glob("*.json"):
                try:
                    task = self._load_task(task_file)
                    if task.parent_task_id == parent_task_id:
                        subtasks.append(task)
                except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError):
                    continue

        return subtasks

    def get_parent_task(self, task: Task) -> Optional[Task]:
        """Load the parent task for a subtask.

        Looks up task.parent_task_id in all queues and completed directory.
        Returns None if task has no parent or parent not found.
        """
        if not task.parent_task_id:
            return None

        return self.find_task(task.parent_task_id)

    def create_fan_in_task(self, parent_task: Task, completed_subtasks: List[Task]) -> Task:
        """Create a continuation task that aggregates subtask results.

        The fan-in task:
        - ID: f"fan-in-{parent_task.id}"
        - Type: matches parent's original type
        - Inherits parent's context with added fan_in=True
        - Aggregates result_summary from all completed subtasks
        - Assigned to the next agent in workflow (typically QA)
        """
        # Aggregate results and collect implementation branches
        aggregated_results = []
        subtask_branches = []
        for subtask in completed_subtasks:
            if subtask.result_summary:
                aggregated_results.append(f"[{subtask.title}]: {subtask.result_summary}")
            branch = (
                subtask.context.get("implementation_branch")
                or subtask.context.get("worktree_branch")
            )
            if branch:
                subtask_branches.append(branch)

        context = {
            **parent_task.context,
            "fan_in": True,
            "parent_task_id": parent_task.id,
            "subtask_count": len(completed_subtasks),
            "aggregated_results": "\n".join(aggregated_results),
        }
        # Carry subtask branches so needs_fix cycles reuse existing code
        # instead of reimplementing from scratch
        if subtask_branches:
            context["implementation_branch"] = subtask_branches[0]
            if len(subtask_branches) > 1:
                context["subtask_branches"] = subtask_branches

        # Aggregate subtask costs for the fan-in task.
        # Subtasks inherit parent context (including the parent's _cumulative_cost
        # as a baseline), then accumulate their own LLM cost on top. To avoid
        # counting the parent baseline N times, subtract it from each subtask.
        parent_baseline = parent_task.context.get("_cumulative_cost", 0.0)
        subtask_own_costs = sum(
            st.context.get("_cumulative_cost", 0.0) - parent_baseline
            for st in completed_subtasks
        )
        context["_cumulative_cost"] = parent_baseline + subtask_own_costs

        fan_in_task = Task(
            id=f"fan-in-{parent_task.id}",
            type=parent_task.type,
            status=TaskStatus.PENDING,
            priority=parent_task.priority,
            created_by="system",
            assigned_to="qa",  # Next step after engineer in default workflow
            created_at=datetime.now(UTC),
            title=f"[fan-in] {parent_task.title}",
            description=parent_task.description,
            context=context,
            result_summary="\n".join(aggregated_results) if aggregated_results else None,
        )
        return fan_in_task

    def delete_task(self, task_id: str) -> bool:
        """Permanently delete a task file from disk.

        Searches queue directories and completed directory. Also cleans up
        the associated lock directory and cache entries.

        Returns True if the task was found and deleted.
        """
        task_file = self._find_task_file(task_id)
        if not task_file:
            return False

        # Remove the JSON file
        queue_name = task_file.parent.name
        task_file.unlink()

        # Clean up lock directory
        lock_path = self.lock_dir / f"{task_id}.lock"
        if lock_path.exists() and lock_path.is_dir():
            shutil.rmtree(lock_path, ignore_errors=True)

        # Invalidate caches
        non_pending = self._non_pending_files.get(queue_name)
        if non_pending:
            non_pending.discard(f"{task_id}.json")
        self._completed_cache.pop(task_id, None)
        # Force directory mtime re-check on next pop/claim
        self._queue_dir_mtime.pop(queue_name, None)

        return True

    def _find_task_file(self, task_id: str) -> Optional[Path]:
        """Locate a task's JSON file by ID across queue and completed directories."""
        filename = f"{task_id}.json"

        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if not queue_dir.is_dir():
                    continue
                candidate = queue_dir / filename
                if candidate.exists():
                    return candidate

        candidate = self.completed_dir / filename
        if candidate.exists():
            return candidate

        return None

    def _iter_all_task_files(self):
        """Yield all task JSON file paths across queues and completed."""
        if self.queue_dir.exists():
            for queue_dir in self.queue_dir.iterdir():
                if not queue_dir.is_dir():
                    continue
                yield from queue_dir.glob("*.json")

        if self.completed_dir.exists():
            yield from self.completed_dir.glob("*.json")

    def _is_already_completed(self, task_id: str) -> bool:
        """Check if a task already exists in the completed directory."""
        return (self.completed_dir / f"{task_id}.json").exists()

    def _has_successor_chain_task(self, task_id: str) -> bool:
        """Check if any queued or completed task is a chain successor of this task.

        A successor is a task with context["source_task_id"] == task_id
        AND context["chain_step"] == True.
        """
        for task_file in self._iter_all_task_files():
            try:
                data = json.loads(task_file.read_text())
                ctx = data.get("context", {})
                if ctx.get("source_task_id") == task_id and ctx.get("chain_step"):
                    return True
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return False

    def _auto_complete_orphan(self, task_file: Path, task: Task, reason: str) -> None:
        """Auto-complete an orphaned task and move it to completed/."""
        task.status = TaskStatus.COMPLETED
        task.context["completed_by"] = "recovery"
        task.context["recovery_reason"] = reason
        completed_file = self.completed_dir / f"{task.id}.json"
        atomic_write_model(completed_file, task)
        if task_file.exists():
            task_file.unlink()

    def _recover_single_orphan(self, task_file: Path, task: Task) -> str:
        """Apply 3-tier recovery to one orphaned IN_PROGRESS task.

        Tier 1: Already in completed/ → remove stale queue copy
        Tier 2: Chain task with successor queued/completed → auto-complete
        Tier 3: Genuine orphan → reset to PENDING

        Returns "auto_completed" or "reset_to_pending".
        """
        if self._is_already_completed(task.id):
            task_file.unlink()
            logger.debug(f"Removed stale queue copy of completed task {task.id}")
            return "auto_completed"

        if task.context.get("chain_step") and self._has_successor_chain_task(task.id):
            self._auto_complete_orphan(task_file, task, "successor chain task already exists")
            logger.debug(f"Auto-completed orphan {task.id}: successor exists")
            return "auto_completed"

        task.status = TaskStatus.PENDING
        task.started_at = None
        task.started_by = None
        atomic_write_model(task_file, task)
        logger.debug(f"Reset genuine orphan {task.id} to pending")
        return "reset_to_pending"

    def recover_orphaned_tasks(self, queue_ids: list[str] | None = None) -> dict:
        """Unified recovery for orphaned in_progress tasks.

        Delegates to _recover_single_orphan() for the 3-tier check.

        Does NOT check heartbeats — assumes all agents for the scanned
        queues are dead. Callers are responsible for ensuring this
        (e.g., at startup or after killing an agent).

        Args:
            queue_ids: If provided, only scan these queue directories.
                       Otherwise scan all queues.

        Returns:
            {"auto_completed": [...], "reset_to_pending": [...], "errors": [...]}
        """
        result = {"auto_completed": [], "reset_to_pending": [], "errors": []}

        if not self.queue_dir.exists():
            return result

        dirs_to_scan = []
        if queue_ids:
            for qid in queue_ids:
                qdir = self.queue_dir / qid
                if qdir.exists() and qdir.is_dir():
                    dirs_to_scan.append(qdir)
        else:
            for qdir in self.queue_dir.iterdir():
                if qdir.is_dir():
                    dirs_to_scan.append(qdir)

        for queue_path in dirs_to_scan:
            for task_file in queue_path.glob("*.json"):
                try:
                    task = self._load_task(task_file)
                    if task.status != TaskStatus.IN_PROGRESS:
                        continue

                    outcome = self._recover_single_orphan(task_file, task)
                    result[outcome].append(task.id)

                except Exception as e:
                    result["errors"].append(str(task_file))
                    logger.error(f"Error recovering task {task_file}: {e}")

        return result

    def _fan_in_already_created(self, parent_task_id: str) -> bool:
        """Check if fan-in task already exists (prevent duplicate creation)."""
        fan_in_id = f"fan-in-{parent_task_id}"
        return self.find_task(fan_in_id) is not None
