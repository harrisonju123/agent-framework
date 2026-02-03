"""Circuit breaker for detecting and preventing system failures."""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

from ..core.task import Task, TaskType


logger = logging.getLogger(__name__)


class CircuitBreakerReport:
    """Report of circuit breaker check results."""

    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.issues: Dict[str, List[str]] = defaultdict(list)
        self.warnings: List[str] = []

    def add_check(self, check_name: str, passed: bool, message: str = ""):
        """Add a check result."""
        self.checks[check_name] = passed
        if not passed and message:
            self.issues[check_name].append(message)

    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)

    @property
    def passed(self) -> bool:
        """Check if all checks passed."""
        return all(self.checks.values())

    def __str__(self) -> str:
        """Format report as string."""
        lines = ["Circuit Breaker Report", "=" * 50]

        for check_name, passed in self.checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"{status}: {check_name}")
            if not passed:
                for issue in self.issues.get(check_name, []):
                    lines.append(f"  - {issue}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


class CircuitBreaker:
    """
    Circuit breaker for detecting system failures.

    Ported from scripts/circuit-breaker.sh with 8 safety checks.
    """

    def __init__(
        self,
        workspace: Path,
        max_queue_size: int = 100,
        max_escalations: int = 50,
        max_task_age_days: int = 7,
        max_circular_depth: int = 5,
    ):
        self.workspace = Path(workspace)
        self.comm_dir = self.workspace / ".agent-communication"
        self.queue_dir = self.comm_dir / "queues"
        self.completed_dir = self.comm_dir / "completed"

        self.max_queue_size = max_queue_size
        self.max_escalations = max_escalations
        self.max_task_age_days = max_task_age_days
        self.max_circular_depth = max_circular_depth

    def run_all_checks(self) -> CircuitBreakerReport:
        """Run all 8 safety checks."""
        report = CircuitBreakerReport()

        # Check 1: Queue size limits
        queue_ok, queue_msg = self.check_queue_sizes()
        report.add_check("queue_sizes", queue_ok, queue_msg)

        # Check 2: Escalation count limits
        esc_ok, esc_msg = self.check_escalation_count()
        report.add_check("escalation_count", esc_ok, esc_msg)

        # Check 3: Circular dependencies
        circ_ok, circ_msg = self.check_circular_dependencies()
        report.add_check("circular_dependencies", circ_ok, circ_msg)

        # Check 4: Stale tasks
        stale_tasks = self.check_stale_tasks()
        report.add_check("stale_tasks", len(stale_tasks) == 0,
                        f"Found {len(stale_tasks)} stale tasks")

        # Check 5: Task creation rate
        rate_ok, rate_msg = self.check_task_rate()
        report.add_check("task_rate", rate_ok, rate_msg)

        # Check 6: Stuck tasks
        stuck_tasks = self.check_stuck_tasks()
        report.add_check("stuck_tasks", len(stuck_tasks) == 0,
                        f"Found {len(stuck_tasks)} stuck tasks with 3+ retries")

        # Check 7: Duplicate IDs
        dupes = self.check_duplicate_ids()
        report.add_check("duplicate_ids", len(dupes) == 0,
                        f"Found {len(dupes)} duplicate task IDs")

        # Check 8: Escalation retries
        esc_retry_ok, esc_retry_msg = self.check_escalation_retries()
        report.add_check("escalation_retries", esc_retry_ok, esc_retry_msg)

        return report

    def check_queue_sizes(self) -> tuple[bool, str]:
        """Check if any queue exceeds max size."""
        if not self.queue_dir.exists():
            return True, ""

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            task_count = len(list(agent_dir.glob("*.json")))
            if task_count > self.max_queue_size:
                return False, f"Queue {agent_dir.name} has {task_count} tasks (max: {self.max_queue_size})"

        return True, ""

    def check_escalation_count(self) -> tuple[bool, str]:
        """Check total escalation count."""
        escalation_count = 0

        if self.queue_dir.exists():
            for agent_dir in self.queue_dir.iterdir():
                if not agent_dir.is_dir():
                    continue

                for task_file in agent_dir.glob("*.json"):
                    try:
                        task_data = json.loads(task_file.read_text())
                        if task_data.get("type") in ("escalation", TaskType.ESCALATION.value):
                            escalation_count += 1
                    except Exception:
                        continue

        if escalation_count > self.max_escalations:
            return False, f"Found {escalation_count} escalations (max: {self.max_escalations})"

        return True, ""

    def check_circular_dependencies(self) -> tuple[bool, str]:
        """Check for circular task dependencies."""
        # Build dependency graph
        task_deps: Dict[str, List[str]] = {}

        if self.queue_dir.exists():
            for agent_dir in self.queue_dir.iterdir():
                if not agent_dir.is_dir():
                    continue

                for task_file in agent_dir.glob("*.json"):
                    try:
                        task_data = json.loads(task_file.read_text())
                        task_id = task_data.get("id")
                        depends_on = task_data.get("depends_on", [])
                        task_deps[task_id] = depends_on
                    except Exception:
                        continue

        # Check for cycles using DFS
        def has_cycle(task_id: str, visited: Set[str], path: Set[str]) -> bool:
            if task_id in path:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            path.add(task_id)

            for dep in task_deps.get(task_id, []):
                if dep and has_cycle(dep, visited, path):
                    return True

            path.remove(task_id)
            return False

        visited: Set[str] = set()
        for task_id in task_deps:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    return False, f"Circular dependency detected involving task {task_id}"

        return True, ""

    def check_stale_tasks(self) -> List[Task]:
        """Find tasks older than max age."""
        stale_tasks = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.max_task_age_days)

        if not self.queue_dir.exists():
            return stale_tasks

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())
                    task = Task(**task_data)

                    if task.created_at < cutoff_date:
                        stale_tasks.append(task)
                except Exception:
                    continue

        return stale_tasks

    def check_task_rate(self) -> tuple[bool, str]:
        """Check task creation rate for cascading failures."""
        # Simple heuristic: if >50% of tasks created in last hour
        if not self.queue_dir.exists():
            return True, ""

        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_count = 0
        total_count = 0

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())
                    task = Task(**task_data)
                    total_count += 1

                    if task.created_at > one_hour_ago:
                        recent_count += 1
                except Exception:
                    continue

        if total_count > 20 and recent_count / total_count > 0.5:
            return False, f"High task creation rate: {recent_count}/{total_count} in last hour"

        return True, ""

    def check_stuck_tasks(self) -> List[Task]:
        """Find tasks with 3+ retries."""
        stuck_tasks = []

        if not self.queue_dir.exists():
            return stuck_tasks

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())
                    task = Task(**task_data)

                    if task.retry_count >= 3:
                        stuck_tasks.append(task)
                except Exception:
                    continue

        return stuck_tasks

    def check_duplicate_ids(self) -> List[str]:
        """Find duplicate task IDs."""
        seen_ids: Set[str] = set()
        duplicates: List[str] = []

        if not self.queue_dir.exists():
            return duplicates

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())
                    task_id = task_data.get("id")

                    if task_id in seen_ids:
                        duplicates.append(task_id)
                    else:
                        seen_ids.add(task_id)
                except Exception:
                    continue

        return duplicates

    def check_escalation_retries(self) -> tuple[bool, str]:
        """Check that escalation tasks don't have retries."""
        if not self.queue_dir.exists():
            return True, ""

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())

                    if task_data.get("type") == "escalation" and task_data.get("retry_count", 0) > 0:
                        task_id = task_data.get("id")
                        return False, f"Escalation task {task_id} has retries (should not retry)"
                except Exception:
                    continue

        return True, ""

    def fix_issues(self, report: CircuitBreakerReport) -> None:
        """Auto-fix detected issues."""
        logger.info("Running auto-fix for circuit breaker issues")

        # Archive stale tasks
        if "stale_tasks" in report.issues:
            stale_tasks = self.check_stale_tasks()
            self._archive_stale_tasks(stale_tasks)

        # Reset escalation retries
        if "escalation_retries" in report.issues:
            self._reset_escalation_retries()

        logger.info("Auto-fix complete")

    def _archive_stale_tasks(self, stale_tasks: List[Task]) -> None:
        """Archive stale tasks."""
        archive_dir = self.workspace / ".agent-communication" / "archived"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for task in stale_tasks:
            logger.info(f"Archiving stale task {task.id}")

            # Find and move task file
            for agent_dir in self.queue_dir.iterdir():
                if not agent_dir.is_dir():
                    continue

                task_file = agent_dir / f"{task.id}.json"
                if task_file.exists():
                    archive_file = archive_dir / f"{task.id}.json"
                    task_file.rename(archive_file)
                    break

    def _reset_escalation_retries(self) -> None:
        """Reset retry count on escalation tasks."""
        if not self.queue_dir.exists():
            return

        for agent_dir in self.queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())

                    if task_data.get("type") in ("escalation", TaskType.ESCALATION.value) and task_data.get("retry_count", 0) > 0:
                        task_data["retry_count"] = 0
                        # Use atomic write pattern to prevent race conditions
                        tmp_file = task_file.with_suffix('.tmp')
                        tmp_file.write_text(json.dumps(task_data, indent=2))
                        tmp_file.rename(task_file)
                        logger.info(f"Reset retry count for escalation {task_data['id']}")
                except Exception as e:
                    logger.error(f"Error fixing {task_file}: {e}")
