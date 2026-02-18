"""Data provider for web dashboard - shared data access layer."""

import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..core.activity import ActivityManager, AgentStatus, TaskPhase
from ..core.config import load_agents, AgentDefinition
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker
from ..core.task import Task, TaskStatus, TaskType
from .models import (
    ActiveTaskData,
    AgentData,
    AgenticsMetrics,
    AgentStatusEnum,
    CheckpointData,
    QueueStats,
    EventData,
    FailedTaskData,
    HealthCheck,
    HealthReport,
    CurrentTaskData,
    LogEntry,
    SpecializationCount,
    TeamSessionData,
    ToolActivityData,
)

logger = logging.getLogger(__name__)


class DashboardDataProvider:
    """Provides data for both TUI and Web dashboards.

    This class abstracts the data access layer so both the TUI dashboard
    and web dashboard can share the same data fetching logic.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.activity_manager = ActivityManager(workspace)
        self.queue = FileQueue(workspace)
        self.circuit_breaker = CircuitBreaker(workspace)
        self._agents_config_cache: Optional[List[AgentDefinition]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 5  # Refresh config/teams every 5 seconds
        self._teams_cache: Optional[List[TeamSessionData]] = None
        self._teams_cache_time: Optional[datetime] = None

    def _get_agents_config(self) -> List[AgentDefinition]:
        """Get agents config with caching."""
        now = datetime.now(timezone.utc)
        if (
            self._agents_config_cache is None
            or self._cache_time is None
            or (now - self._cache_time).total_seconds() > self._cache_ttl_seconds
        ):
            try:
                self._agents_config_cache = load_agents(
                    self.workspace / "config" / "agents.yaml"
                )
                self._cache_time = now
            except FileNotFoundError:
                logger.warning("agents.yaml not found, returning empty list")
                return []
        return self._agents_config_cache or []

    def get_agents_data(self) -> List[AgentData]:
        """Get all agents with their current status."""
        agents_config = self._get_agents_config()
        activities = self.activity_manager.get_all_activities()
        activities_by_id = {a.agent_id: a for a in activities}

        result: List[AgentData] = []

        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            activity = activities_by_id.get(agent_def.id)

            if not activity or activity.status == AgentStatus.IDLE:
                agent_data = AgentData(
                    id=agent_def.id,
                    name=agent_def.name,
                    queue=agent_def.queue,
                    status=AgentStatusEnum.IDLE,
                    phases_completed=0,
                )
            elif activity.status == AgentStatus.COMPLETING and activity.current_task:
                # Task just completed - show brief completing state
                current_task = CurrentTaskData(
                    id=activity.current_task.id,
                    title=activity.current_task.title,
                    type=activity.current_task.type,
                    started_at=activity.current_task.started_at,
                )
                agent_data = AgentData(
                    id=agent_def.id,
                    name=agent_def.name,
                    queue=agent_def.queue,
                    status=AgentStatusEnum.COMPLETING,
                    current_task=current_task,
                    phases_completed=5,  # All phases complete
                    elapsed_seconds=activity.get_elapsed_seconds(),
                    last_updated=activity.last_updated,
                )
            elif activity.status == AgentStatus.WORKING and activity.current_task:
                current_task = CurrentTaskData(
                    id=activity.current_task.id,
                    title=activity.current_task.title,
                    type=activity.current_task.type,
                    started_at=activity.current_task.started_at,
                )
                phases_completed = sum(1 for p in activity.phases if p.completed)
                phase_name = (
                    activity.current_phase.value
                    if activity.current_phase
                    else None
                )

                # Map tool activity if present during LLM execution
                tool_activity_data = None
                if (
                    activity.tool_activity
                    and activity.current_phase == TaskPhase.EXECUTING_LLM
                ):
                    tool_activity_data = ToolActivityData(
                        tool_name=activity.tool_activity.tool_name,
                        tool_input_summary=activity.tool_activity.tool_input_summary,
                        started_at=activity.tool_activity.started_at,
                        tool_call_count=activity.tool_activity.tool_call_count,
                    )

                agent_data = AgentData(
                    id=agent_def.id,
                    name=agent_def.name,
                    queue=agent_def.queue,
                    status=AgentStatusEnum.WORKING,
                    current_task=current_task,
                    current_phase=phase_name,
                    phases_completed=phases_completed,
                    elapsed_seconds=activity.get_elapsed_seconds(),
                    last_updated=activity.last_updated,
                    tool_activity=tool_activity_data,
                )
            else:
                # Dead agent
                agent_data = AgentData(
                    id=agent_def.id,
                    name=agent_def.name,
                    queue=agent_def.queue,
                    status=AgentStatusEnum.DEAD,
                    last_updated=activity.last_updated if activity else None,
                )

            result.append(agent_data)

        return result

    def get_queue_stats(self) -> List[QueueStats]:
        """Get queue statistics for all agents."""
        agents_config = self._get_agents_config()
        result: List[QueueStats] = []

        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            stats = self.queue.get_queue_stats(agent_def.queue)
            oldest_age = None
            if stats.get("oldest"):
                oldest_dt = stats["oldest"]
                # Handle naive datetimes by assuming UTC
                if oldest_dt.tzinfo is None:
                    oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
                oldest_age = int(
                    (datetime.now(timezone.utc) - oldest_dt).total_seconds()
                )

            result.append(
                QueueStats(
                    queue_id=agent_def.queue,
                    agent_name=agent_def.name,
                    pending_count=stats["count"],
                    oldest_task_age=oldest_age,
                )
            )

        return result

    def get_failed_tasks(self, limit: int = 10) -> List[FailedTaskData]:
        """Get list of failed tasks."""
        failed_tasks = self.queue.get_all_failed()
        result: List[FailedTaskData] = []

        for task in failed_tasks[:limit]:
            result.append(
                FailedTaskData(
                    id=task.id,
                    title=task.title,
                    jira_key=task.context.get("jira_key"),
                    assigned_to=task.assigned_to,
                    retry_count=task.retry_count,
                    last_error=task.last_error,
                    failed_at=task.failed_at,
                )
            )

        return result

    def get_recent_events(self, limit: int = 20) -> List[EventData]:
        """Get recent activity events."""
        events = self.activity_manager.get_recent_events(limit=limit)
        result: List[EventData] = []

        for event in events:
            result.append(
                EventData(
                    type=event.type,
                    agent=event.agent,
                    task_id=event.task_id,
                    title=event.title,
                    timestamp=event.timestamp,
                    duration_ms=event.duration_ms,
                    retry_count=event.retry_count,
                    phase=event.phase.value if event.phase else None,
                    error_message=event.error_message,
                    pr_url=event.pr_url,
                )
            )

        return result

    def get_health_status(self) -> HealthReport:
        """Get system health status from circuit breaker."""
        report = self.circuit_breaker.run_all_checks()

        checks: List[HealthCheck] = []
        for check_name, passed in report.checks.items():
            message = None
            if not passed and check_name in report.issues:
                # Join multiple issues into one message
                message = "; ".join(report.issues[check_name])

            checks.append(
                HealthCheck(
                    name=check_name,
                    passed=passed,
                    message=message,
                )
            )

        return HealthReport(
            passed=report.passed,
            checks=checks,
            warnings=report.warnings,
        )

    def is_paused(self) -> bool:
        """Check if agent processing is paused."""
        pause_file = self.workspace / ".agent-communication" / "pause"
        return pause_file.exists()

    def pause(self) -> bool:
        """Pause agent processing."""
        pause_file = self.workspace / ".agent-communication" / "pause"
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        if not pause_file.exists():
            pause_file.write_text(str(int(datetime.now(timezone.utc).timestamp())))
            return True
        return False

    def resume(self) -> bool:
        """Resume agent processing."""
        pause_file = self.workspace / ".agent-communication" / "pause"
        if pause_file.exists():
            pause_file.unlink()
            return True
        return False

    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task.

        Args:
            task_id: Task ID or JIRA key

        Returns:
            True if task was found and requeued
        """
        task = self.queue.get_failed_task(task_id)
        if not task:
            return False

        self.queue.requeue_task(task)
        return True

    def get_task(self, task_id: str) -> Optional[FailedTaskData]:
        """Get a specific task by ID."""
        task = self.queue.get_failed_task(task_id)
        if not task:
            return None

        return FailedTaskData(
            id=task.id,
            title=task.title,
            jira_key=task.context.get("jira_key"),
            assigned_to=task.assigned_to,
            retry_count=task.retry_count,
            last_error=task.last_error,
            failed_at=task.failed_at,
        )

    def get_active_tasks(self, limit: int = 50) -> List[ActiveTaskData]:
        """Get all pending and in-progress tasks across queues."""
        result: List[ActiveTaskData] = []

        if not self.queue.queue_dir.exists():
            return result

        for queue_dir in self.queue.queue_dir.iterdir():
            if not queue_dir.is_dir() or queue_dir.name == "checkpoints":
                continue
            for task_file in queue_dir.glob("*.json"):
                try:
                    task = FileQueue.load_task_file(task_file)
                    if task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                        continue
                    result.append(
                        ActiveTaskData(
                            id=task.id,
                            title=task.title,
                            status=str(task.status),
                            jira_key=task.context.get("jira_key"),
                            assigned_to=task.assigned_to,
                            created_at=task.created_at,
                            started_at=task.started_at,
                            task_type=str(task.type),
                            parent_task_id=task.parent_task_id,
                        )
                    )
                except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
                    logger.warning(f"Error reading task file {task_file}: {e}")
                    continue

        # IN_PROGRESS first, then PENDING; oldest first within each group
        result.sort(key=lambda t: (t.status != "in_progress", t.created_at))
        return result[:limit]

    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a pending or in-progress task.

        Returns:
            True if task was found and cancelled
        """
        task = self.queue.find_task(task_id)
        if not task:
            return False

        if task.status not in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
            return False

        task.mark_cancelled(os.getenv("USER", "dashboard"), reason)
        self.queue.update(task)
        return True

    def get_active_teams(self) -> List[TeamSessionData]:
        """Get active Agent Team sessions from local workspace."""
        now = datetime.now(timezone.utc)
        if (
            self._teams_cache is not None
            and self._teams_cache_time is not None
            and (now - self._teams_cache_time).total_seconds() <= self._cache_ttl_seconds
        ):
            return self._teams_cache

        teams_dir = self.workspace / ".agent-communication" / "teams"
        claude_teams_dir = Path.home() / ".claude" / "teams"

        sessions: List[TeamSessionData] = []

        for search_dir in [teams_dir, claude_teams_dir]:
            if not search_dir.exists():
                continue
            for session_file in search_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text())
                    started_at = None
                    if data.get("started_at"):
                        try:
                            started_at = datetime.fromisoformat(data["started_at"])
                        except (ValueError, TypeError):
                            pass
                    sessions.append(TeamSessionData(
                        team_name=data.get("team_name", session_file.stem),
                        template=data.get("template", "unknown"),
                        started_at=started_at,
                        source_task_id=data.get("source_task_id"),
                        status=data.get("status", "unknown"),
                    ))
                except (json.JSONDecodeError, OSError):
                    continue

        self._teams_cache = sessions
        self._teams_cache_time = now
        return sessions

    def delete_task(self, task_id: str) -> Optional[str]:
        """Permanently delete a task from disk.

        Only allows deletion of terminal/idle tasks (PENDING, FAILED, CANCELLED).
        IN_PROGRESS tasks must be cancelled first.

        Returns:
            None on success, or an error reason string.
        """
        task = self.queue.find_task(task_id)
        if not task:
            return "not_found"

        deletable = {TaskStatus.PENDING, TaskStatus.FAILED, TaskStatus.CANCELLED}
        if task.status not in deletable:
            return "not_deletable"

        self.queue.delete_task(task_id)
        return None

    def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        assigned_to: str,
        repository: Optional[str] = None,
        priority: int = 1,
    ) -> Task:
        """Create a task directly and push it to the specified queue."""
        task_id = f"manual-{int(datetime.now(timezone.utc).timestamp())}-{uuid.uuid4().hex[:6]}"

        context: Dict[str, str] = {}
        if repository:
            context["github_repo"] = repository

        task = Task(
            id=task_id,
            type=TaskType(task_type),
            status=TaskStatus.PENDING,
            priority=priority,
            created_by="web-dashboard",
            assigned_to=assigned_to,
            created_at=datetime.now(timezone.utc),
            title=title,
            description=description,
            context=context,
        )

        self.queue.push(task, assigned_to)
        return task

    # ============== Checkpoint Methods ==============

    def get_pending_checkpoints(self) -> List[CheckpointData]:
        """Get all tasks awaiting checkpoint approval."""
        checkpoint_dir = self.workspace / ".agent-communication" / "queues" / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        result: List[CheckpointData] = []
        for checkpoint_file in sorted(checkpoint_dir.glob("*.json")):
            try:
                task = FileQueue.load_task_file(checkpoint_file)
                if task.status != TaskStatus.AWAITING_APPROVAL:
                    continue

                # Use file mtime as proxy for when checkpoint was reached
                paused_at = datetime.fromtimestamp(
                    checkpoint_file.stat().st_mtime, tz=timezone.utc
                )

                result.append(
                    CheckpointData(
                        id=task.id,
                        title=task.title,
                        checkpoint_id=task.checkpoint_reached or "unknown",
                        checkpoint_message=task.checkpoint_message or "",
                        assigned_to=task.assigned_to,
                        paused_at=paused_at,
                    )
                )
            except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
                logger.warning(f"Error reading checkpoint file {checkpoint_file}: {e}")
                continue

        return result

    def approve_checkpoint(self, task_id: str, message: Optional[str] = None) -> bool:
        """Approve a checkpoint and re-queue the task.

        Mirrors the CLI `agent approve` logic: load checkpoint file,
        verify AWAITING_APPROVAL status, approve, re-queue, delete file.
        """
        if not re.match(r'^[a-zA-Z0-9_.-]+$', task_id):
            raise ValueError(f"Invalid task_id: {task_id}")

        checkpoint_dir = self.workspace / ".agent-communication" / "queues" / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{task_id}.json"
        if not checkpoint_file.exists():
            return False

        task = FileQueue.load_task_file(checkpoint_file)
        if task.status != TaskStatus.AWAITING_APPROVAL:
            return False

        approver = os.getenv("USER", "dashboard")
        task.approve_checkpoint(approver)

        if message:
            task.notes.append(f"Checkpoint approved: {message}")
        else:
            task.notes.append(
                f"Checkpoint approved at {datetime.now(timezone.utc).isoformat()}"
            )

        # Route directly to next workflow step instead of re-queuing to the
        # same agent — avoids duplicate LLM execution on the completed work
        from ..workflow.executor import resume_after_checkpoint

        try:
            routed = resume_after_checkpoint(task, self.queue, self.workspace)
            if routed:
                checkpoint_file.unlink(missing_ok=True)
            else:
                logger.error(
                    f"Checkpoint approved but could not route task {task_id} "
                    "to next step — preserving checkpoint for retry"
                )
        except Exception as e:
            logger.error(f"Error routing checkpoint task {task_id}: {e}")
            return False

        # The approval itself succeeded regardless of routing outcome
        return True

    def reject_checkpoint(self, task_id: str, feedback: str) -> bool:
        """Reject a checkpoint with feedback and re-queue to the same agent.

        The agent will see the feedback in its prompt and redo the work.
        """
        if not re.match(r'^[a-zA-Z0-9_.-]+$', task_id):
            raise ValueError(f"Invalid task_id: {task_id}")

        checkpoint_dir = self.workspace / ".agent-communication" / "queues" / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{task_id}.json"
        if not checkpoint_file.exists():
            return False

        task = FileQueue.load_task_file(checkpoint_file)
        if task.status != TaskStatus.AWAITING_APPROVAL:
            return False

        task.context["rejection_feedback"] = feedback
        task.notes.append(f"Checkpoint rejected: {feedback}")
        task.reset_to_pending()
        task.checkpoint_reached = None
        task.checkpoint_message = None

        self.queue.push(task, task.assigned_to)
        checkpoint_file.unlink(missing_ok=True)
        return True

    # ============== Log Reading Methods ==============

    def _get_known_agent_ids(self) -> Set[str]:
        """Return the set of enabled agent IDs.

        Reuses the cached agents config so this is essentially free.
        """
        return {agent.id for agent in self._get_agents_config() if agent.enabled}

    def get_logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.workspace / "logs"

    def get_available_log_files(self) -> List[str]:
        """Get list of available agent log files.

        Only returns IDs for agents defined in agents.yaml whose log file
        actually exists — avoids picking up CLI/dashboard/artifact logs.
        """
        logs_dir = self.get_logs_dir()
        if not logs_dir.exists():
            return []
        return [
            agent_id
            for agent_id in self._get_known_agent_ids()
            if (logs_dir / f"{agent_id}.log").exists()
        ]

    def get_agent_logs(self, agent_id: str, lines: int = 100) -> List[str]:
        """Get recent log lines for an agent.

        Args:
            agent_id: Agent identifier (e.g., "engineer", "qa")
            lines: Number of lines to return (from end of file)

        Returns:
            List of log lines (most recent last)

        Raises:
            ValueError: If agent_id contains invalid characters (path traversal protection)
        """
        # Validate agent_id contains only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            raise ValueError(f"Invalid agent_id: {agent_id}")

        log_file = self.get_logs_dir() / f"{agent_id}.log"
        if not log_file.exists():
            return []

        try:
            # Read file and return last N lines
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return [line.rstrip() for line in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
            return []

    def read_log_from_position(
        self, agent_id: str, position: int = 0
    ) -> Tuple[List[str], int]:
        """Read log lines from a specific position.

        Args:
            agent_id: Agent identifier
            position: Byte position to start reading from

        Returns:
            Tuple of (new_lines, new_position)

        Raises:
            ValueError: If agent_id contains invalid characters (path traversal protection)
        """
        # Validate agent_id contains only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
            raise ValueError(f"Invalid agent_id: {agent_id}")

        log_file = self.get_logs_dir() / f"{agent_id}.log"
        if not log_file.exists():
            return [], 0

        try:
            file_size = log_file.stat().st_size
            if position >= file_size:
                return [], position

            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(position)
                new_content = f.read()
                new_position = f.tell()

            lines = [line.rstrip() for line in new_content.splitlines() if line.strip()]
            return lines, new_position

        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
            return [], position

    def get_all_log_positions(self) -> Dict[str, int]:
        """Get current file sizes for known agent log files (for tracking).

        Only checks logs for agents defined in agents.yaml — avoids globbing
        hundreds of CLI/dashboard/artifact .log files in the logs directory.

        Returns:
            Dict mapping agent_id to file size (byte position)
        """
        logs_dir = self.get_logs_dir()
        if not logs_dir.exists():
            return {}

        positions = {}
        for agent_id in self._get_known_agent_ids():
            log_file = logs_dir / f"{agent_id}.log"
            if log_file.exists():
                try:
                    positions[agent_id] = log_file.stat().st_size
                except Exception:
                    positions[agent_id] = 0

        return positions

    def parse_log_level(self, line: str) -> Optional[str]:
        """Extract log level from a log line.

        Args:
            line: Log line text

        Returns:
            Log level (DEBUG, INFO, WARNING, ERROR) or None
        """
        # Common log format: "2024-01-01 10:00:00 INFO message"
        # or "2024-01-01 10:00:00,000 - agent - INFO - message"
        match = re.search(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    # ============== Agentic Metrics ==============

    def compute_agentics_metrics(self, hours: int = 24) -> AgenticsMetrics:
        """Compute agentic feature metrics by scanning recent session JSONL logs.

        Reads all session files modified within the past `hours` window and
        aggregates events logged by the agent loop:
        - ``memory_recall`` → memory hit rate
        - ``self_eval``     → self-evaluation catch rate (FAIL verdicts)
        - ``replan``        → replan trigger rate
        - ``task_start`` / ``task_complete`` → task counts for rate denominators
        - activity files    → specialization distribution (current snapshot)

        Context budget utilization is not recorded in session logs today, so
        that metric is omitted until the budget_manager emits a dedicated event.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        sessions_dir = self.workspace / "logs" / "sessions"

        task_ids_seen: Set[str] = set()
        task_ids_with_memory_recall: Set[str] = set()
        task_ids_with_replan: Set[str] = set()

        self_eval_total = 0
        self_eval_fail_count = 0

        if sessions_dir.exists():
            for session_file in sessions_dir.glob("*.jsonl"):
                try:
                    mtime = datetime.fromtimestamp(
                        session_file.stat().st_mtime, tz=timezone.utc
                    )
                    if mtime < cutoff:
                        continue

                    with open(session_file, "r", encoding="utf-8", errors="replace") as fh:
                        for raw_line in fh:
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            try:
                                entry = json.loads(raw_line)
                            except json.JSONDecodeError:
                                continue

                            # Filter by event timestamp, not just file mtime
                            try:
                                ts = datetime.fromisoformat(entry.get("ts", ""))
                                if ts < cutoff:
                                    continue
                            except (ValueError, TypeError):
                                continue

                            event = entry.get("event", "")
                            task_id = entry.get("task_id", "")

                            if event == "task_start" and task_id:
                                task_ids_seen.add(task_id)

                            elif event == "memory_recall" and task_id:
                                task_ids_with_memory_recall.add(task_id)

                            elif event == "self_eval":
                                verdict = entry.get("verdict", "")
                                # AUTO_PASS means no evaluation ran — don't count it
                                if verdict in ("PASS", "FAIL"):
                                    self_eval_total += 1
                                    if verdict == "FAIL":
                                        self_eval_fail_count += 1

                            elif event == "replan" and task_id:
                                task_ids_with_replan.add(task_id)

                except OSError as e:
                    logger.warning(f"Error reading session file {session_file}: {e}")

        task_count = len(task_ids_seen)

        memory_recall_rate = (
            len(task_ids_with_memory_recall) / task_count if task_count else 0.0
        )
        replan_trigger_rate = (
            len(task_ids_with_replan) / task_count if task_count else 0.0
        )
        self_eval_catch_rate = (
            self_eval_fail_count / self_eval_total if self_eval_total else 0.0
        )

        # Specialization distribution from live activity files (current snapshot).
        # Activity files reflect the most recent task each agent processed, so this
        # captures what profiles are in active use rather than historical counts.
        specialization_counts: Dict[str, int] = defaultdict(int)
        activities = self.activity_manager.get_all_activities()
        for activity in activities:
            profile = getattr(activity, "specialization", None)
            if profile:
                specialization_counts[profile] += 1

        specialization_distribution = [
            SpecializationCount(profile=profile, count=count)
            for profile, count in sorted(specialization_counts.items())
        ]

        return AgenticsMetrics(
            memory_recall_rate=round(memory_recall_rate, 4),
            memory_recalls_total=len(task_ids_with_memory_recall),
            self_eval_catch_rate=round(self_eval_catch_rate, 4),
            self_eval_total=self_eval_total,
            replan_trigger_rate=round(replan_trigger_rate, 4),
            replan_total=len(task_ids_with_replan),
            replan_success_rate=0.0,  # Not tracked in session logs yet
            specialization_distribution=specialization_distribution,
            avg_context_budget_utilization=0.0,  # Not emitted by budget_manager yet
            context_budget_samples=0,
            window_hours=hours,
        )

    # ============== Claude CLI Log Methods ==============

    def get_available_claude_cli_logs(self) -> List[str]:
        """Get list of available Claude CLI log files (task IDs).

        Returns:
            List of task_ids that have CLI logs
        """
        logs_dir = self.get_logs_dir()
        if not logs_dir.exists():
            return []

        task_ids = []
        for log_file in logs_dir.glob("claude-cli-*.log"):
            # Extract task_id from "claude-cli-{task_id}.log"
            name = log_file.stem
            if name.startswith("claude-cli-"):
                task_id = name[len("claude-cli-"):]
                task_ids.append(task_id)

        return task_ids

    def get_claude_cli_logs(self, task_id: str, lines: int = 100) -> List[str]:
        """Get recent log lines for a Claude CLI subprocess.

        Args:
            task_id: Task identifier
            lines: Number of lines to return (from end of file)

        Returns:
            List of log lines (most recent last)

        Raises:
            ValueError: If task_id contains invalid characters (path traversal protection)
        """
        # Validate task_id contains only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            raise ValueError(f"Invalid task_id: {task_id}")

        log_file = self.get_logs_dir() / f"claude-cli-{task_id}.log"
        if not log_file.exists():
            return []

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return [line.rstrip() for line in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"Error reading CLI log file {log_file}: {e}")
            return []

    def read_claude_cli_log_from_position(
        self, task_id: str, position: int = 0
    ) -> Tuple[List[str], int]:
        """Read Claude CLI log lines from a specific position.

        Args:
            task_id: Task identifier
            position: Byte position to start reading from

        Returns:
            Tuple of (new_lines, new_position)

        Raises:
            ValueError: If task_id contains invalid characters (path traversal protection)
        """
        # Validate task_id contains only safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            raise ValueError(f"Invalid task_id: {task_id}")

        log_file = self.get_logs_dir() / f"claude-cli-{task_id}.log"
        if not log_file.exists():
            return [], 0

        try:
            file_size = log_file.stat().st_size
            if position >= file_size:
                return [], position

            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(position)
                new_content = f.read()
                new_position = f.tell()

            lines = [line.rstrip() for line in new_content.splitlines() if line.strip()]
            return lines, new_position

        except Exception as e:
            logger.error(f"Error reading CLI log file {log_file}: {e}")
            return [], position

    def get_all_claude_cli_log_positions(self) -> Dict[str, int]:
        """Get current file sizes for all Claude CLI log files (for tracking).

        Returns:
            Dict mapping task_id to file size (byte position)
        """
        logs_dir = self.get_logs_dir()
        if not logs_dir.exists():
            return {}

        positions = {}
        for log_file in logs_dir.glob("claude-cli-*.log"):
            name = log_file.stem
            if name.startswith("claude-cli-"):
                task_id = name[len("claude-cli-"):]
                try:
                    positions[task_id] = log_file.stat().st_size
                except Exception:
                    positions[task_id] = 0

        return positions

    def get_active_claude_cli_tasks(self) -> Dict[str, str]:
        """Get mapping of agent_id to current task_id for active agents.

        Returns:
            Dict mapping agent_id to task_id for agents currently working or completing
        """
        activities = self.activity_manager.get_all_activities()
        result = {}

        for activity in activities:
            # Include both WORKING and COMPLETING statuses
            if activity.status in (AgentStatus.WORKING, AgentStatus.COMPLETING) and activity.current_task:
                result[activity.agent_id] = activity.current_task.id

        return result
