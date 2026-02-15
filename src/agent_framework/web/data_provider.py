"""Data provider for web dashboard - shared data access layer."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..core.activity import ActivityManager, AgentStatus, TaskPhase
from ..core.config import load_agents, AgentDefinition
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker
from .models import (
    AgentData,
    AgentStatusEnum,
    QueueStats,
    EventData,
    FailedTaskData,
    HealthCheck,
    HealthReport,
    CurrentTaskData,
    LogEntry,
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
