"""Bridge between autonomous pipeline and interactive Agent Teams sessions."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..core.task import Task, TaskStatus, TaskType
from ..queue.file_queue import FileQueue
from ..safeguards.escalation import EscalationHandler
from ..utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# Default location for Claude teams data
CLAUDE_TEAMS_DIR = Path.home() / ".claude" / "teams"


class TeamBridge:
    """Bidirectional bridge between autonomous pipeline and interactive Agent Teams."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.teams_dir = self.workspace / ".agent-communication" / "teams"
        self.escalation_handler = EscalationHandler(enable_error_truncation=True)

    def build_escalation_context(self, task: Task) -> str:
        """Gather context from a failed task for interactive team resolution.

        Produces a structured text block with task details, error history,
        and acceptance criteria so the debug team has full context.
        """
        error_msg = task.last_error or "No error message available"
        truncated_error = self.escalation_handler.truncate_error(error_msg)

        context_parts = [
            f"## Failed Task: {task.title}",
            f"- **Task ID:** {task.id}",
            f"- **Type:** {task.type}",
            f"- **Assigned to:** {task.assigned_to}",
            f"- **Retry count:** {task.retry_count}",
        ]

        if task.context.get("jira_key"):
            context_parts.append(f"- **JIRA:** {task.context['jira_key']}")

        if task.context.get("github_repo"):
            context_parts.append(f"- **Repository:** {task.context['github_repo']}")

        context_parts.append(f"\n## Description\n{task.description}")

        if task.acceptance_criteria:
            criteria = "\n".join(f"- {c}" for c in task.acceptance_criteria)
            context_parts.append(f"\n## Acceptance Criteria\n{criteria}")

        context_parts.append(f"\n## Error\n```\n{truncated_error}\n```")

        return "\n".join(context_parts)

    def record_team_session(
        self,
        team_name: str,
        template: str,
        source_task_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Write session metadata for dashboard visibility.

        Returns:
            Path to the created session file
        """
        self.teams_dir.mkdir(parents=True, exist_ok=True)

        session_data = {
            "team_name": team_name,
            "template": template,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_task_id": source_task_id,
            "status": "active",
            **(metadata or {}),
        }

        session_file = self.teams_dir / f"{team_name}.json"
        atomic_write_json(session_file, json.dumps(session_data, indent=2))
        logger.info(f"Recorded team session: {team_name}")
        return session_file

    def handoff_to_autonomous(
        self, tasks: list[dict], workflow: str = "simple"
    ) -> list[str]:
        """Convert team output into framework Tasks and push to queue.

        Args:
            tasks: List of task dicts with at minimum 'title' and 'description'
            workflow: Workflow complexity for routing

        Returns:
            List of queued task IDs
        """
        queue = FileQueue(self.workspace)
        queued_ids = []

        for task_data in tasks:
            task_id = f"team-handoff-{time.time_ns()}-{len(queued_ids)}"

            # Route based on workflow
            if workflow == "full":
                assigned_to = "architect"
            elif workflow == "standard":
                assigned_to = "engineer"
            else:
                assigned_to = "engineer"

            task = Task(
                id=task_id,
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.PENDING,
                priority=2,
                created_by="agent-team",
                assigned_to=assigned_to,
                created_at=datetime.now(timezone.utc),
                title=task_data.get("title", "Team handoff task"),
                description=task_data.get("description", ""),
                context=task_data.get("context", {}),
                acceptance_criteria=task_data.get("acceptance_criteria", []),
            )

            queue.push(task, assigned_to)
            queued_ids.append(task_id)
            logger.info(f"Handed off task {task_id} to {assigned_to}")

        return queued_ids

    def get_active_teams(self) -> list[dict]:
        """Read active team sessions from both local and Claude teams directories."""
        sessions = []

        for teams_dir in [self.teams_dir, CLAUDE_TEAMS_DIR]:
            if not teams_dir.exists():
                continue

            for session_file in teams_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text())
                    data["source_dir"] = str(teams_dir)
                    sessions.append(data)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Could not read team session {session_file}: {e}")

        return sessions

    def mark_session_ended(self, team_name: str) -> bool:
        """Update a team session's status to 'ended'.

        Returns True if the session was found and updated.
        """
        session_file = self.teams_dir / f"{team_name}.json"
        if not session_file.exists():
            logger.warning(f"No session file found for team: {team_name}")
            return False

        try:
            data = json.loads(session_file.read_text())
            data["status"] = "ended"
            data["ended_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(session_file, json.dumps(data, indent=2))
            logger.info(f"Marked team session ended: {team_name}")
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to mark session ended for {team_name}: {e}")
            return False

    def build_team_claude_md(
        self,
        repo_path: Optional[str] = None,
        task_context: Optional[str] = None,
    ) -> str:
        """Build CLAUDE.md content for team sessions explaining pipeline integration.

        This gets written to the team's working directory so teammates
        understand how to interact with the autonomous pipeline.
        """
        team_context_path = self.workspace / "config" / "docs" / "team_context.md"

        if team_context_path.exists():
            pipeline_docs = team_context_path.read_text()
        else:
            pipeline_docs = (
                "Pipeline documentation not found. "
                "Use queue_task_for_agent MCP tool to hand off work."
            )

        parts = [
            "# Agent Team Session",
            "",
            "This is an interactive Agent Team session connected to the autonomous pipeline.",
            "",
            pipeline_docs,
        ]

        if repo_path:
            parts.append(f"\n## Working Repository\n{repo_path}")

        if task_context:
            parts.append(f"\n## Task Context\n{task_context}")

        return "\n".join(parts)
