"""Generate project-level markdown summaries from completed tasks."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.task import Task, TaskStatus, TaskType, PlanDocument
from ..queue.file_queue import FileQueue


class SummaryGenerator:
    """Generate human-readable summaries from completed tasks."""

    def __init__(self, queue: FileQueue, output_dir: Path):
        self.queue = queue
        self.output_dir = Path(output_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure output directories exist."""
        (self.output_dir / "plans").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summaries").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "archives").mkdir(parents=True, exist_ok=True)

    def generate_project_summary(self, jira_key: str) -> Path:
        """
        Aggregate all completed tasks for a JIRA ticket into a summary.

        Args:
            jira_key: The JIRA issue key (e.g., "PTO-1234")

        Returns:
            Path to the generated summary file
        """
        tasks = self._collect_tasks_for_jira(jira_key)

        if not tasks:
            summary_path = self.output_dir / "summaries" / f"{jira_key}.md"
            summary_path.write_text(f"# {jira_key}\n\nNo completed tasks found.\n")
            return summary_path

        # Group tasks by type
        grouped = self._group_tasks_by_type(tasks)

        # Generate markdown
        content = self._render_project_summary(jira_key, grouped)

        summary_path = self.output_dir / "summaries" / f"{jira_key}.md"
        summary_path.write_text(content)
        return summary_path

    def generate_daily_summary(self, date: Optional[datetime] = None) -> Path:
        """
        Generate summary of all work completed on a given date.

        Args:
            date: The date to summarize (defaults to today)

        Returns:
            Path to the generated summary file
        """
        if date is None:
            date = datetime.now(timezone.utc)

        date_str = date.strftime("%Y-%m-%d")
        tasks = self._collect_tasks_for_date(date)

        if not tasks:
            summary_path = self.output_dir / "summaries" / f"daily-{date_str}.md"
            summary_path.write_text(
                f"# Daily Summary: {date_str}\n\nNo completed tasks.\n"
            )
            return summary_path

        # Group by JIRA key
        grouped_by_jira: dict[str, list[Task]] = {}
        ungrouped: list[Task] = []

        for task in tasks:
            jira_key = task.context.get("jira_key")
            if jira_key:
                grouped_by_jira.setdefault(jira_key, []).append(task)
            else:
                ungrouped.append(task)

        content = self._render_daily_summary(date_str, grouped_by_jira, ungrouped)

        summary_path = self.output_dir / "summaries" / f"daily-{date_str}.md"
        summary_path.write_text(content)
        return summary_path

    def save_plan(self, task: Task) -> Optional[Path]:
        """
        Save a task's plan document to the plans directory.

        Args:
            task: Task with a plan field

        Returns:
            Path to the saved plan, or None if no plan exists
        """
        if not task.plan:
            return None

        jira_key = task.context.get("jira_key", "unknown")
        plan_path = self.output_dir / "plans" / f"{jira_key}-{task.id}.md"
        content = self._render_plan(task)
        plan_path.write_text(content)
        return plan_path

    def _collect_tasks_for_jira(self, jira_key: str) -> list[Task]:
        """Collect all completed tasks matching a JIRA key."""
        tasks = []
        completed_dir = self.queue.completed_dir

        if not completed_dir.exists():
            return tasks

        for task_file in completed_dir.glob("*.json"):
            try:
                data = json.loads(task_file.read_text())
                task = Task(**data)
                if task.context.get("jira_key") == jira_key:
                    tasks.append(task)
            except (json.JSONDecodeError, Exception):
                continue

        # Sort by completion time
        return sorted(tasks, key=lambda t: t.completed_at or t.created_at)

    def _collect_tasks_for_date(self, date: datetime) -> list[Task]:
        """Collect all tasks completed on a specific date."""
        tasks = []
        completed_dir = self.queue.completed_dir

        if not completed_dir.exists():
            return tasks

        target_date = date.date()

        for task_file in completed_dir.glob("*.json"):
            try:
                data = json.loads(task_file.read_text())
                task = Task(**data)
                if task.completed_at and task.completed_at.date() == target_date:
                    tasks.append(task)
            except (json.JSONDecodeError, Exception):
                continue

        return sorted(tasks, key=lambda t: t.completed_at or t.created_at)

    def _group_tasks_by_type(self, tasks: list[Task]) -> dict[str, list[Task]]:
        """Group tasks into planning, implementation, and testing categories."""
        grouped: dict[str, list[Task]] = {
            "planning": [],
            "implementation": [],
            "testing": [],
            "other": [],
        }

        planning_types = {TaskType.ARCHITECTURE, TaskType.PLANNING}
        impl_types = {
            TaskType.IMPLEMENTATION,
            TaskType.FIX,
            TaskType.BUGFIX,
            TaskType.BUG_FIX,
            TaskType.ENHANCEMENT,
        }
        testing_types = {TaskType.TESTING, TaskType.VERIFICATION, TaskType.REVIEW}

        for task in tasks:
            task_type = TaskType(task.type) if isinstance(task.type, str) else task.type

            if task_type in planning_types:
                grouped["planning"].append(task)
            elif task_type in impl_types:
                grouped["implementation"].append(task)
            elif task_type in testing_types:
                grouped["testing"].append(task)
            else:
                grouped["other"].append(task)

        return grouped

    def _render_project_summary(
        self, jira_key: str, grouped: dict[str, list[Task]]
    ) -> str:
        """Render a project summary as markdown."""
        lines = [f"# {jira_key} - Project Summary", ""]
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        # Overview stats
        total = sum(len(tasks) for tasks in grouped.values())
        lines.append("## Overview")
        lines.append(f"- Total tasks completed: {total}")
        lines.append(f"- Planning tasks: {len(grouped['planning'])}")
        lines.append(f"- Implementation tasks: {len(grouped['implementation'])}")
        lines.append(f"- Testing tasks: {len(grouped['testing'])}")
        lines.append("")

        # Planning section with plans
        if grouped["planning"]:
            lines.append("## Planning")
            for task in grouped["planning"]:
                lines.append(f"### {task.title}")
                if task.plan:
                    lines.extend(self._render_plan_inline(task.plan))
                elif task.result_summary:
                    lines.append(task.result_summary)
                lines.append("")

        # Implementation section
        if grouped["implementation"]:
            lines.append("## Implementation")
            for task in grouped["implementation"]:
                lines.append(f"### {task.title}")
                if task.result_summary:
                    lines.append(task.result_summary)
                if task.deliverables:
                    lines.append("")
                    lines.append("**Deliverables:**")
                    for d in task.deliverables:
                        lines.append(f"- {d}")
                lines.append("")

        # Testing section
        if grouped["testing"]:
            lines.append("## Testing & Verification")
            for task in grouped["testing"]:
                lines.append(f"### {task.title}")
                if task.result_summary:
                    lines.append(task.result_summary)
                lines.append("")

        # Other tasks
        if grouped["other"]:
            lines.append("## Other Tasks")
            for task in grouped["other"]:
                lines.append(f"- **{task.title}**: {task.result_summary or 'Completed'}")
            lines.append("")

        return "\n".join(lines)

    def _render_daily_summary(
        self,
        date_str: str,
        grouped_by_jira: dict[str, list[Task]],
        ungrouped: list[Task],
    ) -> str:
        """Render a daily summary as markdown."""
        lines = [f"# Daily Summary: {date_str}", ""]

        total = sum(len(tasks) for tasks in grouped_by_jira.values()) + len(ungrouped)
        lines.append(f"Total tasks completed: {total}")
        lines.append("")

        # Tasks grouped by JIRA ticket
        if grouped_by_jira:
            lines.append("## By JIRA Ticket")
            for jira_key, tasks in sorted(grouped_by_jira.items()):
                lines.append(f"### {jira_key}")
                for task in tasks:
                    status = "✓" if task.status == TaskStatus.COMPLETED else "✗"
                    lines.append(f"- {status} {task.title} ({task.type})")
                lines.append("")

        # Ungrouped tasks
        if ungrouped:
            lines.append("## Other Tasks")
            for task in ungrouped:
                status = "✓" if task.status == TaskStatus.COMPLETED else "✗"
                lines.append(f"- {status} {task.title} ({task.type})")
            lines.append("")

        return "\n".join(lines)

    def _render_plan(self, task: Task) -> str:
        """Render a task's plan as standalone markdown."""
        if not task.plan:
            return ""

        jira_key = task.context.get("jira_key", "unknown")
        lines = [f"# Plan: {task.title}", ""]
        lines.append(f"JIRA: {jira_key}")
        lines.append(f"Task ID: {task.id}")
        lines.append(f"Created: {task.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        lines.extend(self._render_plan_inline(task.plan))
        return "\n".join(lines)

    def _render_plan_inline(self, plan: PlanDocument) -> list[str]:
        """Render a PlanDocument as inline markdown lines."""
        lines = []

        lines.append("#### Objectives")
        for obj in plan.objectives:
            lines.append(f"- {obj}")
        lines.append("")

        lines.append("#### Approach")
        for i, step in enumerate(plan.approach, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        if plan.risks:
            lines.append("#### Risks")
            for risk in plan.risks:
                lines.append(f"- {risk}")
            lines.append("")

        lines.append("#### Success Criteria")
        for criterion in plan.success_criteria:
            lines.append(f"- {criterion}")
        lines.append("")

        if plan.files_to_modify:
            lines.append("#### Files to Modify")
            for file in plan.files_to_modify:
                lines.append(f"- `{file}`")
            lines.append("")

        if plan.dependencies:
            lines.append("#### Dependencies")
            for dep in plan.dependencies:
                lines.append(f"- {dep}")
            lines.append("")

        return lines
