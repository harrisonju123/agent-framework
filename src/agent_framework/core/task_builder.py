"""Shared utilities for building tasks across CLI and web interfaces."""

from datetime import datetime, timezone
from typing import Optional

from .task import Task, TaskStatus, TaskType


def build_planning_task(
    goal: str,
    workflow: str,
    github_repo: str,
    repository_name: str,
    jira_project: Optional[str] = None,
    created_by: str = "cli",
) -> Task:
    """Build a planning task for the Architect agent.

    Args:
        goal: The user's goal/request
        workflow: Workflow name ("default" or "analysis")
        github_repo: GitHub repository in owner/repo format
        repository_name: Human-readable repository name
        jira_project: JIRA project key (None if not configured)
        created_by: Source of the task (cli/web-dashboard)

    Returns:
        Task configured for architect queue
    """
    import time

    # Use jira_project or repo slug for task ID (must be filesystem-safe)
    project_id = jira_project or github_repo.replace("/", "-")
    task_id = f"planning-{project_id}-{int(time.time())}"

    # Build task context - only include jira_project if configured
    context = {
        "mode": "planning",
        "workflow": workflow,
        "github_repo": github_repo,
        "repository_name": repository_name,
        "user_goal": goal,
        "jira_available": bool(jira_project),
    }
    if jira_project:
        context["jira_project"] = jira_project

    # Build instructions based on JIRA availability
    instructions = _build_planning_instructions(goal, workflow, jira_project)

    return Task(
        id=task_id,
        type=TaskType.PLANNING,
        status=TaskStatus.PENDING,
        priority=1,
        created_by=created_by,
        assigned_to="architect",
        created_at=datetime.now(timezone.utc),
        title=f"Plan and delegate: {goal}",
        description=instructions,
        context=context,
    )


def build_analysis_task(
    repository: str,
    severity: str,
    max_issues: int,
    dry_run: bool,
    focus: Optional[str] = None,
    jira_project: Optional[str] = None,
    created_by: str = "cli",
) -> Task:
    """Build an analysis task for the Architect agent.

    Args:
        repository: GitHub repository in owner/repo format
        severity: Severity filter (all/critical/high/medium)
        max_issues: Maximum issues to report
        dry_run: If True, don't create JIRA tickets
        focus: Optional focus instructions
        jira_project: JIRA project key (None if not configured)
        created_by: Source of the task (cli/web-dashboard)

    Returns:
        Task configured for architect queue
    """
    import time

    task_id = f"analysis-{repository.replace('/', '-')}-{int(time.time())}"

    # Build task description
    description = _build_analysis_description(
        repository, severity, max_issues, dry_run, focus, jira_project
    )

    # Build context - only include jira_project if configured
    context = {
        "mode": "analysis",
        "workflow": "analysis",
        "github_repo": repository,
        "severity_filter": severity,
        "max_issues": max_issues,
        "dry_run": dry_run,
        "focus_instructions": focus,
        "jira_available": bool(jira_project),
    }
    if jira_project:
        context["jira_project"] = jira_project

    return Task(
        id=task_id,
        type=TaskType.ANALYSIS,
        status=TaskStatus.PENDING,
        priority=1,
        created_by=created_by,
        assigned_to="architect",
        created_at=datetime.now(timezone.utc),
        title=f"Analyze repository: {repository}",
        description=description,
        context=context,
    )


def _build_planning_instructions(
    goal: str, workflow: str, jira_project: Optional[str]
) -> str:
    """Build instructions for planning task based on JIRA availability."""
    jira_note = (
        "6. Create appropriate JIRA ticket to track this work"
        if jira_project
        else "6. No JIRA project configured — skip JIRA operations"
    )
    return f"""User Goal: {goal}

Instructions for Architect Agent:
1. Clone/update repository (use MultiRepoManager)
2. Search for keywords and concepts from the goal — grep for relevant function names, class names, and config keys
3. Stop exploring once you have enough context to produce a concrete plan — do not map the entire codebase
4. Produce a PlanDocument with: objectives, approach (step-by-step), files_to_modify, risks, success_criteria
5. Store plan in task.plan — the framework routes to Engineer automatically
{jira_note}"""


def _build_analysis_description(
    repository: str,
    severity: str,
    max_issues: int,
    dry_run: bool,
    focus: Optional[str],
    jira_project: Optional[str],
) -> str:
    """Build description for analysis task based on JIRA availability."""
    if jira_project:
        description = f"""Perform full repository analysis on {repository}.

Scan for security vulnerabilities, performance issues, and code quality problems.
Group findings by file/module location and create JIRA epic with subtasks.

Settings:
- Severity filter: {severity}
- Max issues: {max_issues}
- Dry run: {dry_run}
"""
    else:
        description = f"""Perform full repository analysis on {repository}.

Scan for security vulnerabilities, performance issues, and code quality problems.
Group findings by file/module location.

Note: No JIRA project configured - results will be output to logs only.

Settings:
- Severity filter: {severity}
- Max issues: {max_issues}
- Dry run: {dry_run}
"""

    if focus:
        if jira_project:
            description += f"""
## Custom Focus Instructions
{focus}

When analyzing this repository:
1. PRIORITIZE the areas and flows mentioned in the focus instructions
2. Look specifically for the issue types mentioned
3. Explore and trace the specified code flows before running static analyzers
4. Include a "Focus Area Analysis" section in the JIRA epic description
"""
        else:
            description += f"""
## Custom Focus Instructions
{focus}

When analyzing this repository:
1. PRIORITIZE the areas and flows mentioned in the focus instructions
2. Look specifically for the issue types mentioned
3. Explore and trace the specified code flows before running static analyzers
4. Output findings to logs (no JIRA available)
"""

    return description


def build_decomposed_subtask(
    parent_task: Task,
    name: str,
    description: str,
    files_to_modify: list[str],
    approach_steps: list[str],
    index: int,
    depends_on: list[str] = None,
) -> Task:
    """Build a subtask from a decomposed parent.

    ID pattern: {parent_task.id}-sub-{index}
    Inherits parent's context, priority, and workflow info.

    Args:
        parent_task: The parent task being decomposed
        name: Short name/title for the subtask
        description: Detailed description of what this subtask should accomplish
        files_to_modify: List of files this subtask will modify
        approach_steps: Step-by-step approach for implementing this subtask
        index: Subtask index (0-based)
        depends_on: Optional list of task IDs this subtask depends on

    Returns:
        Task configured as a subtask of parent_task
    """
    subtask_id = f"{parent_task.id}-sub-{index}"

    # Build context inheriting from parent
    subtask_context = {
        **parent_task.context,
        "parent_task_id": parent_task.id,
        "subtask_index": index,
        "files_to_modify": files_to_modify,
    }

    # Build plan document for the subtask
    from .task import PlanDocument
    plan = PlanDocument(
        objectives=[name],
        approach=approach_steps,
        files_to_modify=files_to_modify,
        success_criteria=[f"Complete implementation of: {name}"],
    )

    return Task(
        id=subtask_id,
        type=parent_task.type,
        status=TaskStatus.PENDING,
        priority=parent_task.priority,
        created_by="decomposer",
        assigned_to=parent_task.assigned_to,
        created_at=datetime.now(timezone.utc),
        title=name,
        description=description,
        context=subtask_context,
        depends_on=depends_on or [],
        parent_task_id=parent_task.id,
        plan=plan,
    )
