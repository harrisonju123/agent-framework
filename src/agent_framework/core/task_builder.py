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
    """Build a planning task for the Product Owner agent.

    Args:
        goal: The user's goal/request
        workflow: Workflow mode (simple/standard/full)
        github_repo: GitHub repository in owner/repo format
        repository_name: Human-readable repository name
        jira_project: JIRA project key (None if not configured)
        created_by: Source of the task (cli/web-dashboard)

    Returns:
        Task configured for product-owner queue
    """
    import time

    # Use jira_project or repository name for task ID
    project_id = jira_project or repository_name
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
        assigned_to="product-owner",
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
    """Build an analysis task for the repo-analyzer agent.

    Args:
        repository: GitHub repository in owner/repo format
        severity: Severity filter (all/critical/high/medium)
        max_issues: Maximum issues to report
        dry_run: If True, don't create JIRA tickets
        focus: Optional focus instructions
        jira_project: JIRA project key (None if not configured)
        created_by: Source of the task (cli/web-dashboard)

    Returns:
        Task configured for repo-analyzer queue
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
        assigned_to="repo-analyzer",
        created_at=datetime.now(timezone.utc),
        title=f"Analyze repository: {repository}",
        description=description,
        context=context,
    )


def _build_planning_instructions(
    goal: str, workflow: str, jira_project: Optional[str]
) -> str:
    """Build instructions for planning task based on JIRA availability."""
    if jira_project:
        return f"""User Goal: {goal}

Instructions for Product Owner Agent:
1. Clone/update repository (use MultiRepoManager)
2. Explore the codebase to understand structure
3. Validate the goal is feasible
4. Check context.workflow ('{workflow}') and route accordingly
5. Create appropriate JIRA ticket and queue tasks per workflow mode"""
    else:
        return f"""User Goal: {goal}

Instructions for Product Owner Agent:
1. Clone/update repository (use MultiRepoManager)
2. Explore the codebase to understand structure
3. Validate the goal is feasible
4. Check context.workflow ('{workflow}') and route accordingly
5. No JIRA project configured - write tasks directly to local queues
6. Generate local task IDs: local-{{workflow}}-{{timestamp}}
7. Write task JSON to: .agent-communication/queues/{{agent}}/{{task_id}}.json"""


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
