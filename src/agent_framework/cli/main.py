"""Main CLI for agent framework."""

import asyncio
import os
import re
import subprocess
from pathlib import Path

# Load .env file into environment before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment

import click
from rich.console import Console
from rich.table import Table

from ..core.config import (
    load_config,
    load_agents,
    load_jira_config,
    load_github_config,
)
from ..core.orchestrator import Orchestrator
from ..core.task import Task, TaskStatus, TaskType
from ..integrations.jira.client import JIRAClient
from ..integrations.github.client import GitHubClient
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker
from ..workspace.multi_repo_manager import MultiRepoManager
from ..workspace.worktree_manager import WorktreeManager
from .team_commands import team


console = Console()


@click.group()
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.pass_context
def cli(ctx, workspace):
    """Agent Framework - AI-powered JIRA ticket automation."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = Path(workspace)


cli.add_command(team)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize a new agent framework project."""
    workspace = ctx.obj["workspace"]
    console.print("[bold green]Initializing agent framework...[/]")

    # Create directory structure
    config_dir = workspace / "config"
    config_dir.mkdir(exist_ok=True)

    # Copy example configs
    from pathlib import Path as P
    import shutil

    example_configs = [
        "agents.yaml.example",
        "jira.yaml.example",
        "github.yaml.example",
    ]

    for example in example_configs:
        src = P(__file__).parent.parent.parent.parent / "config" / example
        dst = config_dir / example
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            console.print(f"  Created {dst}")

    console.print("[green]✓ Initialization complete![/]")
    console.print("\nNext steps:")
    console.print("1. Copy config/*.example files to remove .example extension")
    console.print("2. Edit config files with your JIRA/GitHub credentials")
    console.print("3. Run 'agent start' to start agents")


@cli.command()
@click.option("--repo", "-r", required=True, help="GitHub repo (owner/repo)")
@click.option("--severity", type=click.Choice(["all", "critical", "high", "medium"]), default="high",
              help="Minimum severity to include")
@click.option("--max-issues", default=50, help="Max subtasks to create")
@click.option("--dry-run", is_flag=True, help="Show findings without creating JIRA")
@click.option("--focus", "-f", help="Custom focus instructions for the analysis (e.g., 'review PTO accrual flow for tech debt')")
@click.pass_context
def analyze(ctx, repo, severity, max_issues, dry_run, focus):
    """Analyze repository and create JIRA epic with findings.

    Scans the entire repository for issues (security, performance, code quality)
    and creates a JIRA epic with subtasks grouped by file/module location.

    Examples:
        agent analyze --repo justworkshr/pto
        agent analyze --repo justworkshr/pto --severity critical --dry-run
        agent analyze --repo justworkshr/pto --max-issues 30
    """
    workspace = ctx.obj["workspace"]

    # Validate repo format
    if "/" not in repo:
        console.print(f"[red]Invalid repo format: {repo}[/]")
        console.print("[dim]Expected format: owner/repo (e.g., justworkshr/pto)[/]")
        return

    console.print(f"[bold cyan]Repository Analysis: {repo}[/]")
    console.print()

    # Load config
    framework_config = load_config(workspace / "config" / "agent-framework.yaml")

    # Find matching repository in config
    matching_repo = None
    for registered_repo in framework_config.repositories:
        if registered_repo.github_repo == repo:
            matching_repo = registered_repo
            break

    if not matching_repo:
        console.print(f"[yellow]Warning: Repository {repo} not found in config[/]")
        console.print("[dim]Available repositories:[/]")
        for r in framework_config.repositories:
            console.print(f"  - {r.github_repo} ({r.jira_project})")

        # Prompt for JIRA project
        jira_project = click.prompt("Enter JIRA project key for epic creation", type=str)
    else:
        jira_project = matching_repo.jira_project
        console.print(f"[green]✓ Found in config: JIRA project {jira_project}[/]")

    console.print()
    console.print(f"[bold]Analysis Settings:[/]")
    console.print(f"  Repository: {repo}")
    console.print(f"  JIRA Project: {jira_project}")
    console.print(f"  Severity Filter: {severity}")
    console.print(f"  Max Issues: {max_issues}")
    console.print(f"  Dry Run: {dry_run}")
    if focus:
        console.print(f"  Focus: {focus}")
    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN: Will show findings without creating JIRA tickets[/]")
        console.print()

    # Create analysis task for architect agent
    import time
    from datetime import datetime

    task_id = f"analysis-{repo.replace('/', '-')}-{int(time.time())}"

    # Build task description
    description = f"""Perform full repository analysis on {repo}.

Scan for security vulnerabilities, performance issues, and code quality problems.
Group findings by file/module location and create JIRA epic with subtasks.

Settings:
- Severity filter: {severity}
- Max issues: {max_issues}
- Dry run: {dry_run}
"""

    # Add focus instructions if provided
    if focus:
        description += f"""
## Custom Focus Instructions
{focus}

When analyzing this repository:
1. PRIORITIZE the areas and flows mentioned in the focus instructions
2. Look specifically for the issue types mentioned (tech debt, code style, security, etc.)
3. Explore and trace the specified code flows before running static analyzers
4. Include a "Focus Area Analysis" section in the JIRA epic description
"""

    # Build task context
    task_context = {
        "mode": "analysis",
        "workflow": "analysis",
        "github_repo": repo,
        "jira_project": jira_project,
        "severity_filter": severity,
        "max_issues": max_issues,
        "dry_run": dry_run,
    }

    # Add focus instructions to context
    if focus:
        task_context["focus_instructions"] = focus

    task = Task(
        id=task_id,
        type=TaskType.ANALYSIS,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="cli",
        assigned_to="architect",
        created_at=datetime.utcnow(),
        title=f"Analyze repository: {repo}",
        description=description,
        context=task_context,
    )

    # Queue the analysis task
    queue = FileQueue(workspace)
    queue.push(task, "architect")

    console.print(f"[green]✓ Analysis task queued: {task_id}[/]")

    # Ensure agents are running
    orchestrator = Orchestrator(workspace)
    running = orchestrator.get_running_agents()

    if running:
        console.print(f"[green]✓ Agents already running: {', '.join(running)}[/]")
    else:
        console.print("[bold]Starting agents...[/]")
        orchestrator.setup_signal_handlers()
        orchestrator.spawn_all_agents()
        console.print("[green]✓ Agents started[/]")

    console.print()
    console.print(f"[bold cyan]Analysis in progress...[/]")
    console.print(f"[dim]📋 Monitor progress: agent status --watch[/]")
    console.print(f"[dim]📝 View logs: tail -f logs/architect.log[/]")
    console.print()

    if not dry_run:
        console.print("[yellow]The Architect agent will:[/]")
        console.print(f"  1. Clone/update {repo}")
        console.print(f"  2. Detect languages and run static analyzers")
        console.print(f"  3. Aggregate findings by file/module")
        console.print(f"  4. Create JIRA epic in project {jira_project}")
        console.print(f"  5. Create subtasks for each file group")
    else:
        console.print("[yellow]The Architect agent will:[/]")
        console.print(f"  1. Clone/update {repo}")
        console.print(f"  2. Detect languages and run static analyzers")
        console.print(f"  3. Aggregate findings by file/module")
        console.print(f"  4. Generate analysis report (no JIRA tickets)")


@cli.command()
@click.option("--project", "-p", help="JIRA project key")
@click.option("--max", "-n", default=10, help="Max tickets to pull")
@click.pass_context
def pull(ctx, project, max):
    """Pull unassigned tickets from JIRA backlog."""
    workspace = ctx.obj["workspace"]

    # Load configs
    jira_config = load_jira_config(workspace / "config" / "jira.yaml")
    if not jira_config:
        console.print("[red]Error: JIRA config not found[/]")
        return

    # Create JIRA client
    jira_client = JIRAClient(jira_config)
    console.print(f"[bold]Pulling unassigned tickets from {project or jira_config.project}...[/]")

    try:
        issues = jira_client.pull_unassigned_tickets(max_results=max)
        console.print(f"[bold]Found {len(issues)} unassigned tickets:[/]")

        table = Table()
        table.add_column("Key")
        table.add_column("Summary")
        table.add_column("Type")

        for issue in issues:
            table.add_row(
                issue.key,
                issue.fields.summary,
                issue.fields.issuetype.name,
            )

        console.print(table)

        if click.confirm("Queue these tickets for processing?"):
            # Load agent config to determine assignment
            agents_config = load_agents(workspace / "config" / "agents.yaml")
            queue = FileQueue(workspace)

            for issue in issues:
                # Assign to architect for planning first
                assigned_to = next(
                    (a.queue for a in agents_config if a.enabled and a.id == "architect"),
                    next(
                        (a.queue for a in agents_config if a.enabled),
                        "engineer"
                    )
                )
                task = jira_client.issue_to_task(issue, assigned_to)
                queue.push(task, assigned_to)

            console.print(f"[green]✓ Queued {len(issues)} tickets[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--no-dashboard", is_flag=True, help="Skip live dashboard")
@click.option("--epic", "-e", help="JIRA epic key to process (e.g., PROJ-100)")
@click.option("--parallel", "-p", is_flag=True, help="Process epic tickets in parallel (requires worktrees)")
@click.option("--auto-approve", is_flag=True, help="Skip plan checkpoint, run fully autonomous")
@click.pass_context
def work(ctx, no_dashboard, epic, parallel, auto_approve):
    """Interactive mode: describe what to build, delegate to Architect agent.

    With --epic: Process all tickets in an existing JIRA epic.
    Without --epic: Describe a new feature to implement.

    Use --parallel with --epic to process multiple tickets concurrently.
    Use --auto-approve to skip the plan review checkpoint.
    """
    workspace = ctx.obj["workspace"]

    console.print("[bold cyan]🤖 Agent Framework - Interactive Mode[/]")
    console.print()

    # Load config
    framework_config = load_config(workspace / "config" / "agent-framework.yaml")

    # Handle epic processing mode
    if epic:
        _handle_epic_mode(ctx, workspace, framework_config, epic, no_dashboard, parallel, auto_approve)
        return

    # Check if repos are registered
    if not framework_config.repositories:
        console.print("[red]No repositories registered in config/agent-framework.yaml[/]")
        console.print("[yellow]Add a 'repositories' section with your repos[/]")
        return

    # Step 1: Ask what to work on
    console.print("[bold]What would you like to work on?[/]")
    goal = click.prompt("", type=str)

    if not goal or not goal.strip():
        console.print("[red]Please provide a goal description[/]")
        return

    # Step 2: Select repository
    console.print("\n[bold]Which repository?[/]")
    for i, repo in enumerate(framework_config.repositories, 1):
        jira_info = f"(JIRA: {repo.jira_project})" if repo.jira_project else "(local tasks)"
        console.print(f"  {i}. [cyan]{repo.github_repo}[/] {jira_info}")

    repo_idx = click.prompt(
        "Select repository",
        type=click.IntRange(1, len(framework_config.repositories))
    )
    selected_repo = framework_config.repositories[repo_idx - 1]

    if selected_repo.jira_project:
        console.print(f"\n[green]✓[/] Selected: [bold]{selected_repo.github_repo}[/] (JIRA: {selected_repo.jira_project})")
    else:
        console.print(f"\n[green]✓[/] Selected: [bold]{selected_repo.github_repo}[/]")
        console.print("[dim]No JIRA project configured - using local task tracking[/]")

    workflow = "default_auto" if auto_approve else "default"

    # Step 3: Create planning task for Architect
    from ..core.task_builder import build_planning_task

    task = build_planning_task(
        goal=goal,
        workflow=workflow,
        github_repo=selected_repo.github_repo,
        repository_name=selected_repo.name,
        jira_project=selected_repo.jira_project,
        created_by="cli",
    )

    # Queue the planning task
    queue = FileQueue(workspace)
    queue.push(task, "architect")

    console.print(f"\n[green]✓[/] Task queued: Analyze goal and create JIRA epic with breakdown")

    if not auto_approve:
        console.print("[dim]Workflow will pause after architect plans for your review.[/]")
        console.print("[dim]Run 'agent approve <task-id>' to proceed to implementation.[/]")

    # Step 4: Ensure agents are running (don't block)
    orchestrator = Orchestrator(workspace)

    # Check if agents already running
    running = orchestrator.get_running_agents()

    if running:
        console.print(f"\n[green]✓[/] Agents already running: {', '.join(running)}")
    else:
        console.print("\n[bold]Starting agents...[/]")
        orchestrator.setup_signal_handlers()
        orchestrator.spawn_all_agents()
        console.print("[green]✓[/] Agents started")

    if no_dashboard:
        # Print status without launching dashboard
        if selected_repo.jira_project:
            console.print(f"\n[bold cyan]🎯 Epic will be created in JIRA project: {selected_repo.jira_project}[/]")
        else:
            console.print(f"\n[bold cyan]🎯 Tasks will be created in local queues[/]")
        console.print(f"[dim]📋 Monitor progress: agent status --watch[/]")
        console.print(f"[dim]📝 View logs: tail -f logs/architect.log[/]")
        console.print()
        console.print("[yellow]The Architect agent will:[/]")
        console.print(f"  1. Clone/update {selected_repo.github_repo} to ~/.agent-workspaces/")
        console.print(f"  2. Analyze the codebase")
        if selected_repo.jira_project:
            console.print(f"  3. Create JIRA epic in project {selected_repo.jira_project}")
            console.print(f"  4. Break down into architect → engineer → qa subtasks")
        else:
            console.print(f"  3. Create tasks in local queues (.agent-communication/queues/)")
            console.print(f"  4. Route to appropriate agents based on workflow")
        console.print(f"  5. Queue tasks for other agents")
        console.print()
        console.print("[dim]Dashboard disabled. Use 'agent status --watch' to monitor.[/]")
    else:
        # Start live dashboard
        console.print("\n[bold cyan]✓ Task queued, starting live dashboard...[/]")
        console.print("[dim]Press Ctrl+C to exit dashboard[/]\n")

        from .dashboard import AgentDashboard

        dashboard = AgentDashboard(workspace)
        try:
            asyncio.run(dashboard.run())
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard closed[/]")
            console.print("[dim]Agents continue running in background. Use 'agent status --watch' to monitor.[/]")


@cli.command()
@click.argument("ticket_id")
@click.option("--agent", "-a", help="Specific agent to use")
@click.pass_context
def run(ctx, ticket_id, agent):
    """Work on a specific JIRA ticket."""
    workspace = ctx.obj["workspace"]

    console.print(f"[bold]Processing ticket: {ticket_id}[/]")

    # Load configs
    jira_config = load_jira_config(workspace / "config" / "jira.yaml")
    if not jira_config:
        console.print("[red]Error: JIRA config not found[/]")
        return

    # Create JIRA client
    jira_client = JIRAClient(jira_config)

    try:
        # Fetch ticket
        issue = jira_client.jira.issue(ticket_id)

        # Load agent config
        agents_config = load_agents(workspace / "config" / "agents.yaml")

        # Assign to agent (default to architect for planning first)
        assigned_to = agent or next(
            (a.queue for a in agents_config if a.enabled and a.id == "architect"),
            next(
                (a.queue for a in agents_config if a.enabled),
                "engineer"
            )
        )

        # Create task
        task = jira_client.issue_to_task(issue, assigned_to)

        # Queue the task
        queue = FileQueue(workspace)
        queue.push(task, assigned_to)

        console.print(f"[green]✓ Task queued for {assigned_to}[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--port", "-p", default=8080, help="Server port (default: 8080)")
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser")
@click.option("--dev", is_flag=True, help="Run in dev mode (Vite hot reload)")
@click.pass_context
def dashboard(ctx, port, no_browser, dev):
    """Start the web dashboard.

    Opens a browser-based dashboard for monitoring and controlling agents.
    Uses WebSocket for real-time updates.

    Examples:
        agent dashboard              # Start on port 8080, open browser
        agent dashboard --port 9000  # Use custom port
        agent dashboard --dev        # Dev mode with Vite hot reload
        agent dashboard --no-browser # Don't auto-open browser
    """
    workspace = ctx.obj["workspace"]

    console.print("[bold cyan]Starting Web Dashboard[/]")
    console.print(f"[dim]Server: http://localhost:{port}[/]")

    if dev:
        console.print("[yellow]Running in development mode[/]")
        console.print("[dim]Make sure to run 'npm run dev' in frontend directory[/]")

    from ..web.server import run_dashboard_server

    try:
        run_dashboard_server(
            workspace=workspace,
            port=port,
            open_browser=not no_browser,
            dev_mode=dev,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/]")


@cli.command()
@click.pass_context
def doctor(ctx):
    """Run health checks to validate system configuration.

    Checks:
    - Config files exist and are valid
    - Credentials are set and working
    - JIRA/GitHub connectivity
    - Directory structure
    - Agent definitions

    Examples:
        agent doctor                    # Run all health checks
    """
    workspace = ctx.obj["workspace"]

    console.print("[bold cyan]Running system health checks...[/]")
    console.print()

    from ..health.checker import HealthChecker, CheckStatus

    checker = HealthChecker(workspace)
    results = checker.run_all_checks()

    # Display results
    passed = sum(1 for r in results if r.status == CheckStatus.PASSED)
    failed = sum(1 for r in results if r.status == CheckStatus.FAILED)
    warnings = sum(1 for r in results if r.status == CheckStatus.WARNING)

    for result in results:
        icon = {
            CheckStatus.PASSED: "✓",
            CheckStatus.FAILED: "✗",
            CheckStatus.WARNING: "⚠",
            CheckStatus.SKIPPED: "⊘"
        }[result.status]

        color = {
            CheckStatus.PASSED: "green",
            CheckStatus.FAILED: "red",
            CheckStatus.WARNING: "yellow",
            CheckStatus.SKIPPED: "dim"
        }[result.status]

        console.print(f"[{color}]{icon} {result.name}[/]: {result.message}")

        if result.fix_action:
            console.print(f"  [dim]→ {result.fix_action}[/]")

        if result.documentation:
            console.print(f"  [dim]📖 {result.documentation}[/]")

        console.print()

    # Summary
    console.print("─" * 50)
    if failed == 0 and warnings == 0:
        console.print(f"[bold green]✓ All checks passed ({passed}/{len(results)})[/]")
        console.print("\n[dim]System is ready. Run 'agent start' to begin.[/]")
    elif failed == 0:
        console.print(f"[bold yellow]⚠ {warnings} warning(s), {passed} passed[/]")
        console.print("\n[dim]System will work but some features may be limited.[/]")
    else:
        console.print(f"[bold red]✗ {failed} critical issue(s), {warnings} warning(s)[/]")
        console.print("\n[red]Fix critical issues before starting agents.[/]")
        console.print("[dim]Run 'agent dashboard' and click Setup to reconfigure.[/]")


@cli.command()
@click.option("--watch", "-w", is_flag=True, help="Watch mode: auto-refresh every 2s")
@click.pass_context
def status(ctx, watch):
    """Show agent status and activity."""
    workspace = ctx.obj["workspace"]

    if watch:
        # Launch dashboard in watch mode
        from .dashboard import AgentDashboard

        console.print("[bold cyan]Agent Status - Watch Mode[/]")
        console.print("[dim]Press Ctrl+C to exit[/]\n")

        dashboard = AgentDashboard(workspace)
        try:
            asyncio.run(dashboard.run())
        except KeyboardInterrupt:
            console.print("\n[yellow]Exited watch mode[/]")


@cli.command()
@click.pass_context
def analytics(ctx):
    """Show analytics dashboard with performance metrics and failure analysis."""
    workspace = ctx.obj["workspace"]

    from .analytics_dashboard import AnalyticsDashboard

    console.print("[bold cyan]Agent Analytics Dashboard[/]")
    console.print("[dim]Press Ctrl+C to exit • Tab to switch views[/]\n")

    dashboard = AnalyticsDashboard(workspace)
    try:
        asyncio.run(dashboard.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Analytics dashboard closed[/]")
    else:
        # One-time status display (legacy table view)
        console.print("[bold]Agent Framework Status[/]")

        try:
            # Load agent configs
            agents_config = load_agents(workspace / "config" / "agents.yaml")
            queue = FileQueue(workspace)

            table = Table()
            table.add_column("Agent")
            table.add_column("Status")
            table.add_column("Queue Count")

            for agent_def in agents_config:
                stats = queue.get_queue_stats(agent_def.queue)
                status = "enabled" if agent_def.enabled else "disabled"

                table.add_row(
                    agent_def.name,
                    f"[green]{status}[/]" if agent_def.enabled else f"[dim]{status}[/]",
                    str(stats["count"]),
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--epic", "-e", required=True, help="JIRA epic key (e.g., PROJ-100)")
@click.pass_context
def summary(ctx, epic):
    """Show progress summary for an epic's tickets.

    Displays a table with all tickets in the epic, their current status,
    associated PRs, and error details for failed tasks.
    """
    workspace = ctx.obj["workspace"]

    # Validate epic key format
    if "-" not in epic:
        console.print(f"[red]Invalid epic key format: {epic}[/]")
        console.print("[dim]Expected format: PROJ-123[/]")
        return

    console.print(f"[bold]Epic Summary: {epic}[/]")
    console.print()

    # Load JIRA config
    jira_config = load_jira_config(workspace / "config" / "jira.yaml")
    if not jira_config:
        console.print("[red]Error: JIRA config not found[/]")
        return

    # Create JIRA client and get epic info
    jira_client = JIRAClient(jira_config)

    try:
        epic_data = jira_client.get_epic_with_subtasks(epic)
        epic_issue = epic_data["epic"]
        issues = epic_data["issues"]

        console.print(f"[bold cyan]Epic: {epic} - {epic_issue.fields.summary}[/]")
        console.print()

    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch epic from JIRA: {e}[/]")
        console.print("[dim]Showing local task data only...[/]")
        console.print()
        epic_issue = None
        issues = []

    # Get tasks from local queue
    queue = FileQueue(workspace)
    tasks_by_status = queue.get_tasks_by_epic(epic)

    # Calculate totals
    total_pending = len(tasks_by_status["pending"])
    total_in_progress = len(tasks_by_status["in_progress"])
    total_completed = len(tasks_by_status["completed"])
    total_failed = len(tasks_by_status["failed"])
    total_tasks = total_pending + total_in_progress + total_completed + total_failed

    if total_tasks == 0:
        console.print("[yellow]No tasks found for this epic.[/]")
        console.print("[dim]Make sure tasks have epic_key in their context.[/]")
        return

    # Status summary
    status_text = f"Status: "
    if total_in_progress > 0:
        status_text += f"[yellow]IN_PROGRESS[/yellow]"
    elif total_failed > 0 and total_pending == 0:
        status_text += f"[red]FAILED[/red]"
    elif total_completed == total_tasks:
        status_text += f"[green]COMPLETED[/green]"
    else:
        status_text += f"[cyan]PENDING[/cyan]"

    status_text += f" ({total_completed}/{total_tasks} tickets)"
    console.print(status_text)
    console.print()

    # Build task table
    table = Table(expand=True)
    table.add_column("Ticket", style="cyan", width=12)
    table.add_column("Title", width=35)
    table.add_column("Status", width=12)
    table.add_column("PR", width=15)

    # Combine all tasks and sort by epic_position if available
    all_tasks = (
        tasks_by_status["completed"] +
        tasks_by_status["in_progress"] +
        tasks_by_status["pending"] +
        tasks_by_status["failed"]
    )
    all_tasks.sort(key=lambda t: t.context.get("epic_position", 999))

    prs_ready = []
    failed_tasks = []

    for task in all_tasks:
        jira_key = task.context.get("jira_key", task.id[:12])
        title = task.title[:32] + "..." if len(task.title) > 32 else task.title
        pr_url = task.context.get("pr_url", "")

        # Format status
        if task.status == TaskStatus.COMPLETED:
            status_str = "✓ Done"
            status_style = "green"
            if pr_url:
                # Extract PR number from URL
                pr_num = pr_url.split("/")[-1] if "/" in pr_url else pr_url
                pr_display = f"PR #{pr_num}"
                prs_ready.append(pr_url)
            else:
                pr_display = "-"
        elif task.status == TaskStatus.IN_PROGRESS:
            status_str = "⏳ Running"
            status_style = "yellow"
            pr_display = "-"
        elif task.status == TaskStatus.FAILED:
            status_str = "✗ Failed"
            status_style = "red"
            pr_display = "-"
            failed_tasks.append(task)
        else:  # PENDING
            status_str = "⏸ Queued"
            status_style = "dim"
            pr_display = "-"

        table.add_row(
            jira_key,
            title,
            f"[{status_style}]{status_str}[/{status_style}]",
            pr_display
        )

    console.print(table)
    console.print()

    # Show failed tasks with error details
    if failed_tasks:
        console.print(f"[bold red]Failed Tasks ({len(failed_tasks)}):[/]")
        for task in failed_tasks:
            jira_key = task.context.get("jira_key", task.id[:12])
            console.print(f"  [red]{jira_key}[/]: {task.title}")
            if task.last_error:
                error_preview = task.last_error[:100] + "..." if len(task.last_error) > 100 else task.last_error
                console.print(f"  └─ Error: {error_preview}", style="dim")
            console.print(f"  └─ Retry: [cyan]agent retry {jira_key}[/]")
        console.print()

    # Show PRs ready for review
    if prs_ready:
        console.print(f"[bold green]PRs Ready for Review: {len(prs_ready)}[/]")
        for pr_url in prs_ready:
            console.print(f"  - {pr_url}")
        console.print()


@cli.command()
@click.argument("task_id")
@click.option("--hint", "-h", required=True, help="Human guidance hint to inject for retry")
@click.pass_context
def guide(ctx, task_id, hint):
    """Inject human guidance into a failed task and retry.

    This command allows you to provide specific guidance to help the agent
    overcome a failure. The hint will be injected into the task context and
    the task will be retried with this additional information.

    Examples:
        agent guide task-123 --hint "The API endpoint changed to /v2/users"
        agent guide escalation-456 --hint "Use authentication header X-API-Key instead of Bearer"
    """
    workspace = ctx.obj["workspace"]
    queue = FileQueue(workspace)

    # Try to find the task (could be in failed, escalation, or completed)
    task = queue.get_failed_task(task_id)

    if not task:
        # Check if it's an escalation task
        escalations_dir = workspace / ".agent-communication" / "escalations"
        if escalations_dir.exists():
            escalation_file = escalations_dir / f"{task_id}.json"
            if escalation_file.exists():
                import json
                from ..core.task import Task
                with open(escalation_file, 'r') as f:
                    task_data = json.load(f)
                task = Task(**task_data)

    if not task:
        console.print(f"[red]Error: Task {task_id} not found[/]")
        console.print("[dim]Task must be in FAILED status or logged as escalation[/]")
        return

    # Show current status
    console.print(f"[bold]Task: {task.title}[/]")
    console.print(f"Status: {task.status}")
    console.print(f"Retry count: {task.retry_count}")
    console.print()

    # Show escalation report if available
    if task.escalation_report:
        console.print(f"[bold cyan]Escalation Report:[/]")
        console.print(f"Pattern: {task.escalation_report.failure_pattern}")
        console.print(f"Hypothesis: {task.escalation_report.root_cause_hypothesis}")
        console.print()

    console.print(f"[yellow]Human Guidance:[/] {hint}")
    console.print()

    if not click.confirm("Inject this guidance and retry task?"):
        console.print("[yellow]Cancelled[/]")
        return

    # Inject human guidance
    if task.escalation_report:
        task.escalation_report.human_guidance = hint
    else:
        # Create basic escalation report with guidance
        from ..core.task import EscalationReport
        task.escalation_report = EscalationReport(
            task_id=task.id,
            original_title=task.title,
            total_attempts=task.retry_count,
            attempt_history=task.retry_attempts,
            root_cause_hypothesis="Human guidance provided",
            suggested_interventions=[hint],
            human_guidance=hint,
        )

    # Add to context for agent visibility
    task.context["human_guidance"] = hint
    task.context["guided_retry"] = True

    # Reset retry count to give it fresh attempts with guidance
    task.retry_count = 0
    task.status = TaskStatus.PENDING
    task.notes.append(f"Human guidance injected: {hint}")

    if not task.assigned_to:
        console.print("[red]Error: Task has no assigned_to agent — cannot re-queue[/]")
        return

    # Re-queue the task
    queue.requeue_task(task)

    console.print(f"[green]✓ Guidance injected and task re-queued to {task.assigned_to}[/]")
    console.print(f"[dim]The agent will receive your guidance on next attempt[/]")
    console.print(f"[dim]Monitor with: agent status --watch[/]")


@cli.command()
@click.argument("task_id")
@click.option("--reason", "-r", default=None, help="Reason for cancellation")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def cancel(ctx, task_id, reason, yes):
    """Cancel a queued or in-progress task so it won't be retried.

    TASK_ID can be a task ID or JIRA key (e.g., PROJ-104).
    Cancelled tasks are moved out of the queue and will not be retried
    even if the subprocess is killed.

    Examples:
        agent cancel task-123
        agent cancel PROJ-104 --reason "duplicate task"
        agent cancel task-123 --yes          # Skip confirmation
    """
    workspace = ctx.obj["workspace"]
    queue = FileQueue(workspace)

    task = queue.find_task(task_id)

    if not task:
        console.print(f"[red]Error: Task '{task_id}' not found in any queue[/]")
        return

    jira_key = task.context.get("jira_key", task.id)
    console.print(f"[bold]Task: {jira_key} - {task.title}[/]")
    console.print(f"Status: {task.status}")
    console.print(f"Assigned to: {task.assigned_to}")
    console.print()

    if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
        console.print(f"[yellow]Task is already {task.status} — nothing to cancel[/]")
        return

    if not yes and not click.confirm("Cancel this task?"):
        console.print("[yellow]Aborted[/]")
        return

    cancelled_by = os.getenv("USER", "cli")
    task.mark_cancelled(cancelled_by, reason)

    # Persist the updated status so the agent sees it on next poll
    queue.update(task)

    console.print(f"[green]Task {jira_key} cancelled[/]")
    if reason:
        console.print(f"[dim]Reason: {reason}[/]")
    console.print(f"[dim]If the agent is mid-execution, it will skip retry on exit.[/]")


@cli.command()
@click.argument("identifier", required=False)
@click.option("--reset-retries", is_flag=True, help="Reset retry count to 0")
@click.option("--all", "retry_all", is_flag=True, help="Retry all failed tasks")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation for bulk operations")
@click.pass_context
def retry(ctx, identifier, reset_retries, retry_all, yes):
    """Retry a failed task.

    IDENTIFIER can be either a task ID or a JIRA key (e.g., PROJ-104).
    The task will be reset to PENDING status and re-queued for processing.

    Examples:
        agent retry PROJ-104              # Retry a specific task
        agent retry PROJ-104 --reset-retries  # Retry and reset retry count
        agent retry --all                 # Retry all failed tasks
        agent retry --all --yes           # Retry all without confirmation
    """
    workspace = ctx.obj["workspace"]
    queue = FileQueue(workspace)

    if retry_all:
        # Retry all failed tasks
        failed_tasks = queue.get_all_failed()

        if not failed_tasks:
            console.print("[green]No failed tasks to retry[/]")
            return

        # Show tasks to be retried
        console.print(f"[bold]Found {len(failed_tasks)} failed tasks:[/]")
        for task in failed_tasks[:10]:  # Show first 10
            jira_key = task.context.get("jira_key", task.id[:12])
            console.print(f"  • {jira_key} - {task.title[:40]}...")
        if len(failed_tasks) > 10:
            console.print(f"  ... and {len(failed_tasks) - 10} more")
        console.print()

        # Confirm unless --yes flag is provided
        if not yes:
            if not click.confirm(f"Retry all {len(failed_tasks)} tasks?"):
                console.print("[yellow]Cancelled[/]")
                return

        console.print(f"[bold]Retrying {len(failed_tasks)} failed tasks...[/]")

        for task in failed_tasks:
            jira_key = task.context.get("jira_key", task.id)

            if reset_retries:
                task.retry_count = 0

            queue.requeue_task(task)
            console.print(f"  [green]✓[/] {jira_key} - {task.title[:40]}...")

        console.print(f"\n[green]✓ Queued {len(failed_tasks)} tasks for retry[/]")
        console.print(f"[dim]Monitor with: agent status --watch[/]")
        return

    # Single task retry
    if not identifier:
        console.print("[red]Error: Provide a task identifier or use --all[/]")
        console.print("[dim]Usage: agent retry TASK-ID or agent retry --all[/]")
        return

    task = queue.get_failed_task(identifier)

    if not task:
        console.print(f"[red]Error: No failed task found with identifier '{identifier}'[/]")
        console.print("[dim]Make sure the task exists and has FAILED status.[/]")
        return

    jira_key = task.context.get("jira_key", task.id)
    console.print(f"[bold]Retrying: {jira_key} - {task.title}[/]")

    # Show current error if available
    if task.last_error:
        error_preview = task.last_error[:100] + "..." if len(task.last_error) > 100 else task.last_error
        console.print(f"[dim]Previous error: {error_preview}[/]")

    # Reset retry count if requested
    if reset_retries:
        task.retry_count = 0
        console.print(f"[dim]Retry count reset to 0[/]")

    # Re-queue the task
    queue.requeue_task(task)

    console.print(f"[green]✓ Task queued for {task.assigned_to}[/]")
    console.print(f"[dim]Monitor with: agent status --watch[/]")


@cli.command()
@click.option("--watchdog/--no-watchdog", default=True, help="Start watchdog")
@click.option(
    "--replicas", "-r",
    default=1,
    type=click.IntRange(min=1, max=50),
    help="Number of replicas per agent (1-50, for parallel processing)"
)
@click.option(
    "--log-level", "-l",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level"
)
@click.pass_context
def start(ctx, watchdog, replicas, log_level):
    """Start the agent system."""
    workspace = ctx.obj["workspace"]

    if replicas > 1:
        console.print(f"[bold green]Starting Agent Framework with {replicas} replicas per agent[/]")
    else:
        console.print("[bold green]Starting Agent Framework[/]")

    if log_level != "INFO":
        console.print(f"[dim]Log level: {log_level}[/]")

    try:
        # Create orchestrator
        orchestrator = Orchestrator(workspace)

        # Setup signal handlers for graceful shutdown
        orchestrator.setup_signal_handlers()

        # Spawn all agents with replicas
        processes = orchestrator.spawn_all_agents(replicas=replicas, log_level=log_level)
        console.print(f"[green]✓ Started {len(processes)} agents[/]")

        # Spawn watchdog if requested
        if watchdog:
            orchestrator.spawn_watchdog()
            console.print("[green]✓ Started watchdog[/]")

        console.print("\n[bold]Agents are now running. Press Ctrl+C to stop.[/]")
        console.print("Use 'agent status' to check on agent health and queue status.")

        # Keep main process alive
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping agents...[/]")
            orchestrator.stop_all_agents(graceful=True)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--graceful/--force", default=True, help="Graceful shutdown")
@click.pass_context
def stop(ctx, graceful):
    """Stop the agent system."""
    workspace = ctx.obj["workspace"]

    console.print("[bold yellow]Stopping Agent Framework[/]")

    try:
        orchestrator = Orchestrator(workspace)
        orchestrator.stop_all_agents(graceful=graceful)
        console.print("[green]✓ All agents stopped[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--watchdog/--no-watchdog", default=True, help="Start watchdog")
@click.option("--dashboard/--no-dashboard", default=True, help="Show live dashboard")
@click.pass_context
def restart(ctx, watchdog, dashboard):
    """Restart the agent system (stop then start)."""
    workspace = ctx.obj["workspace"]

    console.print("[bold yellow]Restarting Agent Framework[/]")

    try:
        orchestrator = Orchestrator(workspace)

        # Stop existing agents
        orchestrator.stop_all_agents(graceful=True)
        console.print("[green]✓ Agents stopped[/]")

        import time
        time.sleep(1)

        # Start agents
        orchestrator.setup_signal_handlers()
        processes = orchestrator.spawn_all_agents()
        console.print(f"[green]✓ Started {len(processes)} agents[/]")

        if watchdog:
            orchestrator.spawn_watchdog()
            console.print("[green]✓ Started watchdog[/]")

        if dashboard:
            console.print("\n[bold cyan]Starting live dashboard...[/]")
            console.print("[dim]Press Ctrl+C to exit dashboard (agents continue running)[/]\n")

            from .dashboard import AgentDashboard
            dash = AgentDashboard(workspace)
            try:
                asyncio.run(dash.run())
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard closed[/]")
                console.print("[dim]Agents continue running. Use 'agent status --watch' to monitor.[/]")
        else:
            console.print("\n[bold]Agents are running. Use 'agent status --watch' to monitor.[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command()
@click.option("--agent", "-a", help="Clear only specific agent queue (e.g., 'engineer', 'qa')")
@click.option("--completed", is_flag=True, help="Also clear completed tasks")
@click.option("--locks", is_flag=True, help="Also clear stale lock files")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def clear(ctx, agent, completed, locks, yes):
    """Clear agent task queues."""
    workspace = ctx.obj["workspace"]
    comm_dir = workspace / ".agent-communication"
    queues_dir = comm_dir / "queues"
    completed_dir = comm_dir / "completed"
    locks_dir = comm_dir / "locks"

    if not queues_dir.exists():
        console.print("[yellow]No queues directory found[/]")
        return

    # Count tasks to clear
    pending_count = 0
    completed_count = 0

    if agent:
        # Clear specific agent queue
        agent_dir = queues_dir / agent
        if agent_dir.exists():
            pending_count = len(list(agent_dir.glob("*.json")))
    else:
        # Count all pending tasks
        for agent_dir in queues_dir.iterdir():
            if agent_dir.is_dir():
                pending_count += len(list(agent_dir.glob("*.json")))

    if completed and completed_dir.exists():
        completed_count = len(list(completed_dir.glob("*.json")))

    locks_count = 0
    if locks and locks_dir.exists():
        locks_count = len(list(locks_dir.glob("*.lock")))

    if pending_count == 0 and completed_count == 0 and locks_count == 0:
        console.print("[green]Queues are already empty[/]")
        return

    # Show what will be cleared
    console.print(f"[bold]Tasks to clear:[/]")
    console.print(f"  Pending: {pending_count}")
    if completed:
        console.print(f"  Completed: {completed_count}")
    if locks:
        console.print(f"  Locks: {locks_count}")

    if not yes:
        if not click.confirm("Continue?"):
            console.print("[yellow]Cancelled[/]")
            return

    # Clear tasks
    cleared = 0
    if agent:
        agent_dir = queues_dir / agent
        if agent_dir.exists():
            for f in agent_dir.glob("*.json"):
                f.unlink()
                cleared += 1
    else:
        for agent_dir in queues_dir.iterdir():
            if agent_dir.is_dir():
                for f in agent_dir.glob("*.json"):
                    f.unlink()
                    cleared += 1

    if completed and completed_dir.exists():
        for f in completed_dir.glob("*.json"):
            f.unlink()
            cleared += 1

    if locks and locks_dir.exists():
        import shutil
        for lock_dir in locks_dir.glob("*.lock"):
            if lock_dir.is_dir():
                shutil.rmtree(lock_dir)
                cleared += 1

    console.print(f"[green]✓ Cleared {cleared} items[/]")


@cli.command()
@click.pass_context
def pause(ctx):
    """Pause agent processing after current task completes."""
    workspace = ctx.obj["workspace"]
    comm_dir = workspace / ".agent-communication"
    pause_file = comm_dir / "pause"

    comm_dir.mkdir(parents=True, exist_ok=True)

    if pause_file.exists():
        console.print("[yellow]Agents are already paused[/]")
    else:
        pause_file.write_text(str(int(__import__("time").time())))
        console.print("[yellow]⏸ Pause signal sent[/]")
        console.print("[dim]Agents will pause after completing their current task.[/]")
        console.print("[dim]Use 'agent status --watch' to monitor, 'agent resume' to continue.[/]")


@cli.command()
@click.pass_context
def resume(ctx):
    """Resume paused agent processing."""
    workspace = ctx.obj["workspace"]
    comm_dir = workspace / ".agent-communication"
    pause_file = comm_dir / "pause"

    if not pause_file.exists():
        console.print("[yellow]Agents are not paused[/]")
    else:
        pause_file.unlink()
        console.print("[green]▶ Resume signal sent[/]")
        console.print("[dim]Agents will resume processing on next poll cycle.[/]")


@cli.command()
@click.argument("task_id", required=False)
@click.option("--message", "-m", help="Optional approval message")
@click.pass_context
def approve(ctx, task_id, message):
    """Approve a task waiting at a checkpoint.

    Example:
        agent approve                        # List all checkpoints
        agent approve chain-abc123-engineer  # Approve specific checkpoint
        agent approve chain-abc123-engineer -m "Reviewed and looks good"
    """
    import os
    from datetime import UTC, datetime
    from ..core.task import TaskStatus

    workspace = ctx.obj["workspace"]
    queue = FileQueue(workspace)
    queue_dir = workspace / ".agent-communication" / "queues"
    checkpoint_dir = queue_dir / "checkpoints"

    if not task_id:
        if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.json")):
            console.print("[green]No tasks awaiting approval at checkpoints[/]")
            return

        console.print("[bold]Tasks Awaiting Checkpoint Approval:[/]")
        console.print()

        table = Table()
        table.add_column("Task ID")
        table.add_column("Title")
        table.add_column("Checkpoint")
        table.add_column("Message")

        for checkpoint_file in sorted(checkpoint_dir.glob("*.json")):
            task = FileQueue.load_task_file(checkpoint_file)

            title = task.title[:40] + "..." if len(task.title) > 40 else task.title
            cp_msg = task.checkpoint_message or "N/A"
            if len(cp_msg) > 50:
                cp_msg = cp_msg[:50] + "..."

            table.add_row(
                task.id,
                title,
                task.checkpoint_reached or "N/A",
                cp_msg,
            )

        console.print(table)
        console.print()
        console.print("[dim]Use 'agent approve <task_id>' to approve a specific checkpoint[/]")
        return

    checkpoint_file = checkpoint_dir / f"{task_id}.json"
    if not checkpoint_file.exists():
        console.print(f"[red]Error: No task found at checkpoint with ID '{task_id}'[/]")
        console.print("[dim]Use 'agent approve' to see all checkpoints[/]")
        return

    task = FileQueue.load_task_file(checkpoint_file)

    if task.status != TaskStatus.AWAITING_APPROVAL:
        console.print(f"[yellow]Warning: Task {task_id} is not awaiting approval[/]")
        console.print(f"[dim]Current status: {task.status}[/]")
        return

    console.print(f"[bold]Checkpoint Details:[/]")
    console.print(f"  Task: {task.title}")
    console.print(f"  Checkpoint: {task.checkpoint_reached}")
    console.print(f"  Message: {task.checkpoint_message}")
    console.print()

    if not click.confirm("Approve this checkpoint and continue workflow?"):
        console.print("[yellow]Approval cancelled[/]")
        return

    approver = os.getenv("USER", "user")
    task.approve_checkpoint(approver)

    if message:
        task.notes.append(f"Checkpoint approved: {message}")
    else:
        task.notes.append(f"Checkpoint approved at {datetime.now(UTC).isoformat()}")

    # Re-queue then remove checkpoint file — only delete after successful push
    queue = FileQueue(workspace)
    try:
        queue.push(task, task.assigned_to)
        checkpoint_file.unlink()
    except Exception as e:
        console.print(f"[red]Error re-queuing task: {e}[/]")
        console.print("[dim]Checkpoint file preserved — task not lost[/]")
        return

    console.print(f"[green]Checkpoint approved for task {task_id}[/]")
    console.print(f"[dim]Task re-queued to {task.assigned_to} for continuation[/]")
    console.print(f"[dim]Monitor with: agent status --watch[/]")


@cli.command()
@click.option("--fix", is_flag=True, help="Auto-fix detected issues")
@click.pass_context
def check(ctx, fix):
    """Run circuit breaker safety checks."""
    workspace = ctx.obj["workspace"]

    console.print("[bold]Running Circuit Breaker Checks[/]")

    try:
        # Load config (use config/ subdirectory for consistency)
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")

        # Create circuit breaker
        breaker = CircuitBreaker(
            workspace=framework_config.workspace,
            max_queue_size=framework_config.safeguards.max_queue_size,
            max_escalations=framework_config.safeguards.max_escalations,
            max_task_age_days=framework_config.safeguards.max_task_age_days,
        )

        # Run checks
        report = breaker.run_all_checks()

        # Display report
        console.print(str(report))

        # Auto-fix if requested
        if fix and not report.passed:
            console.print("\n[yellow]Running auto-fix...[/]")
            breaker.fix_issues(report)
            console.print("[green]✓ Auto-fix complete[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("cleanup-worktrees")
@click.option("--max-age", "-a", type=int, help="Max age in hours (overrides config)")
@click.option("--force", is_flag=True, help="Force remove all worktrees")
@click.option("--dry-run", is_flag=True, help="Show what would be removed")
@click.pass_context
def cleanup_worktrees(ctx, max_age, force, dry_run):
    """Clean up orphaned or stale git worktrees.

    By default, removes worktrees older than max_age_hours (from config).
    Use --force to remove all worktrees regardless of age.
    """
    workspace = ctx.obj["workspace"]

    console.print("[bold]Git Worktree Cleanup[/]")
    console.print()

    try:
        # Load config
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")
        worktree_config = framework_config.multi_repo.worktree

        if not worktree_config.enabled:
            console.print("[yellow]Worktree mode is not enabled in config[/]")
            console.print("[dim]Enable with: multi_repo.worktree.enabled: true[/]")
            return

        # Create worktree manager
        github_token = os.environ.get("GITHUB_TOKEN")
        wt_config = worktree_config.to_manager_config()
        # Override max_age if specified via CLI
        if max_age is not None:
            wt_config.max_age_hours = max_age
        manager = WorktreeManager(config=wt_config, github_token=github_token)

        # Get current stats
        stats = manager.get_stats()
        worktrees = manager.list_worktrees()

        console.print(f"Worktree root: [cyan]{wt_config.root}[/]")
        console.print(f"Registered: {stats['total_registered']}, Active: {stats['active']}, Orphaned: {stats['orphaned']}")
        console.print(f"Max age: {wt_config.max_age_hours}h, Max worktrees: {wt_config.max_worktrees}")
        console.print()

        if not worktrees:
            console.print("[green]No worktrees to clean up[/]")
            return

        # Show worktrees
        from rich.table import Table
        from datetime import datetime

        table = Table()
        table.add_column("Agent")
        table.add_column("Task")
        table.add_column("Branch")
        table.add_column("Age")
        table.add_column("Status")

        now = datetime.utcnow()
        to_remove = []

        for wt in worktrees:
            try:
                last_accessed = datetime.fromisoformat(wt.last_accessed)
                age_hours = (now - last_accessed).total_seconds() / 3600
                age_str = f"{age_hours:.1f}h"

                is_stale = age_hours > wt_config.max_age_hours
                exists = Path(wt.path).exists()

                if force or is_stale or not exists:
                    to_remove.append(wt)
                    status = "[red]REMOVE[/]"
                else:
                    status = "[green]KEEP[/]"

                table.add_row(
                    wt.agent_id,
                    wt.task_id[:8],
                    wt.branch[:30] + "..." if len(wt.branch) > 30 else wt.branch,
                    age_str,
                    status,
                )
            except ValueError:
                to_remove.append(wt)
                table.add_row(wt.agent_id, wt.task_id[:8], wt.branch[:30], "?", "[red]REMOVE[/]")

        console.print(table)
        console.print()

        if not to_remove:
            console.print("[green]No worktrees need removal[/]")
            return

        console.print(f"[yellow]{len(to_remove)} worktrees will be removed[/]")

        if dry_run:
            console.print("[dim]Dry run - no changes made[/]")
            return

        if not click.confirm("Continue?"):
            console.print("[yellow]Cancelled[/]")
            return

        # Perform cleanup
        removed = 0
        for wt in to_remove:
            path = Path(wt.path)
            if manager.remove_worktree(path, force=True):
                removed += 1
                console.print(f"  Removed: {path.name}")

        console.print(f"\n[green]✓ Removed {removed} worktrees[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback
        traceback.print_exc()


@cli.command("apply-pattern")
@click.option("--reference", "-r", required=True, help="Reference repo: owner/repo")
@click.option("--files", "-f", required=True, help="Reference files (comma-separated)")
@click.option("--targets", "-t", required=True, help="Target repos (comma-separated)")
@click.option("--description", "-d", required=True, help="What to implement")
@click.option("--branch-prefix", default="feature/agent", help="Branch name prefix")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.pass_context
def apply_pattern(ctx, reference, files, targets, description, branch_prefix, dry_run):
    """Apply a pattern from reference repo to multiple target repos."""
    workspace = ctx.obj["workspace"]

    try:
        # Load config (use config/ subdirectory for consistency)
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")

        # Check for GitHub token
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            console.print("[red]Error: GITHUB_TOKEN environment variable not set[/]")
            return

        # Parse inputs
        ref_repo = reference
        ref_files = [f.strip() for f in files.split(",")]
        target_repos = [t.strip() for t in targets.split(",")]

        console.print(f"[bold]Applying pattern from {ref_repo} to {len(target_repos)} repos[/]")
        console.print(f"Reference files: {', '.join(ref_files)}")

        # Initialize managers
        manager = MultiRepoManager(
            framework_config.multi_repo.workspace_root,
            github_token
        )

        # For multi-repo PR creation, we need to initialize GitHubClient differently
        # We'll use PyGithub directly
        from github import Github
        gh = Github(github_token)

        # Ensure and read reference repo
        console.print(f"\n[cyan]Reading reference: {ref_repo}[/]")
        manager.ensure_repo(ref_repo)
        reference_content = manager.read_files(ref_repo, ref_files)

        if not reference_content:
            console.print("[red]Error: Could not read reference files[/]")
            return

        console.print(f"[green]✓ Read {len(reference_content)} reference files[/]")

        # Process each target repo
        for target in target_repos:
            console.print(f"\n[bold cyan]Processing: {target}[/]")

            if dry_run:
                console.print(f"  [yellow]Would clone, create branch, run Claude, and create PR[/]")
                continue

            # Track state for rollback
            original_branch = None
            branch_created = False
            changes_pushed = False
            target_path = None
            branch = None

            try:
                # Ensure target repo is cloned/updated
                console.print(f"  Ensuring repo is up to date...")
                target_path = manager.ensure_repo(target)
                console.print(f"  [green]✓ Repo ready at {target_path}[/]")

                # Save current branch for rollback
                original_branch = manager.get_current_branch(target)

                # Create branch
                # Slugify description for branch name
                slug = re.sub(r'[^a-z0-9]+', '-', description.lower()).strip('-')[:50]
                branch = f"{branch_prefix}/{slug}"

                console.print(f"  Creating branch: {branch}")
                manager.create_branch(target, branch)
                branch_created = True
                console.print(f"  [green]✓ Branch created[/]")

                # Build prompt with reference context
                prompt = _build_apply_pattern_prompt(description, reference_content, ref_repo, target)

                # Run Claude CLI in target repo directory
                console.print(f"  Running Claude to implement pattern...")
                _run_claude_cli(prompt, target_path, framework_config)
                console.print(f"  [green]✓ Claude completed[/]")

                # Commit and push
                console.print(f"  Committing and pushing...")
                commit_msg = f"Implement: {description}\n\nApplied pattern from {ref_repo}"
                manager.commit_and_push(target, branch, commit_msg)
                changes_pushed = True
                console.print(f"  [green]✓ Changes pushed[/]")

                # Create PR
                console.print(f"  Creating pull request...")
                pr_body = f"""## Description
{description}

## Reference Implementation
Applied pattern from [`{ref_repo}`](https://github.com/{ref_repo})

Reference files:
{chr(10).join(f'- `{f}`' for f in ref_files)}

## Changes
This PR implements the same pattern/functionality as the reference implementation.
"""
                # Create PR using PyGithub directly
                repo = gh.get_repo(target)
                pr = repo.create_pull(
                    title=description,
                    body=pr_body,
                    head=branch,
                    base="main"
                )
                pr.add_to_labels("agent-pr", "pattern-application")
                console.print(f"  [green]✓ PR created: {pr.html_url}[/]")

            except subprocess.TimeoutExpired:
                console.print(f"  [red]✗ Operation timed out for {target}[/]")
                _rollback_changes(manager, target, target_path, original_branch, branch, branch_created, changes_pushed)
                continue
            except ValueError as e:
                console.print(f"  [red]✗ Validation error for {target}: {e}[/]")
                console.print(f"  [yellow]Check your inputs - repository name, branch prefix, or file paths may be invalid[/]")
                _rollback_changes(manager, target, target_path, original_branch, branch, branch_created, changes_pushed)
                continue
            except PermissionError as e:
                console.print(f"  [red]✗ Permission denied for {target}: {e}[/]")
                console.print(f"  [yellow]Verify your GitHub token has write access to this repository[/]")
                _rollback_changes(manager, target, target_path, original_branch, branch, branch_created, changes_pushed)
                continue
            except RuntimeError as e:
                console.print(f"  [red]✗ Claude CLI failed for {target}: {e}[/]")
                console.print(f"  [yellow]Check Claude CLI installation and configuration[/]")
                _rollback_changes(manager, target, target_path, original_branch, branch, branch_created, changes_pushed)
                continue
            except Exception as e:
                console.print(f"  [red]✗ Unexpected error processing {target}: {e}[/]")
                console.print(f"  [yellow]Rolling back changes...[/]")
                _rollback_changes(manager, target, target_path, original_branch, branch, branch_created, changes_pushed)
                continue

        console.print(f"\n[bold green]✓ Pattern application complete![/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback
        traceback.print_exc()


def _handle_epic_mode(ctx, workspace, framework_config, epic_key: str, no_dashboard: bool, parallel: bool = False, auto_approve: bool = False):
    """Handle --epic mode: process tickets in a JIRA epic.

    Args:
        ctx: Click context
        workspace: Workspace path
        framework_config: Framework configuration
        epic_key: JIRA epic key (e.g., PROJ-100)
        no_dashboard: Skip dashboard if True
        parallel: If True, process tickets in parallel (no dependencies)
        auto_approve: If True, skip plan checkpoint (use default_auto workflow)
    """
    from datetime import datetime
    import time

    # Validate epic key format
    if "-" not in epic_key:
        console.print(f"[red]Invalid epic key format: {epic_key}[/]")
        console.print("[dim]Expected format: PROJ-123[/]")
        return

    # Load JIRA config
    jira_config = load_jira_config(workspace / "config" / "jira.yaml")
    if not jira_config:
        console.print("[red]Error: JIRA config not found. Create config/jira.yaml[/]")
        return

    # Create JIRA client
    jira_client = JIRAClient(jira_config)

    console.print(f"[bold]Fetching epic: {epic_key}[/]")

    try:
        epic_data = jira_client.get_epic_with_subtasks(epic_key)
        epic = epic_data["epic"]
        issues = epic_data["issues"]

        if not issues:
            console.print(f"[yellow]No issues found in epic {epic_key}[/]")
            console.print("[dim]Make sure the epic has linked issues or subtasks.[/]")
            return

        console.print(f"[green]✓ Found {len(issues)} issues in epic:[/]")
        console.print(f"  [bold]{epic.fields.summary}[/]")
        console.print()

        # Show issues
        table = Table()
        table.add_column("#", style="dim")
        table.add_column("Key")
        table.add_column("Summary")
        table.add_column("Type")
        table.add_column("Status")

        for i, issue in enumerate(issues, 1):
            table.add_row(
                str(i),
                issue.key,
                issue.fields.summary[:50] + "..." if len(issue.fields.summary) > 50 else issue.fields.summary,
                issue.fields.issuetype.name,
                issue.fields.status.name,
            )

        console.print(table)
        console.print()

        if not click.confirm(f"Queue {len(issues)} tickets for processing?"):
            console.print("[yellow]Cancelled[/]")
            return

        workflow = "default_auto" if auto_approve else "default"

        # Determine target repository from epic or config
        github_repo = None
        jira_project = epic_key.split("-")[0]

        # Try to find matching repo in config
        for repo in framework_config.repositories:
            if repo.jira_project == jira_project:
                github_repo = repo.github_repo
                break

        if not github_repo:
            console.print(f"[yellow]No repository configured for JIRA project {jira_project}[/]")
            if framework_config.repositories:
                console.print("Available repositories:")
                for i, repo in enumerate(framework_config.repositories, 1):
                    console.print(f"  {i}. {repo.github_repo} ({repo.jira_project})")

                repo_idx = click.prompt(
                    "Select repository",
                    type=click.IntRange(1, len(framework_config.repositories))
                )
                selected_repo = framework_config.repositories[repo_idx - 1]
                github_repo = selected_repo.github_repo
            else:
                console.print("[red]No repositories configured. Add them to config/agent-framework.yaml[/]")
                return

        # Queue all issues as tasks
        queue = FileQueue(workspace)

        previous_task_id = None
        queued_tasks = []

        # Filter out completed tickets
        pending_issues = [
            issue for issue in issues
            if issue.fields.status.name.lower() not in ("done", "closed", "resolved", "in progress", "code review", "won't do")
        ]

        if len(pending_issues) < len(issues):
            console.print(f"[dim]Skipping {len(issues) - len(pending_issues)} completed tickets[/]")

        if not pending_issues:
            console.print("[green]All tickets in this epic are already completed![/]")
            return

        for i, issue in enumerate(pending_issues):
            assigned_to = "architect"

            # Override based on issue type
            issue_type = issue.fields.issuetype.name.lower()
            if "test" in issue_type or "qa" in issue_type:
                assigned_to = "qa"
            elif "bug" in issue_type:
                assigned_to = "engineer"

            # Create task
            task = jira_client.issue_to_task(issue, assigned_to)

            # Add epic context
            task.context["epic_key"] = epic_key
            task.context["epic_summary"] = epic.fields.summary
            task.context["github_repo"] = github_repo
            task.context["jira_project"] = jira_project
            task.context["workflow"] = workflow
            task.context["epic_position"] = i + 1
            task.context["epic_total"] = len(pending_issues)

            # Parallel mode: use worktrees, no dependencies
            if parallel:
                task.context["use_worktree"] = True
            else:
                # Sequential: each task depends on previous
                if previous_task_id:
                    task.depends_on = [previous_task_id]

            queue.push(task, assigned_to)
            queued_tasks.append((task.id, issue.key, assigned_to))
            previous_task_id = task.id

        console.print(f"\n[green]✓ Queued {len(queued_tasks)} tasks from epic {epic_key}[/]")
        if parallel:
            console.print(f"[dim]Tasks will process in PARALLEL using worktrees[/]")
        else:
            console.print(f"[dim]Tasks will process sequentially: ticket 1 → ticket 2 → ...[/]")

        # Ensure agents are running
        orchestrator = Orchestrator(workspace)
        running = orchestrator.get_running_agents()

        if running:
            console.print(f"\n[green]✓ Agents already running: {', '.join(running)}[/]")
        else:
            console.print("\n[bold]Starting agents...[/]")
            orchestrator.setup_signal_handlers()
            orchestrator.spawn_all_agents()
            console.print("[green]✓ Agents started[/]")

        if no_dashboard:
            console.print(f"\n[bold cyan]🎯 Processing epic: {epic_key}[/]")
            console.print(f"[dim]📋 Monitor progress: agent status --watch[/]")
            console.print(f"[dim]⏸ Pause anytime: agent pause[/]")
            console.print(f"[dim]▶ Resume: agent resume[/]")
        else:
            console.print("\n[bold cyan]✓ Tasks queued, starting live dashboard...[/]")
            console.print("[dim]Press Ctrl+C to exit dashboard[/]\n")

            from .dashboard import AgentDashboard
            dashboard = AgentDashboard(workspace)
            try:
                asyncio.run(dashboard.run())
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard closed[/]")
                console.print("[dim]Agents continue running. Use 'agent status --watch' to monitor.[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback
        traceback.print_exc()


def _rollback_changes(
    manager,
    repo_name: str,
    repo_path: Path,
    original_branch: str,
    new_branch: str,
    branch_created: bool,
    changes_pushed: bool
):
    """Rollback changes after a failed operation."""
    if not repo_path or not repo_path.exists():
        return

    try:
        console.print(f"  [yellow]Rolling back {repo_name}...[/]")

        # Return to original branch if we switched
        if original_branch:
            try:
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                console.print(f"  [yellow]Returned to branch: {original_branch}[/]")
            except subprocess.CalledProcessError:
                console.print(f"  [yellow]Could not return to original branch[/]")

        # Delete local branch if we created it
        if branch_created and new_branch:
            try:
                manager.delete_branch(repo_name, new_branch)
                console.print(f"  [yellow]Deleted local branch: {new_branch}[/]")
            except Exception:
                pass  # Branch may not exist or already deleted

        # Provide guidance for remote cleanup if changes were pushed
        if changes_pushed and new_branch:
            console.print(f"  [yellow]⚠ Changes were pushed to remote before failure[/]")
            console.print(f"  [yellow]To delete remote branch, run:[/]")
            console.print(f"  [dim]cd {repo_path} && git push origin --delete {new_branch}[/]")

        console.print(f"  [green]✓ Rollback complete for {repo_name}[/]")

    except Exception as e:
        console.print(f"  [red]✗ Rollback failed for {repo_name}: {e}[/]")
        console.print(f"  [yellow]Manual cleanup may be required in: {repo_path}[/]")


def _build_apply_pattern_prompt(description: str, reference_content: dict, ref_repo: str, target_repo: str) -> str:
    """Build prompt for Claude to implement the pattern."""
    ref_files_section = "\n\n".join(
        f"**File: {path}**\n```\n{content}\n```"
        for path, content in reference_content.items()
    )

    return f"""You are tasked with implementing a pattern in the repository: {target_repo}

TASK:
{description}

REFERENCE IMPLEMENTATION:
The reference implementation is from repository: {ref_repo}

{ref_files_section}

INSTRUCTIONS:
1. Analyze the reference implementation above
2. Understand the pattern, architecture, and approach used
3. Implement the same pattern in the current repository ({target_repo})
4. Follow the same code structure and conventions as the reference
5. Adapt the implementation to fit the current repository's structure and conventions
6. Create any necessary files, update existing files as needed
7. Ensure the implementation is complete and functional

IMPORTANT:
- Study the reference files carefully before implementing
- Match the code quality and style of the reference implementation
- Make sure all necessary imports, dependencies, and configurations are included
- Test your implementation if possible

Begin implementing now."""


def _run_claude_cli(prompt: str, cwd: Path, framework_config, timeout: int = 3600):
    """Run Claude CLI with the given prompt in the specified directory.

    Args:
        prompt: The prompt to send to Claude
        cwd: Working directory for Claude CLI
        framework_config: Framework configuration
        timeout: Maximum execution time in seconds (default: 1 hour)

    Raises:
        ValueError: If prompt is invalid
        RuntimeError: If Claude CLI execution fails
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Remove control characters and limit length
    prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t\r')

    # Reasonable prompt size limit (1MB)
    MAX_PROMPT_SIZE = 1024 * 1024
    if len(prompt.encode('utf-8')) > MAX_PROMPT_SIZE:
        raise ValueError(f"Prompt too large (max {MAX_PROMPT_SIZE} bytes)")

    # Validate working directory
    if not cwd or not cwd.exists():
        raise ValueError(f"Invalid working directory: {cwd}")

    # Get Claude CLI executable from config
    claude_cmd = framework_config.llm.claude_cli_executable

    # Validate executable exists
    if not subprocess.run(
        ["which", claude_cmd],
        capture_output=True,
        timeout=5,
    ).returncode == 0:
        raise RuntimeError(f"Claude CLI executable not found: {claude_cmd}")

    # Build env with proxy vars if configured
    env = os.environ.copy()
    env.update(framework_config.llm.get_proxy_env())

    # Run Claude CLI with timeout
    # Send prompt via stdin and use dangerously-skip-permissions for automation
    try:
        result = subprocess.run(
            [claude_cmd, "--dangerously-skip-permissions"],
            cwd=cwd,
            input=prompt,
            capture_output=False,  # Let Claude output directly to console
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI execution exceeded timeout of {timeout} seconds")


if __name__ == "__main__":
    cli()
