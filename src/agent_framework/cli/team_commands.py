"""CLI commands for Claude Agent Teams integration."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _launch_team_session(
    template,
    cwd: Path,
    workspace: Path,
    prompt: str,
    team_name: str,
):
    """Build and run a Claude Agent Teams subprocess.

    Handles env setup, MCP config, template flags, and marks the
    session ended when the process exits.
    """
    from ..core.team_bridge import TeamBridge

    claude_cmd = ["claude"]

    if template.plan_approval:
        claude_cmd.append("--plan-approval")

    if template.delegate_mode:
        claude_cmd.append("--delegate-mode")

    env = os.environ.copy()
    env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"

    mcp_config = workspace / "config" / "mcp-config.json"
    if mcp_config.exists():
        env["CLAUDE_MCP_CONFIG"] = str(mcp_config)

    try:
        subprocess.run(
            claude_cmd,
            cwd=cwd,
            env=env,
            input=prompt,
            text=True,
        )
    except FileNotFoundError:
        console.print("[red]Claude CLI not found. Install it first.[/]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Team session ended[/]")
    finally:
        bridge = TeamBridge(workspace)
        bridge.mark_session_ended(team_name)


@click.group()
def team():
    """Interactive Agent Teams - collaborative multi-agent sessions."""
    pass


@team.command()
@click.option(
    "--template", "-t",
    type=click.Choice(["full", "review", "debug"]),
    default="full",
    help="Team template: full (Architect+Engineer+QA), review (3 reviewers), debug (investigators)",
)
@click.option("--repo", "-r", help="Target repository (owner/repo)")
@click.pass_context
def start(ctx, template, repo):
    """Launch an interactive Agent Team session.

    Spawns a Claude Agent Teams session with pre-configured team structure.
    The team can interact with the autonomous pipeline via MCP tools.

    Examples:
        agent team start --template full --repo myorg/myapp
        agent team start --template review --repo myorg/myapp
        agent team start --template debug
    """
    workspace = ctx.obj["workspace"]

    from ..core.team_templates import load_team_templates, build_spawn_prompt
    from ..core.team_bridge import TeamBridge

    # Load template
    templates = load_team_templates(workspace / "config")
    if template not in templates:
        console.print(f"[red]Template '{template}' not found[/]")
        console.print(f"[dim]Available: {', '.join(templates.keys())}[/]")
        return

    tmpl = templates[template]
    bridge = TeamBridge(workspace)

    # Resolve repo path if specified
    repo_info = None
    repo_path = None
    if repo:
        try:
            from ..core.config import load_config
            framework_config = load_config(workspace / "config" / "agent-framework.yaml")

            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                from ..workspace.multi_repo_manager import MultiRepoManager
                manager = MultiRepoManager(
                    framework_config.multi_repo.workspace_root,
                    github_token,
                )
                repo_path = manager.ensure_repo(repo)
                repo_info = f"Repository: {repo}\nLocal path: {repo_path}"
                console.print(f"[green]Repository ready: {repo_path}[/]")
            else:
                console.print("[yellow]GITHUB_TOKEN not set, skipping repo setup[/]")
                repo_info = f"Repository: {repo} (not cloned - no GITHUB_TOKEN)"
        except Exception as e:
            console.print(f"[yellow]Could not resolve repo: {e}[/]")
            repo_info = f"Repository: {repo} (resolution failed)"

    # Load team context doc
    team_context_doc = None
    context_path = workspace / "config" / "docs" / "team_context.md"
    if context_path.exists():
        team_context_doc = context_path.read_text()

    # Build spawn prompt
    prompt = build_spawn_prompt(
        template=tmpl,
        repo_info=repo_info,
        team_context_doc=team_context_doc,
    )

    # Generate team name
    team_name = f"{tmpl.team_name_prefix}-{int(time.time())}"

    # Record session
    bridge.record_team_session(
        team_name=team_name,
        template=template,
        metadata={"repo": repo} if repo else None,
    )

    console.print(f"\n[bold cyan]Launching Agent Team: {team_name}[/]")
    console.print(f"[dim]Template: {template} | Lead: {tmpl.lead_role}[/]")
    if tmpl.plan_approval:
        console.print("[dim]Plan approval: enabled (teammates must approve plans)[/]")
    if tmpl.delegate_mode:
        console.print("[dim]Delegate mode: enabled[/]")
    console.print()

    cwd = repo_path or workspace

    _launch_team_session(tmpl, cwd, workspace, prompt, team_name)


@team.command()
@click.argument("task_id")
@click.option(
    "--template", "-t",
    type=click.Choice(["full", "review", "debug"]),
    default="debug",
    help="Team template to use for resolution",
)
@click.pass_context
def escalate(ctx, task_id, template):
    """Escalate a failed task to an interactive team for resolution.

    Takes a failed autonomous task and spawns a team session with full
    error context so the team can diagnose and fix the issue.

    Examples:
        agent team escalate task-impl-1234
        agent team escalate task-impl-1234 --template full
    """
    workspace = ctx.obj["workspace"]

    from ..queue.file_queue import FileQueue
    from ..core.team_templates import load_team_templates, build_spawn_prompt
    from ..core.team_bridge import TeamBridge

    # Load the failed task
    queue = FileQueue(workspace)
    task = queue.get_failed_task(task_id)

    if not task:
        console.print(f"[red]No failed task found: {task_id}[/]")
        console.print("[dim]Ensure the task exists and has FAILED status.[/]")
        return

    console.print(f"[bold]Escalating: {task.title}[/]")
    console.print(f"[dim]Task: {task.id} | Retries: {task.retry_count}[/]")

    # Build escalation context
    bridge = TeamBridge(workspace)
    escalation_context = bridge.build_escalation_context(task)

    # Load template
    templates = load_team_templates(workspace / "config")
    if template not in templates:
        console.print(f"[red]Template '{template}' not found[/]")
        return

    tmpl = templates[template]

    # Resolve repo if available
    repo_info = None
    repo_path = None
    github_repo = task.context.get("github_repo")
    if github_repo:
        try:
            from ..core.config import load_config
            framework_config = load_config(workspace / "config" / "agent-framework.yaml")
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token:
                from ..workspace.multi_repo_manager import MultiRepoManager
                manager = MultiRepoManager(
                    framework_config.multi_repo.workspace_root,
                    github_token,
                )
                repo_path = manager.ensure_repo(github_repo)
                repo_info = f"Repository: {github_repo}\nLocal path: {repo_path}"
        except Exception as e:
            console.print(f"[yellow]Could not resolve repo: {e}[/]")

    # Load team context doc
    team_context_doc = None
    context_path = workspace / "config" / "docs" / "team_context.md"
    if context_path.exists():
        team_context_doc = context_path.read_text()

    # Build spawn prompt with escalation context
    prompt = build_spawn_prompt(
        template=tmpl,
        task_context=escalation_context,
        repo_info=repo_info,
        team_context_doc=team_context_doc,
    )

    # Generate team name
    team_name = f"{tmpl.team_name_prefix}-escalation-{int(time.time())}"

    # Record session
    bridge.record_team_session(
        team_name=team_name,
        template=template,
        source_task_id=task.id,
    )

    console.print(f"\n[bold cyan]Launching debug team: {team_name}[/]")
    console.print()

    cwd = repo_path or workspace

    _launch_team_session(tmpl, cwd, workspace, prompt, team_name)


@team.command("status")
@click.pass_context
def team_status(ctx):
    """List active and recent team sessions.

    Shows sessions from both local workspace and ~/.claude/teams/.

    Examples:
        agent team status
    """
    workspace = ctx.obj["workspace"]

    from ..core.team_bridge import TeamBridge

    bridge = TeamBridge(workspace)
    sessions = bridge.get_active_teams()

    if not sessions:
        console.print("[dim]No team sessions found[/]")
        console.print("[dim]Start one with: agent team start --template full[/]")
        return

    table = Table(title="Agent Team Sessions")
    table.add_column("Team", style="cyan")
    table.add_column("Template")
    table.add_column("Started")
    table.add_column("Source Task")
    table.add_column("Status")

    for session in sessions:
        started = session.get("started_at", "?")
        if isinstance(started, str) and len(started) > 19:
            started = started[:19]  # Trim timezone for display

        table.add_row(
            session.get("team_name", "?"),
            session.get("template", "?"),
            started,
            session.get("source_task_id", "-"),
            session.get("status", "?"),
        )

    console.print(table)


@team.command()
@click.argument("team_name")
@click.option(
    "--workflow", "-w",
    type=click.Choice(["simple", "standard", "full"]),
    default="simple",
    help="Workflow for queued tasks",
)
@click.pass_context
def handoff(ctx, team_name, workflow):
    """Queue a team's completed work to the autonomous pipeline.

    Reads task output from the team session and pushes it to the
    autonomous pipeline for further processing.

    Examples:
        agent team handoff impl-1234567890
        agent team handoff impl-1234567890 --workflow standard
    """
    workspace = ctx.obj["workspace"]

    if not re.match(r'^[a-zA-Z0-9_-]+$', team_name):
        console.print(f"[red]Invalid team name: {team_name}[/]")
        console.print("[dim]Team names may only contain letters, digits, hyphens, and underscores.[/]")
        return

    from ..core.team_bridge import TeamBridge

    bridge = TeamBridge(workspace)

    # Look for team tasks in Claude's tasks directory
    claude_tasks_dir = Path.home() / ".claude" / "tasks" / team_name
    local_tasks_dir = workspace / ".agent-communication" / "teams" / team_name

    tasks_to_handoff = []

    for tasks_dir in [claude_tasks_dir, local_tasks_dir]:
        if not tasks_dir.exists():
            continue

        for task_file in tasks_dir.glob("*.json"):
            try:
                data = json.loads(task_file.read_text())
                tasks_to_handoff.append(data)
            except (json.JSONDecodeError, OSError) as e:
                console.print(f"[yellow]Skipping {task_file.name}: {e}[/]")

    if not tasks_to_handoff:
        console.print(f"[yellow]No tasks found for team: {team_name}[/]")
        console.print(f"[dim]Checked: {claude_tasks_dir}[/]")
        console.print(f"[dim]Checked: {local_tasks_dir}[/]")
        return

    console.print(f"[bold]Found {len(tasks_to_handoff)} tasks from team {team_name}[/]")

    for task in tasks_to_handoff:
        title = task.get("title", "Untitled")[:50]
        console.print(f"  - {title}")

    if not click.confirm("\nQueue these tasks to the autonomous pipeline?"):
        console.print("[yellow]Cancelled[/]")
        return

    queued_ids = bridge.handoff_to_autonomous(tasks_to_handoff, workflow=workflow)

    console.print(f"\n[green]Queued {len(queued_ids)} tasks[/]")
    for task_id in queued_ids:
        console.print(f"  [dim]{task_id}[/]")

    console.print(f"\n[dim]Monitor with: agent status --watch[/]")
