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
    framework_config=None,
):
    """Build and run a Claude Agent Teams subprocess.

    Handles env setup, MCP config, template flags, and marks the
    session ended when the process exits.
    """
    from ..core.team_bridge import TeamBridge

    claude_cmd = ["claude"]

    if template.delegate_mode:
        claude_cmd.extend(["--permission-mode", "delegate"])
    elif template.plan_approval:
        claude_cmd.extend(["--permission-mode", "plan"])

    env = os.environ.copy()
    env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"

    # Inject LLM proxy env if configured
    if framework_config is None:
        from ..core.config import load_config
        try:
            framework_config = load_config(workspace / "config" / "agent-framework.yaml")
        except Exception:
            pass
    if framework_config:
        env.update(framework_config.llm.get_proxy_env())

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


def _create_team_worktree(
    repo_path: Path,
    owner_repo: str,
    team_name: str,
    framework_config,
) -> tuple[Path, str]:
    """Create an isolated worktree for a team session.

    Uses the base clone from ensure_repo() as the worktree parent so the
    team works on its own branch without touching the default branch.

    Returns:
        Tuple of (worktree_path, branch_name), or (repo_path, "") if
        worktree creation fails (falls back to bare clone).
    """
    from ..workspace.worktree_manager import WorktreeManager

    branch_name = f"team/{team_name}"

    try:
        wt_config = framework_config.multi_repo.worktree.to_manager_config()
        github_token = os.environ.get("GITHUB_TOKEN")
        wt_manager = WorktreeManager(wt_config, github_token=github_token)

        worktree_path = wt_manager.create_worktree(
            base_repo=repo_path,
            branch_name=branch_name,
            agent_id="team",
            task_id=team_name,
            owner_repo=owner_repo,
        )
        return worktree_path, branch_name
    except Exception as e:
        console.print(f"[yellow]Worktree creation failed, using base clone: {e}[/]")
        return repo_path, ""


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
@click.option("--epic", "-e", help="JIRA epic key - fetches tickets and includes in team context")
@click.pass_context
def start(ctx, template, repo, epic):
    """Launch an interactive Agent Team session.

    Spawns a Claude Agent Teams session with pre-configured team structure.
    The team can interact with the autonomous pipeline via MCP tools.

    Examples:
        agent team start --template full --repo myorg/myapp
        agent team start --template full --repo myorg/myapp --epic ME-443
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

    # Load framework config (used for repo resolution and proxy env)
    from ..core.config import load_config
    framework_config = None
    try:
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")
    except Exception:
        pass

    # Generate team name upfront so worktree branch includes it
    team_name = f"{tmpl.team_name_prefix}-{int(time.time())}"

    # Resolve repo path if specified, then create isolated worktree
    repo_info = None
    repo_path = None
    if repo:
        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token and framework_config:
                from ..workspace.multi_repo_manager import MultiRepoManager
                manager = MultiRepoManager(
                    framework_config.multi_repo.workspace_root,
                    github_token,
                )
                base_clone = manager.ensure_repo(repo)

                worktree_path, branch = _create_team_worktree(
                    base_clone, repo, team_name, framework_config,
                )
                repo_path = worktree_path
                if branch:
                    repo_info = f"Repository: {repo}\nBranch: {branch}\nWorktree: {repo_path}"
                    console.print(f"[green]Worktree ready: {repo_path} (branch: {branch})[/]")
                else:
                    repo_info = f"Repository: {repo}\nLocal path: {repo_path}"
                    console.print(f"[green]Repository ready: {repo_path}[/]")
            elif not github_token:
                console.print("[yellow]GITHUB_TOKEN not set, skipping repo setup[/]")
                repo_info = f"Repository: {repo} (not cloned - no GITHUB_TOKEN)"
            else:
                console.print("[yellow]Config not loaded, skipping repo setup[/]")
                repo_info = f"Repository: {repo} (config unavailable)"
        except Exception as e:
            console.print(f"[yellow]Could not resolve repo: {e}[/]")
            repo_info = f"Repository: {repo} (resolution failed)"

    # Fetch epic tickets if --epic provided
    epic_context = None
    if epic:
        try:
            from ..core.config import load_jira_config
            from ..integrations.jira.client import JIRAClient

            jira_config = load_jira_config(workspace / "config" / "jira.yaml")
            if not jira_config:
                console.print("[yellow]JIRA not configured - proceeding without epic context[/]")
                epic_context = f"Epic: {epic} (JIRA not configured)"
            else:
                jira_client = JIRAClient(jira_config)
                epic_data = jira_client.get_epic_with_subtasks(epic)

                epic_issue = epic_data["epic"]
                lines = [f"### Epic: {epic_issue.key} - {epic_issue.fields.summary}\n"]
                if epic_issue.fields.description:
                    lines.append(f"{epic_issue.fields.description}\n")

                issues = epic_data["issues"]
                lines.append(f"### Tickets ({len(issues)})\n")
                for issue in issues:
                    status = issue.fields.status.name
                    issue_type = issue.fields.issuetype.name
                    lines.append(f"- **{issue.key}** [{issue_type}] ({status}): {issue.fields.summary}")
                    if issue.fields.description:
                        desc = issue.fields.description[:500]
                        lines.append(f"  {desc}")

                epic_context = "\n".join(lines)
                console.print(f"[green]Loaded epic {epic} with {len(issues)} tickets[/]")
        except Exception as e:
            console.print(f"[yellow]Could not fetch epic {epic}: {e}[/]")
            epic_context = f"Epic: {epic} (fetch failed - work with available context)"

    # Load team context doc
    team_context_doc = None
    context_path = workspace / "config" / "docs" / "team_context.md"
    if context_path.exists():
        team_context_doc = context_path.read_text()

    # Build spawn prompt
    prompt = build_spawn_prompt(
        template=tmpl,
        task_context=epic_context,
        repo_info=repo_info,
        team_context_doc=team_context_doc,
    )

    # Record session
    session_metadata = {}
    if repo:
        session_metadata["repo"] = repo
    if epic:
        session_metadata["epic"] = epic
    bridge.record_team_session(
        team_name=team_name,
        template=template,
        metadata=session_metadata or None,
    )

    console.print(f"\n[bold cyan]Launching Agent Team: {team_name}[/]")
    console.print(f"[dim]Template: {template} | Lead: {tmpl.lead_role}[/]")
    if tmpl.plan_approval:
        console.print("[dim]Plan approval: enabled (teammates must approve plans)[/]")
    if tmpl.delegate_mode:
        console.print("[dim]Delegate mode: enabled[/]")
    console.print()

    cwd = repo_path or workspace

    _launch_team_session(tmpl, cwd, workspace, prompt, team_name, framework_config)


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

    # Load framework config (used for repo resolution and proxy env)
    from ..core.config import load_config
    framework_config = None
    try:
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")
    except Exception:
        pass

    # Resolve repo if available, then create isolated worktree
    repo_info = None
    repo_path = None
    github_repo = task.context.get("github_repo")
    team_name = f"{tmpl.team_name_prefix}-escalation-{int(time.time())}"
    if github_repo:
        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token and framework_config:
                from ..workspace.multi_repo_manager import MultiRepoManager
                manager = MultiRepoManager(
                    framework_config.multi_repo.workspace_root,
                    github_token,
                )
                base_clone = manager.ensure_repo(github_repo)

                worktree_path, branch = _create_team_worktree(
                    base_clone, github_repo, team_name, framework_config,
                )
                repo_path = worktree_path
                if branch:
                    repo_info = f"Repository: {github_repo}\nBranch: {branch}\nWorktree: {repo_path}"
                else:
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

    # Record session
    escalation_metadata = {}
    if github_repo:
        escalation_metadata["repo"] = github_repo
    bridge.record_team_session(
        team_name=team_name,
        template=template,
        source_task_id=task.id,
        metadata=escalation_metadata or None,
    )

    console.print(f"\n[bold cyan]Launching debug team: {team_name}[/]")
    console.print()

    cwd = repo_path or workspace

    _launch_team_session(tmpl, cwd, workspace, prompt, team_name, framework_config)


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
