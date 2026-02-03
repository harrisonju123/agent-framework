"""Main CLI for agent framework."""

import asyncio
import os
import re
import subprocess
from pathlib import Path

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
from ..integrations.jira.client import JIRAClient
from ..integrations.github.client import GitHubClient
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker
from ..workspace.multi_repo_manager import MultiRepoManager


console = Console()


@click.group()
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.pass_context
def cli(ctx, workspace):
    """Agent Framework - AI-powered JIRA ticket automation."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = Path(workspace)


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
                # Simple assignment: give to first enabled agent
                assigned_to = next(
                    (a.queue for a in agents_config if a.enabled),
                    "engineer"
                )
                task = jira_client.issue_to_task(issue, assigned_to)
                queue.push(task, assigned_to)

            console.print(f"[green]✓ Queued {len(issues)} tickets[/]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


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

        # Assign to agent
        assigned_to = agent or next(
            (a.queue for a in agents_config if a.enabled),
            "engineer"
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
@click.pass_context
def status(ctx):
    """Show system status."""
    workspace = ctx.obj["workspace"]

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
@click.option("--watchdog/--no-watchdog", default=True, help="Start watchdog")
@click.pass_context
def start(ctx, watchdog):
    """Start the agent system."""
    workspace = ctx.obj["workspace"]

    console.print("[bold green]Starting Agent Framework[/]")

    try:
        # Create orchestrator
        orchestrator = Orchestrator(workspace)

        # Setup signal handlers for graceful shutdown
        orchestrator.setup_signal_handlers()

        # Spawn all agents
        processes = orchestrator.spawn_all_agents()
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
@click.option("--fix", is_flag=True, help="Auto-fix detected issues")
@click.pass_context
def check(ctx, fix):
    """Run circuit breaker safety checks."""
    workspace = ctx.obj["workspace"]

    console.print("[bold]Running Circuit Breaker Checks[/]")

    try:
        # Load config
        framework_config = load_config(workspace / "agent-framework.yaml")

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
        # Load config
        framework_config = load_config(workspace / "agent-framework.yaml")

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

    # Run Claude CLI with timeout
    try:
        result = subprocess.run(
            [claude_cmd, "-m", prompt],
            cwd=cwd,
            capture_output=False,  # Let Claude output directly to console
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI execution exceeded timeout of {timeout} seconds")


if __name__ == "__main__":
    cli()
