"""Main CLI for agent framework."""

import asyncio
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
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker


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


if __name__ == "__main__":
    cli()
