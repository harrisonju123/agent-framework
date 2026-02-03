"""Live TUI dashboard for agent activity."""

import asyncio
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.activity import ActivityManager, AgentStatus
from ..core.config import load_agents
from ..queue.file_queue import FileQueue


class AgentDashboard:
    """Live TUI dashboard for agent activity."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.activity_manager = ActivityManager(workspace)
        self.console = Console()
        self.start_time = datetime.utcnow()

    def make_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )

        layout["main"].split_row(
            Layout(name="agents"),
            Layout(name="queues", ratio=1)
        )

        return layout

    def render_header(self) -> Panel:
        """Render dashboard header."""
        uptime = datetime.utcnow() - self.start_time
        uptime_str = f"{int(uptime.total_seconds() // 60)}m {int(uptime.total_seconds() % 60)}s"

        header_text = Text()
        header_text.append("🤖 Agent Activity Dashboard", style="bold cyan")
        header_text.append(f" • Press Ctrl+C to exit • Updates every 2s • Uptime: {uptime_str}", style="dim")

        return Panel(header_text, style="blue")

    def render_agents_table(self) -> Table:
        """Render agent status table."""
        table = Table(title="Agent Status", expand=True)
        table.add_column("Agent", style="cyan", width=15)
        table.add_column("Status", width=12)
        table.add_column("Current Activity", style="white")
        table.add_column("Elapsed", justify="right", width=12)

        activities = self.activity_manager.get_all_activities()

        # Load agent configs to get all agents
        agents_config = load_agents(self.workspace / "config" / "agents.yaml")

        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            # Find activity for this agent
            activity = next((a for a in activities if a.agent_id == agent_def.id), None)

            if not activity or activity.status == AgentStatus.IDLE:
                table.add_row(
                    agent_def.name,
                    "⏸  [yellow]Idle[/yellow]",
                    "[dim]Waiting for tasks[/dim]",
                    "-"
                )
            elif activity.status == AgentStatus.WORKING and activity.current_task:
                # Format task title
                task_title = activity.current_task.title[:40] + "..." if len(activity.current_task.title) > 40 else activity.current_task.title
                task_id_short = activity.current_task.id[:20] + "..."

                # Format phase
                phase_text = activity.current_phase.value.replace("_", " ").title() if activity.current_phase else "Processing"

                # Calculate elapsed time
                elapsed = activity.get_elapsed_seconds()
                elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed else "-"

                table.add_row(
                    agent_def.name,
                    "🔄 [green]Working[/green]",
                    f"{phase_text}\n[dim]{task_id_short}[/dim]",
                    elapsed_str
                )
            else:
                table.add_row(
                    agent_def.name,
                    "❌ [red]Dead[/red]",
                    "[dim]No heartbeat[/dim]",
                    "-"
                )

        return table

    def render_recent_activity(self) -> Panel:
        """Render recent activity events."""
        events = self.activity_manager.get_recent_events(limit=5)

        text = Text()

        if not events:
            text.append("No recent activity", style="dim")
        else:
            for event in events:
                timestamp_str = event.timestamp.strftime("%H:%M:%S")

                if event.type == "complete":
                    duration_sec = event.duration_ms // 1000 if event.duration_ms else 0
                    duration_str = f"{duration_sec // 60}m {duration_sec % 60}s"
                    text.append(f"✓ {timestamp_str}", style="green bold")
                    text.append(f" - {event.agent} completed: {event.title} ({duration_str})\n")
                elif event.type == "fail":
                    text.append(f"✗ {timestamp_str}", style="red bold")
                    text.append(f" - {event.agent} failed: {event.title} (retry {event.retry_count}/5)\n")
                elif event.type == "start":
                    text.append(f"▶ {timestamp_str}", style="blue bold")
                    text.append(f" - {event.agent} started: {event.title}\n")

        return Panel(text, title="Recent Activity", border_style="blue")

    def render_queue_stats(self) -> Panel:
        """Render queue statistics."""
        queue = FileQueue(self.workspace)
        agents_config = load_agents(self.workspace / "config" / "agents.yaml")

        text = Text()

        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            stats = queue.get_queue_stats(agent_def.queue)
            count = stats["count"]

            style = "yellow" if count > 0 else "dim"
            text.append(f"• {agent_def.name}: ", style="cyan")
            text.append(f"{count} pending\n", style=style)

        return Panel(text, title="Queue Status", border_style="blue")

    def render(self) -> Layout:
        """Render complete dashboard."""
        layout = self.make_layout()

        layout["header"].update(self.render_header())
        layout["agents"].update(self.render_agents_table())
        layout["queues"].update(self.render_queue_stats())
        layout["footer"].update(self.render_recent_activity())

        return layout

    async def run(self, refresh_interval: float = 2.0):
        """Run dashboard with live updates."""
        with Live(self.render(), console=self.console, refresh_per_second=0.5) as live:
            try:
                while True:
                    await asyncio.sleep(refresh_interval)
                    live.update(self.render())
            except KeyboardInterrupt:
                pass
