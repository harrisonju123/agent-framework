"""Live TUI dashboard for agent activity with keyboard controls."""

import asyncio
import select
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Unix-only imports for terminal handling - graceful fallback for Windows
try:
    import termios
    import tty
    HAS_TTY = True
except ImportError:
    HAS_TTY = False

from ..core.activity import ActivityManager, AgentStatus, TaskPhase
from ..core.config import load_agents
from ..core.orchestrator import Orchestrator
from ..queue.file_queue import FileQueue
from ..safeguards.circuit_breaker import CircuitBreaker


class FocusPanel(str, Enum):
    """Which panel has focus for selection."""
    AGENTS = "agents"
    FAILED = "failed"


class AgentDashboard:
    """Live TUI dashboard for agent activity with keyboard controls."""

    # Spinner frames for animated progress
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Max failed tasks to display
    MAX_FAILED_DISPLAY = 5

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.activity_manager = ActivityManager(workspace)
        self.console = Console()
        self.start_time = datetime.utcnow()
        self.circuit_breaker = CircuitBreaker(workspace)
        self.queue = FileQueue(workspace)
        self.orchestrator = Orchestrator(workspace)

        # Selection state
        self.focus_panel = FocusPanel.AGENTS
        self.agent_index = 0
        self.failed_index = 0
        self.show_help = False

        # Animation state
        self.spinner_frame = 0
        self.tick = 0

        # Cached data for selection
        self._agents_list: List[str] = []
        self._failed_tasks_list: List = []
        self._cached_agents_config = None

    def _get_agents_config(self):
        """Get agents config, caching for the current render cycle."""
        if self._cached_agents_config is None:
            self._cached_agents_config = load_agents(self.workspace / "config" / "agents.yaml")
        return self._cached_agents_config

    def _clear_cache(self):
        """Clear cached data at start of render cycle."""
        self._cached_agents_config = None

    def make_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="health", size=6),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )

        layout["main"].split_row(
            Layout(name="agents"),
            Layout(name="queues", ratio=1)
        )

        # Split footer into activity and failed tasks
        layout["footer"].split_row(
            Layout(name="activity"),
            Layout(name="failed")
        )

        return layout

    def render_header(self) -> Panel:
        """Render dashboard header with key hints."""
        uptime = datetime.utcnow() - self.start_time
        uptime_str = f"{int(uptime.total_seconds() // 60)}m {int(uptime.total_seconds() % 60)}s"

        # Check pause status
        pause_file = self.workspace / ".agent-communication" / "pause"
        is_paused = pause_file.exists()

        header_text = Text()
        header_text.append("🤖 Agent Dashboard", style="bold cyan")
        if is_paused:
            header_text.append(" [PAUSED]", style="bold yellow")
        header_text.append(f" • Uptime: {uptime_str}", style="dim")
        header_text.append(" • ", style="dim")
        header_text.append("j/k", style="cyan")
        header_text.append(":nav ", style="dim")
        header_text.append("r", style="cyan")
        header_text.append(":retry ", style="dim")
        header_text.append("R", style="cyan")
        header_text.append(":restart ", style="dim")
        header_text.append("p", style="cyan")
        header_text.append(":pause ", style="dim")
        header_text.append("?", style="cyan")
        header_text.append(":help ", style="dim")
        header_text.append("q", style="cyan")
        header_text.append(":quit", style="dim")

        return Panel(header_text, style="blue")

    # Maps tool names to short action verbs for TUI display
    TOOL_VERBS = {
        "Read": "Read",
        "Edit": "Edit",
        "Write": "Write",
        "Bash": "Run",
        "Grep": "Grep",
        "Glob": "Glob",
        "Task": "Task",
    }

    def _get_phase_display(self, phase: Optional[TaskPhase], phase_started: Optional[datetime], tool_activity=None) -> Tuple[str, str]:
        """Get phase display text with spinner and progress dots.

        Args:
            tool_activity: Optional ToolActivity from activity file, shown
                during EXECUTING_LLM to give visibility into Claude's actions.

        Returns:
            Tuple of (phase_text, elapsed_str)
        """
        if not phase:
            return "Processing", ""

        phase_val = phase.value if hasattr(phase, 'value') else str(phase)
        phase_text = phase_val.replace("_", " ").title()

        # Show tool activity instead of generic "Executing Llm" when available
        if phase == TaskPhase.EXECUTING_LLM:
            spinner = self.SPINNER_FRAMES[self.spinner_frame % len(self.SPINNER_FRAMES)]
            if tool_activity:
                verb = self.TOOL_VERBS.get(tool_activity.tool_name, tool_activity.tool_name)
                summary = f": {tool_activity.tool_input_summary}" if tool_activity.tool_input_summary else ""
                # Truncate to fit column width
                tool_text = f"{verb}{summary}"
                if len(tool_text) > 25:
                    tool_text = tool_text[:24] + "\u2026"
                phase_text = f"{spinner} {tool_text}"
            else:
                phase_text = f"{spinner} {phase_text}"

        # Calculate phase elapsed time
        elapsed_str = ""
        if phase_started:
            elapsed = (datetime.utcnow() - phase_started).total_seconds()
            elapsed_str = f"{int(elapsed)}s"

        return phase_text, elapsed_str

    def _get_progress_dots(self, phases_completed: int, total_phases: int = 5) -> str:
        """Generate progress dots visualization."""
        filled = min(phases_completed, total_phases)
        empty = total_phases - filled
        return "●" * filled + "○" * empty

    def render_agents_table(self) -> Panel:
        """Render agent status table with selection highlighting."""
        table = Table(expand=True, show_header=True, header_style="bold")
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Agent", style="cyan", width=15)
        table.add_column("Status", width=12)
        table.add_column("Phase", style="white", width=20)
        table.add_column("Progress", width=8)
        table.add_column("Elapsed", justify="right", width=10)

        activities = self.activity_manager.get_all_activities()
        agents_config = self._get_agents_config()

        # Update cached agents list (only enabled agents)
        self._agents_list = [a.id for a in agents_config if a.enabled]

        # Clamp agent_index to valid range
        if self._agents_list:
            self.agent_index = max(0, min(self.agent_index, len(self._agents_list) - 1))
        else:
            self.agent_index = 0

        # Track enabled agent index separately to handle disabled agents correctly
        enabled_idx = 0
        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            # Selection indicator based on enabled index
            is_selected = self.focus_panel == FocusPanel.AGENTS and enabled_idx == self.agent_index
            indicator = "▶" if is_selected else " "
            row_style = "reverse" if is_selected else None

            activity = next((a for a in activities if a.agent_id == agent_def.id), None)

            if not activity or activity.status == AgentStatus.IDLE:
                table.add_row(
                    indicator,
                    agent_def.name,
                    "⏸  [yellow]Idle[/yellow]",
                    "[dim]Waiting for tasks[/dim]",
                    "",
                    "-",
                    style=row_style
                )
            elif activity.status == AgentStatus.COMPLETING and activity.current_task:
                # Task just completed - show brief completing state
                title = activity.current_task.title
                title_display = f"{title[:30]}..." if len(title) > 30 else title
                table.add_row(
                    indicator,
                    agent_def.name,
                    "✓  [blue]Completing[/blue]",
                    f"[dim]{title_display}[/dim]",
                    "[cyan]●●●●●[/cyan]",
                    "-",
                    style=row_style
                )
            elif activity.status == AgentStatus.WORKING and activity.current_task:
                # Get phase info with spinner (pass tool_activity for live tool visibility)
                phase_text, phase_elapsed = self._get_phase_display(
                    activity.current_phase,
                    activity.phases[-1].started_at if activity.phases else None,
                    tool_activity=activity.tool_activity,
                )

                # Progress dots based on completed phases
                progress = self._get_progress_dots(sum(1 for p in activity.phases if p.completed))

                # Total elapsed
                elapsed = activity.get_elapsed_seconds()
                elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed else "-"

                table.add_row(
                    indicator,
                    agent_def.name,
                    "🔄 [green]Working[/green]",
                    f"{phase_text}",
                    f"[cyan]{progress}[/cyan]",
                    elapsed_str,
                    style=row_style
                )
            else:
                table.add_row(
                    indicator,
                    agent_def.name,
                    "❌ [red]Dead[/red]",
                    "[dim]No heartbeat[/dim]",
                    "",
                    "-",
                    style=row_style
                )

            enabled_idx += 1

        title = "Agents"
        if self.focus_panel == FocusPanel.AGENTS:
            title = "[bold cyan]» Agents[/bold cyan]"

        return Panel(table, title=title, border_style="cyan" if self.focus_panel == FocusPanel.AGENTS else "blue")

    def render_failed_tasks(self) -> Panel:
        """Render failed tasks panel with selection."""
        failed_tasks = self.queue.get_all_failed()
        self._failed_tasks_list = failed_tasks

        # Clamp failed_index to valid range (limited to displayed tasks)
        max_idx = min(self.MAX_FAILED_DISPLAY, len(failed_tasks)) - 1
        if max_idx >= 0:
            self.failed_index = max(0, min(self.failed_index, max_idx))
        else:
            self.failed_index = 0

        text = Text()

        if not failed_tasks:
            text.append("No failed tasks", style="dim green")
        else:
            # Show up to MAX_FAILED_DISPLAY failed tasks
            for idx, task in enumerate(failed_tasks[:self.MAX_FAILED_DISPLAY]):
                is_selected = self.focus_panel == FocusPanel.FAILED and idx == self.failed_index
                indicator = "▶ " if is_selected else "  "

                jira_key = task.context.get("jira_key", task.id[:12])
                title = task.title[:30] + "..." if len(task.title) > 30 else task.title
                retry_count = task.retry_count or 0

                if is_selected:
                    text.append(indicator, style="bold cyan")
                    text.append(f"{jira_key}", style="bold red reverse")
                    text.append(f" {title}", style="reverse")
                    text.append(f" (×{retry_count})\n", style="dim reverse")
                else:
                    text.append(indicator)
                    text.append(f"{jira_key}", style="red bold")
                    text.append(f" {title}", style="white")
                    text.append(f" (×{retry_count})\n", style="dim")

                # Show error snippet
                if task.last_error:
                    error_preview = task.last_error[:50] + "..." if len(task.last_error) > 50 else task.last_error
                    text.append(f"   └─ {error_preview}\n", style="dim red")

        title = "Failed Tasks"
        if self.focus_panel == FocusPanel.FAILED:
            title = "[bold cyan]» Failed Tasks[/bold cyan]"

        border_style = "cyan" if self.focus_panel == FocusPanel.FAILED else "red" if failed_tasks else "blue"
        return Panel(text, title=title, border_style=border_style)

    def render_recent_activity(self) -> Panel:
        """Render recent activity events."""
        events = self.activity_manager.get_recent_events(limit=4)

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
                    text.append(f" {event.agent}: {event.title[:25]}... ({duration_str})\n")
                elif event.type == "fail":
                    text.append(f"✗ {timestamp_str}", style="red bold")
                    retry_info = f" ×{event.retry_count}" if event.retry_count else ""
                    text.append(f" {event.agent}: {event.title[:25]}...{retry_info}\n")
                elif event.type == "start":
                    text.append(f"▶ {timestamp_str}", style="blue bold")
                    text.append(f" {event.agent}: {event.title[:25]}...\n")

        return Panel(text, title="Recent Activity", border_style="blue")

    def render_queue_stats(self) -> Panel:
        """Render queue statistics."""
        agents_config = self._get_agents_config()

        text = Text()

        for agent_def in agents_config:
            if not agent_def.enabled:
                continue

            stats = self.queue.get_queue_stats(agent_def.queue)
            count = stats["count"]

            style = "yellow" if count > 0 else "dim"
            text.append(f"• {agent_def.name}: ", style="cyan")
            text.append(f"{count} pending\n", style=style)

        return Panel(text, title="Queue Status", border_style="blue")

    def render_health_status(self) -> Panel:
        """Render system health status."""
        report = self.circuit_breaker.run_all_checks()

        text = Text()

        if report.passed:
            text.append("✓ System Health: ", style="green bold")
            text.append("HEALTHY\n\n", style="green")
        else:
            text.append("⚠ System Health: ", style="red bold")
            text.append("DEGRADED\n\n", style="red")

        for check_name, passed in report.checks.items():
            status_icon = "✓" if passed else "✗"
            status_style = "green" if passed else "red"
            check_label = check_name.replace("_", " ").title()

            text.append(f"{status_icon} {check_label}", style=status_style)

            if not passed and check_name in report.issues:
                for issue in report.issues[check_name][:1]:
                    text.append(f" - {issue[:60]}...", style="dim red")
            text.append("\n")

        pause_marker = self.workspace / ".agent-communication" / "PAUSE_INTAKE"
        if pause_marker.exists():
            text.append("\n⏸  ", style="yellow bold")
            text.append("Task intake PAUSED\n", style="yellow")

        border_style = "green" if report.passed else "red"
        return Panel(text, title="System Health", border_style=border_style)

    def render_help_overlay(self) -> Panel:
        """Render help overlay."""
        text = Text()
        text.append("Keyboard Controls\n\n", style="bold cyan")
        text.append("Navigation:\n", style="bold")
        text.append("  j/↓     Move selection down\n")
        text.append("  k/↑     Move selection up\n")
        text.append("  Tab     Switch panel focus\n\n")
        text.append("Actions:\n", style="bold")
        text.append("  r       Retry selected failed task\n")
        text.append("  R       Restart selected dead agent\n")
        text.append("  p       Toggle pause/resume\n\n")
        text.append("Other:\n", style="bold")
        text.append("  ?       Toggle this help\n")
        text.append("  q       Quit dashboard\n")
        text.append("\n[dim]Press any key to close[/dim]")

        return Panel(text, title="Help", border_style="cyan")

    def render(self) -> Layout:
        """Render complete dashboard."""
        # Clear cache at start of render cycle
        self._clear_cache()

        # Update animation state
        self.tick += 1
        if self.tick % 2 == 0:  # Update spinner every other tick
            self.spinner_frame += 1

        if self.show_help:
            # Show help overlay
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="help"),
            )
            layout["header"].update(self.render_header())
            layout["help"].update(self.render_help_overlay())
            return layout

        layout = self.make_layout()

        layout["header"].update(self.render_header())
        layout["health"].update(self.render_health_status())
        layout["agents"].update(self.render_agents_table())
        layout["queues"].update(self.render_queue_stats())
        layout["activity"].update(self.render_recent_activity())
        layout["failed"].update(self.render_failed_tasks())

        return layout

    def _handle_key(self, key: str) -> Optional[str]:
        """Handle keyboard input. Returns action message or None."""
        if key == 'q':
            return "QUIT"

        if key == '?':
            self.show_help = not self.show_help
            return None

        if self.show_help:
            self.show_help = False
            return None

        # Navigation
        if key in ('j', '\x1b[B'):  # j or down arrow
            if self.focus_panel == FocusPanel.AGENTS:
                if self._agents_list:
                    self.agent_index = min(self.agent_index + 1, len(self._agents_list) - 1)
            else:
                max_idx = min(self.MAX_FAILED_DISPLAY, len(self._failed_tasks_list)) - 1
                if max_idx >= 0:
                    self.failed_index = min(self.failed_index + 1, max_idx)
            return None

        if key in ('k', '\x1b[A'):  # k or up arrow
            if self.focus_panel == FocusPanel.AGENTS:
                self.agent_index = max(self.agent_index - 1, 0)
            else:
                self.failed_index = max(self.failed_index - 1, 0)
            return None

        if key == '\t':  # Tab - switch focus
            if self.focus_panel == FocusPanel.AGENTS:
                self.focus_panel = FocusPanel.FAILED
            else:
                self.focus_panel = FocusPanel.AGENTS
            return None

        # Actions
        if key == 'r':  # Retry failed task
            if self.focus_panel == FocusPanel.FAILED and self._failed_tasks_list:
                # Only allow retrying tasks that are displayed
                max_idx = min(self.MAX_FAILED_DISPLAY, len(self._failed_tasks_list)) - 1
                if 0 <= self.failed_index <= max_idx:
                    task = self._failed_tasks_list[self.failed_index]
                    try:
                        self.queue.requeue_task(task)
                        jira_key = task.context.get("jira_key", task.id[:12])
                        return f"Retried task: {jira_key}"
                    except Exception as e:
                        return f"Failed to retry: {str(e)[:30]}"
            return "Select a failed task first (Tab to switch)"

        if key == 'R':  # Restart dead agent
            if self.focus_panel == FocusPanel.AGENTS and self._agents_list:
                if 0 <= self.agent_index < len(self._agents_list):
                    agent_id = self._agents_list[self.agent_index]
                    # Check if agent is dead
                    activity = self.activity_manager.get_activity(agent_id)
                    if activity and activity.status == AgentStatus.DEAD:
                        try:
                            # Stop and restart the agent
                            self.orchestrator.stop_agent(agent_id, graceful=False)
                            self.orchestrator.spawn_agent(agent_id)
                            return f"Restarted agent: {agent_id}"
                        except Exception as e:
                            return f"Failed to restart: {str(e)[:30]}"
                    return f"Agent {agent_id} is not dead"
            return "Select a dead agent first"

        if key == 'p':  # Toggle pause
            pause_file = self.workspace / ".agent-communication" / "pause"
            pause_file.parent.mkdir(parents=True, exist_ok=True)
            if pause_file.exists():
                pause_file.unlink()
                return "Resumed agent processing"
            else:
                pause_file.write_text(str(int(time.time())))
                return "Paused agent processing"

        return None

    async def _read_key(self) -> Optional[str]:
        """Read a single keypress without blocking.

        Returns None on Windows or when stdin is not a TTY.
        """
        if not HAS_TTY:
            return None

        if not sys.stdin.isatty():
            return None

        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, OSError, AttributeError):
            return None

        try:
            tty.setraw(fd)
            # Use select to check if input is available
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                # Handle escape sequences (arrow keys)
                if ch == '\x1b':
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch += sys.stdin.read(2)
                return ch
            return None
        except Exception:
            return None
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except (termios.error, OSError):
                pass

    async def run(self, refresh_interval: float = 0.5):
        """Run dashboard with live updates and keyboard handling."""
        action_message = None
        action_message_time = None

        with Live(self.render(), console=self.console, refresh_per_second=4) as live:
            try:
                while True:
                    # Check for keyboard input
                    key = await self._read_key()
                    if key:
                        result = self._handle_key(key)
                        if result == "QUIT":
                            break
                        elif result:
                            action_message = result
                            action_message_time = datetime.utcnow()

                    # Clear action message after 3 seconds
                    if action_message_time:
                        if (datetime.utcnow() - action_message_time).total_seconds() > 3:
                            action_message = None
                            action_message_time = None

                    # Update display
                    layout = self.render()

                    # Show action message in header area if present
                    if action_message:
                        header = layout["header"]
                        msg_text = Text()
                        msg_text.append("🤖 Agent Dashboard", style="bold cyan")
                        msg_text.append(f" • {action_message}", style="bold green")
                        header.update(Panel(msg_text, style="blue"))

                    live.update(layout)
                    await asyncio.sleep(refresh_interval)

            except KeyboardInterrupt:
                pass
