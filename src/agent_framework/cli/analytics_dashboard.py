"""
Analytics dashboard with performance metrics, failure analysis, and optimization insights.

Provides tabbed interface showing:
- Performance metrics (success rates, token efficiency, costs)
- Failure analysis (common patterns, recommendations)
- Shadow mode optimization results
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..analytics.performance_metrics import PerformanceMetrics
from ..analytics.failure_analyzer import FailureAnalyzer
from ..analytics.shadow_mode_analyzer import ShadowModeAnalyzer
from ..analytics.llm_metrics import LlmMetrics
from ..analytics.chain_metrics import ChainMetrics


class AnalyticsDashboard:
    """Live TUI dashboard for agent analytics."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.console = Console()
        self.start_time = datetime.now(timezone.utc)

        # Analytics modules
        self.performance_metrics = PerformanceMetrics(workspace)
        self.failure_analyzer = FailureAnalyzer(workspace)
        self.shadow_analyzer = ShadowModeAnalyzer(workspace)
        self.llm_metrics = LlmMetrics(workspace)
        self.chain_metrics = ChainMetrics(workspace)

        # Current tab (0=Performance, 1=Failures, 2=Optimizations, 3=LLM Costs, 4=Workflow)
        self.current_tab = 0
        self.tab_names = ["Performance", "Failures", "Optimizations", "LLM Costs", "Workflow"]

    def make_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="tabs", size=1),
            Layout(name="main"),
            Layout(name="footer", size=2)
        )

        return layout

    def render_header(self) -> Panel:
        """Render dashboard header."""
        uptime = datetime.now(timezone.utc) - self.start_time
        uptime_str = f"{int(uptime.total_seconds() // 60)}m {int(uptime.total_seconds() % 60)}s"

        header_text = Text()
        header_text.append("📊 Agent Analytics Dashboard", style="bold cyan")
        header_text.append(
            f" • Press Ctrl+C to exit • Tab to switch views • Uptime: {uptime_str}",
            style="dim"
        )

        return Panel(header_text, style="blue")

    def render_tabs(self) -> Panel:
        """Render tab selector."""
        text = Text()

        for i, tab_name in enumerate(self.tab_names):
            if i == self.current_tab:
                text.append(f" [{tab_name}] ", style="bold cyan on blue")
            else:
                text.append(f"  {tab_name}  ", style="dim")

        return Panel(text, style="blue")

    def render_performance_tab(self) -> Layout:
        """Render performance metrics tab."""
        try:
            report = self.performance_metrics.generate_report(hours=24)
        except Exception as e:
            return Layout(Panel(f"Error generating performance report: {e}", style="red"))

        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=8),
            Layout(name="agents", ratio=1),
            Layout(name="failures", size=12)
        )

        # Overview panel
        overview_text = Text()
        overview_text.append(f"📈 Overall Success Rate: ", style="bold")
        success_style = "green" if report.overall_success_rate >= 85 else "yellow" if report.overall_success_rate >= 70 else "red"
        overview_text.append(f"{report.overall_success_rate:.1f}%\n", style=success_style)

        overview_text.append(f"📊 Total Tasks: ", style="bold")
        overview_text.append(f"{report.total_tasks}\n")

        overview_text.append(f"💰 Total Cost: ", style="bold")
        overview_text.append(f"${report.total_cost:.2f}\n")

        overview_text.append(f"⏱  Time Range: ", style="bold")
        overview_text.append(f"Last {report.time_range_hours} hours\n")

        overview_text.append(f"🔄 Last Updated: ", style="bold")
        overview_text.append(f"{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n", style="dim")

        layout["overview"].update(Panel(overview_text, title="Performance Overview", border_style="blue"))

        # Agent performance table
        agent_table = Table(title="Agent Performance", expand=True)
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Tasks", justify="right")
        agent_table.add_column("Success Rate", justify="right")
        agent_table.add_column("Avg Duration", justify="right")
        agent_table.add_column("Avg Cost", justify="right")
        agent_table.add_column("Retry Rate", justify="right")

        for agent in report.agent_performance[:5]:  # Top 5 agents
            success_style = "green" if agent.success_rate >= 85 else "yellow" if agent.success_rate >= 70 else "red"
            retry_style = "green" if agent.retry_rate < 10 else "yellow" if agent.retry_rate < 30 else "red"

            agent_table.add_row(
                agent.agent_id,
                str(agent.total_tasks),
                f"[{success_style}]{agent.success_rate:.1f}%[/{success_style}]",
                f"{agent.avg_duration_seconds:.1f}s",
                f"${agent.avg_cost_per_task:.4f}",
                f"[{retry_style}]{agent.retry_rate:.1f}%[/{retry_style}]"
            )

        layout["agents"].update(agent_table)

        # Top failures panel
        failures_text = Text()
        failures_text.append("Top Failure Patterns:\n\n", style="bold red")

        for i, failure in enumerate(report.top_failures[:5], 1):
            failures_text.append(f"{i}. ", style="bold")
            failures_text.append(f"{failure['error_pattern']}\n", style="white")
            failures_text.append(f"   Count: {failure['count']} ({failure['percentage']:.1f}%)\n", style="dim")
            failures_text.append(f"   Agents: {', '.join(failure['affected_agents'])}\n", style="dim cyan")
            failures_text.append(f"   Samples: {', '.join(failure['sample_task_ids'][:2])}\n\n", style="dim yellow")

        layout["failures"].update(Panel(failures_text, title="Failure Analysis", border_style="red"))

        return layout

    def render_failures_tab(self) -> Layout:
        """Render failure analysis tab."""
        try:
            report = self.failure_analyzer.analyze(hours=168)
        except Exception as e:
            return Layout(Panel(f"Error generating failure report: {e}", style="red"))

        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=10),
            Layout(name="categories", ratio=1),
            Layout(name="recommendations", size=10)
        )

        # Overview panel
        overview_text = Text()
        overview_text.append(f"🔍 Total Failures: ", style="bold")
        overview_text.append(f"{report.total_failures}\n", style="red")

        overview_text.append(f"📉 Failure Rate: ", style="bold")
        rate_style = "green" if report.failure_rate < 10 else "yellow" if report.failure_rate < 25 else "red"
        overview_text.append(f"{report.failure_rate:.1f}%\n", style=rate_style)

        overview_text.append(f"⏱  Time Range: ", style="bold")
        overview_text.append(f"Last {report.time_range_hours} hours ({report.time_range_hours // 24} days)\n")

        overview_text.append(f"🔄 Last Updated: ", style="bold")
        overview_text.append(f"{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n", style="dim")

        if report.trends:
            overview_text.append("📈 Trending Issues:\n", style="bold yellow")
            for trend in report.trends[:3]:
                if trend.is_increasing:
                    overview_text.append(f"  ⬆ {trend.category}: ", style="red")
                    overview_text.append(f"+{trend.weekly_change_pct:.1f}% ({trend.weekly_count} cases)\n", style="red")

        layout["overview"].update(Panel(overview_text, title="Failure Overview", border_style="red"))

        # Categories table
        cat_table = Table(title="Failure Categories", expand=True)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Percentage", justify="right")
        cat_table.add_column("Affected Agents")

        for category in report.categories[:10]:
            cat_table.add_row(
                category.category.replace('_', ' ').title(),
                str(category.count),
                f"{category.percentage:.1f}%",
                ', '.join(category.affected_agents)
            )

        layout["categories"].update(cat_table)

        # Recommendations panel
        rec_text = Text()
        rec_text.append("💡 Top Recommendations:\n\n", style="bold green")

        for i, rec in enumerate(report.top_recommendations, 1):
            rec_text.append(f"{i}. ", style="bold")
            rec_text.append(f"{rec}\n\n", style="white")

        layout["recommendations"].update(Panel(rec_text, title="Recommendations", border_style="green"))

        return layout

    def render_optimizations_tab(self) -> Layout:
        """Render shadow mode optimization analysis tab."""
        try:
            report = self.shadow_analyzer.analyze(hours=168)
        except Exception as e:
            return Layout(Panel(f"Error generating optimization report: {e}", style="red"))

        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=8),
            Layout(name="strategies", ratio=1),
            Layout(name="recommendation", size=6)
        )

        # Overview panel
        overview_text = Text()
        overview_text.append(f"🔬 Shadow Mode Runs: ", style="bold")
        overview_text.append(f"{report.total_shadow_runs}\n")

        overview_text.append(f"📊 Strategies Analyzed: ", style="bold")
        overview_text.append(f"{report.strategies_analyzed}\n")

        overview_text.append(f"✅ Safe Strategies: ", style="bold")
        safe_count = len(report.safe_strategies)
        safe_style = "green" if safe_count > 0 else "yellow"
        overview_text.append(f"{safe_count}\n", style=safe_style)

        if report.safe_strategies:
            overview_text.append(f"  → {', '.join(report.safe_strategies)}\n", style="dim green")

        overview_text.append(f"⏱  Time Range: ", style="bold")
        overview_text.append(f"Last {report.time_range_hours} hours ({report.time_range_hours // 24} days)\n")

        layout["overview"].update(Panel(overview_text, title="Shadow Mode Overview", border_style="blue"))

        # Strategies table
        strat_table = Table(title="Optimization Strategies", expand=True)
        strat_table.add_column("Strategy", style="cyan")
        strat_table.add_column("Comparisons", justify="right")
        strat_table.add_column("Avg Savings", justify="right")
        strat_table.add_column("Status")

        for metric in report.metrics:
            status = "✅ Safe" if metric.is_safe_to_enable else "⚠ Needs More Data"
            status_style = "green" if metric.is_safe_to_enable else "yellow"

            strat_table.add_row(
                metric.strategy_name,
                str(metric.total_comparisons),
                f"{metric.avg_token_savings_pct:.1f}%",
                f"[{status_style}]{status}[/{status_style}]"
            )

        layout["strategies"].update(strat_table)

        # Overall recommendation
        rec_text = Text()
        rec_text.append("💡 Recommendation:\n\n", style="bold green")
        rec_text.append(f"{report.overall_recommendation}\n\n", style="white")

        if report.metrics:
            rec_text.append("Strategy Details:\n", style="bold")
            for metric in report.metrics:
                if metric.is_safe_to_enable:
                    rec_text.append(f"✅ {metric.strategy_name}: ", style="green")
                    rec_text.append(f"{metric.recommendation}\n", style="white")

        layout["recommendation"].update(Panel(rec_text, title="Deployment Recommendations", border_style="green"))

        return layout

    def render_llm_costs_tab(self) -> Layout:
        """Render LLM cost and token tracking tab."""
        try:
            report = self.llm_metrics.generate_report(hours=24)
        except Exception as e:
            return Layout(Panel(f"Error generating LLM metrics report: {e}", style="red"))

        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=10),
            Layout(name="model_usage", ratio=1),
            Layout(name="top_tasks", size=12),
        )

        # Overview panel
        overview_text = Text()
        overview_text.append(f"Total Cost: ", style="bold")
        overview_text.append(f"${report.total_cost:.4f}\n")

        overview_text.append(f"Total LLM Calls: ", style="bold")
        overview_text.append(f"{report.total_llm_calls}\n")

        overview_text.append(f"Token Efficiency (out/in): ", style="bold")
        eff_style = "green" if report.overall_token_efficiency > 0.3 else "yellow"
        overview_text.append(f"{report.overall_token_efficiency:.3f}\n", style=eff_style)

        overview_text.append(f"Total Tokens In: ", style="bold")
        overview_text.append(f"{report.total_tokens_in:,}\n")

        overview_text.append(f"Total Tokens Out: ", style="bold")
        overview_text.append(f"{report.total_tokens_out:,}\n")

        overview_text.append(f"Time Range: ", style="bold")
        overview_text.append(f"Last {report.time_range_hours} hours\n")

        layout["overview"].update(Panel(overview_text, title="LLM Cost Overview", border_style="blue"))

        # Model usage table
        model_table = Table(title="Model Usage", expand=True)
        model_table.add_column("Tier", style="cyan")
        model_table.add_column("Calls", justify="right")
        model_table.add_column("Total Cost", justify="right")
        model_table.add_column("Avg Cost/Call", justify="right")
        model_table.add_column("Avg Latency", justify="right")
        model_table.add_column("Cost Share %", justify="right")

        for tier in report.model_tiers:
            model_table.add_row(
                tier.tier,
                str(tier.call_count),
                f"${tier.total_cost:.4f}",
                f"${tier.avg_cost_per_call:.6f}",
                f"{tier.avg_duration_ms:.0f}ms",
                f"{tier.cost_share_pct:.1f}%",
            )

        layout["model_usage"].update(model_table)

        # Top cost tasks table
        task_table = Table(title="Top Cost Tasks", expand=True)
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Cost", justify="right")
        task_table.add_column("LLM Calls", justify="right")
        task_table.add_column("Tokens In", justify="right")
        task_table.add_column("Tokens Out", justify="right")
        task_table.add_column("Efficiency", justify="right")

        for task in report.top_cost_tasks[:10]:
            task_table.add_row(
                task.task_id[:12],
                f"${task.total_cost:.4f}",
                str(task.llm_call_count),
                f"{task.total_tokens_in:,}",
                f"{task.total_tokens_out:,}",
                f"{task.token_efficiency:.3f}",
            )

        layout["top_tasks"].update(task_table)

        return layout

    def render_workflow_tab(self) -> Layout:
        """Render workflow chain metrics tab."""
        try:
            report = self.chain_metrics.generate_report(hours=24)
        except Exception as e:
            return Layout(Panel(f"Error generating workflow report: {e}", style="red"))

        layout = Layout()
        layout.split_column(
            Layout(name="overview", size=10),
            Layout(name="step_metrics", ratio=1),
            Layout(name="recent_chains", size=14),
        )

        # Overview panel
        overview_text = Text()
        overview_text.append(f"Total Chains: ", style="bold")
        overview_text.append(f"{report.total_chains}\n")

        overview_text.append(f"Completed: ", style="bold")
        overview_text.append(f"{report.completed_chains}\n")

        overview_text.append(f"Completion Rate: ", style="bold")
        rate_style = "green" if report.chain_completion_rate >= 80 else "yellow" if report.chain_completion_rate >= 50 else "red"
        overview_text.append(f"{report.chain_completion_rate:.1f}%\n", style=rate_style)

        overview_text.append(f"Avg Chain Depth: ", style="bold")
        overview_text.append(f"{report.avg_chain_depth:.1f} steps\n")

        overview_text.append(f"Avg Files Modified: ", style="bold")
        overview_text.append(f"{report.avg_files_modified:.1f}\n")

        overview_text.append(f"Avg Attempts: ", style="bold")
        attempt_style = "green" if report.avg_attempts < 1.5 else "yellow" if report.avg_attempts < 3 else "red"
        overview_text.append(f"{report.avg_attempts:.1f}\n", style=attempt_style)

        overview_text.append(f"Time Range: ", style="bold")
        overview_text.append(f"Last {report.time_range_hours} hours\n")

        layout["overview"].update(Panel(overview_text, title="Workflow Overview", border_style="blue"))

        # Step metrics table
        step_table = Table(title="Step Type Metrics", expand=True)
        step_table.add_column("Step", style="cyan")
        step_table.add_column("Count", justify="right")
        step_table.add_column("Success Rate", justify="right")
        step_table.add_column("Avg Duration", justify="right")
        step_table.add_column("p50", justify="right")
        step_table.add_column("p90", justify="right")

        for metric in report.step_type_metrics:
            success_style = "green" if metric.success_rate >= 90 else "yellow" if metric.success_rate >= 70 else "red"
            step_table.add_row(
                metric.step_id,
                str(metric.total_count),
                f"[{success_style}]{metric.success_rate:.1f}%[/{success_style}]",
                f"{metric.avg_duration_seconds:.1f}s",
                f"{metric.p50_duration_seconds:.1f}s",
                f"{metric.p90_duration_seconds:.1f}s",
            )

        layout["step_metrics"].update(step_table)

        # Recent chains panel
        chain_table = Table(title="Recent Chains", expand=True)
        chain_table.add_column("Root Task", style="cyan")
        chain_table.add_column("Workflow")
        chain_table.add_column("Steps", justify="right")
        chain_table.add_column("Attempts", justify="right")
        chain_table.add_column("Files", justify="right")
        chain_table.add_column("Duration", justify="right")
        chain_table.add_column("Status")

        for chain in report.recent_chains:
            status = "[green]Done[/green]" if chain.completed else "[yellow]In Progress[/yellow]"
            chain_table.add_row(
                chain.root_task_id[:12],
                chain.workflow,
                str(chain.step_count),
                str(chain.attempt),
                str(chain.files_modified_count),
                f"{chain.total_duration_seconds:.0f}s",
                status,
            )

        layout["recent_chains"].update(chain_table)

        return layout

    def render_footer(self) -> Panel:
        """Render footer with help text."""
        text = Text()
        text.append("Press ", style="dim")
        text.append("Tab", style="bold")
        text.append(" to cycle through tabs • ", style="dim")
        text.append("Ctrl+C", style="bold")
        text.append(" to exit", style="dim")

        return Panel(text, style="blue")

    def render(self) -> Layout:
        """Render complete dashboard."""
        layout = self.make_layout()

        layout["header"].update(self.render_header())
        layout["tabs"].update(self.render_tabs())
        layout["footer"].update(self.render_footer())

        # Render active tab content
        if self.current_tab == 0:
            layout["main"].update(self.render_performance_tab())
        elif self.current_tab == 1:
            layout["main"].update(self.render_failures_tab())
        elif self.current_tab == 2:
            layout["main"].update(self.render_optimizations_tab())
        elif self.current_tab == 3:
            layout["main"].update(self.render_llm_costs_tab())
        elif self.current_tab == 4:
            layout["main"].update(self.render_workflow_tab())
        else:
            layout["main"].update(self.render_performance_tab())

        return layout

    async def run(self, refresh_interval: float = 5.0):
        """Run dashboard with live updates."""
        with Live(self.render(), console=self.console, refresh_per_second=0.5) as live:
            try:
                while True:
                    await asyncio.sleep(refresh_interval)
                    live.update(self.render())
            except KeyboardInterrupt:
                pass

    def next_tab(self):
        """Switch to next tab."""
        self.current_tab = (self.current_tab + 1) % len(self.tab_names)

    def prev_tab(self):
        """Switch to previous tab."""
        self.current_tab = (self.current_tab - 1) % len(self.tab_names)
