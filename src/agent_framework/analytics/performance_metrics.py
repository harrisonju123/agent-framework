"""
Performance metrics aggregation from activity stream.

Analyzes agent performance including success rates, completion times,
token efficiency, and cost per task.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskMetrics(BaseModel):
    """Metrics for a single task."""
    task_id: str
    agent: str
    title: str
    status: str  # "completed", "failed", "in_progress"
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    pr_url: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class AgentPerformance(BaseModel):
    """Performance metrics for an agent."""
    agent_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    success_rate: float
    avg_duration_seconds: float
    avg_tokens_per_task: int
    avg_cost_per_task: float
    total_cost: float
    retry_rate: float  # Percentage of tasks that needed retries


class TaskTypeMetrics(BaseModel):
    """Metrics aggregated by task type."""
    task_type: str
    total_tasks: int
    success_rate: float
    avg_duration_seconds: float
    p50_tokens: int
    p90_tokens: int
    p99_tokens: int
    avg_tokens: int
    avg_cost: float


class PerformanceReport(BaseModel):
    """Complete performance report."""
    generated_at: datetime
    time_range_hours: int
    overall_success_rate: float
    total_tasks: int
    total_cost: float
    agent_performance: List[AgentPerformance]
    task_type_metrics: List[TaskTypeMetrics]
    top_failures: List[Dict[str, Any]]  # Top 5 failure patterns


class PerformanceMetrics:
    """Aggregates and analyzes agent performance metrics."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.metrics_dir / "performance.json"

    def generate_report(self, hours: int = 24) -> PerformanceReport:
        """
        Generate performance report for the specified time range.

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            PerformanceReport with aggregated metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Parse activity stream
        events = self._read_events(cutoff_time)

        # Build task metrics
        task_metrics = self._build_task_metrics(events)

        # Calculate agent performance
        agent_performance = self._calculate_agent_performance(task_metrics)

        # Calculate task type metrics
        task_type_metrics = self._calculate_task_type_metrics(task_metrics)

        # Find top failure patterns
        top_failures = self._find_top_failures(task_metrics)

        # Calculate overall metrics
        total_tasks = len(task_metrics)
        completed = sum(1 for t in task_metrics.values() if t.status == "completed")
        total_cost = sum(t.cost for t in task_metrics.values())
        success_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0.0

        report = PerformanceReport(
            generated_at=datetime.utcnow(),
            time_range_hours=hours,
            overall_success_rate=success_rate,
            total_tasks=total_tasks,
            total_cost=total_cost,
            agent_performance=agent_performance,
            task_type_metrics=task_type_metrics,
            top_failures=top_failures,
        )

        # Save report
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Performance report saved to {self.output_file}")

        return report

    def _read_events(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Read and filter events from activity stream."""
        if not self.stream_file.exists():
            return []

        events = []
        for line in self.stream_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug(f"Failed to parse event: {e}")

        return events

    def _build_task_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, TaskMetrics]:
        """Build task metrics from events."""
        tasks: Dict[str, TaskMetrics] = {}

        for event in events:
            task_id = event.get('task_id')
            if not task_id:
                continue

            event_type = event.get('type')

            if event_type == 'start':
                tasks[task_id] = TaskMetrics(
                    task_id=task_id,
                    agent=event.get('agent', 'unknown'),
                    title=event.get('title', ''),
                    status='in_progress',
                    started_at=datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')),
                )

            elif event_type == 'complete' and task_id in tasks:
                task = tasks[task_id]
                task.status = 'completed'
                task.completed_at = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                task.duration_ms = event.get('duration_ms', 0)
                task.pr_url = event.get('pr_url')
                # Extract token usage and cost from event
                task.input_tokens = event.get('input_tokens', 0)
                task.output_tokens = event.get('output_tokens', 0)
                task.cost = event.get('cost', 0.0)

            elif event_type == 'fail' and task_id in tasks:
                task = tasks[task_id]
                task.status = 'failed'
                task.completed_at = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                task.retry_count = event.get('retry_count', 0)
                task.error_message = event.get('error_message')

        return tasks

    def _calculate_agent_performance(self, task_metrics: Dict[str, TaskMetrics]) -> List[AgentPerformance]:
        """Calculate performance metrics per agent."""
        agent_tasks: Dict[str, List[TaskMetrics]] = defaultdict(list)

        for task in task_metrics.values():
            agent_tasks[task.agent].append(task)

        performance = []
        for agent_id, tasks in agent_tasks.items():
            total = len(tasks)
            completed = sum(1 for t in tasks if t.status == 'completed')
            failed = sum(1 for t in tasks if t.status == 'failed')

            durations = [t.duration_ms / 1000 for t in tasks if t.duration_ms]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            total_tokens = sum(t.input_tokens + t.output_tokens for t in tasks)
            avg_tokens = total_tokens // total if total > 0 else 0

            total_cost = sum(t.cost for t in tasks)
            avg_cost = total_cost / total if total > 0 else 0.0

            retried = sum(1 for t in tasks if t.retry_count > 0)
            retry_rate = (retried / total * 100) if total > 0 else 0.0

            success_rate = (completed / total * 100) if total > 0 else 0.0

            performance.append(AgentPerformance(
                agent_id=agent_id,
                total_tasks=total,
                completed_tasks=completed,
                failed_tasks=failed,
                success_rate=success_rate,
                avg_duration_seconds=avg_duration,
                avg_tokens_per_task=avg_tokens,
                avg_cost_per_task=avg_cost,
                total_cost=total_cost,
                retry_rate=retry_rate,
            ))

        return sorted(performance, key=lambda x: x.total_tasks, reverse=True)

    def _calculate_task_type_metrics(self, task_metrics: Dict[str, TaskMetrics]) -> List[TaskTypeMetrics]:
        """Calculate metrics aggregated by task type (inferred from title)."""
        # Simple heuristic: group by first word in title or JIRA prefix
        type_tasks: Dict[str, List[TaskMetrics]] = defaultdict(list)

        for task in task_metrics.values():
            task_type = self._infer_task_type(task.title)
            type_tasks[task_type].append(task)

        metrics = []
        for task_type, tasks in type_tasks.items():
            total = len(tasks)
            completed = sum(1 for t in tasks if t.status == 'completed')
            success_rate = (completed / total * 100) if total > 0 else 0.0

            durations = [t.duration_ms / 1000 for t in tasks if t.duration_ms]
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            tokens = sorted([t.input_tokens + t.output_tokens for t in tasks])
            p50 = tokens[len(tokens) // 2] if tokens else 0
            p90 = tokens[int(len(tokens) * 0.9)] if tokens else 0
            p99 = tokens[int(len(tokens) * 0.99)] if tokens else 0
            avg_tokens = sum(tokens) // len(tokens) if tokens else 0

            avg_cost = sum(t.cost for t in tasks) / total if total > 0 else 0.0

            metrics.append(TaskTypeMetrics(
                task_type=task_type,
                total_tasks=total,
                success_rate=success_rate,
                avg_duration_seconds=avg_duration,
                p50_tokens=p50,
                p90_tokens=p90,
                p99_tokens=p99,
                avg_tokens=avg_tokens,
                avg_cost=avg_cost,
            ))

        return sorted(metrics, key=lambda x: x.total_tasks, reverse=True)

    def _infer_task_type(self, title: str) -> str:
        """Infer task type from title."""
        title_lower = title.lower()

        if 'escalation' in title_lower:
            return 'escalation'
        elif title.startswith('[') and ']' in title:
            # Extract tag like [Bug Fix] or [Feature]
            return title[1:title.index(']')]
        elif 'jira-' in title_lower:
            return 'jira_task'
        elif 'test' in title_lower:
            return 'testing'
        elif 'refactor' in title_lower:
            return 'refactoring'
        elif 'implement' in title_lower or 'add' in title_lower:
            return 'implementation'
        elif 'fix' in title_lower or 'bug' in title_lower:
            return 'bug_fix'
        else:
            return 'other'

    def _find_top_failures(self, task_metrics: Dict[str, TaskMetrics], limit: int = 5) -> List[Dict[str, Any]]:
        """Find the most common failure patterns."""
        failed_tasks = [t for t in task_metrics.values() if t.status == 'failed']

        if not failed_tasks:
            return []

        # Group by error message prefix (first 100 chars)
        error_groups: Dict[str, List[TaskMetrics]] = defaultdict(list)
        for task in failed_tasks:
            error_key = (task.error_message or 'Unknown error')[:100]
            error_groups[error_key].append(task)

        # Sort by frequency
        top_errors = sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True)[:limit]

        return [
            {
                'error_pattern': error_key,
                'count': len(tasks),
                'percentage': len(tasks) / len(failed_tasks) * 100,
                'affected_agents': list(set(t.agent for t in tasks)),
                'sample_task_ids': [t.task_id for t in tasks[:3]],
            }
            for error_key, tasks in top_errors
        ]
