"""
Performance metrics aggregation from activity stream.

Analyzes agent performance including success rates, completion times,
token efficiency, and cost per task.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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


class HandoffRecord(BaseModel):
    """Single handoff between two workflow steps."""
    root_task_id: str
    from_agent: str
    to_agent: str
    from_task_id: str
    to_task_id: str
    completed_at: datetime
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    total_handoff_ms: Optional[int] = None
    queue_wait_ms: Optional[int] = None
    post_completion_ms: Optional[int] = None
    status: str  # "completed", "pending", "delayed"


class HandoffSummary(BaseModel):
    """Aggregate handoff metrics for a transition type."""
    transition: str  # e.g. "architect→engineer"
    count: int
    avg_total_ms: float
    p50_total_ms: int
    p90_total_ms: int
    avg_queue_wait_ms: float
    failed_count: int  # queued but never started
    delayed_count: int  # > threshold


class HandoffReport(BaseModel):
    """Complete handoff analysis."""
    generated_at: datetime
    records: List[HandoffRecord]
    summaries: List[HandoffSummary]
    pending_handoffs: List[HandoffRecord]


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
    handoff_summaries: List[HandoffSummary] = []


class PerformanceMetrics:
    """Aggregates and analyzes agent performance metrics."""

    # Handoffs queued longer than this are flagged "delayed"
    DELAYED_THRESHOLD_MS = 60_000

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.metrics_dir / "performance.json"
        self.handoff_log_file = self.metrics_dir / "handoffs.jsonl"

    def generate_report(self, hours: int = 24) -> PerformanceReport:
        """
        Generate performance report for the specified time range.

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            PerformanceReport with aggregated metrics
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

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

        # Compute handoff metrics and persist new records
        handoff_records = self._compute_handoff_records(events)
        self._persist_handoff_records(handoff_records)
        handoff_summaries = self._aggregate_handoff_summaries(handoff_records)

        report = PerformanceReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            overall_success_rate=success_rate,
            total_tasks=total_tasks,
            total_cost=total_cost,
            agent_performance=agent_performance,
            task_type_metrics=task_type_metrics,
            top_failures=top_failures,
            handoff_summaries=handoff_summaries,
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

    # -- Handoff metrics --

    def _parse_timestamp(self, ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))

    def _compute_handoff_records(self, events: List[Dict[str, Any]]) -> List[HandoffRecord]:
        """Match complete→queued→start triples to build handoff records.

        Groups events by root_task_id (or falls back to chain-ID parsing)
        then correlates chronologically within each group.
        """
        # Index events by type for correlation
        complete_events: Dict[str, Dict[str, Any]] = {}  # task_id → event
        queued_events: Dict[str, Dict[str, Any]] = {}    # task_id → event (keyed by to_task_id)
        start_events: Dict[str, Dict[str, Any]] = {}     # task_id → event

        for event in events:
            etype = event.get('type')
            task_id = event.get('task_id', '')
            if etype == 'complete':
                complete_events[task_id] = event
            elif etype == 'queued':
                queued_events[task_id] = event
            elif etype == 'start':
                start_events[task_id] = event

        records: List[HandoffRecord] = []

        # Queued events are the anchor: each one represents a handoff
        for to_task_id, q_event in queued_events.items():
            source_task_id = q_event.get('source_task_id', '')
            root_task_id = q_event.get('root_task_id', '')

            c_event = complete_events.get(source_task_id)
            s_event = start_events.get(to_task_id)

            completed_at = self._parse_timestamp(c_event['timestamp']) if c_event else None
            queued_at = self._parse_timestamp(q_event['timestamp'])
            started_at = self._parse_timestamp(s_event['timestamp']) if s_event else None

            # Fall back: if no complete event, use queued_at as the anchor
            if completed_at is None:
                completed_at = queued_at

            total_handoff_ms = None
            queue_wait_ms = None
            post_completion_ms = None
            status = "pending"

            if started_at is not None:
                total_handoff_ms = int((started_at - completed_at).total_seconds() * 1000)
                queue_wait_ms = int((started_at - queued_at).total_seconds() * 1000)
                status = "completed"
                if total_handoff_ms > self.DELAYED_THRESHOLD_MS:
                    status = "delayed"

            post_completion_ms = int((queued_at - completed_at).total_seconds() * 1000)

            from_agent = c_event.get('agent', 'unknown') if c_event else 'unknown'
            to_agent = q_event.get('agent', 'unknown')

            records.append(HandoffRecord(
                root_task_id=root_task_id or source_task_id,
                from_agent=from_agent,
                to_agent=to_agent,
                from_task_id=source_task_id,
                to_task_id=to_task_id,
                completed_at=completed_at,
                queued_at=queued_at,
                started_at=started_at,
                total_handoff_ms=total_handoff_ms,
                queue_wait_ms=queue_wait_ms,
                post_completion_ms=post_completion_ms,
                status=status,
            ))

        # Also handle pre-instrumentation data: complete→start without queued
        for task_id, s_event in start_events.items():
            if task_id in queued_events:
                continue  # already handled above
            root_task_id = s_event.get('root_task_id', '')
            if not root_task_id:
                continue
            # Find a complete event in the same root chain
            source_complete = None
            for c_id, c_event in complete_events.items():
                if c_event.get('root_task_id') == root_task_id and c_id != task_id:
                    ts = self._parse_timestamp(c_event['timestamp'])
                    s_ts = self._parse_timestamp(s_event['timestamp'])
                    if ts <= s_ts:
                        if source_complete is None or ts > self._parse_timestamp(source_complete['timestamp']):
                            source_complete = c_event
            if source_complete is None:
                continue

            completed_at = self._parse_timestamp(source_complete['timestamp'])
            started_at = self._parse_timestamp(s_event['timestamp'])
            total_handoff_ms = int((started_at - completed_at).total_seconds() * 1000)
            status = "delayed" if total_handoff_ms > self.DELAYED_THRESHOLD_MS else "completed"

            records.append(HandoffRecord(
                root_task_id=root_task_id,
                from_agent=source_complete.get('agent', 'unknown'),
                to_agent=s_event.get('agent', 'unknown'),
                from_task_id=source_complete.get('task_id', ''),
                to_task_id=task_id,
                completed_at=completed_at,
                queued_at=None,
                started_at=started_at,
                total_handoff_ms=total_handoff_ms,
                queue_wait_ms=None,
                post_completion_ms=None,
                status=status,
            ))

        return records

    def _aggregate_handoff_summaries(
        self, records: List[HandoffRecord],
    ) -> List[HandoffSummary]:
        """Aggregate handoff records by transition type (from_agent→to_agent)."""
        by_transition: Dict[str, List[HandoffRecord]] = defaultdict(list)
        for r in records:
            key = f"{r.from_agent}\u2192{r.to_agent}"
            by_transition[key].append(r)

        summaries: List[HandoffSummary] = []
        for transition, group in sorted(by_transition.items()):
            totals = sorted([
                r.total_handoff_ms for r in group if r.total_handoff_ms is not None
            ])
            waits = [r.queue_wait_ms for r in group if r.queue_wait_ms is not None]

            count = len(group)
            avg_total = sum(totals) / len(totals) if totals else 0.0
            p50 = totals[len(totals) // 2] if totals else 0
            p90 = totals[int(len(totals) * 0.9)] if totals else 0
            avg_wait = sum(waits) / len(waits) if waits else 0.0
            failed = sum(1 for r in group if r.status == "pending")
            delayed = sum(1 for r in group if r.status == "delayed")

            summaries.append(HandoffSummary(
                transition=transition,
                count=count,
                avg_total_ms=avg_total,
                p50_total_ms=p50,
                p90_total_ms=p90,
                avg_queue_wait_ms=avg_wait,
                failed_count=failed,
                delayed_count=delayed,
            ))

        return summaries

    def _persist_handoff_records(self, records: List[HandoffRecord]) -> None:
        """Append new handoff records to the persistent JSONL log."""
        if not records:
            return
        existing_ids = set()
        if self.handoff_log_file.exists():
            for line in self.handoff_log_file.read_text().strip().split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    existing_ids.add((data.get('from_task_id'), data.get('to_task_id')))
                except (json.JSONDecodeError, KeyError):
                    continue

        new_lines = []
        for r in records:
            if (r.from_task_id, r.to_task_id) not in existing_ids:
                new_lines.append(r.model_dump_json())

        if new_lines:
            with open(self.handoff_log_file, 'a') as f:
                f.write('\n'.join(new_lines) + '\n')

    def read_handoff_log(self) -> List[HandoffRecord]:
        """Read all records from the persistent handoff log."""
        if not self.handoff_log_file.exists():
            return []
        records = []
        for line in self.handoff_log_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                records.append(HandoffRecord.model_validate_json(line))
            except Exception:
                continue
        return records

    def generate_handoff_report(self) -> HandoffReport:
        """Generate a standalone handoff report from the persistent log."""
        records = self.read_handoff_log()
        summaries = self._aggregate_handoff_summaries(records)
        pending = [r for r in records if r.status == "pending"]
        return HandoffReport(
            generated_at=datetime.now(timezone.utc),
            records=records,
            summaries=summaries,
            pending_handoffs=pending,
        )
