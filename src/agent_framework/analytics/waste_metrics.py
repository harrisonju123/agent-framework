"""
Token waste ratio metric — per-root-task cost efficiency.

Correlates activity stream events with session logs to determine
how much spend produced deliverables (PRs) vs. was lost to failures.

Formula: waste_ratio = wasted_cost / total_cost (0.0 = fully productive, 1.0 = all waste)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


# --- Pydantic report models ---


class RootTaskWaste(BaseModel):
    """Waste breakdown for a single root task."""
    root_task_id: str
    total_cost: float
    wasted_cost: float
    waste_ratio: float
    productive_cost: float
    total_tasks: int
    failed_tasks: int
    completed_tasks: int
    has_pr: bool
    title: str


class WasteMetricsReport(BaseModel):
    """Token waste ratio report across all root tasks."""
    generated_at: datetime
    time_range_hours: int
    roots_analyzed: int
    total_cost: float
    total_wasted_cost: float
    aggregate_waste_ratio: float
    avg_waste_ratio: float
    max_waste_ratio: float
    roots_with_zero_delivery: int
    top_waste_roots: List[RootTaskWaste]


# --- Collector ---


class WasteMetrics:
    """
    Aggregates token waste ratio from activity stream and session logs.

    Reads activity stream to group tasks by root_task_id, then
    cross-references session logs for cost data on failed tasks
    (which don't carry cost in the activity stream).
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> WasteMetricsReport:
        """Generate a waste metrics report for the given lookback window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events = self._read_events(cutoff)
        roots = self._group_by_root(events)

        # Load session logs for failed tasks that lack cost in the activity stream
        session_costs = self._load_session_costs(cutoff)

        root_wastes: List[RootTaskWaste] = []
        for root_id, task_events in roots.items():
            root_wastes.append(
                self._compute_root_waste(root_id, task_events, session_costs)
            )

        # Aggregate
        total_cost = sum(r.total_cost for r in root_wastes)
        total_wasted = sum(r.wasted_cost for r in root_wastes)
        ratios = [r.waste_ratio for r in root_wastes]

        aggregate_ratio = total_wasted / total_cost if total_cost > 0 else 0.0
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        max_ratio = max(ratios) if ratios else 0.0
        zero_delivery = sum(1 for r in root_wastes if not r.has_pr)

        # Top waste roots: sorted by wasted_cost desc, capped at 10
        top_roots = sorted(root_wastes, key=lambda r: r.wasted_cost, reverse=True)[:10]

        return WasteMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            roots_analyzed=len(root_wastes),
            total_cost=round(total_cost, 6),
            total_wasted_cost=round(total_wasted, 6),
            aggregate_waste_ratio=round(aggregate_ratio, 6),
            avg_waste_ratio=round(avg_ratio, 6),
            max_waste_ratio=round(max_ratio, 6),
            roots_with_zero_delivery=zero_delivery,
            top_waste_roots=top_roots,
        )

    # --- Private helpers ---

    def _read_events(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """Read and filter events from the activity stream."""
        if not self.stream_file.exists():
            return []

        events = []
        for line in self.stream_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                timestamp = datetime.fromisoformat(
                    event["timestamp"].replace("Z", "+00:00")
                )
                if timestamp >= cutoff:
                    events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        return events

    def _group_by_root(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group activity events by root_task_id.

        For tasks without root_task_id, fall back to task_id as root
        (standalone tasks).
        """
        roots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            root_id = event.get("root_task_id") or event.get("task_id")
            if root_id:
                roots[root_id].append(event)
        return roots

    def _load_session_costs(self, cutoff: datetime) -> Dict[str, float]:
        """Load per-task cost from session logs (for failed tasks missing cost)."""
        events_by_task = load_session_events(self.sessions_dir, cutoff)
        costs: Dict[str, float] = {}
        for task_id, events in events_by_task.items():
            total = sum(
                e.get("cost") or 0.0
                for e in events
                if e.get("event") == "llm_complete"
            )
            if total > 0:
                costs[task_id] = total
        return costs

    def _compute_root_waste(
        self,
        root_id: str,
        events: List[Dict[str, Any]],
        session_costs: Dict[str, float],
    ) -> RootTaskWaste:
        """Compute waste metrics for a single root task."""
        # Track per-task state within this root
        task_costs: Dict[str, float] = {}
        task_status: Dict[str, str] = {}  # task_id → "completed" | "failed"
        has_pr = False
        title = ""

        for event in events:
            task_id = event.get("task_id", "")
            event_type = event.get("type")

            if event_type == "start" and not title:
                title = event.get("title", "")

            elif event_type == "complete":
                task_status[task_id] = "completed"
                task_costs[task_id] = event.get("cost") or 0.0
                if event.get("pr_url"):
                    has_pr = True

            elif event_type == "fail":
                task_status[task_id] = "failed"
                # Fail events typically lack cost — pull from session logs
                task_costs[task_id] = session_costs.get(task_id, 0.0)

        completed = sum(1 for s in task_status.values() if s == "completed")
        failed = sum(1 for s in task_status.values() if s == "failed")
        total_cost = sum(task_costs.values())

        if not has_pr:
            # No deliverable — all cost is waste
            wasted_cost = total_cost
        else:
            # Only failed task costs are waste
            wasted_cost = sum(
                cost for tid, cost in task_costs.items()
                if task_status.get(tid) == "failed"
            )

        waste_ratio = wasted_cost / total_cost if total_cost > 0 else 0.0

        return RootTaskWaste(
            root_task_id=root_id,
            total_cost=round(total_cost, 6),
            wasted_cost=round(wasted_cost, 6),
            waste_ratio=round(waste_ratio, 6),
            productive_cost=round(total_cost - wasted_cost, 6),
            total_tasks=len(task_status),
            failed_tasks=failed,
            completed_tasks=completed,
            has_pr=has_pr,
            title=title,
        )
