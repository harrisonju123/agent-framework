"""
Review cycle enforcement metrics from session log events.

Aggregates `review_cycle_check` events emitted by WorkflowExecutor._route_to_step()
to surface enforcement rates, cap violations, and phase reset patterns.
Data source: logs/sessions/{task_id}.jsonl
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


class StepBreakdown(BaseModel):
    """Per-step review cycle counts."""
    workflow_step: str
    checks: int
    enforcements: int
    phase_resets: int


class ReviewCycleMetrics(BaseModel):
    """Aggregated review cycle enforcement metrics."""
    total_checks: int
    total_enforcements: int
    total_phase_resets: int
    total_halts: int
    enforcement_rate: float
    cap_violations: int
    violation_task_ids: List[str]
    enforcement_count_distribution: Dict[int, int]
    by_step: List[StepBreakdown]


class ReviewCycleMetricsReport(BaseModel):
    """Complete review cycle metrics report."""
    generated_at: datetime
    time_range_hours: int
    metrics: ReviewCycleMetrics
    raw_events: List[Dict[str, Any]]


class ReviewCycleAnalyzer:
    """Aggregates review cycle enforcement metrics from session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> ReviewCycleMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        raw_events = self._extract_review_cycle_events(events_by_task)
        metrics = self._aggregate(raw_events)

        return ReviewCycleMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            metrics=metrics,
            raw_events=raw_events,
        )

    def _extract_review_cycle_events(
        self, events_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Filter to only review_cycle_check events across all tasks."""
        result = []
        for events in events_by_task.values():
            for e in events:
                if e.get("event") == "review_cycle_check":
                    result.append(e)
        return result

    def _aggregate(self, events: List[Dict[str, Any]]) -> ReviewCycleMetrics:
        total_checks = len(events)
        total_enforcements = sum(1 for e in events if e.get("enforced"))
        total_phase_resets = sum(1 for e in events if e.get("phase_reset"))
        total_halts = sum(1 for e in events if e.get("halted"))

        # Cap violation: count_after >= max but neither enforced nor phase_reset
        # This is the bug signal — should always be zero in healthy operation
        violation_seen: set[str] = set()
        violation_task_ids: list[str] = []
        for e in events:
            count_after = e.get("count_after", 0)
            cap = e.get("max", 0)
            if (count_after >= cap
                    and not e.get("enforced")
                    and not e.get("phase_reset")
                    and not e.get("halted")):
                tid = e.get("task_id", "unknown")
                if tid not in violation_seen:
                    violation_seen.add(tid)
                    violation_task_ids.append(tid)

        enforcement_rate = (
            (total_enforcements / total_checks * 100)
            if total_checks > 0 else 0.0
        )

        # Distribution: how many enforcements happened at each count_after value
        enforcement_dist: Dict[int, int] = defaultdict(int)
        for e in events:
            if e.get("enforced"):
                enforcement_dist[e.get("count_after", 0)] += 1

        # Per-step breakdown
        step_data: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"checks": 0, "enforcements": 0, "phase_resets": 0}
        )
        for e in events:
            step = e.get("workflow_step", "unknown")
            step_data[step]["checks"] += 1
            if e.get("enforced"):
                step_data[step]["enforcements"] += 1
            if e.get("phase_reset"):
                step_data[step]["phase_resets"] += 1

        by_step = [
            StepBreakdown(workflow_step=step, **counts)
            for step, counts in sorted(step_data.items())
        ]

        return ReviewCycleMetrics(
            total_checks=total_checks,
            total_enforcements=total_enforcements,
            total_phase_resets=total_phase_resets,
            total_halts=total_halts,
            enforcement_rate=enforcement_rate,
            cap_violations=len(violation_task_ids),
            violation_task_ids=violation_task_ids,
            enforcement_count_distribution=dict(enforcement_dist),
            by_step=by_step,
        )
