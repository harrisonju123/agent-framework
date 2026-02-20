"""Task decomposition metrics collected from session JSONL logs and chain state files.

Surfaces 4 key signals about the decomposition subsystem:
- Decomposition rate: how often architect tasks trigger decomposition
- Subtask distribution: how many subtasks per decomposed task
- Estimation accuracy: architect estimated lines vs actual lines written
- Fan-in reliability: fraction of decomposed tasks that complete fan-in
"""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


# --- Pydantic report models ---


class DecompositionRateMetrics(BaseModel):
    """How often architect tasks trigger decomposition."""
    tasks_evaluated: int
    tasks_decomposed: int
    decomposition_rate: float


class SubtaskDistribution(BaseModel):
    """Distribution of subtask counts across decomposed tasks."""
    distribution: Dict[int, int]  # {2: 5, 3: 3} = 5 tasks had 2 subtasks
    avg_subtask_count: float
    min_subtask_count: int
    max_subtask_count: int


class EstimationSample(BaseModel):
    """Single data point for estimation accuracy."""
    task_id: str
    estimated_lines: int
    actual_lines: int
    ratio: float  # actual / estimated; <1 = overestimate, >1 = underestimate


class EstimationAccuracy(BaseModel):
    """How well architect line estimates match actual implementation."""
    sample_count: int
    avg_estimated: float
    avg_actual: float
    avg_ratio: float
    samples: List[EstimationSample]


class FanInMetrics(BaseModel):
    """Reliability of the fan-in aggregation step."""
    decomposed_tasks: int
    fan_ins_created: int
    fan_in_success_rate: float


class DecompositionReport(BaseModel):
    """Complete decomposition observability report."""
    generated_at: datetime
    time_range_hours: int
    rate: DecompositionRateMetrics
    distribution: SubtaskDistribution
    estimation: EstimationAccuracy
    fan_in: FanInMetrics


# --- Collector ---


class DecompositionMetrics:
    """Aggregates decomposition metrics from session JSONL logs and chain state.

    Session events consumed:
    - decomposition_evaluated: emitted by should_decompose_task()
    - task_decomposed: emitted by decompose_and_queue_subtasks()
    - fan_in_created: emitted by check_and_create_fan_in_task()

    Chain state files provide actual line counts for estimation accuracy.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"
        self.chain_state_dir = self.workspace / ".agent-communication" / "chain-state"

    def generate_report(self, hours: int = 24) -> DecompositionReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        # Flatten all events for easier filtering
        all_events = []
        for events in events_by_task.values():
            all_events.extend(events)

        rate = self._aggregate_rate(all_events)
        distribution = self._aggregate_distribution(all_events)
        estimation = self._aggregate_estimation(all_events)
        fan_in = self._aggregate_fan_in(all_events)

        return DecompositionReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            rate=rate,
            distribution=distribution,
            estimation=estimation,
            fan_in=fan_in,
        )

    def _aggregate_rate(self, events: List[Dict[str, Any]]) -> DecompositionRateMetrics:
        """Count decomposition_evaluated events, split by should_decompose."""
        eval_events = [e for e in events if e.get("event") == "decomposition_evaluated"]

        # Deduplicate by task_id — take the last evaluation per task
        by_task: Dict[str, Dict[str, Any]] = {}
        for e in eval_events:
            tid = e.get("task_id", "unknown")
            by_task[tid] = e

        tasks_evaluated = len(by_task)
        tasks_decomposed = sum(
            1 for e in by_task.values() if e.get("should_decompose")
        )

        return DecompositionRateMetrics(
            tasks_evaluated=tasks_evaluated,
            tasks_decomposed=tasks_decomposed,
            decomposition_rate=round(tasks_decomposed / tasks_evaluated, 3) if tasks_evaluated > 0 else 0.0,
        )

    def _aggregate_distribution(self, events: List[Dict[str, Any]]) -> SubtaskDistribution:
        """From task_decomposed events, build Counter of subtask_count."""
        decomposed = [e for e in events if e.get("event") == "task_decomposed"]

        # Deduplicate by task_id
        by_task: Dict[str, Dict[str, Any]] = {}
        for e in decomposed:
            tid = e.get("task_id", "unknown")
            by_task[tid] = e

        counts = [e.get("subtask_count", 0) for e in by_task.values()]

        if not counts:
            return SubtaskDistribution(
                distribution={},
                avg_subtask_count=0.0,
                min_subtask_count=0,
                max_subtask_count=0,
            )

        dist = dict(Counter(counts))

        return SubtaskDistribution(
            distribution=dist,
            avg_subtask_count=round(sum(counts) / len(counts), 1),
            min_subtask_count=min(counts),
            max_subtask_count=max(counts),
        )

    def _aggregate_estimation(self, events: List[Dict[str, Any]]) -> EstimationAccuracy:
        """Cross-reference estimated lines from events with actual from chain state."""
        decomposed = [e for e in events if e.get("event") == "task_decomposed"]

        # Deduplicate by task_id
        by_task: Dict[str, Dict[str, Any]] = {}
        for e in decomposed:
            tid = e.get("task_id", "unknown")
            by_task[tid] = e

        samples: List[EstimationSample] = []
        for task_id, event in by_task.items():
            estimated = event.get("estimated_lines", 0)
            if estimated <= 0:
                continue

            subtask_ids = event.get("subtask_ids", [])
            actual = self._compute_actual_lines(subtask_ids)
            if actual <= 0:
                continue

            ratio = round(actual / estimated, 3)
            samples.append(EstimationSample(
                task_id=task_id,
                estimated_lines=estimated,
                actual_lines=actual,
                ratio=ratio,
            ))

        if not samples:
            return EstimationAccuracy(
                sample_count=0,
                avg_estimated=0.0,
                avg_actual=0.0,
                avg_ratio=0.0,
                samples=[],
            )

        return EstimationAccuracy(
            sample_count=len(samples),
            avg_estimated=round(sum(s.estimated_lines for s in samples) / len(samples), 1),
            avg_actual=round(sum(s.actual_lines for s in samples) / len(samples), 1),
            avg_ratio=round(sum(s.ratio for s in samples) / len(samples), 3),
            samples=samples,
        )

    def _aggregate_fan_in(self, events: List[Dict[str, Any]]) -> FanInMetrics:
        """Cross-reference task_decomposed parent IDs with fan_in_created parent IDs."""
        decomposed_parents = set()
        for e in events:
            if e.get("event") == "task_decomposed":
                decomposed_parents.add(e.get("task_id", "unknown"))

        fan_in_parents = set()
        for e in events:
            if e.get("event") == "fan_in_created":
                fan_in_parents.add(e.get("parent_task_id", "unknown"))

        decomposed_count = len(decomposed_parents)
        fan_in_count = len(decomposed_parents & fan_in_parents)

        return FanInMetrics(
            decomposed_tasks=decomposed_count,
            fan_ins_created=fan_in_count,
            fan_in_success_rate=round(fan_in_count / decomposed_count, 3) if decomposed_count > 0 else 0.0,
        )

    def _compute_actual_lines(self, subtask_ids: List[str]) -> int:
        """Load chain state for each subtask, sum lines_added from implement steps."""
        if not self.chain_state_dir.exists():
            return 0

        total = 0
        for subtask_id in subtask_ids:
            path = self.chain_state_dir / f"{subtask_id}.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                steps = data.get("steps", [])
                if not isinstance(steps, list):
                    continue
                for step in steps:
                    if isinstance(step, dict) and step.get("step_id") == "implement":
                        total += step.get("lines_added", 0)
            except (json.JSONDecodeError, OSError):
                continue

        return total
