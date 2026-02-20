"""
Workflow chain metrics aggregation from chain state files.

Analyzes workflow chain behavior including step success rates,
inter-step durations, chain completion rates, and retry patterns.
Data source: .agent-communication/chain-state/{root_task_id}.json
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from ..core.chain_state import ChainState, StepRecord, load_chain_state

logger = logging.getLogger(__name__)


class StepTypeMetrics(BaseModel):
    """Aggregated metrics for a single workflow step type (e.g. 'implement')."""
    step_id: str
    total_count: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_duration_seconds: float
    p50_duration_seconds: float
    p90_duration_seconds: float


class ChainSummary(BaseModel):
    """Summary metrics for a single workflow chain."""
    root_task_id: str
    workflow: str
    step_count: int
    attempt: int
    completed: bool
    files_modified_count: int
    total_duration_seconds: float


class ChainMetricsReport(BaseModel):
    """Complete workflow chain metrics report."""
    generated_at: datetime
    time_range_hours: int
    total_chains: int
    completed_chains: int
    chain_completion_rate: float
    avg_chain_depth: float
    avg_files_modified: float
    avg_attempts: float
    step_type_metrics: List[StepTypeMetrics]
    top_failing_steps: List[StepTypeMetrics]
    recent_chains: List[ChainSummary]


class ChainMetrics:
    """Aggregates workflow chain metrics from chain state files."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.chain_state_dir = self.workspace / ".agent-communication" / "chain-state"

    def generate_report(self, hours: int = 24) -> ChainMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        states = self._load_chain_states(cutoff)

        step_type_metrics = self._compute_step_type_metrics(states)
        chain_summaries = self._compute_chain_summaries(states)

        total = len(states)
        completed = sum(1 for s in chain_summaries if s.completed)
        completion_rate = (completed / total * 100) if total > 0 else 0.0

        avg_depth = (
            sum(s.step_count for s in chain_summaries) / total
            if total > 0 else 0.0
        )
        avg_files = (
            sum(s.files_modified_count for s in chain_summaries) / total
            if total > 0 else 0.0
        )
        avg_attempts = (
            sum(s.attempt for s in chain_summaries) / total
            if total > 0 else 0.0
        )

        top_failing = sorted(
            [m for m in step_type_metrics if m.failure_count > 0],
            key=lambda m: m.failure_count / m.total_count,
            reverse=True,
        )

        # Recent chains: last 10 by root_task_id (stable ordering)
        recent = sorted(chain_summaries, key=lambda s: s.root_task_id, reverse=True)[:10]

        return ChainMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_chains=total,
            completed_chains=completed,
            chain_completion_rate=completion_rate,
            avg_chain_depth=avg_depth,
            avg_files_modified=avg_files,
            avg_attempts=avg_attempts,
            step_type_metrics=step_type_metrics,
            top_failing_steps=top_failing,
            recent_chains=recent,
        )

    def _load_chain_states(self, cutoff: datetime) -> List[ChainState]:
        """Load chain states from disk, filtering by file mtime."""
        if not self.chain_state_dir.exists():
            return []

        states = []
        for path in self.chain_state_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    continue
            except OSError:
                continue

            # Extract root_task_id from filename (strip .json)
            root_task_id = path.stem
            state = load_chain_state(self.workspace, root_task_id)
            if state is not None:
                states.append(state)

        return states

    def _compute_step_type_metrics(self, states: List[ChainState]) -> List[StepTypeMetrics]:
        """Aggregate metrics by step_id across all chains."""
        step_data: Dict[str, List[StepRecord]] = {}
        # Also collect durations keyed by step_id
        step_durations: Dict[str, List[float]] = {}

        for state in states:
            for i, step in enumerate(state.steps):
                step_data.setdefault(step.step_id, []).append(step)

                duration = self._compute_step_duration(state.steps, i)
                if duration is not None:
                    step_durations.setdefault(step.step_id, []).append(duration)

        metrics = []
        for step_id, records in sorted(step_data.items()):
            total = len(records)
            failures = sum(1 for r in records if self._is_step_failure(r))
            successes = total - failures
            success_rate = (successes / total * 100) if total > 0 else 0.0

            durations = sorted(step_durations.get(step_id, []))
            avg_dur = sum(durations) / len(durations) if durations else 0.0
            p50_dur = durations[len(durations) // 2] if durations else 0.0
            p90_dur = durations[int(len(durations) * 0.9)] if durations else 0.0

            metrics.append(StepTypeMetrics(
                step_id=step_id,
                total_count=total,
                success_count=successes,
                failure_count=failures,
                success_rate=success_rate,
                avg_duration_seconds=avg_dur,
                p50_duration_seconds=p50_dur,
                p90_duration_seconds=p90_dur,
            ))

        return metrics

    def _compute_chain_summaries(self, states: List[ChainState]) -> List[ChainSummary]:
        summaries = []
        for state in states:
            all_files = set()
            for step in state.steps:
                all_files.update(step.files_modified)

            completed = self._is_chain_completed(state)
            total_duration = self._compute_chain_duration(state)

            summaries.append(ChainSummary(
                root_task_id=state.root_task_id,
                workflow=state.workflow,
                step_count=len(state.steps),
                attempt=state.attempt,
                completed=completed,
                files_modified_count=len(all_files),
                total_duration_seconds=total_duration,
            ))

        return summaries

    def _compute_step_duration(self, steps: List[StepRecord], index: int) -> Optional[float]:
        """Compute time between this step and the previous step's completion.

        Returns None for the first step (no prior reference point).
        """
        if index == 0:
            return None

        try:
            current_ts = datetime.fromisoformat(steps[index].completed_at)
            prev_ts = datetime.fromisoformat(steps[index - 1].completed_at)
            delta = (current_ts - prev_ts).total_seconds()
            return max(delta, 0.0)
        except (ValueError, TypeError):
            return None

    def _compute_chain_duration(self, state: ChainState) -> float:
        """Total wall time from first step completion to last step completion."""
        if len(state.steps) < 2:
            return 0.0

        try:
            first_ts = datetime.fromisoformat(state.steps[0].completed_at)
            last_ts = datetime.fromisoformat(state.steps[-1].completed_at)
            return max((last_ts - first_ts).total_seconds(), 0.0)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _is_step_failure(record: StepRecord) -> bool:
        if record.error is not None:
            return True
        if record.verdict == "needs_fix":
            return True
        return False

    @staticmethod
    def _is_chain_completed(state: ChainState) -> bool:
        """A chain is completed if it has a create_pr step or all review steps approved."""
        for step in state.steps:
            if step.step_id == "create_pr":
                return True

        # Avoid false-positives on in-progress chains: require no active step
        # and at least one explicit approval from a review step
        if state.current_step is not None:
            return False

        has_approval = any(step.verdict == "approved" for step in state.steps)
        no_failures = all(
            step.verdict in ("approved", None) and step.error is None
            for step in state.steps
        )
        return has_approval and no_failures
