"""
Git-specific behavioral metrics from chain state files and session logs.

Surfaces four metrics the observation reports couldn't answer:
1. Commits per task — aggregate commit count across chain steps
2. Lines changed per commit — insertions/deletions per step
3. Push success/failure rate — from session log git_push events
4. Time from first edit to first commit — derived from tool_call timestamps
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..core.chain_state import ChainState, load_chain_state
from .session_loader import load_session_events

logger = logging.getLogger(__name__)


# --- Pydantic report models ---


class TaskGitMetrics(BaseModel):
    """Per-task git metrics derived from chain state + session logs."""
    root_task_id: str
    total_commits: int
    total_insertions: int
    total_deletions: int
    push_attempts: int
    push_successes: int
    first_edit_to_commit_secs: Optional[float] = None


class GitMetricsSummary(BaseModel):
    """Aggregate git metrics across all observed tasks."""
    avg_commits_per_task: float
    avg_insertions_per_commit: float
    avg_deletions_per_commit: float
    push_success_rate: float
    p50_edit_to_commit_secs: Optional[float] = None
    p90_edit_to_commit_secs: Optional[float] = None


class GitMetricsReport(BaseModel):
    """Complete git metrics report."""
    generated_at: datetime
    time_range_hours: int
    total_tasks: int
    per_task: List[TaskGitMetrics]
    summary: GitMetricsSummary


# --- Collector ---


class GitMetrics:
    """Aggregates git behavioral metrics from chain state files and session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.chain_state_dir = self.workspace / ".agent-communication" / "chain-state"
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> GitMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        chain_states = self._load_chain_states(cutoff)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        # Aggregate push events across all sessions
        push_totals = self._aggregate_push_events(events_by_task)

        per_task: List[TaskGitMetrics] = []
        for state in chain_states:
            task_metrics = self._compute_task_metrics(state, events_by_task, push_totals)
            per_task.append(task_metrics)

        summary = self._compute_summary(per_task, push_totals)

        return GitMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_tasks=len(per_task),
            per_task=per_task,
            summary=summary,
        )

    def _load_chain_states(self, cutoff: datetime) -> List[ChainState]:
        if not self.chain_state_dir.exists():
            return []

        states = []
        cutoff_ts = cutoff.timestamp()
        for path in self.chain_state_dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff_ts:
                    continue
            except OSError:
                continue

            state = load_chain_state(self.workspace, path.stem)
            if state is not None:
                states.append(state)

        return states

    def _aggregate_push_events(
        self, events_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate git_push events by task_id → {attempts, successes}."""
        totals: Dict[str, Dict[str, int]] = {}
        for task_id, events in events_by_task.items():
            attempts = 0
            successes = 0
            for e in events:
                if e.get("event") != "git_push":
                    continue
                attempts += 1
                if e.get("success"):
                    successes += 1
            if attempts > 0:
                totals[task_id] = {"attempts": attempts, "successes": successes}
        return totals

    def _compute_task_metrics(
        self,
        state: ChainState,
        events_by_task: Dict[str, List[Dict[str, Any]]],
        push_totals: Dict[str, Dict[str, int]],
    ) -> TaskGitMetrics:
        total_commits = 0
        total_insertions = 0
        total_deletions = 0

        for step in state.steps:
            total_commits += len(step.commit_shas)
            total_insertions += step.lines_added
            total_deletions += step.lines_removed

        # Push events — check both root task id and individual step task ids
        push_attempts = 0
        push_successes = 0
        task_ids = {state.root_task_id} | {s.task_id for s in state.steps}
        for tid in task_ids:
            if tid in push_totals:
                push_attempts += push_totals[tid]["attempts"]
                push_successes += push_totals[tid]["successes"]

        # Edit-to-commit timing from session events
        edit_to_commit = self._compute_edit_to_commit_seconds(
            state, events_by_task
        )

        return TaskGitMetrics(
            root_task_id=state.root_task_id,
            total_commits=total_commits,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            push_attempts=push_attempts,
            push_successes=push_successes,
            first_edit_to_commit_secs=edit_to_commit,
        )

    def _compute_edit_to_commit_seconds(
        self,
        state: ChainState,
        events_by_task: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[float]:
        """Derive first-edit to first-commit latency from session tool_call events.

        Scans all task sessions associated with this chain for:
        - First Write or Edit tool call → first edit timestamp
        - First Bash call containing 'git commit' → first commit timestamp
        """
        task_ids = {state.root_task_id} | {s.task_id for s in state.steps}

        first_edit_ts: Optional[datetime] = None
        first_commit_ts: Optional[datetime] = None

        for tid in task_ids:
            events = events_by_task.get(tid, [])
            for e in events:
                if e.get("event") != "tool_call":
                    continue
                ts_str = e.get("ts")
                if not ts_str:
                    continue

                tool = e.get("tool", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

                # Detect edit tools
                if tool in ("Write", "Edit"):
                    if first_edit_ts is None or ts < first_edit_ts:
                        first_edit_ts = ts

                # Detect git commit in Bash calls
                if tool == "Bash":
                    cmd = (e.get("input") or {}).get("command", "")
                    if "git commit" in cmd:
                        if first_commit_ts is None or ts < first_commit_ts:
                            first_commit_ts = ts

        if first_edit_ts is None or first_commit_ts is None:
            return None

        delta = (first_commit_ts - first_edit_ts).total_seconds()
        # Commit before edit shouldn't happen, but guard against clock skew
        return max(delta, 0.0)

    def _compute_summary(
        self,
        per_task: List[TaskGitMetrics],
        push_totals: Dict[str, Dict[str, int]],
    ) -> GitMetricsSummary:
        n = len(per_task)

        # Commits per task
        total_commits = sum(t.total_commits for t in per_task)
        avg_commits = round(total_commits / n, 1) if n > 0 else 0.0

        # Lines per commit
        total_ins = sum(t.total_insertions for t in per_task)
        total_del = sum(t.total_deletions for t in per_task)
        avg_ins = round(total_ins / total_commits, 1) if total_commits > 0 else 0.0
        avg_del = round(total_del / total_commits, 1) if total_commits > 0 else 0.0

        # Push success rate across ALL sessions (not just chain-state tasks)
        all_attempts = sum(v["attempts"] for v in push_totals.values())
        all_successes = sum(v["successes"] for v in push_totals.values())
        push_rate = round(all_successes / all_attempts, 3) if all_attempts > 0 else 0.0

        # Edit-to-commit latency percentiles
        latencies = [
            t.first_edit_to_commit_secs
            for t in per_task
            if t.first_edit_to_commit_secs is not None
        ]
        p50 = None
        p90 = None
        if latencies:
            sorted_lat = sorted(latencies)
            p50 = round(median(sorted_lat), 1)
            if len(sorted_lat) >= 2:
                p90 = round(quantiles(sorted_lat, n=10)[8], 1)
            else:
                p90 = round(sorted_lat[-1], 1)

        return GitMetricsSummary(
            avg_commits_per_task=avg_commits,
            avg_insertions_per_commit=avg_ins,
            avg_deletions_per_commit=avg_del,
            push_success_rate=push_rate,
            p50_edit_to_commit_secs=p50,
            p90_edit_to_commit_secs=p90,
        )
