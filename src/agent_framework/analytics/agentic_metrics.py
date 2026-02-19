"""Agentic feature metrics aggregator for observability dashboard.

Reads from existing data stores — memory files, session logs, activity stream,
and activity state files — to surface metrics about features that aren't
exposed by the main dashboard: memory recall, self-eval, replan, specialization,
and context budget utilization.

No new instrumentation is added here; this is purely a read-side aggregation layer.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    total_entries: int = 0
    accessed_entries: int = 0  # access_count >= 1
    # Fraction of stored memories that have been recalled at least once.
    # Low values mean agents are writing memories they never use.
    hit_rate: float = 0.0


@dataclass
class SelfEvalStats:
    total_evaluations: int = 0
    # Self-eval retries: evaluations where verdict was FAIL (agent rewrote output)
    failed_evaluations: int = 0
    # Fraction of evaluations that caught issues before QA saw them
    retry_rate: float = 0.0


@dataclass
class ReplanStats:
    # Distinct sessions (task IDs) that triggered at least one replan
    sessions_with_replan: int = 0
    total_sessions: int = 0
    trigger_rate: float = 0.0


@dataclass
class ContextBudgetStats:
    # Complete events with token data in the activity stream
    total_tasks_with_tokens: int = 0
    # Events where tokens exceeded budget threshold
    budget_exceeded_events: int = 0
    # Average utilization pct across tasks that completed with token data.
    # None when no such tasks exist.
    avg_utilization_pct: Optional[float] = None


@dataclass
class AgenticMetrics:
    memory: MemoryStats = field(default_factory=MemoryStats)
    self_eval: SelfEvalStats = field(default_factory=SelfEvalStats)
    replan: ReplanStats = field(default_factory=ReplanStats)
    # Specialization profile -> count of agents currently using it.
    # "none" represents agents without a profile assigned.
    specialization_distribution: Dict[str, int] = field(default_factory=dict)
    context_budget: ContextBudgetStats = field(default_factory=ContextBudgetStats)


class AgenticMetricsAggregator:
    """Aggregates agentic feature metrics from on-disk data.

    All reads are best-effort: any file that can't be parsed is silently skipped.
    """

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._memory_base = self._workspace / ".agent-communication" / "memory"
        self._activity_dir = self._workspace / ".agent-communication" / "activity"
        self._activity_stream = self._workspace / ".agent-communication" / "activity-stream.jsonl"
        self._sessions_dir = self._workspace / "logs" / "sessions"

    def get_all_metrics(self) -> AgenticMetrics:
        return AgenticMetrics(
            memory=self._compute_memory_stats(),
            self_eval=self._compute_self_eval_stats(),
            replan=self._compute_replan_stats(),
            specialization_distribution=self._compute_specialization_distribution(),
            context_budget=self._compute_context_budget_stats(),
        )

    # -- Memory --

    def _compute_memory_stats(self) -> MemoryStats:
        """Walk all memory JSON files, tally total and accessed entries."""
        total = 0
        accessed = 0

        if not self._memory_base.exists():
            return MemoryStats()

        for json_file in self._memory_base.rglob("*.json"):
            entries = self._load_json_list(json_file)
            for entry in entries:
                total += 1
                if entry.get("access_count", 0) >= 1:
                    accessed += 1

        hit_rate = (accessed / total) if total > 0 else 0.0
        return MemoryStats(
            total_entries=total,
            accessed_entries=accessed,
            hit_rate=round(hit_rate, 3),
        )

    # -- Self-eval --

    def _compute_self_eval_stats(self) -> SelfEvalStats:
        """Count self_eval events across all session logs."""
        total = 0
        failed = 0

        for event in self._iter_session_events("self_eval"):
            total += 1
            verdict = event.get("verdict", "")
            if verdict.upper() == "FAIL":
                failed += 1

        retry_rate = (failed / total) if total > 0 else 0.0
        return SelfEvalStats(
            total_evaluations=total,
            failed_evaluations=failed,
            retry_rate=round(retry_rate, 3),
        )

    # -- Replan --

    def _compute_replan_stats(self) -> ReplanStats:
        """Count distinct sessions that triggered a replan."""
        sessions_with_replan: set = set()
        all_sessions: set = set()

        for json_file in self._iter_session_files():
            task_id = json_file.stem
            has_replan = False
            has_events = False

            for event in self._parse_jsonl(json_file):
                event_type = event.get("event", "")
                if event_type in ("task_complete", "self_eval", "replan", "tool_call"):
                    has_events = True
                if event_type == "replan":
                    has_replan = True

            if has_events:
                all_sessions.add(task_id)
            if has_replan:
                sessions_with_replan.add(task_id)

        total = len(all_sessions)
        with_replan = len(sessions_with_replan)
        trigger_rate = (with_replan / total) if total > 0 else 0.0
        return ReplanStats(
            sessions_with_replan=with_replan,
            total_sessions=total,
            trigger_rate=round(trigger_rate, 3),
        )

    # -- Specialization --

    def _compute_specialization_distribution(self) -> Dict[str, int]:
        """Read current specialization from each agent's activity file."""
        distribution: Dict[str, int] = {}

        if not self._activity_dir.exists():
            return distribution

        for activity_file in self._activity_dir.glob("*.json"):
            try:
                data = json.loads(activity_file.read_text())
                specialization = data.get("specialization") or "none"
                distribution[specialization] = distribution.get(specialization, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue

        return distribution

    # -- Context budget --

    def _compute_context_budget_stats(self) -> ContextBudgetStats:
        """Derive budget utilization from complete events in the activity stream."""
        if not self._activity_stream.exists():
            return ContextBudgetStats()

        total_with_tokens = 0
        budget_exceeded = 0
        utilization_sum = 0.0

        try:
            content = self._activity_stream.read_text()
        except OSError:
            return ContextBudgetStats()

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "token_budget_exceeded":
                budget_exceeded += 1

            elif event_type == "complete":
                input_tokens = event.get("input_tokens")
                output_tokens = event.get("output_tokens")
                if input_tokens is not None and output_tokens is not None:
                    total_with_tokens += 1
                    total_tokens = input_tokens + output_tokens
                    # Use a conservative default budget of 40000 for utilization
                    budget = 40000
                    utilization_sum += min((total_tokens / budget) * 100, 200.0)

        avg_util = (utilization_sum / total_with_tokens) if total_with_tokens > 0 else None
        if avg_util is not None:
            avg_util = round(avg_util, 1)

        return ContextBudgetStats(
            total_tasks_with_tokens=total_with_tokens,
            budget_exceeded_events=budget_exceeded,
            avg_utilization_pct=avg_util,
        )

    # -- Helpers --

    def _iter_session_files(self):
        if not self._sessions_dir.exists():
            return
        yield from self._sessions_dir.glob("*.jsonl")

    def _iter_session_events(self, event_type: str):
        """Yield all session log events matching a given event type."""
        for json_file in self._iter_session_files():
            for event in self._parse_jsonl(json_file):
                if event.get("event") == event_type:
                    yield event

    def _parse_jsonl(self, path: Path) -> List[dict]:
        try:
            events = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return events
        except OSError:
            return []

    def _load_json_list(self, path: Path) -> List[dict]:
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
