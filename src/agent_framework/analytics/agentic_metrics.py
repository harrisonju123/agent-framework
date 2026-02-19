"""Agentic feature metrics aggregator.

Reads existing session logs, activity stream, memory stores, and profile
registry to surface agentic-specific observability metrics. Read-only — no
agent code changes required.

Data sources:
- logs/sessions/{task_id}.jsonl       → self_eval, replan, llm_complete events
- .agent-communication/activity-stream.jsonl → task outcomes, budget warnings
- .agent-communication/memory/**/*.json      → memory access counts
- .agent-communication/profile-registry/profiles.json → specialization usage
"""

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# How long to hold the cached report before re-computing
_CACHE_TTL_SECONDS = 30


class MemoryMetrics(BaseModel):
    """Memory store utilization metrics."""
    total_memories: int
    accessed_memories: int  # entries with access_count > 0
    hit_rate: float         # percentage of memories that were ever recalled
    by_category: Dict[str, int]  # category → entry count


class SelfEvalMetrics(BaseModel):
    """Self-evaluation verdict distribution and retry rates."""
    total_tasks_evaluated: int
    tasks_with_failures: int  # tasks where at least one FAIL verdict was issued
    retry_rate: float         # % of evaluated tasks that triggered a self-eval retry
    pass_count: int
    fail_count: int
    auto_pass_count: int      # skipped evals (no objective evidence)


class ReplanMetrics(BaseModel):
    """Dynamic replanning trigger and outcome metrics."""
    total_tasks_with_replan: int
    total_replan_events: int
    success_after_replan: int  # replanned tasks that ultimately completed
    trigger_rate_pct: float    # % of all terminal tasks that triggered at least one replan


class SpecializationMetrics(BaseModel):
    """Specialization profile usage distribution."""
    profiles: Dict[str, int]   # profile_id → total match count
    total_specializations: int


class ContextBudgetMetrics(BaseModel):
    """Context window budget utilization metrics."""
    critical_budget_events: int       # context_budget_critical events in window
    total_token_budget_warnings: int  # critical + token_budget_exceeded events
    avg_output_token_ratio_pct: float  # avg (output / (in + out)) across LLM calls


class AgenticMetricsReport(BaseModel):
    """Complete agentic observability report for the dashboard."""
    generated_at: datetime
    time_range_hours: int
    memory: MemoryMetrics
    self_eval: SelfEvalMetrics
    replan: ReplanMetrics
    specialization: SpecializationMetrics
    context_budget: ContextBudgetMetrics


class AgenticMetrics:
    """Aggregates agentic-feature metrics from existing data sources.

    All reads are best-effort — missing files or parse errors are silently
    skipped so the dashboard always returns a usable (possibly partial) report.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self._sessions_dir = self.workspace / "logs" / "sessions"
        self._stream_file = (
            self.workspace / ".agent-communication" / "activity-stream.jsonl"
        )
        self._memory_dir = self.workspace / ".agent-communication" / "memory"
        self._profile_registry = (
            self.workspace
            / ".agent-communication"
            / "profile-registry"
            / "profiles.json"
        )

        # Simple monotonic-clock cache — not thread-safe but the dashboard
        # runs in a single-process async server, so this is fine.
        self._cache: Optional[AgenticMetricsReport] = None
        self._cache_at: float = 0.0

    def get_report(self, hours: int = 24) -> AgenticMetricsReport:
        """Return the aggregated report, rebuilding at most once per 30 seconds."""
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_at) < _CACHE_TTL_SECONDS:
            return self._cache

        report = self._build_report(hours)
        self._cache = report
        self._cache_at = now
        return report

    # ------------------------------------------------------------------ #
    #  Report builder                                                       #
    # ------------------------------------------------------------------ #

    def _build_report(self, hours: int) -> AgenticMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        session_events = self._read_session_events(cutoff)
        stream_events = self._read_stream_events(cutoff)

        return AgenticMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            memory=self._compute_memory_metrics(),
            self_eval=self._compute_self_eval_metrics(session_events),
            replan=self._compute_replan_metrics(session_events, stream_events),
            specialization=self._compute_specialization_metrics(),
            context_budget=self._compute_context_budget_metrics(
                stream_events, session_events
            ),
        )

    # ------------------------------------------------------------------ #
    #  Data readers                                                         #
    # ------------------------------------------------------------------ #

    def _read_session_events(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """Read all events from per-task session JSONL files within the window."""
        if not self._sessions_dir.exists():
            return []

        all_events: List[Dict[str, Any]] = []
        for session_file in self._sessions_dir.glob("*.jsonl"):
            try:
                for line in session_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        ts_str = event.get("ts", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts >= cutoff:
                                all_events.append(event)
                    except (json.JSONDecodeError, ValueError):
                        continue
            except OSError:
                continue
        return all_events

    def _read_stream_events(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """Read activity stream events within the time window."""
        if not self._stream_file.exists():
            return []

        events: List[Dict[str, Any]] = []
        try:
            for line in self._stream_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    ts_str = event.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts >= cutoff:
                            events.append(event)
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            pass
        return events

    # ------------------------------------------------------------------ #
    #  Metric computations                                                  #
    # ------------------------------------------------------------------ #

    def _compute_memory_metrics(self) -> MemoryMetrics:
        """Aggregate memory access statistics across all memory store files."""
        if not self._memory_dir.exists():
            return MemoryMetrics(
                total_memories=0,
                accessed_memories=0,
                hit_rate=0.0,
                by_category={},
            )

        total = 0
        accessed = 0
        by_category: Counter = Counter()

        for store_file in self._memory_dir.glob("**/*.json"):
            try:
                entries = json.loads(store_file.read_text(encoding="utf-8"))
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    total += 1
                    if entry.get("access_count", 0) > 0:
                        accessed += 1
                    by_category[entry.get("category", "unknown")] += 1
            except (json.JSONDecodeError, OSError):
                continue

        hit_rate = (accessed / total * 100) if total > 0 else 0.0
        return MemoryMetrics(
            total_memories=total,
            accessed_memories=accessed,
            hit_rate=round(hit_rate, 1),
            by_category=dict(by_category),
        )

    def _compute_self_eval_metrics(
        self, session_events: List[Dict[str, Any]]
    ) -> SelfEvalMetrics:
        """Derive self-evaluation verdict distribution and retry rate."""
        # Group self_eval events by task so we can identify per-task failures
        evals_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in session_events:
            if event.get("event") == "self_eval":
                task_id = event.get("task_id", "unknown")
                evals_by_task[task_id].append(event)

        pass_count = sum(
            1
            for evals in evals_by_task.values()
            for e in evals
            if e.get("verdict") == "PASS"
        )
        fail_count = sum(
            1
            for evals in evals_by_task.values()
            for e in evals
            if e.get("verdict") == "FAIL"
        )
        auto_pass_count = sum(
            1
            for evals in evals_by_task.values()
            for e in evals
            if e.get("verdict") == "AUTO_PASS"
        )
        tasks_with_failures = sum(
            1
            for evals in evals_by_task.values()
            if any(e.get("verdict") == "FAIL" for e in evals)
        )

        total_evaluated = len(evals_by_task)
        retry_rate = (
            (tasks_with_failures / total_evaluated * 100) if total_evaluated > 0 else 0.0
        )

        return SelfEvalMetrics(
            total_tasks_evaluated=total_evaluated,
            tasks_with_failures=tasks_with_failures,
            retry_rate=round(retry_rate, 1),
            pass_count=pass_count,
            fail_count=fail_count,
            auto_pass_count=auto_pass_count,
        )

    def _compute_replan_metrics(
        self,
        session_events: List[Dict[str, Any]],
        stream_events: List[Dict[str, Any]],
    ) -> ReplanMetrics:
        """Compute replan trigger rate and post-replan success rate."""
        tasks_with_replan: Set[str] = set()
        replan_count = 0

        for event in session_events:
            if event.get("event") == "replan":
                task_id = event.get("task_id", "")
                if task_id:
                    tasks_with_replan.add(task_id)
                    replan_count += 1

        # Determine task outcomes from the activity stream
        completed_tasks: Set[str] = {
            e["task_id"]
            for e in stream_events
            if e.get("type") == "complete" and e.get("task_id")
        }
        failed_tasks: Set[str] = {
            e["task_id"]
            for e in stream_events
            if e.get("type") == "fail" and e.get("task_id")
        }

        success_after_replan = len(tasks_with_replan & completed_tasks)
        all_terminal = len(completed_tasks | failed_tasks)
        trigger_rate = (
            (len(tasks_with_replan) / all_terminal * 100) if all_terminal > 0 else 0.0
        )

        return ReplanMetrics(
            total_tasks_with_replan=len(tasks_with_replan),
            total_replan_events=replan_count,
            success_after_replan=success_after_replan,
            trigger_rate_pct=round(trigger_rate, 1),
        )

    def _compute_specialization_metrics(self) -> SpecializationMetrics:
        """Read specialization profile match counts from the profile registry."""
        profiles: Dict[str, int] = {}

        if self._profile_registry.exists():
            try:
                data = json.loads(self._profile_registry.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for entry in data:
                        profile_id = entry.get("profile", {}).get("id", "unknown")
                        match_count = entry.get("match_count", 0)
                        profiles[profile_id] = (
                            profiles.get(profile_id, 0) + match_count
                        )
            except (json.JSONDecodeError, OSError):
                pass

        total = sum(profiles.values())
        return SpecializationMetrics(
            profiles=profiles,
            total_specializations=total,
        )

    def _compute_context_budget_metrics(
        self,
        stream_events: List[Dict[str, Any]],
        session_events: List[Dict[str, Any]],
    ) -> ContextBudgetMetrics:
        """Aggregate context budget utilization from stream and session events."""
        critical_events = sum(
            1 for e in stream_events if e.get("type") == "context_budget_critical"
        )
        budget_exceeded = sum(
            1 for e in stream_events if e.get("type") == "token_budget_exceeded"
        )

        # Output token ratio as a proxy for context utilization — higher means
        # the model is generating more relative to its input context.
        utilizations: List[float] = []
        for event in session_events:
            if event.get("event") == "llm_complete":
                tokens_in = event.get("tokens_in", 0) or 0
                tokens_out = event.get("tokens_out", 0) or 0
                total = tokens_in + tokens_out
                if total > 0:
                    utilizations.append(tokens_out / total * 100)

        avg_util = (
            round(sum(utilizations) / len(utilizations), 1) if utilizations else 0.0
        )

        return ContextBudgetMetrics(
            critical_budget_events=critical_events,
            total_token_budget_warnings=critical_events + budget_exceeded,
            avg_output_token_ratio_pct=avg_util,
        )
