"""
Agentic feature metrics aggregation from session logs and activity stream.

Reads session-level events (memory, self-eval, replan, specialization, context
budget) to produce a unified metrics snapshot for the observability dashboard.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..utils.atomic_io import atomic_write_model

logger = logging.getLogger(__name__)


class MemoryMetrics(BaseModel):
    total_recalls: int
    total_stores: int
    avg_chars_injected: float
    categories_distribution: Dict[str, int]
    recall_rate: float  # recalls per task


class SelfEvalMetrics(BaseModel):
    total_evaluations: int
    pass_count: int
    fail_count: int
    auto_pass_count: int
    catch_rate: float  # fail_count / total_evaluations * 100
    avg_eval_attempts: float  # average _self_eval_count across tasks that used self-eval


class ReplanMetrics(BaseModel):
    total_replans: int
    success_after_replan: int
    failure_after_replan: int
    replan_success_rate: float  # success_after_replan / total_replans * 100
    trigger_rate: float  # total_replans / total_tasks * 100


class SpecializationMetrics(BaseModel):
    profile_distribution: Dict[str, int]  # profile_id -> count from prompt_built events
    total_specialized_tasks: int
    total_generic_tasks: int


class ContextBudgetMetrics(BaseModel):
    avg_utilization_percent: float
    tasks_near_limit: int  # >= 80% utilization
    tasks_over_budget: int  # tasks with token_budget_exceeded events
    avg_input_tokens: int
    avg_output_tokens: int


class AgenticMetrics(BaseModel):
    generated_at: datetime
    time_range_hours: int
    memory: MemoryMetrics
    self_eval: SelfEvalMetrics
    replan: ReplanMetrics
    specialization: SpecializationMetrics
    context_budget: ContextBudgetMetrics
    debate: Optional[Dict[str, Any]] = None  # placeholder for future debate metrics


class AgenticMetricsCollector:
    """Aggregates agentic feature metrics from activity stream and session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.sessions_dir = self.workspace / "logs" / "sessions"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.metrics_dir / "agentic-metrics.json"

    def collect(self, hours: int = 24) -> AgenticMetrics:
        """
        Aggregate agentic feature metrics for the given time window.

        Reads the activity stream for task lifecycle events, then scans session
        log files (filtered by mtime) for fine-grained agentic events.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        activity_events = self._read_activity_events(cutoff)
        session_events = self._read_session_events(cutoff)

        total_tasks = self._count_tasks(activity_events)

        metrics = AgenticMetrics(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            memory=self._aggregate_memory(session_events, total_tasks),
            self_eval=self._aggregate_self_eval(session_events, activity_events),
            replan=self._aggregate_replan(session_events, activity_events, total_tasks),
            specialization=self._aggregate_specialization(session_events, total_tasks),
            context_budget=self._aggregate_context_budget(session_events, activity_events),
        )

        atomic_write_model(self.output_file, metrics)
        logger.info(f"Agentic metrics saved to {self.output_file}")

        return metrics

    # ------------------------------------------------------------------ reads

    def _read_activity_events(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """Stream activity-stream.jsonl, returning only events after cutoff."""
        if not self.stream_file.exists():
            return []

        events: List[Dict[str, Any]] = []
        with self.stream_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    if ts >= cutoff:
                        events.append(event)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.debug(f"Skipping malformed activity event: {exc}")
        return events

    def _read_session_events(self, cutoff: datetime) -> List[Dict[str, Any]]:
        """
        Scan session JSONL files modified after cutoff and return their events.

        Filtering by file mtime avoids opening every historical session log.
        """
        if not self.sessions_dir.exists():
            return []

        cutoff_ts = cutoff.timestamp()
        events: List[Dict[str, Any]] = []

        for session_file in self.sessions_dir.glob("*.jsonl"):
            try:
                if session_file.stat().st_mtime < cutoff_ts:
                    continue
            except OSError:
                continue

            with session_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        # Session logs use "ts" not "timestamp"
                        raw_ts = event.get("ts", "")
                        if raw_ts:
                            ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                            if ts < cutoff:
                                continue
                        events.append(event)
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
                        logger.debug(f"Skipping malformed session event: {exc}")

        return events

    # ----------------------------------------------------------- task count

    def _count_tasks(self, activity_events: List[Dict[str, Any]]) -> int:
        """Count distinct tasks that started in the time window."""
        return len({e["task_id"] for e in activity_events if e.get("type") == "start" and e.get("task_id")})

    # ------------------------------------------------------- memory metrics

    def _aggregate_memory(self, session_events: List[Dict[str, Any]], total_tasks: int) -> MemoryMetrics:
        recalls = [e for e in session_events if e.get("event") == "memory_recall"]
        stores = [e for e in session_events if e.get("event") == "memory_store"]

        chars_injected = [e.get("chars_injected", 0) for e in recalls if "chars_injected" in e]
        avg_chars = sum(chars_injected) / len(chars_injected) if chars_injected else 0.0

        categories: Dict[str, int] = defaultdict(int)
        for e in recalls:
            cat = e.get("category")
            if cat:
                categories[cat] += 1

        total_recalls = len(recalls)
        recall_rate = total_recalls / total_tasks if total_tasks > 0 else 0.0

        return MemoryMetrics(
            total_recalls=total_recalls,
            total_stores=len(stores),
            avg_chars_injected=avg_chars,
            categories_distribution=dict(categories),
            recall_rate=recall_rate,
        )

    # --------------------------------------------------- self-eval metrics

    def _aggregate_self_eval(
        self,
        session_events: List[Dict[str, Any]],
        activity_events: List[Dict[str, Any]],
    ) -> SelfEvalMetrics:
        evals = [e for e in session_events if e.get("event") == "self_eval"]

        pass_count = sum(1 for e in evals if e.get("result") == "pass")
        fail_count = sum(1 for e in evals if e.get("result") == "fail")
        auto_pass_count = sum(1 for e in evals if e.get("auto_pass") is True)
        total = len(evals)
        catch_rate = (fail_count / total * 100) if total > 0 else 0.0

        # avg _self_eval_count across tasks that ran self-eval (from complete events)
        eval_counts = [
            e.get("_self_eval_count", 0)
            for e in activity_events
            if e.get("type") == "complete" and e.get("_self_eval_count") is not None
        ]
        avg_attempts = sum(eval_counts) / len(eval_counts) if eval_counts else 0.0

        return SelfEvalMetrics(
            total_evaluations=total,
            pass_count=pass_count,
            fail_count=fail_count,
            auto_pass_count=auto_pass_count,
            catch_rate=catch_rate,
            avg_eval_attempts=avg_attempts,
        )

    # ------------------------------------------------------- replan metrics

    def _aggregate_replan(
        self,
        session_events: List[Dict[str, Any]],
        activity_events: List[Dict[str, Any]],
        total_tasks: int,
    ) -> ReplanMetrics:
        replans = [e for e in session_events if e.get("event") == "replan"]
        total_replans = len(replans)

        # Determine post-replan task outcome from activity events.
        # Use distinct task IDs for success/failure so a task that replans multiple
        # times counts once — avoiding a denominator mismatch with the set intersection.
        replanned_task_ids = {e.get("task_id") for e in replans if e.get("task_id")}
        completed_ids = {e.get("task_id") for e in activity_events if e.get("type") == "complete"}
        failed_ids = {e.get("task_id") for e in activity_events if e.get("type") == "fail"}

        success_after = len(replanned_task_ids & completed_ids)
        failure_after = len(replanned_task_ids & failed_ids)

        total_replanned_tasks = len(replanned_task_ids)
        replan_success_rate = (success_after / total_replanned_tasks * 100) if total_replanned_tasks > 0 else 0.0
        trigger_rate = (total_replans / total_tasks * 100) if total_tasks > 0 else 0.0

        return ReplanMetrics(
            total_replans=total_replans,
            success_after_replan=success_after,
            failure_after_replan=failure_after,
            replan_success_rate=replan_success_rate,
            trigger_rate=trigger_rate,
        )

    # ------------------------------------------------ specialization metrics

    def _aggregate_specialization(
        self, session_events: List[Dict[str, Any]], total_tasks: int
    ) -> SpecializationMetrics:
        prompt_built = [e for e in session_events if e.get("event") == "prompt_built"]

        profile_dist: Dict[str, int] = defaultdict(int)
        for e in prompt_built:
            profile = e.get("profile_id")
            if profile:
                profile_dist[profile] += 1

        specialized_task_ids = {e.get("task_id") for e in prompt_built if e.get("profile_id") and e.get("task_id")}
        total_specialized = len(specialized_task_ids)
        total_generic = max(0, total_tasks - total_specialized)

        return SpecializationMetrics(
            profile_distribution=dict(profile_dist),
            total_specialized_tasks=total_specialized,
            total_generic_tasks=total_generic,
        )

    # --------------------------------------------- context budget metrics

    def _aggregate_context_budget(
        self,
        session_events: List[Dict[str, Any]],
        activity_events: List[Dict[str, Any]],
    ) -> ContextBudgetMetrics:
        llm_events = [e for e in session_events if e.get("event") == "llm_complete"]

        utilizations = [e.get("utilization_percent", 0.0) for e in llm_events if "utilization_percent" in e]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0
        tasks_near = sum(1 for u in utilizations if u >= 80.0)

        over_budget_ids = {
            e.get("task_id")
            for e in activity_events
            if e.get("type") == "token_budget_exceeded" and e.get("task_id")
        }

        input_tokens = [e.get("input_tokens", 0) for e in llm_events if "input_tokens" in e]
        output_tokens = [e.get("output_tokens", 0) for e in llm_events if "output_tokens" in e]
        avg_input = int(sum(input_tokens) / len(input_tokens)) if input_tokens else 0
        avg_output = int(sum(output_tokens) / len(output_tokens)) if output_tokens else 0

        return ContextBudgetMetrics(
            avg_utilization_percent=avg_util,
            tasks_near_limit=tasks_near,
            tasks_over_budget=len(over_budget_ids),
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
        )
