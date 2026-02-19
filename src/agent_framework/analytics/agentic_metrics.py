"""
Agentic feature metrics collected from per-task session JSONL logs.

Surfaces observable signals from the framework's agentic subsystems:
- Memory recall frequency
- Self-evaluation catch rate (how often self-eval caught issues before QA)
- Replanning trigger and success rate
- Specialization profile distribution (from current agent activity)
- Context budget utilization (prompt length distribution)
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Pydantic report models ---


class MemoryMetrics(BaseModel):
    """Memory recall frequency from session logs."""
    total_recalls: int
    tasks_with_recall: int
    avg_chars_injected: float
    # Fraction of tasks that had at least one memory injection
    recall_rate: float


class SelfEvalMetrics(BaseModel):
    """Self-evaluation outcomes from session logs."""
    total_evals: int
    pass_count: int
    fail_count: int
    auto_pass_count: int
    # How often self-eval surfaced a real issue (FAIL / total non-auto evals)
    catch_rate: float


class ReplanMetrics(BaseModel):
    """Replanning trigger and outcome from session logs."""
    total_replans: int
    tasks_with_replan: int
    tasks_completed_after_replan: int
    trigger_rate: float  # tasks_with_replan / total_observed_tasks
    # Of tasks that replanned, fraction that ultimately completed
    success_rate_after_replan: float


class SpecializationMetrics(BaseModel):
    """Engineer specialization profile distribution (current agent snapshot)."""
    # profile name → agent count; None key = "unspecialized"
    distribution: Dict[str, int]
    total_active_agents: int


class DebateMetrics(BaseModel):
    """Debate subsystem metrics (not yet implemented in core)."""
    available: bool = False
    note: str = "Debate system not yet implemented"


class ContextBudgetMetrics(BaseModel):
    """Prompt length distribution as a proxy for context budget utilization."""
    sample_count: int
    avg_prompt_length: int
    max_prompt_length: int
    min_prompt_length: int
    p50_prompt_length: int
    p90_prompt_length: int


class AgenticMetricsReport(BaseModel):
    """Complete agentic observability report."""
    generated_at: datetime
    time_range_hours: int
    total_observed_tasks: int
    memory: MemoryMetrics
    self_eval: SelfEvalMetrics
    replan: ReplanMetrics
    specialization: SpecializationMetrics
    debate: DebateMetrics
    context_budget: ContextBudgetMetrics


# --- Collector ---


class AgenticMetrics:
    """
    Aggregates agentic feature metrics from session JSONL logs.

    Each task writes events to logs/sessions/{task_id}.jsonl. This collector
    scans those files (filtered by modification time) and aggregates the signals
    defined in AgenticMetricsReport.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"
        self.activity_dir = self.workspace / ".agent-communication" / "activity"

    def generate_report(self, hours: int = 24) -> AgenticMetricsReport:
        """
        Generate an agentic metrics report for the given lookback window.

        Scans session JSONL files modified within the last `hours` hours and
        aggregates the four observable agentic signals. Specialization is read
        from the live agent activity files rather than session logs (not yet
        logged there).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = self._load_session_events(cutoff)

        memory = self._aggregate_memory(events_by_task)
        self_eval = self._aggregate_self_eval(events_by_task)
        replan = self._aggregate_replan(events_by_task)
        specialization = self._read_specialization_distribution()
        context_budget = self._aggregate_context_budget(events_by_task)

        return AgenticMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_observed_tasks=len(events_by_task),
            memory=memory,
            self_eval=self_eval,
            replan=replan,
            specialization=specialization,
            debate=DebateMetrics(),
            context_budget=context_budget,
        )

    # --- private helpers ---

    def _load_session_events(self, cutoff: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read session JSONL files modified after cutoff and bucket events by task_id.

        We use file mtime as a fast pre-filter — if a file hasn't been touched
        since the cutoff there's nothing new in it.
        """
        if not self.sessions_dir.exists():
            return {}

        events_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        cutoff_ts = cutoff.timestamp()

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff_ts:
                    continue
            except OSError:
                continue

            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by event timestamp, not just file mtime
                    raw_ts = event.get("ts")
                    if raw_ts:
                        try:
                            event_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                            if event_ts < cutoff:
                                continue
                        except (ValueError, TypeError):
                            pass

                    task_id = event.get("task_id", path.stem)
                    events_by_task[task_id].append(event)

            except OSError as e:
                logger.debug(f"Could not read session log {path}: {e}")

        return events_by_task

    def _aggregate_memory(self, events_by_task: Dict[str, List[Dict[str, Any]]]) -> MemoryMetrics:
        """Aggregate memory_recall events across all tasks."""
        total_recalls = 0
        tasks_with_recall = 0
        total_chars = 0

        for events in events_by_task.values():
            recalls = [e for e in events if e.get("event") == "memory_recall"]
            if recalls:
                tasks_with_recall += 1
                total_recalls += len(recalls)
                total_chars += sum(e.get("chars_injected", 0) for e in recalls)

        total_tasks = len(events_by_task)
        return MemoryMetrics(
            total_recalls=total_recalls,
            tasks_with_recall=tasks_with_recall,
            avg_chars_injected=round(total_chars / total_recalls, 1) if total_recalls > 0 else 0.0,
            recall_rate=round(tasks_with_recall / total_tasks, 3) if total_tasks > 0 else 0.0,
        )

    def _aggregate_self_eval(self, events_by_task: Dict[str, List[Dict[str, Any]]]) -> SelfEvalMetrics:
        """Aggregate self_eval events (verdicts: PASS / FAIL / AUTO_PASS)."""
        pass_count = 0
        fail_count = 0
        auto_pass_count = 0

        for events in events_by_task.values():
            for e in events:
                if e.get("event") != "self_eval":
                    continue
                verdict = (e.get("verdict") or "").upper()
                if verdict == "PASS":
                    pass_count += 1
                elif verdict == "FAIL":
                    fail_count += 1
                elif verdict == "AUTO_PASS":
                    auto_pass_count += 1

        total_evals = pass_count + fail_count + auto_pass_count
        # catch_rate: of real evals (non-auto), how often did it catch a problem?
        real_evals = pass_count + fail_count
        catch_rate = round(fail_count / real_evals, 3) if real_evals > 0 else 0.0

        return SelfEvalMetrics(
            total_evals=total_evals,
            pass_count=pass_count,
            fail_count=fail_count,
            auto_pass_count=auto_pass_count,
            catch_rate=catch_rate,
        )

    def _aggregate_replan(self, events_by_task: Dict[str, List[Dict[str, Any]]]) -> ReplanMetrics:
        """
        Aggregate replan events.

        A task "completed after replan" if it has both a replan event and a
        task_complete event in its session log.
        """
        tasks_with_replan: set[str] = set()
        tasks_completed_after_replan = 0
        total_replans = 0

        for task_id, events in events_by_task.items():
            replans = [e for e in events if e.get("event") == "replan"]
            if replans:
                tasks_with_replan.add(task_id)
                total_replans += len(replans)

                # Check if the task ultimately completed
                completed = any(e.get("event") == "task_complete" for e in events)
                if completed:
                    tasks_completed_after_replan += 1

        total_tasks = len(events_by_task)
        n_replan_tasks = len(tasks_with_replan)

        return ReplanMetrics(
            total_replans=total_replans,
            tasks_with_replan=n_replan_tasks,
            tasks_completed_after_replan=tasks_completed_after_replan,
            trigger_rate=round(n_replan_tasks / total_tasks, 3) if total_tasks > 0 else 0.0,
            success_rate_after_replan=round(
                tasks_completed_after_replan / n_replan_tasks, 3
            ) if n_replan_tasks > 0 else 0.0,
        )

    def _read_specialization_distribution(self) -> SpecializationMetrics:
        """
        Read current specialization from live agent activity JSON files.

        Specialization is not yet written to session logs, so we derive the
        current distribution from the activity snapshot that each agent writes
        on every state transition.
        """
        distribution: Dict[str, int] = defaultdict(int)
        total_agents = 0

        if not self.activity_dir.exists():
            return SpecializationMetrics(distribution={}, total_active_agents=0)

        for path in self.activity_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                total_agents += 1
                spec = data.get("specialization")
                profile = spec if spec else "none"
                distribution[profile] += 1
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"Could not read activity file {path}: {e}")

        return SpecializationMetrics(
            distribution=dict(distribution),
            total_active_agents=total_agents,
        )

    def _aggregate_context_budget(self, events_by_task: Dict[str, List[Dict[str, Any]]]) -> ContextBudgetMetrics:
        """
        Derive context budget utilization from prompt_built event prompt lengths.

        prompt_length (chars) is a reasonable proxy for how much of the context
        window is consumed by the system prompt + injected context.
        """
        lengths: List[int] = []

        for events in events_by_task.values():
            for e in events:
                if e.get("event") != "prompt_built":
                    continue
                length = e.get("prompt_length")
                if isinstance(length, int) and length > 0:
                    lengths.append(length)

        if not lengths:
            return ContextBudgetMetrics(
                sample_count=0,
                avg_prompt_length=0,
                max_prompt_length=0,
                min_prompt_length=0,
                p50_prompt_length=0,
                p90_prompt_length=0,
            )

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        p50 = int(median(sorted_lengths))
        # quantiles() requires n >= 2; fall back to max for single-element lists
        p90 = int(quantiles(sorted_lengths, n=10)[8]) if n >= 2 else sorted_lengths[-1]

        return ContextBudgetMetrics(
            sample_count=n,
            avg_prompt_length=int(sum(lengths) / n),
            max_prompt_length=sorted_lengths[-1],
            min_prompt_length=sorted_lengths[0],
            p50_prompt_length=p50,
            p90_prompt_length=p90,
        )
