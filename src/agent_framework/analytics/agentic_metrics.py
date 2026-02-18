"""Agentic observability metrics from session logs.

Aggregates metrics about the framework's agentic features:
- Memory recall hit rate and context injection stats
- Self-evaluation catch rate (issues found before QA)
- Replan trigger rate and success after replanning
- Context budget utilization bucketed by usage bands
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Pydantic models ---

class MemoryMetrics(BaseModel):
    """How often and how much memory context is injected."""
    sessions_with_recall: int = 0
    sessions_without_recall: int = 0
    total_sessions: int = 0
    hit_rate_pct: float = 0.0
    avg_chars_injected: float = 0.0
    total_recalls: int = 0


class SelfEvalMetrics(BaseModel):
    """Self-evaluation verdict distribution and catch rate."""
    total_evals: int = 0
    auto_pass_count: int = 0
    pass_count: int = 0
    fail_count: int = 0
    catch_rate_pct: float = 0.0
    auto_pass_rate_pct: float = 0.0


class ReplanMetrics(BaseModel):
    """Replan trigger frequency and post-replan outcomes."""
    total_replans: int = 0
    tasks_with_replans: int = 0
    tasks_completed_after_replan: int = 0
    replan_success_rate_pct: float = 0.0


class ContextBudgetMetrics(BaseModel):
    """Token usage bucketed into utilization bands."""
    total_completions: int = 0
    band_0_25_pct: int = 0
    band_25_50_pct: int = 0
    band_50_75_pct: int = 0
    band_75_100_pct: int = 0
    band_over_100_pct: int = 0
    avg_utilization_pct: float = 0.0


class AgenticMetricsReport(BaseModel):
    """Full agentic observability report."""
    generated_at: datetime
    time_range_hours: int
    memory: MemoryMetrics
    self_eval: SelfEvalMetrics
    replan: ReplanMetrics
    context_budget: ContextBudgetMetrics


# --- Default token budgets (mirrors BudgetManager defaults) ---

_DEFAULT_BUDGETS: Dict[str, int] = {
    "planning": 30_000,
    "implementation": 50_000,
    "testing": 20_000,
    "escalation": 80_000,
    "review": 25_000,
    "architecture": 40_000,
    "coordination": 15_000,
    "documentation": 15_000,
    "fix": 30_000,
    "bugfix": 30_000,
    "bug_fix": 30_000,
    "verification": 20_000,
    "status_report": 10_000,
    "enhancement": 40_000,
}

_FALLBACK_BUDGET = 40_000


class AgenticMetrics:
    """Reads session JSONL logs and computes agentic observability metrics."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"
        self.output_file = (
            self.workspace / ".agent-communication" / "metrics" / "agentics.json"
        )

    def generate_report(self, hours: int = 24) -> AgenticMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        sessions = self._load_sessions(cutoff)

        memory = self._compute_memory_metrics(sessions)
        self_eval = self._compute_self_eval_metrics(sessions)
        replan = self._compute_replan_metrics(sessions)
        context_budget = self._compute_context_budget_metrics(sessions)

        report = AgenticMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            memory=memory,
            self_eval=self_eval,
            replan=replan,
            context_budget=context_budget,
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Agentic metrics report saved to {self.output_file}")
        return report

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def _load_sessions(self, cutoff: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Load session JSONL files, returning {task_id: [events]} for recent sessions."""
        if not self.sessions_dir.exists():
            return {}

        sessions: Dict[str, List[Dict[str, Any]]] = {}

        for path in self.sessions_dir.glob("*.jsonl"):
            # Quick mtime check to skip old files
            try:
                if path.stat().st_mtime < cutoff.timestamp():
                    continue
            except OSError:
                continue

            task_id = path.stem
            events: List[Dict[str, Any]] = []
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(
                        entry.get("ts", "").replace("Z", "+00:00")
                    )
                    if ts >= cutoff:
                        events.append(entry)
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

            if events:
                sessions[task_id] = events

        return sessions

    # ------------------------------------------------------------------
    # Memory metrics
    # ------------------------------------------------------------------

    def _compute_memory_metrics(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> MemoryMetrics:
        total_sessions = len(sessions)
        if total_sessions == 0:
            return MemoryMetrics()

        sessions_with = 0
        total_recalls = 0
        total_chars = 0

        for events in sessions.values():
            recalls = [e for e in events if e.get("event") == "memory_recall"]
            if recalls:
                sessions_with += 1
                total_recalls += len(recalls)
                total_chars += sum(e.get("chars_injected", 0) for e in recalls)

        avg_chars = total_chars / total_recalls if total_recalls else 0.0
        hit_rate = (sessions_with / total_sessions * 100) if total_sessions else 0.0

        return MemoryMetrics(
            sessions_with_recall=sessions_with,
            sessions_without_recall=total_sessions - sessions_with,
            total_sessions=total_sessions,
            hit_rate_pct=round(hit_rate, 1),
            avg_chars_injected=round(avg_chars, 1),
            total_recalls=total_recalls,
        )

    # ------------------------------------------------------------------
    # Self-eval metrics
    # ------------------------------------------------------------------

    def _compute_self_eval_metrics(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> SelfEvalMetrics:
        auto_pass = 0
        passed = 0
        failed = 0

        for events in sessions.values():
            for e in events:
                if e.get("event") != "self_eval":
                    continue
                verdict = (e.get("verdict") or "").upper()
                if verdict == "AUTO_PASS":
                    auto_pass += 1
                elif verdict == "PASS":
                    passed += 1
                elif verdict == "FAIL":
                    failed += 1

        total = auto_pass + passed + failed
        # Catch rate: how often self-eval caught issues (FAIL / non-auto evals)
        non_auto = passed + failed
        catch_rate = (failed / non_auto * 100) if non_auto else 0.0
        auto_pass_rate = (auto_pass / total * 100) if total else 0.0

        return SelfEvalMetrics(
            total_evals=total,
            auto_pass_count=auto_pass,
            pass_count=passed,
            fail_count=failed,
            catch_rate_pct=round(catch_rate, 1),
            auto_pass_rate_pct=round(auto_pass_rate, 1),
        )

    # ------------------------------------------------------------------
    # Replan metrics
    # ------------------------------------------------------------------

    def _compute_replan_metrics(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> ReplanMetrics:
        total_replans = 0
        tasks_with_replans: set = set()
        tasks_completed_after: set = set()

        for task_id, events in sessions.items():
            had_replan = False
            completed = False

            for e in events:
                ev = e.get("event")
                if ev == "replan":
                    total_replans += 1
                    had_replan = True
                elif ev == "task_complete":
                    completed = True

            if had_replan:
                tasks_with_replans.add(task_id)
                if completed:
                    tasks_completed_after.add(task_id)

        replan_count = len(tasks_with_replans)
        success_count = len(tasks_completed_after)
        success_rate = (success_count / replan_count * 100) if replan_count else 0.0

        return ReplanMetrics(
            total_replans=total_replans,
            tasks_with_replans=replan_count,
            tasks_completed_after_replan=success_count,
            replan_success_rate_pct=round(success_rate, 1),
        )

    # ------------------------------------------------------------------
    # Context budget utilization
    # ------------------------------------------------------------------

    def _compute_context_budget_metrics(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> ContextBudgetMetrics:
        bands = {
            "0_25": 0,
            "25_50": 0,
            "50_75": 0,
            "75_100": 0,
            "over_100": 0,
        }
        total_util = 0.0
        total_completions = 0

        for task_id, events in sessions.items():
            # Determine task type from task_start event
            budget = _FALLBACK_BUDGET
            for e in events:
                if e.get("event") == "task_start":
                    task_type = (e.get("task_type") or "").lower().replace("-", "_")
                    budget = _DEFAULT_BUDGETS.get(task_type, _FALLBACK_BUDGET)
                    break

            # Sum tokens across all llm_complete events in the session
            for e in events:
                if e.get("event") != "llm_complete":
                    continue
                tokens_in = e.get("tokens_in", 0)
                tokens_out = e.get("tokens_out", 0)
                total_tokens = tokens_in + tokens_out
                if total_tokens <= 0:
                    continue

                utilization = total_tokens / budget * 100
                total_util += utilization
                total_completions += 1

                if utilization > 100:
                    bands["over_100"] += 1
                elif utilization > 75:
                    bands["75_100"] += 1
                elif utilization > 50:
                    bands["50_75"] += 1
                elif utilization > 25:
                    bands["25_50"] += 1
                else:
                    bands["0_25"] += 1

        avg_util = (total_util / total_completions) if total_completions else 0.0

        return ContextBudgetMetrics(
            total_completions=total_completions,
            band_0_25_pct=bands["0_25"],
            band_25_50_pct=bands["25_50"],
            band_50_75_pct=bands["50_75"],
            band_75_100_pct=bands["75_100"],
            band_over_100_pct=bands["over_100"],
            avg_utilization_pct=round(avg_util, 1),
        )
