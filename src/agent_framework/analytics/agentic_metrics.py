"""Agentic observability metrics aggregated from session logs.

Reads per-task JSONL session logs (logs/sessions/{task_id}.jsonl) and
aggregates the agentic-feature events that the activity stream doesn't
surface: memory recalls, self-eval verdicts, replans, and token budgets.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MemoryHitRateMetrics(BaseModel):
    """How often recalled memories were actually used (vs. sessions without recall)."""

    sessions_with_recall: int
    sessions_without_recall: int
    # Chars injected is a proxy for recall depth; higher = more content surfaced
    avg_chars_injected: float
    total_recalls: int


class SelfEvalMetrics(BaseModel):
    """Self-evaluation verdicts across all tasks in the window."""

    total_evals: int
    auto_pass_count: int  # Skipped because no objective evidence
    pass_count: int
    fail_count: int
    # % of evals that caught a real issue (FAIL) before QA hand-off
    catch_rate_percent: float
    # % of evals that were skipped automatically
    auto_pass_rate_percent: float


class ReplanMetrics(BaseModel):
    """Replan trigger rate across all tasks."""

    tasks_with_replan: int
    total_replan_events: int
    # Of tasks that replanned, how many ultimately completed (approximated by
    # finding a task_complete after a replan in the same session file)
    tasks_completed_after_replan: int
    success_after_replan_percent: float


class ContextBudgetMetrics(BaseModel):
    """Token budget utilisation across LLM calls."""

    total_llm_calls: int
    # Snapshots bucketed into 0-25 / 25-50 / 50-75 / 75-100 %
    utilization_buckets: Dict[str, int]
    avg_tokens_in: float
    avg_tokens_out: float
    avg_total_tokens: float


class AgenticsReport(BaseModel):
    """Full agentic observability report."""

    generated_at: datetime
    time_range_hours: int
    sessions_scanned: int
    memory_hit_rate: MemoryHitRateMetrics
    self_eval: SelfEvalMetrics
    replan: ReplanMetrics
    context_budget: ContextBudgetMetrics


class AgenticsMetrics:
    """Aggregates agentic-feature observability from session log files.

    Session logs live at logs/sessions/{task_id}.jsonl and contain events
    emitted by the framework around memory recall, self-evaluation, replanning,
    and LLM token usage.  Unlike the activity stream these files are per-task
    and survive crashes, making them the authoritative source for agentic ops.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.metrics_dir / "agentics.json"

    def generate_report(self, hours: int = 24) -> AgenticsReport:
        """Aggregate agentic metrics from session logs in the given window.

        Args:
            hours: How far back to look based on event timestamps.

        Returns:
            AgenticsReport with all agentic panel data.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        session_events = self._read_sessions(cutoff)

        report = AgenticsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            sessions_scanned=len(session_events),
            memory_hit_rate=self._compute_memory_hit_rate(session_events),
            self_eval=self._compute_self_eval(session_events),
            replan=self._compute_replan(session_events),
            context_budget=self._compute_context_budget(session_events),
        )

        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Agentics report saved to {self.output_file}")
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_sessions(
        self, cutoff: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return {task_id: [events]} for sessions with activity after cutoff."""
        if not self.sessions_dir.exists():
            return {}

        result: Dict[str, List[Dict[str, Any]]] = {}
        for jsonl_file in self.sessions_dir.glob("*.jsonl"):
            task_id = jsonl_file.stem
            events: List[Dict[str, Any]] = []
            try:
                for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        ts_raw = ev.get("ts", "")
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                        if ts >= cutoff:
                            events.append(ev)
                    except (json.JSONDecodeError, ValueError):
                        continue
            except OSError as e:
                logger.debug(f"Could not read session file {jsonl_file}: {e}")
                continue

            if events:
                result[task_id] = events

        return result

    def _compute_memory_hit_rate(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> MemoryHitRateMetrics:
        sessions_with = 0
        sessions_without = 0
        total_recalls = 0
        total_chars = 0

        for events in sessions.values():
            recalls = [e for e in events if e.get("event") == "memory_recall"]
            if recalls:
                sessions_with += 1
                total_recalls += len(recalls)
                total_chars += sum(
                    int(e.get("chars_injected", 0)) for e in recalls
                )
            else:
                sessions_without += 1

        avg_chars = (total_chars / total_recalls) if total_recalls > 0 else 0.0

        return MemoryHitRateMetrics(
            sessions_with_recall=sessions_with,
            sessions_without_recall=sessions_without,
            avg_chars_injected=avg_chars,
            total_recalls=total_recalls,
        )

    def _compute_self_eval(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> SelfEvalMetrics:
        total = 0
        auto_pass = 0
        passed = 0
        failed = 0

        for events in sessions.values():
            for ev in events:
                if ev.get("event") != "self_eval":
                    continue
                total += 1
                verdict = ev.get("verdict", "").upper()
                if verdict == "AUTO_PASS":
                    auto_pass += 1
                elif verdict.startswith("PASS"):
                    passed += 1
                elif verdict.startswith("FAIL"):
                    failed += 1

        catch_rate = (failed / total * 100) if total > 0 else 0.0
        auto_pass_rate = (auto_pass / total * 100) if total > 0 else 0.0

        return SelfEvalMetrics(
            total_evals=total,
            auto_pass_count=auto_pass,
            pass_count=passed,
            fail_count=failed,
            catch_rate_percent=round(catch_rate, 1),
            auto_pass_rate_percent=round(auto_pass_rate, 1),
        )

    def _compute_replan(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> ReplanMetrics:
        tasks_with_replan = 0
        total_replan_events = 0
        tasks_completed_after_replan = 0

        for events in sessions.values():
            replan_events = [e for e in events if e.get("event") == "replan"]
            if not replan_events:
                continue

            tasks_with_replan += 1
            total_replan_events += len(replan_events)

            # A task "succeeded after replan" if task_complete with status=completed
            # appears after the first replan event in the same session file.
            first_replan_ts = replan_events[0].get("ts", "")
            for ev in events:
                if ev.get("event") != "task_complete":
                    continue
                if ev.get("status") != "completed":
                    continue
                if ev.get("ts", "") >= first_replan_ts:
                    tasks_completed_after_replan += 1
                    break

        success_rate = (
            tasks_completed_after_replan / tasks_with_replan * 100
            if tasks_with_replan > 0
            else 0.0
        )

        return ReplanMetrics(
            tasks_with_replan=tasks_with_replan,
            total_replan_events=total_replan_events,
            tasks_completed_after_replan=tasks_completed_after_replan,
            success_after_replan_percent=round(success_rate, 1),
        )

    def _compute_context_budget(
        self, sessions: Dict[str, List[Dict[str, Any]]]
    ) -> ContextBudgetMetrics:
        # Bucket by cumulative utilisation within a session: sum tokens_in+tokens_out
        # across llm_complete events and compare against a nominal 200k budget.
        _NOMINAL_BUDGET = 200_000

        buckets: Dict[str, int] = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
        total_calls = 0
        total_tokens_in = 0
        total_tokens_out = 0

        for events in sessions.values():
            session_tokens_used = 0
            for ev in events:
                if ev.get("event") != "llm_complete":
                    continue
                tokens_in = int(ev.get("tokens_in", 0))
                tokens_out = int(ev.get("tokens_out", 0))
                session_tokens_used += tokens_in + tokens_out
                total_calls += 1
                total_tokens_in += tokens_in
                total_tokens_out += tokens_out

            if session_tokens_used > 0:
                pct = session_tokens_used / _NOMINAL_BUDGET * 100
                if pct <= 25:
                    buckets["0-25%"] += 1
                elif pct <= 50:
                    buckets["25-50%"] += 1
                elif pct <= 75:
                    buckets["50-75%"] += 1
                else:
                    buckets["75-100%"] += 1

        avg_in = total_tokens_in / total_calls if total_calls > 0 else 0.0
        avg_out = total_tokens_out / total_calls if total_calls > 0 else 0.0
        avg_total = (total_tokens_in + total_tokens_out) / total_calls if total_calls > 0 else 0.0

        return ContextBudgetMetrics(
            total_llm_calls=total_calls,
            utilization_buckets=buckets,
            avg_tokens_in=round(avg_in, 1),
            avg_tokens_out=round(avg_out, 1),
            avg_total_tokens=round(avg_total, 1),
        )
