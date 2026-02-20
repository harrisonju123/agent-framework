"""
LLM cost and token tracking from per-task session JSONL logs.

Surfaces cost per task, token efficiency, model distribution,
latency percentiles, and cost trends over time by reading
llm_complete events from session logs.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, List

from pydantic import BaseModel

from ..llm.model_success_store import normalize_model_tier
from .session_loader import load_session_events

logger = logging.getLogger(__name__)


# --- Pydantic report models ---


class TaskCostSummary(BaseModel):
    """Per-task cost and token summary."""
    task_id: str
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    total_duration_ms: int
    llm_call_count: int
    token_efficiency: float  # tokens_out / tokens_in
    primary_model: str       # most-used tier


class ModelTierMetrics(BaseModel):
    """Aggregated metrics for a single model tier (haiku/sonnet/opus)."""
    tier: str
    call_count: int
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    avg_cost_per_call: float
    avg_duration_ms: float
    cost_share_pct: float


class CostTrendBucket(BaseModel):
    """Hourly time-series bucket for cost trends."""
    timestamp: datetime
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    call_count: int
    avg_duration_ms: float


class LatencyMetrics(BaseModel):
    """Latency distribution across all LLM calls."""
    sample_count: int
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


class LlmMetricsReport(BaseModel):
    """Complete LLM cost and token tracking report."""
    generated_at: datetime
    time_range_hours: int
    tasks_with_llm_calls: int
    total_llm_calls: int
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    overall_token_efficiency: float
    model_tiers: List[ModelTierMetrics]
    latency: LatencyMetrics
    top_cost_tasks: List[TaskCostSummary]
    trends: List[CostTrendBucket]


# --- Collector ---


class LlmMetrics:
    """
    Aggregates LLM cost and token metrics from session JSONL logs.

    Each task writes events to logs/sessions/{task_id}.jsonl. This collector
    scans those files (filtered by modification time) and aggregates the
    llm_complete events into the LlmMetricsReport.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> LlmMetricsReport:
        """Generate an LLM metrics report for the given lookback window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)
        llm_by_task = self._extract_llm_events(events_by_task)

        model_tiers = self._aggregate_model_tiers(llm_by_task)
        latency = self._aggregate_latency(llm_by_task)
        top_cost_tasks = self._aggregate_task_costs(llm_by_task)
        trends = self._aggregate_trends(llm_by_task)

        # Compute totals from flattened events
        all_events = [e for events in llm_by_task.values() for e in events]
        total_cost = sum(e.get("cost") or 0.0 for e in all_events)
        total_tokens_in = sum(e.get("tokens_in") or 0 for e in all_events)
        total_tokens_out = sum(e.get("tokens_out") or 0 for e in all_events)
        efficiency = round(total_tokens_out / total_tokens_in, 3) if total_tokens_in > 0 else 0.0

        return LlmMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            tasks_with_llm_calls=len(llm_by_task),
            total_llm_calls=len(all_events),
            total_cost=round(total_cost, 6),
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            overall_token_efficiency=efficiency,
            model_tiers=model_tiers,
            latency=latency,
            top_cost_tasks=top_cost_tasks,
            trends=trends,
        )

    # --- Private helpers ---

    @staticmethod
    def _extract_llm_events(
        events_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Filter to only llm_complete events, dropping tasks with none."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for task_id, events in events_by_task.items():
            llm_events = [e for e in events if e.get("event") == "llm_complete"]
            if llm_events:
                result[task_id] = llm_events
        return result

    @staticmethod
    def _aggregate_model_tiers(
        llm_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> List[ModelTierMetrics]:
        """Flatten all llm_complete events, group by tier, compute per-tier totals."""
        tier_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "cost": 0.0, "tokens_in": 0, "tokens_out": 0, "duration_ms": 0.0}
        )

        for events in llm_by_task.values():
            for e in events:
                tier = normalize_model_tier(e.get("model") or "unknown")
                d = tier_data[tier]
                d["count"] += 1
                d["cost"] += e.get("cost") or 0.0
                d["tokens_in"] += e.get("tokens_in") or 0
                d["tokens_out"] += e.get("tokens_out") or 0
                d["duration_ms"] += e.get("duration_ms") or 0

        grand_total_cost = sum(d["cost"] for d in tier_data.values())

        result = []
        for tier, d in sorted(tier_data.items(), key=lambda x: -x[1]["cost"]):
            count = d["count"]
            result.append(ModelTierMetrics(
                tier=tier,
                call_count=count,
                total_cost=round(d["cost"], 6),
                total_tokens_in=d["tokens_in"],
                total_tokens_out=d["tokens_out"],
                avg_cost_per_call=round(d["cost"] / count, 6) if count > 0 else 0.0,
                avg_duration_ms=round(d["duration_ms"] / count, 1) if count > 0 else 0.0,
                cost_share_pct=round(d["cost"] / grand_total_cost * 100, 1) if grand_total_cost > 0 else 0.0,
            ))

        return result

    @staticmethod
    def _aggregate_latency(
        llm_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> LatencyMetrics:
        """Collect duration_ms values and compute percentiles."""
        durations: List[float] = []
        for events in llm_by_task.values():
            for e in events:
                dur = e.get("duration_ms")
                if dur is not None and dur > 0:
                    durations.append(float(dur))

        if not durations:
            return LatencyMetrics(
                sample_count=0, avg_ms=0.0, p50_ms=0.0, p90_ms=0.0, p99_ms=0.0, max_ms=0.0,
            )

        sorted_dur = sorted(durations)
        n = len(sorted_dur)
        p50 = median(sorted_dur)

        if n >= 2:
            # quantiles(n=100) gives 99 cut points; index 89 = p90, index 98 = p99
            q = quantiles(sorted_dur, n=100)
            p90 = q[89]
            p99 = q[98]
        else:
            p90 = sorted_dur[-1]
            p99 = sorted_dur[-1]

        return LatencyMetrics(
            sample_count=n,
            avg_ms=round(sum(durations) / n, 1),
            p50_ms=round(p50, 1),
            p90_ms=round(p90, 1),
            p99_ms=round(p99, 1),
            max_ms=round(sorted_dur[-1], 1),
        )

    @staticmethod
    def _aggregate_task_costs(
        llm_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> List[TaskCostSummary]:
        """Per-task sum, return top 10 by cost descending."""
        summaries: List[TaskCostSummary] = []

        for task_id, events in llm_by_task.items():
            total_cost = sum(e.get("cost") or 0.0 for e in events)
            total_in = sum(e.get("tokens_in") or 0 for e in events)
            total_out = sum(e.get("tokens_out") or 0 for e in events)
            total_dur = int(sum(e.get("duration_ms") or 0 for e in events))

            tier_counts = Counter(
                normalize_model_tier(e.get("model") or "unknown") for e in events
            )
            primary_model = tier_counts.most_common(1)[0][0] if tier_counts else "unknown"

            summaries.append(TaskCostSummary(
                task_id=task_id,
                total_cost=round(total_cost, 6),
                total_tokens_in=total_in,
                total_tokens_out=total_out,
                total_duration_ms=total_dur,
                llm_call_count=len(events),
                token_efficiency=round(total_out / total_in, 3) if total_in > 0 else 0.0,
                primary_model=primary_model,
            ))

        summaries.sort(key=lambda s: s.total_cost, reverse=True)
        return summaries[:10]

    @staticmethod
    def _aggregate_trends(
        llm_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> List[CostTrendBucket]:
        """Hourly bucket by flooring ts, accumulate cost/tokens/count/duration."""
        buckets: Dict[datetime, Dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "count": 0, "duration_ms": 0.0}
        )

        for events in llm_by_task.values():
            for e in events:
                raw_ts = e.get("ts")
                if not raw_ts:
                    continue
                try:
                    event_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    bucket_ts = event_ts.replace(minute=0, second=0, microsecond=0)
                except (ValueError, TypeError):
                    continue

                d = buckets[bucket_ts]
                d["cost"] += e.get("cost") or 0.0
                d["tokens_in"] += e.get("tokens_in") or 0
                d["tokens_out"] += e.get("tokens_out") or 0
                d["count"] += 1
                d["duration_ms"] += e.get("duration_ms") or 0

        result = []
        for bucket_ts in sorted(buckets.keys()):
            d = buckets[bucket_ts]
            count = d["count"]
            result.append(CostTrendBucket(
                timestamp=bucket_ts,
                total_cost=round(d["cost"], 6),
                total_tokens_in=d["tokens_in"],
                total_tokens_out=d["tokens_out"],
                call_count=count,
                avg_duration_ms=round(d["duration_ms"] / count, 1) if count > 0 else 0.0,
            ))

        return result
