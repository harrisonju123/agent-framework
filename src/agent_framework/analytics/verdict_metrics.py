"""
Verdict audit metrics from session JSONL event logs.

Aggregates verdict_audit and condition_eval_audit events to surface
pattern match frequencies, fallback rates, override rates, and
ambiguous verdict rates — key diagnostics for detecting pattern
matching false positives before they cause P0s.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


class VerdictMethodDistribution(BaseModel):
    """How verdicts were determined across tasks."""
    total_verdicts: int
    by_method: Dict[str, int]       # review_outcome, ambiguous_halt, ambiguous_default, no_changes_marker
    by_value: Dict[str, int]        # approved, needs_fix, no_changes, None
    fallback_rate: float            # fraction using keyword_fallback in condition evals
    override_rate: float            # fraction where CRITICAL/MAJOR overrode APPROVE
    ambiguous_rate: float           # fraction of ambiguous_halt + ambiguous_default


class VerdictPatternFrequency(BaseModel):
    """Per-pattern match statistics across all audits."""
    category: str                   # "approve", "critical_issues", etc.
    pattern: str                    # regex string
    match_count: int
    suppression_count: int          # negation-suppressed matches
    false_positive_risk: float      # suppression_count / (match_count + suppression_count)


class VerdictMetricsReport(BaseModel):
    """Aggregated verdict audit report."""
    generated_at: datetime
    time_range_hours: int
    total_tasks_with_verdicts: int
    distribution: VerdictMethodDistribution
    pattern_frequencies: List[VerdictPatternFrequency]
    recent_audits: List[Dict[str, Any]]  # last 20 verdict_audit events


class VerdictMetrics:
    """Aggregates verdict audit metrics from session JSONL files."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> VerdictMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        verdict_audits: list[dict] = []
        condition_audits: list[dict] = []

        for task_events in events_by_task.values():
            for event in task_events:
                event_name = event.get("event")
                if event_name == "verdict_audit":
                    verdict_audits.append(event)
                elif event_name == "condition_eval_audit":
                    condition_audits.append(event)

        distribution = self._compute_distribution(verdict_audits, condition_audits)
        pattern_freqs = self._compute_pattern_frequencies(verdict_audits)
        recent = sorted(verdict_audits, key=lambda e: e.get("ts", ""), reverse=True)[:20]

        return VerdictMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_tasks_with_verdicts=len(verdict_audits),
            distribution=distribution,
            pattern_frequencies=pattern_freqs,
            recent_audits=recent,
        )

    def _compute_distribution(
        self, verdict_audits: list[dict], condition_audits: list[dict],
    ) -> VerdictMethodDistribution:
        total = len(verdict_audits)
        method_counts: Counter = Counter()
        value_counts: Counter = Counter()
        override_count = 0

        for audit in verdict_audits:
            method_counts[audit.get("method", "unknown")] += 1
            value_counts[str(audit.get("value"))] += 1
            if audit.get("override_applied"):
                override_count += 1

        # Condition eval fallback rate
        fallback_count = sum(
            1 for c in condition_audits if c.get("method") == "keyword_fallback"
        )
        total_condition_evals = len(condition_audits)
        fallback_rate = (fallback_count / total_condition_evals) if total_condition_evals > 0 else 0.0

        ambiguous = method_counts.get("ambiguous_halt", 0) + method_counts.get("ambiguous_default", 0)
        ambiguous_rate = (ambiguous / total) if total > 0 else 0.0
        override_rate = (override_count / total) if total > 0 else 0.0

        return VerdictMethodDistribution(
            total_verdicts=total,
            by_method=dict(method_counts),
            by_value=dict(value_counts),
            fallback_rate=round(fallback_rate, 4),
            override_rate=round(override_rate, 4),
            ambiguous_rate=round(ambiguous_rate, 4),
        )

    def _compute_pattern_frequencies(
        self, verdict_audits: list[dict],
    ) -> list[VerdictPatternFrequency]:
        # Key: (category, pattern) -> {match_count, suppression_count}
        stats: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"match": 0, "suppress": 0})

        for audit in verdict_audits:
            for pm in audit.get("matched_patterns", []):
                key = (pm.get("category", ""), pm.get("pattern", ""))
                stats[key]["match"] += 1

            for pm in audit.get("negation_suppressed", []):
                key = (pm.get("category", ""), pm.get("pattern", ""))
                stats[key]["suppress"] += 1

        result = []
        for (category, pattern), counts in sorted(stats.items()):
            total = counts["match"] + counts["suppress"]
            risk = (counts["suppress"] / total) if total > 0 else 0.0
            result.append(VerdictPatternFrequency(
                category=category,
                pattern=pattern,
                match_count=counts["match"],
                suppression_count=counts["suppress"],
                false_positive_risk=round(risk, 4),
            ))

        return result
