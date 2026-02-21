"""
Cross-feature learning loop metrics from session log events.

Aggregates feedback_emitted, qa_pattern_injected, and specialization_adjusted
events to surface how the four feedback loops are performing:
  1. Self-eval failures → memory
  2. QA recurring patterns → engineer prompt warnings
  3. Replan successes → memory
  4. Debate decisions → specialization adjustments

Data source: logs/sessions/{task_id}.jsonl
"""

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


class FeedbackMetricsReport(BaseModel):
    """Complete cross-feature learning metrics report."""
    generated_at: datetime
    time_range_hours: int

    # Aggregate counts per feedback loop
    self_eval_memories_stored: int
    qa_findings_persisted: int
    qa_recurring_patterns_detected: int
    qa_warnings_injected: int
    specialization_adjustments: int
    replan_events: int

    # Top items for dashboard display
    top_missed_criteria: List[str]
    top_recurring_findings: List[str]

    # Per-category event breakdown
    events_by_category: Dict[str, int]

    # Raw events for drill-down
    recent_adjustments: List[Dict[str, Any]]


class FeedbackMetrics:
    """Aggregates cross-feature learning metrics from session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> FeedbackMetricsReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        # Flatten all events
        all_events: List[Dict[str, Any]] = []
        for task_events in events_by_task.values():
            all_events.extend(task_events)

        # Count by event type
        feedback_emitted = [e for e in all_events if e.get("event") == "feedback_emitted"]
        qa_pattern_events = [e for e in all_events if e.get("event") == "qa_pattern_injected"]
        adjustment_events = [e for e in all_events if e.get("event") == "specialization_adjusted"]
        qa_persisted_events = [e for e in all_events if e.get("event") == "qa_findings_persisted"]

        # Categorize feedback_emitted events by source
        category_counts: Dict[str, int] = Counter()
        for evt in feedback_emitted:
            cat = evt.get("category", "unknown")
            category_counts[cat] += 1

        self_eval_count = category_counts.get("self_eval_failures", 0)
        replan_count = category_counts.get("replan", 0)

        # Sum QA findings persisted
        qa_findings_total = sum(e.get("count", 0) for e in qa_persisted_events)

        # QA pattern injection stats
        qa_patterns_detected = sum(e.get("pattern_count", 0) for e in qa_pattern_events)
        qa_warnings_injected = len(qa_pattern_events)

        # Extract top missed criteria from self-eval feedback content
        top_missed = self._extract_top_items(
            [e.get("content_preview", "") for e in feedback_emitted
             if e.get("category") == "self_eval_failures"],
            limit=5,
        )

        # Extract top recurring findings from QA pattern events
        top_findings: List[str] = []
        for evt in qa_pattern_events:
            patterns = evt.get("top_patterns", [])
            top_findings.extend(patterns)
        top_findings = list(dict.fromkeys(top_findings))[:5]  # Dedupe, keep order

        # Recent specialization adjustments for drill-down
        recent_adjustments = sorted(
            adjustment_events,
            key=lambda e: e.get("ts", ""),
            reverse=True,
        )[:10]

        return FeedbackMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            self_eval_memories_stored=self_eval_count,
            qa_findings_persisted=qa_findings_total,
            qa_recurring_patterns_detected=qa_patterns_detected,
            qa_warnings_injected=qa_warnings_injected,
            specialization_adjustments=len(adjustment_events),
            replan_events=replan_count,
            top_missed_criteria=top_missed,
            top_recurring_findings=top_findings,
            events_by_category=dict(category_counts),
            recent_adjustments=recent_adjustments,
        )

    @staticmethod
    def _extract_top_items(contents: List[str], limit: int = 5) -> List[str]:
        """Extract and deduplicate the most common content previews."""
        if not contents:
            return []
        counter = Counter(c.strip() for c in contents if c.strip())
        return [item for item, _ in counter.most_common(limit)]
