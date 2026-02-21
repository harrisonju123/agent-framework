"""
Feedback loop metrics from session log events.

Aggregates feedback bus events (feedback_bus_self_eval, feedback_bus_qa_pattern,
qa_warnings_injected) to surface cross-feature learning health:
- Store counts by category (missed_criteria, qa_patterns, specialization_hints)
- Top recurring patterns
- Specialization hint hit rate
- QA warning injection frequency

Data source: logs/sessions/{task_id}.jsonl
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from .session_loader import load_session_events

logger = logging.getLogger(__name__)


class CategoryBreakdown(BaseModel):
    """Per-category feedback bus store counts."""
    category: str
    store_count: int
    total_memories: int


class TopPattern(BaseModel):
    """Most frequently stored feedback pattern."""
    pattern: str
    count: int
    category: str


class FeedbackLoopMetrics(BaseModel):
    """Aggregated feedback loop metrics."""
    total_self_eval_stores: int
    total_qa_pattern_stores: int
    total_qa_warnings_injected: int
    total_specialization_hints: int
    self_eval_memories_stored: int
    qa_pattern_memories_stored: int
    qa_warnings_chars_injected: int
    by_category: List[CategoryBreakdown]
    top_patterns: List[TopPattern]
    specialization_hint_hit_rate: float


class FeedbackLoopReport(BaseModel):
    """Complete feedback loop metrics report."""
    generated_at: datetime
    time_range_hours: int
    metrics: FeedbackLoopMetrics
    raw_events: List[Dict[str, Any]]


class FeedbackLoopAnalyzer:
    """Aggregates feedback loop metrics from session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> FeedbackLoopReport:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        events_by_task = load_session_events(self.sessions_dir, cutoff)

        raw_events = self._extract_feedback_events(events_by_task)
        metrics = self._aggregate(raw_events)

        return FeedbackLoopReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            metrics=metrics,
            raw_events=raw_events,
        )

    def _extract_feedback_events(
        self, events_by_task: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Filter to feedback-bus-related events across all tasks."""
        feedback_event_types = {
            "feedback_bus_self_eval",
            "feedback_bus_qa_pattern",
            "qa_warnings_injected",
            "memory_store",
        }
        result = []
        for events in events_by_task.values():
            for e in events:
                if e.get("event") in feedback_event_types:
                    result.append(e)
        return result

    def _aggregate(self, events: List[Dict[str, Any]]) -> FeedbackLoopMetrics:
        # Count events by type
        self_eval_stores = 0
        qa_pattern_stores = 0
        qa_warnings_injected = 0
        self_eval_memories = 0
        qa_pattern_memories = 0
        qa_warnings_chars = 0

        # Pattern frequency tracking
        category_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"store_count": 0, "total_memories": 0}
        )

        for e in events:
            event_type = e.get("event")

            if event_type == "feedback_bus_self_eval":
                self_eval_stores += 1
                memories = e.get("memories_stored", 0)
                self_eval_memories += memories
                category_counts["missed_criteria"]["store_count"] += 1
                category_counts["missed_criteria"]["total_memories"] += memories

            elif event_type == "feedback_bus_qa_pattern":
                qa_pattern_stores += 1
                memories = e.get("memories_stored", 0)
                qa_pattern_memories += memories
                category_counts["qa_patterns"]["store_count"] += 1
                category_counts["qa_patterns"]["total_memories"] += memories

            elif event_type == "qa_warnings_injected":
                qa_warnings_injected += 1
                qa_warnings_chars += e.get("chars", 0)

        # Build category breakdown
        by_category = [
            CategoryBreakdown(
                category=cat,
                store_count=counts["store_count"],
                total_memories=counts["total_memories"],
            )
            for cat, counts in sorted(category_counts.items())
        ]

        # Specialization hint metrics from memory_store events
        # (these appear as regular memory_store events with the repo context)
        total_specialization_hints = sum(
            1 for e in events
            if e.get("event") == "memory_store"
            and "specialization" in str(e).lower()
        )

        # Hint hit rate: ratio of qa_warnings_injected to total engineer tasks
        # that had qa_patterns stored (how often stored patterns are surfaced)
        hint_hit_rate = 0.0
        total_stores = self_eval_stores + qa_pattern_stores
        if total_stores > 0:
            hint_hit_rate = qa_warnings_injected / total_stores * 100

        # Top patterns: aggregate from event descriptions
        pattern_counter: Counter = Counter()
        for e in events:
            event_type = e.get("event", "")
            if event_type == "feedback_bus_self_eval":
                pattern_counter[("missed_criteria", f"self_eval_failure (task {e.get('task_id', '?')[:8]})")] += 1
            elif event_type == "feedback_bus_qa_pattern":
                pattern_counter[("qa_patterns", f"qa_findings ({e.get('memories_stored', 0)} patterns)")] += 1

        top_patterns = [
            TopPattern(category=cat, pattern=pat, count=cnt)
            for (cat, pat), cnt in pattern_counter.most_common(10)
        ]

        return FeedbackLoopMetrics(
            total_self_eval_stores=self_eval_stores,
            total_qa_pattern_stores=qa_pattern_stores,
            total_qa_warnings_injected=qa_warnings_injected,
            total_specialization_hints=total_specialization_hints,
            self_eval_memories_stored=self_eval_memories,
            qa_pattern_memories_stored=qa_pattern_memories,
            qa_warnings_chars_injected=qa_warnings_chars,
            by_category=by_category,
            top_patterns=top_patterns,
            specialization_hint_hit_rate=hint_hit_rate,
        )
