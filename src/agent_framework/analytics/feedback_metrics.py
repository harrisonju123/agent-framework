"""
Cross-feature feedback loop metrics from session logs.

Aggregates how the four feedback loops are performing:
1. Self-eval failures → memory (missed acceptance criteria)
2. Replan successes → memory (recovery patterns)
3. QA recurring findings → engineer prompt (warning injection)
4. Debate decisions → specialization (profile adjustment)

Data source: logs/sessions/{task_id}.jsonl
"""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _load_events_from_file(path: Path) -> List[Dict]:
    """Parse all events from a JSONL session file."""
    events = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass
    return events


class FeedbackMetricsReport(BaseModel):
    """Complete feedback loop metrics report."""
    generated_at: datetime
    time_range_hours: int
    total_sessions_scanned: int

    # Self-eval → memory
    self_eval_failures_stored: int
    top_missed_criteria: List[str]

    # QA findings → memory
    qa_findings_persisted: int

    # QA patterns → engineer prompt
    qa_warnings_injected: int
    qa_pattern_top_findings: List[str]

    # Debate → specialization
    specialization_adjustments: int
    adjustment_details: List[Dict[str, str]]

    # FeedbackBus totals
    feedback_events_emitted: int


class FeedbackMetrics:
    """Aggregates cross-feature feedback loop metrics from session logs."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "logs" / "sessions"

    def generate_report(self, hours: int = 24) -> FeedbackMetricsReport:
        """Generate feedback metrics report for the given time range."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        self_eval_stored = 0
        qa_persisted = 0
        qa_injected = 0
        specialization_adjustments = 0
        feedback_emitted = 0
        missed_criteria: Counter = Counter()
        top_qa_patterns: List[str] = []
        adjustment_details: List[Dict[str, str]] = []
        sessions_scanned = 0

        if not self.sessions_dir.exists():
            return self._empty_report(hours)

        for session_file in self.sessions_dir.glob("*.jsonl"):
            try:
                if session_file.stat().st_mtime < cutoff.timestamp():
                    continue
            except OSError:
                continue

            sessions_scanned += 1
            events = _load_events_from_file(session_file)

            for event in events:
                event_type = event.get("event", "")

                if event_type == "self_eval_failure_stored":
                    self_eval_stored += 1
                    preview = event.get("content_preview", "")
                    if preview:
                        missed_criteria[preview[:80]] += 1

                elif event_type == "qa_findings_persisted":
                    qa_persisted += event.get("findings_count", 0)

                elif event_type == "qa_pattern_injected":
                    qa_injected += 1
                    patterns = event.get("top_patterns", [])
                    for p in patterns:
                        if p not in top_qa_patterns:
                            top_qa_patterns.append(p)

                elif event_type == "specialization_adjusted":
                    specialization_adjustments += 1
                    adjustment_details.append({
                        "original": event.get("original_profile", ""),
                        "adjusted": event.get("adjusted_profile", ""),
                        "signal_strength": str(event.get("signal_strength", 0)),
                    })

                elif event_type == "feedback_emitted":
                    feedback_emitted += 1

        return FeedbackMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_sessions_scanned=sessions_scanned,
            self_eval_failures_stored=self_eval_stored,
            top_missed_criteria=[f"{desc} ({count}x)" for desc, count in missed_criteria.most_common(5)],
            qa_findings_persisted=qa_persisted,
            qa_warnings_injected=qa_injected,
            qa_pattern_top_findings=top_qa_patterns[:5],
            specialization_adjustments=specialization_adjustments,
            adjustment_details=adjustment_details[:5],
            feedback_events_emitted=feedback_emitted,
        )

    def _empty_report(self, hours: int) -> FeedbackMetricsReport:
        return FeedbackMetricsReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_sessions_scanned=0,
            self_eval_failures_stored=0,
            top_missed_criteria=[],
            qa_findings_persisted=0,
            qa_warnings_injected=0,
            qa_pattern_top_findings=[],
            specialization_adjustments=0,
            adjustment_details=[],
            feedback_events_emitted=0,
        )
