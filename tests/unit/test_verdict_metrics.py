"""Tests for verdict metrics analytics module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent_framework.analytics.verdict_metrics import VerdictMetrics


def _write_session_events(sessions_dir: Path, task_id: str, events: list[dict]):
    """Write mock session events to a JSONL file."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{task_id}.jsonl"
    lines = []
    for event in events:
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        event.setdefault("task_id", task_id)
        lines.append(json.dumps(event))
    path.write_text("\n".join(lines))


class TestVerdictMetrics:
    def test_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = VerdictMetrics(Path(tmpdir))
            report = metrics.generate_report(hours=24)

            assert report.total_tasks_with_verdicts == 0
            assert report.distribution.total_verdicts == 0
            assert report.pattern_frequencies == []
            assert report.recent_audits == []

    def test_single_verdict_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            _write_session_events(sessions_dir, "task-1", [
                {
                    "event": "verdict_audit",
                    "method": "review_outcome",
                    "value": "approved",
                    "agent_id": "qa",
                    "workflow_step": "qa_review",
                    "override_applied": False,
                    "matched_patterns": [
                        {"category": "approve", "pattern": "\\bAPPROVE[D]?\\b"},
                    ],
                    "negation_suppressed": [],
                },
            ])

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            assert report.total_tasks_with_verdicts == 1
            assert report.distribution.by_method["review_outcome"] == 1
            assert report.distribution.by_value["approved"] == 1
            assert report.distribution.override_rate == 0.0

    def test_override_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            _write_session_events(sessions_dir, "task-1", [
                {
                    "event": "verdict_audit",
                    "method": "review_outcome",
                    "value": "needs_fix",
                    "override_applied": True,
                    "matched_patterns": [],
                    "negation_suppressed": [],
                },
                {
                    "event": "verdict_audit",
                    "method": "review_outcome",
                    "value": "approved",
                    "override_applied": False,
                    "matched_patterns": [],
                    "negation_suppressed": [],
                },
            ])

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            assert report.distribution.total_verdicts == 2
            assert report.distribution.override_rate == 0.5

    def test_ambiguous_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            _write_session_events(sessions_dir, "task-1", [
                {"event": "verdict_audit", "method": "ambiguous_halt", "value": None,
                 "override_applied": False, "matched_patterns": [], "negation_suppressed": []},
                {"event": "verdict_audit", "method": "ambiguous_default", "value": "approved",
                 "override_applied": False, "matched_patterns": [], "negation_suppressed": []},
                {"event": "verdict_audit", "method": "review_outcome", "value": "approved",
                 "override_applied": False, "matched_patterns": [], "negation_suppressed": []},
            ])

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            # 2 out of 3 are ambiguous
            assert abs(report.distribution.ambiguous_rate - 2 / 3) < 0.01

    def test_pattern_frequencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            _write_session_events(sessions_dir, "task-1", [
                {
                    "event": "verdict_audit",
                    "method": "review_outcome",
                    "value": "approved",
                    "override_applied": False,
                    "matched_patterns": [
                        {"category": "approve", "pattern": "\\bAPPROVE[D]?\\b"},
                        {"category": "approve", "pattern": "\\bLGTM\\b"},
                    ],
                    "negation_suppressed": [
                        {"category": "critical_issues", "pattern": "\\bCRITICAL\\b.*?:"},
                    ],
                },
            ])

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            freq_map = {(f.category, f.pattern): f for f in report.pattern_frequencies}
            assert ("approve", "\\bAPPROVE[D]?\\b") in freq_map
            approve_freq = freq_map[("approve", "\\bAPPROVE[D]?\\b")]
            assert approve_freq.match_count == 1
            assert approve_freq.suppression_count == 0

            critical_freq = freq_map[("critical_issues", "\\bCRITICAL\\b.*?:")]
            assert critical_freq.match_count == 0
            assert critical_freq.suppression_count == 1
            assert critical_freq.false_positive_risk == 1.0

    def test_fallback_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            _write_session_events(sessions_dir, "task-1", [
                {"event": "condition_eval_audit", "method": "keyword_fallback",
                 "condition_type": "approved", "result": True},
                {"event": "condition_eval_audit", "method": "condition_verdict",
                 "condition_type": "approved", "result": True},
            ])

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            assert report.distribution.fallback_rate == 0.5

    def test_recent_audits_limited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            sessions_dir = workspace / "logs" / "sessions"

            events = [
                {"event": "verdict_audit", "method": "review_outcome", "value": f"v{i}",
                 "override_applied": False, "matched_patterns": [], "negation_suppressed": []}
                for i in range(30)
            ]
            _write_session_events(sessions_dir, "task-1", events)

            metrics = VerdictMetrics(workspace)
            report = metrics.generate_report(hours=24)

            assert len(report.recent_audits) <= 20
