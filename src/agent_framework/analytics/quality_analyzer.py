"""
Quality trend analysis over time.

Tracks code quality metrics:
- Test pass rate per PR
- Test coverage trends
- Static analysis violations
- PR revision count (fewer revisions = better initial quality)
- Time from PR creation to approval
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QualitySnapshot(BaseModel):
    """Quality metrics for a specific time period."""
    date: datetime
    test_pass_rate: float
    avg_test_coverage: Optional[float] = None
    static_analysis_warnings: int = 0
    avg_pr_revisions: float = 0.0
    avg_time_to_approval_hours: Optional[float] = None


class QualityTrendReport(BaseModel):
    """Complete quality trend report."""
    generated_at: datetime
    time_range_days: int
    current_quality_score: float  # 0-100
    trend_direction: str  # "improving", "stable", "declining"
    snapshots: List[QualitySnapshot]
    recommendations: List[str]


class QualityAnalyzer:
    """Analyzes code quality trends over time."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.output_file = self.metrics_dir / "quality-trends.json"

    def analyze(self, days: int = 30) -> QualityTrendReport:
        """
        Analyze quality trends over the specified period.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            QualityTrendReport with trend analysis
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Read events to extract quality metrics
        events = self._read_events(cutoff_time)

        if not events:
            return self._empty_report(days)

        # Build daily snapshots
        snapshots = self._build_snapshots(events, days)

        # Calculate current quality score
        current_score = self._calculate_quality_score(snapshots[-1] if snapshots else None)

        # Determine trend direction
        trend = self._calculate_trend(snapshots)

        # Generate recommendations
        recommendations = self._generate_recommendations(snapshots, trend)

        report = QualityTrendReport(
            generated_at=datetime.utcnow(),
            time_range_days=days,
            current_quality_score=current_score,
            trend_direction=trend,
            snapshots=snapshots[-7:],  # Last 7 days for display
            recommendations=recommendations,
        )

        # Save report
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Quality trend report saved to {self.output_file}")

        return report

    def _read_events(self, cutoff_time: datetime) -> List[Dict]:
        """Read events from activity stream."""
        if not self.stream_file.exists():
            return []

        events = []
        for line in self.stream_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return events

    def _build_snapshots(self, events: List[Dict], days: int) -> List[QualitySnapshot]:
        """Build daily quality snapshots from events."""
        # Group events by day
        events_by_day: Dict[str, List[Dict]] = defaultdict(list)

        for event in events:
            timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            day_key = timestamp.strftime('%Y-%m-%d')
            events_by_day[day_key].append(event)

        # Build snapshots for each day
        snapshots = []
        start_date = datetime.utcnow() - timedelta(days=days)

        for i in range(days):
            date = start_date + timedelta(days=i)
            day_key = date.strftime('%Y-%m-%d')
            day_events = events_by_day.get(day_key, [])

            snapshot = self._calculate_daily_metrics(date, day_events)
            snapshots.append(snapshot)

        return snapshots

    def _calculate_daily_metrics(self, date: datetime, events: List[Dict]) -> QualitySnapshot:
        """Calculate quality metrics for a single day."""
        # Test success rate
        test_events = [e for e in events if e.get('type') in ['test_complete', 'test_fail']]
        if test_events:
            passed = sum(1 for e in test_events if e.get('type') == 'test_complete')
            test_pass_rate = (passed / len(test_events)) * 100
        else:
            test_pass_rate = 100.0  # Default to 100% if no tests run

        # Extract test coverage from test events (if available in error messages)
        coverages = []
        for event in test_events:
            coverage = self._extract_coverage(event.get('title', ''))
            if coverage is not None:
                coverages.append(coverage)

        avg_coverage = sum(coverages) / len(coverages) if coverages else None

        # Static analysis warnings (from error messages)
        static_warnings = 0
        for event in events:
            if 'static' in event.get('title', '').lower() or 'lint' in event.get('title', '').lower():
                warnings = self._extract_warning_count(event.get('error_message', ''))
                static_warnings += warnings

        # PR metrics (from completed tasks with PR URLs)
        pr_events = [e for e in events if e.get('type') == 'complete' and e.get('pr_url')]
        avg_revisions = 1.0  # Placeholder - would need GitHub API integration

        # Time to approval (from task start to PR creation)
        approval_times = []
        for event in pr_events:
            duration_ms = event.get('duration_ms')
            if duration_ms:
                approval_times.append(duration_ms / (1000 * 60 * 60))  # Convert to hours

        avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else None

        return QualitySnapshot(
            date=date,
            test_pass_rate=test_pass_rate,
            avg_test_coverage=avg_coverage,
            static_analysis_warnings=static_warnings,
            avg_pr_revisions=avg_revisions,
            avg_time_to_approval_hours=avg_approval_time,
        )

    def _extract_coverage(self, text: str) -> Optional[float]:
        """Extract test coverage percentage from text."""
        # Look for patterns like "coverage: 85%", "85% coverage", "coverage=85"
        patterns = [
            r'coverage[:\s=]+(\d+\.?\d*)%?',
            r'(\d+\.?\d*)%?\s*coverage',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return None

    def _extract_warning_count(self, text: str) -> int:
        """Extract warning count from static analysis output."""
        if not text:
            return 0

        # Look for patterns like "5 warnings", "warnings: 5", "found 5 issues"
        patterns = [
            r'(\d+)\s+warnings?',
            r'warnings?[:\s]+(\d+)',
            r'found\s+(\d+)\s+issues?',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        return 0

    def _calculate_quality_score(self, snapshot: Optional[QualitySnapshot]) -> float:
        """
        Calculate overall quality score (0-100) from metrics.

        Weighted scoring:
        - Test pass rate: 40%
        - Test coverage: 30%
        - Static analysis: 20%
        - PR efficiency: 10%
        """
        if not snapshot:
            return 0.0

        score = 0.0

        # Test pass rate (40 points max)
        score += (snapshot.test_pass_rate / 100) * 40

        # Test coverage (30 points max)
        if snapshot.avg_test_coverage is not None:
            score += (snapshot.avg_test_coverage / 100) * 30
        else:
            score += 15  # Neutral score if no data

        # Static analysis (20 points max, fewer warnings = better)
        if snapshot.static_analysis_warnings == 0:
            score += 20
        elif snapshot.static_analysis_warnings <= 5:
            score += 15
        elif snapshot.static_analysis_warnings <= 10:
            score += 10
        else:
            score += 5

        # PR efficiency (10 points max)
        if snapshot.avg_pr_revisions <= 1.5:
            score += 10
        elif snapshot.avg_pr_revisions <= 2.5:
            score += 7
        else:
            score += 4

        return min(100.0, score)

    def _calculate_trend(self, snapshots: List[QualitySnapshot]) -> str:
        """Determine if quality is improving, stable, or declining."""
        if len(snapshots) < 7:
            return "insufficient_data"

        # Compare recent week to previous week
        recent_scores = [self._calculate_quality_score(s) for s in snapshots[-7:]]
        previous_scores = [self._calculate_quality_score(s) for s in snapshots[-14:-7]] if len(snapshots) >= 14 else None

        recent_avg = sum(recent_scores) / len(recent_scores)

        if previous_scores:
            previous_avg = sum(previous_scores) / len(previous_scores)
            diff = recent_avg - previous_avg

            if diff > 5:
                return "improving"
            elif diff < -5:
                return "declining"
            else:
                return "stable"
        else:
            # Only one week of data, check if trending up or down within the week
            if recent_scores[-1] > recent_scores[0] + 5:
                return "improving"
            elif recent_scores[-1] < recent_scores[0] - 5:
                return "declining"
            else:
                return "stable"

    def _generate_recommendations(self, snapshots: List[QualitySnapshot], trend: str) -> List[str]:
        """Generate actionable recommendations based on trends."""
        recommendations = []

        if not snapshots:
            return ["Insufficient data for recommendations. Continue running tasks."]

        recent = snapshots[-1] if snapshots else None
        if not recent:
            return recommendations

        # Test pass rate recommendations
        if recent.test_pass_rate < 80:
            recommendations.append(
                f"⚠ Test pass rate is low ({recent.test_pass_rate:.1f}%). "
                "Review and fix flaky tests."
            )
        elif recent.test_pass_rate < 90:
            recommendations.append(
                f"Test pass rate ({recent.test_pass_rate:.1f}%) has room for improvement. "
                "Investigate intermittent failures."
            )

        # Test coverage recommendations
        if recent.avg_test_coverage is not None:
            if recent.avg_test_coverage < 70:
                recommendations.append(
                    f"⚠ Test coverage is low ({recent.avg_test_coverage:.1f}%). "
                    "Add tests for uncovered code paths."
                )
            elif recent.avg_test_coverage < 80:
                recommendations.append(
                    f"Test coverage ({recent.avg_test_coverage:.1f}%) could be improved. "
                    "Target 80%+ coverage for critical paths."
                )

        # Static analysis recommendations
        if recent.static_analysis_warnings > 10:
            recommendations.append(
                f"⚠ High number of static analysis warnings ({recent.static_analysis_warnings}). "
                "Run linters before committing."
            )
        elif recent.static_analysis_warnings > 5:
            recommendations.append(
                f"Static analysis found {recent.static_analysis_warnings} issues. "
                "Address warnings to improve code quality."
            )

        # Trend-based recommendations
        if trend == "declining":
            recommendations.append(
                "⚠ Quality metrics are declining. Review recent changes and reinforce quality gates."
            )
        elif trend == "improving":
            recommendations.append(
                "✅ Quality metrics are improving. Continue current practices."
            )

        # Overall assessment
        current_score = self._calculate_quality_score(recent)
        if current_score >= 90:
            recommendations.append("Excellent quality metrics! Keep up the great work.")
        elif current_score >= 75:
            recommendations.append("Good quality overall. Focus on the specific areas noted above.")
        else:
            recommendations.append(
                "Quality metrics need attention. Prioritize fixing identified issues."
            )

        return recommendations or ["Continue monitoring quality metrics."]

    def _empty_report(self, days: int) -> QualityTrendReport:
        """Generate empty report when no data available."""
        return QualityTrendReport(
            generated_at=datetime.utcnow(),
            time_range_days=days,
            current_quality_score=0.0,
            trend_direction="insufficient_data",
            snapshots=[],
            recommendations=[
                "Insufficient data for quality analysis.",
                "Continue running tasks and tests to collect metrics.",
            ],
        )
