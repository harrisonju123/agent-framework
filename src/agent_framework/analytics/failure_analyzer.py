"""
Failure pattern detection and root cause analysis.

Automatically analyzes escalation tasks and failed tasks to identify:
- Common error patterns
- Problematic agent types or workflows
- Systemic issues
- Trending failure causes
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FailureCategory(BaseModel):
    """A categorized failure pattern."""
    category: str  # "syntax_error", "test_failure", "timeout", "mcp_error", etc.
    pattern: str  # Regex or substring that matches this category
    count: int
    percentage: float
    affected_agents: List[str]
    sample_errors: List[str]
    recommendation: Optional[str] = None


class FailureTrend(BaseModel):
    """Trending failure over time."""
    category: str
    weekly_count: int
    weekly_change_pct: float  # Percentage change from previous week
    is_increasing: bool


class FailureAnalysisReport(BaseModel):
    """Complete failure analysis report."""
    generated_at: datetime
    time_range_hours: int
    total_failures: int
    failure_rate: float
    categories: List[FailureCategory]
    trends: List[FailureTrend]
    top_recommendations: List[str]


class FailureAnalyzer:
    """Analyzes failure patterns and provides recommendations."""

    # Failure categorization patterns (regex patterns for common errors)
    FAILURE_PATTERNS = {
        'syntax_error': [
            r'syntax error',
            r'SyntaxError:',
            r'invalid syntax',
            r'unexpected token',
        ],
        'test_failure': [
            r'test.*failed',
            r'FAIL:',
            r'assertion.*failed',
            r'expected.*but got',
            r'\d+ failed,',
        ],
        'timeout': [
            r'timeout',
            r'timed out',
            r'deadline exceeded',
            r'context deadline exceeded',
        ],
        'mcp_error': [
            r'MCP.*error',
            r'mcp.*timeout',
            r'mcp_.*failed',
            r'MCP server.*not responding',
        ],
        'compilation_error': [
            r'compilation error',
            r'build failed',
            r'cannot compile',
            r'undefined:',
        ],
        'nil_pointer': [
            r'nil pointer',
            r'null pointer',
            r'NullPointerException',
            r'cannot.*nil',
        ],
        'import_error': [
            r'import error',
            r'cannot import',
            r'module.*not found',
            r'no module named',
        ],
        'permission_error': [
            r'permission denied',
            r'access denied',
            r'forbidden',
        ],
        'network_error': [
            r'network error',
            r'connection.*failed',
            r'connection refused',
            r'DNS.*failed',
        ],
    }

    # Recommendations for common failure categories
    RECOMMENDATIONS = {
        'timeout': 'Consider increasing timeout values or optimizing slow operations',
        'mcp_error': 'Check MCP server configuration and add retries for MCP calls',
        'test_failure': 'Review test stability and add better error messages',
        'syntax_error': 'Enable pre-commit linting hooks to catch syntax errors early',
        'compilation_error': 'Add compilation check before test phase',
        'nil_pointer': 'Add nil checks and validation for optional fields',
        'import_error': 'Verify dependencies are installed and import paths are correct',
        'permission_error': 'Check file permissions and service account access',
        'network_error': 'Add retry logic for network calls and check service availability',
    }

    def __init__(self, workspace: Path, threshold: int = 3):
        """
        Initialize failure analyzer.

        Args:
            workspace: Project workspace path
            threshold: Minimum number of occurrences to report a pattern (default: 3)
        """
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.reports_dir = self.workspace / ".agent-communication" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.reports_dir / "failure-analysis.json"
        self.threshold = threshold

    def analyze(self, hours: int = 168) -> FailureAnalysisReport:
        """
        Analyze failures from the specified time range.

        Args:
            hours: Number of hours to look back (default: 168 = 1 week)

        Returns:
            FailureAnalysisReport with categorized failures and recommendations
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Read failure events
        failures = self._read_failures(cutoff_time)

        if not failures:
            return self._empty_report(hours)

        # Categorize failures
        categories = self._categorize_failures(failures)

        # Calculate trends (compare to previous period)
        trends = self._calculate_trends(hours)

        # Generate top recommendations
        top_recommendations = self._generate_recommendations(categories)

        # Calculate failure rate
        all_events = self._read_all_events(cutoff_time)
        total_tasks = len(set(e['task_id'] for e in all_events if e.get('type') == 'start'))
        failure_rate = (len(failures) / total_tasks * 100) if total_tasks > 0 else 0.0

        report = FailureAnalysisReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_failures=len(failures),
            failure_rate=failure_rate,
            categories=categories,
            trends=trends,
            top_recommendations=top_recommendations,
        )

        # Save report
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Failure analysis report saved to {self.output_file}")

        return report

    def _read_failures(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Read failure events from activity stream."""
        if not self.stream_file.exists():
            return []

        failures = []
        for line in self.stream_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get('type') != 'fail':
                    continue

                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    failures.append(event)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug(f"Failed to parse event: {e}")

        return failures

    def _read_all_events(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Read all events from activity stream for statistics."""
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

    def _categorize_failures(self, failures: List[Dict[str, Any]]) -> List[FailureCategory]:
        """Categorize failures by error pattern."""
        categorized: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        uncategorized = []

        for failure in failures:
            error_msg = failure.get('error_message', '').lower()
            matched = False

            for category, patterns in self.FAILURE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, error_msg, re.IGNORECASE):
                        categorized[category].append(failure)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                uncategorized.append(failure)

        # Add uncategorized as a category if significant
        if len(uncategorized) >= self.threshold:
            categorized['other'] = uncategorized

        # Build category objects
        categories = []
        for category, failures_in_cat in categorized.items():
            if len(failures_in_cat) < self.threshold:
                continue

            affected_agents = list(set(f['agent'] for f in failures_in_cat))
            sample_errors = [
                f.get('error_message', 'Unknown error')[:200]
                for f in failures_in_cat[:3]
            ]

            categories.append(FailureCategory(
                category=category,
                pattern=', '.join(self.FAILURE_PATTERNS.get(category, [''])),
                count=len(failures_in_cat),
                percentage=len(failures_in_cat) / len(failures) * 100,
                affected_agents=affected_agents,
                sample_errors=sample_errors,
                recommendation=self.RECOMMENDATIONS.get(category),
            ))

        return sorted(categories, key=lambda x: x.count, reverse=True)

    def _calculate_trends(self, hours: int) -> List[FailureTrend]:
        """Calculate failure trends by comparing to previous period."""
        current_cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        previous_cutoff = datetime.now(timezone.utc) - timedelta(hours=hours * 2)

        current_failures = self._read_failures(current_cutoff)
        previous_failures = self._read_failures(previous_cutoff)

        # Remove current period from previous
        previous_failures = [
            f for f in previous_failures
            if datetime.fromisoformat(f['timestamp'].replace('Z', '+00:00')) < current_cutoff
        ]

        # Categorize both periods
        current_by_category = self._group_by_category(current_failures)
        previous_by_category = self._group_by_category(previous_failures)

        trends = []
        for category in set(list(current_by_category.keys()) + list(previous_by_category.keys())):
            current_count = len(current_by_category.get(category, []))
            previous_count = len(previous_by_category.get(category, []))

            if current_count < self.threshold:
                continue

            if previous_count == 0:
                change_pct = 100.0 if current_count > 0 else 0.0
            else:
                change_pct = ((current_count - previous_count) / previous_count) * 100

            trends.append(FailureTrend(
                category=category,
                weekly_count=current_count,
                weekly_change_pct=change_pct,
                is_increasing=change_pct > 10,  # 10% threshold for "increasing"
            ))

        return sorted(trends, key=lambda x: abs(x.weekly_change_pct), reverse=True)

    def _group_by_category(self, failures: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group failures by category."""
        categorized: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for failure in failures:
            error_msg = failure.get('error_message', '').lower()
            matched = False

            for category, patterns in self.FAILURE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, error_msg, re.IGNORECASE):
                        categorized[category].append(failure)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                categorized['other'].append(failure)

        return categorized

    def _generate_recommendations(self, categories: List[FailureCategory]) -> List[str]:
        """Generate top recommendations based on failure categories."""
        recommendations = []

        for category in categories[:5]:  # Top 5 categories
            if category.recommendation:
                rec = f"{category.category.replace('_', ' ').title()}: {category.recommendation}"
                recommendations.append(rec)

        return recommendations

    def _empty_report(self, hours: int) -> FailureAnalysisReport:
        """Generate an empty report when no failures found."""
        return FailureAnalysisReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_failures=0,
            failure_rate=0.0,
            categories=[],
            trends=[],
            top_recommendations=['No failures detected - great job!'],
        )
