"""
Dynamic token budget optimization.

Analyzes historical token usage to calculate optimal budgets per task type:
- Calculates P50, P90, P99 token usage from actual data
- Adjusts budgets based on percentile thresholds
- Provides recommendations for model downgrades
- Monitors budget effectiveness over time
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskTypeBudget(BaseModel):
    """Budget recommendation for a task type."""
    task_type: str
    current_budget: int
    recommended_budget: int
    p50_usage: int
    p90_usage: int
    p99_usage: int
    sample_count: int
    avg_usage: int
    max_usage: int
    potential_savings_pct: float
    confidence: str  # "low", "medium", "high"


class BudgetOptimizationReport(BaseModel):
    """Complete budget optimization report."""
    generated_at: datetime
    time_range_days: int
    task_types_analyzed: int
    total_potential_savings_pct: float
    budgets: List[TaskTypeBudget]
    recommendations: List[str]


class BudgetOptimizer:
    """Analyzes token usage and optimizes budgets."""

    # Minimum samples required for confident recommendations
    MIN_SAMPLES_LOW = 5
    MIN_SAMPLES_MEDIUM = 10
    MIN_SAMPLES_HIGH = 20

    # Default budgets (from agent.py)
    DEFAULT_BUDGETS = {
        "planning": 30000,
        "implementation": 50000,
        "testing": 20000,
        "escalation": 80000,
        "review": 25000,
        "architecture": 40000,
        "coordination": 15000,
        "documentation": 15000,
        "fix": 30000,
        "bugfix": 30000,
        "bug-fix": 30000,
        "verification": 20000,
        "status_report": 10000,
        "enhancement": 40000,
    }

    def __init__(self, workspace: Path, percentile: int = 90):
        """
        Initialize budget optimizer.

        Args:
            workspace: Project workspace path
            percentile: Percentile to use for budget calculation (default: 90)
        """
        self.workspace = Path(workspace)
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.metrics_dir = self.workspace / ".agent-communication" / "metrics"
        self.output_file = self.metrics_dir / "budget-optimization.json"
        self.percentile = percentile

    def analyze(self, days: int = 7) -> BudgetOptimizationReport:
        """
        Analyze token usage and generate budget recommendations.

        Args:
            days: Number of days to look back (default: 7)

        Returns:
            BudgetOptimizationReport with recommendations
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Read activity events to get task completion data
        tasks = self._read_completed_tasks(cutoff_time)

        if not tasks:
            return self._empty_report(days)

        # Group by task type and extract token usage
        usage_by_type = self._group_by_task_type(tasks)

        # Calculate budget recommendations
        budgets = []
        total_savings = 0.0
        total_types = 0

        for task_type, usage_data in usage_by_type.items():
            if len(usage_data) < self.MIN_SAMPLES_LOW:
                continue

            budget_rec = self._calculate_budget(task_type, usage_data)
            budgets.append(budget_rec)

            if budget_rec.potential_savings_pct > 0:
                total_savings += budget_rec.potential_savings_pct
                total_types += 1

        avg_savings = total_savings / total_types if total_types > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(budgets)

        report = BudgetOptimizationReport(
            generated_at=datetime.utcnow(),
            time_range_days=days,
            task_types_analyzed=len(budgets),
            total_potential_savings_pct=avg_savings,
            budgets=sorted(budgets, key=lambda x: x.potential_savings_pct, reverse=True),
            recommendations=recommendations,
        )

        # Save report
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Budget optimization report saved to {self.output_file}")

        return report

    def _read_completed_tasks(self, cutoff_time: datetime) -> List[Dict]:
        """Read completed task events from activity stream."""
        if not self.stream_file.exists():
            return []

        # Build task map from events
        tasks = {}
        for line in self.stream_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                if timestamp < cutoff_time:
                    continue

                task_id = event.get('task_id')
                if not task_id:
                    continue

                event_type = event.get('type')

                if event_type == 'start':
                    # Initialize task
                    tasks[task_id] = {
                        'id': task_id,
                        'title': event.get('title', ''),
                        'started_at': timestamp,
                        'status': 'started',
                    }
                elif event_type == 'complete' and task_id in tasks:
                    tasks[task_id]['status'] = 'completed'
                    tasks[task_id]['completed_at'] = timestamp
                    # Extract token usage
                    input_tokens = event.get('input_tokens', 0)
                    output_tokens = event.get('output_tokens', 0)
                    tasks[task_id]['total_tokens'] = input_tokens + output_tokens

            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Filter to only completed tasks
        completed = [t for t in tasks.values() if t.get('status') == 'completed']

        # Try to extract token usage from logs (if available)
        # For now, we'll need to enhance ActivityEvent to include token data
        # This is a placeholder - in real implementation, we'd need token data in events
        return completed

    def _group_by_task_type(self, tasks: List[Dict]) -> Dict[str, List[int]]:
        """Group tasks by type and extract token usage."""
        usage_by_type: Dict[str, List[int]] = defaultdict(list)

        for task in tasks:
            task_type = self._infer_task_type(task['title'])
            total_tokens = task.get('total_tokens', 0)

            # Only include tasks with token data
            if total_tokens > 0:
                usage_by_type[task_type].append(total_tokens)

        return usage_by_type

    def _infer_task_type(self, title: str) -> str:
        """Infer task type from title."""
        title_lower = title.lower()

        if 'escalation' in title_lower:
            return 'escalation'
        elif 'test' in title_lower:
            return 'testing'
        elif 'review' in title_lower:
            return 'review'
        elif 'architect' in title_lower or 'design' in title_lower:
            return 'architecture'
        elif 'implement' in title_lower or 'add' in title_lower:
            return 'implementation'
        elif 'fix' in title_lower or 'bug' in title_lower:
            return 'fix'
        elif 'refactor' in title_lower:
            return 'enhancement'
        elif 'plan' in title_lower:
            return 'planning'
        else:
            return 'other'

    def _calculate_budget(self, task_type: str, usage_data: List[int]) -> TaskTypeBudget:
        """Calculate optimal budget for a task type."""
        if not usage_data:
            current_budget = self.DEFAULT_BUDGETS.get(task_type, 40000)
            return TaskTypeBudget(
                task_type=task_type,
                current_budget=current_budget,
                recommended_budget=current_budget,
                p50_usage=0,
                p90_usage=0,
                p99_usage=0,
                sample_count=0,
                avg_usage=0,
                max_usage=0,
                potential_savings_pct=0.0,
                confidence="low",
            )

        sorted_usage = sorted(usage_data)
        n = len(sorted_usage)

        p50 = sorted_usage[n // 2]
        p90 = sorted_usage[int(n * 0.9)] if n > 1 else sorted_usage[0]
        p99 = sorted_usage[int(n * 0.99)] if n > 2 else sorted_usage[-1]
        avg = sum(sorted_usage) // n
        max_usage = sorted_usage[-1]

        # Recommended budget at specified percentile (with 10% buffer)
        recommended = int(p90 * 1.1) if self.percentile == 90 else int(p99 * 1.1)

        # Current budget
        current_budget = self.DEFAULT_BUDGETS.get(task_type, 40000)

        # Calculate potential savings
        if current_budget > recommended:
            savings_pct = ((current_budget - recommended) / current_budget) * 100
        else:
            savings_pct = 0.0

        # Determine confidence level
        if n >= self.MIN_SAMPLES_HIGH:
            confidence = "high"
        elif n >= self.MIN_SAMPLES_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        return TaskTypeBudget(
            task_type=task_type,
            current_budget=current_budget,
            recommended_budget=recommended,
            p50_usage=p50,
            p90_usage=p90,
            p99_usage=p99,
            sample_count=n,
            avg_usage=avg,
            max_usage=max_usage,
            potential_savings_pct=savings_pct,
            confidence=confidence,
        )

    def _generate_recommendations(self, budgets: List[TaskTypeBudget]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High-confidence savings opportunities
        high_savings = [b for b in budgets if b.confidence == "high" and b.potential_savings_pct > 10]
        if high_savings:
            recommendations.append(
                f"Enable dynamic budgets immediately: {len(high_savings)} task types "
                f"show 10%+ savings with high confidence"
            )

        # Opportunities for model downgrades
        low_token_tasks = [b for b in budgets if b.p90_usage < 20000 and b.sample_count >= 10]
        if low_token_tasks:
            task_types = ', '.join([b.task_type for b in low_token_tasks[:3]])
            recommendations.append(
                f"Consider using Haiku for: {task_types} (P90 usage < 20K tokens)"
            )

        # High variance warnings
        high_variance = [
            b for b in budgets
            if b.p99_usage > b.p50_usage * 2 and b.sample_count >= 10
        ]
        if high_variance:
            task_types = ', '.join([b.task_type for b in high_variance[:3]])
            recommendations.append(
                f"High variance detected in: {task_types}. Consider splitting into subtypes."
            )

        # Overall assessment
        avg_savings = sum(b.potential_savings_pct for b in budgets) / len(budgets) if budgets else 0
        if avg_savings > 15:
            recommendations.append(
                f"Strong optimization opportunity: Average {avg_savings:.1f}% savings across all task types"
            )
        elif avg_savings > 5:
            recommendations.append(
                f"Moderate optimization opportunity: Average {avg_savings:.1f}% savings possible"
            )
        else:
            recommendations.append(
                "Current budgets are well-tuned. Continue monitoring for changes."
            )

        return recommendations or ["Insufficient data for recommendations. Continue collecting metrics."]

    def _empty_report(self, days: int) -> BudgetOptimizationReport:
        """Generate empty report when no data available."""
        return BudgetOptimizationReport(
            generated_at=datetime.utcnow(),
            time_range_days=days,
            task_types_analyzed=0,
            total_potential_savings_pct=0.0,
            budgets=[],
            recommendations=[
                "Insufficient historical data for budget optimization.",
                "Continue running tasks to collect token usage metrics.",
                "Minimum 5 samples per task type required for recommendations.",
            ],
        )

    def export_config_snippet(self) -> Dict:
        """
        Export optimized budgets as config snippet.

        Returns:
            Dict suitable for agent-framework.yaml optimization.token_budgets
        """
        try:
            if not self.output_file.exists():
                self.analyze()

            report_data = json.loads(self.output_file.read_text())
            report = BudgetOptimizationReport(**report_data)

            config_budgets = {}
            for budget in report.budgets:
                if budget.confidence in ["high", "medium"] and budget.potential_savings_pct > 5:
                    config_budgets[budget.task_type] = budget.recommended_budget

            return {
                "token_budgets": config_budgets,
                "_generated_at": report.generated_at.isoformat(),
                "_confidence_threshold": "medium",
                "_savings_threshold_pct": 5.0,
            }

        except Exception as e:
            logger.error(f"Failed to export config snippet: {e}")
            return {}
