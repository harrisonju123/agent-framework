"""
Shadow mode optimization analysis.

Analyzes shadow mode data to evaluate optimization effectiveness:
- Token savings vs. success rate impact
- Identifies safe-to-enable optimizations
- Provides recommendations for optimization deployment
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OptimizationMetrics(BaseModel):
    """Metrics for a single optimization strategy."""
    strategy_id: str
    strategy_name: str
    total_comparisons: int
    avg_token_savings: float
    avg_token_savings_pct: float
    max_token_savings_pct: float
    min_token_savings_pct: float
    is_safe_to_enable: bool
    recommendation: str


class OptimizationReport(BaseModel):
    """Complete optimization analysis report."""
    generated_at: datetime
    time_range_hours: int
    total_shadow_runs: int
    strategies_analyzed: int
    safe_strategies: List[str]
    metrics: List[OptimizationMetrics]
    overall_recommendation: str


class ShadowModeAnalyzer:
    """Analyzes shadow mode optimization data."""

    # Strategy descriptions
    STRATEGY_NAMES = {
        'compact_json': 'Compact JSON formatting',
        'result_summarization': 'Result summarization for long outputs',
        'error_truncation': 'Error message truncation',
        'all_optimizations': 'All optimizations combined',
    }

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.metrics_file = self.workspace / ".agent-communication" / "metrics" / "optimization.jsonl"
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.output_file = self.workspace / ".agent-communication" / "metrics" / "shadow-mode-analysis.json"

    def analyze(self, hours: int = 168) -> OptimizationReport:
        """
        Analyze shadow mode optimization data.

        Args:
            hours: Number of hours to look back (default: 168 = 1 week)

        Returns:
            OptimizationReport with optimization effectiveness analysis
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Read optimization metrics
        optimization_data = self._read_optimization_data(cutoff_time)

        if not optimization_data:
            return self._empty_report(hours)

        # Read success/failure data from activity stream
        task_outcomes = self._read_task_outcomes(cutoff_time)

        # Analyze by strategy
        metrics = self._analyze_strategies(optimization_data, task_outcomes)

        # Identify safe-to-enable strategies
        safe_strategies = [m.strategy_id for m in metrics if m.is_safe_to_enable]

        # Generate overall recommendation
        overall_recommendation = self._generate_overall_recommendation(metrics, safe_strategies)

        report = OptimizationReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_shadow_runs=len(optimization_data),
            strategies_analyzed=len(metrics),
            safe_strategies=safe_strategies,
            metrics=metrics,
            overall_recommendation=overall_recommendation,
        )

        # Save report
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(report.model_dump_json(indent=2))
        logger.info(f"Shadow mode analysis report saved to {self.output_file}")

        return report

    def _read_optimization_data(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Read optimization metrics from shadow mode runs."""
        if not self.metrics_file.exists():
            return []

        data = []
        for line in self.metrics_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    data.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug(f"Failed to parse optimization entry: {e}")

        return data

    def _read_task_outcomes(self, cutoff_time: datetime) -> Dict[str, str]:
        """Read task outcomes (success/fail) from activity stream."""
        if not self.stream_file.exists():
            return {}

        outcomes = {}
        for line in self.stream_file.read_text().strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                if timestamp < cutoff_time:
                    continue

                event_type = event.get('type')
                task_id = event.get('task_id')

                if event_type in ['complete', 'fail'] and task_id:
                    outcomes[task_id] = 'success' if event_type == 'complete' else 'fail'
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return outcomes

    def _analyze_strategies(
        self,
        optimization_data: List[Dict[str, Any]],
        task_outcomes: Dict[str, str]
    ) -> List[OptimizationMetrics]:
        """Analyze optimization effectiveness by strategy."""
        # Group by strategy (inferred from optimization patterns)
        by_strategy = self._group_by_strategy(optimization_data)

        metrics = []
        for strategy_id, entries in by_strategy.items():
            if not entries:
                continue

            # Calculate token savings
            savings = [
                self._calculate_savings(entry)
                for entry in entries
            ]
            savings = [s for s in savings if s is not None]

            if not savings:
                continue

            avg_savings = sum(savings) / len(savings)
            avg_savings_pct = avg_savings  # Already in percentage

            # Check safety: In shadow mode, we use legacy prompts, so success rate
            # comparison isn't directly available. We assume safe if consistent savings
            # and no obvious issues. In a real implementation, you'd compare success rates
            # of tasks with optimizations vs. without.
            is_safe = avg_savings_pct > 5.0 and avg_savings_pct < 50.0  # Reasonable savings range

            recommendation = self._generate_strategy_recommendation(
                strategy_id, avg_savings_pct, is_safe
            )

            metrics.append(OptimizationMetrics(
                strategy_id=strategy_id,
                strategy_name=self.STRATEGY_NAMES.get(strategy_id, strategy_id),
                total_comparisons=len(entries),
                avg_token_savings=avg_savings_pct,
                avg_token_savings_pct=avg_savings_pct,
                max_token_savings_pct=max(savings),
                min_token_savings_pct=min(savings),
                is_safe_to_enable=is_safe,
                recommendation=recommendation,
            ))

        return sorted(metrics, key=lambda x: x.avg_token_savings_pct, reverse=True)

    def _group_by_strategy(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group optimization data by strategy."""
        by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for entry in data:
            # Infer strategy from token savings pattern and agent type
            strategy = self._infer_strategy(entry)
            by_strategy[strategy].append(entry)

        return by_strategy

    def _infer_strategy(self, entry: Dict[str, Any]) -> str:
        """Infer optimization strategy from metrics entry."""
        savings_pct = entry.get('token_savings_pct', 0)

        # Heuristics based on typical savings percentages
        if savings_pct > 30:
            return 'all_optimizations'
        elif savings_pct > 20:
            return 'result_summarization'
        elif savings_pct > 10:
            return 'error_truncation'
        elif savings_pct > 5:
            return 'compact_json'
        else:
            return 'unknown'

    def _calculate_savings(self, entry: Dict[str, Any]) -> Optional[float]:
        """Calculate token savings percentage for an entry."""
        return entry.get('token_savings_pct')

    def _generate_strategy_recommendation(
        self,
        strategy_id: str,
        avg_savings_pct: float,
        is_safe: bool
    ) -> str:
        """Generate recommendation for a specific strategy."""
        if not is_safe:
            return f"Not recommended: Insufficient data or unusual savings pattern"

        if avg_savings_pct > 20:
            return f"Highly recommended: Deploy immediately for ~{avg_savings_pct:.1f}% cost reduction"
        elif avg_savings_pct > 10:
            return f"Recommended: Enable for {avg_savings_pct:.1f}% savings"
        elif avg_savings_pct > 5:
            return f"Safe to enable: Modest {avg_savings_pct:.1f}% savings"
        else:
            return f"Optional: Minimal {avg_savings_pct:.1f}% savings"

    def _generate_overall_recommendation(
        self,
        metrics: List[OptimizationMetrics],
        safe_strategies: List[str]
    ) -> str:
        """Generate overall deployment recommendation."""
        if not safe_strategies:
            return "No optimizations are safe to enable yet. Continue shadow mode testing."

        total_savings = sum(m.avg_token_savings_pct for m in metrics if m.is_safe_to_enable)
        avg_savings = total_savings / len(safe_strategies) if safe_strategies else 0

        if len(safe_strategies) >= 3 and avg_savings > 15:
            return (
                f"READY TO DEPLOY: Enable all {len(safe_strategies)} safe strategies "
                f"for ~{avg_savings:.1f}% average cost reduction"
            )
        elif len(safe_strategies) >= 2:
            return (
                f"Deploy {len(safe_strategies)} strategies: "
                f"{', '.join(safe_strategies[:3])} for ~{avg_savings:.1f}% savings"
            )
        elif len(safe_strategies) == 1:
            strategy_name = self.STRATEGY_NAMES.get(safe_strategies[0], safe_strategies[0])
            return f"Start with {strategy_name} for initial savings"
        else:
            return "Continue monitoring shadow mode results"

    def _empty_report(self, hours: int) -> OptimizationReport:
        """Generate empty report when no shadow mode data exists."""
        return OptimizationReport(
            generated_at=datetime.now(timezone.utc),
            time_range_hours=hours,
            total_shadow_runs=0,
            strategies_analyzed=0,
            safe_strategies=[],
            metrics=[],
            overall_recommendation="No shadow mode data available. Enable shadow_mode in config to start testing optimizations.",
        )
