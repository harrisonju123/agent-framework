"""
Analytics and metrics collection for the agent framework.

This package provides tools for:
- Performance metrics aggregation
- Failure pattern detection
- Shadow mode optimization analysis
- Quality trend tracking
"""

from .performance_metrics import PerformanceMetrics
from .failure_analyzer import FailureAnalyzer
from .shadow_mode_analyzer import ShadowModeAnalyzer
from .budget_optimizer import BudgetOptimizer
from .quality_analyzer import QualityAnalyzer

__all__ = [
    'PerformanceMetrics',
    'FailureAnalyzer',
    'ShadowModeAnalyzer',
    'BudgetOptimizer',
    'QualityAnalyzer',
]
