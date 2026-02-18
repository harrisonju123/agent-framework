"""
Analytics and metrics collection for the agent framework.

This package provides tools for:
- Performance metrics aggregation
- Failure pattern detection
- Shadow mode optimization analysis
- Agentic feature observability (memory, self-eval, replan, budget)
"""

from .performance_metrics import PerformanceMetrics
from .failure_analyzer import FailureAnalyzer
from .shadow_mode_analyzer import ShadowModeAnalyzer
from .agentic_metrics import AgenticMetrics

__all__ = [
    'PerformanceMetrics',
    'FailureAnalyzer',
    'ShadowModeAnalyzer',
    'AgenticMetrics',
]
