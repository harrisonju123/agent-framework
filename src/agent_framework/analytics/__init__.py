"""
Analytics and metrics collection for the agent framework.

This package provides tools for:
- Performance metrics aggregation
- Failure pattern detection
- Shadow mode optimization analysis
"""

from .performance_metrics import PerformanceMetrics
from .failure_analyzer import FailureAnalyzer
from .shadow_mode_analyzer import ShadowModeAnalyzer
from .agentic_metrics import AgenticMetrics, AgenticMetricsReport
from .llm_metrics import LlmMetrics, LlmMetricsReport
from .decomposition_metrics import DecompositionMetrics, DecompositionReport
from .chain_metrics import ChainMetrics, ChainMetricsReport
from .git_metrics import GitMetrics, GitMetricsReport
from .waste_metrics import WasteMetrics, WasteMetricsReport
from .review_cycle_metrics import ReviewCycleAnalyzer, ReviewCycleMetricsReport
from .verdict_metrics import VerdictMetrics, VerdictMetricsReport
from .feedback_loop_metrics import FeedbackLoopAnalyzer, FeedbackLoopReport

__all__ = [
    'PerformanceMetrics',
    'FailureAnalyzer',
    'ShadowModeAnalyzer',
    'AgenticMetrics',
    'AgenticMetricsReport',
    'LlmMetrics',
    'LlmMetricsReport',
    'DecompositionMetrics',
    'DecompositionReport',
    'ChainMetrics',
    'ChainMetricsReport',
    'GitMetrics',
    'GitMetricsReport',
    'WasteMetrics',
    'WasteMetricsReport',
    'ReviewCycleAnalyzer',
    'ReviewCycleMetricsReport',
    'VerdictMetrics',
    'VerdictMetricsReport',
    'FeedbackLoopAnalyzer',
    'FeedbackLoopReport',
]
