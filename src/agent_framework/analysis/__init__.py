"""Analysis module for repository scanning and issue aggregation."""

from .aggregator import (
    Finding,
    FindingSeverity,
    FindingType,
    FileGroup,
    AnalysisResult,
    FindingAggregator,
)

__all__ = [
    "Finding",
    "FindingSeverity",
    "FindingType",
    "FileGroup",
    "AnalysisResult",
    "FindingAggregator",
]
