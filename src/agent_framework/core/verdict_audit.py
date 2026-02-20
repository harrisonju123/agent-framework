"""Verdict audit trail — structured diagnostics for every verdict determination.

When pattern matching false-positives fire (e.g. "already exist" triggering
no_changes, or negated phrases like "No issues found" matching rejection),
the audit trail captures HOW the verdict was reached: which patterns matched,
which were suppressed by negation context, and which code path was taken.

Separate from ReviewOutcome because audit trace is a different concern than
parse result — ReviewOutcome drives routing, VerdictAudit drives diagnostics.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PatternMatch:
    """A single regex pattern match during review outcome parsing."""

    category: str           # "approve", "critical_issues", etc.
    pattern: str            # regex string that matched
    matched_text: str       # actual text matched (truncated to 50 chars)
    position: int           # char offset in content where match started
    suppressed_by_negation: bool  # True if negation prefix ("no ", "zero ", etc.) nullified this match


@dataclass
class VerdictAudit:
    """Full audit trail for a single verdict determination.

    Captures the reasoning chain so false positives can be diagnosed
    from session logs without reproducing the exact LLM output.
    """

    method: str             # "review_outcome" | "no_changes_marker" | "ambiguous_halt" | "ambiguous_default"
    value: Optional[str]    # final verdict string or None if not set
    agent_id: str
    workflow_step: str
    task_id: str

    # ReviewOutcome flags at the point of verdict
    outcome_flags: Optional[Dict[str, bool]] = None  # {approved, has_critical, has_major, ...}

    # All pattern matches (both successful and negation-suppressed)
    matched_patterns: List[PatternMatch] = field(default_factory=list)
    negation_suppressed: List[PatternMatch] = field(default_factory=list)

    # Override decisions
    override_applied: bool = False          # CRITICAL/MAJOR/test_fail overrode APPROVE
    severity_tag_default_deny: bool = False  # _SEVERITY_TAG_RE default-deny rule fired
    no_changes_marker_found: bool = False

    content_snippet: str = ""               # first 200 chars of content for context

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for session logging and task context storage."""
        return asdict(self)
