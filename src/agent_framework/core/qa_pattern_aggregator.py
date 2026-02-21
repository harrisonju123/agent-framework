"""QA pattern aggregator: detects recurring findings across tasks.

Scans memory entries with category='qa_findings' to find patterns that
appear 3+ times across different tasks. Returns top-N patterns as
warning strings for injection into engineer prompts.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Findings that recur this many times across different tasks are flagged
RECURRENCE_THRESHOLD = 3

# Cap the warnings section to avoid prompt bloat
MAX_WARNING_CHARS = 500
MAX_WARNINGS = 5


@dataclass
class RecurringPattern:
    """A QA finding pattern that recurs across multiple tasks."""
    category: str       # e.g. "security", "performance"
    description: str    # Normalized description
    severity: str       # Most severe occurrence
    file_pattern: str   # Common file path or pattern
    occurrence_count: int
    task_ids: List[str]


class QAPatternAggregator:
    """Aggregates QA findings from memory to detect recurring patterns.

    Reads qa_findings entries from MemoryStore and groups them by
    (category, normalized_description) to find patterns appearing
    across 3+ different tasks.
    """

    def __init__(self, memory_store, repo_slug: str):
        self._memory_store = memory_store
        self._repo_slug = repo_slug

    def get_recurring_patterns(
        self,
        agent_type: str = "shared",
        threshold: int = RECURRENCE_THRESHOLD,
        limit: int = MAX_WARNINGS,
    ) -> List[RecurringPattern]:
        """Find QA findings that recur across multiple tasks.

        Args:
            agent_type: Memory namespace to query (default "shared").
            threshold: Min occurrences to flag as recurring.
            limit: Max patterns to return.

        Returns:
            List of RecurringPattern sorted by occurrence count descending.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return []

        entries = self._memory_store.recall(
            repo_slug=self._repo_slug,
            agent_type=agent_type,
            category="qa_findings",
            limit=200,
        )

        if not entries:
            return []

        # Group by (category_tag, normalized_description)
        groups: Dict[str, dict] = {}
        for entry in entries:
            key = self._normalize_finding(entry.content)
            if key not in groups:
                groups[key] = {
                    "contents": [],
                    "task_ids": set(),
                    "severities": [],
                    "categories": [],
                    "files": [],
                }
            g = groups[key]
            g["contents"].append(entry.content)
            if entry.source_task_id:
                g["task_ids"].add(entry.source_task_id)
            # Parse structured info from tags
            for tag in entry.tags:
                if tag.startswith("severity:"):
                    g["severities"].append(tag.split(":", 1)[1])
                elif tag.startswith("category:"):
                    g["categories"].append(tag.split(":", 1)[1])
                elif tag.startswith("file:"):
                    g["files"].append(tag.split(":", 1)[1])

        # Filter to patterns appearing in threshold+ different tasks
        patterns = []
        for key, g in groups.items():
            task_count = len(g["task_ids"])
            if task_count < threshold:
                continue

            severity = self._highest_severity(g["severities"]) if g["severities"] else "MEDIUM"
            category = Counter(g["categories"]).most_common(1)[0][0] if g["categories"] else "general"
            file_pattern = Counter(g["files"]).most_common(1)[0][0] if g["files"] else ""

            patterns.append(RecurringPattern(
                category=category,
                description=key,
                severity=severity,
                file_pattern=file_pattern,
                occurrence_count=task_count,
                task_ids=sorted(g["task_ids"])[:5],
            ))

        # Sort by occurrence count descending, then severity
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MAJOR": 2, "MEDIUM": 3, "LOW": 4}
        patterns.sort(key=lambda p: (-p.occurrence_count, severity_rank.get(p.severity, 5)))

        return patterns[:limit]

    def get_warnings_for_files(
        self,
        target_files: List[str],
        agent_type: str = "shared",
    ) -> List[RecurringPattern]:
        """Get recurring patterns relevant to specific files.

        Filters patterns to those whose file_pattern matches any of
        the target files (suffix match).
        """
        all_patterns = self.get_recurring_patterns(agent_type=agent_type)
        if not all_patterns or not target_files:
            return all_patterns

        relevant = []
        for pattern in all_patterns:
            if not pattern.file_pattern:
                # Patterns without file info are always relevant
                relevant.append(pattern)
                continue
            for target in target_files:
                if (target.endswith(pattern.file_pattern) or
                        pattern.file_pattern.endswith(target) or
                        pattern.file_pattern in target):
                    relevant.append(pattern)
                    break

        return relevant

    def format_warnings_section(
        self,
        patterns: List[RecurringPattern],
        max_chars: int = MAX_WARNING_CHARS,
    ) -> str:
        """Format recurring patterns as a prompt section.

        Returns empty string if no patterns.
        """
        if not patterns:
            return ""

        lines = ["## RECURRING QA WARNINGS", ""]
        lines.append(
            "These issues have been found repeatedly across previous tasks. "
            "Check your implementation for these patterns:"
        )
        lines.append("")

        for i, p in enumerate(patterns, 1):
            location = f" ({p.file_pattern})" if p.file_pattern else ""
            lines.append(
                f"{i}. [{p.severity}] {p.category}{location}: {p.description} "
                f"(seen {p.occurrence_count}x)"
            )

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars - 15] + "\n[truncated]\n"

        return result

    @staticmethod
    def _normalize_finding(content: str) -> str:
        """Normalize a finding description for grouping.

        Strips line numbers, file paths, and other variable parts to
        group semantically similar findings together.
        """
        import re
        # Strip line numbers like ":123"
        normalized = re.sub(r':\d+', '', content)
        # Strip specific file paths, keeping just the description
        normalized = re.sub(r'\S+/\S+\.\w+', '<file>', normalized)
        # Collapse whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        # Cap length for grouping key
        return normalized[:200]

    @staticmethod
    def _highest_severity(severities: List[str]) -> str:
        """Return the highest severity from a list."""
        rank = {"CRITICAL": 0, "HIGH": 1, "MAJOR": 2, "MEDIUM": 3, "MINOR": 4, "LOW": 5, "SUGGESTION": 6}
        if not severities:
            return "MEDIUM"
        return min(severities, key=lambda s: rank.get(s, 5))
