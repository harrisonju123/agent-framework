"""QA pattern aggregator for detecting recurring findings across tasks.

Scans memory entries with category='qa_findings' to find patterns that
recur across different tasks (same file path, same severity, same category
appearing 3+ times). Returns top-N patterns as warning strings for injection
into engineer prompts.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Threshold: a finding pattern must appear in at least this many distinct tasks
RECURRENCE_THRESHOLD = 3
# Cap on how many warnings to return
MAX_WARNINGS = 5
# Cap on total chars in the warnings section
MAX_WARNINGS_CHARS = 500


@dataclass
class RecurringPattern:
    """A QA finding pattern that recurs across multiple tasks."""
    description: str
    occurrence_count: int
    severity: str
    category: str
    # File paths where this pattern appeared
    file_paths: List[str]


class QAPatternAggregator:
    """Detects recurring QA findings across tasks from memory.

    Queries the MemoryStore for qa_findings entries and groups them
    by (severity, category, description_prefix) to find patterns.
    """

    def __init__(self, memory_store: Optional["MemoryStore"] = None):
        self._memory_store = memory_store

    def get_recurring_patterns(
        self,
        repo_slug: str,
        agent_type: str = "qa",
        relevant_files: Optional[Set[str]] = None,
    ) -> List[RecurringPattern]:
        """Find QA findings that recur across multiple tasks.

        Args:
            repo_slug: Repository identifier.
            agent_type: Agent type to query memories for.
            relevant_files: If provided, only return patterns for these files.

        Returns:
            List of RecurringPattern sorted by occurrence count (descending).
        """
        if not self._memory_store or not self._memory_store.enabled:
            return []

        memories = self._memory_store.recall(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="qa_findings",
            limit=100,
        )

        if not memories:
            return []

        # Group by normalized description to detect recurrence.
        # Memory content format: "QA finding: {severity} {category} in {file}: {description}"
        pattern_counts: Counter = Counter()
        pattern_details: Dict[str, dict] = {}

        for mem in memories:
            key = self._normalize_key(mem.content)
            if not key:
                continue

            pattern_counts[key] += 1

            if key not in pattern_details:
                # Parse severity and category from tags
                severity = ""
                category = ""
                file_path = ""
                for tag in mem.tags:
                    if tag.upper() in ("CRITICAL", "HIGH", "MAJOR", "MEDIUM", "LOW", "MINOR", "SUGGESTION"):
                        severity = tag.upper()
                    elif tag.startswith("file:"):
                        file_path = tag[5:]
                    elif not severity:
                        category = tag

                pattern_details[key] = {
                    "description": mem.content,
                    "severity": severity or "UNKNOWN",
                    "category": category or "general",
                    "file_paths": [],
                }

            if mem.tags:
                for tag in mem.tags:
                    if tag.startswith("file:"):
                        fp = tag[5:]
                        if fp not in pattern_details[key]["file_paths"]:
                            pattern_details[key]["file_paths"].append(fp)

        # Filter to patterns exceeding the recurrence threshold
        recurring = []
        for key, count in pattern_counts.most_common():
            if count < RECURRENCE_THRESHOLD:
                break

            details = pattern_details[key]

            # If relevant_files is provided, only include patterns for those files
            if relevant_files and details["file_paths"]:
                if not any(self._file_matches(fp, relevant_files) for fp in details["file_paths"]):
                    continue

            recurring.append(RecurringPattern(
                description=details["description"],
                occurrence_count=count,
                severity=details["severity"],
                category=details["category"],
                file_paths=details["file_paths"],
            ))

            if len(recurring) >= MAX_WARNINGS:
                break

        return recurring

    def format_warnings(self, patterns: List[RecurringPattern]) -> str:
        """Format recurring patterns as a prompt warning section.

        Returns empty string if no patterns, otherwise a section capped at
        MAX_WARNINGS_CHARS.
        """
        if not patterns:
            return ""

        lines = ["## RECURRING QA WARNINGS",
                  "These issues have been flagged repeatedly across previous tasks. Proactively avoid them:\n"]

        for i, p in enumerate(patterns, 1):
            files_hint = f" (in {', '.join(p.file_paths[:3])})" if p.file_paths else ""
            lines.append(f"{i}. [{p.severity}] {p.description[:100]}{files_hint} — seen {p.occurrence_count}x")

        lines.append("")
        result = "\n".join(lines)

        if len(result) > MAX_WARNINGS_CHARS:
            result = result[:MAX_WARNINGS_CHARS - 4] + "\n..."

        return result

    @staticmethod
    def _normalize_key(content: str) -> str:
        """Normalize finding content to a grouping key.

        Strips variable parts (task IDs, line numbers) to group similar findings.
        """
        if not content:
            return ""
        # Use first 80 chars of content as grouping key (stable across minor variations)
        normalized = content.strip().lower()[:80]
        return normalized

    @staticmethod
    def _file_matches(file_path: str, relevant_files: Set[str]) -> bool:
        """Check if a file path matches any in the relevant set (suffix match)."""
        if file_path in relevant_files:
            return True
        for rf in relevant_files:
            if file_path.endswith("/" + rf) or rf.endswith("/" + file_path):
                return True
        return False
