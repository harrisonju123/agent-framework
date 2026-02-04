"""Task validation to detect vague/underspecified tasks before processing.

Only validates internally-generated tasks (from agents like architect, product-owner).
JIRA tickets are skipped since they come from an external system with their own review process.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .task import Task, TaskType


@dataclass
class ValidationResult:
    """Result of task validation."""
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False  # True if validation was skipped (e.g., JIRA ticket)


# Vague patterns that indicate underspecified tasks
# Each tuple is (regex_pattern, message_template)
VAGUE_PATTERNS = [
    # Vague verb at end of title without target
    (r'\b(improve|enhance|optimize|refactor|update|fix|clean|handle)\s*$',
     "'{word}' at end of title without specific target"),

    # Vague verb + generic target
    (r'\b(improve|enhance|optimize)\s+(the\s+)?(code|system|performance|quality|handling|logic)\s*$',
     "'{match}' is too vague - specify what aspect to improve"),

    # Generic "error handling" without specifics
    (r'\b(improve|add|enhance|update)\s+(the\s+)?error\s+handling\s*$',
     "'{match}' needs specifics - which errors? which components?"),

    # "consistency" without specifics
    (r'\b(improve|ensure|add)\s+.*consistency\s*$',
     "'{match}' is vague - specify what should be consistent"),

    # Generic "logging" without specifics
    (r'\b(improve|add|enhance)\s+(the\s+)?logging\s*$',
     "'{match}' needs specifics - which components? what level?"),
]


def validate_task(task: Task, mode: str = "warn") -> ValidationResult:
    """
    Validate task for vagueness and underspecification.

    Args:
        task: The task to validate
        mode: "warn" logs warnings but allows task, "reject" fails vague tasks

    Returns:
        ValidationResult with is_valid, warnings, errors, and skipped flag

    Skips validation for:
    - JIRA tickets (have jira_key in context or created_by == "jira")
    - CLI-created tasks (created_by == "cli") - user explicitly created these
    """
    # Skip validation for external tickets (JIRA, etc.)
    if task.context.get("jira_key") or task.created_by == "jira":
        return ValidationResult(is_valid=True, skipped=True)

    # Skip validation for CLI-created tasks (user explicitly created these)
    if task.created_by == "cli":
        return ValidationResult(is_valid=True, skipped=True)

    issues: list[str] = []

    # Check 1: Title matches vague patterns
    for pattern, message in VAGUE_PATTERNS:
        match = re.search(pattern, task.title, re.IGNORECASE)
        if match:
            # Build message with matched text
            msg = message.format(
                word=match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0),
                match=match.group(0).strip()
            )
            issues.append(msg)

    # Check 2: Very short title with no description context
    # Remove any JIRA-key prefix like "[PROJ-123] " before checking length
    clean_title = re.sub(r'^\[[A-Z]+-\d+\]\s*', '', task.title)
    if len(clean_title) < 15 and len(task.description) < 50:
        issues.append("Title and description too brief - add specific details")

    # Check 3: Agent-generated IMPLEMENTATION tasks should have acceptance_criteria
    impl_types = {TaskType.IMPLEMENTATION, TaskType.ARCHITECTURE, TaskType.ENHANCEMENT}
    agent_creators = {"architect", "product-owner", "planner"}
    if task.type in impl_types and not task.acceptance_criteria:
        if task.created_by in agent_creators:
            issues.append("Agent-generated implementation task should have acceptance_criteria")

    # Check 4: Tasks with "and" in title might be too broad
    if " and " in task.title.lower():
        words_before_and = task.title.lower().split(" and ")[0].split()
        words_after_and = task.title.lower().split(" and ")[1].split()
        # Only flag if both sides have substantial content (not just "X and Y")
        if len(words_before_and) > 2 and len(words_after_and) > 2:
            issues.append("Task may be too broad - consider splitting into separate tasks")

    # Build result based on mode
    if mode == "reject":
        return ValidationResult(
            is_valid=len(issues) == 0,
            errors=issues,
        )
    else:  # mode == "warn"
        return ValidationResult(
            is_valid=True,
            warnings=issues,
        )
