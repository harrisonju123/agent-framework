"""Cross-feature learning loop: connects feature outputs to feature inputs.

FeedbackBus is a synchronous event processor called at existing lifecycle hooks:
- Self-eval FAIL verdicts → memory (acceptance_gaps category)
- Replan successes → enriched memory (full approach chain)
- QA recurring findings → memory (qa_recurring) → prompt warnings
- Debate decisions → shared memory + specialization adjustment
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .memory_store import MemoryStore
    from ..core.profile_registry import ProfileRegistry
    from ..core.session_logger import SessionLogger
    from ..core.task import Task

logger = logging.getLogger(__name__)

# Domain keywords detected in debate synthesis → specialization adjustment
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "backend": ["api", "database", "server", "endpoint", "migration", "query", "grpc", "rest"],
    "frontend": ["component", "css", "react", "vue", "ui", "ux", "layout", "responsive"],
    "infrastructure": ["terraform", "docker", "kubernetes", "k8s", "ci/cd", "deploy", "helm", "infra"],
    "fullstack": ["frontend", "backend", "fullstack", "full-stack"],
}

# Cap QA warnings stored per repo to prevent memory bloat
MAX_QA_RECURRING_ENTRIES = 20
# Frequency threshold before a QA finding is considered "recurring"
QA_RECURRING_THRESHOLD = 2
# Max QA warnings injected into prompt
MAX_QA_WARNINGS_IN_PROMPT = 5
MAX_QA_WARNINGS_CHARS = 500

# Cap domain feedback adjustment to prevent drift
DOMAIN_FEEDBACK_MAX_ADJUSTMENT = 0.1
DOMAIN_FEEDBACK_MIN_SIGNALS = 3


class FeedbackBus:
    """Lightweight synchronous event processor connecting feature outputs to inputs.

    Called at existing lifecycle hooks — no async infrastructure needed.
    """

    def __init__(
        self,
        memory_store: "MemoryStore",
        session_logger: Optional["SessionLogger"] = None,
        profile_registry: Optional["ProfileRegistry"] = None,
    ):
        self._memory_store = memory_store
        self._session_logger = session_logger
        self._profile_registry = profile_registry

    def on_self_eval_fail(
        self,
        task: "Task",
        verdict: str,
        repo_slug: str,
        agent_type: str,
    ) -> None:
        """Persist self-eval FAIL patterns to memory under acceptance_gaps category.

        Extracts missed criteria keywords from the verdict so future tasks
        can avoid the same gaps.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        # Extract the gap description (everything after "FAIL")
        gap_text = verdict
        fail_idx = verdict.upper().find("FAIL")
        if fail_idx >= 0:
            gap_text = verdict[fail_idx + 4:].strip().lstrip(":").strip()

        if not gap_text:
            return

        # Truncate to reasonable length
        gap_text = gap_text[:500]

        # Build tags from file extensions in task context and task type
        tags = _extract_tags_from_task(task)

        content = f"Acceptance gap: {gap_text}"

        self._memory_store.remember(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="acceptance_gaps",
            content=content,
            source_task_id=task.id,
            tags=tags,
        )

        if self._session_logger:
            self._session_logger.log(
                "feedback_self_eval_stored",
                task_id=task.id,
                repo=repo_slug,
                gap_length=len(gap_text),
                tags=tags,
            )

        logger.debug("Stored self-eval acceptance gap for task %s", task.id)

    def on_replan_success(
        self,
        task: "Task",
        repo_slug: str,
        agent_type: str,
    ) -> None:
        """Store enriched replan outcome with full approach chain.

        Instead of a one-liner, stores all attempts tried and which one worked,
        giving future replans richer context.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        if not task.replan_history:
            return

        # Build structured multi-line entry with all attempts
        lines = []
        for i, entry in enumerate(task.replan_history):
            error_type = entry.get("error_type", "unknown")
            approach = entry.get("approach_tried", "not recorded")
            files = entry.get("files_involved", [])
            files_str = ", ".join(files[:3]) if files else "unknown"

            if i < len(task.replan_history) - 1:
                lines.append(f"  Attempt {i+1}: {approach} → failed ({error_type}) in {files_str}")
            else:
                # Last entry is the winning approach
                revised = entry.get("revised_plan", "")
                plan_summary = revised.split("\n")[0].strip("- *").strip()
                if not plan_summary:
                    plan_summary = revised[:100]
                lines.append(f"  Winning approach: {plan_summary}")

        last_entry = task.replan_history[-1]
        error_type = last_entry.get("error_type", "unknown")
        files = last_entry.get("files_involved", [])
        files_str = ", ".join(files[:3]) if files else "unknown"

        content = (
            f"Recovery from {error_type} in {files_str} "
            f"({len(task.replan_history)} attempts):\n" + "\n".join(lines)
        )

        tags = [error_type] if error_type != "unknown" else []

        self._memory_store.remember(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="past_failures",
            content=content,
            source_task_id=task.id,
            tags=tags,
        )

        if self._session_logger:
            self._session_logger.log(
                "feedback_replan_stored",
                task_id=task.id,
                repo=repo_slug,
                attempts=len(task.replan_history),
                error_type=error_type,
            )

        logger.debug(
            "Stored enriched replan outcome for task %s (%d attempts)",
            task.id, len(task.replan_history),
        )

    def on_qa_findings(
        self,
        task: "Task",
        findings: list,
        repo_slug: str,
        agent_type: str = "shared",
    ) -> None:
        """Aggregate QA findings and track recurring patterns across tasks.

        Findings that recur 2+ times are stored under qa_recurring category
        with frequency counts, later injected as warnings in engineer prompts.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        if not findings:
            return

        # Group findings by (category, file_pattern) for dedup
        stored_count = 0
        for finding in findings:
            category = getattr(finding, "category", "general")
            file_path = getattr(finding, "file", "")
            severity = getattr(finding, "severity", "UNKNOWN")
            description = getattr(finding, "description", "")

            if not description:
                continue

            # Normalize file to extension pattern for cross-task matching
            file_pattern = _file_to_pattern(file_path)

            # Check existing qa_recurring memories for this pattern
            existing = self._memory_store.recall(
                repo_slug=repo_slug,
                agent_type=agent_type,
                category="qa_recurring",
                limit=MAX_QA_RECURRING_ENTRIES,
            )

            # Look for matching existing entry to increment
            matched = False
            for mem in existing:
                if _qa_finding_matches(mem.content, category, file_pattern, description):
                    # Touch the existing memory to bump its frequency
                    # (MemoryStore.remember deduplicates by content)
                    matched = True
                    break

            content = (
                f"[{severity}] {category} in {file_pattern}: {description[:200]}"
            )
            tags = [category, severity.lower()]
            if file_pattern != "*":
                tags.append(file_pattern)

            self._memory_store.remember(
                repo_slug=repo_slug,
                agent_type=agent_type,
                category="qa_recurring",
                content=content,
                source_task_id=task.id,
                tags=tags,
            )
            stored_count += 1

        if stored_count > 0 and self._session_logger:
            self._session_logger.log(
                "feedback_qa_recurring_stored",
                task_id=task.id,
                repo=repo_slug,
                findings_count=len(findings),
                stored_count=stored_count,
            )

        logger.debug(
            "Stored %d QA findings for task %s", stored_count, task.id,
        )

    def on_debate_complete(
        self,
        debate_result: dict,
        repo_slug: str,
        task_id: Optional[str] = None,
        original_profile_id: Optional[str] = None,
    ) -> None:
        """Auto-persist debate decisions to shared memory and detect domain mismatches.

        Extracts topic/recommendation/confidence from debate JSON and stores
        under architectural_decisions. If debate synthesis reveals domain mismatch,
        records feedback in ProfileRegistry.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        synthesis = debate_result.get("synthesis", {})
        if not isinstance(synthesis, dict):
            return

        recommendation = synthesis.get("recommendation", "")
        if not recommendation:
            return

        topic = debate_result.get("topic", "unknown")
        confidence = synthesis.get("confidence", "unknown")
        trade_offs = synthesis.get("trade_offs", [])

        # Build memory content
        trade_offs_str = ""
        if trade_offs and isinstance(trade_offs, list):
            trade_offs_str = " Trade-offs: " + "; ".join(str(t)[:80] for t in trade_offs[:3])

        content = (
            f"Decision on '{topic}': {recommendation[:300]}"
            f" (confidence: {confidence}){trade_offs_str}"
        )

        # Tags from topic words
        tags = [w.lower() for w in topic.split() if len(w) > 3][:5]

        self._memory_store.remember(
            repo_slug=repo_slug,
            agent_type="shared",
            category="architectural_decisions",
            content=content,
            source_task_id=task_id,
            tags=tags,
        )

        if self._session_logger:
            self._session_logger.log(
                "feedback_debate_stored",
                task_id=task_id,
                repo=repo_slug,
                topic=topic[:100],
                confidence=confidence,
            )

        # Specialization adjustment: detect domain mismatch
        if self._profile_registry and original_profile_id:
            detected_domain = _detect_domain_from_text(
                f"{topic} {recommendation} {' '.join(str(t) for t in trade_offs)}"
            )
            if detected_domain and detected_domain != original_profile_id:
                self._profile_registry.record_domain_feedback(
                    task_id=task_id or "unknown",
                    detected_domain=detected_domain,
                    original_profile_id=original_profile_id,
                )

                if self._session_logger:
                    self._session_logger.log(
                        "feedback_domain_mismatch",
                        task_id=task_id,
                        repo=repo_slug,
                        detected_domain=detected_domain,
                        original_profile_id=original_profile_id,
                    )

                logger.info(
                    "Domain mismatch detected: debate suggests '%s' but profile was '%s'",
                    detected_domain, original_profile_id,
                )

        logger.debug("Stored debate decision for topic '%s'", topic)


def _extract_tags_from_task(task: "Task") -> List[str]:
    """Extract tags from task context: file extensions and task type."""
    tags = []

    # Task type
    if task.type:
        type_str = task.type.value if hasattr(task.type, "value") else str(task.type)
        tags.append(type_str)

    # File extensions from deliverables or files_to_modify
    files = (
        task.context.get("files_to_modify", [])
        or task.context.get("deliverables", [])
        or []
    )
    exts = set()
    for f in files:
        if isinstance(f, str):
            _, ext = os.path.splitext(f)
            if ext:
                exts.add(ext.lower())
    tags.extend(sorted(exts)[:5])

    return tags


def _file_to_pattern(file_path: str) -> str:
    """Convert a file path to an extension pattern for cross-task matching."""
    if not file_path:
        return "*"
    _, ext = os.path.splitext(file_path)
    return f"*{ext}" if ext else "*"


def _qa_finding_matches(
    memory_content: str,
    category: str,
    file_pattern: str,
    description: str,
) -> bool:
    """Check if a QA finding matches an existing memory entry."""
    # Match on category and file pattern being present in the memory
    if category not in memory_content:
        return False
    if file_pattern != "*" and file_pattern not in memory_content:
        return False
    # Fuzzy match: check if key words from description appear
    desc_words = {w.lower() for w in description.split() if len(w) > 4}
    if not desc_words:
        return False
    mem_lower = memory_content.lower()
    matches = sum(1 for w in desc_words if w in mem_lower)
    return matches >= max(1, len(desc_words) // 3)


def _detect_domain_from_text(text: str) -> Optional[str]:
    """Detect domain from text using keyword matching."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[domain] = count

    if not scores:
        return None

    # Return domain with highest keyword count
    best = max(scores, key=scores.get)
    # Require at least 2 keyword matches to avoid spurious detection
    if scores[best] < 2:
        return None
    return best
