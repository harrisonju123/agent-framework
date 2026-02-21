"""Cross-feature learning loop: routes post-task learnings to persistent stores.

Connects feature outputs to feature inputs:
- Self-eval failures -> memory (commonly missed acceptance criteria)
- Replan successes -> memory (delegates to ErrorRecoveryManager)
- QA recurring findings -> shared memory (cross-task pattern detection)
- Debate decisions -> specialization hints
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .error_recovery import ErrorRecoveryManager
    from .session_logger import SessionLogger

from ..memory.memory_store import MemoryStore
from .task import Task, TaskStatus

logger = logging.getLogger(__name__)

# Category constants for memory store entries
CATEGORY_SELF_EVAL_GAPS = "self_eval_gaps"
CATEGORY_QA_RECURRING = "qa_recurring_warnings"
CATEGORY_SPECIALIZATION_HINT = "specialization_hint"

# "shared" namespace stores cross-agent memories visible to all agents
SHARED_AGENT_TYPE = "shared"

# Require this many occurrences across distinct tasks before flagging as recurring
QA_RECURRENCE_THRESHOLD = 2

# Domain keywords that indicate a debate outcome is relevant to specialization
_SPECIALIZATION_KEYWORDS = frozenset({
    "technology", "framework", "language", "library", "database",
    "architecture", "pattern", "approach", "stack", "tooling",
})


class FeedbackBus:
    """Post-task learning coordinator.

    Runs after task completion to extract cross-feature learnings and
    persist them to the appropriate stores. Each collector fires only
    when the task context contains relevant data.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        session_logger: Optional["SessionLogger"] = None,
        error_recovery: Optional["ErrorRecoveryManager"] = None,
    ):
        self._memory_store = memory_store
        self._session_logger = session_logger
        self._error_recovery = error_recovery

    def process(self, task: Task, repo_slug: str) -> None:
        """Run all applicable collectors for a completed task."""
        if not self._memory_store.enabled:
            return

        self._store_self_eval_failure(task, repo_slug)
        self._store_replan_success(task, repo_slug)
        self._aggregate_qa_findings(task, repo_slug)
        self._update_specialization_from_debate(task, repo_slug)

    # -- Collectors --

    def _store_self_eval_failure(self, task: Task, repo_slug: str) -> None:
        """Persist missed acceptance criteria from failed self-evaluations.

        Only fires when the task went through self-eval retry (has critique)
        AND still completed — meaning the agent eventually passed but the
        initial failures reveal gaps worth remembering.
        """
        critique = task.context.get("_self_eval_critique")
        if not critique:
            return

        # Extract which criteria were missed from the critique text
        criteria = task.acceptance_criteria or []
        missed = _extract_missed_criteria(critique, criteria)

        if not missed:
            # Store the raw critique as a general gap if we can't parse specifics
            content = f"Self-eval failed: {critique[:500]}"
            self._memory_store.remember(
                repo_slug=repo_slug,
                agent_type=task.assigned_to or "engineer",
                category=CATEGORY_SELF_EVAL_GAPS,
                content=content,
                source_task_id=task.id,
                tags=["self_eval"],
            )
        else:
            for criterion in missed:
                self._memory_store.remember(
                    repo_slug=repo_slug,
                    agent_type=task.assigned_to or "engineer",
                    category=CATEGORY_SELF_EVAL_GAPS,
                    content=f"Commonly missed criterion: {criterion}",
                    source_task_id=task.id,
                    tags=["self_eval", "missed_criterion"],
                )

        if self._session_logger:
            self._session_logger.log(
                "feedback_bus_self_eval_stored",
                repo=repo_slug,
                missed_count=len(missed) if missed else 1,
                task_id=task.id,
            )

    def _store_replan_success(self, task: Task, repo_slug: str) -> None:
        """Delegate successful replan persistence to ErrorRecoveryManager.

        Centralizes all cross-feature learning through FeedbackBus so callers
        only need bus.process() instead of separate error_recovery calls.
        """
        if not self._error_recovery:
            return
        if not task.replan_history or task.status != TaskStatus.COMPLETED:
            return

        self._error_recovery.store_replan_outcome(task, repo_slug)

    def _aggregate_qa_findings(self, task: Task, repo_slug: str) -> None:
        """Detect QA findings recurring across distinct tasks and promote to shared memory.

        Checks structured_findings from the current task against existing
        qa_recurring_warnings. If a finding description already exists in memory
        from a different task, it has recurred — store/touch it in the shared namespace.
        """
        structured = task.context.get("structured_findings")
        if not structured or not isinstance(structured, dict):
            return

        findings_list = structured.get("findings", [])
        if not findings_list:
            return

        # Load existing QA recurring warnings to check for matches
        existing = self._memory_store.recall(
            repo_slug=repo_slug,
            agent_type=SHARED_AGENT_TYPE,
            category=CATEGORY_QA_RECURRING,
            limit=200,
        )
        existing_contents = {e.content for e in existing}

        # Also load per-agent QA findings to detect cross-task recurrence
        agent_type = task.assigned_to or "engineer"
        agent_findings = self._memory_store.recall(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category=CATEGORY_QA_RECURRING,
            limit=200,
        )

        # Build lookup: finding description -> set of source_task_ids
        finding_task_map: Dict[str, set] = {}
        for entry in agent_findings:
            finding_task_map.setdefault(entry.content, set())
            if entry.source_task_id:
                finding_task_map[entry.content].add(entry.source_task_id)

        promoted_count = 0
        for finding in findings_list:
            desc = finding.get("description", "").strip()
            if not desc:
                continue

            normalized = f"QA finding: {desc}"

            # Always store per-agent so we can track recurrence
            self._memory_store.remember(
                repo_slug=repo_slug,
                agent_type=agent_type,
                category=CATEGORY_QA_RECURRING,
                content=normalized,
                source_task_id=task.id,
                tags=_extract_file_tags(finding),
            )

            # Check recurrence: appeared in a different task before?
            prior_tasks = finding_task_map.get(normalized, set())
            prior_tasks.discard(task.id)  # Don't count current task

            if len(prior_tasks) >= (QA_RECURRENCE_THRESHOLD - 1):
                # Promote to shared namespace so all agents see it
                if normalized not in existing_contents:
                    file_path = finding.get("file", "")
                    tags = ["qa_recurring"]
                    if file_path:
                        tags.append(file_path)

                    self._memory_store.remember(
                        repo_slug=repo_slug,
                        agent_type=SHARED_AGENT_TYPE,
                        category=CATEGORY_QA_RECURRING,
                        content=normalized,
                        source_task_id=task.id,
                        tags=tags,
                    )
                    promoted_count += 1

        if promoted_count > 0 and self._session_logger:
            self._session_logger.log(
                "feedback_bus_qa_recurring_detected",
                repo=repo_slug,
                promoted_count=promoted_count,
                total_findings=len(findings_list),
                task_id=task.id,
            )

    def _update_specialization_from_debate(self, task: Task, repo_slug: str) -> None:
        """Store high-confidence debate outcomes as specialization hints.

        Only fires when the task context contains debate_result (set by the
        debate MCP tool) and the topic relates to domain expertise.
        """
        debate_result = task.context.get("debate_result")
        if not debate_result or not isinstance(debate_result, dict):
            return

        confidence = debate_result.get("confidence", "").lower()
        if confidence != "high":
            return

        topic = debate_result.get("topic", "")
        recommendation = debate_result.get("recommendation", "")
        if not topic or not recommendation:
            return

        # Only store if the debate topic relates to specialization domains
        topic_lower = topic.lower()
        if not any(kw in topic_lower for kw in _SPECIALIZATION_KEYWORDS):
            return

        content = f"Debate: {topic[:200]} -> {recommendation[:300]}"
        self._memory_store.remember(
            repo_slug=repo_slug,
            agent_type=SHARED_AGENT_TYPE,
            category=CATEGORY_SPECIALIZATION_HINT,
            content=content,
            source_task_id=task.id,
            tags=["debate", "specialization"],
        )

        if self._session_logger:
            self._session_logger.log(
                "feedback_bus_specialization_updated",
                repo=repo_slug,
                topic=topic[:200],
                confidence=confidence,
                task_id=task.id,
            )


# -- Helpers --

def _extract_missed_criteria(critique: str, criteria: List[str]) -> List[str]:
    """Match self-eval critique text against acceptance criteria.

    Returns criteria that appear to be referenced in the failure critique.
    Uses case-insensitive substring matching against the criteria list.
    """
    if not criteria:
        return []

    critique_lower = critique.lower()
    missed = []
    for criterion in criteria:
        # Check if meaningful words from the criterion appear in the critique
        words = [w for w in criterion.lower().split() if len(w) > 3]
        if not words:
            continue
        match_count = sum(1 for w in words if w in critique_lower)
        if match_count >= max(1, len(words) // 2):
            missed.append(criterion)

    return missed


def _extract_file_tags(finding: Dict[str, Any]) -> List[str]:
    """Extract file-related tags from a QA finding."""
    tags = ["qa_finding"]
    file_path = finding.get("file", "")
    if file_path:
        tags.append(file_path)
    return tags
