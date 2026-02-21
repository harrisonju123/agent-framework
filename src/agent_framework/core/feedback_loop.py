"""Cross-feature feedback bus connecting feature outputs to feature inputs.

Self-eval failures, replan successes, QA findings, and debate insights
are normalised into structured memories so downstream features (prompt
builder, specialization, replanning) can learn from past task outcomes.
"""

import hashlib
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..memory.memory_store import MemoryStore
    from .session_logger import SessionLogger

logger = logging.getLogger(__name__)

# Cap QA warnings section to prevent prompt bloat
QA_WARNINGS_MAX_CHARS = 1000


class FeedbackBus:
    """Routes cross-feature learning events into the memory store.

    Each method normalises one feature's output into a category/tag
    scheme that other features query at read time:

      self_eval_failures      → replanning + memory recall
      replan_success          → replanning memory context
      qa_recurring_findings   → prompt_builder QA warnings injection
      debate_specialization_hints → specialization profile adjustment
    """

    def __init__(
        self,
        memory_store: "MemoryStore",
        session_logger: Optional["SessionLogger"] = None,
    ):
        self._memory = memory_store
        self._session_logger = session_logger

    # ------------------------------------------------------------------
    # 1. Self-eval failures → memory
    # ------------------------------------------------------------------

    def store_self_eval_failure(
        self,
        repo_slug: str,
        agent_type: str,
        criteria_missed: List[str],
        task_id: str,
        critique: str = "",
    ) -> bool:
        """Persist which acceptance criteria were missed on a FAIL verdict.

        Stored under category ``self_eval_failures`` so replanning and
        future self-eval prompts can reference commonly missed criteria.
        """
        if not criteria_missed and not critique:
            return False

        if criteria_missed:
            content = "Missed criteria: " + "; ".join(criteria_missed)
        else:
            # Fallback: store raw critique when structured extraction fails
            content = f"Self-eval critique: {critique[:500]}"

        tags = ["self_eval", task_id]
        for criterion in criteria_missed[:5]:
            # Hash long criteria into short tags for dedup filtering
            tag = hashlib.md5(criterion.encode(), usedforsecurity=False).hexdigest()[:8]
            tags.append(f"criterion:{tag}")

        ok = self._memory.remember(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="self_eval_failures",
            content=content,
            source_task_id=task_id,
            tags=tags,
        )

        self._emit_log(
            "self_eval_failure",
            repo_slug=repo_slug,
            agent_type=agent_type,
            criteria_count=len(criteria_missed),
            task_id=task_id,
            stored=ok,
        )
        return ok

    # ------------------------------------------------------------------
    # 2. Replan successes → memory
    # ------------------------------------------------------------------

    def store_replan_success(
        self,
        repo_slug: str,
        agent_type: str,
        error_type: str,
        files_involved: List[str],
        revised_plan: str,
        task_id: str,
    ) -> bool:
        """Persist a successful recovery pattern with full approach context.

        Richer than the terse one-liner that ``store_replan_outcome``
        previously produced — includes error type, files, and the
        complete revised plan that resolved the issue.
        """
        files_str = ", ".join(files_involved[:5]) if files_involved else "unknown"
        content = (
            f"Error: {error_type}\n"
            f"Files: {files_str}\n"
            f"Resolution:\n{revised_plan[:800]}"
        )

        tags = ["replan_success", error_type]
        for f in files_involved[:3]:
            tags.append(f"file:{f}")

        ok = self._memory.remember(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="replan_success",
            content=content,
            source_task_id=task_id,
            tags=tags,
        )

        self._emit_log(
            "replan_success",
            repo_slug=repo_slug,
            agent_type=agent_type,
            error_type=error_type,
            files_count=len(files_involved),
            task_id=task_id,
            stored=ok,
        )
        return ok

    # ------------------------------------------------------------------
    # 3. QA recurring findings → memory
    # ------------------------------------------------------------------

    def store_qa_findings(
        self,
        repo_slug: str,
        agent_type: str,
        findings: List[Dict],
        task_id: str,
    ) -> int:
        """Persist QA findings as cross-task memories keyed by file pattern.

        Deduplicates by (category, file_pattern, description_hash) before
        storing to prevent memory bloat from repeated findings.

        Returns number of findings stored.
        """
        stored = 0
        seen: set[str] = set()

        for finding in findings:
            severity = finding.get("severity", "UNKNOWN")
            category = finding.get("category", "general")
            description = finding.get("description", "")
            file_path = finding.get("file", "unknown")

            # Dedup key: category + file pattern + description hash
            desc_hash = hashlib.md5(
                description.encode(), usedforsecurity=False
            ).hexdigest()[:8]
            dedup_key = f"{category}:{file_path}:{desc_hash}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            content = f"[{severity}] {file_path}: {description}"
            tags = [
                "qa_finding",
                f"severity:{severity.lower()}",
                f"category:{category}",
                task_id,
            ]

            ok = self._memory.remember(
                repo_slug=repo_slug,
                agent_type=agent_type,
                category="qa_recurring_findings",
                content=content,
                source_task_id=task_id,
                tags=tags,
            )
            if ok:
                stored += 1

        self._emit_log(
            "qa_findings",
            repo_slug=repo_slug,
            agent_type=agent_type,
            findings_total=len(findings),
            findings_stored=stored,
            task_id=task_id,
        )
        return stored

    # ------------------------------------------------------------------
    # 4. Debate insights → specialization hints
    # ------------------------------------------------------------------

    def store_debate_insight(
        self,
        repo_slug: str,
        agent_type: str,
        domain: str,
        topic: str,
        recommendation: str,
        reasoning: str,
        task_id: str = "",
    ) -> bool:
        """Persist a debate-derived specialization hint.

        Only called for high-confidence debates with domain-specific
        keywords. Stored under the ``shared`` agent namespace so all
        agents can query these hints.
        """
        content = (
            f"Domain: {domain}\n"
            f"Topic: {topic}\n"
            f"Recommendation: {recommendation}\n"
            f"Reasoning: {reasoning}"
        )

        tags = [
            "debate_insight",
            f"domain:{domain}",
            f"origin:{agent_type}",
        ]

        ok = self._memory.remember(
            repo_slug=repo_slug,
            agent_type="shared",
            category="debate_specialization_hints",
            content=content,
            source_task_id=task_id,
            tags=tags,
        )

        self._emit_log(
            "debate_insight",
            repo_slug=repo_slug,
            agent_type=agent_type,
            domain=domain,
            task_id=task_id,
            stored=ok,
        )
        return ok

    # ------------------------------------------------------------------
    # Session logging helper
    # ------------------------------------------------------------------

    def _emit_log(self, sub_event: str, **data) -> None:
        """Emit a ``feedback_loop_store`` session log event."""
        if self._session_logger:
            self._session_logger.log(
                "feedback_loop_store",
                feedback_type=sub_event,
                **data,
            )
