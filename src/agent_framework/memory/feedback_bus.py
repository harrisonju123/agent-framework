"""Cross-feature learning loop: connects feature outputs to feature inputs.

The FeedbackBus listens to events from self-eval, replan, QA, and debate
subsystems and translates their outputs into memory entries, prompt warnings,
and specialization corrections.

Feedback channels:
- Self-eval failures -> memory (missed acceptance criteria patterns)
- Replan successes  -> memory (error->resolution recovery patterns)
- QA findings       -> memory (recurring finding patterns as warnings)
- Debate decisions  -> specialization (domain mismatch -> profile adjustment)
"""

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.profile_registry import ProfileRegistry

from .memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Minimum occurrences of same (category, severity, file_pattern) before
# a QA finding is promoted to a recurring warning stored in memory.
_QA_RECURRENCE_THRESHOLD = 2

# Max QA warnings stored per repo — prevents memory bloat
_MAX_QA_WARNINGS = 20


def _extract_tags_from_task(task) -> List[str]:
    """Build tag list from task context for memory entries."""
    tags = []
    if task.type:
        from ..utils.type_helpers import get_type_str
        tags.append(get_type_str(task.type))
    repo = task.context.get("github_repo")
    if repo:
        tags.append(repo.split("/")[-1])
    jira_project = task.context.get("jira_project")
    if jira_project:
        tags.append(jira_project)
    return tags


def _file_to_pattern(file_path: str) -> str:
    """Generalize a file path to a glob pattern for grouping.

    Examples:
        src/foo/bar.py     -> **/*.py
        tests/unit/test_x.py -> **/tests/**/*.py
        src/api/handler.go -> **/*.go
    """
    if not file_path:
        return "**/*"

    _, ext = os.path.splitext(file_path)
    ext = ext.lower() if ext else ""

    # Keep test path structure for test files
    parts = file_path.replace("\\", "/").split("/")
    if any(p in ("tests", "test", "__tests__", "spec") for p in parts):
        return f"**/tests/**/*{ext}" if ext else "**/tests/**/*"

    return f"**/*{ext}" if ext else "**/*"


def _qa_finding_matches(a: Dict[str, str], b: Dict[str, str]) -> bool:
    """Check if two QA findings match on (category, severity, file_pattern)."""
    return (
        a.get("category") == b.get("category")
        and a.get("severity") == b.get("severity")
        and a.get("file_pattern") == b.get("file_pattern")
    )


def _detect_domain_from_keywords(text: str) -> List[str]:
    """Extract domain-specific keywords from debate/critique text.

    Returns lowercase keyword list for comparison against profile tags.
    """
    # Domain keyword patterns — these signal specific technical domains
    domain_patterns = {
        "database": r"\b(?:sql|database|migration|schema|query|orm|postgres|mysql|mongo)\b",
        "frontend": r"\b(?:react|vue|angular|css|html|dom|component|browser|webpack)\b",
        "backend": r"\b(?:api|rest|grpc|graphql|endpoint|middleware|server|handler)\b",
        "infrastructure": r"\b(?:docker|kubernetes|k8s|terraform|ci/cd|deploy|helm|aws|gcp)\b",
        "security": r"\b(?:auth|oauth|jwt|encryption|csrf|xss|injection|certificate)\b",
        "testing": r"\b(?:unit test|integration test|e2e|selenium|pytest|jest|mock)\b",
    }
    found = []
    text_lower = text.lower()
    for domain, pattern in domain_patterns.items():
        if re.search(pattern, text_lower):
            found.append(domain)
    return found


class FeedbackBus:
    """Connects feature outputs to feature inputs for cross-task learning.

    Instantiated by Agent.__init__ and called from post-completion hooks.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        agent_type: str,
        profile_registry: Optional["ProfileRegistry"] = None,
    ):
        self._memory_store = memory_store
        self._agent_type = agent_type
        self._profile_registry = profile_registry

    def on_self_eval_fail(self, task, critique: str) -> None:
        """Store missed acceptance criteria patterns from self-eval failures.

        Called after self_evaluate() returns False. Parses the critique to
        extract which criteria were missed and stores them under 'missed_criteria'
        so the prompt builder can inject proactive warnings for future tasks.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return

        # Extract individual missed criteria from the critique
        missed = self._parse_missed_criteria(critique, task)
        if not missed:
            # Fall back to storing the raw critique summary
            missed = [critique[:500]]

        tags = _extract_tags_from_task(task)

        for criterion in missed:
            self._memory_store.remember(
                repo_slug=repo_slug,
                agent_type=self._agent_type,
                category="missed_criteria",
                content=criterion,
                source_task_id=task.id,
                tags=tags,
            )

        logger.debug(
            "FeedbackBus: stored %d missed criteria from self-eval failure for %s",
            len(missed), task.id,
        )

    def on_replan_success(self, task, repo_slug: str) -> None:
        """Store enriched recovery patterns from successful replanning.

        Extends the existing store_replan_outcome by also storing under
        'recovery_patterns' with file extension tags for broader matching.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        if not task.replan_history:
            return

        last_entry = task.replan_history[-1]
        error_type = last_entry.get("error_type", "unknown")
        files = last_entry.get("files_involved", [])
        revised_plan = last_entry.get("revised_plan", "")

        plan_summary = revised_plan.split("\n")[0].strip("- *").strip()
        if not plan_summary:
            plan_summary = revised_plan[:100]

        # Enrich tags with file extensions for specialization matching
        tags = [error_type]
        ext_set = set()
        for f in files:
            _, ext = os.path.splitext(f)
            if ext:
                ext_set.add(ext.lower())
        tags.extend(sorted(ext_set))

        files_str = ", ".join(files[:3]) if files else "unknown files"
        content = f"{error_type} in {files_str}: {plan_summary} -> resolved"

        self._memory_store.remember(
            repo_slug=repo_slug,
            agent_type=self._agent_type,
            category="recovery_patterns",
            content=content,
            source_task_id=task.id,
            tags=tags,
        )

        logger.debug(
            "FeedbackBus: stored recovery pattern for %s (error_type=%s)",
            task.id, error_type,
        )

    def on_qa_findings(self, task, findings: List[Dict[str, Any]]) -> None:
        """Detect recurring QA finding patterns and store as warnings.

        Groups findings by (file_pattern, category, severity). Patterns
        seen >= _QA_RECURRENCE_THRESHOLD times are stored as 'qa_warnings'
        in memory for injection into future engineer prompts.
        """
        if not self._memory_store or not self._memory_store.enabled:
            return

        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return

        if not findings:
            return

        # Normalize findings to dicts with file_pattern
        normalized = []
        for f in findings:
            if hasattr(f, "__dict__"):
                # QAFinding dataclass
                d = {
                    "file": getattr(f, "file", ""),
                    "severity": getattr(f, "severity", "UNKNOWN"),
                    "category": getattr(f, "category", "unknown"),
                    "description": getattr(f, "description", ""),
                }
            else:
                d = dict(f)

            d["file_pattern"] = _file_to_pattern(d.get("file", ""))
            normalized.append(d)

        # Count occurrences of (file_pattern, category, severity) groups
        group_key = lambda d: (d["file_pattern"], d.get("category", ""), d.get("severity", ""))
        counter: Counter = Counter()
        examples: Dict[tuple, str] = {}
        for d in normalized:
            key = group_key(d)
            counter[key] += 1
            if key not in examples:
                examples[key] = d.get("description", "")[:200]

        tags = _extract_tags_from_task(task)
        stored = 0

        for (file_pattern, category, severity), count in counter.items():
            if count < _QA_RECURRENCE_THRESHOLD:
                continue

            content = (
                f"[{severity}] {category} in {file_pattern}: "
                f"{examples[(file_pattern, category, severity)]} "
                f"(seen {count}x)"
            )

            self._memory_store.remember(
                repo_slug=repo_slug,
                agent_type=self._agent_type,
                category="qa_warnings",
                content=content,
                source_task_id=task.id,
                tags=tags + [category, severity.lower()],
            )
            stored += 1

            if stored >= _MAX_QA_WARNINGS:
                break

        if stored:
            logger.debug(
                "FeedbackBus: stored %d recurring QA warning patterns for %s",
                stored, task.id,
            )

    def on_debate_complete(
        self,
        task,
        debate_result: Dict[str, Any],
        current_profile_id: Optional[str] = None,
    ) -> None:
        """Detect domain mismatch from debate and feed back into specialization.

        Checks if the debate synthesis reveals the current specialization was
        a poor fit (low confidence + domain keywords not matching profile).
        If so, records feedback via ProfileRegistry.record_domain_feedback().
        """
        if not self._profile_registry:
            return

        if not current_profile_id:
            return

        confidence = debate_result.get("confidence")
        synthesis = debate_result.get("synthesis", "")

        # Only signal mismatch on low confidence debates
        if confidence is None or confidence > 0.5:
            return

        # Extract domain keywords from synthesis
        domain_tags = _detect_domain_from_keywords(synthesis)
        if not domain_tags:
            return

        # Record the mismatch signal
        self._profile_registry.record_domain_feedback(
            profile_id=current_profile_id,
            domain_tags=domain_tags,
            mismatch_signal=True,
        )

        logger.debug(
            "FeedbackBus: recorded domain mismatch for profile %s "
            "(confidence=%.2f, domains=%s)",
            current_profile_id, confidence, domain_tags,
        )

    def _parse_missed_criteria(self, critique: str, task) -> List[str]:
        """Extract specific missed acceptance criteria from self-eval critique.

        Looks for lines that reference specific criteria being unmet.
        Falls back to the raw critique if no structured items are found.
        """
        criteria = getattr(task, "acceptance_criteria", None) or []
        if not criteria:
            return []

        missed = []
        critique_lower = critique.lower()

        for criterion in criteria:
            # Check if the critique mentions this criterion as failed/missing
            criterion_words = set(criterion.lower().split())
            # Require at least 3 significant words to match (avoid noise)
            significant = {w for w in criterion_words if len(w) > 3}
            if not significant:
                continue

            matches = sum(1 for w in significant if w in critique_lower)
            # If more than half the significant words appear in the critique,
            # this criterion was likely flagged as missed
            if matches >= max(len(significant) // 2, 1):
                missed.append(f"Commonly missed: {criterion}")

        return missed
