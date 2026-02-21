"""Cross-feature feedback bus for persisting learning signals to memory.

Connects feature outputs to feature inputs:
- Self-eval failures -> memory (missed acceptance criteria patterns)
- QA recurring findings -> memory (qa_patterns for prompt injection)
- Debate decisions -> specialization (hints when domain mismatch detected)

Wraps MemoryStore so callers don't scatter memory calls across modules.
"""

import logging
import re
from typing import List, Optional

from ..memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Cap on QA warning text injected into prompts to avoid bloat
QA_WARNINGS_MAX_CHARS = 500

# Specialization keywords used to detect domain signals in debate topics
SPECIALIZATION_KEYWORDS = {
    "frontend": {"frontend", "ui", "ux", "react", "vue", "css", "component", "browser", "dom", "svelte", "angular"},
    "backend": {"backend", "api", "database", "server", "endpoint", "query", "sql", "grpc", "rest", "microservice"},
    "infrastructure": {"infrastructure", "devops", "ci/cd", "docker", "kubernetes", "terraform", "deploy", "helm", "k8s", "pipeline"},
}


class FeedbackBus:
    """Single write interface for cross-feature learning signals.

    All feedback loop writes go through this class so memory interactions
    are consolidated rather than scattered across modules.
    """

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    @property
    def enabled(self) -> bool:
        return self._store.enabled

    def store_self_eval_failure(
        self,
        repo_slug: str,
        agent_type: str,
        task_id: str,
        critique: str,
        acceptance_criteria: Optional[List[str]] = None,
    ) -> int:
        """Parse self-eval FAIL critique and store missed criteria patterns.

        Extracts which acceptance criteria were missed from the critique text
        and stores them as category='missed_criteria' memories so future tasks
        can avoid the same gaps.

        Returns number of memories stored.
        """
        if not self._store.enabled:
            return 0

        count = 0

        # Strategy 1: Match critique against known acceptance criteria
        if acceptance_criteria:
            critique_lower = critique.lower()
            for criterion in acceptance_criteria:
                # Check if the critique references this criterion (keyword overlap)
                criterion_words = set(
                    w.lower() for w in criterion.split() if len(w) > 3
                )
                matched_words = [w for w in criterion_words if w in critique_lower]
                if len(matched_words) >= 2 or (len(criterion_words) <= 2 and matched_words):
                    content = f"Commonly missed: {criterion}"
                    # Tag with significant words for retrieval filtering
                    tags = ["self_eval"] + list(matched_words)[:3]
                    stored = self._store.remember(
                        repo_slug=repo_slug,
                        agent_type=agent_type,
                        category="missed_criteria",
                        content=content,
                        source_task_id=task_id,
                        tags=tags,
                    )
                    if stored:
                        count += 1

        # Strategy 2: Extract gap descriptions directly from critique
        # Look for lines that describe specific failures
        gap_patterns = [
            r"(?:missing|lacks?|no|without|didn'?t|absent)\s+(.{10,80})",
            r"(?:FAIL|failed|gap).*?:\s*(.{10,80})",
        ]
        for pattern in gap_patterns:
            for match in re.finditer(pattern, critique, re.IGNORECASE):
                gap_text = match.group(1).strip().rstrip(".")
                if len(gap_text) > 10:
                    content = f"Self-eval gap: {gap_text}"
                    stored = self._store.remember(
                        repo_slug=repo_slug,
                        agent_type=agent_type,
                        category="missed_criteria",
                        content=content,
                        source_task_id=task_id,
                        tags=["self_eval"],
                    )
                    if stored:
                        count += 1
                    # Cap at 3 extracted gaps per eval to avoid noise
                    if count >= 3:
                        break
            if count >= 3:
                break

        if count > 0:
            logger.info(
                "Stored %d missed_criteria memories from self-eval failure (task %s)",
                count, task_id,
            )

        return count

    def store_qa_pattern(
        self,
        repo_slug: str,
        agent_type: str,
        task_id: str,
        structured_findings: dict,
    ) -> int:
        """Extract recurring QA finding categories and store as qa_patterns.

        Groups findings by category (security, performance, testing, etc.)
        and stores aggregated patterns so future prompts can warn about them.

        Returns number of memories stored.
        """
        if not self._store.enabled:
            return 0

        findings = structured_findings.get("findings", [])
        if not findings:
            return 0

        # Group findings by severity and category
        category_counts: dict[str, list[str]] = {}
        for finding in findings:
            severity = finding.get("severity", "UNKNOWN").upper()
            desc = finding.get("description", "")
            file_path = finding.get("file", "")

            # Classify finding into a domain category
            domain = self._classify_finding(desc, file_path)
            category_counts.setdefault(domain, []).append(
                f"[{severity}] {desc[:80]}"
            )

        count = 0
        for domain, examples in category_counts.items():
            # Only store patterns that appear 2+ times (recurring signal)
            # or are CRITICAL severity
            has_critical = any("[CRITICAL]" in e for e in examples)
            if len(examples) >= 2 or has_critical:
                example_text = "; ".join(examples[:3])
                content = f"QA pattern ({domain}): {len(examples)} findings — {example_text}"
                stored = self._store.remember(
                    repo_slug=repo_slug,
                    agent_type=agent_type,
                    category="qa_patterns",
                    content=content,
                    source_task_id=task_id,
                    tags=["qa", domain],
                )
                if stored:
                    count += 1

        if count > 0:
            logger.info(
                "Stored %d qa_patterns memories from QA findings (task %s)",
                count, task_id,
            )

        return count

    def store_specialization_signal(
        self,
        repo_slug: str,
        task_id: str,
        topic: str,
        recommendation: str,
        current_specialization: Optional[str] = None,
    ) -> bool:
        """Store debate-driven specialization hint when domain mismatch detected.

        Analyzes the debate topic and recommendation for specialization keywords.
        If the recommendation suggests a different domain than what was used,
        stores a specialization_hint memory under the shared namespace.

        Returns True if a hint was stored.
        """
        if not self._store.enabled:
            return False

        # Detect domain signals in topic and recommendation
        topic_domains = self._detect_domains(topic)
        rec_domains = self._detect_domains(recommendation)

        # Combine all detected domains
        all_domains = topic_domains | rec_domains
        if not all_domains:
            return False

        # Check if there's a mismatch with the current specialization
        if current_specialization and current_specialization in all_domains:
            # Current specialization matches — no hint needed
            return False

        # If recommendation points to a specific domain, store the hint
        # Prefer recommendation domains over topic domains
        suggested = rec_domains or all_domains
        if not suggested:
            return False

        # Pick the strongest signal (most keyword matches)
        best_domain = max(suggested, key=lambda d: len(suggested))

        content = (
            f"Debate suggested '{best_domain}' specialization. "
            f"Topic: {topic[:100]}. "
            f"Recommendation: {recommendation[:150]}"
        )

        stored = self._store.remember(
            repo_slug=repo_slug,
            agent_type="shared",
            category="specialization_hints",
            content=content,
            source_task_id=task_id,
            tags=["debate", best_domain],
        )

        if stored:
            logger.info(
                "Stored specialization_hint '%s' from debate (task %s)",
                best_domain, task_id,
            )

        return stored

    def get_qa_warnings(
        self,
        repo_slug: str,
        agent_type: str,
        max_chars: int = QA_WARNINGS_MAX_CHARS,
    ) -> str:
        """Format top recurring QA patterns as explicit warnings for prompt injection.

        Returns a formatted string for the '## QA Pattern Warnings' prompt section,
        or empty string if no patterns exist.
        """
        # Recall both agent-specific and shared qa_patterns + missed_criteria
        patterns = self._store.recall(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="qa_patterns",
            limit=5,
        )
        missed = self._store.recall(
            repo_slug=repo_slug,
            agent_type=agent_type,
            category="missed_criteria",
            limit=5,
        )

        if not patterns and not missed:
            return ""

        lines = ["## QA Pattern Warnings\n"]
        lines.append(
            "Based on previous tasks, watch out for these recurring issues:\n"
        )
        total_chars = sum(len(l) for l in lines)

        for mem in patterns:
            line = f"- {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        for mem in missed:
            line = f"- {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        lines.append("")
        return "\n".join(lines)

    def get_specialization_hints(
        self,
        repo_slug: str,
        limit: int = 3,
    ) -> list:
        """Retrieve recent specialization hints from the shared namespace.

        Returns list of MemoryEntry objects for specialization_hints category.
        """
        return self._store.recall(
            repo_slug=repo_slug,
            agent_type="shared",
            category="specialization_hints",
            limit=limit,
        )

    @staticmethod
    def _classify_finding(description: str, file_path: str) -> str:
        """Classify a QA finding into a domain category."""
        text = f"{description} {file_path}".lower()

        if any(w in text for w in ("sql injection", "xss", "csrf", "auth", "secret", "credential", "vulnerability", "injection")):
            return "security"
        if any(w in text for w in ("n+1", "performance", "slow", "cache", "memory leak", "optimization")):
            return "performance"
        if any(w in text for w in ("test", "coverage", "assert", "mock", "fixture")):
            return "testing"
        if any(w in text for w in ("error handling", "exception", "try", "catch", "panic", "recover")):
            return "error_handling"
        if any(w in text for w in ("type", "typing", "annotation", "interface", "schema")):
            return "type_safety"
        if any(w in text for w in ("log", "debug", "trace", "monitor", "metric")):
            return "observability"
        return "code_quality"

    @staticmethod
    def _detect_domains(text: str) -> set:
        """Detect specialization domains mentioned in text."""
        text_lower = text.lower()
        found = set()
        for domain, keywords in SPECIALIZATION_KEYWORDS.items():
            matches = sum(1 for k in keywords if k in text_lower)
            if matches >= 2:
                found.add(domain)
        return found
