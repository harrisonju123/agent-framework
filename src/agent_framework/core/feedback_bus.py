"""Cross-feature learning loop: feedback bus connecting feature outputs to feature inputs.

Three feedback paths:
1. Self-eval failures → memory: stores commonly missed acceptance criteria
2. QA recurring findings → memory: aggregates cross-task findings for prompt injection
3. Debate decisions → specialization: stores domain hints from debate synthesis
"""

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..memory.memory_store import MemoryStore
    from .session_logger import SessionLogger

logger = logging.getLogger(__name__)

# Cap stored missed criteria per repo to avoid memory bloat
MAX_MISSED_CRITERIA_PER_REPO = 5

# Only store QA findings that recur this many times or more
QA_RECURRING_THRESHOLD = 2

# Domain keywords for debate specialization hints
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "frontend": ["react", "vue", "angular", "css", "html", "dom", "browser", "ui", "ux", "component"],
    "backend": ["api", "database", "server", "endpoint", "middleware", "auth", "cache", "queue"],
    "infrastructure": ["docker", "kubernetes", "ci/cd", "pipeline", "deploy", "terraform", "aws", "gcp"],
    "data": ["etl", "pipeline", "schema", "migration", "analytics", "warehouse", "sql"],
}

# Max chars for the Known Pitfalls prompt section
MAX_PITFALLS_SECTION_CHARS = 1000


def _finding_key(finding: Dict[str, Any]) -> str:
    """Stable dedup key for a QA finding: (file_pattern, category, description_hash).

    Uses a truncated hash of the description to group similar findings
    without exact-match sensitivity to wording variations.
    """
    file_path = finding.get("file", "")
    # Generalize file path to a pattern (strip specific line numbers, keep filename)
    file_pattern = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

    category = finding.get("category", finding.get("severity", "general"))

    desc = finding.get("description", finding.get("issue", ""))
    # Truncate to first 80 chars before hashing for fuzzy grouping
    desc_hash = hashlib.md5(desc[:80].lower().encode(), usedforsecurity=False).hexdigest()[:8]

    return f"{file_pattern}|{category}|{desc_hash}"


def store_self_eval_failure(
    memory_store: "MemoryStore",
    session_logger: Optional["SessionLogger"],
    *,
    task_id: str,
    repo_slug: str,
    agent_type: str,
    acceptance_criteria: List[str],
    critique: str,
) -> int:
    """Store which acceptance criteria were missed during self-eval failure.

    Parses the critique text to identify which criteria were flagged,
    then stores each as a memory under category='missed_criteria'.

    Returns number of criteria stored.
    """
    if not memory_store or not memory_store.enabled:
        return 0

    if not acceptance_criteria or not critique:
        return 0

    critique_lower = critique.lower()
    stored = 0

    for criterion in acceptance_criteria:
        # Check if this criterion appears to be mentioned as failing in the critique.
        # Extract distinctive words (>4 chars) from the criterion to match.
        words = [w.lower() for w in re.findall(r'\b\w{5,}\b', criterion)]
        if not words:
            continue

        # Require at least 2 keyword matches (or 1 if criterion has only 1 keyword)
        min_matches = min(2, len(words))
        matched = sum(1 for w in words[:6] if w in critique_lower)

        if matched >= min_matches:
            content = f"Commonly missed: {criterion[:200]}"
            memory_store.remember(
                repo_slug=repo_slug,
                agent_type=agent_type,
                category="missed_criteria",
                content=content,
                source_task_id=task_id,
                tags=words[:3],
            )
            stored += 1

            if stored >= MAX_MISSED_CRITERIA_PER_REPO:
                break

    if stored > 0 and session_logger:
        session_logger.log(
            "feedback_bus_self_eval_stored",
            task_id=task_id,
            repo=repo_slug,
            criteria_stored=stored,
            total_criteria=len(acceptance_criteria),
        )

    return stored


def aggregate_qa_findings(
    memory_store: "MemoryStore",
    session_logger: Optional["SessionLogger"],
    *,
    task_id: str,
    repo_slug: str,
    structured_findings: List[Dict[str, Any]],
) -> int:
    """Aggregate QA findings and store those that recur across tasks.

    Groups findings by (file_pattern, category, description_hash). Only persists
    findings that appear 2+ times total (across this and previous calls).

    Returns number of recurring findings stored.
    """
    if not memory_store or not memory_store.enabled:
        return 0

    if not structured_findings:
        return 0

    # Load existing recurring findings to check occurrence counts
    existing = memory_store.recall(
        repo_slug=repo_slug,
        agent_type="shared",
        category="recurring_qa_findings",
        limit=200,
    )
    existing_keys: Dict[str, int] = {}
    for mem in existing:
        # Parse occurrence count from content prefix "({N}x) ..."
        match = re.match(r'\((\d+)x\)', mem.content)
        count = int(match.group(1)) if match else 1
        # Reconstruct the key from stored content
        existing_keys[mem.content.split("] ", 1)[-1][:80] if "] " in mem.content else mem.content[:80]] = count

    stored = 0
    seen_keys: Dict[str, Dict[str, Any]] = {}

    for finding in structured_findings:
        key = _finding_key(finding)
        if key in seen_keys:
            # Duplicate within same task — increment
            seen_keys[key]["count"] += 1
        else:
            desc = finding.get("description", finding.get("issue", "unknown"))
            category = finding.get("category", finding.get("severity", "general"))
            seen_keys[key] = {"desc": desc[:150], "category": category, "count": 1}

    for key, info in seen_keys.items():
        desc_prefix = info["desc"][:80]

        # Check if this finding already exists in memory
        prev_count = 0
        for existing_content, cnt in existing_keys.items():
            if desc_prefix in existing_content or existing_content in desc_prefix:
                prev_count = cnt
                break

        total_count = prev_count + info["count"]
        if total_count < QA_RECURRING_THRESHOLD:
            continue

        content = f"({total_count}x) [{info['category']}] {info['desc']}"
        memory_store.remember(
            repo_slug=repo_slug,
            agent_type="shared",
            category="recurring_qa_findings",
            content=content,
            source_task_id=task_id,
            tags=["qa_recurring", info["category"]],
        )
        stored += 1

    if stored > 0 and session_logger:
        session_logger.log(
            "feedback_bus_qa_aggregated",
            task_id=task_id,
            repo=repo_slug,
            findings_input=len(structured_findings),
            recurring_stored=stored,
        )

    return stored


def store_debate_specialization_hint(
    memory_store: "MemoryStore",
    session_logger: Optional["SessionLogger"],
    *,
    repo_slug: str,
    debate_topic: str,
    synthesis: str,
    confidence: str,
) -> Optional[str]:
    """Extract domain signal from debate synthesis and store as specialization hint.

    Only acts on high-confidence debates that mention domain-specific terms.
    Returns the detected domain, or None if no strong signal found.
    """
    if not memory_store or not memory_store.enabled:
        return None

    # Only store hints from high-confidence debates
    if confidence.lower() not in ("high", "very high"):
        return None

    synthesis_lower = synthesis.lower()
    topic_lower = debate_topic.lower()
    combined = f"{topic_lower} {synthesis_lower}"

    # Score each domain by keyword matches
    domain_scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score >= 2:
            domain_scores[domain] = score

    if not domain_scores:
        return None

    # Pick the strongest domain signal
    best_domain = max(domain_scores, key=domain_scores.get)  # type: ignore[arg-type]

    content = (
        f"Debate on '{debate_topic[:100]}' strongly suggests {best_domain} expertise. "
        f"Confidence: {confidence}."
    )

    memory_store.remember(
        repo_slug=repo_slug,
        agent_type="shared",
        category="specialization_hints",
        content=content,
        tags=[best_domain, "debate_hint"],
    )

    if session_logger:
        session_logger.log(
            "feedback_bus_specialization_hint",
            repo=repo_slug,
            domain=best_domain,
            score=domain_scores[best_domain],
            confidence=confidence,
        )

    return best_domain


def format_known_pitfalls(
    memory_store: "MemoryStore",
    *,
    repo_slug: str,
    agent_type: str,
    max_chars: int = MAX_PITFALLS_SECTION_CHARS,
) -> str:
    """Build a '## Known Pitfalls' prompt section from recurring findings and missed criteria.

    Queries MemoryStore for recurring_qa_findings and missed_criteria,
    formats them with occurrence counts, and caps output to max_chars.

    Returns empty string if nothing to inject.
    """
    if not memory_store or not memory_store.enabled:
        return ""

    recurring = memory_store.recall(
        repo_slug=repo_slug,
        agent_type="shared",
        category="recurring_qa_findings",
        limit=10,
    )

    missed = memory_store.recall(
        repo_slug=repo_slug,
        agent_type=agent_type,
        category="missed_criteria",
        limit=10,
    )

    if not recurring and not missed:
        return ""

    lines = ["## Known Pitfalls\n"]
    lines.append("Based on previous tasks in this repo, watch out for:\n")
    total_chars = sum(len(l) for l in lines)

    if recurring:
        lines.append("### Recurring QA Findings")
        total_chars += len(lines[-1])
        for mem in recurring:
            line = f"- {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

    if missed and total_chars < max_chars:
        lines.append("\n### Commonly Missed Criteria")
        total_chars += len(lines[-1])
        for mem in missed:
            line = f"- {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

    lines.append("")
    return "\n".join(lines)
