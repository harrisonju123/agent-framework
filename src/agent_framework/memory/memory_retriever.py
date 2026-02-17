"""Relevance scoring and prompt formatting for agent memories.

Retrieves memories from the store and ranks them by relevance
to the current task, applying recency decay so stale memories
gradually lose priority.
"""

import math
import time
from typing import List, Optional

from .memory_store import MemoryEntry, MemoryStore


# Recency half-life: memories lose half their recency score after this many seconds
RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600  # 1 week

# Maximum characters of memory context injected into prompts
MAX_MEMORY_PROMPT_CHARS = 3000

# Maximum characters for replan context (should be more concise)
MAX_REPLAN_MEMORY_CHARS = 1500


def _recency_score(entry: MemoryEntry) -> float:
    """Exponential decay based on time since last access."""
    age_seconds = time.time() - entry.last_accessed
    return math.exp(-0.693 * age_seconds / RECENCY_HALF_LIFE_SECONDS)


def _frequency_score(entry: MemoryEntry) -> float:
    """Logarithmic boost for frequently accessed memories."""
    return math.log1p(entry.access_count)


def _debate_confidence_boost(entry: MemoryEntry) -> float:
    """Multiplier for high-confidence architectural decisions from debates.

    Debates produce structured synthesis with an explicit confidence level.
    Surfacing high-confidence outcomes above general memories prevents
    re-litigating settled architectural choices.
    """
    if entry.category != "architectural_decisions" or "debate" not in (entry.tags or []):
        return 1.0

    # Confidence is stored as "Confidence: high|medium|low" in the content string
    for line in entry.content.splitlines():
        if line.startswith("Confidence:"):
            level = line.split(":", 1)[1].strip().lower()
            if level == "high":
                return 2.0
            if level == "medium":
                return 1.5
            break  # "low" or unrecognised -> no boost

    return 1.0


def _relevance_score(entry: MemoryEntry, task_tags: Optional[List[str]] = None) -> float:
    """Combined relevance score: recency * frequency * tag overlap * debate boost."""
    score = _recency_score(entry) * (1.0 + _frequency_score(entry))

    # Boost if tags overlap with current task context
    if task_tags and entry.tags:
        overlap = len(set(entry.tags) & set(task_tags))
        score *= (1.0 + overlap * 0.5)

    # High-confidence architectural decisions from debates rank above general memories
    score *= _debate_confidence_boost(entry)

    return score


class MemoryRetriever:
    """Retrieves and formats relevant memories for prompt injection."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def get_relevant_memories(
        self,
        repo_slug: str,
        agent_type: str,
        task_tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memories ranked by relevance to current task."""
        all_memories = self._store.recall_all(repo_slug, agent_type)
        if not all_memories:
            return []

        scored = [(m, _relevance_score(m, task_tags)) for m in all_memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:limit]]

    def format_for_prompt(
        self,
        repo_slug: str,
        agent_type: str,
        task_tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> str:
        """Build a prompt section with relevant memories.

        Returns empty string if no memories or memory system disabled.
        """
        memories = self.get_relevant_memories(repo_slug, agent_type, task_tags, limit)
        if not memories:
            return ""

        lines = ["## Memories from Previous Tasks\n"]
        total_chars = 0

        for mem in memories:
            line = f"- [{mem.category}] {mem.content}"
            if total_chars + len(line) > MAX_MEMORY_PROMPT_CHARS:
                break
            lines.append(line)
            total_chars += len(line)

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def format_for_replan(
        self,
        repo_slug: str,
        agent_type: str,
        task_tags: Optional[List[str]] = None,
        limit: int = 15,
        current_error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> str:
        """Build a replan-specific prompt section with relevant memories.

        Queries 5 priority categories with error-type-aware limits:
        - conventions: coding standards, patterns to follow
        - test_commands: how to run/fix tests
        - repo_structure: where key files live
        - past_failures: what went wrong before and how it was resolved
        - architectural_decisions: design choices that constrain implementation

        Uses error-type-aware retrieval to prioritize the most relevant memories.
        For past_failures, boosts memories tagged with the current error type.

        Returns empty string if no memories or memory system disabled.
        """
        all_memories = self._store.recall_all(repo_slug, agent_type)
        if not all_memories:
            return ""

        # Priority categories for replanning — these are most actionable
        priority_categories = {
            "conventions",
            "test_commands",
            "repo_structure",
            "past_failures",
            "architectural_decisions",
        }

        # Error-type-aware category boosting: give more weight to relevant categories
        category_boosts = {cat: 1.0 for cat in priority_categories}

        if error_type == "test_failure":
            category_boosts["test_commands"] = 3.0
            category_boosts["past_failures"] = 3.0
        elif error_type in ("dependency", "import_error"):
            category_boosts["repo_structure"] = 3.0
            category_boosts["past_failures"] = 3.0
        elif error_type in ("logic", "type_error", "validation"):
            category_boosts["conventions"] = 3.0
            category_boosts["past_failures"] = 3.0
        else:
            # Default: boost past_failures and conventions
            category_boosts["past_failures"] = 2.0
            category_boosts["conventions"] = 2.0

        # Score with category and error-type boosts
        scored = []
        for mem in all_memories:
            # Skip memories not in priority categories
            if mem.category not in priority_categories:
                continue

            score = _relevance_score(mem, task_tags)

            # Apply category boost
            score *= category_boosts.get(mem.category, 1.0)

            # Additional boost for past_failures matching current error type
            if mem.category == "past_failures" and error_type:
                error_tag = f"error:{error_type}"
                if mem.tags and error_tag in mem.tags:
                    score *= 2.0  # Double score for matching error types

            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, _ in scored[:limit]]

        lines = ["## Repository Knowledge (from previous tasks)\n"]
        lines.append("You've worked on this repo before. Here's what you know:\n")
        total_chars = 0

        for mem in top_memories:
            line = f"- [{mem.category}] {mem.content}"
            if total_chars + len(line) > MAX_REPLAN_MEMORY_CHARS:
                break
            lines.append(line)
            total_chars += len(line)

        if len(lines) == 2:  # Only header, no actual memories added
            return ""

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def extract_memories_from_response(
        self,
        response_content: str,
        repo_slug: str,
        agent_type: str,
        task_id: str,
    ) -> int:
        """Parse agent response for learnings and store them.

        Looks for a structured section like:
          ## Learnings
          - [category] content

        Returns number of memories stored.
        """
        if not response_content:
            return 0

        count = 0
        in_learnings = False

        for line in response_content.splitlines():
            stripped = line.strip()

            if stripped.lower().startswith("## learning"):
                in_learnings = True
                continue

            if in_learnings and stripped.startswith("##"):
                break

            if in_learnings and stripped.startswith("- ["):
                # Parse "- [category] content"
                try:
                    bracket_end = stripped.index("]", 3)
                    category = stripped[3:bracket_end]
                    content = stripped[bracket_end + 1:].strip()
                    if category and content:
                        self._store.remember(
                            repo_slug=repo_slug,
                            agent_type=agent_type,
                            category=category,
                            content=content,
                            source_task_id=task_id,
                        )
                        count += 1
                except (ValueError, IndexError):
                    continue

        return count
