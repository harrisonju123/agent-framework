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


def _recency_score(entry: MemoryEntry) -> float:
    """Exponential decay based on time since last access."""
    age_seconds = time.time() - entry.last_accessed
    return math.exp(-0.693 * age_seconds / RECENCY_HALF_LIFE_SECONDS)


def _frequency_score(entry: MemoryEntry) -> float:
    """Logarithmic boost for frequently accessed memories."""
    return math.log1p(entry.access_count)


def _relevance_score(entry: MemoryEntry, task_tags: Optional[List[str]] = None) -> float:
    """Combined relevance score: recency * frequency * tag overlap."""
    score = _recency_score(entry) * (1.0 + _frequency_score(entry))

    # Boost if tags overlap with current task context
    if task_tags and entry.tags:
        overlap = len(set(entry.tags) & set(task_tags))
        score *= (1.0 + overlap * 0.5)

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
        max_chars: int = 1500,
    ) -> str:
        """Build a memory section for replan context.

        Prioritizes categories most useful for recovering from failures:
        conventions, test_commands, repo_structure.

        Returns empty string if no memories exist.
        """
        PRIORITY_CATEGORIES = {"conventions", "test_commands", "repo_structure"}

        memories = self.get_relevant_memories(
            repo_slug, agent_type, task_tags, limit=20,
        )
        if not memories:
            return ""

        # Sort: priority categories first, then by existing relevance order
        priority = [m for m in memories if m.category in PRIORITY_CATEGORIES]
        others = [m for m in memories if m.category not in PRIORITY_CATEGORIES]
        ordered = priority + others

        lines = ["## Relevant Memories from This Repo\n"]
        total_chars = 0

        for mem in ordered:
            line = f"- [{mem.category}] {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if len(lines) == 1:  # only header
            return ""

        lines.append("")
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
