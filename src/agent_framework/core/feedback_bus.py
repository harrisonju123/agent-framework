"""Cross-feature learning loop: FeedbackBus connects feature outputs to feature inputs.

Self-eval failures → memory, replan successes → memory (existing),
QA recurring findings → engineer prompt, debate decisions → specialization.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Single feedback event emitted by a producer."""
    source: str        # e.g. "self_eval", "qa_findings", "replan", "debate"
    category: str      # e.g. "self_eval_failures", "qa_findings"
    content: str       # The actual feedback content
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# Type alias for consumer callbacks
ConsumerCallback = Callable[[FeedbackEvent], None]


class FeedbackBus:
    """Lightweight pub/sub bus connecting feature outputs to feature inputs.

    Producers emit FeedbackEvents by category. Consumers subscribe to
    categories and receive matching events. Optionally persists events
    to MemoryStore for cross-session survival.
    """

    def __init__(self, memory_store=None, repo_slug: Optional[str] = None, agent_type: str = "shared"):
        self._consumers: Dict[str, List[ConsumerCallback]] = {}
        self._memory_store = memory_store
        self._repo_slug = repo_slug
        self._agent_type = agent_type
        self._event_log: List[FeedbackEvent] = []

    def register_consumer(self, category: str, callback: ConsumerCallback) -> None:
        """Subscribe a consumer to events of a given category."""
        self._consumers.setdefault(category, []).append(callback)

    def emit(self, event: FeedbackEvent, *, persist: bool = True) -> None:
        """Emit a feedback event to all registered consumers.

        Args:
            event: The feedback event to emit.
            persist: If True and memory_store is configured, persist to memory.
        """
        self._event_log.append(event)

        # Dispatch to category-specific consumers
        for callback in self._consumers.get(event.category, []):
            try:
                callback(event)
            except Exception as e:
                logger.warning(
                    "FeedbackBus consumer error for category %r: %s",
                    event.category, e,
                )

        # Persist to memory store for cross-session survival
        if persist and self._memory_store and self._repo_slug:
            try:
                self._memory_store.remember(
                    repo_slug=self._repo_slug,
                    agent_type=self._agent_type,
                    category=event.category,
                    content=event.content,
                    tags=event.tags,
                    source_task_id=event.metadata.get("task_id"),
                )
            except Exception as e:
                logger.warning("FeedbackBus memory persist failed: %s", e)

    @property
    def event_count(self) -> int:
        """Total events emitted in this session."""
        return len(self._event_log)

    def get_events(self, category: Optional[str] = None) -> List[FeedbackEvent]:
        """Get logged events, optionally filtered by category."""
        if category is None:
            return list(self._event_log)
        return [e for e in self._event_log if e.category == category]

    def get_event_counts_by_category(self) -> Dict[str, int]:
        """Return event counts grouped by category."""
        counts: Dict[str, int] = {}
        for event in self._event_log:
            counts[event.category] = counts.get(event.category, 0) + 1
        return counts
