"""Cross-feature feedback bus for connecting feature outputs to feature inputs.

Routes structured feedback events between producers (self-eval, QA, replan,
debate) and consumers (memory store, prompt builder, specialization).
Events are persisted through MemoryStore so learnings survive across sessions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.memory_store import MemoryStore
    from .session_logger import SessionLogger

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """A single feedback event emitted by a producer."""
    source: str          # e.g. "self_eval", "qa_review", "replan", "debate"
    category: str        # memory category for persistence
    content: str         # human-readable description of the feedback
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Type alias for consumer callables
Consumer = Callable[[FeedbackEvent], None]


class FeedbackBus:
    """Lightweight publish-subscribe bus for cross-feature feedback.

    Producers emit FeedbackEvents. Consumers subscribe by category and
    receive matching events. All events are optionally persisted to
    MemoryStore for cross-session learning.
    """

    def __init__(
        self,
        memory_store: Optional["MemoryStore"] = None,
        session_logger: Optional["SessionLogger"] = None,
        repo_slug: Optional[str] = None,
        agent_type: Optional[str] = None,
    ):
        self._memory_store = memory_store
        self._session_logger = session_logger
        self._repo_slug = repo_slug
        self._agent_type = agent_type
        # category -> list of consumer callables
        self._consumers: Dict[str, List[Consumer]] = {}
        # Counters for observability
        self._emit_count = 0
        self._persist_count = 0

    def register_consumer(self, category: str, consumer: Consumer) -> None:
        """Subscribe a consumer to events of a specific category."""
        self._consumers.setdefault(category, []).append(consumer)

    def emit(self, event: FeedbackEvent, *, persist: bool = True) -> None:
        """Emit a feedback event to all registered consumers.

        Args:
            event: The feedback event to distribute.
            persist: If True and memory_store is available, persist to memory.
        """
        self._emit_count += 1

        # Notify consumers subscribed to this category
        consumers = self._consumers.get(event.category, [])
        for consumer in consumers:
            try:
                consumer(event)
            except Exception as e:
                logger.debug(f"Feedback consumer error (non-fatal): {e}")

        # Persist to memory store for cross-session learning
        if persist and self._memory_store and self._repo_slug and self._agent_type:
            try:
                self._memory_store.remember(
                    repo_slug=self._repo_slug,
                    agent_type=self._agent_type,
                    category=event.category,
                    content=event.content,
                    tags=event.tags,
                )
                self._persist_count += 1
            except Exception as e:
                logger.debug(f"Feedback persistence error (non-fatal): {e}")

        # Session logging
        if self._session_logger:
            self._session_logger.log(
                "feedback_emitted",
                source=event.source,
                category=event.category,
                content_preview=event.content[:200],
                tags=event.tags,
                persisted=persist,
            )

    @property
    def emit_count(self) -> int:
        return self._emit_count

    @property
    def persist_count(self) -> int:
        return self._persist_count

    def update_context(self, repo_slug: str, agent_type: str) -> None:
        """Update repo/agent context (called per-task when context changes)."""
        self._repo_slug = repo_slug
        self._agent_type = agent_type
