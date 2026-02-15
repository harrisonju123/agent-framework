"""Context window management for long-running tasks.

Manages token budgets, progressive message history, and priority-based
context inclusion to prevent quality decay in long tasks.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Priority levels for context inclusion."""
    CRITICAL = 1  # Task definition, acceptance criteria
    HIGH = 2      # Recent messages, error context
    MEDIUM = 3    # Tool outputs, summaries
    LOW = 4       # Historical context, metadata


@dataclass
class ContextItem:
    """A single item that can be included in the context window."""
    content: str
    priority: ContextPriority
    token_estimate: int
    category: str  # "task_definition", "message", "tool_output", "error", etc.
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBudget:
    """Token budget tracking for a task."""
    total_budget: int
    reserved_for_output: int
    available_for_input: int
    used_so_far: int = 0

    @property
    def remaining(self) -> int:
        """Remaining tokens in budget."""
        return self.total_budget - self.used_so_far

    @property
    def utilization_percent(self) -> float:
        """Percentage of budget used."""
        if self.total_budget == 0:
            return 0.0
        return (self.used_so_far / self.total_budget) * 100

    @property
    def is_near_limit(self) -> bool:
        """Check if approaching budget limit (>=80% used)."""
        return self.utilization_percent >= 80.0


class ContextWindowManager:
    """
    Manages context window for long-running tasks.

    Features:
    - Token budget tracking with configurable limits
    - Progressive message history management (keep recent, summarize old)
    - Priority-based context inclusion
    - Automatic truncation when approaching limits
    """

    def __init__(
        self,
        total_budget: int,
        output_reserve: int = 4096,
        summary_threshold: int = 10,
        min_message_retention: int = 3,
    ):
        """
        Initialize context window manager.

        Args:
            total_budget: Total token budget for the task
            output_reserve: Tokens reserved for model output (default 4K)
            summary_threshold: Number of messages before summarization kicks in
            min_message_retention: Minimum recent messages to keep verbatim
        """
        self.budget = ContextBudget(
            total_budget=total_budget,
            reserved_for_output=output_reserve,
            available_for_input=total_budget - output_reserve,
        )
        self.summary_threshold = summary_threshold
        self.min_message_retention = min_message_retention

        self._context_items: List[ContextItem] = []
        self._message_history: List[ContextItem] = []
        self._summarized_history: Optional[str] = None
        self._checkpoint_triggered: bool = False

    def add_context_item(
        self,
        content: str,
        priority: ContextPriority,
        category: str,
        token_estimate: Optional[int] = None,
        **metadata: Any,
    ) -> None:
        """Add a context item with estimated token count."""
        if token_estimate is None:
            token_estimate = self._estimate_tokens(content)

        item = ContextItem(
            content=content,
            priority=priority,
            token_estimate=token_estimate,
            category=category,
            metadata=metadata,
        )
        self._context_items.append(item)

        if category == "message":
            self._message_history.append(item)

    def add_message(
        self,
        content: str,
        role: str = "assistant",
        token_estimate: Optional[int] = None,
    ) -> None:
        """Add a message to history (recent messages have HIGH priority)."""
        self.add_context_item(
            content=content,
            priority=ContextPriority.HIGH,
            category="message",
            token_estimate=token_estimate,
            role=role,
        )

    def add_tool_output(
        self,
        tool_name: str,
        output: str,
        token_estimate: Optional[int] = None,
        summarize: bool = True,
    ) -> None:
        """
        Add tool output to context.

        Large outputs are automatically summarized if summarize=True.
        """
        if token_estimate is None:
            token_estimate = self._estimate_tokens(output)

        # Summarize large tool outputs (>1000 tokens)
        if summarize and token_estimate > 1000:
            output = self._summarize_tool_output(tool_name, output)
            token_estimate = self._estimate_tokens(output)

        self.add_context_item(
            content=output,
            priority=ContextPriority.MEDIUM,
            category="tool_output",
            token_estimate=token_estimate,
            tool_name=tool_name,
        )

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update budget with actual token usage from LLM call."""
        self.budget.used_so_far += input_tokens + output_tokens

        if self.budget.is_near_limit:
            logger.warning(
                f"Context budget near limit: {self.budget.utilization_percent:.1f}% "
                f"({self.budget.used_so_far}/{self.budget.total_budget} tokens)"
            )

    def build_context(self, max_tokens: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build context string respecting token budget.

        Returns:
            Tuple of (context_string, metadata_dict)

        Metadata includes:
            - total_tokens: Estimated tokens in context
            - items_included: Number of context items
            - items_dropped: Number of items dropped due to budget
            - priority_breakdown: Token counts by priority level
        """
        if max_tokens is None:
            max_tokens = self.budget.available_for_input

        # Check if we need progressive summarization
        if len(self._message_history) > self.summary_threshold:
            self._apply_progressive_summarization()

        # Sort items by priority (CRITICAL first)
        sorted_items = sorted(self._context_items, key=lambda x: x.priority.value)

        # Build context respecting budget
        included_items: List[ContextItem] = []
        token_count = 0
        priority_counts: Dict[str, int] = {}

        for item in sorted_items:
            if token_count + item.token_estimate <= max_tokens:
                included_items.append(item)
                token_count += item.token_estimate

                priority_name = item.priority.name
                priority_counts[priority_name] = priority_counts.get(priority_name, 0) + item.token_estimate
            else:
                # Budget exhausted
                logger.debug(
                    f"Dropping {item.priority.name} item ({item.category}): "
                    f"would exceed budget ({token_count + item.token_estimate} > {max_tokens})"
                )

        # Build final context string
        context_parts = []

        # Add summarized history if exists
        if self._summarized_history:
            context_parts.append(f"## Previous Activity Summary\n{self._summarized_history}\n")

        # Add included items grouped by category, preserving priority order
        _CATEGORY_ORDER = ["task_definition", "error", "message", "tool_output", "metadata"]
        seen = set()
        for category in _CATEGORY_ORDER:
            category_items = [i for i in included_items if i.category == category]
            if category_items:
                context_parts.extend(item.content for item in category_items)
                seen.add(category)
        # Include any items with categories not in the predefined list
        remaining = [i for i in included_items if i.category not in seen]
        if remaining:
            context_parts.extend(item.content for item in remaining)

        context_string = "\n\n".join(context_parts)

        metadata = {
            "total_tokens": token_count,
            "items_included": len(included_items),
            "items_dropped": len(sorted_items) - len(included_items),
            "priority_breakdown": priority_counts,
            "budget_utilization_percent": self.budget.utilization_percent,
        }

        return context_string, metadata

    def _apply_progressive_summarization(self) -> None:
        """
        Summarize old messages while keeping recent ones verbatim.

        Strategy:
        - Keep last N messages verbatim (HIGH priority)
        - Summarize older messages into a compact summary
        """
        if len(self._message_history) <= self.min_message_retention:
            return  # Not enough messages to summarize

        # Split into old (to summarize) and recent (to keep)
        messages_to_summarize = self._message_history[:-self.min_message_retention]
        recent_messages = self._message_history[-self.min_message_retention:]

        if not messages_to_summarize:
            return

        # Create summary of old messages
        summary_parts = []
        for msg in messages_to_summarize:
            # Extract key actions/outcomes
            role = msg.metadata.get("role", "assistant")
            content_preview = msg.content[:200].replace("\n", " ")
            summary_parts.append(f"- [{role}] {content_preview}...")

        self._summarized_history = "\n".join(summary_parts)

        # Remove old messages from context items, keep recent
        recent_ids = {id(m) for m in recent_messages}
        self._context_items = [
            item for item in self._context_items
            if item.category != "message" or id(item) in recent_ids
        ]
        self._message_history = recent_messages

        logger.info(
            f"Applied progressive summarization: {len(messages_to_summarize)} old messages "
            f"summarized, {len(recent_messages)} recent messages retained"
        )

    def _summarize_tool_output(self, tool_name: str, output: str) -> str:
        """
        Summarize large tool output to reduce token usage.

        Extracts key information based on tool type.
        """
        MAX_LINES = 50
        lines = output.split("\n")

        if len(lines) <= MAX_LINES:
            return output

        # Tool-specific summarization
        if tool_name in ("Read", "Grep"):
            # Keep first and last portions
            head = lines[:20]
            tail = lines[-20:]
            omitted = len(lines) - 40
            return "\n".join(head + [f"... ({omitted} lines omitted) ..."] + tail)

        elif tool_name == "Bash":
            # Keep errors and last few lines
            error_lines = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
            if error_lines:
                return "\n".join(error_lines[:10] + ["..."] + lines[-10:])
            return "\n".join(lines[-20:])

        else:
            # Generic: keep start and end
            head = lines[:25]
            tail = lines[-25:]
            omitted = len(lines) - 50
            return "\n".join(head + [f"... ({omitted} lines omitted) ..."] + tail)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (~4 characters per token for English text)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status for logging/monitoring."""
        return {
            "total_budget": self.budget.total_budget,
            "used_so_far": self.budget.used_so_far,
            "remaining": self.budget.remaining,
            "utilization_percent": self.budget.utilization_percent,
            "is_near_limit": self.budget.is_near_limit,
            "message_count": len(self._message_history),
            "context_items": len(self._context_items),
            "has_summarized_history": self._summarized_history is not None,
        }

    def should_trigger_checkpoint(self) -> bool:
        """Check if a checkpoint should fire due to budget exhaustion.

        Returns True exactly once when budget crosses the 90% threshold.
        """
        if self._checkpoint_triggered:
            return False
        if self.budget.utilization_percent >= 90.0:
            self._checkpoint_triggered = True
            return True
        return False
