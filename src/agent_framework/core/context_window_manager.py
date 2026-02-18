"""Context window management for long-running tasks.

Manages token budgets, progressive message history, and priority-based
context inclusion to prevent quality decay in long tasks.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MEMORY_BUDGET_RATIO = 3000 / 50_000  # 3000 chars at the 50K default implementation budget
MIN_MEMORY_CHARS = 800               # Floor: enough for 5-6 useful memory entries
MAX_MEMORY_CHARS = 6000              # Ceiling: prevent excessive injection for huge budgets


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
    """Token budget tracking for a task.

    Tracks peak input tokens (the largest single prompt sent) plus
    cumulative output tokens (total generated across all turns).
    This avoids double-counting input context that's re-sent each turn.
    """
    total_budget: int
    reserved_for_output: int
    available_for_input: int
    peak_input_tokens: int = 0
    cumulative_output_tokens: int = 0

    @property
    def used_so_far(self) -> int:
        """Effective usage: peak input context + total output generated."""
        return self.peak_input_tokens + self.cumulative_output_tokens

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
        """Update budget with actual token usage from LLM call.

        Tracks peak input (largest prompt across all turns) and cumulative
        output (total tokens generated) to avoid double-counting the prompt
        which is re-sent on every turn.
        """
        self.budget.peak_input_tokens = max(self.budget.peak_input_tokens, input_tokens)
        self.budget.cumulative_output_tokens += output_tokens

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
            role = msg.metadata.get("role", "assistant")
            flat = msg.content.replace("\n", " ")
            # Truncate at word boundary to avoid garbled output
            if len(flat) > 200:
                cut = flat[:200].rfind(" ")
                flat = flat[:cut if cut > 100 else 200]
            summary_parts.append(f"- [{role}] {flat}...")

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

        # Tool-specific summarization with non-overlapping head/tail
        half = MAX_LINES // 2

        if tool_name in ("Read", "Grep"):
            head = lines[:half]
            tail = lines[-half:]
            omitted = len(lines) - len(head) - len(tail)
            return "\n".join(head + [f"... ({omitted} lines omitted) ..."] + tail)

        elif tool_name == "Bash":
            error_lines = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
            if error_lines:
                return "\n".join(error_lines[:10] + ["..."] + lines[-10:])
            return "\n".join(lines[-MAX_LINES:])

        else:
            head = lines[:half]
            tail = lines[-half:]
            omitted = len(lines) - len(head) - len(tail)
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

    def compute_memory_budget(self) -> int:
        """Compute memory budget proportional to the task's token budget.

        Scales with total_budget so larger tasks (escalation, architecture) get more
        memory context, while small tasks (status_report) don't waste their budget.
        Decays linearly from 70-90% utilization rather than cliff-dropping at 70%.

        - < 70% used:  base proportional allocation (clamped to MIN/MAX_MEMORY_CHARS)
        - 70-90% used: linear decay from base → 0 (continuous at 70% boundary)
        - >= 90% used: 0 (omit memories to preserve critical context)

        Note: tasks at MIN_MEMORY_CHARS floor (small budgets) reach effectively zero
        allocation around 87-88% utilization rather than 90%, because the decay
        formula produces sub-useful char counts before the hard cutoff.
        """
        base = int(self.budget.total_budget * MEMORY_BUDGET_RATIO)
        base = max(MIN_MEMORY_CHARS, min(MAX_MEMORY_CHARS, base))

        utilization = self.budget.utilization_percent
        if utilization < 70.0:
            return base
        elif utilization < 90.0:
            # Linear decay from base → 0 between 70% and 90%
            return int(base * (90.0 - utilization) / 20.0)
        else:
            return 0
