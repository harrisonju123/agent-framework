"""Tests for context window management."""

import pytest

from agent_framework.core.context_window_manager import (
    ContextWindowManager,
    ContextPriority,
    ContextItem,
    ContextBudget,
)


class TestContextBudget:
    """Test context budget tracking."""

    def test_budget_initialization(self):
        """Verify budget is initialized correctly."""
        budget = ContextBudget(
            total_budget=10000,
            reserved_for_output=2000,
            available_for_input=8000,
        )

        assert budget.total_budget == 10000
        assert budget.reserved_for_output == 2000
        assert budget.available_for_input == 8000
        assert budget.used_so_far == 0
        assert budget.remaining == 10000

    def test_budget_utilization(self):
        """Verify budget utilization is calculated correctly."""
        budget = ContextBudget(
            total_budget=10000,
            reserved_for_output=2000,
            available_for_input=8000,
            used_so_far=5000,
        )

        assert budget.utilization_percent == 50.0
        assert budget.remaining == 5000

    def test_budget_near_limit_detection(self):
        """Verify near-limit detection works at 80% threshold."""
        budget = ContextBudget(
            total_budget=10000,
            reserved_for_output=2000,
            available_for_input=8000,
            used_so_far=8000,
        )

        assert budget.is_near_limit is True  # 80% is at limit

        budget.used_so_far = 7999
        assert budget.is_near_limit is False  # < 80%


class TestContextWindowManager:
    """Test context window manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a context window manager for testing."""
        return ContextWindowManager(
            total_budget=10000,
            output_reserve=2000,
            summary_threshold=5,
            min_message_retention=2,
        )

    def test_initialization(self, manager):
        """Verify manager initializes correctly."""
        assert manager.budget.total_budget == 10000
        assert manager.budget.reserved_for_output == 2000
        assert manager.budget.available_for_input == 8000
        assert manager.summary_threshold == 5
        assert manager.min_message_retention == 2

    def test_add_context_item(self, manager):
        """Verify context items can be added."""
        manager.add_context_item(
            content="Test content",
            priority=ContextPriority.HIGH,
            category="test",
        )

        assert len(manager._context_items) == 1
        item = manager._context_items[0]
        assert item.content == "Test content"
        assert item.priority == ContextPriority.HIGH
        assert item.category == "test"

    def test_add_message(self, manager):
        """Verify messages are added with HIGH priority."""
        manager.add_message("User message", role="user")

        assert len(manager._message_history) == 1
        msg = manager._message_history[0]
        assert msg.priority == ContextPriority.HIGH
        assert msg.category == "message"
        assert msg.metadata["role"] == "user"

    def test_add_tool_output_small(self, manager):
        """Verify small tool outputs are not summarized."""
        manager.add_tool_output(
            tool_name="Read",
            output="Short output",
            summarize=True,
        )

        assert len(manager._context_items) == 1
        item = manager._context_items[0]
        assert item.content == "Short output"
        assert item.category == "tool_output"

    def test_add_tool_output_large(self, manager):
        """Verify large tool outputs are summarized."""
        # Create output with >1000 tokens (~3500 chars = ~1166 tokens)
        large_output = "\n".join([f"Line {i} with some extra text to make it longer" for i in range(100)])
        manager.add_tool_output(
            tool_name="Read",
            output=large_output,
            summarize=True,
        )

        assert len(manager._context_items) == 1
        item = manager._context_items[0]
        # Should be summarized (truncated)
        assert "omitted" in item.content
        assert len(item.content) < len(large_output)

    def test_update_token_usage(self, manager):
        """Verify token usage updates budget."""
        manager.update_token_usage(input_tokens=1000, output_tokens=500)

        assert manager.budget.used_so_far == 1500
        assert manager.budget.remaining == 8500

    def test_build_context_priority_ordering(self, manager):
        """Verify context items are included by priority."""
        # Add items with different priorities
        manager.add_context_item(
            content="Low priority",
            priority=ContextPriority.LOW,
            category="metadata",
        )
        manager.add_context_item(
            content="Critical item",
            priority=ContextPriority.CRITICAL,
            category="task_definition",
        )
        manager.add_context_item(
            content="High priority",
            priority=ContextPriority.HIGH,
            category="message",
        )

        context, metadata = manager.build_context()

        # CRITICAL should come first
        assert context.index("Critical item") < context.index("High priority")
        assert context.index("High priority") < context.index("Low priority")

    def test_build_context_budget_limit(self, manager):
        """Verify low-priority items are dropped when budget is tight."""
        # Fill budget with critical items (each ~166 tokens)
        for i in range(10):
            manager.add_context_item(
                content="A" * 500,  # ~166 tokens each
                priority=ContextPriority.CRITICAL,
                category="task_definition",
            )

        # Add low-priority item that won't fit
        manager.add_context_item(
            content="Low priority that won't fit",
            priority=ContextPriority.LOW,
            category="metadata",
        )

        # Use small budget so low priority gets dropped
        context, metadata = manager.build_context(max_tokens=1500)

        # Low priority should be dropped
        assert "Low priority that won't fit" not in context
        assert metadata["items_dropped"] >= 1  # At least the low priority item

    def test_progressive_summarization(self, manager):
        """Verify old messages are summarized while recent ones are kept."""
        # Add 6 messages (threshold is 5)
        for i in range(6):
            manager.add_message(f"Message {i}", role="assistant")

        # Trigger summarization
        manager._apply_progressive_summarization()

        # Should have summary + 2 recent messages
        assert manager._summarized_history is not None
        assert len(manager._message_history) == 2  # min_message_retention

        # Recent messages should be intact
        assert any("Message 5" in item.content for item in manager._message_history)
        assert any("Message 4" in item.content for item in manager._message_history)

    def test_get_budget_status(self, manager):
        """Verify budget status is reported correctly."""
        manager.add_message("Test message")
        manager.update_token_usage(1000, 500)

        status = manager.get_budget_status()

        assert status["total_budget"] == 10000
        assert status["used_so_far"] == 1500
        assert status["remaining"] == 8500
        assert status["message_count"] == 1
        assert status["context_items"] == 1

    def test_should_trigger_checkpoint(self, manager):
        """Verify checkpoint trigger at 90% budget usage."""
        manager.update_token_usage(9000, 0)

        assert manager.should_trigger_checkpoint() is True  # 90% is at checkpoint

        # Reset and use less
        manager.budget.used_so_far = 8999
        assert manager.should_trigger_checkpoint() is False  # < 90%

    def test_token_estimation(self, manager):
        """Verify token estimation heuristic."""
        # Empty string
        assert manager._estimate_tokens("") == 0

        # ~350 characters should be ~100-120 tokens (1 token per ~3 chars)
        text = "x" * 350
        estimated = manager._estimate_tokens(text)
        assert 90 <= estimated <= 120

    def test_summarize_tool_output_read(self, manager):
        """Verify Read tool output summarization keeps head and tail."""
        lines = [f"Line {i}" for i in range(100)]
        output = "\n".join(lines)

        summarized = manager._summarize_tool_output("Read", output)

        # Should have head, omitted marker, and tail
        assert "Line 0" in summarized
        assert "Line 99" in summarized
        assert "omitted" in summarized
        assert len(summarized.split("\n")) < 100

    def test_summarize_tool_output_bash_errors(self, manager):
        """Verify Bash errors are preserved in summarization."""
        lines = ["Output line 1", "Error: something failed", "Output line 2"] + \
                [f"Line {i}" for i in range(100)]
        output = "\n".join(lines)

        summarized = manager._summarize_tool_output("Bash", output)

        # Should preserve error lines
        assert "Error: something failed" in summarized

    def test_summarize_tool_output_short(self, manager):
        """Verify short outputs are not summarized."""
        short_output = "\n".join([f"Line {i}" for i in range(10)])

        summarized = manager._summarize_tool_output("Read", short_output)

        # Should be unchanged
        assert summarized == short_output

    def test_build_context_with_summarized_history(self, manager):
        """Verify summarized history is included in context."""
        # Add messages to trigger summarization
        for i in range(6):
            manager.add_message(f"Message {i}")

        manager._apply_progressive_summarization()

        context, metadata = manager.build_context()

        # Should include summary section
        assert "Previous Activity Summary" in context
        assert "Message 0" in context  # From summary

    def test_context_metadata_breakdown(self, manager):
        """Verify context metadata includes priority breakdown."""
        manager.add_context_item(
            content="A" * 300,
            priority=ContextPriority.CRITICAL,
            category="task_definition",
        )
        manager.add_context_item(
            content="B" * 300,
            priority=ContextPriority.HIGH,
            category="message",
        )

        context, metadata = manager.build_context()

        # Should have priority breakdown
        assert "priority_breakdown" in metadata
        assert "CRITICAL" in metadata["priority_breakdown"]
        assert "HIGH" in metadata["priority_breakdown"]
