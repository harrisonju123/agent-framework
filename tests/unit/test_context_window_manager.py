"""Tests for context window management."""

import pytest

from agent_framework.core.context_window_manager import (
    ContextWindowManager,
    ContextPriority,
    ContextItem,
    ContextBudget,
    MEMORY_BUDGET_RATIO,
    MIN_MEMORY_CHARS,
    MAX_MEMORY_CHARS,
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
            peak_input_tokens=3000,
            cumulative_output_tokens=2000,
        )

        assert budget.used_so_far == 5000
        assert budget.utilization_percent == 50.0
        assert budget.remaining == 5000

    def test_budget_near_limit_detection(self):
        """Verify near-limit detection works at 80% threshold."""
        budget = ContextBudget(
            total_budget=10000,
            reserved_for_output=2000,
            available_for_input=8000,
            peak_input_tokens=4000,
            cumulative_output_tokens=4000,
        )

        assert budget.is_near_limit is True  # 80% is at limit

        budget.cumulative_output_tokens = 3999
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
        # Fill budget with critical items (each 125 tokens at ~4 chars/token)
        for i in range(10):
            manager.add_context_item(
                content="A" * 500,  # 125 tokens each
                priority=ContextPriority.CRITICAL,
                category="task_definition",
            )

        # Add low-priority item that won't fit
        manager.add_context_item(
            content="Low priority that won't fit",
            priority=ContextPriority.LOW,
            category="metadata",
        )

        # Budget of 1250 fits exactly the 10 critical items, drops low priority
        context, metadata = manager.build_context(max_tokens=1250)

        assert "Low priority that won't fit" not in context
        assert metadata["items_dropped"] >= 1

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
        """Verify checkpoint fires exactly once when crossing 90% threshold."""
        assert manager.should_trigger_checkpoint() is False  # Under threshold

        manager.update_token_usage(9000, 0)
        assert manager.should_trigger_checkpoint() is True   # First check at 90%
        assert manager.should_trigger_checkpoint() is False   # Already fired — won't repeat

    def test_token_estimation(self, manager):
        """Verify token estimation heuristic."""
        # Empty string
        assert manager._estimate_tokens("") == 0

        # 400 characters at ~4 chars/token = 100 tokens
        text = "x" * 400
        estimated = manager._estimate_tokens(text)
        assert estimated == 100

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

    def test_compute_memory_budget_healthy(self, manager):
        """Verify full (floor-clamped) memory budget when utilization is healthy (<70%).

        10K budget → base = int(10000 * 0.06) = 600, clamped to MIN_MEMORY_CHARS = 800.
        """
        # No usage yet — 0% utilization
        assert manager.compute_memory_budget() == MIN_MEMORY_CHARS

        # 50% utilization
        manager.update_token_usage(2500, 2500)
        assert manager.compute_memory_budget() == MIN_MEMORY_CHARS

        # Right at 69% boundary
        manager.budget.peak_input_tokens = 3450
        manager.budget.cumulative_output_tokens = 3450
        assert manager.budget.utilization_percent == 69.0
        assert manager.compute_memory_budget() == MIN_MEMORY_CHARS

    def test_compute_memory_budget_tight(self, manager):
        """Verify linearly-decayed memory budget when utilization is tight (70-90%).

        10K budget → base = 800 (floor). Decay: int(800 * (90 - util) / 20).
        """
        # Exactly 70% utilization — top of decay range, returns full base
        manager.update_token_usage(3500, 3500)
        assert manager.budget.utilization_percent == 70.0
        assert manager.compute_memory_budget() == 800  # int(800 * 20/20)

        # 80% utilization — midpoint of decay range
        manager.budget.peak_input_tokens = 4000
        manager.budget.cumulative_output_tokens = 4000
        assert manager.budget.utilization_percent == 80.0
        assert manager.compute_memory_budget() == 400  # int(800 * 10/20)

        # Right below 90% boundary — nearly zero
        manager.budget.peak_input_tokens = 4499
        manager.budget.cumulative_output_tokens = 4499
        assert manager.budget.utilization_percent == 89.98
        assert manager.compute_memory_budget() == 0  # int(800 * 0.02/20) = 0

    def test_compute_memory_budget_critical(self, manager):
        """Verify memory budget is zero when utilization is critical (>=90%)."""
        # Exactly 90% utilization
        manager.update_token_usage(4500, 4500)
        assert manager.budget.utilization_percent == 90.0
        assert manager.compute_memory_budget() == 0

        # 95% utilization
        manager.budget.peak_input_tokens = 4750
        manager.budget.cumulative_output_tokens = 4750
        assert manager.budget.utilization_percent == 95.0
        assert manager.compute_memory_budget() == 0

        # 100% utilization
        manager.budget.peak_input_tokens = 5000
        manager.budget.cumulative_output_tokens = 5000
        assert manager.budget.utilization_percent == 100.0
        assert manager.compute_memory_budget() == 0

    def test_compute_memory_budget_edge_cases(self, manager):
        """Verify memory budget at tier boundaries (10K budget, base=800)."""
        # Just below 70% threshold (69.99%) — still in healthy tier
        manager.budget.peak_input_tokens = 3499
        manager.budget.cumulative_output_tokens = 3500
        assert 69.9 <= manager.budget.utilization_percent < 70.0
        assert manager.compute_memory_budget() == MIN_MEMORY_CHARS

        # Exactly 70% — enters decay range but (90-70)/20 = 1.0, so full base returned
        manager.budget.peak_input_tokens = 3500
        manager.budget.cumulative_output_tokens = 3500
        assert manager.budget.utilization_percent == 70.0
        assert manager.compute_memory_budget() == 800  # int(800 * 20/20)

        # Just below 90% threshold (89.99%) — decay is nearly zero
        manager.budget.peak_input_tokens = 4499
        manager.budget.cumulative_output_tokens = 4500
        assert 89.9 <= manager.budget.utilization_percent < 90.0
        assert manager.compute_memory_budget() == 0  # int(800 * 0.01/20) = 0

        # Exactly 90% threshold — hard cutoff
        manager.budget.peak_input_tokens = 4500
        manager.budget.cumulative_output_tokens = 4500
        assert manager.budget.utilization_percent == 90.0
        assert manager.compute_memory_budget() == 0

    @pytest.mark.parametrize("total_budget,expected_chars", [
        (10_000, 800),   # 600 raw → clamped to MIN_MEMORY_CHARS floor
        (30_000, 1800),  # 1800 raw → unclamped
        (50_000, 3000),  # 3000 raw → unclamped (matches pre-scaling default)
        (80_000, 4800),  # 4800 raw → unclamped
    ])
    def test_compute_memory_budget_scales_with_budget(self, total_budget, expected_chars):
        """Verify memory allocation scales proportionally with the task's token budget."""
        m = ContextWindowManager(total_budget=total_budget, output_reserve=2000)
        assert m.compute_memory_budget() == expected_chars

    def test_compute_memory_budget_linear_decay(self):
        """Verify smooth linear decay between 70-90% utilization (50K budget, base=3000)."""
        m = ContextWindowManager(total_budget=50_000, output_reserve=10_000)
        base = 3000

        cases = [
            (70.0, int(base * 20 / 20)),  # 3000
            (75.0, int(base * 15 / 20)),  # 2250
            (80.0, int(base * 10 / 20)),  # 1500
            (85.0, int(base * 5 / 20)),   # 750
        ]
        for utilization, expected in cases:
            # Set peak and cumulative to yield the target utilization
            half = int(utilization / 100 * m.budget.total_budget / 2)
            m.budget.peak_input_tokens = half
            m.budget.cumulative_output_tokens = half
            assert m.compute_memory_budget() == expected, f"Expected {expected} at {utilization}% utilization"

    def test_compute_memory_budget_min_clamp(self):
        """Verify MIN_MEMORY_CHARS floor prevents starvation on very small budgets."""
        m = ContextWindowManager(total_budget=1_000, output_reserve=200)
        # Raw: int(1000 * 0.06) = 60 — well below floor
        assert m.compute_memory_budget() == MIN_MEMORY_CHARS

    def test_compute_memory_budget_max_clamp(self):
        """Verify MAX_MEMORY_CHARS ceiling prevents excessive injection on huge budgets."""
        m = ContextWindowManager(total_budget=200_000, output_reserve=40_000)
        # Raw: int(200000 * 0.06) = 12000 — above ceiling
        assert m.compute_memory_budget() == MAX_MEMORY_CHARS
