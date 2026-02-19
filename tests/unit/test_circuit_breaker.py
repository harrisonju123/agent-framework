"""Tests for the stuck-agent circuit breaker in _execute_llm_with_interruption_watch."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


@pytest.fixture
def task():
    return Task(
        id="test-cb-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.now(timezone.utc),
        title="Implement auth feature",
        description="Add authentication",
        context={"github_repo": "org/repo"},
    )


@pytest.fixture
def agent():
    """Minimal mock agent with the real _execute_llm_with_interruption_watch bound."""
    a = MagicMock()
    a._execute_llm_with_interruption_watch = (
        Agent._execute_llm_with_interruption_watch.__get__(a)
    )
    a._update_phase = MagicMock()
    a._session_logger = MagicMock()
    a._session_logger.log = MagicMock()
    a._session_logger.log_tool_call = MagicMock()
    a._context_window_manager = None
    a._current_specialization = None
    a._current_file_count = 0
    a._mcp_enabled = False
    a._max_consecutive_tool_calls = 5  # Low threshold for fast tests
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.logger = MagicMock()
    a.queue = MagicMock()
    a.activity_manager = MagicMock()
    return a


def _make_slow_llm(on_tool_activity_ref):
    """Create a mock LLM whose complete() captures on_tool_activity then hangs.

    The caller can fire tool callbacks to simulate Claude Code tool use while
    the LLM coroutine is "running".
    """
    llm = MagicMock()

    async def _complete(*args, **kwargs):
        on_tool_activity_ref.append(kwargs.get("on_tool_activity"))
        # Hang until cancelled — circuit breaker or test timeout will end this
        await asyncio.sleep(999)

    llm.complete = _complete
    llm.cancel = MagicMock()
    llm.get_partial_output = MagicMock(return_value="")
    return llm


class TestCircuitBreakerFires:
    """Circuit breaker trips after N consecutive Bash calls."""

    @pytest.mark.asyncio
    async def test_fires_after_threshold(self, agent, task):
        """Consecutive Bash calls >= threshold → synthetic failure returned."""
        cb_ref = []
        agent.llm = _make_slow_llm(cb_ref)
        async def _never_interrupt():
            await asyncio.sleep(999)
        agent._watch_for_interruption = _never_interrupt

        async def _run():
            return await agent._execute_llm_with_interruption_watch(
                task, "implement auth", MagicMock(), None
            )

        # Start the method — it will block on the three-way race
        result_task = asyncio.create_task(_run())

        # Wait for the callback to be captured
        for _ in range(50):
            if cb_ref:
                break
            await asyncio.sleep(0.01)
        assert cb_ref, "on_tool_activity callback not captured"

        on_tool = cb_ref[0]

        # Fire consecutive Bash calls to hit the threshold (5)
        for i in range(5):
            on_tool("Bash", f"git status attempt {i}")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result is not None
        assert result.success is False
        assert result.finish_reason == "circuit_breaker"
        assert "consecutive Bash calls" in result.error
        agent.llm.cancel.assert_called()

    @pytest.mark.asyncio
    async def test_non_bash_resets_counter(self, agent, task):
        """Interleaving a non-Bash tool resets the counter — no trip."""
        cb_ref = []
        agent.llm = _make_slow_llm(cb_ref)
        async def _never_interrupt():
            await asyncio.sleep(999)
        agent._watch_for_interruption = _never_interrupt

        async def _run():
            return await agent._execute_llm_with_interruption_watch(
                task, "implement auth", MagicMock(), None
            )

        result_task = asyncio.create_task(_run())

        for _ in range(50):
            if cb_ref:
                break
            await asyncio.sleep(0.01)
        assert cb_ref

        on_tool = cb_ref[0]

        # Fire 4 Bash calls (threshold=5), then a Read, then 4 more Bash
        for i in range(4):
            on_tool("Bash", f"cmd {i}")
        on_tool("Read", "some/file.py")  # Resets counter
        for i in range(4):
            on_tool("Bash", f"cmd {i}")

        # Give a moment for the event loop — circuit breaker should NOT have tripped
        await asyncio.sleep(0.05)
        assert not result_task.done(), "Circuit breaker should not have tripped"

        # Clean up
        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass


class TestCircuitBreakerEvents:
    """Circuit breaker emits the right activity event and session log."""

    @pytest.mark.asyncio
    async def test_emits_activity_event(self, agent, task):
        """Circuit breaker appends an ActivityEvent with type='circuit_breaker'."""
        cb_ref = []
        agent.llm = _make_slow_llm(cb_ref)
        async def _never_interrupt():
            await asyncio.sleep(999)
        agent._watch_for_interruption = _never_interrupt

        async def _run():
            return await agent._execute_llm_with_interruption_watch(
                task, "implement auth", MagicMock(), None
            )

        result_task = asyncio.create_task(_run())

        for _ in range(50):
            if cb_ref:
                break
            await asyncio.sleep(0.01)

        on_tool = cb_ref[0]
        for i in range(5):
            on_tool("Bash", f"diagnostic {i}")

        await asyncio.wait_for(result_task, timeout=5.0)

        # Verify activity event
        agent.activity_manager.append_event.assert_called_once()
        event = agent.activity_manager.append_event.call_args[0][0]
        assert event.type == "circuit_breaker"
        assert event.agent == "test-agent"
        assert event.task_id == "test-cb-1"
        assert "5 consecutive Bash calls" in event.title

    @pytest.mark.asyncio
    async def test_logs_to_session_logger(self, agent, task):
        """Circuit breaker logs a 'circuit_breaker' event to session logger."""
        cb_ref = []
        agent.llm = _make_slow_llm(cb_ref)
        async def _never_interrupt():
            await asyncio.sleep(999)
        agent._watch_for_interruption = _never_interrupt

        async def _run():
            return await agent._execute_llm_with_interruption_watch(
                task, "implement auth", MagicMock(), None
            )

        result_task = asyncio.create_task(_run())

        for _ in range(50):
            if cb_ref:
                break
            await asyncio.sleep(0.01)

        on_tool = cb_ref[0]
        for i in range(5):
            on_tool("Bash", f"whoami {i}")

        await asyncio.wait_for(result_task, timeout=5.0)

        # Find the circuit_breaker log call
        log_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0][0] == "circuit_breaker"
        ]
        assert len(log_calls) == 1
        assert log_calls[0][1]["consecutive_bash"] == 5
        assert log_calls[0][1]["threshold"] == 5


class TestCircuitBreakerConfig:
    """Config field wiring."""

    def test_safeguards_config_default(self):
        from agent_framework.core.config import SafeguardsConfig
        cfg = SafeguardsConfig()
        assert cfg.max_consecutive_tool_calls == 15

    def test_safeguards_config_custom(self):
        from agent_framework.core.config import SafeguardsConfig
        cfg = SafeguardsConfig(max_consecutive_tool_calls=25)
        assert cfg.max_consecutive_tool_calls == 25
