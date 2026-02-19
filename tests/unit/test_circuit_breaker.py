"""Tests for the stuck-agent circuit breaker in _execute_llm_with_interruption_watch."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

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
    a._auto_commit_wip = AsyncMock()
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


async def _setup_and_get_callback(agent, task):
    """Start the LLM execution and return (result_task, on_tool_activity callback)."""
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
    assert cb_ref, "on_tool_activity callback not captured"

    return result_task, cb_ref[0]


class TestCircuitBreakerFires:
    """Circuit breaker trips after N consecutive Bash calls with low diversity."""

    @pytest.mark.asyncio
    async def test_fires_after_threshold_low_diversity(self, agent, task):
        """Repeated commands at threshold → low diversity → trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Same command repeated 5 times → diversity = 1/5 = 0.2
        for _ in range(5):
            on_tool("Bash", "git status")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result is not None
        assert result.success is False
        assert result.finish_reason == "circuit_breaker"
        assert "consecutive Bash calls" in result.error
        agent.llm.cancel.assert_called()

    @pytest.mark.asyncio
    async def test_non_bash_resets_counter(self, agent, task):
        """Interleaving a non-Bash tool resets the counter — no trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Fire 4 Bash calls (threshold=5), then a Read, then 4 more Bash
        for i in range(4):
            on_tool("Bash", "git status")
        on_tool("Read", "some/file.py")  # Resets counter
        for i in range(4):
            on_tool("Bash", "git status")

        # Give a moment for the event loop — circuit breaker should NOT have tripped
        await asyncio.sleep(0.05)
        assert not result_task.done(), "Circuit breaker should not have tripped"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_diverse_commands_defer_to_hard_cap(self, agent, task):
        """All-unique commands at threshold → high diversity → no trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 5 unique commands at threshold=5 → diversity = 5/5 = 1.0 > 0.5
        for i in range(5):
            on_tool("Bash", f"unique-command-{i}")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Should not trip with high diversity"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_hard_cap_trips_despite_diversity(self, agent, task):
        """Even fully diverse commands trip at hard cap (2x threshold)."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 10 unique commands at threshold=5 → hard cap = 10
        for i in range(10):
            on_tool("Bash", f"unique-command-{i}")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"
        assert "diversity=1.00" in result.error
        agent.llm.cancel.assert_called()

    @pytest.mark.asyncio
    async def test_diversity_at_boundary_trips(self, agent, task):
        """Exactly 0.5 diversity (<=) → trips at soft threshold."""
        # threshold=5, need exactly 0.5 diversity at count=6
        # 3 unique out of 6 = 0.5 — but we need to reach threshold first.
        # At count=5: need 2-3 unique. Use 2 unique commands repeated.
        # Actually at count=5: 2 unique → 0.4, 3 unique → 0.6
        # We need exactly 0.5 at count >= 5. So at count=6: 3 unique = 0.5
        # But the check happens at each Bash call >= threshold.
        # At count=5 with pattern [A, A, B, B, C]: 3/5 = 0.6 → no trip
        # At count=6 with pattern [A, A, B, B, C, C]: 3/6 = 0.5 → trips
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 3 commands each repeated twice → diversity = 3/6 = 0.5
        for cmd in ["cmd-a", "cmd-a", "cmd-b", "cmd-b", "cmd-c", "cmd-c"]:
            on_tool("Bash", cmd)

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_non_bash_resets_command_history(self, agent, task):
        """Non-Bash tool resets both counter AND command list."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Build up 4 repeated commands (would be low diversity if counted)
        for _ in range(4):
            on_tool("Bash", "git status")

        # Reset with a non-Bash tool
        on_tool("Read", "file.py")

        # Now 5 unique commands — should NOT trip (diversity = 1.0 > 0.5)
        for i in range(5):
            on_tool("Bash", f"fresh-unique-{i}")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Previous repeated commands should not affect post-reset diversity"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass


class TestCircuitBreakerEvents:
    """Circuit breaker emits the right activity event and session log."""

    @pytest.mark.asyncio
    async def test_emits_activity_event(self, agent, task):
        """Circuit breaker appends an ActivityEvent with diversity info."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Repeated commands for low diversity
        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        agent.activity_manager.append_event.assert_called_once()
        event = agent.activity_manager.append_event.call_args[0][0]
        assert event.type == "circuit_breaker"
        assert event.agent == "test-agent"
        assert event.task_id == "test-cb-1"
        assert "5 consecutive Bash calls" in event.title
        assert "diversity=" in event.title

    @pytest.mark.asyncio
    async def test_logs_to_session_logger(self, agent, task):
        """Circuit breaker logs diversity metrics to session logger."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Repeated commands for low diversity
        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        log_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0][0] == "circuit_breaker"
        ]
        assert len(log_calls) == 1
        assert log_calls[0][1]["consecutive_bash"] == 5
        assert log_calls[0][1]["threshold"] == 5
        assert log_calls[0][1]["unique_commands"] == 1
        assert log_calls[0][1]["diversity"] == 0.2


class TestCircuitBreakerWipCommit:
    """Circuit breaker auto-commits WIP before killing the session."""

    @pytest.mark.asyncio
    async def test_auto_commit_called_on_trip(self, agent, task):
        """_auto_commit_wip is called before LLM cancellation."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        agent._auto_commit_wip.assert_awaited_once()
        call_args = agent._auto_commit_wip.call_args
        assert call_args[0][0] == task
        assert call_args[0][2] == 5  # bash_count

    @pytest.mark.asyncio
    async def test_auto_commit_with_changes(self):
        """_auto_commit_wip commits when git status shows changes."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a.logger = MagicMock()
        a._session_logger = MagicMock()

        t = Task(
            id="wip-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout=" M file.py\n")
            await a._auto_commit_wip(t, Path("/tmp/work"), 15)

            assert mock_git.call_count == 3  # status, add, commit
            commit_call = mock_git.call_args_list[2]
            assert "WIP: auto-save" in commit_call[0][0][2]

    @pytest.mark.asyncio
    async def test_auto_commit_no_changes(self):
        """_auto_commit_wip skips commit when working tree is clean."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a.logger = MagicMock()
        a._session_logger = MagicMock()

        t = Task(
            id="wip-2", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(stdout="")
            await a._auto_commit_wip(t, Path("/tmp/work"), 15)

            # Only status check, no add/commit
            assert mock_git.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_commit_exception_non_fatal(self):
        """_auto_commit_wip swallows exceptions — never propagates."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a.logger = MagicMock()
        a._session_logger = MagicMock()

        t = Task(
            id="wip-3", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        with patch("agent_framework.utils.subprocess_utils.run_git_command", side_effect=Exception("git broken")):
            # Should not raise
            await a._auto_commit_wip(t, Path("/tmp/work"), 15)

        a.logger.debug.assert_called()


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
