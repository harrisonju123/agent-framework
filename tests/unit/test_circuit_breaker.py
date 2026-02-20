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
    a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
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
    a._max_consecutive_diagnostic_calls = 5  # Diagnostic breaker threshold
    a._exploration_alert_threshold = 999  # Don't trigger in circuit breaker tests
    a._exploration_alert_thresholds = {}
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.workspace = Path("/tmp/test-workspace")
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
            task, "implement auth", agent.workspace, None
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
    async def test_diverse_commands_pass_at_threshold(self, agent, task):
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
    async def test_diverse_commands_never_trip(self, agent, task):
        """Fully diverse commands never trip, even far beyond threshold."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 50 unique commands at threshold=5 — high diversity should never trip
        for i in range(50):
            on_tool("Bash", f"unique-command-{i}")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Diverse commands should never trip the circuit breaker"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

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
        """Circuit breaker appends an ActivityEvent with diversity info and working_dir."""
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
        """Circuit breaker logs diversity metrics and working_dir to session logger."""
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
        kwargs = log_calls[0][1]
        assert kwargs["consecutive_bash"] == 5
        assert kwargs["threshold"] == 5
        assert kwargs["unique_commands"] == 1
        assert kwargs["diversity"] == 0.2
        assert kwargs["working_dir"] == str(agent.workspace)
        assert kwargs["working_dir_exists"] is False  # /tmp/test-workspace doesn't exist
        assert kwargs["last_commands"] == ["git status"] * 5


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
    async def test_auto_commit_delegates_to_safety_commit(self):
        """_auto_commit_wip delegates to GitOperationsManager.safety_commit."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = True
        a._session_logger = MagicMock()

        t = Task(
            id="wip-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        await a._auto_commit_wip(t, Path("/tmp/work"), 15)

        a._git_ops.safety_commit.assert_called_once()
        assert "circuit breaker" in a._git_ops.safety_commit.call_args[0][1]
        a._session_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_commit_no_session_log_when_clean(self):
        """_auto_commit_wip skips session log when safety_commit returns False."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = False
        a._session_logger = MagicMock()

        t = Task(
            id="wip-2", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        await a._auto_commit_wip(t, Path("/tmp/work"), 15)

        a._git_ops.safety_commit.assert_called_once()
        a._session_logger.log.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_commit_passes_bash_count(self):
        """_auto_commit_wip includes the bash count in the commit message."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = False
        a._session_logger = MagicMock()

        t = Task(
            id="wip-3", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="test-agent",
            created_at=datetime.now(timezone.utc), title="Test", description="Test",
            context={},
        )

        await a._auto_commit_wip(t, Path("/tmp/work"), 42)

        msg = a._git_ops.safety_commit.call_args[0][1]
        assert "42 consecutive Bash calls" in msg


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


class TestCircuitBreakerPartialOutput:
    """Circuit breaker harvests partial LLM output before killing the session."""

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_captures_partial_output_on_trip(self, _mock_record, agent, task):
        """Partial progress is stored in task context when circuit breaker fires."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Override get_partial_output to return meaningful content
        agent.llm.get_partial_output.return_value = (
            "I analyzed the auth module and found:\n"
            "[Tool Call: Read src/auth.py]\n"
            "The existing implementation uses session-based auth.\n"
            "I'll refactor to JWT-based tokens."
        )
        agent._extract_partial_progress = Agent._extract_partial_progress

        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        assert "_previous_attempt_summary" in task.context
        summary = task.context["_previous_attempt_summary"]
        # Tool call noise should be filtered out
        assert "[Tool Call:" not in summary
        assert "auth" in summary.lower()

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_no_partial_output_leaves_context_empty(self, _mock_record, agent, task):
        """Empty partial output does not populate _previous_attempt_summary."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        agent.llm.get_partial_output.return_value = ""

        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        assert "_previous_attempt_summary" not in task.context

    @pytest.mark.asyncio
    async def test_partial_output_exception_non_fatal(self, agent, task):
        """get_partial_output raising does not prevent circuit breaker response."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        agent.llm.get_partial_output.side_effect = RuntimeError("buffer gone")

        for _ in range(5):
            on_tool("Bash", "git status")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        # Should still return the circuit breaker response
        assert result.success is False
        assert result.finish_reason == "circuit_breaker"


class TestCircuitBreakerProductiveCommands:
    """Productive commands (test/build/lint) get a higher threshold before tripping."""

    @pytest.mark.asyncio
    async def test_productive_commands_bypass_at_threshold(self, agent, task):
        """All-productive commands at base threshold → deferred (effective=15)."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 5× pytest at threshold=5 → productive_ratio=1.0 → effective=15
        for _ in range(5):
            on_tool("Bash", "pytest tests/ -v")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Productive commands should not trip at base threshold"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_productive_commands_trip_at_hard_ceiling(self, agent, task):
        """All-productive commands at 3× threshold → trip (hard ceiling)."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 15× pytest at threshold=5 → effective=15 → trip
        for _ in range(15):
            on_tool("Bash", "pytest tests/ -v")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_non_productive_commands_trip_normally(self, agent, task):
        """Non-productive repeated commands trip at base threshold."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 5× ls at threshold=5 → productive_ratio=0.0 → trip immediately
        for _ in range(5):
            on_tool("Bash", "ls -la")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_mixed_commands_below_productive_threshold(self, agent, task):
        """60% productive (below 70% threshold) → trips at base threshold."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 3 productive + 2 non-productive = 60% productive < 70% → trips normally
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "ls -la")
        on_tool("Bash", "ls -la")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_mixed_commands_above_productive_threshold(self, agent, task):
        """80% productive (above 70% threshold) → deferred."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # 4 productive + 1 non-productive = 80% productive > 70% → deferred
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "pytest tests/")
        on_tool("Bash", "ls -la")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "80% productive should defer the circuit breaker"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_productive_ratio_logged_on_deferral(self, agent, task):
        """Deferral log message includes productive_ratio."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "pytest tests/ -v")

        await asyncio.sleep(0.05)

        # Check that the info log with productive_ratio was emitted
        info_calls = [
            str(c) for c in agent.logger.info.call_args_list
            if "productive_ratio" in str(c)
        ]
        assert len(info_calls) >= 1, "Should log productive_ratio on deferral"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_productive_ratio_in_activity_event(self, agent, task):
        """Activity event includes productive_ratio when circuit breaker trips."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        # Trip with non-productive, non-diagnostic commands
        for _ in range(5):
            on_tool("Bash", "cat foo.py")

        await asyncio.wait_for(result_task, timeout=5.0)

        event = agent.activity_manager.append_event.call_args[0][0]
        assert "productive_ratio=" in event.title

    @pytest.mark.asyncio
    async def test_productive_ratio_in_session_log(self, agent, task):
        """Session log includes productive_ratio field."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.wait_for(result_task, timeout=5.0)

        log_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0][0] == "circuit_breaker"
        ]
        assert len(log_calls) == 1
        assert "productive_ratio" in log_calls[0][1]


class TestDiagnosticCircuitBreaker:
    """Diagnostic circuit breaker trips on consecutive environment-probing commands."""

    @pytest.mark.asyncio
    async def test_pwd_trips_at_diagnostic_threshold(self, agent, task):
        """pwd x5 trips diagnostic breaker even though diversity=1.0."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "pwd")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"
        assert "diagnostic" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_prefixed_commands_trip(self, agent, task):
        """/bin/echo x5 trips — path prefix is stripped before matching."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "/usr/bin/echo hello")

        result = await asyncio.wait_for(result_task, timeout=5.0)

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"

    @pytest.mark.asyncio
    async def test_non_diagnostic_does_not_trip(self, agent, task):
        """git status x5 does NOT trip the diagnostic breaker."""
        # Raise volume threshold so only the diagnostic breaker is under test
        agent._max_consecutive_tool_calls = 50
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "git status")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Non-diagnostic commands should not trip diagnostic breaker"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_counter_resets_on_non_bash(self, agent, task):
        """Diagnostic counter resets on non-Bash tool: pwd x4 → Read → pwd x4 → no trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(4):
            on_tool("Bash", "pwd")
        on_tool("Read", "some/file.py")  # Resets both counters
        for _ in range(4):
            on_tool("Bash", "pwd")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Diagnostic breaker should not trip after reset"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_counter_resets_on_non_diagnostic_bash(self, agent, task):
        """Diagnostic counter resets on non-diagnostic Bash: pwd x4 → git status → pwd x4."""
        # Raise volume threshold so only the diagnostic breaker is under test
        agent._max_consecutive_tool_calls = 50
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(4):
            on_tool("Bash", "pwd")
        on_tool("Bash", "git status")  # Non-diagnostic Bash resets diagnostic counter
        for _ in range(4):
            on_tool("Bash", "pwd")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Diagnostic breaker should not trip after non-diagnostic reset"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_session_log_has_diagnostic_trigger(self, agent, task):
        """Session log has trigger='diagnostic' field on diagnostic trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "pwd")

        await asyncio.wait_for(result_task, timeout=5.0)

        log_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0][0] == "circuit_breaker"
        ]
        assert len(log_calls) == 1
        assert log_calls[0][1]["trigger"] == "diagnostic"
        assert log_calls[0][1]["consecutive_diagnostic"] == 5
        assert log_calls[0][1]["threshold"] == 5  # diagnostic threshold, not volume
        kwargs = log_calls[0][1]
        assert kwargs["working_dir"] == str(agent.workspace)
        assert kwargs["working_dir_exists"] is False
        assert kwargs["last_commands"] == ["pwd"] * 5

    @pytest.mark.asyncio
    async def test_activity_event_title_contains_diagnostic(self, agent, task):
        """Activity event title contains 'Diagnostic' on diagnostic trip."""
        result_task, on_tool = await _setup_and_get_callback(agent, task)

        for _ in range(5):
            on_tool("Bash", "ls")

        await asyncio.wait_for(result_task, timeout=5.0)

        agent.activity_manager.append_event.assert_called_once()
        event = agent.activity_manager.append_event.call_args[0][0]
        assert "Diagnostic" in event.title

    def test_config_default(self):
        """SafeguardsConfig has max_consecutive_diagnostic_calls=5 by default."""
        from agent_framework.core.config import SafeguardsConfig
        cfg = SafeguardsConfig()
        assert cfg.max_consecutive_diagnostic_calls == 5


def _make_exploration_agent(threshold: int = 10, thresholds: dict | None = None):
    """Create a mock agent for exploration alert tests.

    Circuit breaker thresholds are set high so only the exploration alert is under test.
    """
    a = MagicMock()
    a._execute_llm_with_interruption_watch = (
        Agent._execute_llm_with_interruption_watch.__get__(a)
    )
    a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
    a._auto_commit_wip = AsyncMock()
    a._update_phase = MagicMock()
    a._session_logger = MagicMock()
    a._session_logger.log = MagicMock()
    a._session_logger.log_tool_call = MagicMock()
    a._context_window_manager = None
    a._current_specialization = None
    a._current_file_count = 0
    a._mcp_enabled = False
    a._max_consecutive_tool_calls = 999
    a._max_consecutive_diagnostic_calls = 999
    a._exploration_alert_threshold = threshold
    a._exploration_alert_thresholds = thresholds or {}
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.config.base_id = "engineer"
    a.workspace = Path("/tmp/test-workspace")
    a.logger = MagicMock()
    a.queue = MagicMock()
    a.activity_manager = MagicMock()
    return a


class TestExplorationAlert:
    """Exploration alert fires once when total tool calls exceed threshold."""

    @pytest.mark.asyncio
    async def test_alert_fires_at_threshold(self, task):
        """10 diverse tool calls → session log event + activity event."""
        a = _make_exploration_agent(threshold=10)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(10):
            on_tool("Read", f"file_{i}.py")

        await asyncio.sleep(0.05)

        # Verify session log event
        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["total_tool_calls"] == 10
        assert alert_logs[0][1]["threshold"] == 10

        # Verify activity event
        alert_events = [
            c for c in a.activity_manager.append_event.call_args_list
            if c[0][0].type == "exploration_alert"
        ]
        assert len(alert_events) == 1
        assert "10 calls" in alert_events[0][0][0].title
        assert "standalone" in alert_events[0][0][0].title

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_alert_fires_once(self, task):
        """20 calls past threshold=10 → only 1 alert."""
        a = _make_exploration_agent(threshold=10)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(20):
            on_tool("Read", f"file_{i}.py")

        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1, "Alert should fire exactly once"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, task):
        """49 calls with threshold=50 → no alert."""
        a = _make_exploration_agent(threshold=50)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(49):
            on_tool("Read", f"file_{i}.py")

        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 0, "No alert below threshold"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_agent_continues_after_alert(self, task):
        """LLM task is NOT cancelled after alert — agent keeps working."""
        a = _make_exploration_agent(threshold=10)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(15):
            on_tool("Read", f"file_{i}.py")

        await asyncio.sleep(0.05)
        assert not result_task.done(), "Agent should continue working after alert"

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    def test_config_default_threshold(self):
        """OptimizationConfig has exploration_alert_threshold=50 by default."""
        from agent_framework.core.config import OptimizationConfig
        cfg = OptimizationConfig()
        assert cfg.exploration_alert_threshold == 50
        assert cfg.exploration_alert_thresholds == {
            "plan": 80,
            "implement": 50,
            "code_review": 40,
            "qa_review": 40,
            "create_pr": 25,
        }

    @pytest.mark.asyncio
    async def test_plan_step_uses_plan_threshold(self):
        """Plan step with threshold=80: 60 calls → no alert; 85 calls → alert."""
        thresholds = {"plan": 80, "implement": 50}
        task = Task(
            id="test-plan-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Plan task",
            description="Plan",
            context={"github_repo": "org/repo", "workflow_step": "plan"},
        )
        a = _make_exploration_agent(threshold=50, thresholds=thresholds)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        # 60 calls — below plan threshold of 80
        for i in range(60):
            on_tool("Read", f"file_{i}.py")
        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 0, "60 calls should not trigger plan threshold of 80"

        # 25 more calls → 85 total, exceeds plan threshold
        for i in range(25):
            on_tool("Read", f"extra_{i}.py")
        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["workflow_step"] == "plan"
        assert alert_logs[0][1]["threshold"] == 80

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_implement_step_uses_implement_threshold(self):
        """Implement step fires alert at implement threshold (50)."""
        thresholds = {"plan": 80, "implement": 50}
        task = Task(
            id="test-impl-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Implement task",
            description="Implement",
            context={"github_repo": "org/repo", "workflow_step": "implement"},
        )
        a = _make_exploration_agent(threshold=999, thresholds=thresholds)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(50):
            on_tool("Read", f"file_{i}.py")
        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["workflow_step"] == "implement"
        assert alert_logs[0][1]["threshold"] == 50

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_standalone_task_uses_global_default(self):
        """Task without workflow_step falls back to global scalar threshold."""
        thresholds = {"plan": 80, "implement": 50}
        task = Task(
            id="test-standalone-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Standalone task",
            description="Standalone",
            context={"github_repo": "org/repo"},
        )
        a = _make_exploration_agent(threshold=30, thresholds=thresholds)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(30):
            on_tool("Read", f"file_{i}.py")
        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["workflow_step"] is None
        assert alert_logs[0][1]["threshold"] == 30

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_event_includes_workflow_step_and_agent_type(self):
        """Alert events carry workflow_step and agent_type for analytics."""
        thresholds = {"implement": 10}
        task = Task(
            id="test-enriched-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Enriched event task",
            description="Test enrichment",
            context={"github_repo": "org/repo", "workflow_step": "implement"},
        )
        a = _make_exploration_agent(threshold=999, thresholds=thresholds)
        a.config.base_id = "engineer"
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(10):
            on_tool("Read", f"file_{i}.py")
        await asyncio.sleep(0.05)

        # Session log fields
        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["workflow_step"] == "implement"
        assert alert_logs[0][1]["agent_type"] == "engineer"

        # Activity event title
        alert_events = [
            c for c in a.activity_manager.append_event.call_args_list
            if c[0][0].type == "exploration_alert"
        ]
        assert len(alert_events) == 1
        assert "step=implement" in alert_events[0][0][0].title

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_unknown_step_falls_back_to_global(self):
        """Workflow step not in thresholds dict falls back to scalar default."""
        thresholds = {"plan": 80, "implement": 50}
        task = Task(
            id="test-unknown-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Unknown step task",
            description="Unknown step",
            context={"github_repo": "org/repo", "workflow_step": "lint"},
        )
        # Global default=30, thresholds dict has no "lint" entry
        a = _make_exploration_agent(threshold=30, thresholds=thresholds)
        result_task, on_tool = await _setup_and_get_callback(a, task)

        for i in range(30):
            on_tool("Read", f"file_{i}.py")
        await asyncio.sleep(0.05)

        alert_logs = [
            c for c in a._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_logs) == 1
        assert alert_logs[0][1]["workflow_step"] == "lint"
        assert alert_logs[0][1]["threshold"] == 30

        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass
