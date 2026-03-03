"""Tests for llm_executor.py — LLMExecutionManager and helper functions."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_framework.core.llm_executor import (
    LLMExecutionManager,
    is_productive_command,
)
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# is_productive_command
# ---------------------------------------------------------------------------

class TestIsProductiveCommand:
    @pytest.mark.parametrize("cmd", [
        "pytest tests/",
        "python -m pytest",
        "git commit -m 'fix'",
        "git push origin main",
        "ruff check src/",
        "npm install",
        "cargo build",
        "docker-compose up",
    ])
    def test_productive_commands(self, cmd):
        assert is_productive_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "cat file.txt",
        "echo hello",
        "curl localhost",
        "whoami",
    ])
    def test_non_productive_commands(self, cmd):
        assert is_productive_command(cmd) is False

    def test_case_insensitive(self):
        assert is_productive_command("PYTEST tests/") is True

    def test_leading_whitespace(self):
        assert is_productive_command("  git push") is True


# ---------------------------------------------------------------------------
# LLMExecutionManager construction
# ---------------------------------------------------------------------------

def _make_manager():
    return LLMExecutionManager(
        config=MagicMock(id="engineer", base_id="engineer"),
        llm=MagicMock(),
        git_ops=MagicMock(active_worktree=None, worktree_env_vars=None),
        logger=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )


def _make_task(**ctx_overrides):
    ctx = {"github_repo": "org/repo", **ctx_overrides}
    return Task(
        id="task-exec",
        title="Exec task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        context=ctx,
        created_at=datetime.now(timezone.utc),
        created_by="architect",
        assigned_to="engineer",
        description="Exec task description",
        priority=50,
    )


class TestLLMExecutionManagerInit:
    def test_construction(self):
        mgr = _make_manager()
        assert mgr.config.id == "engineer"

    def test_set_session_logger(self):
        mgr = _make_manager()
        new_logger = MagicMock()
        mgr.set_session_logger(new_logger)
        assert mgr.session_logger is new_logger


# ---------------------------------------------------------------------------
# process_completion
# ---------------------------------------------------------------------------

class TestProcessCompletion:
    def test_logs_completion(self):
        mgr = _make_manager()
        response = LLMResponse(
            content="done",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="end_turn",
            latency_ms=500,
            success=True,
        )
        task = _make_task()
        mgr.process_completion(response, task)
        mgr.session_logger.log.assert_called_once()
        call_args = mgr.session_logger.log.call_args
        assert call_args[0][0] == "llm_complete"

    def test_updates_context_window_manager(self):
        mgr = _make_manager()
        response = LLMResponse(
            content="done",
            model_used="sonnet",
            input_tokens=100,
            output_tokens=50,
            finish_reason="end_turn",
            latency_ms=500,
            success=True,
        )
        task = _make_task()
        cwm = MagicMock()
        cwm.get_budget_status.return_value = {
            "utilization_percent": 50.0,
            "used_so_far": 5000,
            "total_budget": 10000,
            "remaining": 5000,
        }
        cwm.should_trigger_checkpoint.return_value = False
        mgr.process_completion(response, task, context_window_manager=cwm)
        cwm.update_token_usage.assert_called_once_with(100, 50)


# ---------------------------------------------------------------------------
# log_routing_decision
# ---------------------------------------------------------------------------

class TestLogRoutingDecision:
    def test_no_decision_is_noop(self):
        mgr = _make_manager()
        mgr.llm.model_selector = None
        response = MagicMock()
        task = _make_task()
        mgr.log_routing_decision(task, response)
        mgr.session_logger.log.assert_not_called()

    def test_logs_decision_when_present(self):
        mgr = _make_manager()
        decision = MagicMock()
        decision.chosen_tier = "sonnet"
        decision.scores = {"sonnet": 0.9}
        decision.signals = {}
        decision.fallback = False
        selector = MagicMock()
        selector._last_routing_decision = decision
        mgr.llm.model_selector = selector

        response = MagicMock(model_used="sonnet")
        task = _make_task()
        mgr.log_routing_decision(task, response)
        mgr.session_logger.log.assert_called_once()
        assert selector._last_routing_decision is None


# ---------------------------------------------------------------------------
# auto_commit_wip
# ---------------------------------------------------------------------------

class TestAutoCommitWip:
    @pytest.mark.asyncio
    async def test_commits_on_success(self):
        mgr = _make_manager()
        mgr.git_ops.safety_commit.return_value = True
        task = _make_task()
        await mgr.auto_commit_wip(task, Path("/tmp/work"), 15)
        mgr.git_ops.safety_commit.assert_called_once()
        mgr.session_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_swallows_exception(self):
        mgr = _make_manager()
        mgr.git_ops.safety_commit.side_effect = RuntimeError("git error")
        task = _make_task()
        # Should not raise
        await mgr.auto_commit_wip(task, Path("/tmp/work"), 15)


# ---------------------------------------------------------------------------
# Efficiency directive content
# ---------------------------------------------------------------------------

class TestEfficiencyDirective:
    """Verify the append_system_prompt sent to the LLM contains anti-re-read rules."""

    @pytest.mark.asyncio
    async def test_efficiency_directive_contains_anti_reread_rules(self):
        """The LLM request's append_system_prompt should include file-read rules."""
        mgr = _make_manager()

        captured_request = []

        async def _capture_complete(request, *, task_id=None, **kwargs):
            captured_request.append(request)
            return LLMResponse(
                content="done",
                model_used="sonnet",
                input_tokens=100,
                output_tokens=50,
                finish_reason="end_turn",
                latency_ms=100,
                success=True,
            )

        mgr.llm.complete = _capture_complete
        mgr.llm.cancel = MagicMock()
        mgr.llm.get_partial_output = MagicMock(return_value="")

        task = _make_task()

        async def _never_interrupt():
            await asyncio.sleep(999)

        await mgr.execute(
            task, "test prompt", Path("/tmp/work"), None,
            watch_for_interruption_coro=_never_interrupt,
        )

        assert len(captured_request) == 1
        directive = captured_request[0].append_system_prompt
        assert "NEVER read the same file twice" in directive
        assert "session interrupt" in directive
        assert "NEVER use offset or limit" in directive


# ---------------------------------------------------------------------------
# _handle_circuit_breaker: re-read trigger
# ---------------------------------------------------------------------------

class TestHandleCircuitBreakerReread:
    """Verify _handle_circuit_breaker logs reread_interrupt when the trigger is re-reads."""

    @pytest.mark.asyncio
    async def test_reread_trigger_logs_reread_interrupt(self):
        mgr = _make_manager()
        task = _make_task()
        wd = MagicMock()
        wd.exists.return_value = True
        wd.__str__ = lambda self: "/tmp/work"

        # Build a mock checkpoint_mgr that looks like a re-read interrupt
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.consecutive_bash = 0
        checkpoint_mgr.consecutive_diagnostic = 0
        checkpoint_mgr.diagnostic_tripped = False
        checkpoint_mgr._reread_interrupted = True
        checkpoint_mgr.bash_commands = []
        checkpoint_mgr._max_consecutive = 15
        checkpoint_mgr._max_diagnostic = 10
        checkpoint_mgr.get_worst_reread.return_value = ("core/agent.py", 4)
        checkpoint_mgr.get_read_stats.return_value = {"core/agent.py": 4, "core/config.py": 2}

        mgr.git_ops.safety_commit.return_value = False
        mgr.llm.cancel = MagicMock()
        mgr.llm.get_partial_output = MagicMock(return_value="")

        llm_task = asyncio.create_task(asyncio.sleep(999))
        watcher_task = asyncio.create_task(asyncio.sleep(999))

        result = await mgr._handle_circuit_breaker(
            task, wd, checkpoint_mgr, llm_task, watcher_task,
        )

        assert result.success is False
        assert result.finish_reason == "circuit_breaker"
        assert "core/agent.py" in result.error

        # Verify the reread_interrupt session event was logged
        log_calls = mgr.session_logger.log.call_args_list
        reread_calls = [c for c in log_calls if c[0][0] == "reread_interrupt"]
        assert len(reread_calls) == 1
        assert reread_calls[0][1]["worst_file"] == "core/agent.py"
        assert reread_calls[0][1]["worst_count"] == 4
        assert reread_calls[0][1]["read_stats"] == {"core/agent.py": 4, "core/config.py": 2}

        # Verify the circuit_breaker event has trigger="reread"
        cb_calls = [c for c in log_calls if c[0][0] == "circuit_breaker"]
        assert len(cb_calls) == 1
        assert cb_calls[0][1]["trigger"] == "reread"

    @pytest.mark.asyncio
    async def test_non_reread_trigger_skips_reread_interrupt_log(self):
        """When the trigger is volume (not re-read), no reread_interrupt event is logged."""
        mgr = _make_manager()
        task = _make_task()
        wd = MagicMock()
        wd.exists.return_value = True
        wd.__str__ = lambda self: "/tmp/work"

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.consecutive_bash = 15
        checkpoint_mgr.consecutive_diagnostic = 0
        checkpoint_mgr.diagnostic_tripped = False
        checkpoint_mgr._reread_interrupted = False
        checkpoint_mgr.bash_commands = ["git status"] * 15
        checkpoint_mgr._max_consecutive = 15
        checkpoint_mgr._max_diagnostic = 10

        mgr.git_ops.safety_commit.return_value = False
        mgr.llm.cancel = MagicMock()
        mgr.llm.get_partial_output = MagicMock(return_value="")

        llm_task = asyncio.create_task(asyncio.sleep(999))
        watcher_task = asyncio.create_task(asyncio.sleep(999))

        await mgr._handle_circuit_breaker(
            task, wd, checkpoint_mgr, llm_task, watcher_task,
        )

        log_calls = mgr.session_logger.log.call_args_list
        reread_calls = [c for c in log_calls if c[0][0] == "reread_interrupt"]
        assert len(reread_calls) == 0

        cb_calls = [c for c in log_calls if c[0][0] == "circuit_breaker"]
        assert cb_calls[0][1]["trigger"] == "volume"
