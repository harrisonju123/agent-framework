"""Tests for the exception guards around _cleanup_task_execution and the run() polling loop.

These guard two crash paths confirmed by QA agent PID 7169:
  1. _cleanup_task_execution raises inside a finally block → escapes to run() → loop dies
  2. _handle_task raises despite the finally guard → run() loop dies
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(**overrides):
    defaults = dict(
        id="task-guard-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Guard test task",
        description="Task used to test exception guards",
        retry_count=0,
        replan_history=[],
        acceptance_criteria=[],
        notes=[],
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_agent_with_real_cleanup_guard():
    """Agent mock with the real _handle_task finally guard bound."""
    agent = MagicMock()
    agent._handle_task = Agent._handle_task.__get__(agent)
    return agent


class TestCleanupGuardInFinallyBlock:
    """_cleanup_task_execution raising must not escape the finally block."""

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_propagate(self):
        """Exception from cleanup must be swallowed, not re-raised."""
        agent = _make_agent_with_real_cleanup_guard()
        task = _make_task()
        lock = MagicMock()

        # Make the body succeed, cleanup blow up
        agent._normalize_workflow = MagicMock()
        agent._validate_task_or_reject = MagicMock(return_value=True)
        agent._setup_task_context = MagicMock()
        agent._setup_context_window_manager_for_task = MagicMock()
        agent._initialize_task_execution = MagicMock()
        agent._git_ops.get_working_directory = MagicMock(return_value="/tmp/work")
        agent._try_index_codebase = MagicMock()
        agent._prompt_builder.build = MagicMock(return_value="prompt")
        agent._prompt_builder.get_current_specialization = MagicMock(return_value=None)
        agent._prompt_builder.get_current_file_count = MagicMock(return_value=0)
        agent._compose_team_for_task = MagicMock(return_value=[])
        response = MagicMock(success=True)
        agent._execute_llm_with_interruption_watch = AsyncMock(return_value=response)
        agent._process_llm_completion = MagicMock()
        agent._handle_successful_response = AsyncMock()
        agent._context_window_manager = MagicMock()
        agent._session_logger = MagicMock()
        agent._cleanup_task_execution = MagicMock(side_effect=OSError("disk full"))

        # Must not raise
        await agent._handle_task(task, lock=lock)

        agent.logger.error.assert_called()
        error_msg = agent.logger.error.call_args[0][0]
        assert "task-guard-001" in error_msg

    @pytest.mark.asyncio
    async def test_lock_released_when_cleanup_fails(self):
        """Lock must be released directly when cleanup raises before releasing it."""
        agent = _make_agent_with_real_cleanup_guard()
        task = _make_task()
        lock = MagicMock()

        agent._normalize_workflow = MagicMock()
        agent._validate_task_or_reject = MagicMock(return_value=True)
        agent._setup_task_context = MagicMock()
        agent._setup_context_window_manager_for_task = MagicMock()
        agent._initialize_task_execution = MagicMock()
        agent._git_ops.get_working_directory = MagicMock(return_value="/tmp/work")
        agent._try_index_codebase = MagicMock()
        agent._prompt_builder.build = MagicMock(return_value="prompt")
        agent._prompt_builder.get_current_specialization = MagicMock(return_value=None)
        agent._prompt_builder.get_current_file_count = MagicMock(return_value=0)
        agent._compose_team_for_task = MagicMock(return_value=[])
        response = MagicMock(success=True)
        agent._execute_llm_with_interruption_watch = AsyncMock(return_value=response)
        agent._process_llm_completion = MagicMock()
        agent._handle_successful_response = AsyncMock()
        agent._context_window_manager = MagicMock()
        agent._session_logger = MagicMock()
        agent._cleanup_task_execution = MagicMock(side_effect=OSError("disk full"))

        await agent._handle_task(task, lock=lock)

        agent.queue.release_lock.assert_called_once_with(lock)

    @pytest.mark.asyncio
    async def test_current_task_id_cleared_when_cleanup_fails(self):
        """_current_task_id must be cleared even when cleanup raises."""
        agent = _make_agent_with_real_cleanup_guard()
        task = _make_task()
        lock = MagicMock()

        agent._normalize_workflow = MagicMock()
        agent._validate_task_or_reject = MagicMock(return_value=True)
        agent._setup_task_context = MagicMock()
        agent._setup_context_window_manager_for_task = MagicMock()
        agent._initialize_task_execution = MagicMock()
        agent._git_ops.get_working_directory = MagicMock(return_value="/tmp/work")
        agent._try_index_codebase = MagicMock()
        agent._prompt_builder.build = MagicMock(return_value="prompt")
        agent._prompt_builder.get_current_specialization = MagicMock(return_value=None)
        agent._prompt_builder.get_current_file_count = MagicMock(return_value=0)
        agent._compose_team_for_task = MagicMock(return_value=[])
        response = MagicMock(success=True)
        agent._execute_llm_with_interruption_watch = AsyncMock(return_value=response)
        agent._process_llm_completion = MagicMock()
        agent._handle_successful_response = AsyncMock()
        agent._context_window_manager = MagicMock()
        agent._session_logger = MagicMock()
        agent._cleanup_task_execution = MagicMock(side_effect=OSError("disk full"))

        await agent._handle_task(task, lock=lock)

        assert agent._current_task_id is None

    @pytest.mark.asyncio
    async def test_lock_release_failure_is_logged_not_raised(self):
        """Secondary failure in release_lock must be logged at DEBUG, not raised."""
        agent = _make_agent_with_real_cleanup_guard()
        task = _make_task()
        lock = MagicMock()

        agent._normalize_workflow = MagicMock()
        agent._validate_task_or_reject = MagicMock(return_value=True)
        agent._setup_task_context = MagicMock()
        agent._setup_context_window_manager_for_task = MagicMock()
        agent._initialize_task_execution = MagicMock()
        agent._git_ops.get_working_directory = MagicMock(return_value="/tmp/work")
        agent._try_index_codebase = MagicMock()
        agent._prompt_builder.build = MagicMock(return_value="prompt")
        agent._prompt_builder.get_current_specialization = MagicMock(return_value=None)
        agent._prompt_builder.get_current_file_count = MagicMock(return_value=0)
        agent._compose_team_for_task = MagicMock(return_value=[])
        response = MagicMock(success=True)
        agent._execute_llm_with_interruption_watch = AsyncMock(return_value=response)
        agent._process_llm_completion = MagicMock()
        agent._handle_successful_response = AsyncMock()
        agent._context_window_manager = MagicMock()
        agent._session_logger = MagicMock()
        agent._cleanup_task_execution = MagicMock(side_effect=OSError("disk full"))
        agent.queue.release_lock = MagicMock(side_effect=RuntimeError("lock gone"))

        # Both failures must be swallowed
        await agent._handle_task(task, lock=lock)

        debug_calls = [str(c) for c in agent.logger.debug.call_args_list]
        assert any("task-guard-001" in c for c in debug_calls)


class TestPollingLoopGuard:
    """_handle_task escaping must not kill the run() polling loop."""

    def _make_run_agent(self, handle_task_side_effect):
        """Agent mock with enough state for run() to iterate once then stop."""
        agent = MagicMock()
        agent._running = True
        agent._paused = False
        agent.config.poll_interval = 0
        agent.config.queue = "engineer"
        agent.config.id = "eng-01"

        task = _make_task()
        lock = MagicMock()

        agent._check_pause_signal = MagicMock(return_value=False)
        agent._write_heartbeat = MagicMock()
        agent._review_cycle.purge_orphaned_review_tasks = MagicMock()
        agent.activity_manager.update_activity = MagicMock()

        # Claim returns task on first call, then agent stops
        call_count = {"n": 0}

        def claim_side_effect(queue, agent_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return (task, lock)
            agent._running = False
            return None

        agent.queue.claim = MagicMock(side_effect=claim_side_effect)
        agent._handle_task = AsyncMock(side_effect=handle_task_side_effect)
        agent.run = Agent.run.__get__(agent)

        return agent, task, lock

    @pytest.mark.asyncio
    async def test_handle_task_exception_does_not_crash_loop(self):
        """An exception escaping _handle_task must be caught; loop continues."""
        agent, task, lock = self._make_run_agent(RuntimeError("escaped"))

        # run() must return normally (loop exits because _running → False)
        await agent.run()

        agent.logger.error.assert_called()
        error_msg = agent.logger.error.call_args[0][0]
        assert task.id in error_msg

    @pytest.mark.asyncio
    async def test_lock_released_when_handle_task_escapes(self):
        """Lock must be released in the outer catch when _handle_task escapes."""
        agent, task, lock = self._make_run_agent(RuntimeError("escaped"))

        await agent.run()

        agent.queue.release_lock.assert_called_with(lock)
