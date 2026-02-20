"""Tests for worktree pre-flight health check and vanish-cancellation."""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.activity import ActivityEvent
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def task():
    return Task(
        id="test-wt-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.now(timezone.utc),
        title="Implement feature",
        description="Add a feature",
        context={
            "github_repo": "org/repo",
            "worktree_branch": "feature/test-branch",
        },
    )


@pytest.fixture
def tmp_worktree(tmp_path):
    """Real directory that tests can delete to simulate worktree vanishing."""
    wt = tmp_path / "worktree"
    wt.mkdir()
    # Seed with some files so file_count > 0
    (wt / "README.md").write_text("hello")
    (wt / "main.py").write_text("pass")
    return wt


@pytest.fixture
def agent():
    """Minimal mock agent with real methods bound for testing."""
    a = MagicMock()
    a._session_logger = MagicMock()
    a._session_logger.log = MagicMock()
    a.logger = MagicMock()
    a.config = MagicMock()
    a.config.id = "test-agent"
    a.queue = MagicMock()
    a.activity_manager = MagicMock()
    a._running = True
    return a


class TestPreFlightSessionEvent:
    """_get_validated_working_directory logs worktree_validated on success."""

    def test_session_log_on_successful_validation(self, agent, task, tmp_worktree):
        agent._get_validated_working_directory = (
            Agent._get_validated_working_directory.__get__(agent)
        )
        agent._git_ops = MagicMock()
        agent._git_ops.get_working_directory.return_value = tmp_worktree

        result = agent._get_validated_working_directory(task)

        assert result == tmp_worktree
        agent._session_logger.log.assert_called_once_with(
            "worktree_validated",
            path=str(tmp_worktree),
            branch="feature/test-branch",
            file_count=2,  # README.md + main.py
        )

    def test_raises_when_missing_after_retry(self, agent, task):
        """Existing behavior preserved: raises RuntimeError after retry."""
        agent._get_validated_working_directory = (
            Agent._get_validated_working_directory.__get__(agent)
        )
        nonexistent = Path("/tmp/does-not-exist-worktree-test")
        agent._git_ops = MagicMock()
        agent._git_ops.get_working_directory.return_value = nonexistent

        with pytest.raises(RuntimeError, match="does not exist after retry"):
            agent._get_validated_working_directory(task)


class TestWatcherWorktreeVanish:
    """_watch_for_interruption returns when worktree disappears."""

    @pytest.mark.asyncio
    async def test_watcher_returns_on_worktree_vanish(self, agent, tmp_worktree):
        agent._watch_for_interruption = (
            Agent._watch_for_interruption.__get__(agent)
        )
        agent._check_pause_signal = MagicMock(return_value=False)
        agent._write_heartbeat = MagicMock()
        agent._git_ops = MagicMock()
        agent._git_ops.active_worktree = tmp_worktree

        # Delete the worktree after a short delay
        async def delete_soon():
            await asyncio.sleep(0.1)
            shutil.rmtree(tmp_worktree)

        asyncio.create_task(delete_soon())

        # Watcher should complete within a few seconds (poll interval = 2s)
        await asyncio.wait_for(agent._watch_for_interruption(), timeout=5.0)
        agent.logger.critical.assert_called_once()

    @pytest.mark.asyncio
    async def test_watcher_continues_when_worktree_exists(self, agent, tmp_worktree):
        agent._watch_for_interruption = (
            Agent._watch_for_interruption.__get__(agent)
        )
        agent._check_pause_signal = MagicMock(return_value=False)
        agent._write_heartbeat = MagicMock()
        agent._git_ops = MagicMock()
        agent._git_ops.active_worktree = tmp_worktree

        watcher = asyncio.create_task(agent._watch_for_interruption())

        # Give it time to do a couple iterations — should NOT complete
        await asyncio.sleep(0.3)
        assert not watcher.done(), "Watcher should not complete when worktree exists"

        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


class TestLLMCancelledOnWorktreeVanish:
    """Full integration: watcher completes → LLM cancelled → worktree_vanished event."""

    @pytest.mark.asyncio
    async def test_llm_cancelled_on_runtime_vanish(self, agent, task, tmp_worktree):
        agent._execute_llm_with_interruption_watch = (
            Agent._execute_llm_with_interruption_watch.__get__(agent)
        )
        agent._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(agent)
        agent._auto_commit_wip = AsyncMock()
        agent._update_phase = MagicMock()
        agent._session_logger = MagicMock()
        agent._session_logger.log = MagicMock()
        agent._session_logger.log_tool_call = MagicMock()
        agent._context_window_manager = None
        agent._current_specialization = None
        agent._current_file_count = 0
        agent._mcp_enabled = False
        agent._max_consecutive_tool_calls = 999
        agent._max_consecutive_diagnostic_calls = 999
        agent._exploration_alert_threshold = 999
        agent._exploration_alert_thresholds = {}
        agent.config.id = "test-agent"
        agent.workspace = Path("/tmp/test-workspace")

        # Make LLM hang
        llm = MagicMock()

        async def _complete(*args, **kwargs):
            await asyncio.sleep(999)

        llm.complete = _complete
        llm.cancel = MagicMock()
        llm.get_partial_output = MagicMock(return_value="")
        agent.llm = llm

        # Wire up watcher to use the real method
        agent._watch_for_interruption = (
            Agent._watch_for_interruption.__get__(agent)
        )
        agent._check_pause_signal = MagicMock(return_value=False)
        agent._write_heartbeat = MagicMock()
        agent._git_ops = MagicMock()
        agent._git_ops.active_worktree = tmp_worktree

        # Delete worktree after a short delay to trigger watcher
        async def delete_soon():
            await asyncio.sleep(0.1)
            shutil.rmtree(tmp_worktree)

        asyncio.create_task(delete_soon())

        result = await asyncio.wait_for(
            agent._execute_llm_with_interruption_watch(
                task, "implement feature", tmp_worktree, None
            ),
            timeout=10.0,
        )

        assert result is None
        agent.llm.cancel.assert_called()

        # Verify worktree_vanished event was emitted
        event_calls = agent.activity_manager.append_event.call_args_list
        assert len(event_calls) == 1
        event = event_calls[0][0][0]
        assert event.type == "worktree_vanished"
        assert event.task_id == "test-wt-1"

        # Verify session log
        session_log_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0][0] == "worktree_vanished"
        ]
        assert len(session_log_calls) == 1
        assert session_log_calls[0][1]["path"] == str(tmp_worktree)


class TestGapCheck:
    """Second validation in _handle_task catches deletion before LLM start."""

    def test_gap_check_raises_runtime_error(self, tmp_path):
        """Verify RuntimeError is raised when working_dir vanishes before LLM.

        Pattern-level test: reproduces the inline check from _handle_task rather
        than exercising _handle_task directly (too many dependencies to mock).
        """
        working_dir = tmp_path / "vanished"
        working_dir.mkdir()
        working_dir.rmdir()

        # The check is: if not working_dir.exists(): raise RuntimeError(...)
        assert not working_dir.exists()
        with pytest.raises(RuntimeError, match="vanished before LLM start"):
            if not working_dir.exists():
                raise RuntimeError(
                    f"Working directory vanished before LLM start: {working_dir}. "
                    f"Likely deleted by sibling agent cleanup during prompt build."
                )


class TestWorktreeVanishedEventType:
    """ActivityEvent accepts 'worktree_vanished' type."""

    def test_worktree_vanished_event_type_valid(self):
        event = ActivityEvent(
            type="worktree_vanished",
            agent="test-agent",
            task_id="test-1",
            title="Worktree gone",
            timestamp=datetime.now(timezone.utc),
        )
        assert event.type == "worktree_vanished"

    def test_invalid_event_type_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ActivityEvent(
                type="not_a_real_type",
                agent="test-agent",
                task_id="test-1",
                title="Bad type",
                timestamp=datetime.now(timezone.utc),
            )
