"""Tests for Agent._maybe_run_periodic_worktree_cleanup()."""

import time

import pytest
from unittest.mock import MagicMock

from agent_framework.core.agent import Agent, _WORKTREE_CLEANUP_INTERVAL_SECONDS


@pytest.fixture
def agent():
    """Minimal Agent mock with the real periodic cleanup method bound."""
    mock = MagicMock()
    mock._last_worktree_cleanup = time.time()
    mock._maybe_run_periodic_worktree_cleanup = (
        Agent._maybe_run_periodic_worktree_cleanup.__get__(mock)
    )
    return mock


class TestPeriodicWorktreeCleanup:

    def test_skips_when_no_worktree_manager(self, agent):
        agent.worktree_manager = None
        agent._maybe_run_periodic_worktree_cleanup()
        # Nothing to assert — just verifying it doesn't crash

    def test_skips_when_timer_not_elapsed(self, agent):
        agent.worktree_manager = MagicMock()
        agent._last_worktree_cleanup = time.time()

        agent._maybe_run_periodic_worktree_cleanup()

        agent.worktree_manager.cleanup_orphaned_worktrees.assert_not_called()

    def test_runs_cleanup_when_timer_elapsed(self, agent):
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.cleanup_orphaned_worktrees.return_value = {"total": 2}
        agent._last_worktree_cleanup = time.time() - _WORKTREE_CLEANUP_INTERVAL_SECONDS - 1

        agent._maybe_run_periodic_worktree_cleanup()

        agent.worktree_manager.cleanup_orphaned_worktrees.assert_called_once()
        agent.logger.info.assert_called_once()

    def test_updates_timer_after_cleanup(self, agent):
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.cleanup_orphaned_worktrees.return_value = {"total": 0}
        agent._last_worktree_cleanup = 0.0

        before = time.time()
        agent._maybe_run_periodic_worktree_cleanup()

        assert agent._last_worktree_cleanup >= before

    def test_exception_does_not_propagate(self, agent):
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.cleanup_orphaned_worktrees.side_effect = RuntimeError("disk full")
        agent._last_worktree_cleanup = 0.0

        # Should not raise
        agent._maybe_run_periodic_worktree_cleanup()
        agent.logger.debug.assert_called_once()

    def test_silent_when_nothing_removed(self, agent):
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.cleanup_orphaned_worktrees.return_value = {"total": 0}
        agent._last_worktree_cleanup = 0.0

        agent._maybe_run_periodic_worktree_cleanup()

        agent.logger.info.assert_not_called()
