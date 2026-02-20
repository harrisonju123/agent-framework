"""Tests for Agent._maybe_run_periodic_worktree_cleanup() — now a no-op."""

import pytest
from unittest.mock import MagicMock

from agent_framework.core.agent import Agent


@pytest.fixture
def agent():
    """Minimal Agent mock with the real periodic cleanup method bound."""
    mock = MagicMock()
    mock._maybe_run_periodic_worktree_cleanup = (
        Agent._maybe_run_periodic_worktree_cleanup.__get__(mock)
    )
    return mock


class TestPeriodicWorktreeCleanup:

    def test_is_noop(self, agent):
        """Periodic cleanup is disabled — never touches worktree_manager."""
        agent.worktree_manager = MagicMock()
        agent._maybe_run_periodic_worktree_cleanup()
        agent.worktree_manager.cleanup_orphaned_worktrees.assert_not_called()

    def test_noop_without_worktree_manager(self, agent):
        """No crash when worktree_manager is None."""
        agent.worktree_manager = None
        agent._maybe_run_periodic_worktree_cleanup()
