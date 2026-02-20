"""Tests for worktree reference counting (P0 worktree deletion fix).

Covers:
- WorktreeInfo.active_users field and backward compat
- acquire_worktree / release_worktree ref counting
- Stale PID cleanup via _prune_stale_users
- remove_worktree refuses when active_users non-empty
- Subtask intermediate detection preserves worktree
- Hot-reload version detection
"""

import json
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_framework.workspace.worktree_manager import (
    WorktreeManager,
    WorktreeConfig,
    WorktreeInfo,
)


class TestActiveUsersField:
    """Tests for the active_users field on WorktreeInfo."""

    def test_default_empty_list(self):
        info = WorktreeInfo(
            path="/wt", branch="b", agent_id="a", task_id="t",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/repo",
        )
        assert info.active_users == []

    def test_to_dict_includes_active_users(self):
        info = WorktreeInfo(
            path="/wt", branch="b", agent_id="a", task_id="t",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/repo",
            active_users=[{"agent_id": "eng", "pid": 1234}],
        )
        d = info.to_dict()
        assert d["active_users"] == [{"agent_id": "eng", "pid": 1234}]

    def test_from_dict_backward_compat_no_active_users(self):
        """Old registries without active_users still deserialize."""
        data = {
            "path": "/wt", "branch": "b", "agent_id": "a", "task_id": "t",
            "created_at": "2025-01-01T00:00:00", "last_accessed": "2025-01-01T00:00:00",
            "base_repo": "/repo", "active": True,
        }
        info = WorktreeInfo.from_dict(data)
        assert info.active_users == []
        assert info.active is True

    def test_from_dict_with_active_users(self):
        data = {
            "path": "/wt", "branch": "b", "agent_id": "a", "task_id": "t",
            "created_at": "2025-01-01T00:00:00", "last_accessed": "2025-01-01T00:00:00",
            "base_repo": "/repo", "active": True,
            "active_users": [{"agent_id": "eng", "pid": 42}],
        }
        info = WorktreeInfo.from_dict(data)
        assert len(info.active_users) == 1
        assert info.active_users[0]["agent_id"] == "eng"

    def test_from_dict_ignores_unknown_fields(self):
        """Extra fields in registry JSON are silently dropped."""
        data = {
            "path": "/wt", "branch": "b", "agent_id": "a", "task_id": "t",
            "created_at": "2025-01-01T00:00:00", "last_accessed": "2025-01-01T00:00:00",
            "base_repo": "/repo", "future_field": "ignored",
        }
        info = WorktreeInfo.from_dict(data)
        assert not hasattr(info, "future_field")


class TestAcquireRelease:
    """Tests for acquire_worktree / release_worktree ref counting."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config=config)
        # Seed a worktree entry
        wt_path = tmp_path / "worktrees" / "test-wt"
        wt_path.mkdir(parents=True)
        mgr._registry["test-key"] = WorktreeInfo(
            path=str(wt_path), branch="b", agent_id="chain", task_id="t",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            base_repo="/repo", active=True,
        )
        mgr._save_registry()
        return mgr, wt_path

    def test_acquire_adds_user(self, manager):
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "architect")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 1
        assert entry.active_users[0]["agent_id"] == "architect"
        assert entry.active is True

    def test_acquire_deduplicates_same_user(self, manager):
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "architect")
        mgr.acquire_worktree(wt_path, "architect")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 1

    def test_acquire_multiple_users(self, manager):
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "architect")
        mgr.acquire_worktree(wt_path, "engineer")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 2

    def test_release_removes_user(self, manager):
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "architect")
        mgr.acquire_worktree(wt_path, "engineer")

        mgr.release_worktree(wt_path, "architect")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 1
        assert entry.active_users[0]["agent_id"] == "engineer"
        assert entry.active is True

    def test_release_last_user_sets_inactive(self, manager):
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "engineer")

        mgr.release_worktree(wt_path, "engineer")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 0
        assert entry.active is False

    def test_release_nonexistent_user_is_noop(self, manager):
        """Releasing a user that never acquired doesn't crash."""
        mgr, wt_path = manager
        mgr.release_worktree(wt_path, "nonexistent")
        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 0

    def test_acquire_persists_to_disk(self, manager, tmp_path):
        """Acquire writes to disk so other processes see it."""
        mgr, wt_path = manager
        mgr.acquire_worktree(wt_path, "engineer", pid=12345)

        # Load fresh from disk
        registry_path = tmp_path / "worktrees" / ".worktree-registry.json"
        data = json.loads(registry_path.read_text())
        assert data["test-key"]["active_users"] == [{"agent_id": "engineer", "pid": 12345}]


class TestPruneStaleUsers:
    """Tests for _prune_stale_users - dead PID cleanup."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config=config)
        wt_path = tmp_path / "worktrees" / "test-wt"
        wt_path.mkdir(parents=True)
        mgr._registry["test-key"] = WorktreeInfo(
            path=str(wt_path), branch="b", agent_id="chain", task_id="t",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            base_repo="/repo", active=True,
            active_users=[
                {"agent_id": "alive", "pid": os.getpid()},
                {"agent_id": "dead", "pid": 99999999},
            ],
        )
        return mgr

    def test_prune_removes_dead_pid(self, manager):
        with patch.object(WorktreeManager, '_is_pid_alive', side_effect=lambda pid: pid == os.getpid()):
            pruned = manager._prune_stale_users()
        assert pruned == 1
        entry = manager._registry["test-key"]
        assert len(entry.active_users) == 1
        assert entry.active_users[0]["agent_id"] == "alive"

    def test_prune_all_dead_clears_users(self, manager):
        manager._registry["test-key"].active_users = [
            {"agent_id": "dead1", "pid": 99999998},
            {"agent_id": "dead2", "pid": 99999999},
        ]
        with patch.object(WorktreeManager, '_is_pid_alive', return_value=False):
            pruned = manager._prune_stale_users()
        assert pruned == 2
        entry = manager._registry["test-key"]
        assert len(entry.active_users) == 0

    def test_acquire_prunes_before_adding(self, tmp_path):
        """acquire_worktree prunes stale users before adding the new one."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config=config)
        wt_path = tmp_path / "worktrees" / "test-wt"
        wt_path.mkdir(parents=True)
        mgr._registry["test-key"] = WorktreeInfo(
            path=str(wt_path), branch="b", agent_id="chain", task_id="t",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            base_repo="/repo", active=True,
            active_users=[{"agent_id": "dead", "pid": 99999999}],
        )
        mgr._save_registry()

        with patch.object(WorktreeManager, '_is_pid_alive', side_effect=lambda pid: pid == os.getpid()):
            mgr.acquire_worktree(wt_path, "new_agent")

        entry = mgr._registry["test-key"]
        assert len(entry.active_users) == 1
        assert entry.active_users[0]["agent_id"] == "new_agent"


class TestRemoveWorktreeRefCount:
    """Tests that remove_worktree respects active_users."""

    @pytest.fixture
    def setup(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config=config)
        wt_path = tmp_path / "worktrees" / "test-wt"
        wt_path.mkdir(parents=True)
        (wt_path / ".git").write_text("gitdir: /fake/.git/worktrees/test")
        mgr._registry["test-key"] = WorktreeInfo(
            path=str(wt_path), branch="b", agent_id="chain", task_id="t",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            base_repo=str(tmp_path / "base"),
            active=True,
            active_users=[{"agent_id": "engineer", "pid": os.getpid()}],
        )
        mgr._save_registry()
        return mgr, wt_path

    def test_refuses_when_active_users_present(self, setup):
        mgr, wt_path = setup
        result = mgr.remove_worktree(wt_path, force=False)
        assert result is False
        assert wt_path.exists()

    def test_force_overrides_active_users(self, setup):
        mgr, wt_path = setup
        result = mgr.remove_worktree(wt_path, force=True)
        assert result is True

    def test_allows_removal_when_no_active_users(self, setup):
        mgr, wt_path = setup
        mgr._registry["test-key"].active_users = []
        mgr._registry["test-key"].active = False
        mgr._save_registry()
        result = mgr.remove_worktree(wt_path, force=False)
        assert result is True


class TestMarkWorktreeInactiveRefCounted:
    """Tests for mark_worktree_inactive with user_id parameter."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config=config)
        wt_path = tmp_path / "worktrees" / "test-wt"
        wt_path.mkdir(parents=True)
        mgr._registry["test-key"] = WorktreeInfo(
            path=str(wt_path), branch="b", agent_id="chain", task_id="t",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_accessed=datetime.now(timezone.utc).isoformat(),
            base_repo="/repo", active=True,
            active_users=[
                {"agent_id": "architect", "pid": os.getpid()},
                {"agent_id": "engineer", "pid": os.getpid()},
            ],
        )
        mgr._save_registry()
        return mgr, wt_path

    def test_release_one_keeps_active(self, manager):
        mgr, wt_path = manager
        mgr.mark_worktree_inactive(wt_path, user_id="architect")
        entry = mgr._registry["test-key"]
        assert entry.active is True
        assert len(entry.active_users) == 1

    def test_release_all_marks_inactive(self, manager):
        mgr, wt_path = manager
        mgr.mark_worktree_inactive(wt_path, user_id="architect")
        mgr.mark_worktree_inactive(wt_path, user_id="engineer")
        entry = mgr._registry["test-key"]
        assert entry.active is False
        assert len(entry.active_users) == 0

    def test_legacy_no_user_id_clears_all(self, manager):
        """Without user_id, legacy behavior clears everything."""
        mgr, wt_path = manager
        mgr.mark_worktree_inactive(wt_path)
        entry = mgr._registry["test-key"]
        assert entry.active is False
        assert len(entry.active_users) == 0


class TestSubtaskIntermediateDetection:
    """Tests that subtasks with workflow keep worktree active at non-terminal steps."""

    def _make_git_ops(self, mock_config, mock_logger, mock_queue, mock_wt_mgr, tmp_path):
        from agent_framework.core.git_operations import GitOperationsManager
        return GitOperationsManager(
            config=mock_config,
            workspace=tmp_path,
            queue=mock_queue,
            logger=mock_logger,
            worktree_manager=mock_wt_mgr,
        )

    def test_subtask_with_chain_step_keeps_worktree_active(self, tmp_path):
        """Subtask at a non-terminal chain step should touch worktree, not mark inactive."""
        from agent_framework.core.task import Task

        mock_config = MagicMock()
        mock_config.id = "engineer"
        mock_config.base_id = "engineer"
        mock_logger = MagicMock()
        mock_queue = MagicMock()
        mock_wt_mgr = MagicMock()
        mock_wt_mgr.has_unpushed_commits.return_value = False

        git_ops = self._make_git_ops(
            mock_config, mock_logger, mock_queue, mock_wt_mgr, tmp_path,
        )
        # Mock the terminal step check - engineer at implement is non-terminal
        git_ops._is_at_terminal_workflow_step = MagicMock(return_value=False)
        wt_path = tmp_path / "wt"
        wt_path.mkdir()
        git_ops._active_worktree = wt_path

        task = Task(
            id="subtask-1",
            title="Implement feature",
            description="...",
            type="implementation",
            status="pending",
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            parent_task_id="parent-1",
            context={
                "chain_step": True,
                "workflow": "default",
                "workflow_step": "implement",
            },
        )

        git_ops.cleanup_worktree(task, success=True)

        mock_wt_mgr.touch_worktree.assert_called_once_with(wt_path)
        mock_wt_mgr.mark_worktree_inactive.assert_not_called()


class TestHotReloadVersionDetection:
    """Tests for _get_source_code_version and _should_hot_restart."""

    def test_get_source_code_version_returns_string(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._get_source_code_version = Agent._get_source_code_version.__get__(agent)
        version = agent._get_source_code_version()
        assert version is not None
        assert len(version) > 0

    def test_should_hot_restart_false_when_same_version(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._get_source_code_version = Agent._get_source_code_version.__get__(agent)
        agent._should_hot_restart = Agent._should_hot_restart.__get__(agent)
        agent._startup_code_version = agent._get_source_code_version()
        agent._last_version_check = 0.0  # bypass rate limiter
        assert agent._should_hot_restart() is False

    def test_should_hot_restart_true_when_version_differs(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._should_hot_restart = Agent._should_hot_restart.__get__(agent)
        agent._get_source_code_version = MagicMock(return_value="new_hash")
        agent._startup_code_version = "old_hash"
        agent._last_version_check = 0.0
        assert agent._should_hot_restart() is True

    def test_should_hot_restart_false_when_no_startup_version(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._should_hot_restart = Agent._should_hot_restart.__get__(agent)
        agent._get_source_code_version = MagicMock(return_value="some_hash")
        del agent._startup_code_version
        assert agent._should_hot_restart() is False

    def test_should_hot_restart_false_when_version_returns_none(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._should_hot_restart = Agent._should_hot_restart.__get__(agent)
        agent._get_source_code_version = MagicMock(return_value=None)
        agent._startup_code_version = "old_hash"
        agent._last_version_check = 0.0
        assert agent._should_hot_restart() is False

    def test_rate_limits_version_check(self):
        """Calls within 60s window return False without checking git."""
        import time
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._should_hot_restart = Agent._should_hot_restart.__get__(agent)
        agent._startup_code_version = "old_hash"
        agent._last_version_check = time.time()  # just checked
        agent._get_source_code_version = MagicMock(return_value="new_hash")
        assert agent._should_hot_restart() is False
        # git was never called because rate limiter short-circuited
        agent._get_source_code_version.assert_not_called()


class TestCheckpointIntervalConditional:
    """Test that checkpoint interval is 15 for implementation steps, 25 otherwise."""

    def test_implementation_step_uses_15(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._is_implementation_step = Agent._is_implementation_step.__get__(agent)
        agent.config = MagicMock()
        agent.config.base_id = "engineer"

        task = MagicMock()
        task.context = {"workflow_step": "implement"}
        assert agent._is_implementation_step(task) is True

    def test_plan_step_is_not_implementation(self):
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent._is_implementation_step = Agent._is_implementation_step.__get__(agent)
        agent.config = MagicMock()
        agent.config.base_id = "architect"

        task = MagicMock()
        task.context = {"workflow_step": "plan"}
        assert agent._is_implementation_step(task) is False
