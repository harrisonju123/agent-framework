"""Unit tests for GitOperationsManager."""

import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

from agent_framework.core.git_operations import GitOperationsManager
from agent_framework.core.task import Task, TaskStatus


@pytest.fixture
def mock_config():
    """Mock agent configuration."""
    config = MagicMock()
    config.id = "engineer"
    config.base_id = "engineer"
    return config


@pytest.fixture
def mock_logger():
    """Mock logger."""
    return MagicMock()


@pytest.fixture
def mock_queue():
    """Mock file queue."""
    queue = MagicMock()
    queue.queue_dir = Path("/mock/queue")
    # completed_dir must be a real Path so .exists() returns False rather than a truthy MagicMock
    queue.completed_dir = Path("/mock/queue/completed")
    queue.get_completed.return_value = None
    return queue


@pytest.fixture
def mock_worktree_manager():
    """Mock worktree manager."""
    manager = MagicMock()
    manager.config = MagicMock()
    manager.config.cleanup_on_complete = True
    manager.config.cleanup_on_failure = False
    return manager


@pytest.fixture
def mock_multi_repo_manager():
    """Mock multi-repo manager."""
    return MagicMock()


@pytest.fixture
def git_ops(mock_config, mock_logger, mock_queue, tmp_path):
    """Create GitOperationsManager instance."""
    return GitOperationsManager(
        config=mock_config,
        workspace=tmp_path,
        queue=mock_queue,
        logger=mock_logger,
    )


@pytest.fixture
def git_ops_with_worktree(mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path):
    """Create GitOperationsManager with worktree manager."""
    return GitOperationsManager(
        config=mock_config,
        workspace=tmp_path,
        queue=mock_queue,
        logger=mock_logger,
        worktree_manager=mock_worktree_manager,
    )


@pytest.fixture
def sample_task():
    """Create a sample task."""
    from datetime import datetime, timezone
    return Task(
        id="task-123",
        title="Test Task",
        description="Test description",
        type="implementation",
        status="pending",
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        context={"github_repo": "owner/repo"},
    )


class TestGetWorkingDirectory:
    """Tests for get_working_directory method."""

    def test_uses_workspace_when_no_github_repo(self, git_ops, sample_task):
        """Test that workspace is used when no github_repo in context."""
        sample_task.context = {}
        result = git_ops.get_working_directory(sample_task)
        assert result == git_ops.workspace

    def test_uses_shared_clone_for_pr_creation_task(self, git_ops, sample_task):
        """Test that PR creation tasks use shared clone without worktree."""
        git_ops.multi_repo_manager = MagicMock()
        git_ops.multi_repo_manager.ensure_repo.return_value = Path("/shared/clone")

        sample_task.context["pr_creation_step"] = True
        sample_task.context["implementation_branch"] = "feature/test"

        result = git_ops.get_working_directory(sample_task)

        assert result == Path("/shared/clone")
        git_ops.multi_repo_manager.ensure_repo.assert_called_once_with("owner/repo")

    def test_creates_worktree_when_enabled(self, git_ops_with_worktree, sample_task):
        """Test worktree creation when worktree mode is enabled."""
        git_ops_with_worktree.multi_repo_manager = MagicMock()
        git_ops_with_worktree.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")
        git_ops_with_worktree.worktree_manager.find_worktree_by_branch.return_value = None
        git_ops_with_worktree.worktree_manager.create_worktree.return_value = Path("/worktree")

        result = git_ops_with_worktree.get_working_directory(sample_task)

        assert result == Path("/worktree")
        assert git_ops_with_worktree._active_worktree == Path("/worktree")
        git_ops_with_worktree.worktree_manager.create_worktree.assert_called_once()

    def test_reuses_existing_worktree(self, git_ops_with_worktree, sample_task, tmp_path):
        """Test that existing worktree is reused."""
        existing_wt = tmp_path / "existing-worktree"
        existing_wt.mkdir()

        git_ops_with_worktree.multi_repo_manager = MagicMock()
        git_ops_with_worktree.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")
        git_ops_with_worktree.worktree_manager.find_worktree_by_branch.return_value = existing_wt

        sample_task.context["worktree_branch"] = "agent/engineer/task-abc123"

        result = git_ops_with_worktree.get_working_directory(sample_task)

        assert result == existing_wt
        assert git_ops_with_worktree._active_worktree == existing_wt
        git_ops_with_worktree.worktree_manager.create_worktree.assert_not_called()

    def test_falls_back_to_shared_clone_on_worktree_failure(self, git_ops_with_worktree, sample_task):
        """Test fallback to shared clone when worktree creation fails."""
        git_ops_with_worktree.multi_repo_manager = MagicMock()
        git_ops_with_worktree.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")
        git_ops_with_worktree.worktree_manager.find_worktree_by_branch.return_value = None
        git_ops_with_worktree.worktree_manager.create_worktree.side_effect = Exception("Failed")

        result = git_ops_with_worktree.get_working_directory(sample_task)

        assert result == Path("/base/repo")
        assert git_ops_with_worktree._active_worktree is None


class TestShouldUseWorktree:
    """Tests for _should_use_worktree method."""

    def test_respects_task_override_true(self, git_ops_with_worktree, sample_task):
        """Test that task override for worktree=True is respected."""
        sample_task.context["use_worktree"] = True
        assert git_ops_with_worktree._should_use_worktree(sample_task) is True

    def test_respects_task_override_false(self, git_ops_with_worktree, sample_task):
        """Test that task override for worktree=False is respected."""
        sample_task.context["use_worktree"] = False
        assert git_ops_with_worktree._should_use_worktree(sample_task) is False

    def test_returns_false_when_no_worktree_manager(self, git_ops, sample_task):
        """Test that False is returned when no worktree manager is available."""
        assert git_ops._should_use_worktree(sample_task) is False

    def test_returns_true_when_worktree_manager_available(self, git_ops_with_worktree, sample_task):
        """Test that True is returned when worktree manager is available."""
        assert git_ops_with_worktree._should_use_worktree(sample_task) is True


class TestSyncWorktreeQueuedTasks:
    """Tests for sync_worktree_queued_tasks method."""

    def test_does_nothing_when_no_active_worktree(self, git_ops):
        """Test that method does nothing when no active worktree."""
        git_ops.sync_worktree_queued_tasks()
        git_ops.queue.push.assert_not_called()

    def test_syncs_task_files_from_worktree(self, git_ops, tmp_path):
        """Test that task files are synced from worktree to main queue."""
        from datetime import datetime, timezone

        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path

        # Create worktree queue structure
        worktree_queue = worktree_path / ".agent-communication" / "queues" / "qa"
        worktree_queue.mkdir(parents=True)

        # Create a task file
        task_data = {
            "id": "task-456",
            "title": "QA Task",
            "description": "Test",
            "type": "qa_verification",
            "status": "pending",
            "priority": 1,
            "created_by": "test",
            "assigned_to": "qa",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": {"chain_step": True},
        }
        task_file = worktree_queue / "task-456.json"
        task_file.write_text(json.dumps(task_data))

        git_ops.sync_worktree_queued_tasks()

        assert git_ops.queue.push.called
        assert not task_file.exists()  # File should be deleted after sync

    def test_skips_completed_tasks(self, git_ops, tmp_path):
        """Completed tasks in worktree queue are skipped but cleaned up."""
        from datetime import datetime, timezone

        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path

        worktree_queue = worktree_path / ".agent-communication" / "queues" / "engineer"
        worktree_queue.mkdir(parents=True)

        task_data = {
            "id": "task-already-done",
            "title": "Completed Task",
            "description": "Test",
            "type": "implementation",
            "status": "pending",
            "priority": 1,
            "created_by": "test",
            "assigned_to": "engineer",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": {"chain_step": True},
        }
        task_file = worktree_queue / "task-already-done.json"
        task_file.write_text(json.dumps(task_data))

        # Simulate task already completed by placing its file in the completed_dir
        completed_dir = tmp_path / "completed"
        completed_dir.mkdir()
        (completed_dir / "task-already-done.json").write_text(json.dumps(task_data))
        git_ops.queue.completed_dir = completed_dir

        git_ops.sync_worktree_queued_tasks()

        git_ops.queue.push.assert_not_called()
        assert not task_file.exists()  # Stale file still cleaned up


class TestCleanupWorktree:
    """Tests for cleanup_worktree method."""

    def test_does_nothing_when_no_active_worktree(self, git_ops, sample_task):
        """Test that cleanup does nothing when no active worktree."""
        git_ops.cleanup_worktree(sample_task, success=True)
        # No assertions needed - just checking it doesn't crash

    def test_cleans_up_on_success_when_configured(self, git_ops_with_worktree, sample_task, tmp_path):
        """Test that worktree is cleaned up on success when configured."""
        worktree_path = tmp_path / "worktree"
        git_ops_with_worktree._active_worktree = worktree_path
        git_ops_with_worktree.worktree_manager.has_unpushed_commits.return_value = False
        git_ops_with_worktree.worktree_manager.has_uncommitted_changes.return_value = False

        git_ops_with_worktree.cleanup_worktree(sample_task, success=True)

        git_ops_with_worktree.worktree_manager.remove_worktree.assert_called_once()
        assert git_ops_with_worktree._active_worktree is None

    def test_skips_cleanup_when_unpushed_commits(self, git_ops_with_worktree, sample_task, tmp_path):
        """Test that cleanup is skipped when there are unpushed commits."""
        worktree_path = tmp_path / "worktree"
        git_ops_with_worktree._active_worktree = worktree_path
        git_ops_with_worktree.worktree_manager.has_unpushed_commits.return_value = True

        git_ops_with_worktree.cleanup_worktree(sample_task, success=True)

        git_ops_with_worktree.worktree_manager.remove_worktree.assert_not_called()
        assert git_ops_with_worktree._active_worktree is None  # Still cleared

    def test_skips_cleanup_when_uncommitted_changes(self, git_ops_with_worktree, sample_task, tmp_path):
        """Test that cleanup is skipped when there are uncommitted changes."""
        worktree_path = tmp_path / "worktree"
        git_ops_with_worktree._active_worktree = worktree_path
        git_ops_with_worktree.worktree_manager.has_unpushed_commits.return_value = False
        git_ops_with_worktree.worktree_manager.has_uncommitted_changes.return_value = True

        git_ops_with_worktree.cleanup_worktree(sample_task, success=True)

        git_ops_with_worktree.worktree_manager.remove_worktree.assert_not_called()
        assert git_ops_with_worktree._active_worktree is None


class TestGetChangedFiles:
    """Tests for get_changed_files method."""

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_changed_files(self, mock_run_git, git_ops):
        """Test that changed files are returned from git diff."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file1.py\nfile2.py\nfile3.py"
        mock_run_git.return_value = mock_result

        files = git_ops.get_changed_files()

        assert files == ["file1.py", "file2.py", "file3.py"]

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_empty_list_on_error(self, mock_run_git, git_ops):
        """Test that empty list is returned on git command error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: not a git repository"
        mock_run_git.return_value = mock_result

        files = git_ops.get_changed_files()

        assert files == []


class TestPushAndCreatePRIfNeeded:
    """Tests for push_and_create_pr_if_needed method."""

    def test_skips_when_pr_already_exists(self, git_ops, sample_task):
        """Test that PR creation is skipped when PR already exists."""
        sample_task.context["pr_url"] = "https://github.com/owner/repo/pull/123"

        git_ops.push_and_create_pr_if_needed(sample_task)

        git_ops.logger.debug.assert_called()

    def test_skips_when_no_active_worktree(self, git_ops, sample_task):
        """Test that PR creation is skipped when no active worktree."""
        git_ops.push_and_create_pr_if_needed(sample_task)

        git_ops.logger.debug.assert_called()


class TestManagePRLifecycle:
    """Tests for manage_pr_lifecycle method."""

    def test_does_nothing_when_no_manager(self, git_ops, sample_task):
        """Test that method does nothing when no PR lifecycle manager."""
        git_ops.manage_pr_lifecycle(sample_task)
        # No assertions needed - just checking it doesn't crash

    def test_manages_pr_when_manager_available(self, git_ops, sample_task):
        """Test that PR is managed when manager is available."""
        mock_manager = MagicMock()
        mock_manager.should_manage.return_value = True
        mock_manager.manage.return_value = False
        git_ops._pr_lifecycle_manager = mock_manager

        sample_task.context["pr_url"] = "https://github.com/owner/repo/pull/123"

        git_ops.manage_pr_lifecycle(sample_task)

        mock_manager.should_manage.assert_called_once_with(sample_task)
        mock_manager.manage.assert_called_once()


class TestActiveWorktreeProperty:
    """Tests for active_worktree property."""

    def test_get_active_worktree(self, git_ops, tmp_path):
        """Test getting active worktree."""
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path

        assert git_ops.active_worktree == worktree_path

    def test_set_active_worktree(self, git_ops, tmp_path):
        """Test setting active worktree."""
        worktree_path = tmp_path / "worktree"
        git_ops.active_worktree = worktree_path

        assert git_ops._active_worktree == worktree_path


class TestCleanupWorktreeInactiveMarking:
    """Tests that cleanup_worktree marks worktrees inactive."""

    def test_marks_worktree_inactive_on_cleanup(self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task):
        """mark_worktree_inactive is called during normal cleanup."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = False
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.mark_worktree_inactive.assert_called_once_with(worktree_path)

    def test_marks_worktree_inactive_even_when_cleanup_skipped(self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task):
        """mark_worktree_inactive is called even when unpushed commits block removal."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = True

        git_ops.cleanup_worktree(sample_task, success=True)

        # Worktree not removed, but still marked inactive
        mock_worktree_manager.remove_worktree.assert_not_called()
        mock_worktree_manager.mark_worktree_inactive.assert_called_once_with(worktree_path)


class TestCleanupWorktreeChainStepProtection:
    """Tests that intermediate chain steps keep worktrees protected from eviction."""

    def _make_git_ops(self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, workflows_config=None):
        return GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
            workflows_config=workflows_config,
        )

    def test_intermediate_chain_step_skips_inactive_marking(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Intermediate chain step: touch_worktree called, mark_worktree_inactive NOT called."""
        git_ops = self._make_git_ops(mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path)
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        git_ops._is_at_terminal_workflow_step = MagicMock(return_value=False)

        sample_task.context["chain_step"] = True

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.mark_worktree_inactive.assert_not_called()
        mock_worktree_manager.touch_worktree.assert_called_once_with(worktree_path)
        assert git_ops._active_worktree is None

    def test_terminal_chain_step_marks_inactive(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Terminal chain step: existing behavior preserved (mark_worktree_inactive called)."""
        git_ops = self._make_git_ops(mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path)
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = False
        mock_worktree_manager.has_uncommitted_changes.return_value = False
        git_ops._is_at_terminal_workflow_step = MagicMock(return_value=True)

        sample_task.context["chain_step"] = True

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.mark_worktree_inactive.assert_called_once_with(worktree_path)
        mock_worktree_manager.touch_worktree.assert_not_called()

    def test_non_chain_task_marks_inactive(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Standalone task (no chain_step): existing behavior preserved."""
        git_ops = self._make_git_ops(mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path)
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = False
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.mark_worktree_inactive.assert_called_once_with(worktree_path)
        mock_worktree_manager.touch_worktree.assert_not_called()

    def test_intermediate_chain_step_skips_removal(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Even with cleanup_on_complete=True, intermediate step doesn't remove worktree."""
        mock_worktree_manager.config.cleanup_on_complete = True
        git_ops = self._make_git_ops(mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path)
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        git_ops._is_at_terminal_workflow_step = MagicMock(return_value=False)

        sample_task.context["chain_step"] = True

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.remove_worktree.assert_not_called()
        mock_worktree_manager.mark_worktree_inactive.assert_not_called()


class TestCleanupWorktreePushBeforeRemoval:
    """Tests for push-then-cleanup behavior in cleanup_worktree."""

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_pushes_unpushed_commits_then_removes(
        self, mock_run_git, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """When push succeeds, worktree is removed instead of left behind."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = True
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        # Simulate successful push: rev-parse returns branch, push succeeds
        mock_rev_parse = MagicMock(returncode=0, stdout="agent/engineer/PROJ-123\n")
        mock_push = MagicMock(returncode=0)
        mock_run_git.side_effect = [mock_rev_parse, mock_push]

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.remove_worktree.assert_called_once()
        mock_logger.info.assert_any_call("Pushed unpushed commits during cleanup")

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_falls_through_when_push_fails(
        self, mock_run_git, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """When push fails, falls back to existing skip-with-warning behavior."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = True
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        # Push fails
        mock_rev_parse = MagicMock(returncode=0, stdout="agent/engineer/PROJ-123\n")
        mock_push = MagicMock(returncode=1, stderr="permission denied")
        mock_run_git.side_effect = [mock_rev_parse, mock_push]

        git_ops.cleanup_worktree(sample_task, success=True)

        mock_worktree_manager.remove_worktree.assert_not_called()

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_skips_push_for_main_branch(
        self, mock_run_git, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Worktrees on main/master are not pushed — they shouldn't have unpushed commits."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = True
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        mock_rev_parse = MagicMock(returncode=0, stdout="main\n")
        mock_run_git.side_effect = [mock_rev_parse]

        git_ops.cleanup_worktree(sample_task, success=True)

        # Push not attempted, worktree not removed
        mock_worktree_manager.remove_worktree.assert_not_called()

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_push_exception_handled_gracefully(
        self, mock_run_git, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path, sample_task
    ):
        """Exception during push attempt doesn't break cleanup flow."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        worktree_path = tmp_path / "worktree"
        git_ops._active_worktree = worktree_path
        mock_worktree_manager.has_unpushed_commits.return_value = True
        mock_worktree_manager.has_uncommitted_changes.return_value = False

        mock_run_git.side_effect = Exception("network timeout")

        git_ops.cleanup_worktree(sample_task, success=True)

        # Should not crash, worktree not removed
        mock_worktree_manager.remove_worktree.assert_not_called()
        assert git_ops._active_worktree is None


class TestDetectImplementationBranch:
    """Tests for detect_implementation_branch method."""

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_sets_branch_from_worktree(self, mock_run_git, git_ops, sample_task, tmp_path):
        """Feature branch is stored in task context."""
        git_ops._active_worktree = tmp_path / "worktree"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "agent/engineer/PROJ-123-abc\n"
        mock_run_git.return_value = mock_result

        git_ops.detect_implementation_branch(sample_task)

        assert sample_task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc"

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_skips_main_branch(self, mock_run_git, git_ops, sample_task, tmp_path):
        """main/master branches are not stored — they're not feature branches."""
        git_ops._active_worktree = tmp_path / "worktree"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "main\n"
        mock_run_git.return_value = mock_result

        git_ops.detect_implementation_branch(sample_task)

        assert "implementation_branch" not in sample_task.context

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_overwrites_stale_branch(self, mock_run_git, git_ops, sample_task, tmp_path):
        """Stale architect branch is replaced by the engineer's actual branch."""
        git_ops._active_worktree = tmp_path / "worktree"
        sample_task.context["implementation_branch"] = "agent/architect/task-old"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "agent/engineer/task-new\n"
        mock_run_git.return_value = mock_result

        git_ops.detect_implementation_branch(sample_task)

        assert sample_task.context["implementation_branch"] == "agent/engineer/task-new"
        git_ops.logger.info.assert_called_once_with(
            "Updated implementation branch: agent/architect/task-old → agent/engineer/task-new"
        )

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_preserves_branch_when_worktree_matches(self, mock_run_git, git_ops, sample_task, tmp_path):
        """No unnecessary churn when worktree HEAD matches what's already stored."""
        git_ops._active_worktree = tmp_path / "worktree"
        sample_task.context["implementation_branch"] = "agent/engineer/task-abc"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "agent/engineer/task-abc\n"
        mock_run_git.return_value = mock_result

        git_ops.detect_implementation_branch(sample_task)

        assert sample_task.context["implementation_branch"] == "agent/engineer/task-abc"

    def test_skips_when_no_worktree(self, git_ops, sample_task):
        """No active worktree → no-op."""
        git_ops.detect_implementation_branch(sample_task)

        assert "implementation_branch" not in sample_task.context

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_handles_git_failure(self, mock_run_git, git_ops, sample_task, tmp_path):
        """Git command failure is handled gracefully."""
        git_ops._active_worktree = tmp_path / "worktree"
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_run_git.return_value = mock_result

        git_ops.detect_implementation_branch(sample_task)

        assert "implementation_branch" not in sample_task.context


class TestRegistryReloadInGetWorkingDirectory:
    """Tests that get_working_directory reloads the registry before branch lookup."""

    def test_reloads_registry_before_branch_lookup(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path
    ):
        """reload_registry() is called before find_worktree_by_branch()."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        git_ops.multi_repo_manager = MagicMock()
        git_ops.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")
        mock_worktree_manager.find_worktree_by_branch.return_value = Path("/existing/wt")

        from datetime import datetime, timezone
        task = Task(
            id="task-reload", title="Test", description="Test", type="implementation",
            status="pending", priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            context={"github_repo": "owner/repo", "implementation_branch": "agent/engineer/ME-1"},
        )

        git_ops.get_working_directory(task)

        # Verify ordering: reload_registry called before find_worktree_by_branch
        calls = mock_worktree_manager.method_calls
        reload_idx = next(i for i, c in enumerate(calls) if c[0] == "reload_registry")
        find_idx = next(i for i, c in enumerate(calls) if c[0] == "find_worktree_by_branch")
        assert reload_idx < find_idx

    def test_architect_creates_own_worktree_from_engineer_branch(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path
    ):
        """Architect gets its own worktree based on the engineer's branch, not the engineer's worktree."""
        # Simulate architect agent (different base_id than the engineer branch)
        mock_config.id = "architect"
        mock_config.base_id = "architect"
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        git_ops.multi_repo_manager = MagicMock()
        git_ops.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")

        new_worktree = Path("/worktrees/owner/repo/architect-task-chain")
        mock_worktree_manager.find_worktree_by_branch.return_value = None
        mock_worktree_manager.create_worktree.return_value = new_worktree

        from datetime import datetime, timezone
        task = Task(
            id="task-chain", title="Code Review", description="Review", type="review",
            status="pending", priority=1, created_by="test", assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            context={
                "github_repo": "owner/repo",
                "implementation_branch": "agent/engineer/ME-1-abc12345",
            },
        )

        result = git_ops.get_working_directory(task)

        assert result == new_worktree
        # Architect created its own branch, not reusing engineer's
        create_call = mock_worktree_manager.create_worktree.call_args
        assert create_call.kwargs.get("start_point") == "agent/engineer/ME-1-abc12345"
        # Branch name should be the architect's own
        assert create_call.kwargs["branch_name"].startswith("agent/architect/")

    def test_engineer_reuses_own_implementation_branch(
        self, mock_config, mock_logger, mock_queue, mock_worktree_manager, tmp_path
    ):
        """Engineer reuses its own implementation_branch in fix cycles."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        git_ops.multi_repo_manager = MagicMock()
        git_ops.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")

        worktree_path = tmp_path / "engineer-ME-1"
        worktree_path.mkdir()
        mock_worktree_manager.find_worktree_by_branch.return_value = worktree_path

        from datetime import datetime, timezone
        task = Task(
            id="task-fix", title="Fix", description="Fix", type="implementation",
            status="pending", priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            context={
                "github_repo": "owner/repo",
                "implementation_branch": "agent/engineer/ME-1-abc12345",
            },
        )

        result = git_ops.get_working_directory(task)

        assert result == worktree_path
        assert git_ops._active_worktree == worktree_path
        mock_worktree_manager.create_worktree.assert_not_called()


class TestIsOwnBranch:
    """Tests for _is_own_branch method."""

    def test_same_agent_branch(self, mock_config, mock_logger, mock_queue, tmp_path):
        """Branch created by the same agent is recognized as own."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue, logger=mock_logger,
        )
        assert git_ops._is_own_branch("agent/engineer/PROJ-123-abc") is True

    def test_replica_branch(self, mock_config, mock_logger, mock_queue, tmp_path):
        """Branch created by a replica (engineer-2) is recognized as own."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue, logger=mock_logger,
        )
        assert git_ops._is_own_branch("agent/engineer-2/PROJ-123-abc") is True

    def test_different_agent_branch(self, mock_config, mock_logger, mock_queue, tmp_path):
        """Branch created by a different agent is not own."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue, logger=mock_logger,
        )
        assert git_ops._is_own_branch("agent/architect/PROJ-123-abc") is False

    def test_non_agent_branch(self, mock_config, mock_logger, mock_queue, tmp_path):
        """Non-agent branch pattern is not own."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue, logger=mock_logger,
        )
        assert git_ops._is_own_branch("feature/some-branch") is False

    def test_partial_prefix_match_rejected(self, mock_config, mock_logger, mock_queue, tmp_path):
        """agent/engineering/... should not match agent/engineer."""
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue, logger=mock_logger,
        )
        assert git_ops._is_own_branch("agent/engineering/PROJ-123") is False


class TestPreflightWorktreeExistence:
    """Tests for pre-flight worktree path existence check in get_working_directory."""

    def test_recreates_worktree_when_path_missing(self, mock_config, mock_logger, mock_queue, tmp_path):
        """When find_worktree_by_branch returns a path that no longer exists, recreate it."""
        mock_worktree_manager = MagicMock()
        git_ops = GitOperationsManager(
            config=mock_config, workspace=tmp_path, queue=mock_queue,
            logger=mock_logger, worktree_manager=mock_worktree_manager,
        )
        git_ops.multi_repo_manager = MagicMock()
        git_ops.multi_repo_manager.ensure_repo.return_value = Path("/base/repo")

        # find_worktree_by_branch returns a path that doesn't exist on disk
        ghost_path = tmp_path / "ghost-worktree"
        mock_worktree_manager.find_worktree_by_branch.return_value = ghost_path
        mock_worktree_manager.create_worktree.return_value = tmp_path / "new-worktree"

        from datetime import datetime, timezone
        task = Task(
            id="task-preflight", title="T", description="D", type="implementation",
            status="pending", priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            context={
                "github_repo": "owner/repo",
                "worktree_branch": "agent/engineer/ME-1-abc",
            },
        )

        result = git_ops.get_working_directory(task)

        # Should have removed the stale entry and created a new worktree
        mock_worktree_manager.remove_worktree.assert_called_once_with(ghost_path, force=True)
        mock_worktree_manager.create_worktree.assert_called_once()
        assert result == tmp_path / "new-worktree"
