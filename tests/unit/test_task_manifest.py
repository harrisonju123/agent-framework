"""Tests for task manifest module."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.task_manifest import (
    TaskManifest,
    load_manifest,
    save_manifest,
    get_or_create_manifest,
    _manifest_path,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def sample_manifest():
    return TaskManifest(
        root_task_id="root-123",
        branch="agent/engineer/PROJ-42-abc",
        github_repo="owner/repo",
        user_goal="Implement feature X",
        workflow="default",
        working_directory="/tmp/workspace",
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by="engineer",
    )


class TestSaveAndLoadRoundtrip:
    def test_roundtrip(self, workspace, sample_manifest):
        save_manifest(workspace, sample_manifest)
        loaded = load_manifest(workspace, "root-123")

        assert loaded is not None
        assert loaded.root_task_id == "root-123"
        assert loaded.branch == "agent/engineer/PROJ-42-abc"
        assert loaded.github_repo == "owner/repo"
        assert loaded.user_goal == "Implement feature X"
        assert loaded.workflow == "default"
        assert loaded.working_directory == "/tmp/workspace"
        assert loaded.created_by == "engineer"

    def test_creates_directory(self, workspace, sample_manifest):
        save_manifest(workspace, sample_manifest)
        manifest_dir = workspace / ".agent-communication" / "manifests"
        assert manifest_dir.exists()

    def test_file_is_valid_json(self, workspace, sample_manifest):
        save_manifest(workspace, sample_manifest)
        path = _manifest_path(workspace, "root-123")
        data = json.loads(path.read_text())
        assert data["branch"] == "agent/engineer/PROJ-42-abc"


class TestGetOrCreate:
    def test_creates_new(self, workspace):
        manifest = get_or_create_manifest(
            workspace,
            root_task_id="new-task",
            branch="agent/engineer/PROJ-99-xyz",
            github_repo="owner/repo",
            created_by="engineer",
        )

        assert manifest.root_task_id == "new-task"
        assert manifest.branch == "agent/engineer/PROJ-99-xyz"
        assert manifest.created_at != ""

        # File exists on disk
        path = _manifest_path(workspace, "new-task")
        assert path.exists()

    def test_returns_existing_ignores_new_branch(self, workspace):
        """The immutability invariant: second call with different branch returns original."""
        first = get_or_create_manifest(
            workspace,
            root_task_id="task-1",
            branch="agent/engineer/original-branch",
        )

        second = get_or_create_manifest(
            workspace,
            root_task_id="task-1",
            branch="agent/engineer/different-branch",
        )

        assert second.branch == "agent/engineer/original-branch"
        assert second.branch == first.branch

    def test_kwargs_ignored_on_existing(self, workspace):
        get_or_create_manifest(
            workspace,
            root_task_id="task-2",
            branch="branch-a",
            github_repo="owner/repo-a",
        )

        manifest = get_or_create_manifest(
            workspace,
            root_task_id="task-2",
            branch="branch-b",
            github_repo="owner/repo-b",
        )

        assert manifest.github_repo == "owner/repo-a"


class TestLoadEdgeCases:
    def test_load_missing_returns_none(self, workspace):
        assert load_manifest(workspace, "nonexistent") is None

    def test_load_corrupt_returns_none(self, workspace):
        manifest_dir = workspace / ".agent-communication" / "manifests"
        manifest_dir.mkdir(parents=True)
        path = manifest_dir / "corrupt-task.json"
        path.write_text("not valid json {{{")

        assert load_manifest(workspace, "corrupt-task") is None

    def test_load_missing_branch_returns_none(self, workspace):
        manifest_dir = workspace / ".agent-communication" / "manifests"
        manifest_dir.mkdir(parents=True)
        path = manifest_dir / "no-branch.json"
        path.write_text(json.dumps({"root_task_id": "no-branch"}))

        assert load_manifest(workspace, "no-branch") is None

    def test_load_tolerates_unknown_fields(self, workspace):
        """Future schema additions don't crash old code."""
        manifest_dir = workspace / ".agent-communication" / "manifests"
        manifest_dir.mkdir(parents=True)
        path = manifest_dir / "future-task.json"
        path.write_text(json.dumps({
            "root_task_id": "future-task",
            "branch": "feature/x",
            "some_future_field": "ignored",
        }))

        manifest = load_manifest(workspace, "future-task")
        assert manifest is not None
        assert manifest.branch == "feature/x"


class TestSafetyCommitBranchVerification:
    """Test that safety_commit verifies HEAD matches the manifest branch."""

    def test_corrects_branch_mismatch(self, tmp_path):
        from agent_framework.core.git_operations import GitOperationsManager

        config = MagicMock()
        config.id = "engineer"
        config.base_id = "engineer"
        session_logger = MagicMock()

        git_ops = GitOperationsManager(
            config=config,
            workspace=tmp_path,
            queue=MagicMock(),
            logger=MagicMock(),
            session_logger=session_logger,
        )

        # Write a manifest
        get_or_create_manifest(
            tmp_path, root_task_id="root-abc", branch="feature/correct"
        )
        git_ops._active_root_task_id = "root-abc"

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            # Simulate: status says dirty, branch --show-current says wrong branch
            status_result = MagicMock(returncode=0, stdout=" M file.py\n")
            branch_result = MagicMock(returncode=0, stdout="main\n")
            checkout_result = MagicMock(returncode=0)
            add_result = MagicMock(returncode=0)
            commit_result = MagicMock(returncode=0, stderr="")

            mock_git.side_effect = [
                status_result,    # status --porcelain
                branch_result,    # branch --show-current
                checkout_result,  # checkout feature/correct
                add_result,       # add -A
                commit_result,    # commit
            ]

            working_dir = tmp_path / "repo"
            working_dir.mkdir()

            git_ops.safety_commit(working_dir, "test commit")

            # Verify checkout was called with the manifest branch
            checkout_call = mock_git.call_args_list[2]
            assert checkout_call[0][0] == ["checkout", "feature/correct"]

            # Verify session event was logged
            session_logger.log.assert_called_once_with(
                "manifest_branch_correction",
                expected="feature/correct",
                actual="main",
                root_task_id="root-abc",
            )

    def test_no_correction_when_branch_matches(self, tmp_path):
        from agent_framework.core.git_operations import GitOperationsManager

        config = MagicMock()
        config.id = "engineer"
        config.base_id = "engineer"

        git_ops = GitOperationsManager(
            config=config,
            workspace=tmp_path,
            queue=MagicMock(),
            logger=MagicMock(),
        )

        get_or_create_manifest(
            tmp_path, root_task_id="root-ok", branch="feature/correct"
        )
        git_ops._active_root_task_id = "root-ok"

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            status_result = MagicMock(returncode=0, stdout=" M file.py\n")
            branch_result = MagicMock(returncode=0, stdout="feature/correct\n")
            add_result = MagicMock(returncode=0)
            commit_result = MagicMock(returncode=0, stderr="")

            mock_git.side_effect = [
                status_result,   # status --porcelain
                branch_result,   # branch --show-current (matches manifest)
                add_result,      # add -A
                commit_result,   # commit
            ]

            working_dir = tmp_path / "repo"
            working_dir.mkdir()

            result = git_ops.safety_commit(working_dir, "test")
            assert result is True

            # No checkout call — only status, branch check, add, commit
            assert mock_git.call_count == 4


class TestDetectImplementationBranchPrefersManifest:
    def test_prefers_manifest(self, tmp_path):
        from agent_framework.core.git_operations import GitOperationsManager

        config = MagicMock()
        config.id = "engineer"
        config.base_id = "engineer"

        git_ops = GitOperationsManager(
            config=config,
            workspace=tmp_path,
            queue=MagicMock(),
            logger=MagicMock(),
        )

        task = Task(
            id="task-detect",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test",
            description="Test",
            context={"_root_task_id": "root-detect"},
        )

        get_or_create_manifest(
            tmp_path, root_task_id="root-detect", branch="agent/engineer/manifest-branch"
        )

        # Even without an active worktree, manifest branch is used
        git_ops.detect_implementation_branch(task)
        assert task.context["implementation_branch"] == "agent/engineer/manifest-branch"


class TestChainTaskStampsManifestBranch:
    """Test that _build_chain_task stamps manifest branch into context."""

    def test_stamps_manifest_branch(self, tmp_path):
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        queue = MagicMock()
        executor = WorkflowExecutor(
            queue=queue,
            queue_dir=tmp_path / "queues",
            workspace=tmp_path,
        )

        get_or_create_manifest(
            tmp_path, root_task_id="root-chain", branch="agent/engineer/chain-branch"
        )

        task = Task(
            id="chain-root-chain-plan-d1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan task",
            description="Plan the work",
            context={
                "_root_task_id": "root-chain",
                "workflow_step": "plan",
                "workflow": "default",
                "worktree_branch": "agent/engineer/chain-branch",
            },
        )

        step = WorkflowStep(
            id="implement",
            agent="engineer",
        )

        chain_task = executor._build_chain_task(task, step, "architect")

        assert chain_task.context["implementation_branch"] == "agent/engineer/chain-branch"
        assert chain_task.context["worktree_branch"] == "agent/engineer/chain-branch"

    def test_no_manifest_no_stamp(self, tmp_path):
        """Without a manifest, the branch comes from task.context as before."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        queue = MagicMock()
        executor = WorkflowExecutor(
            queue=queue,
            queue_dir=tmp_path / "queues",
            workspace=tmp_path,
        )

        task = Task(
            id="chain-root-nomnfst-plan-d1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan task",
            description="Plan the work",
            context={
                "_root_task_id": "root-nomnfst",
                "workflow_step": "plan",
                "workflow": "default",
                "worktree_branch": "agent/architect/original",
            },
        )

        step = WorkflowStep(id="implement", agent="engineer")

        chain_task = executor._build_chain_task(task, step, "architect")

        # worktree_branch propagated from parent context, not overwritten
        assert chain_task.context["worktree_branch"] == "agent/architect/original"


class TestRetryLoadsManifestBranch:
    """Test that _get_validated_working_directory loads manifest on retries."""

    def test_retry_sets_context_from_manifest(self, tmp_path):
        from agent_framework.core.agent import Agent

        get_or_create_manifest(
            tmp_path, root_task_id="root-retry", branch="agent/engineer/retry-branch"
        )

        agent = MagicMock()
        agent.workspace = tmp_path

        task = Task(
            id="task-retry",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Retry task",
            description="Retry test",
            retry_count=1,
            context={"_root_task_id": "root-retry"},
        )

        # Mock _git_ops.get_working_directory to return a real directory
        mock_working_dir = tmp_path / "work"
        mock_working_dir.mkdir()
        # Create at least one file so iterdir() works
        (mock_working_dir / "dummy").touch()

        agent._git_ops = MagicMock()
        agent._git_ops.get_working_directory.return_value = mock_working_dir
        agent._session_logger = MagicMock()

        # Bind the real method to our mock
        agent._get_validated_working_directory = Agent._get_validated_working_directory.__get__(agent)

        result = agent._get_validated_working_directory(task)

        assert task.context["implementation_branch"] == "agent/engineer/retry-branch"
        assert task.context["worktree_branch"] == "agent/engineer/retry-branch"
        assert result == mock_working_dir

    def test_no_retry_no_manifest_load(self, tmp_path):
        """First attempt (retry_count=0) doesn't pre-load from manifest."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent.workspace = tmp_path

        task = Task(
            id="task-first",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="First attempt",
            description="First test",
            retry_count=0,
            context={"_root_task_id": "root-first"},
        )

        mock_working_dir = tmp_path / "work"
        mock_working_dir.mkdir()
        (mock_working_dir / "dummy").touch()

        agent._git_ops = MagicMock()
        agent._git_ops.get_working_directory.return_value = mock_working_dir
        agent._session_logger = MagicMock()

        agent._get_validated_working_directory = Agent._get_validated_working_directory.__get__(agent)

        agent._get_validated_working_directory(task)

        # No branch keys set by manifest loading
        assert "implementation_branch" not in task.context
        assert "worktree_branch" not in task.context


class TestPromptBuilderManifestContext:
    """Test that prompt builder injects manifest context."""

    def test_manifest_rendered_in_prompt(self, tmp_path):
        from agent_framework.core.prompt_builder import PromptBuilder, PromptContext

        get_or_create_manifest(
            tmp_path,
            root_task_id="root-prompt",
            branch="agent/engineer/PROJ-42-abc",
            github_repo="owner/repo",
        )

        ctx = MagicMock(spec=PromptContext)
        ctx.workspace = tmp_path
        ctx.config = MagicMock()
        ctx.config.id = "engineer"
        ctx.mcp_enabled = False
        ctx.agent_definition = None
        ctx.optimization_config = {}
        ctx.workflows_config = {}

        builder = PromptBuilder(ctx)

        task = Task(
            id="task-prompt",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test",
            description="Test",
            context={"_root_task_id": "root-prompt"},
        )

        context = builder._load_manifest_context(task)

        assert "agent/engineer/PROJ-42-abc" in context
        assert "owner/repo" in context
        assert "TASK MANIFEST" in context

    def test_no_manifest_returns_empty(self, tmp_path):
        from agent_framework.core.prompt_builder import PromptBuilder, PromptContext

        ctx = MagicMock(spec=PromptContext)
        ctx.workspace = tmp_path

        builder = PromptBuilder(ctx)

        task = Task(
            id="task-nomanifest",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test",
            description="Test",
            context={"_root_task_id": "root-nomanifest"},
        )

        context = builder._load_manifest_context(task)
        assert context == ""


class TestWriteManifestIfNeeded:
    """Test manifest creation during get_working_directory."""

    def test_manifest_created_on_direct_repo_checkout(self, tmp_path):
        from agent_framework.core.git_operations import GitOperationsManager

        config = MagicMock()
        config.id = "engineer"
        config.base_id = "engineer"

        git_ops = GitOperationsManager(
            config=config,
            workspace=tmp_path,
            queue=MagicMock(),
            logger=MagicMock(),
        )

        task = Task(
            id="task-wd",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Test",
            description="Test feature",
            context={
                "_root_task_id": "root-wd",
                "github_repo": "owner/repo",
                "worktree_branch": "agent/engineer/PROJ-1-abc",
            },
        )

        with patch("agent_framework.core.git_operations.GitOperationsManager._checkout_or_create_branch"):
            working_dir = git_ops.get_working_directory(task)

        manifest = load_manifest(tmp_path, "root-wd")
        assert manifest is not None
        assert manifest.branch == "agent/engineer/PROJ-1-abc"
        assert manifest.github_repo == "owner/repo"
        assert manifest.created_by == "engineer"
