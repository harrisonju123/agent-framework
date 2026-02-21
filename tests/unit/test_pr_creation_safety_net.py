"""Tests for PR creation safety net (_push_and_create_pr_if_needed and helpers)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import subprocess

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.config import WorkflowDefinition
from agent_framework.core.git_operations import GitOperationsManager
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Fixtures --

def _make_task(workflow="default", task_id="task-abc123def456", **ctx_overrides):
    context = {"workflow": workflow, "github_repo": "org/repo", **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


TERMINAL_WORKFLOW = WorkflowDefinition(
    description="Single-step workflow (terminal)",
    agents=["engineer"],
)

CHAIN_WORKFLOW = WorkflowDefinition(
    description="Multi-step workflow",
    agents=["engineer", "qa"],
)


@pytest.fixture
def queue(tmp_path):
    q = MagicMock()
    q.queue_dir = tmp_path / "queues"
    q.queue_dir.mkdir()
    return q


@pytest.fixture
def agent(queue, tmp_path):
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a.workspace = tmp_path
    a._workflows_config = {
        "default": TERMINAL_WORKFLOW,
        "chain": CHAIN_WORKFLOW,
    }
    a._agents_config = [
        SimpleNamespace(id="engineer"),
        SimpleNamespace(id="qa"),
    ]
    a._team_mode_enabled = False
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a._active_worktree = None
    a.worktree_manager = None
    a.multi_repo_manager = None

    from agent_framework.workflow.executor import WorkflowExecutor
    a._workflow_executor = WorkflowExecutor(queue, queue.queue_dir)

    # Initialize GitOperationsManager
    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=a.config,
        workspace=a.workspace,
        queue=a.queue,
        logger=a.logger,
        worktree_manager=a.worktree_manager,
        multi_repo_manager=a.multi_repo_manager,
        session_logger=a._session_logger if hasattr(a, '_session_logger') else None,
        workflows_config=a._workflows_config,
    )

    # Initialize workflow router (required after extraction)
    from agent_framework.core.workflow_router import WorkflowRouter
    a._workflow_router = WorkflowRouter(
        config=config,
        queue=queue,
        workspace=tmp_path,
        logger=a.logger,
        session_logger=a._session_logger,
        workflows_config=a._workflows_config,
        workflow_executor=a._workflow_executor,
        agents_config=a._agents_config,
        multi_repo_manager=None,
    )

    return a


# -- _push_and_create_pr_if_needed --

class TestPushAndCreatePrIfNeeded:
    def test_skips_when_pr_url_exists(self, agent):
        """Short-circuits immediately when task already has a pr_url."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/1")
        agent._git_ops.push_and_create_pr_if_needed(task)
        agent.logger.debug.assert_any_call(
            f"PR already exists for {task.id}: https://github.com/org/repo/pull/1"
        )

    def test_skips_without_active_working_dir(self, agent):
        """No active working directory + no implementation branch = nothing to do."""
        task = _make_task()
        agent._git_ops._active_worktree = None
        agent._git_ops.push_and_create_pr_if_needed(task)
        agent.logger.debug.assert_any_call(
            f"No active working directory for {task.id}, skipping PR creation"
        )

    @patch.object(GitOperationsManager, '_has_unpushed_commits', return_value=True)
    def test_skips_without_github_repo(self, mock_has_unpushed, agent, tmp_path):
        """Has active working dir + unpushed commits but no github_repo in context."""
        task = _make_task()
        del task.context["github_repo"]

        agent._git_ops._active_worktree = tmp_path

        agent._git_ops.push_and_create_pr_if_needed(task)
        agent.logger.debug.assert_any_call(
            f"No github_repo in task context for {task.id}, skipping PR creation"
        )

    def test_delegates_to_create_pr_from_branch_for_pr_creation_step(self, agent):
        """pr_creation_step tasks with implementation_branch use _create_pr_from_branch."""
        task = _make_task(
            pr_creation_step=True,
            implementation_branch="feat/my-branch",
        )
        agent._git_ops._create_pr_from_branch = MagicMock()
        agent._git_ops.push_and_create_pr_if_needed(task)
        agent._git_ops._create_pr_from_branch.assert_called_once_with(task, "feat/my-branch")

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    @patch.object(GitOperationsManager, '_has_unpushed_commits', return_value=True)
    def test_skips_pr_on_intermediate_step(self, mock_has_unpushed, mock_git, agent, tmp_path):
        """Intermediate workflow steps push but don't create a PR."""
        task = _make_task(workflow="chain")
        agent._git_ops._active_worktree = tmp_path

        # git rev-parse returns a feature branch
        branch_result = MagicMock(returncode=0, stdout="feat/my-branch\n")
        push_result = MagicMock(returncode=0, stderr="")
        mock_git.side_effect = [branch_result, push_result]

        agent._git_ops.push_and_create_pr_if_needed(task)

        assert task.context["implementation_branch"] == "feat/my-branch"
        agent.logger.info.assert_any_call(
            "Intermediate step — pushed feat/my-branch but skipped PR creation"
        )

    @patch("agent_framework.utils.subprocess_utils.run_command")
    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    @patch.object(GitOperationsManager, '_has_unpushed_commits', return_value=True)
    def test_creates_pr_on_terminal_step(self, mock_has_unpushed, mock_git, mock_cmd, agent, tmp_path):
        """Terminal workflow step pushes and creates PR."""
        task = _make_task(workflow="default")
        agent._git_ops._active_worktree = tmp_path

        # git rev-parse returns feature branch, push succeeds
        branch_result = MagicMock(returncode=0, stdout="feat/my-branch\n")
        push_result = MagicMock(returncode=0, stderr="")
        mock_git.side_effect = [branch_result, push_result]

        # gh pr create succeeds
        mock_cmd.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/org/repo/pull/42\n",
            stderr="",
        )

        agent._git_ops.push_and_create_pr_if_needed(task)

        assert task.context["pr_url"] == "https://github.com/org/repo/pull/42"

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    @patch.object(GitOperationsManager, '_has_unpushed_commits', return_value=True)
    def test_skips_pr_from_main_branch(self, mock_has_unpushed, mock_git, agent, tmp_path):
        """Never create PRs when on main/master."""
        task = _make_task()
        agent._git_ops._active_worktree = tmp_path

        mock_git.return_value = MagicMock(returncode=0, stdout="main\n")

        agent._git_ops.push_and_create_pr_if_needed(task)
        assert "pr_url" not in task.context


# -- _create_pr_via_gh --

class TestCreatePrViaGh:
    @patch("agent_framework.utils.subprocess_utils.run_command")
    def test_creates_pr_and_stores_url(self, mock_cmd, agent, tmp_path):
        task = _make_task()
        mock_cmd.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/org/repo/pull/99\n",
            stderr="",
        )

        agent._git_ops._create_pr_via_gh(task, "org/repo", "feat/branch", cwd=tmp_path)
        assert task.context["pr_url"] == "https://github.com/org/repo/pull/99"

    @patch("agent_framework.utils.subprocess_utils.run_command")
    def test_handles_pr_already_exists(self, mock_cmd, agent, tmp_path):
        task = _make_task()
        mock_cmd.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="a]pull request already exists for branch",
        )

        agent._git_ops._create_pr_via_gh(task, "org/repo", "feat/branch", cwd=tmp_path)
        assert "pr_url" not in task.context
        agent.logger.info.assert_any_call("PR already exists for this branch")

    @patch("agent_framework.utils.subprocess_utils.run_command")
    def test_strips_chain_prefix_from_title(self, mock_cmd, agent, tmp_path):
        task = _make_task()
        task.title = "[chain] Implement feature X"
        mock_cmd.return_value = MagicMock(returncode=0, stdout="url\n", stderr="")

        agent._git_ops._create_pr_via_gh(task, "org/repo", "feat/branch", cwd=tmp_path)

        call_args = mock_cmd.call_args[0][0]
        title_idx = call_args.index("--title") + 1
        assert not call_args[title_idx].startswith("[chain]")


# -- _remote_branch_exists --

class TestRemoteBranchExists:
    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_true_when_remote_has_branch(self, mock_git, agent, tmp_path):
        mock_git.side_effect = [
            MagicMock(returncode=0, stdout="feat/branch\n"),  # rev-parse
            MagicMock(returncode=0, stdout="abc123\trefs/heads/feat/branch\n"),  # ls-remote
        ]
        assert agent._git_ops._remote_branch_exists(tmp_path) is True

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_false_for_main(self, mock_git, agent, tmp_path):
        mock_git.return_value = MagicMock(returncode=0, stdout="main\n")
        assert agent._git_ops._remote_branch_exists(tmp_path) is False

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_false_when_no_remote_branch(self, mock_git, agent, tmp_path):
        mock_git.side_effect = [
            MagicMock(returncode=0, stdout="feat/branch\n"),
            MagicMock(returncode=0, stdout=""),  # empty ls-remote
        ]
        assert agent._git_ops._remote_branch_exists(tmp_path) is False

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_false_on_detached_head(self, mock_git, agent, tmp_path):
        mock_git.return_value = MagicMock(returncode=0, stdout="HEAD\n")
        assert agent._git_ops._remote_branch_exists(tmp_path) is False

    @patch("agent_framework.utils.subprocess_utils.run_git_command")
    def test_returns_false_on_subprocess_error(self, mock_git, agent, tmp_path):
        from agent_framework.utils.subprocess_utils import SubprocessError
        mock_git.side_effect = SubprocessError(
            cmd="git rev-parse", returncode=128, stderr="fatal"
        )
        assert agent._git_ops._remote_branch_exists(tmp_path) is False
