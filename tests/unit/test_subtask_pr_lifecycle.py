"""Tests for subtask PR lifecycle: prompt suppression and orphan cleanup."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.config import WorkflowDefinition
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(task_id="task-abc", parent_task_id=None, **ctx_overrides):
    context = {"workflow": "default", "github_repo": "org/repo", **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
        parent_task_id=parent_task_id,
    )


@pytest.fixture
def agent(tmp_path):
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = MagicMock()
    a.workspace = tmp_path
    a._workflows_config = {
        "default": WorkflowDefinition(description="default", agents=["engineer"]),
    }
    a._agents_config = [SimpleNamespace(id="engineer")]
    a._team_mode_enabled = False
    a._mcp_enabled = False
    a._optimization_config = {}
    a._guidance_cache = {}
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a._active_worktree = None
    a.worktree_manager = None
    a.multi_repo_manager = None
    a.jira_client = None
    a._agent_definition = None

    # Create prompt builder with mock context
    prompt_ctx = PromptContext(
        config=config,
        workspace=tmp_path,
        mcp_enabled=False,
        optimization_config={},
    )
    prompt_builder = PromptBuilder(prompt_ctx)
    a._prompt_builder = prompt_builder

    # Initialize GitOperationsManager
    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=a.config,
        workspace=a.workspace,
        queue=a.queue,
        logger=a.logger,
        session_logger=a._session_logger if hasattr(a, '_session_logger') else None,
    )
    return a


# -- Prompt suppression --

class TestSubtaskPromptSuppression:
    def test_subtask_prompt_includes_suppression(self, agent):
        """Subtask LLMs should be told not to create PRs."""
        task = _make_task(parent_task_id="parent-1")
        prompt = agent._prompt_builder._build_prompt_legacy(task)
        assert "do NOT create a pull request" in prompt
        assert "SUBTASK" in prompt

    def test_non_subtask_prompt_excludes_suppression(self, agent):
        """Regular tasks should NOT see subtask PR suppression."""
        task = _make_task()
        prompt = agent._prompt_builder._build_prompt_legacy(task)
        assert "You are a SUBTASK" not in prompt

    def test_optimized_prompt_includes_suppression_for_subtask(self, agent):
        """Optimized prompt path also suppresses PR creation for subtasks."""
        task = _make_task(parent_task_id="parent-1")
        prompt = agent._prompt_builder._build_prompt_optimized(task)
        assert "do NOT create a pull request" in prompt


# -- _close_subtask_prs --

class TestCloseSubtaskPrs:
    def test_noop_for_non_fan_in(self, agent):
        """Should not close anything for regular tasks."""
        task = _make_task()
        agent._git_ops._close_subtask_prs = Agent._close_subtask_prs.__get__(agent)
        agent._git_ops._close_subtask_prs(task, "https://github.com/org/repo/pull/10")
        agent.queue.find_task.assert_not_called()

    @patch("agent_framework.utils.subprocess_utils.run_command")
    def test_closes_subtask_prs(self, mock_run, agent):
        """Fan-in should close orphaned PRs from each subtask."""
        # Fan-in tasks store parent_task_id in context, not at model level
        fan_in_task = _make_task(task_id="fan-in-1", fan_in=True)
        fan_in_task.context["parent_task_id"] = "parent-1"

        parent = _make_task(task_id="parent-1")
        parent.subtask_ids = ["sub-0", "sub-1"]
        agent.queue.find_task.return_value = parent

        sub0 = _make_task(task_id="sub-0", pr_url="https://github.com/org/repo/pull/18")
        sub1 = _make_task(task_id="sub-1")  # no PR
        agent.queue.get_completed.side_effect = lambda sid: {"sub-0": sub0, "sub-1": sub1}[sid]

        agent._git_ops._close_subtask_prs = Agent._close_subtask_prs.__get__(agent)
        agent._git_ops._close_subtask_prs(fan_in_task, "https://github.com/org/repo/pull/20")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["gh", "pr", "close"]
        assert "18" in cmd

    def test_skips_when_no_parent(self, agent):
        """Fan-in without parent_task_id in context does nothing."""
        task = _make_task(fan_in=True)  # no parent_task_id in context
        agent._git_ops._close_subtask_prs = Agent._close_subtask_prs.__get__(agent)
        agent._git_ops._close_subtask_prs(task, "https://github.com/org/repo/pull/10")
        agent.queue.find_task.assert_not_called()


# -- _cleanup_subtask_branches --

class TestCleanupSubtaskBranches:
    def test_noop_for_non_fan_in(self, agent):
        """Should not delete anything for regular tasks."""
        task = _make_task()
        agent._git_ops._cleanup_subtask_branches = Agent._cleanup_subtask_branches.__get__(agent)
        agent._git_ops._cleanup_subtask_branches(task)
        agent.queue.find_task.assert_not_called()

    @patch("agent_framework.utils.subprocess_utils.run_command")
    def test_deletes_subtask_branches(self, mock_run, agent):
        """Fan-in should delete remote branches from each subtask."""
        # Fan-in tasks store parent_task_id in context, not at model level
        fan_in_task = _make_task(task_id="fan-in-1", fan_in=True)
        fan_in_task.context["parent_task_id"] = "parent-1"

        parent = _make_task(task_id="parent-1")
        parent.subtask_ids = ["sub-0", "sub-1"]
        agent.queue.find_task.return_value = parent

        sub0 = _make_task(task_id="sub-0", implementation_branch="agent/engineer/sub-0")
        sub1 = _make_task(task_id="sub-1", worktree_branch="agent/engineer/sub-1")
        agent.queue.get_completed.side_effect = lambda sid: {"sub-0": sub0, "sub-1": sub1}[sid]

        agent._git_ops._cleanup_subtask_branches = Agent._cleanup_subtask_branches.__get__(agent)
        agent._git_ops._cleanup_subtask_branches(fan_in_task)

        assert mock_run.call_count == 2
        branches_deleted = [c[0][0][-1] for c in mock_run.call_args_list]
        assert "agent/engineer/sub-0" in branches_deleted
        assert "agent/engineer/sub-1" in branches_deleted
