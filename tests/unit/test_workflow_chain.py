"""Tests for workflow chain enforcement."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.config import WorkflowDefinition
from agent_framework.core.routing import RoutingSignal, WORKFLOW_COMPLETE
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Fixtures --

def _make_task(workflow="default", task_id="task-abc123def456", **ctx_overrides):
    context = {"workflow": workflow, **ctx_overrides}
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
    )


def _make_response(content="Done.", pr_url=None):
    """Minimal LLM response stub."""
    return SimpleNamespace(
        content=content if not pr_url else f"Created PR: {pr_url}",
        error=None,
        input_tokens=100,
        output_tokens=50,
        model_used="sonnet",
        latency_ms=1000,
        finish_reason="end_turn",
    )


DEFAULT_WORKFLOW = WorkflowDefinition(
    description="Default workflow",
    agents=["architect", "engineer", "qa"],
)

PR_WORKFLOW = WorkflowDefinition(
    description="Workflow with PR creator",
    agents=["architect", "engineer", "qa"],
    pr_creator="architect",
)

ANALYSIS_WORKFLOW = WorkflowDefinition(
    description="Analysis only",
    agents=["architect"],
)


@pytest.fixture
def queue(tmp_path):
    """FileQueue mock with real queue_dir and completed_dir for file existence checks."""
    q = MagicMock()
    q.queue_dir = tmp_path / "queues"
    q.queue_dir.mkdir()
    q.completed_dir = tmp_path / "completed"
    q.completed_dir.mkdir()
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
    a._workflows_config = {"default": DEFAULT_WORKFLOW, "analysis": ANALYSIS_WORKFLOW}
    a._agents_config = [
        SimpleNamespace(id="architect"),
        SimpleNamespace(id="engineer"),
        SimpleNamespace(id="qa"),
    ]
    a._team_mode_enabled = False
    a.logger = MagicMock()
    a._session_logger = MagicMock()

    # Initialize workflow executor (required by new DAG implementation)
    from agent_framework.workflow.executor import WorkflowExecutor
    a._workflow_executor = WorkflowExecutor(queue, queue.queue_dir)

    return a


# -- AgentConfig.base_id --

class TestBaseId:
    def test_plain_id(self):
        c = AgentConfig(id="engineer", name="E", queue="e", prompt="p")
        assert c.base_id == "engineer"

    def test_replica_id(self):
        c = AgentConfig(id="engineer-2", name="E", queue="e", prompt="p")
        assert c.base_id == "engineer"

    def test_hyphenated_name(self):
        """Agent IDs with hyphens but no numeric suffix stay unchanged."""
        c = AgentConfig(id="code-reviewer", name="CR", queue="cr", prompt="p")
        assert c.base_id == "code-reviewer"

    def test_replica_of_hyphenated(self):
        c = AgentConfig(id="code-reviewer-3", name="CR", queue="cr", prompt="p")
        assert c.base_id == "code-reviewer"


# -- _enforce_workflow_chain --

class TestEnforceWorkflowChain:
    def test_queues_next_agent_no_pr(self, agent, queue):
        """When no PR is created, chain task is queued to the next agent."""
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"
        assert chain_task.id == f"chain-{task.id}-qa"
        assert chain_task.assigned_to == "qa"
        assert chain_task.context["source_task_id"] == task.id
        assert chain_task.context["chain_step"] is True
        assert chain_task.context["workflow_step"] == "qa"

    def test_continues_chain_when_pr_created_at_non_terminal_step(self, agent, queue):
        """PR on intermediate agent (engineer) doesn't kill the chain — QA still runs."""
        task = _make_task(workflow="default")
        task.context["pr_url"] = "https://github.com/org/repo/pull/42"
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_chains_even_with_team_mode(self, agent, queue):
        """Team mode provides advisory subagents but doesn't suppress chain routing."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_chains_with_team_override_false(self, agent, queue):
        """team_override=False still chains to next agent."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default", team_override=False)
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()

    def test_chains_with_team_override_true(self, agent, queue):
        """team_override=True still chains to next agent."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default", team_override=True)
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_skips_duplicate_chain_task(self, agent, queue):
        """If the chain task file already exists, don't queue again."""
        task = _make_task(workflow="default")
        response = _make_response()

        # Pre-create the chain task file
        chain_id = f"chain-{task.id}-qa"
        qa_dir = queue.queue_dir / "qa"
        qa_dir.mkdir()
        (qa_dir / f"{chain_id}.json").write_text("{}")

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_skips_last_agent_in_chain(self, agent, queue):
        """Last agent in the chain has nobody to forward to."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_skips_unknown_workflow(self, agent, queue):
        task = _make_task(workflow="nonexistent")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_skips_single_agent_workflow(self, agent, queue):
        """Single-agent workflows (like analysis) have no chain to enforce."""
        agent.config = AgentConfig(id="architect", name="A", queue="a", prompt="p")
        task = _make_task(workflow="analysis")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_chain_task_type_engineer(self, agent, queue):
        """Engineer tasks get IMPLEMENTATION type."""
        # architect -> engineer chain
        agent.config = AgentConfig(id="architect", name="A", queue="architect", prompt="p")
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.type == TaskType.IMPLEMENTATION

    def test_chain_task_type_qa(self, agent, queue):
        """QA tasks get QA_VERIFICATION type."""
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.type == TaskType.QA_VERIFICATION

    def test_skips_no_workflow_in_context(self, agent, queue):
        """Tasks without a workflow context key are ignored."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_queue_error_is_caught(self, agent, queue):
        """Push failures are logged by executor but don't raise."""
        task = _make_task(workflow="default")
        response = _make_response()
        queue.push.side_effect = OSError("disk full")
        agent._workflow_executor.logger = MagicMock()

        agent._enforce_workflow_chain(task, response)

        agent._workflow_executor.logger.error.assert_called_once()


# -- _normalize_workflow --

class TestNormalizeWorkflow:
    def test_normalize_workflow_maps_old_names(self, agent):
        """Old workflow names get normalized to 'default'."""
        for old_name in ["simple", "standard", "full"]:
            task = _make_task(workflow=old_name)
            agent._normalize_workflow(task)
            assert task.context["workflow"] == "default"

    def test_normalize_workflow_preserves_default(self, agent):
        task = _make_task(workflow="default")
        agent._normalize_workflow(task)
        assert task.context["workflow"] == "default"

    def test_normalize_workflow_preserves_analysis(self, agent):
        task = _make_task(workflow="analysis")
        agent._normalize_workflow(task)
        assert task.context["workflow"] == "analysis"

    def test_normalize_workflow_preserves_unknown(self, agent):
        task = _make_task(workflow="custom")
        agent._normalize_workflow(task)
        assert task.context["workflow"] == "custom"

    def test_normalize_workflow_no_workflow_key(self, agent):
        """Tasks without a workflow key in context are left untouched."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        agent._normalize_workflow(task)
        assert "workflow" not in task.context

    def test_normalize_workflow_none_context(self, agent):
        """Handles task with None context without raising."""
        task = _make_task(workflow="default")
        task.context = None
        agent._normalize_workflow(task)
        assert task.context is None


# -- Routing signal integration --

def _make_signal(target="qa", reason="PR ready for review"):
    return RoutingSignal(
        target_agent=target,
        reason=reason,
        timestamp="2026-02-13T16:30:00Z",
        source_agent="engineer",
    )


class TestRoutingSignalChain:
    def test_signal_overrides_default_chain(self, agent, queue):
        task = _make_task(workflow="default")
        response = _make_response()
        signal = _make_signal(target="architect")

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert chain_task.assigned_to == "architect"

    def test_signal_fallback_on_self_route(self, agent, queue):
        task = _make_task(workflow="default")
        response = _make_response()
        signal = _make_signal(target="engineer")

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_workflow_complete_skips_chain(self, agent, queue):
        task = _make_task(workflow="default")
        response = _make_response()
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_not_called()

    def test_workflow_complete_signal_stops_chain_even_with_pr(self, agent, queue):
        """WORKFLOW_COMPLETE signal terminates chain regardless of pr_url."""
        task = _make_task(workflow="default")
        response = _make_response(pr_url="https://github.com/org/repo/pull/99")
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_not_called()

    def test_no_signal_uses_default_chain(self, agent, queue):
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response, routing_signal=None)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_signal_without_workflow_routes_directly(self, agent, queue):
        task = _make_task(workflow="default")
        del task.context["workflow"]
        response = _make_response()
        signal = _make_signal(target="qa")

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_escalation_task_rejects_signal(self, agent, queue):
        task = _make_task(workflow="default")
        task.type = TaskType.ESCALATION
        response = _make_response()
        signal = _make_signal(target="qa")

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"


# -- PR creation from last agent --

class TestPRCreation:
    """Tests for _queue_pr_creation_if_needed triggered by last agent in chain."""

    @pytest.fixture
    def pr_agent(self, queue, tmp_path):
        """QA agent with a workflow that has pr_creator set."""
        config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.workspace = tmp_path
        a._workflows_config = {
            "default": DEFAULT_WORKFLOW,
            "pr_workflow": PR_WORKFLOW,
            "analysis": ANALYSIS_WORKFLOW,
        }
        a._agents_config = [
            SimpleNamespace(id="architect"),
            SimpleNamespace(id="engineer"),
            SimpleNamespace(id="qa"),
        ]
        a._team_mode_enabled = False
        a.logger = MagicMock()
        a._session_logger = MagicMock()

        from agent_framework.workflow.executor import WorkflowExecutor
        a._workflow_executor = WorkflowExecutor(queue, queue.queue_dir)

        return a

    def test_last_agent_queues_pr_creation(self, pr_agent, queue):
        """QA (last agent) completes → PR_REQUEST queued to architect."""
        task = _make_task(workflow="pr_workflow")
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert pr_task.type == TaskType.PR_REQUEST
        assert pr_task.context["pr_creation_step"] is True
        assert pr_task.title.startswith("[pr]")
        assert pr_task.id == f"chain-{task.id}-architect-pr"

    def test_last_agent_no_pr_creator_configured(self, pr_agent, queue):
        """No pr_creator on workflow → no task queued (DEFAULT_WORKFLOW)."""
        task = _make_task(workflow="default")
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_pr_creation_task_does_not_chain(self, pr_agent, queue):
        """PR creation task landing on architect doesn't chain to engineer."""
        pr_agent.config = AgentConfig(
            id="architect", name="Architect", queue="architect", prompt="p",
        )
        task = _make_task(workflow="pr_workflow", pr_creation_step=True)
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_pr_creation_dedup(self, pr_agent, queue):
        """Duplicate PR creation tasks are not queued."""
        task = _make_task(workflow="pr_workflow")
        response = _make_response()

        # Pre-create the PR task file
        pr_task_id = f"chain-{task.id}-architect-pr"
        architect_dir = queue.queue_dir / "architect"
        architect_dir.mkdir()
        (architect_dir / f"{pr_task_id}.json").write_text("{}")

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_pr_creation_skipped_when_pr_exists(self, pr_agent, queue):
        """PR already exists in task context → chain returns early before PR creation."""
        task = _make_task(workflow="pr_workflow")
        task.context["pr_url"] = "https://github.com/org/repo/pull/42"
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_pr_creation_task_carries_implementation_branch(self, pr_agent, queue):
        """PR creation task inherits implementation_branch from upstream context."""
        task = _make_task(
            workflow="pr_workflow",
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        pr_task = queue.push.call_args[0][0]
        assert pr_task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"


# -- Terminal step detection --

class TestTerminalStepDetection:
    """Tests for _is_at_terminal_workflow_step."""

    def test_intermediate_agent_is_not_terminal(self, agent):
        """Engineer (middle of architect→engineer→qa) is not terminal."""
        task = _make_task(workflow="default")
        assert agent._is_at_terminal_workflow_step(task) is False

    def test_last_agent_is_terminal(self, agent):
        """QA (last in architect→engineer→qa) is terminal."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default")
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_first_agent_is_not_terminal(self, agent):
        """Architect (first in architect→engineer→qa) is not terminal."""
        agent.config = AgentConfig(id="architect", name="A", queue="a", prompt="p")
        task = _make_task(workflow="default")
        assert agent._is_at_terminal_workflow_step(task) is False

    def test_single_agent_workflow_is_terminal(self, agent):
        """Single-agent workflow (analysis: [architect]) — architect is terminal."""
        agent.config = AgentConfig(id="architect", name="A", queue="a", prompt="p")
        task = _make_task(workflow="analysis")
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_no_workflow_is_terminal(self, agent):
        """Standalone tasks (no workflow key) default to terminal."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_unknown_workflow_is_terminal(self, agent):
        """Unknown workflow name defaults to terminal (safe fallback)."""
        task = _make_task(workflow="nonexistent")
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_explicit_workflow_step_used(self, agent):
        """workflow_step in context takes precedence over agent ID lookup."""
        # Engineer agent, but workflow_step says "qa" (terminal)
        task = _make_task(workflow="default", workflow_step="qa")
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_explicit_workflow_step_intermediate(self, agent):
        """workflow_step pointing to engineer (intermediate) returns False."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default", workflow_step="engineer")
        assert agent._is_at_terminal_workflow_step(task) is False


# -- Intermediate step PR suppression --

class TestIntermediateStepPRSuppression:
    """Tests for _push_and_create_pr_if_needed skipping PRs on intermediate steps."""

    def test_intermediate_step_stores_branch_skips_pr(self, agent, tmp_path):
        """Engineer (intermediate) pushes branch but doesn't create a PR."""
        task = _make_task(workflow="default", github_repo="org/repo")

        # Set up a mock worktree with a feature branch
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        agent._active_worktree = worktree_dir
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.has_unpushed_commits.return_value = True

        import subprocess
        from unittest.mock import patch

        # Mock git rev-parse to return a feature branch name
        mock_rev_parse = MagicMock(returncode=0, stdout="agent/engineer/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_rev_parse, mock_push]
            agent._push_and_create_pr_if_needed(task)

        # Branch should be stored for downstream agents
        assert task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"
        # Should NOT have called gh pr create (only git rev-parse + git push = 2 calls)
        assert mock_run.call_count == 2
        assert "pr_url" not in task.context

    def test_terminal_step_creates_pr(self, agent, tmp_path):
        """QA (terminal) creates a PR normally."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default", github_repo="org/repo")

        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        agent._active_worktree = worktree_dir
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.has_unpushed_commits.return_value = True

        from unittest.mock import patch

        mock_rev_parse = MagicMock(returncode=0, stdout="agent/qa/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0)
        mock_pr_create = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/10\n")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [mock_rev_parse, mock_push, mock_pr_create]
            agent._push_and_create_pr_if_needed(task)

        assert task.context["pr_url"] == "https://github.com/org/repo/pull/10"

    def test_pr_creation_task_uses_implementation_branch(self, agent, tmp_path):
        """PR creation task with implementation_branch calls _create_pr_from_branch."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        agent._active_worktree = None
        agent.worktree_manager = None
        agent.multi_repo_manager = MagicMock()
        agent.multi_repo_manager.ensure_repo.return_value = tmp_path

        from unittest.mock import patch

        mock_pr_create = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/11\n")

        with patch("subprocess.run", return_value=mock_pr_create):
            agent._push_and_create_pr_if_needed(task)

        assert task.context["pr_url"] == "https://github.com/org/repo/pull/11"

    def test_pr_creation_task_without_impl_branch_falls_through(self, agent):
        """PR creation task without implementation_branch uses normal flow."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
        )
        agent._active_worktree = None
        agent.worktree_manager = None

        # No worktree + no implementation_branch → early return (no PR)
        agent._push_and_create_pr_if_needed(task)
        assert "pr_url" not in task.context


# -- Worktree skip for PR creation tasks --

class TestWorktreeSkipForPRCreation:
    """Tests for _get_working_directory skipping worktree on PR creation tasks."""

    def test_pr_creation_with_impl_branch_uses_shared_clone(self, agent, tmp_path):
        """PR creation task with implementation_branch uses shared clone, not worktree."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        repo_path = tmp_path / "repos" / "org" / "repo"
        repo_path.mkdir(parents=True)

        agent.multi_repo_manager = MagicMock()
        agent.multi_repo_manager.ensure_repo.return_value = repo_path

        result = agent._get_working_directory(task)

        assert result == repo_path
        agent.multi_repo_manager.ensure_repo.assert_called_once_with("org/repo")

    def test_normal_task_still_creates_worktree(self, agent, tmp_path):
        """Non-PR-creation tasks still go through the normal worktree flow."""
        task = _make_task(workflow="default", github_repo="org/repo")

        repo_path = tmp_path / "repos" / "org" / "repo"
        repo_path.mkdir(parents=True)
        worktree_path = tmp_path / "worktrees" / "agent-engineer"
        worktree_path.mkdir(parents=True)

        agent.multi_repo_manager = MagicMock()
        agent.multi_repo_manager.ensure_repo.return_value = repo_path
        agent.worktree_manager = MagicMock()
        agent.worktree_manager.find_worktree_by_branch.return_value = None
        agent.worktree_manager.create_worktree.return_value = worktree_path

        result = agent._get_working_directory(task)

        assert result == worktree_path
        agent.worktree_manager.create_worktree.assert_called_once()


# -- Preview mode --

class TestPreviewMode:
    """Tests for PREVIEW task type routing and prompt injection."""

    def test_preview_routes_back_to_architect(self, agent, queue):
        """Engineer completing a PREVIEW task routes back to architect, not QA."""
        task = _make_task(workflow="default")
        task.type = TaskType.PREVIEW
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert chain_task.assigned_to == "architect"

    def test_preview_does_not_route_to_qa(self, agent, queue):
        """PREVIEW tasks must skip QA — only architect reviews previews."""
        task = _make_task(workflow="default")
        task.type = TaskType.PREVIEW
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        target_queue = queue.push.call_args[0][1]
        assert target_queue != "qa"

    def test_non_preview_still_routes_normally(self, agent, queue):
        """Regular IMPLEMENTATION tasks still follow the default workflow chain."""
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

    def test_preview_routing_only_applies_to_engineer(self, agent, queue):
        """Architect completing a PREVIEW task should NOT trigger preview routing."""
        agent.config = AgentConfig(
            id="architect", name="Architect", queue="architect", prompt="You are an architect.",
        )
        task = _make_task(workflow="default")
        task.type = TaskType.PREVIEW
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        # Should follow normal workflow chain, not preview routing
        if queue.push.called:
            target_queue = queue.push.call_args[0][1]
            assert target_queue != "architect"

    def test_inject_preview_mode_prepends_constraints(self, agent):
        """_inject_preview_mode prepends read-only constraints before the original prompt."""
        task = _make_task()
        task.type = TaskType.PREVIEW
        original_prompt = "Build feature X with tests."

        result = agent._inject_preview_mode(original_prompt, task)

        assert result.endswith(original_prompt)
        assert "PREVIEW MODE" in result
        assert "Do NOT use Write, Edit, or NotebookEdit" in result
        assert result.index("PREVIEW MODE") < result.index(original_prompt)

    def test_inject_preview_mode_includes_required_sections(self, agent):
        """Preview prompt includes all required output sections."""
        task = _make_task()
        result = agent._inject_preview_mode("original", task)

        assert "### Files to Modify" in result
        assert "### New Files to Create" in result
        assert "### Implementation Approach" in result
        assert "### Risks and Edge Cases" in result
        assert "### Estimated Total Change Size" in result

    def test_preview_stores_artifact_in_context(self, agent, queue):
        """PREVIEW routing stores result_summary as preview_artifact in task context."""
        task = _make_task(workflow="default")
        task.type = TaskType.PREVIEW
        task.result_summary = "## Files to Modify\n- src/foo.py: add bar method"
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        assert task.context["preview_artifact"] == task.result_summary
        queue.update.assert_called_once_with(task)

    def test_preview_artifact_injected_in_implementation_prompt(self, agent):
        """Implementation tasks with preview_artifact get it injected into prompt."""
        task = _make_task()
        task.context["preview_artifact"] = "Approved plan: modify foo.py"

        result = agent._inject_preview_artifact("Build feature X.", task)

        assert "APPROVED PREVIEW" in result
        assert "Approved plan: modify foo.py" in result

    def test_no_preview_artifact_leaves_prompt_unchanged(self, agent):
        """Tasks without preview_artifact don't modify the prompt."""
        task = _make_task()
        original = "Build feature X."

        result = agent._inject_preview_artifact(original, task)

        assert result == original


# -- Chain ID collision and title accumulation --

class TestChainIdCollision:
    """Regression tests for chain ID truncation bug that caused task dedup collisions."""

    def test_similar_prefix_tasks_produce_different_chain_ids(self, agent, queue):
        """Two tasks sharing a 12-char prefix must get distinct chain IDs."""
        task_a = _make_task(task_id="planning-s6-auth-login")
        task_b = _make_task(task_id="planning-s6-auth-signup")

        agent._enforce_workflow_chain(task_a, _make_response())
        chain_a = queue.push.call_args[0][0]

        queue.reset_mock()
        agent._enforce_workflow_chain(task_b, _make_response())
        chain_b = queue.push.call_args[0][0]

        assert chain_a.id != chain_b.id
        assert "planning-s6-auth-login" in chain_a.id
        assert "planning-s6-auth-signup" in chain_b.id

    def test_chain_title_does_not_accumulate_prefixes(self, agent, queue):
        """Chaining a [chain]-prefixed task produces a single [chain] prefix, not nested."""
        task = _make_task()
        task.title = "[chain] [chain] Implement feature X"

        agent._enforce_workflow_chain(task, _make_response())

        chain_task = queue.push.call_args[0][0]
        assert chain_task.title == "[chain] Implement feature X"
        assert not chain_task.title.startswith("[chain] [chain]")


class TestExecutorCompletedCheck:
    """Tests for executor dedup checking the correct completed directory."""

    def test_executor_completed_check_uses_correct_directory(self, queue, tmp_path):
        """Executor checks queue.completed_dir, not queue_dir/completed."""
        from agent_framework.workflow.executor import WorkflowExecutor

        # Set up completed_dir on the mock queue (simulates real FileQueue)
        completed_dir = tmp_path / "comms" / "completed"
        completed_dir.mkdir(parents=True)
        queue.completed_dir = completed_dir

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # Place a completed chain task in the correct completed_dir
        chain_id = "chain-task-abc-engineer"
        (completed_dir / f"{chain_id}.json").write_text("{}")

        assert executor._is_chain_task_already_queued("engineer", "task-abc") is True

    def test_executor_completed_check_ignores_wrong_directory(self, queue, tmp_path):
        """Completed tasks in queue_dir/completed (wrong path) are NOT found."""
        from agent_framework.workflow.executor import WorkflowExecutor

        # Set completed_dir to the correct location (empty)
        completed_dir = tmp_path / "comms" / "completed"
        completed_dir.mkdir(parents=True)
        queue.completed_dir = completed_dir

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # Place a file in the WRONG directory (queue_dir/completed)
        wrong_dir = queue.queue_dir / "completed"
        wrong_dir.mkdir(parents=True)
        chain_id = "chain-task-abc-engineer"
        (wrong_dir / f"{chain_id}.json").write_text("{}")

        # Should NOT find it — the bug was checking the wrong path
        assert executor._is_chain_task_already_queued("engineer", "task-abc") is False
