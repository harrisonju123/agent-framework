"""Tests for workflow chain enforcement."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
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

    # Initialize review cycle manager (required by delegation shims)
    from agent_framework.core.review_cycle import ReviewCycleManager
    a._review_cycle = ReviewCycleManager(
        config=config,
        queue=queue,
        logger=a.logger,
        agent_definition=None,
        session_logger=a._session_logger,
        activity_manager=MagicMock(),
    )

    # Initialize GitOperationsManager
    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=a.config,
        workspace=a.workspace,
        queue=a.queue,
        logger=a.logger,
        session_logger=a._session_logger if hasattr(a, '_session_logger') else None,
        workflows_config=a._workflows_config,
    )

    # Create prompt builder with mock context
    prompt_ctx = PromptContext(
        config=config,
        workspace=tmp_path,
        mcp_enabled=False,
        optimization_config={},
    )
    prompt_builder = PromptBuilder(prompt_ctx)
    a._prompt_builder = prompt_builder

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
        # Stable format: chain-{root_task_id}-{step_id}-d{depth}
        assert chain_task.id == f"chain-{task.id}-qa-d1"
        assert chain_task.assigned_to == "qa"
        assert chain_task.context["source_task_id"] == task.id
        assert chain_task.context["chain_step"] is True
        assert chain_task.context["workflow_step"] == "qa"
        assert chain_task.context["_root_task_id"] == task.id
        assert chain_task.context["_global_cycle_count"] == 1

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

        # Pre-create the chain task file with stable format
        chain_id = f"chain-{task.id}-qa-d1"
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

        # Mock executor's logger to verify error logging
        agent._workflow_executor.logger = MagicMock()

        agent._enforce_workflow_chain(task, response)

        # Error is caught and logged by the executor (called from router.enforce_chain)
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

        # Initialize GitOperationsManager
        from agent_framework.core.git_operations import GitOperationsManager
        a._git_ops = GitOperationsManager(
            config=a.config,
            workspace=a.workspace,
            queue=a.queue,
            logger=a.logger,
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

    def test_last_agent_queues_pr_creation(self, pr_agent, queue):
        """QA (last agent) completes → PR_REQUEST queued to architect."""
        task = _make_task(workflow="pr_workflow", implementation_branch="feature/test")
        response = _make_response()

        pr_agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert pr_task.type == TaskType.PR_REQUEST
        assert pr_task.context["pr_creation_step"] is True
        assert pr_task.title.startswith("[pr]")
        # PR creation still uses the old deterministic format (not DAG chain)
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
        agent._git_ops._active_worktree = worktree_dir
        agent._git_ops.worktree_manager = MagicMock()
        agent._git_ops.worktree_manager.has_unpushed_commits.return_value = True

        from unittest.mock import patch

        # Mock git rev-parse to return a feature branch name
        mock_rev_parse = MagicMock(returncode=0, stdout="agent/engineer/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0, stdout="", stderr="")

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_run:
            mock_run.side_effect = [mock_rev_parse, mock_push]
            agent._git_ops.push_and_create_pr_if_needed(task)

        # Branch should be stored for downstream agents
        assert task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"
        # Should NOT have called gh pr create (only git rev-parse + git push = 2 calls)
        assert mock_run.call_count == 2
        assert "pr_url" not in task.context

    def test_terminal_step_creates_pr(self, agent, tmp_path):
        """QA (terminal) creates a PR normally."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        agent._git_ops.config = agent.config  # Update git_ops config too
        task = _make_task(workflow="default", github_repo="org/repo")

        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        agent._git_ops._active_worktree = worktree_dir
        agent._git_ops.worktree_manager = MagicMock()
        agent._git_ops.worktree_manager.has_unpushed_commits.return_value = True

        from unittest.mock import patch

        mock_rev_parse = MagicMock(returncode=0, stdout="agent/qa/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0)
        mock_pr_create = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/10\n")

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git, \
             patch("agent_framework.utils.subprocess_utils.run_command") as mock_cmd:
            mock_git.side_effect = [mock_rev_parse, mock_push]
            mock_cmd.return_value = mock_pr_create
            agent._git_ops.push_and_create_pr_if_needed(task)

        assert task.context["pr_url"] == "https://github.com/org/repo/pull/10"

    def test_pr_creation_task_uses_implementation_branch(self, agent, tmp_path):
        """PR creation task with implementation_branch calls _create_pr_from_branch."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        agent._git_ops._active_worktree = None
        agent._git_ops.worktree_manager = None
        agent._git_ops.multi_repo_manager = MagicMock()
        agent._git_ops.multi_repo_manager.ensure_repo.return_value = tmp_path

        from unittest.mock import patch

        mock_pr_create = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/11\n")

        with patch("agent_framework.utils.subprocess_utils.run_command", return_value=mock_pr_create):
            agent._git_ops.push_and_create_pr_if_needed(task)

        assert task.context["pr_url"] == "https://github.com/org/repo/pull/11"

    def test_pr_creation_task_without_impl_branch_falls_through(self, agent):
        """PR creation task without implementation_branch uses normal flow."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
        )
        agent._git_ops._active_worktree = None
        agent._git_ops.worktree_manager = None

        # No worktree + no implementation_branch → early return (no PR)
        agent._git_ops.push_and_create_pr_if_needed(task)
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
        agent._git_ops.multi_repo_manager = agent.multi_repo_manager
        agent.multi_repo_manager.ensure_repo.return_value = repo_path

        result = agent._git_ops.get_working_directory(task)

        assert result == repo_path
        agent.multi_repo_manager.ensure_repo.assert_called_once_with("org/repo")

    def test_normal_task_still_creates_worktree(self, agent, tmp_path):
        """Non-PR-creation tasks still go through the normal worktree flow."""
        task = _make_task(workflow="default", github_repo="org/repo")

        repo_path = tmp_path / "repos" / "org" / "repo"
        repo_path.mkdir(parents=True)
        worktree_path = tmp_path / "worktrees" / "agent-engineer"
        worktree_path.mkdir(parents=True)

        agent._git_ops.multi_repo_manager = MagicMock()
        agent._git_ops.multi_repo_manager.ensure_repo.return_value = repo_path
        agent._git_ops.worktree_manager = MagicMock()
        agent._git_ops.worktree_manager.find_worktree_by_branch.return_value = None
        agent._git_ops.worktree_manager.create_worktree.return_value = worktree_path

        result = agent._git_ops.get_working_directory(task)

        assert result == worktree_path
        agent._git_ops.worktree_manager.create_worktree.assert_called_once()


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

        result = agent._prompt_builder._inject_preview_mode(original_prompt, task)

        assert result.endswith(original_prompt)
        assert "PREVIEW MODE" in result
        assert "Do NOT use Write, Edit, or NotebookEdit" in result
        assert result.index("PREVIEW MODE") < result.index(original_prompt)

    def test_inject_preview_mode_includes_required_sections(self, agent):
        """Preview prompt includes all required output sections."""
        task = _make_task()
        result = agent._prompt_builder._inject_preview_mode("original", task)

        assert "### Files to Modify" in result
        assert "### New Files to Create" in result
        assert "### Implementation Approach" in result
        assert "### Risks and Edge Cases" in result
        assert "### Estimated Total Change Size" in result


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
        # Root task IDs are embedded in the chain ID
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
        chain_id = "chain-task-abc-engineer-d1"
        (completed_dir / f"{chain_id}.json").write_text("{}")

        # Use explicit chain_id param (the new stable format)
        assert executor._is_chain_task_already_queued("engineer", "task-abc", chain_id=chain_id) is True

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
        chain_id = "chain-task-abc-engineer-d1"
        (wrong_dir / f"{chain_id}.json").write_text("{}")

        # Should NOT find it — the bug was checking the wrong path
        assert executor._is_chain_task_already_queued("engineer", "task-abc", chain_id=chain_id) is False


# -- Bounce loop prevention --

class TestBounceLoopPrevention:
    """Tests for root task identity, global cycle cap, and chain ID stability."""

    def test_root_task_id_propagated_through_chain(self, agent, queue):
        """_root_task_id is stamped on first hop and preserved in subsequent hops."""
        task = _make_task(workflow="default")
        agent._enforce_workflow_chain(task, _make_response())

        chain_task = queue.push.call_args[0][0]
        assert chain_task.context["_root_task_id"] == task.id

        # Simulate second hop: QA → (would route further if there were a next step)
        # Root ID should carry forward, not re-stamp with chain task ID
        assert chain_task.context["_root_task_id"] == task.id
        assert chain_task.context["_chain_depth"] == 1
        assert chain_task.context["_global_cycle_count"] == 1

    def test_chain_ids_are_bounded_no_nesting(self, agent, queue):
        """Chain IDs use root_task_id, not task.id, so they never nest chain-chain-chain-..."""
        task = _make_task(workflow="default")
        agent._enforce_workflow_chain(task, _make_response())

        chain_task = queue.push.call_args[0][0]
        # New format: chain-{root_id}-{step_id}-d{depth}
        assert chain_task.id == f"chain-{task.id}-qa-d1"
        assert chain_task.id.count("chain-") == 1, "chain ID should not nest 'chain-' prefixes"

    def test_global_cycle_count_increments_across_hops(self, agent, queue):
        """Each chain hop increments _global_cycle_count."""
        task = _make_task(workflow="default", _global_cycle_count=5)
        agent._enforce_workflow_chain(task, _make_response())

        chain_task = queue.push.call_args[0][0]
        assert chain_task.context["_global_cycle_count"] == 6

    def test_global_cycle_cap_halts_workflow(self, queue, tmp_path):
        """When _global_cycle_count reaches MAX_GLOBAL_CYCLES, no chain task is created."""
        from agent_framework.workflow.executor import WorkflowExecutor, MAX_GLOBAL_CYCLES

        executor = WorkflowExecutor(queue, queue.queue_dir)
        executor.logger = MagicMock()

        task = _make_task(
            workflow="default",
            _global_cycle_count=MAX_GLOBAL_CYCLES,
            _chain_depth=2,
        )

        from agent_framework.workflow.dag import WorkflowStep
        target_step = WorkflowStep(id="qa", agent="qa")

        executor._route_to_step(task, target_step, MagicMock(), "engineer", None)

        queue.push.assert_not_called()
        executor.logger.warning.assert_called_once()
        assert "Global cycle count" in executor.logger.warning.call_args[0][0]

    def test_escalation_propagates_root_task_id(self, agent, queue):
        """_escalate_review_to_architect carries _root_task_id from the source task."""
        task = _make_task(
            workflow="default",
            _root_task_id="original-root-123",
            _global_cycle_count=4,
            _chain_depth=3,
        )
        outcome = SimpleNamespace(
            approved=False,
            findings_summary="Tests still failing",
        )

        agent._escalate_review_to_architect(task, outcome, cycle_count=3)

        queue.push.assert_called_once()
        escalation = queue.push.call_args[0][0]
        assert escalation.context["_root_task_id"] == "original-root-123"
        assert escalation.context["_global_cycle_count"] == 4
        assert escalation.context["_chain_depth"] == 3
        # Escalation ID uses root_task_id, not task.id
        assert escalation.id == "review-escalation-original-root-123"

    def test_escalation_stamps_root_when_missing(self, agent, queue):
        """If _root_task_id is missing, escalation stamps task.id as root."""
        task = _make_task(workflow="default")
        outcome = SimpleNamespace(
            approved=False,
            findings_summary="Tests still failing",
        )

        agent._escalate_review_to_architect(task, outcome, cycle_count=3)

        escalation = queue.push.call_args[0][0]
        assert escalation.context["_root_task_id"] == task.id

    def test_second_hop_chain_id_uses_root_not_chain_id(self, queue, tmp_path):
        """Second chain hop uses root_task_id in chain ID, not the intermediate chain task ID."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # Simulate a first-hop chain task
        first_hop = _make_task(
            task_id="chain-original-task-qa-d1",
            workflow="default",
            _root_task_id="original-task",
            _chain_depth=1,
            _global_cycle_count=1,
        )

        target_step = WorkflowStep(id="engineer", agent="engineer")
        chain_task = executor._build_chain_task(first_hop, target_step, "qa")

        # Chain ID should reference the original root, not the intermediate chain task
        assert chain_task.id == "chain-original-task-engineer-d2"
        assert chain_task.context["_root_task_id"] == "original-task"
        assert chain_task.context["_chain_depth"] == 2
        assert chain_task.context["_global_cycle_count"] == 2


# -- Fix 2: MAX_DAG_REVIEW_CYCLES lowered to 2 --

class TestDAGReviewCycleCap:
    """Verify QA->engineer fix cycles cap at MAX_DAG_REVIEW_CYCLES=2."""

    def test_dag_review_cycles_cap_is_two(self):
        from agent_framework.workflow.executor import MAX_DAG_REVIEW_CYCLES
        assert MAX_DAG_REVIEW_CYCLES == 2

    def test_qa_routes_to_pr_after_two_fix_cycles(self, queue, tmp_path):
        """After 2 fix cycles, QA->engineer route redirects to create_pr step."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            _dag_review_cycles=2,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step, "create_pr": pr_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        # Redirected to PR step instead of back to engineer
        assert target_queue == "architect"

    def test_qa_allows_first_two_fix_cycles(self, queue, tmp_path):
        """First 2 fix cycles route normally to engineer."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            _dag_review_cycles=1,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "engineer"
        assert chain_task.context["_dag_review_cycles"] == 2


# -- Fix 3: QA findings injected in chain task description --

class TestUpstreamSummaryInChainTask:
    """Verify QA findings get embedded in fix-cycle chain task descriptions."""

    def test_qa_to_engineer_injects_upstream_summary(self, queue, tmp_path):
        """When QA routes to engineer, upstream_summary is prepended to description."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            upstream_summary="Tests failing in module X",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.description.startswith("## QA FINDINGS TO ADDRESS")
        assert "Tests failing in module X" in chain_task.description
        assert "## ORIGINAL TASK" in chain_task.description
        assert "Build the thing." in chain_task.description

    def test_non_qa_route_preserves_original_description(self, agent, queue):
        """Engineer->QA route keeps the original description unchanged."""
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.description == "Build the thing."

    def test_qa_to_engineer_without_summary_preserves_description(self, queue, tmp_path):
        """QA->engineer without upstream_summary keeps original description."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.description == "Build the thing."


# -- Fix 4: PR creation skipped for planning-only tasks --

class TestPRCreationPlanningSkip:
    """Verify planning-only tasks (no implementation_branch) skip PR creation."""

    @pytest.fixture
    def pr_agent(self, queue, tmp_path):
        config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.workspace = tmp_path
        a._workflows_config = {
            "default": DEFAULT_WORKFLOW,
            "pr_workflow": PR_WORKFLOW,
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

        from agent_framework.core.git_operations import GitOperationsManager
        a._git_ops = GitOperationsManager(
            config=a.config,
            workspace=a.workspace,
            queue=a.queue,
            logger=a.logger,
            session_logger=a._session_logger,
            workflows_config=a._workflows_config,
        )
        return a

    def test_no_branch_no_pr_skips_creation(self, pr_agent, queue):
        """Task without implementation_branch or pr_number skips PR creation."""
        task = _make_task(workflow="pr_workflow")
        pr_agent._queue_pr_creation_if_needed(task, PR_WORKFLOW)
        queue.push.assert_not_called()

    def test_with_implementation_branch_creates_pr(self, pr_agent, queue):
        """Task with implementation_branch proceeds to queue PR creation."""
        task = _make_task(
            workflow="pr_workflow",
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        pr_agent._queue_pr_creation_if_needed(task, PR_WORKFLOW)
        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        assert pr_task.type == TaskType.PR_REQUEST

    def test_with_pr_number_creates_pr(self, pr_agent, queue):
        """Task with pr_number proceeds to queue PR creation."""
        task = _make_task(workflow="pr_workflow", pr_number=42)
        pr_agent._queue_pr_creation_if_needed(task, PR_WORKFLOW)
        queue.push.assert_called_once()


# -- Fix 5: Title-based dedup --

class TestTitleBasedDedup:
    """Verify title-based dedup catches same work queued under different IDs."""

    def test_title_dedup_blocks_duplicate(self, queue, tmp_path):
        """Existing task with same title (after stripping prefixes) blocks new task."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        existing_task = {"title": "[chain] Implement feature X", "id": "impl-1"}
        (engineer_dir / "impl-1.json").write_text(json.dumps(existing_task))

        result = executor._is_chain_task_already_queued(
            "engineer", "other-source",
            chain_id="chain-other-engineer-d1",
            title="[chain] Implement feature X",
        )
        assert result is True

    def test_title_dedup_allows_different_titles(self, queue, tmp_path):
        """Different titles are not blocked by dedup."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        existing_task = {"title": "[chain] Implement feature Y", "id": "impl-1"}
        (engineer_dir / "impl-1.json").write_text(json.dumps(existing_task))

        result = executor._is_chain_task_already_queued(
            "engineer", "other-source",
            chain_id="chain-other-engineer-d1",
            title="[chain] Implement feature X",
        )
        assert result is False

    def test_title_dedup_strips_chain_prefixes(self, queue, tmp_path):
        """Prefix stripping normalizes '[chain] [chain] X' to match '[chain] X'."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        existing_task = {"title": "Implement feature X", "id": "impl-plan-1"}
        (engineer_dir / "impl-plan-1.json").write_text(json.dumps(existing_task))

        result = executor._is_chain_task_already_queued(
            "engineer", "other-source",
            chain_id="chain-other-engineer-d1",
            title="[chain] Implement feature X",
        )
        assert result is True

    def test_title_dedup_case_insensitive(self, queue, tmp_path):
        """Title comparison is case-insensitive."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        existing_task = {"title": "[chain] IMPLEMENT FEATURE X", "id": "impl-1"}
        (engineer_dir / "impl-1.json").write_text(json.dumps(existing_task))

        result = executor._is_chain_task_already_queued(
            "engineer", "other-source",
            chain_id="chain-other-engineer-d1",
            title="[chain] implement feature x",
        )
        assert result is True

    def test_no_title_skips_dedup(self, queue, tmp_path):
        """When title is None, only ID-based check is performed."""
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        result = executor._is_chain_task_already_queued(
            "engineer", "task-1",
            chain_id="chain-task-1-engineer-d1",
        )
        assert result is False
