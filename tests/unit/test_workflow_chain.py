"""Tests for workflow chain enforcement."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig, _strip_tool_call_markers
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.config import WorkflowDefinition
from tests.unit.workflow_fixtures import PREVIEW_WORKFLOW
from agent_framework.core.routing import RoutingSignal, WORKFLOW_COMPLETE
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.utils.type_helpers import get_type_str


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

    # Budget manager and optimization config (required by cost accumulation)
    from types import MappingProxyType
    a._optimization_config = MappingProxyType({"enable_effort_budget_ceilings": False})
    a._budget = MagicMock()
    a._budget.estimate_cost.return_value = 0.0

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
        agent._workflow_router.config = agent.config
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
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="analysis")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_chain_task_type_engineer(self, agent, queue):
        """Engineer tasks get IMPLEMENTATION type."""
        # architect -> engineer chain
        agent.config = AgentConfig(id="architect", name="A", queue="architect", prompt="p")
        agent._workflow_router.config = agent.config
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
        """Tasks without a workflow key get assigned 'default'."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        agent._normalize_workflow(task)
        assert task.context["workflow"] == "default"

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
        # DEFAULT_WORKFLOW has no pr_creator, so queue_pr_creation_if_needed
        # bails early even though __complete__ now triggers the PR path.
        task = _make_task(workflow="default")
        response = _make_response()
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_not_called()

    def test_workflow_complete_signal_stops_chain_even_with_pr(self, agent, queue):
        """WORKFLOW_COMPLETE signal terminates chain regardless of pr_url."""
        # pr_url is in the response — executor now saves it to task.context,
        # so queue_pr_creation_if_needed detects it and returns early.
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
        pr_agent._workflow_router.config = pr_agent.config
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

    def test_workflow_complete_signal_queues_pr_creation(self, pr_agent, queue):
        """__complete__ signal at non-terminal step + implementation_branch → PR queued."""
        task = _make_task(
            workflow="pr_workflow",
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )
        response = _make_response()
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        pr_agent._enforce_workflow_chain(task, response, routing_signal=signal)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert pr_task.type == TaskType.PR_REQUEST
        assert pr_task.context["pr_creation_step"] is True

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
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="default")
        assert agent._is_at_terminal_workflow_step(task) is True

    def test_first_agent_is_not_terminal(self, agent):
        """Architect (first in architect→engineer→qa) is not terminal."""
        agent.config = AgentConfig(id="architect", name="A", queue="a", prompt="p")
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="default")
        assert agent._is_at_terminal_workflow_step(task) is False

    def test_single_agent_workflow_is_terminal(self, agent):
        """Single-agent workflow (analysis: [architect]) — architect is terminal."""
        agent.config = AgentConfig(id="architect", name="A", queue="a", prompt="p")
        agent._workflow_router.config = agent.config
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
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="default", workflow_step="engineer")
        assert agent._is_at_terminal_workflow_step(task) is False


# -- Intermediate step PR suppression --

class TestIntermediateStepPRSuppression:
    """Tests for _push_and_create_pr_if_needed skipping PRs on intermediate steps."""

    def test_intermediate_step_stores_branch_skips_pr(self, agent, tmp_path):
        """Engineer (intermediate) pushes branch but doesn't create a PR."""
        task = _make_task(workflow="default", github_repo="org/repo")

        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        agent._git_ops._active_worktree = worktree_dir

        from unittest.mock import patch

        mock_rev_parse = MagicMock(returncode=0, stdout="agent/engineer/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0, stdout="", stderr="")

        with patch.object(agent._git_ops, '_has_unpushed_commits', return_value=True), \
             patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_run:
            mock_run.side_effect = [mock_rev_parse, mock_push]
            agent._git_ops.push_and_create_pr_if_needed(task)

        assert task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"
        assert mock_run.call_count == 2
        assert "pr_url" not in task.context

    def test_terminal_step_creates_pr(self, agent, tmp_path):
        """QA (terminal) creates a PR normally."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        agent._git_ops.config = agent.config
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="default", github_repo="org/repo")

        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        agent._git_ops._active_worktree = worktree_dir

        from unittest.mock import patch

        mock_rev_parse = MagicMock(returncode=0, stdout="agent/qa/PROJ-123-abc12345\n")
        mock_push = MagicMock(returncode=0)
        mock_pr_create = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/10\n")

        with patch.object(agent._git_ops, '_has_unpushed_commits', return_value=True), \
             patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git, \
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

    def test_pr_creation_with_impl_branch_uses_workspace(self, agent, tmp_path):
        """PR creation task with implementation_branch uses workspace directly."""
        task = _make_task(
            workflow="default",
            github_repo="org/repo",
            pr_creation_step=True,
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )

        result = agent._git_ops.get_working_directory(task)

        assert result == agent._git_ops.workspace

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

    @pytest.fixture
    def preview_agent(self, agent):
        """Extend the module-level agent with the preview workflow.

        The router holds the same dict reference as agent._workflows_config,
        so adding a key here is immediately visible to both.
        """
        agent._workflows_config["preview"] = PREVIEW_WORKFLOW
        return agent

    def test_preview_routes_to_preview_review(self, preview_agent, queue):
        """Engineer completing the preview step routes to architect for preview_review."""
        task = _make_task(
            workflow="preview",
            workflow_step="preview",
            _chain_depth=1,
            _root_task_id="root-preview-1",
            _global_cycle_count=1,
        )
        task.type = TaskType.PREVIEW
        response = _make_response()

        preview_agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert chain_task.assigned_to == "architect"
        assert chain_task.context.get("workflow_step") == "preview_review"

    def test_preview_does_not_route_to_qa(self, preview_agent, queue):
        """PREVIEW tasks route to architect for review, not directly to QA."""
        task = _make_task(
            workflow="preview",
            workflow_step="preview",
            _chain_depth=1,
            _root_task_id="root-preview-2",
            _global_cycle_count=1,
        )
        task.type = TaskType.PREVIEW
        response = _make_response()

        preview_agent._enforce_workflow_chain(task, response)

        target_queue = queue.push.call_args[0][1]
        assert target_queue != "qa"

    def test_non_preview_still_routes_normally(self, agent, queue):
        """Regular IMPLEMENTATION tasks still follow the default workflow chain."""
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"

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


class TestExecutorWorkflowCompleteSignal:
    """Tests for __complete__ routing signal PR info capture in executor."""

    def test_complete_signal_saves_pr_info_to_task_context(self, queue):
        """__complete__ signal with PR in response → pr_url/pr_number saved to task.context."""
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)
        dag = DEFAULT_WORKFLOW.to_dag("default")

        task = _make_task(workflow="default", workflow_step="plan")
        response = _make_response(pr_url="https://github.com/org/repo/pull/77")
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        routed = executor.execute_step(
            workflow=dag,
            task=task,
            response=response,
            current_agent_id="architect",
            routing_signal=signal,
        )

        assert routed is False
        assert task.context["pr_url"] == "https://github.com/org/repo/pull/77"
        assert task.context["pr_number"] == 77

    def test_complete_signal_without_pr_does_not_add_pr_context(self, queue):
        """__complete__ signal without PR → no pr_url in task.context."""
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)
        dag = DEFAULT_WORKFLOW.to_dag("default")

        task = _make_task(workflow="default", workflow_step="plan")
        response = _make_response()
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        routed = executor.execute_step(
            workflow=dag,
            task=task,
            response=response,
            current_agent_id="architect",
            routing_signal=signal,
        )

        assert routed is False
        assert "pr_url" not in task.context


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

    def test_qa_routes_to_pr_after_max_fix_cycles(self, queue, tmp_path):
        """At cycle limit, QA->engineer route redirects to create_pr step."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
            implementation_branch="feature/xyz",
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

    def test_qa_allows_first_fix_cycle(self, queue, tmp_path):
        """First fix cycle routes normally to engineer."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=0,
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
        assert chain_task.context["_dag_review_cycles"] == 1

    def test_review_cycle_emits_event_on_increment(self, queue, tmp_path):
        """First fix cycle emits review_cycle_check with count_before=0, count_after=1."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        session_logger = MagicMock()
        executor = WorkflowExecutor(queue, queue.queue_dir, session_logger=session_logger)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=0,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        session_logger.log.assert_called_once()
        call_kwargs = session_logger.log.call_args
        assert call_kwargs[0][0] == "review_cycle_check"
        assert call_kwargs[1]["count_before"] == 0
        assert call_kwargs[1]["count_after"] == 1
        assert call_kwargs[1]["enforced"] is False
        assert call_kwargs[1]["phase_reset"] is False

    def test_review_cycle_emits_event_on_enforcement(self, queue, tmp_path):
        """At cap, emits review_cycle_check with enforced=True."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        session_logger = MagicMock()
        executor = WorkflowExecutor(queue, queue.queue_dir, session_logger=session_logger)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
            implementation_branch="feature/xyz",
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step, "create_pr": pr_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        session_logger.log.assert_called_once()
        call_kwargs = session_logger.log.call_args
        assert call_kwargs[0][0] == "review_cycle_check"
        assert call_kwargs[1]["count_before"] == 1
        assert call_kwargs[1]["count_after"] == 2
        assert call_kwargs[1]["enforced"] is True
        assert call_kwargs[1]["target_step"] == "create_pr"

    def test_no_event_without_logger(self, queue, tmp_path):
        """No session_logger configured — no crash, no event."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=0,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"engineer": engineer_step}

        # Should not raise
        executor._route_to_step(task, engineer_step, workflow, "qa", None)
        queue.push.assert_called_once()

    def test_review_cycle_emits_event_on_phase_reset(self, queue, tmp_path):
        """preview_review → implement emits phase_reset=True and resets counter."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        session_logger = MagicMock()
        executor = WorkflowExecutor(queue, queue.queue_dir, session_logger=session_logger)

        task = _make_task(
            workflow="default",
            workflow_step="preview_review",
            _dag_review_cycles=1,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        implement_step = WorkflowStep(id="implement", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"implement": implement_step}

        executor._route_to_step(task, implement_step, workflow, "architect", None)

        session_logger.log.assert_called_once()
        call_kwargs = session_logger.log.call_args
        assert call_kwargs[0][0] == "review_cycle_check"
        assert call_kwargs[1]["phase_reset"] is True
        assert call_kwargs[1]["count_before"] == 1
        assert call_kwargs[1]["count_after"] == 0

    def test_review_cycle_emits_halted_when_no_pr_step(self, queue, tmp_path):
        """Cap hit without create_pr step emits halted=True and stops the chain."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        session_logger = MagicMock()
        executor = WorkflowExecutor(queue, queue.queue_dir, session_logger=session_logger)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        workflow = MagicMock()
        # No create_pr step in workflow
        workflow.steps = {"engineer": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        # Chain halted — no task queued
        queue.push.assert_not_called()
        session_logger.log.assert_called_once()
        call_kwargs = session_logger.log.call_args
        assert call_kwargs[0][0] == "review_cycle_check"
        assert call_kwargs[1]["halted"] is True
        assert call_kwargs[1]["enforced"] is False


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
            workflow_step="qa_review",
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
            workflow_step="qa_review",
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


# -- Step-aware descriptions --

class TestStepAwareDescriptions:
    """Verify chain tasks get step-appropriate descriptions when user_goal is set."""

    @pytest.fixture
    def executor(self, queue):
        from agent_framework.workflow.executor import WorkflowExecutor
        return WorkflowExecutor(queue, queue.queue_dir)

    def _task_with_goal(self, step="plan", **extra):
        return _make_task(
            workflow="default",
            workflow_step=step,
            user_goal="Add authentication to the API",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
            **extra,
        )

    def test_implement_step_gets_directive(self, executor):
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(step="plan")
        step = WorkflowStep(id="implement", agent="engineer")

        chain = executor._build_chain_task(task, step, "architect")

        assert chain.description.startswith("## Implement the following changes")
        assert "Add authentication to the API" in chain.description

    def test_code_review_step_gets_directive(self, executor):
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(step="implement")
        step = WorkflowStep(id="code_review", agent="architect")

        chain = executor._build_chain_task(task, step, "engineer")

        assert chain.description.startswith("## Review the implementation")
        assert "Add authentication to the API" in chain.description

    def test_qa_review_step_gets_directive(self, executor):
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(step="code_review")
        step = WorkflowStep(id="qa_review", agent="qa")

        chain = executor._build_chain_task(task, step, "architect")

        assert chain.description.startswith("## Test and verify")
        assert "Add authentication to the API" in chain.description

    def test_create_pr_step_gets_directive(self, executor):
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(step="qa_review")
        step = WorkflowStep(id="create_pr", agent="architect")

        chain = executor._build_chain_task(task, step, "qa")

        assert chain.description.startswith("## Create a pull request")
        assert "Add authentication to the API" in chain.description

    def test_same_step_fallback_to_original_description(self, executor):
        """When target step == current step, directive guard skips rewrite."""
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(step="implement")
        step = WorkflowStep(id="implement", agent="engineer")

        chain = executor._build_chain_task(task, step, "engineer")

        assert chain.description == "Build the thing."

    def test_no_user_goal_auto_sets_from_description(self, executor):
        """Without user_goal, it's auto-set from task.description and used for step directives."""
        from agent_framework.workflow.dag import WorkflowStep
        task = _make_task(
            workflow="default",
            workflow_step="plan",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )
        step = WorkflowStep(id="implement", agent="engineer")

        chain = executor._build_chain_task(task, step, "architect")

        # user_goal auto-populated from description, triggering step directive
        assert "Implement the following changes" in chain.description
        assert "Build the thing." in chain.description
        assert chain.context["user_goal"] == "Build the thing."

    def test_review_to_engineer_uses_user_goal_in_original_section(self, executor):
        """QA→engineer description uses user_goal for ORIGINAL TASK section."""
        from agent_framework.workflow.dag import WorkflowStep
        task = self._task_with_goal(
            step="qa_review",
            upstream_summary="Tests failing in auth module",
            upstream_source_step="qa_review",
        )
        step = WorkflowStep(id="implement", agent="engineer")

        chain = executor._build_chain_task(
            task, step, "qa", is_review_to_engineer=True,
        )

        assert "## QA FINDINGS TO ADDRESS" in chain.description
        assert "Tests failing in auth module" in chain.description
        assert "## ORIGINAL TASK" in chain.description
        assert "Add authentication to the API" in chain.description
        # user_goal should be used, not the raw task description
        assert "Build the thing." not in chain.description


# -- Code review cycle cap and findings header --

class TestCodeReviewCycleCap:
    """Verify code_review→engineer fix cycles share the same cap as QA."""

    def test_code_review_routes_to_pr_after_max_fix_cycles(self, queue, tmp_path):
        """At cycle limit, architect at code_review→engineer redirects to create_pr."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="code_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
            implementation_branch="feature/xyz",
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step, "create_pr": pr_step}

        executor._route_to_step(task, engineer_step, workflow, "architect", None)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"

    def test_code_review_allows_first_fix_cycle(self, queue, tmp_path):
        """First fix cycle from code_review routes normally to engineer."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="code_review",
            _dag_review_cycles=0,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "architect", None)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "engineer"
        assert chain_task.context["_dag_review_cycles"] == 1

    def test_shared_counter_across_review_stages(self, queue, tmp_path):
        """Counter from code_review carries into qa_review cap."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # code_review used 1 cycle, now qa_review tries to use another
        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _dag_review_cycles=1,
            _chain_depth=5,
            _root_task_id="root-1",
            _global_cycle_count=5,
            implementation_branch="feature/xyz",
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step, "create_pr": pr_step}

        executor._route_to_step(task, engineer_step, workflow, "qa", None)

        queue.push.assert_called_once()
        target_queue = queue.push.call_args[0][1]
        # Counter is 1 + 1 = 2 >= MAX_DAG_REVIEW_CYCLES(2), redirect to PR
        assert target_queue == "architect"

    def test_code_review_injects_upstream_summary_with_header(self, queue, tmp_path):
        """code_review→engineer uses 'CODE REVIEW FINDINGS' header."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="code_review",
            upstream_summary="Missing error handling in auth module",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "architect", None)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.description.startswith("## CODE REVIEW FINDINGS TO ADDRESS")
        assert "Missing error handling in auth module" in chain_task.description
        assert "## ORIGINAL TASK" in chain_task.description

    def test_plan_to_engineer_does_not_increment(self, queue, tmp_path):
        """plan→implement doesn't touch the review cycle counter."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            _chain_depth=0,
            _root_task_id="root-1",
            _global_cycle_count=0,
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step}

        executor._route_to_step(task, engineer_step, workflow, "architect", None)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        assert chain_task.context["_dag_review_cycles"] == 0


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


# -- Cross-type dedup --

class TestCrossTypeDedup:
    """Verify chain tasks are blocked when subtasks already exist for the same root."""

    def test_blocks_chain_when_subtasks_exist(self, queue, tmp_path):
        """Subtask in queue with same root_task_id blocks chain task creation."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # Place a subtask in the engineer queue with a matching root
        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        subtask_data = {
            "id": "root-1-sub-0",
            "parent_task_id": "root-1",
            "title": "Subtask 0",
            "context": {"_root_task_id": "root-1"},
        }
        (engineer_dir / "root-1-sub-0.json").write_text(json.dumps(subtask_data))

        result = executor._is_chain_task_already_queued(
            "qa", "some-source",
            chain_id="chain-root-1-qa-d1",
            root_task_id="root-1",
        )
        assert result is True

    def test_allows_chain_when_no_subtasks(self, queue, tmp_path):
        """No subtasks in queue → chain task creation is allowed."""
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        result = executor._is_chain_task_already_queued(
            "qa", "some-source",
            chain_id="chain-root-1-qa-d1",
            root_task_id="root-1",
        )
        assert result is False

    def test_ignores_subtasks_with_different_root(self, queue, tmp_path):
        """Subtasks for a different root don't block the chain task."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        subtask_data = {
            "id": "other-root-sub-0",
            "parent_task_id": "other-root",
            "title": "Subtask 0",
            "context": {"_root_task_id": "other-root"},
        }
        (engineer_dir / "other-root-sub-0.json").write_text(json.dumps(subtask_data))

        result = executor._is_chain_task_already_queued(
            "qa", "some-source",
            chain_id="chain-root-1-qa-d1",
            root_task_id="root-1",
        )
        assert result is False

    def test_no_root_task_id_skips_cross_type_check(self, queue, tmp_path):
        """When root_task_id is None, cross-type dedup is skipped entirely."""
        import json
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        subtask_data = {
            "id": "root-1-sub-0",
            "parent_task_id": "root-1",
            "title": "Subtask 0",
            "context": {"_root_task_id": "root-1"},
        }
        (engineer_dir / "root-1-sub-0.json").write_text(json.dumps(subtask_data))

        result = executor._is_chain_task_already_queued(
            "qa", "some-source",
            chain_id="chain-root-1-qa-d1",
        )
        assert result is False


# -- Verdict storage and clearing --

class TestVerdictStorageAndClearing:
    """Verify verdict is stored before workflow routing and cleared in chain tasks."""

    def test_verdict_stored_before_workflow_routing(self, agent, queue):
        """_set_structured_verdict stores verdict in task.context for review agents."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default")
        response = _make_response("All checks pass, approved")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "approved"

    def test_needs_fix_verdict_stored(self, agent, queue):
        """Verdict 'needs_fix' stored when review finds issues."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default")
        response = _make_response("CRITICAL: auth module has SQL injection vulnerability")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "needs_fix"

    def test_engineer_does_not_store_verdict(self, agent, queue):
        """Engineer agent skips verdict storage to avoid false positives from stray keywords."""
        task = _make_task(workflow="default")
        response = _make_response("Implemented feature. CRITICAL log line was added.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert "verdict" not in task.context

    def test_verdict_cleared_in_chain_task(self, queue, tmp_path):
        """Chain task context does NOT inherit verdict from parent task."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            verdict="approved",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        chain_task = executor._build_chain_task(task, engineer_step, "qa")

        assert "verdict" not in chain_task.context

    def test_no_verdict_stored_for_non_workflow_tasks(self, agent, queue):
        """Tasks without a workflow don't get a verdict stored."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        response = _make_response("approved")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert "verdict" not in task.context

    def test_preview_review_approval_stores_preview_approved(self, agent, queue):
        """Architect approving at preview_review sets verdict='preview_approved', not 'approved'.

        This is the critical distinction that fires the preview_approved DAG edge
        instead of the generic approved edge, routing to implement rather than qa_review.
        """
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="preview", workflow_step="preview_review")
        response = _make_response("VERDICT: APPROVE")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "preview_approved"

    def test_preview_review_ambiguous_does_not_set_verdict(self, agent, queue):
        """Architect at preview_review with ambiguous output → no verdict stored.

        Review steps with ambiguous output should halt the chain rather than
        default to approval. The preview_review "always" fallback edge will
        still route to implement via the DAG, but no verdict is recorded.
        """
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="preview", workflow_step="preview_review")
        response = _make_response("The plan looks comprehensive and well-structured.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert "verdict" not in task.context

    def test_code_review_approval_still_stores_approved(self, agent, queue):
        """Architect at code_review (not preview_review) still gets verdict='approved'.

        Regression guard: the new preview_review branching must not affect code_review.
        """
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="default", workflow_step="code_review")
        response = _make_response("VERDICT: APPROVE")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "approved"

    def test_ambiguous_at_code_review_does_not_set_verdict(self, agent, queue):
        """Ambiguous output at code_review → no verdict, chain will halt."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="default", workflow_step="code_review")
        response = _make_response("I reviewed the changes. Some observations noted.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert "verdict" not in task.context

    def test_ambiguous_at_qa_review_does_not_set_verdict(self, agent, queue):
        """Ambiguous output at qa_review → no verdict, chain will halt."""
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")
        task = _make_task(workflow="default", workflow_step="qa_review")
        response = _make_response("Ran the test suite. Results are inconclusive.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert "verdict" not in task.context

    def test_ambiguous_at_plan_step_still_sets_approved(self, agent, queue):
        """Ambiguous output at plan step → verdict='approved' (no regression)."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="default", workflow_step="plan")
        response = _make_response("Here is the implementation plan for the feature.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "approved"


# -- No-changes verdict detection --

class TestNoChangesVerdict:
    """Verify no_changes verdict is set at plan step when work is already done."""

    def test_no_changes_verdict_set_at_plan_step(self, agent, queue):
        """Architect at plan step with [NO_CHANGES_NEEDED] marker → verdict='no_changes'."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="default", workflow_step="plan")
        response = _make_response("[NO_CHANGES_NEEDED]\nThe feature already exists in production.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "no_changes"

    def test_no_changes_verdict_set_for_original_planning_task(self, agent, queue):
        """Original planning task (type=PLANNING, no workflow_step) → verdict='no_changes'."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        # Original planning task: type=PLANNING but no workflow_step in context
        task = Task(
            id="task-abc123def456",
            type=TaskType.PLANNING,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan feature X",
            description="Evaluate whether feature X needs work.",
            context={"workflow": "default"},
        )
        response = _make_response("[NO_CHANGES_NEEDED]\nThe feature already exists in production.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") == "no_changes"

    def test_no_changes_verdict_not_set_at_code_review(self, agent, queue):
        """Architect at code_review step → no 'no_changes' verdict even with marker."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        task = _make_task(workflow="default", workflow_step="code_review")
        response = _make_response("[NO_CHANGES_NEEDED]\nNothing to change, already implemented.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        # Should get "approved" verdict from review outcome, not "no_changes"
        assert task.context.get("verdict") != "no_changes"

    def test_no_changes_verdict_not_set_for_engineer(self, agent, queue):
        """Engineer at plan step (edge case) → no no_changes verdict."""
        task = _make_task(workflow="default", workflow_step="plan")
        response = _make_response("[NO_CHANGES_NEEDED]\nFeature already exists.")

        agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
        agent._set_structured_verdict(task, response)

        assert task.context.get("verdict") != "no_changes"

    def test_no_changes_skips_enforce_chain(self, agent, queue):
        """When verdict is no_changes at plan step, workflow terminates — no chain enforcement."""
        agent.config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        agent._workflow_router.config = agent.config
        task = _make_task(workflow="default", workflow_step="plan")
        # Verdict is now set by _set_structured_verdict before _run_post_completion_flow
        task.context["verdict"] = "no_changes"
        response = _make_response("[NO_CHANGES_NEEDED]\nNo engineering work needed.")

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._enforce_workflow_chain = MagicMock()
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        agent._run_post_completion_flow(task, response, None, 0)

        assert task.context.get("verdict") == "no_changes"
        agent._enforce_workflow_chain.assert_not_called()

    def test_missing_workflow_warning_on_root_task(self, agent, queue):
        """Root task without workflow in context emits a warning log."""
        task = _make_task()  # No workflow set
        task.context.pop("workflow", None)
        response = _make_response("Done.")

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        agent._run_post_completion_flow(task, response, None, 0)

        agent.logger.warning.assert_any_call(
            f"Task {task.id} has no workflow in context — "
            f"expected for CLI/web-created tasks"
        )

    def test_no_changes_marker_detected(self):
        """Responses starting with [NO_CHANGES_NEEDED] marker are detected."""
        positives = [
            "[NO_CHANGES_NEEDED]\nThe feature already exists in the codebase.",
            "[NO_CHANGES_NEEDED]\nAlready shipped as part of the v2 release.",
            "[NO_CHANGES_NEEDED] No engineering work needed.",
            "[NO_CHANGES_NEEDED]",
        ]

        for phrase in positives:
            assert Agent._is_no_changes_response(phrase), \
                f"Failed to detect no-changes marker in: {phrase}"

    def test_no_changes_without_marker_not_detected(self):
        """Responses WITHOUT the marker are NOT detected, even with 'already exists' text."""
        negatives = [
            "The feature already exists in the codebase.",
            "This has already been merged into main.",
            "No code changes needed for this task.",
            "Nothing to implement here.",
            "Already done in a previous sprint.",
            "No work required for this ticket.",
        ]

        for phrase in negatives:
            assert not Agent._is_no_changes_response(phrase), \
                f"False positive no-changes (no marker) on: {phrase}"

    def test_no_changes_marker_buried_in_plan_not_detected(self):
        """Marker buried past 200 chars in a plan body is NOT detected."""
        plan_text = "## Plan\n" + "x" * 200 + "\n[NO_CHANGES_NEEDED]\nThis is buried."
        assert not Agent._is_no_changes_response(plan_text), \
            "Marker after 200 chars should not trigger no_changes"

    def test_p0_scenario_already_exist_no_false_positive(self):
        """Exact P0 scenario: planner mentions 'already exist' while describing current state."""
        plan_text = (
            "## Plan: Enhance Observability Dashboard\n\n"
            "### Data Sources (all already exist)\n"
            "- Session logs in /var/log/agent/\n"
            "- Profile registry already exists in config/\n\n"
            "### Files to Modify\n"
            "1. server.py - add metrics endpoint\n"
        )
        assert not Agent._is_no_changes_response(plan_text), \
            "Plan with incidental 'already exist' must NOT trigger no_changes"

    def test_empty_and_none_content(self):
        """Empty string and edge cases return False."""
        assert not Agent._is_no_changes_response("")
        assert not Agent._is_no_changes_response("   ")

    def test_normal_plan_not_detected_as_no_changes(self):
        """Normal planning output should NOT trigger no_changes."""
        normal_plans = [
            "We need to implement the auth module with JWT tokens.",
            "The approach is to modify the existing handler.",
            "Plan: 1. Create new endpoint 2. Add tests",
            "The work is progressing well and we should continue.",
        ]

        for phrase in normal_plans:
            assert not Agent._is_no_changes_response(phrase), \
                f"False positive no-changes on: {phrase}"


# -- No-changes workflow termination --

class TestNoChangesRouting:
    """Verify plan step with no_changes terminates workflow (no create_pr routing)."""

    def test_no_changes_verdict_terminates_workflow(self, queue, tmp_path):
        """Plan step with no_changes has no DAG edge to follow — workflow ends."""
        from agent_framework.workflow.dag import (
            WorkflowStep, WorkflowEdge,
            EdgeCondition, EdgeConditionType,
        )

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="no_changes",
        )

        response = _make_response("[NO_CHANGES_NEEDED]\nFeature already exists.")

        # Current config: plan only has always → implement (no no_changes edge)
        edges = [
            WorkflowEdge(
                target="implement",
                condition=EdgeCondition(EdgeConditionType.ALWAYS),
            ),
        ]

        from agent_framework.workflow.conditions import ConditionRegistry
        # ALWAYS edge would match, but agent.py's skip_chain prevents
        # _enforce_workflow_chain from ever evaluating these edges.
        # This test just documents that the DAG no longer has a
        # no_changes → create_pr shortcut.
        matched_target = None
        for edge in edges:
            if ConditionRegistry.evaluate(edge.condition, task, response):
                matched_target = edge.target
                break
        # ALWAYS matches, but agent.py skip_chain guard prevents this
        # from firing — tested in test_no_changes_skips_enforce_chain
        assert matched_target == "implement"

    def test_plan_routes_to_implement_without_no_changes(self, queue, tmp_path):
        """Plan step without no_changes verdict falls through to implement."""
        from agent_framework.workflow.dag import (
            WorkflowEdge,
            EdgeCondition, EdgeConditionType,
        )

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="approved",
        )

        response = _make_response("Here is the implementation plan...")

        edges = [
            WorkflowEdge(
                target="implement",
                condition=EdgeCondition(EdgeConditionType.ALWAYS),
            ),
        ]

        from agent_framework.workflow.conditions import ConditionRegistry
        matched_target = None
        for edge in edges:
            if ConditionRegistry.evaluate(edge.condition, task, response):
                matched_target = edge.target
                break

        assert matched_target == "implement"


# -- Same-agent upstream context clearing --

class TestSameAgentUpstreamClearing:
    """Prevent self-referential upstream context when chain routes back to same agent."""

    def test_chain_to_same_step_clears_upstream_summary(self, queue):
        """Chain task targeting the same step that produced upstream clears it."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            upstream_summary="Previous code review analysis...",
            upstream_context_file="/tmp/ctx.md",
            upstream_source_agent="architect",
            upstream_source_step="code_review",
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        # Route back to code_review — same step that produced the upstream
        architect_step = WorkflowStep(id="code_review", agent="architect")
        chain_task = executor._build_chain_task(task, architect_step, "qa")

        assert "upstream_summary" not in chain_task.context
        assert "upstream_context_file" not in chain_task.context
        assert "upstream_source_agent" not in chain_task.context
        assert "upstream_source_step" not in chain_task.context

    def test_chain_to_same_agent_different_step_preserves_upstream(self, queue):
        """Chain from architect/plan to architect/code_review keeps upstream context."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            upstream_summary="Plan output: implement auth module",
            upstream_context_file="/tmp/plan-ctx.md",
            upstream_source_agent="architect",
            upstream_source_step="plan",
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        # Route to code_review — different step, same agent
        code_review_step = WorkflowStep(id="code_review", agent="architect")
        chain_task = executor._build_chain_task(task, code_review_step, "architect")

        assert chain_task.context["upstream_summary"] == "Plan output: implement auth module"
        assert chain_task.context["upstream_context_file"] == "/tmp/plan-ctx.md"
        assert chain_task.context["upstream_source_agent"] == "architect"
        assert chain_task.context["upstream_source_step"] == "plan"

    def test_chain_to_different_agent_preserves_upstream_summary(self, queue):
        """Chain task targeting a different agent keeps upstream context intact."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            upstream_summary="QA findings: tests failing",
            upstream_context_file="/tmp/qa-ctx.md",
            upstream_source_agent="qa",
            upstream_source_step="qa_review",
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )

        # Route to engineer — different from qa who produced the upstream
        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        chain_task = executor._build_chain_task(task, engineer_step, "qa")

        assert chain_task.context["upstream_summary"] == "QA findings: tests failing"
        assert chain_task.context["upstream_context_file"] == "/tmp/qa-ctx.md"
        assert chain_task.context["upstream_source_agent"] == "qa"

    def test_upstream_source_agent_and_step_stored_on_save(self, tmp_path):
        """_save_upstream_context stores both upstream_source_agent and upstream_source_step."""
        agent = MagicMock()
        agent.config = AgentConfig(
            id="architect", name="Architect", queue="architect", prompt="p",
        )
        agent.workspace = tmp_path
        agent.UPSTREAM_CONTEXT_MAX_CHARS = Agent.UPSTREAM_CONTEXT_MAX_CHARS
        agent.logger = MagicMock()

        task = _make_task(workflow_step="plan")
        response = _make_response("Analysis complete: all looks good.")

        Agent._save_upstream_context(agent, task, response)

        assert task.context["upstream_source_agent"] == "architect"
        assert task.context["upstream_source_step"] == "plan"


# -- Stale worktree_branch clearing --

class TestUpstreamInlineLimit:
    """Verify UPSTREAM_INLINE_MAX_CHARS is used for inline summary truncation."""

    def test_upstream_inline_uses_15kb_limit(self, tmp_path):
        """_save_upstream_context uses UPSTREAM_INLINE_MAX_CHARS (15KB) for inline summary."""
        agent = MagicMock()
        agent.config = AgentConfig(
            id="architect", name="Architect", queue="architect", prompt="p",
        )
        agent.workspace = tmp_path
        agent.UPSTREAM_CONTEXT_MAX_CHARS = Agent.UPSTREAM_CONTEXT_MAX_CHARS
        agent.UPSTREAM_INLINE_MAX_CHARS = Agent.UPSTREAM_INLINE_MAX_CHARS
        agent.logger = MagicMock()

        task = _make_task()
        # Content longer than 4KB but shorter than 12KB should be preserved fully
        content = "x" * 10000
        response = _make_response(content)

        Agent._save_upstream_context(agent, task, response)

        assert len(task.context["upstream_summary"]) == 10000

    def test_upstream_inline_truncates_at_15kb(self, tmp_path):
        """Content longer than 15KB is truncated at UPSTREAM_INLINE_MAX_CHARS."""
        agent = MagicMock()
        agent.config = AgentConfig(
            id="architect", name="Architect", queue="architect", prompt="p",
        )
        agent.workspace = tmp_path
        agent.UPSTREAM_CONTEXT_MAX_CHARS = Agent.UPSTREAM_CONTEXT_MAX_CHARS
        agent.UPSTREAM_INLINE_MAX_CHARS = Agent.UPSTREAM_INLINE_MAX_CHARS
        agent.logger = MagicMock()

        task = _make_task()
        content = "y" * 20000
        response = _make_response(content)

        Agent._save_upstream_context(agent, task, response)

        assert len(task.context["upstream_summary"]) == 15000


class TestPlanPropagation:
    """Verify _build_chain_task propagates task.plan through the workflow chain."""

    def test_chain_task_propagates_plan(self, queue):
        """_build_chain_task carries task.plan through to the chain task."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep
        from agent_framework.core.task import PlanDocument

        executor = WorkflowExecutor(queue, queue.queue_dir)

        plan = PlanDocument(
            objectives=["Add auth endpoint"],
            approach=["Create handler", "Add middleware", "Write tests"],
            files_to_modify=["src/auth.py", "src/middleware.py"],
            risks=["Token expiry edge case"],
            success_criteria=["All tests pass"],
        )

        task = _make_task(
            workflow="default",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )
        task.plan = plan

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        chain_task = executor._build_chain_task(task, engineer_step, "architect")

        assert chain_task.plan is not None
        assert chain_task.plan.objectives == ["Add auth endpoint"]
        assert len(chain_task.plan.approach) == 3
        assert "src/auth.py" in chain_task.plan.files_to_modify

    def test_chain_task_none_plan_stays_none(self, queue):
        """_build_chain_task with no plan produces chain task with plan=None."""
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
        chain_task = executor._build_chain_task(task, engineer_step, "architect")

        assert chain_task.plan is None


class TestWorktreeBranchPropagation:
    """worktree_branch propagates through the chain so all steps share one worktree."""

    def test_chain_task_propagates_worktree_branch(self, queue):
        """_build_chain_task preserves worktree_branch in context."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            worktree_branch="agent/architect/task-184b366e",
            implementation_branch="agent/engineer/PROJ-123-abc12345",
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        engineer_step = WorkflowStep(id="engineer", agent="engineer")
        chain_task = executor._build_chain_task(task, engineer_step, "architect")

        assert chain_task.context["worktree_branch"] == "agent/architect/task-184b366e"
        assert chain_task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"

    def test_pr_creation_task_clears_worktree_branch(self, queue, tmp_path):
        """queue_pr_creation_if_needed strips worktree_branch from PR task context."""
        config = AgentConfig(id="qa", name="QA", queue="qa", prompt="p")

        from agent_framework.core.workflow_router import WorkflowRouter
        from agent_framework.workflow.executor import WorkflowExecutor

        executor = WorkflowExecutor(queue, queue.queue_dir)
        router = WorkflowRouter(
            config=config,
            queue=queue,
            workspace=tmp_path,
            logger=MagicMock(),
            session_logger=MagicMock(),
            workflows_config={"pr_workflow": PR_WORKFLOW},
            workflow_executor=executor,
            agents_config=[
                SimpleNamespace(id="architect"),
                SimpleNamespace(id="engineer"),
                SimpleNamespace(id="qa"),
            ],
            multi_repo_manager=None,
        )

        task = _make_task(
            workflow="pr_workflow",
            worktree_branch="agent/qa/task-xyz",
            implementation_branch="agent/engineer/PROJ-123-abc12345",
        )

        router.queue_pr_creation_if_needed(task, PR_WORKFLOW)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        assert "worktree_branch" not in pr_task.context
        assert pr_task.context["implementation_branch"] == "agent/engineer/PROJ-123-abc12345"


# -- Push-after-chain ordering --

class TestPushAfterChainRouting:
    """Push runs after _enforce_workflow_chain so git side-effects happen after routing."""

    def test_push_runs_after_enforce_workflow_chain(self, agent, queue):
        """push_and_create_pr_if_needed runs after _enforce_workflow_chain."""
        task = _make_task(workflow="default")
        response = _make_response("Done.")

        call_order = []

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        def track_push(*args, **kwargs):
            call_order.append("push")

        def track_chain(*args, **kwargs):
            call_order.append("chain")

        agent._git_ops.push_and_create_pr_if_needed = track_push
        agent._enforce_workflow_chain = track_chain

        agent._run_post_completion_flow(task, response, None, 0)

        assert "push" in call_order
        assert "chain" in call_order
        assert call_order.index("chain") < call_order.index("push")


# -- Phantom parent_task_id guard --

class TestPhantomParentTaskIdGuard:
    """LLMs can fabricate parent_task_id when writing task JSON directly.
    The guard in _run_post_completion_flow must clear phantom references
    so the task flows through the normal workflow chain."""

    @pytest.fixture(autouse=True)
    def _bind_post_completion(self, agent):
        """Bind the real method and stub side-effects for all tests."""
        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()
        agent._enforce_workflow_chain = MagicMock()
        agent._git_ops.push_and_create_pr_if_needed = MagicMock()

    def test_phantom_parent_cleared_and_task_flows_through_chain(self, agent, queue):
        """Phantom parent_task_id gets cleared, task routes through normal workflow."""
        task = _make_task(workflow="default")
        task.parent_task_id = "planning-phantom-1739894400"
        response = _make_response("Done.")

        queue.find_task.return_value = None

        agent._run_post_completion_flow(task, response, None, 0)

        assert task.parent_task_id is None
        agent._enforce_workflow_chain.assert_called_once()

    def test_real_parent_preserves_subtask_behavior(self, agent, queue):
        """Valid parent_task_id still skips workflow chain (fan-in handles it)."""
        task = _make_task(workflow="default")
        task.parent_task_id = "real-parent-123"
        response = _make_response("Done.")

        real_parent = _make_task(task_id="real-parent-123")
        queue.find_task.return_value = real_parent

        agent._run_post_completion_flow(task, response, None, 0)

        assert task.parent_task_id == "real-parent-123"
        agent._enforce_workflow_chain.assert_not_called()

    def test_no_validation_when_parent_task_id_is_none(self, agent, queue):
        """Regular tasks (no parent) skip the phantom validation entirely."""
        task = _make_task(workflow="default")
        assert task.parent_task_id is None
        response = _make_response("Done.")

        agent._run_post_completion_flow(task, response, None, 0)

        queue.find_task.assert_not_called()
        agent._enforce_workflow_chain.assert_called_once()


class TestNoDiffGuard:
    """Tests for _has_diff_for_pr — prevents dispatching create_pr with zero changes."""

    def test_no_impl_branch_skips_create_pr(self, queue):
        """Layer 1: no implementation_branch and no pr_number → skip."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)
        task = _make_task(workflow="default")
        step = WorkflowStep(id="create_pr", agent="architect")

        assert executor._has_diff_for_pr(task, step) is False

    def test_impl_branch_set_proceeds(self, queue):
        """Layer 1 passes when implementation_branch is present (no workspace → skip Layer 2)."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)
        task = _make_task(workflow="default", implementation_branch="feature/xyz")
        step = WorkflowStep(id="create_pr", agent="architect")

        assert executor._has_diff_for_pr(task, step) is True

    def test_non_create_pr_always_proceeds(self, queue):
        """Guard only fires for create_pr — other steps pass through."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)
        task = _make_task(workflow="default")
        step = WorkflowStep(id="implement", agent="engineer")

        assert executor._has_diff_for_pr(task, step) is True

    def test_git_log_empty_skips_create_pr(self, queue, tmp_path):
        """Layer 2: branch exists but has no commits vs main → skip."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep
        from unittest.mock import patch
        import subprocess

        executor = WorkflowExecutor(queue, queue.queue_dir, workspace=tmp_path)
        task = _make_task(workflow="default", implementation_branch="feature/empty")
        step = WorkflowStep(id="create_pr", agent="architect")

        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("agent_framework.utils.subprocess_utils.run_git_command", return_value=mock_result):
            assert executor._has_diff_for_pr(task, step) is False

    def test_git_failure_blocks_create_pr(self, queue, tmp_path):
        """Layer 2: git command fails → return False (fail closed)."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep
        from unittest.mock import patch

        executor = WorkflowExecutor(queue, queue.queue_dir, workspace=tmp_path)
        task = _make_task(workflow="default", implementation_branch="feature/broken")
        step = WorkflowStep(id="create_pr", agent="architect")

        with patch("agent_framework.utils.subprocess_utils.run_git_command", side_effect=Exception("git error")):
            assert executor._has_diff_for_pr(task, step) is False


# -- Fix: Review cycle counter propagation across all chain hops --

class TestReviewCycleCounterPropagation:
    """Verify _dag_review_cycles survives every chain hop, not just review→engineer."""

    def test_counter_survives_implement_to_code_review_hop(self, queue, tmp_path):
        """Counter set on implement task propagates to code_review chain task."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="implement",
            _dag_review_cycles=1,
            _chain_depth=3,
            _root_task_id="root-1",
            _global_cycle_count=3,
        )

        code_review_step = WorkflowStep(id="code_review", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"code_review": code_review_step}

        executor._route_to_step(task, code_review_step, workflow, "engineer", None)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        assert chain_task.context["_dag_review_cycles"] == 1

    def test_full_cycle_cap_fires_on_second_review(self, queue, tmp_path):
        """Simulates d1(plan)→d2(implement)→d3(code_review)→d4(implement)→d5(code_review).

        Counter starts at 0, increments to 1 on d3→d4, then d4→d5 is a
        non-review hop so counter stays at 1. On d5→d6 (code_review→implement),
        counter increments to 2 >= MAX(2), redirects to create_pr.
        """
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        code_review_step = WorkflowStep(id="code_review", agent="architect")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {
            "implement": engineer_step,
            "code_review": code_review_step,
            "create_pr": pr_step,
        }

        # Hop 1: code_review→implement (first fix cycle, counter 0→1)
        task1 = _make_task(
            workflow="default",
            workflow_step="code_review",
            _dag_review_cycles=0,
            _chain_depth=2,
            _root_task_id="root-1",
            _global_cycle_count=2,
        )
        executor._route_to_step(task1, engineer_step, workflow, "architect", None)
        chain1 = queue.push.call_args[0][0]
        assert chain1.context["_dag_review_cycles"] == 1
        assert queue.push.call_args[0][1] == "engineer"
        queue.push.reset_mock()

        # Hop 2: implement→code_review (non-review hop, counter stays 1)
        task2 = _make_task(
            workflow="default",
            workflow_step="implement",
            _dag_review_cycles=1,
            _chain_depth=3,
            _root_task_id="root-1",
            _global_cycle_count=3,
        )
        executor._route_to_step(task2, code_review_step, workflow, "engineer", None)
        chain2 = queue.push.call_args[0][0]
        assert chain2.context["_dag_review_cycles"] == 1
        queue.push.reset_mock()

        # Hop 3: code_review→implement (second fix cycle, counter 1→2 >= MAX)
        task3 = _make_task(
            workflow="default",
            workflow_step="code_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
            implementation_branch="feature/xyz",
        )
        executor._route_to_step(task3, engineer_step, workflow, "architect", None)
        chain3 = queue.push.call_args[0][0]
        target = queue.push.call_args[0][1]
        # Cap fired — redirected to create_pr (architect)
        assert target == "architect"
        assert chain3.context["workflow_step"] == "create_pr"

    def test_counter_starts_at_zero_on_fresh_chain(self, queue, tmp_path):
        """First task in a chain without prior counter gets 0."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            _chain_depth=0,
            _root_task_id="root-1",
            _global_cycle_count=0,
        )
        # No _dag_review_cycles in context at all
        assert "_dag_review_cycles" not in task.context

        implement_step = WorkflowStep(id="implement", agent="engineer")
        workflow = MagicMock()
        workflow.steps = {"implement": implement_step}

        executor._route_to_step(task, implement_step, workflow, "architect", None)

        chain_task = queue.push.call_args[0][0]
        assert chain_task.context["_dag_review_cycles"] == 0

    def test_cap_redirect_bypasses_diff_check(self, queue, tmp_path):
        """Cap redirect to create_pr works even without implementation_branch."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        # No implementation_branch — would normally fail _has_diff_for_pr
        task = _make_task(
            workflow="default",
            workflow_step="code_review",
            _dag_review_cycles=1,
            _chain_depth=4,
            _root_task_id="root-1",
            _global_cycle_count=4,
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        pr_step = WorkflowStep(id="create_pr", agent="architect")
        workflow = MagicMock()
        workflow.steps = {"implement": engineer_step, "create_pr": pr_step}

        executor._route_to_step(task, engineer_step, workflow, "architect", None)

        # Without cap_redirect bypass, _has_diff_for_pr would block this
        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target = queue.push.call_args[0][1]
        assert target == "architect"
        assert chain_task.context["workflow_step"] == "create_pr"


# -- Worktree race condition fixes --

class TestPRCreationStepInChainBuilder:
    """Fix 1: _build_chain_task sets pr_creation_step for PR chain tasks."""

    def test_pr_request_step_sets_pr_creation_step(self, queue):
        """Chain task targeting a pr_request step gets pr_creation_step=True."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            workflow_step="qa_review",
            _chain_depth=3,
            _root_task_id="root-1",
            _global_cycle_count=3,
        )

        pr_step = WorkflowStep(
            id="create_pr", agent="architect", task_type_override="pr_request",
        )
        chain_task = executor._build_chain_task(task, pr_step, "qa")

        assert chain_task.context["pr_creation_step"] is True

    def test_non_pr_step_does_not_set_pr_creation_step(self, queue):
        """Chain task targeting a normal step does NOT get pr_creation_step."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            _chain_depth=0,
            _root_task_id="root-1",
            _global_cycle_count=0,
        )

        engineer_step = WorkflowStep(id="implement", agent="engineer")
        chain_task = executor._build_chain_task(task, engineer_step, "architect")

        assert "pr_creation_step" not in chain_task.context

    def test_pr_creation_step_stripped_from_non_pr_steps(self, queue):
        """pr_creation_step inherited from source context is popped for non-PR steps."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(
            workflow="default",
            pr_creation_step=True,
            _chain_depth=1,
            _root_task_id="root-1",
            _global_cycle_count=1,
        )

        qa_step = WorkflowStep(id="qa_review", agent="qa")
        chain_task = executor._build_chain_task(task, qa_step, "engineer")

        assert "pr_creation_step" not in chain_task.context


class TestCleanupTaskExecutionOrdering:
    """Fix 2 & 3: IDLE set after cleanup, periodic cleanup skipped for intermediate steps."""

    @pytest.fixture
    def cleanup_agent(self, queue, tmp_path):
        """Minimal agent with enough wiring for _cleanup_task_execution."""
        from agent_framework.core.activity import AgentStatus, AgentActivity, ActivityManager

        config = AgentConfig(id="engineer", name="Engineer", queue="engineer", prompt="p")
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.workspace = tmp_path
        a.logger = MagicMock()
        a._current_task_id = "task-1"
        a.worktree_manager = MagicMock()
        a._last_worktree_cleanup = 0

        # Track the order of calls to verify IDLE happens after cleanup
        call_order = []

        # Activity manager mock that records when IDLE is set
        activity_mgr = MagicMock(spec=ActivityManager)
        def record_update(activity):
            if activity.status == AgentStatus.IDLE:
                call_order.append("idle")
        activity_mgr.update_activity.side_effect = record_update
        a.activity_manager = activity_mgr

        # Git ops mock that records cleanup calls
        a._git_ops = MagicMock()
        def record_cleanup(*args, **kwargs):
            call_order.append("cleanup_worktree")
        a._git_ops.cleanup_worktree.side_effect = record_cleanup

        a._workflows_config = {"default": DEFAULT_WORKFLOW}

        # Store call_order on agent for assertions
        a._test_call_order = call_order
        return a

    def test_idle_set_after_worktree_cleanup(self, cleanup_agent, queue):
        """Agent stays WORKING during cleanup — IDLE is set after cleanup_worktree."""
        task = _make_task(workflow="default")
        task.status = TaskStatus.COMPLETED

        cleanup_agent._cleanup_task_execution(task, lock=None)

        order = cleanup_agent._test_call_order
        assert "cleanup_worktree" in order
        assert "idle" in order
        assert order.index("cleanup_worktree") < order.index("idle")

    def test_periodic_cleanup_not_called_during_task_execution(self, cleanup_agent):
        """Periodic worktree cleanup is disabled — never called from _cleanup_task_execution."""
        task = _make_task(workflow="default")
        task.status = TaskStatus.COMPLETED

        cleanup_agent._maybe_run_periodic_worktree_cleanup = MagicMock()
        cleanup_agent._cleanup_task_execution(task, lock=None)

        cleanup_agent._maybe_run_periodic_worktree_cleanup.assert_not_called()


class TestWorkingDirectoryValidation:
    """Fix 4: _get_validated_working_directory retries when path vanishes."""

    @pytest.fixture
    def validated_agent(self, queue, tmp_path):
        """Minimal agent for testing _get_validated_working_directory."""
        config = AgentConfig(id="engineer", name="Engineer", queue="engineer", prompt="p")
        a = Agent.__new__(Agent)
        a.config = config
        a.logger = MagicMock()
        a._session_logger = MagicMock()
        a._git_ops = MagicMock()
        a._get_validated_working_directory = Agent._get_validated_working_directory.__get__(a)
        return a

    def test_returns_existing_directory(self, validated_agent, tmp_path):
        """Happy path — directory exists on first call."""
        real_dir = tmp_path / "work"
        real_dir.mkdir()
        validated_agent._git_ops.get_working_directory.return_value = real_dir

        task = _make_task(workflow="default")
        result = validated_agent._get_validated_working_directory(task)

        assert result == real_dir
        assert validated_agent._git_ops.get_working_directory.call_count == 1

    def test_retries_when_working_dir_vanishes(self, validated_agent, tmp_path):
        """When working dir doesn't exist on first call, retries once."""
        vanished_dir = tmp_path / "vanished"
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        validated_agent._git_ops.get_working_directory.side_effect = [vanished_dir, real_dir]

        task = _make_task(workflow="default")
        result = validated_agent._get_validated_working_directory(task)

        assert result == real_dir
        assert validated_agent._git_ops.get_working_directory.call_count == 2
        # Upgraded to error-level with diagnostic context (branch, root_id)
        validated_agent.logger.error.assert_called_once()
        log_msg = validated_agent.logger.error.call_args[0][0]
        assert str(vanished_dir) in log_msg

    def test_raises_when_working_dir_vanishes_permanently(self, validated_agent, tmp_path):
        """When working dir doesn't exist after retry, raises RuntimeError."""
        vanished_dir = tmp_path / "vanished"
        validated_agent._git_ops.get_working_directory.return_value = vanished_dir

        task = _make_task(workflow="default")
        with pytest.raises(RuntimeError, match="does not exist after retry"):
            validated_agent._get_validated_working_directory(task)

        assert validated_agent._git_ops.get_working_directory.call_count == 2


# -- Workflow Summary Emission --

class TestEmitWorkflowSummary:
    """Tests for _emit_workflow_summary and its integration with _run_post_completion_flow."""

    def _make_chain_state_file(self, workspace, root_task_id, steps):
        """Write a chain state file to disk for load_chain_state to find."""
        from agent_framework.core.chain_state import ChainState, save_chain_state

        state = ChainState(
            root_task_id=root_task_id,
            user_goal="test goal",
            workflow="default",
            steps=steps,
        )
        save_chain_state(workspace, state)

    def test_emit_workflow_summary_at_terminal_step(self, agent):
        """At terminal workflow step, workflow_summary is logged to session and activity."""
        from agent_framework.core.chain_state import StepRecord

        task = _make_task(
            workflow="default",
            workflow_step="create_pr",
            chain_step=True,
        )

        self._make_chain_state_file(agent.workspace, task.root_id, [
            StepRecord(
                step_id="plan", agent_id="architect", task_id="t1",
                started_at="2026-02-19T10:00:00+00:00",
                completed_at="2026-02-19T10:01:00+00:00",
                duration_seconds=60.0,
                summary="planned",
            ),
            StepRecord(
                step_id="create_pr", agent_id="qa", task_id="t2",
                started_at="2026-02-19T10:02:00+00:00",
                completed_at="2026-02-19T10:02:30+00:00",
                duration_seconds=30.0,
                summary="PR created",
            ),
        ])

        agent._session_logging_enabled = True
        agent.activity_manager = MagicMock()
        agent._emit_workflow_summary(task)

        # Session logger should have been called with workflow_summary
        agent._session_logger.log.assert_called_once()
        call_args = agent._session_logger.log.call_args
        assert call_args[0][0] == "workflow_summary"
        logged_data = call_args[1]
        assert logged_data["root_task_id"] == task.root_id
        assert logged_data["outcome"] == "completed"
        assert len(logged_data["steps"]) == 2

        # Activity stream should have the event
        agent.activity_manager.append_event.assert_called_once()
        event = agent.activity_manager.append_event.call_args[0][0]
        assert event.type == "workflow_summary"

    @staticmethod
    def _prepare_for_post_completion(agent):
        """Set attributes needed by _run_post_completion_flow beyond base fixture."""
        agent._session_logging_enabled = True
        agent.activity_manager = MagicMock()
        agent._git_ops = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 0.0
        agent._memory_enabled = False
        agent._analyze_tool_patterns = MagicMock(return_value=0)
        agent._log_task_completion_metrics = MagicMock()

    def test_no_workflow_summary_when_chain_continues(self, agent):
        """When not at terminal step, no summary is emitted in _run_post_completion_flow."""
        task = _make_task(
            workflow="default",
            workflow_step="implement",
            chain_step=True,
        )

        self._prepare_for_post_completion(agent)

        response = _make_response()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        # Session logger should NOT have workflow_summary call
        for call in agent._session_logger.log.call_args_list:
            if call[0] and call[0][0] == "workflow_summary":
                pytest.fail("workflow_summary should not be emitted at non-terminal step")

    def test_emit_workflow_summary_on_no_changes(self, agent):
        """skip_chain=True (no_changes verdict) still emits workflow summary."""
        from agent_framework.core.chain_state import StepRecord

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            chain_step=True,
            verdict="no_changes",
        )

        self._make_chain_state_file(agent.workspace, task.root_id, [
            StepRecord(
                step_id="plan", agent_id="architect", task_id="t1",
                started_at="2026-02-19T10:00:00+00:00",
                completed_at="2026-02-19T10:00:15+00:00",
                duration_seconds=15.0,
                summary="no changes needed",
                verdict="no_changes",
            ),
        ])

        self._prepare_for_post_completion(agent)

        response = _make_response()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        # Should have workflow_summary in session logger calls
        summary_calls = [
            c for c in agent._session_logger.log.call_args_list
            if c[0] and c[0][0] == "workflow_summary"
        ]
        assert len(summary_calls) == 1
        assert summary_calls[0][1]["outcome"] == "no_changes"

    def test_emit_workflow_summary_no_chain_state_is_noop(self, agent):
        """When chain state file doesn't exist, emission is silently skipped."""
        task = _make_task(workflow="default", workflow_step="create_pr")

        agent._session_logging_enabled = True
        agent.activity_manager = MagicMock()
        agent._emit_workflow_summary(task)

        # Nothing should be logged
        agent._session_logger.log.assert_not_called()
        agent.activity_manager.append_event.assert_not_called()

    def test_emit_includes_pr_url_from_context(self, agent):
        """PR URL from task context is included in the summary."""
        from agent_framework.core.chain_state import StepRecord

        task = _make_task(
            workflow="default",
            workflow_step="create_pr",
            pr_url="https://github.com/org/repo/pull/99",
        )

        self._make_chain_state_file(agent.workspace, task.root_id, [
            StepRecord(
                step_id="create_pr", agent_id="qa", task_id="t1",
                completed_at="2026-02-19T10:00:00+00:00",
                summary="PR created",
            ),
        ])

        agent._session_logging_enabled = True
        agent.activity_manager = MagicMock()
        agent._emit_workflow_summary(task)

        logged_data = agent._session_logger.log.call_args[1]
        assert logged_data["pr_url"] == "https://github.com/org/repo/pull/99"

class TestStripToolCallMarkers:
    """Module-level _strip_tool_call_markers strips CLI-injected noise."""

    def test_removes_single_marker(self):
        content = "Analysis complete.\n[Tool Call: Read]\nThe code looks good."
        result = _strip_tool_call_markers(content)
        assert "[Tool Call:" not in result
        assert "Analysis complete." in result
        assert "The code looks good." in result

    def test_removes_multiple_markers(self):
        content = (
            "I'll analyze the codebase.\n"
            "[Tool Call: Read]\n"
            "Found the auth module.\n"
            "[Tool Call: Bash]\n"
            "Tests pass.\n"
            "[Tool Call: Grep]\n"
            "No issues found."
        )
        result = _strip_tool_call_markers(content)
        assert result.count("[Tool Call:") == 0
        assert "I'll analyze the codebase." in result
        assert "No issues found." in result

    def test_compresses_triple_newlines(self):
        content = "Line 1.\n[Tool Call: Read]\n\n\nLine 2."
        result = _strip_tool_call_markers(content)
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert _strip_tool_call_markers("") == ""

    def test_none_treated_as_empty(self):
        assert _strip_tool_call_markers(None) == ""

    def test_no_markers_passthrough(self):
        content = "Clean content with no markers."
        assert _strip_tool_call_markers(content) == content

    def test_marker_with_nested_brackets(self):
        content = "Before\n[Tool Call: Read (src/auth.py)]\nAfter"
        result = _strip_tool_call_markers(content)
        assert "[Tool Call:" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_leading_trailing_whitespace(self):
        content = "\n[Tool Call: Read]\nActual content.\n[Tool Call: Bash]\n"
        result = _strip_tool_call_markers(content)
        assert result == "Actual content."


class TestUpstreamContextFiltering:
    """_save_upstream_context strips tool call markers before storing."""

    @pytest.fixture
    def ctx_agent(self, queue, tmp_path):
        config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
        a = Agent.__new__(Agent)
        a.config = config
        a.workspace = tmp_path
        a.logger = MagicMock()
        a.UPSTREAM_CONTEXT_MAX_CHARS = 15000
        a.UPSTREAM_INLINE_MAX_CHARS = 15000
        a._save_upstream_context = Agent._save_upstream_context.__get__(a)
        return a

    def test_markers_stripped_from_file(self, ctx_agent, tmp_path):
        task = _make_task(workflow="default")
        response = _make_response(
            content="Plan:\n[Tool Call: Read]\nStep 1: Add auth.\n[Tool Call: Bash]\nStep 2: Test."
        )
        ctx_agent._save_upstream_context(task, response)

        saved = task.context["upstream_summary"]
        assert "[Tool Call:" not in saved
        assert "Step 1: Add auth." in saved
        assert "Step 2: Test." in saved

    def test_markers_stripped_from_inline_summary(self, ctx_agent, tmp_path):
        task = _make_task(workflow="default")
        response = _make_response(content="Good.\n[Tool Call: Read]\nDone.")
        ctx_agent._save_upstream_context(task, response)

        assert "[Tool Call:" not in task.context["upstream_summary"]


# -- Routing signal verdict override --

class TestRoutingSignalVerdictOverride:
    """Tests for verdict reconciliation when architect uses mark_workflow_complete
    MCP tool instead of the [NO_CHANGES_NEEDED] text marker."""

    @pytest.fixture
    def architect_agent(self, queue, tmp_path):
        """Architect agent with minimal setup for verdict override tests."""
        config = AgentConfig(id="architect", name="Architect", queue="architect", prompt="p")
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

        from agent_framework.workflow.executor import WorkflowExecutor
        a._workflow_executor = WorkflowExecutor(queue, queue.queue_dir)

        from agent_framework.core.review_cycle import ReviewCycleManager
        a._review_cycle = ReviewCycleManager(
            config=config,
            queue=queue,
            logger=a.logger,
            agent_definition=None,
            session_logger=a._session_logger,
            activity_manager=MagicMock(),
        )

        from agent_framework.core.git_operations import GitOperationsManager
        a._git_ops = GitOperationsManager(
            config=a.config,
            workspace=a.workspace,
            queue=a.queue,
            logger=a.logger,
            session_logger=a._session_logger,
            workflows_config=a._workflows_config,
        )

        prompt_ctx = PromptContext(
            config=config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        a._prompt_builder = PromptBuilder(prompt_ctx)

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

        from types import MappingProxyType
        a._optimization_config = MappingProxyType({"enable_effort_budget_ceilings": False})
        a._budget = MagicMock()
        a._budget.estimate_cost.return_value = 0.0

        return a

    def _apply_override(self, agent, task, routing_signal):
        """Simulate the verdict override block from _handle_successful_response."""
        if (routing_signal
                and routing_signal.target_agent == WORKFLOW_COMPLETE
                and agent.config.base_id == "architect"
                and task.context.get("workflow_step", get_type_str(task.type)) in ("plan", "planning")
                and task.context.get("verdict") != "no_changes"):
            prev_verdict = task.context.get("verdict")
            task.context["verdict"] = "no_changes"
            audit = task.context.get("verdict_audit")
            if isinstance(audit, dict):
                audit["method"] = "routing_signal_complete"
                audit["value"] = "no_changes"
            agent.logger.info(
                f"Verdict overridden: {prev_verdict!r} → 'no_changes' "
                f"(routing signal WORKFLOW_COMPLETE at plan step)"
            )
            agent._session_logger.log(
                "verdict_override", task_id=task.id,
                prev_verdict=prev_verdict, new_verdict="no_changes",
                method="routing_signal_complete",
                routing_signal_reason=routing_signal.reason,
            )
            agent._patch_chain_state_verdict(task)

    def test_override_fires_at_plan_step(self, architect_agent):
        """Verdict changes from 'approved' → 'no_changes', audit updated, session log emitted."""
        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="approved",
            verdict_audit={"method": "ambiguous_default", "value": "approved"},
        )
        signal = _make_signal(target=WORKFLOW_COMPLETE, reason="No code changes needed")

        self._apply_override(architect_agent, task, signal)

        assert task.context["verdict"] == "no_changes"
        assert task.context["verdict_audit"]["method"] == "routing_signal_complete"
        assert task.context["verdict_audit"]["value"] == "no_changes"
        architect_agent._session_logger.log.assert_called_once_with(
            "verdict_override",
            task_id=task.id,
            prev_verdict="approved",
            new_verdict="no_changes",
            method="routing_signal_complete",
            routing_signal_reason="No code changes needed",
        )

    def test_no_override_when_already_no_changes(self, architect_agent):
        """Idempotent — no double-override when text marker also present."""
        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="no_changes",
            verdict_audit={"method": "no_changes_marker", "value": "no_changes"},
        )
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        self._apply_override(architect_agent, task, signal)

        # Verdict unchanged, no session log emitted
        assert task.context["verdict"] == "no_changes"
        assert task.context["verdict_audit"]["method"] == "no_changes_marker"
        architect_agent._session_logger.log.assert_not_called()

    def test_ignored_at_non_plan_step(self, architect_agent):
        """No override at code_review/qa_review steps."""
        for step in ("code_review", "qa_review", "implement"):
            architect_agent._session_logger.reset_mock()
            task = _make_task(
                workflow="default",
                workflow_step=step,
                verdict="approved",
                verdict_audit={"method": "review_outcome", "value": "approved"},
            )
            signal = _make_signal(target=WORKFLOW_COMPLETE)

            self._apply_override(architect_agent, task, signal)

            assert task.context["verdict"] == "approved", f"Override should not fire at {step}"
            architect_agent._session_logger.log.assert_not_called()

    def test_ignored_for_non_architect(self, agent):
        """Engineer with WORKFLOW_COMPLETE doesn't get verdict override."""
        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="approved",
        )
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        # Reuse the same conditions as production code — engineer base_id
        # means the guard rejects the override
        self._apply_override(agent, task, signal)

        assert task.context["verdict"] == "approved"

    def test_chain_state_patched(self, architect_agent, tmp_path):
        """Chain state file's last step gets corrected verdict."""
        from agent_framework.core.chain_state import save_chain_state, load_chain_state, ChainState, StepRecord

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="approved",
            verdict_audit={"method": "ambiguous_default", "value": "approved"},
        )

        # Write chain state with "approved" verdict for the plan step
        state = ChainState(
            root_task_id=task.id,
            user_goal="Test goal",
            workflow="default",
            steps=[StepRecord(
                step_id="plan",
                agent_id="architect",
                task_id=task.id,
                completed_at="2026-02-20T10:00:00+00:00",
                summary="Plan completed",
                verdict="approved",
                verdict_audit={"method": "ambiguous_default", "value": "approved"},
            )],
        )
        save_chain_state(tmp_path, state)

        signal = _make_signal(target=WORKFLOW_COMPLETE, reason="No changes needed")
        self._apply_override(architect_agent, task, signal)

        # Verify chain state was patched
        patched = load_chain_state(tmp_path, task.id)
        assert patched.steps[-1].verdict == "no_changes"
        assert patched.steps[-1].verdict_audit["method"] == "routing_signal_complete"

    def test_chain_state_wrong_task_id(self, architect_agent, tmp_path):
        """Defensive: skips patch if last step belongs to different task."""
        from agent_framework.core.chain_state import save_chain_state, load_chain_state, ChainState, StepRecord

        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="approved",
            verdict_audit={"method": "ambiguous_default", "value": "approved"},
        )

        # Chain state has a step from a DIFFERENT task
        state = ChainState(
            root_task_id=task.id,
            user_goal="Test goal",
            workflow="default",
            steps=[StepRecord(
                step_id="plan",
                agent_id="architect",
                task_id="different-task-id",
                completed_at="2026-02-20T10:00:00+00:00",
                summary="Plan completed",
                verdict="approved",
            )],
        )
        save_chain_state(tmp_path, state)

        signal = _make_signal(target=WORKFLOW_COMPLETE)
        self._apply_override(architect_agent, task, signal)

        # Verdict on task context is overridden, but chain state is NOT patched
        assert task.context["verdict"] == "no_changes"
        patched = load_chain_state(tmp_path, task.id)
        assert patched.steps[-1].verdict == "approved"

    def test_skip_chain_fires_with_corrected_verdict(self, architect_agent, queue):
        """With corrected no_changes verdict, _enforce_workflow_chain is NOT called."""
        task = _make_task(
            workflow="default",
            workflow_step="plan",
            verdict="no_changes",
        )
        response = _make_response()
        signal = _make_signal(target=WORKFLOW_COMPLETE)

        # Stub downstream methods called after chain routing in _run_post_completion_flow
        architect_agent._extract_and_store_memories = MagicMock()
        architect_agent._analyze_tool_patterns = MagicMock(return_value=0)
        architect_agent._log_task_completion_metrics = MagicMock()
        architect_agent._emit_workflow_summary = MagicMock()

        architect_agent._run_post_completion_flow(task, response, signal, datetime.now(timezone.utc))

        # No chain task should be queued — the no_changes verdict terminates the workflow
        queue.push.assert_not_called()
