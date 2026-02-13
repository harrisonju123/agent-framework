"""Tests for workflow chain enforcement."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.config import WorkflowDefinition
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
        created_at=datetime.utcnow(),
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

ANALYSIS_WORKFLOW = WorkflowDefinition(
    description="Analysis only",
    agents=["architect"],
)


@pytest.fixture
def queue(tmp_path):
    """FileQueue mock with real queue_dir for file existence checks."""
    q = MagicMock()
    q.queue_dir = tmp_path / "queues"
    q.queue_dir.mkdir()
    return q


@pytest.fixture
def agent(queue):
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a._workflows_config = {"default": DEFAULT_WORKFLOW, "analysis": ANALYSIS_WORKFLOW}
    a._team_mode_enabled = False
    a.logger = MagicMock()
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
        assert chain_task.id == f"chain-{task.id[:12]}-qa"
        assert chain_task.assigned_to == "qa"
        assert chain_task.context["source_task_id"] == task.id
        assert chain_task.context["chain_step"] is True

    def test_skips_when_pr_created(self, agent, queue):
        """If a PR was created, chain enforcement is skipped."""
        task = _make_task(workflow="default")
        task.context["pr_url"] = "https://github.com/org/repo/pull/42"
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_skips_when_team_mode_handled(self, agent, queue):
        """When team mode handles the workflow, chain is skipped."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default")
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_fires_when_team_override_false(self, agent, queue):
        """When team_override=False, team mode is skipped so chain must fire."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default", team_override=False)
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_called_once()

    def test_skips_when_team_override_true(self, agent, queue):
        """team_override=True forces teams on."""
        agent._team_mode_enabled = True
        task = _make_task(workflow="default", team_override=True)
        response = _make_response()

        agent._enforce_workflow_chain(task, response)

        queue.push.assert_not_called()

    def test_skips_duplicate_chain_task(self, agent, queue):
        """If the chain task file already exists, don't queue again."""
        task = _make_task(workflow="default")
        response = _make_response()

        # Pre-create the chain task file
        chain_id = f"chain-{task.id[:12]}-qa"
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
        """Push failures are logged but don't raise."""
        task = _make_task(workflow="default")
        response = _make_response()
        queue.push.side_effect = OSError("disk full")

        agent._enforce_workflow_chain(task, response)

        agent.logger.error.assert_called_once()


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
