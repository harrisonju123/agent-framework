"""Tests for workflow.step_utils.is_at_terminal_workflow_step."""

from unittest.mock import MagicMock

from agent_framework.workflow.step_utils import is_at_terminal_workflow_step
from agent_framework.workflow.dag import (
    WorkflowDAG,
    WorkflowStep,
    WorkflowEdge,
)


def _make_task(context=None):
    task = MagicMock()
    task.context = context or {}
    return task


def _make_workflow_def(dag):
    """Create a mock WorkflowDefinition whose to_dag() returns the given DAG."""
    wf = MagicMock()
    wf.to_dag.return_value = dag
    return wf


def _linear_dag():
    """plan → implement → create_pr (3-step linear)."""
    return WorkflowDAG(
        name="default",
        description="",
        steps={
            "plan": WorkflowStep(
                id="plan", agent="architect",
                next=[WorkflowEdge(target="implement")],
            ),
            "implement": WorkflowStep(
                id="implement", agent="engineer",
                next=[WorkflowEdge(target="create_pr")],
            ),
            "create_pr": WorkflowStep(
                id="create_pr", agent="qa",
                next=[],
            ),
        },
        start_step="plan",
    )


class TestIsAtTerminalWorkflowStep:
    def test_standalone_task_returns_true(self):
        """No workflow → terminal (backward compat for standalone agents)."""
        task = _make_task()
        assert is_at_terminal_workflow_step(task, {}, "engineer") is True

    def test_unknown_workflow_returns_true(self):
        task = _make_task({"workflow": "nonexistent"})
        assert is_at_terminal_workflow_step(task, {}, "engineer") is True

    def test_terminal_step_by_workflow_step_context(self):
        dag = _linear_dag()
        wf = _make_workflow_def(dag)
        task = _make_task({"workflow": "default", "workflow_step": "create_pr"})
        assert is_at_terminal_workflow_step(task, {"default": wf}, "qa") is True

    def test_non_terminal_step_by_workflow_step_context(self):
        dag = _linear_dag()
        wf = _make_workflow_def(dag)
        task = _make_task({"workflow": "default", "workflow_step": "implement"})
        assert is_at_terminal_workflow_step(task, {"default": wf}, "engineer") is False

    def test_fallback_to_agent_base_id(self):
        """When workflow_step is absent, looks up step by agent_base_id."""
        dag = _linear_dag()
        wf = _make_workflow_def(dag)
        task = _make_task({"workflow": "default"})
        # engineer is at "implement" which is non-terminal
        assert is_at_terminal_workflow_step(task, {"default": wf}, "engineer") is False
        # qa is at "create_pr" which is terminal
        assert is_at_terminal_workflow_step(task, {"default": wf}, "qa") is True

    def test_agent_not_in_dag_returns_true(self):
        dag = _linear_dag()
        wf = _make_workflow_def(dag)
        task = _make_task({"workflow": "default"})
        assert is_at_terminal_workflow_step(task, {"default": wf}, "unknown_agent") is True

    def test_dag_construction_failure_returns_true(self):
        wf = MagicMock()
        wf.to_dag.side_effect = ValueError("bad config")
        task = _make_task({"workflow": "default"})
        assert is_at_terminal_workflow_step(task, {"default": wf}, "engineer") is True

    def test_empty_workflows_config_returns_true(self):
        task = _make_task({"workflow": "default"})
        assert is_at_terminal_workflow_step(task, {}, "engineer") is True

    def test_none_workflows_config_returns_true(self):
        """Prompt builder passes {} for None configs; verify graceful handling."""
        task = _make_task({"workflow": "default"})
        assert is_at_terminal_workflow_step(task, {}, "engineer") is True

    def test_multi_step_same_agent_uses_workflow_step(self):
        """When an agent appears at multiple steps, workflow_step disambiguates."""
        dag = WorkflowDAG(
            name="default",
            description="",
            steps={
                "plan": WorkflowStep(
                    id="plan", agent="architect",
                    next=[WorkflowEdge(target="implement")],
                ),
                "implement": WorkflowStep(
                    id="implement", agent="engineer",
                    next=[WorkflowEdge(target="code_review")],
                ),
                "code_review": WorkflowStep(
                    id="code_review", agent="architect",
                    next=[],
                ),
            },
            start_step="plan",
        )
        wf = _make_workflow_def(dag)
        # With explicit workflow_step, returns correct answer regardless of agent appearing twice
        task_at_plan = _make_task({"workflow": "default", "workflow_step": "plan"})
        assert is_at_terminal_workflow_step(task_at_plan, {"default": wf}, "architect") is False

        task_at_review = _make_task({"workflow": "default", "workflow_step": "code_review"})
        assert is_at_terminal_workflow_step(task_at_review, {"default": wf}, "architect") is True
