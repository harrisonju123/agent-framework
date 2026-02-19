"""Tests for per-step instructions threading through the workflow chain.

Covers the data flow: YAML → WorkflowStepDefinition → WorkflowStep → chain task context → prompt injection.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import AgentConfig
from agent_framework.core.config import WorkflowDefinition, WorkflowStepDefinition
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.workflow.dag import WorkflowStep
from agent_framework.workflow.executor import WorkflowExecutor


# -- Helpers --

def _make_task(task_id="task-abc123", **ctx_overrides):
    context = {"workflow": "default", **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


def _make_prompt_builder(tmp_path, agent_id="architect"):
    config = AgentConfig(
        id=agent_id,
        name=agent_id.title(),
        queue=agent_id,
        prompt="You are an architect. Plan, route, analyze, break down work.",
    )
    ctx = PromptContext(
        config=config,
        workspace=tmp_path,
        mcp_enabled=False,
        optimization_config={},
    )
    return PromptBuilder(ctx)


# -- dag.py --

class TestWorkflowStepInstructions:
    def test_instructions_default_none(self):
        """WorkflowStep.instructions defaults to None."""
        step = WorkflowStep(id="plan", agent="architect")
        assert step.instructions is None

    def test_instructions_stores_value(self):
        """WorkflowStep preserves instructions when set."""
        step = WorkflowStep(
            id="create_pr",
            agent="architect",
            instructions="Create a PR. Do NOT plan.",
        )
        assert step.instructions == "Create a PR. Do NOT plan."


# -- config.py --

class TestWorkflowStepDefinitionInstructions:
    def test_instructions_default_none(self):
        """WorkflowStepDefinition.instructions defaults to None."""
        step_def = WorkflowStepDefinition(agent="architect")
        assert step_def.instructions is None

    def test_instructions_parses_from_dict(self):
        """WorkflowStepDefinition parses instructions from raw dict (YAML-like)."""
        step_def = WorkflowStepDefinition(
            agent="architect",
            instructions="Only create a PR.",
        )
        assert step_def.instructions == "Only create a PR."

    def test_to_dag_propagates_instructions(self):
        """to_dag() threads instructions from WorkflowStepDefinition to WorkflowStep."""
        workflow = WorkflowDefinition(
            description="test",
            start_step="create_pr",
            steps={
                "create_pr": WorkflowStepDefinition(
                    agent="architect",
                    instructions="Create a PR only.",
                ),
            },
        )
        dag = workflow.to_dag("test")
        assert dag.steps["create_pr"].instructions == "Create a PR only."

    def test_to_dag_none_instructions(self):
        """to_dag() passes None when instructions not set."""
        workflow = WorkflowDefinition(
            description="test",
            start_step="plan",
            steps={
                "plan": WorkflowStepDefinition(agent="architect"),
            },
        )
        dag = workflow.to_dag("test")
        assert dag.steps["plan"].instructions is None


# -- executor.py --

class TestBuildChainTaskInstructions:
    @pytest.fixture
    def executor(self, tmp_path):
        queue = MagicMock()
        queue.queue_dir = tmp_path / "queues"
        queue.queue_dir.mkdir()
        queue.completed_dir = tmp_path / "completed"
        queue.completed_dir.mkdir()
        return WorkflowExecutor(queue, queue.queue_dir)

    def test_stores_instructions_in_context(self, executor):
        """_build_chain_task stores target step instructions in context."""
        task = _make_task()
        target = WorkflowStep(
            id="create_pr",
            agent="architect",
            instructions="Only create a PR.",
        )

        chain = executor._build_chain_task(task, target, "engineer")

        assert chain.context["_step_instructions"] == "Only create a PR."

    def test_clears_stale_instructions(self, executor):
        """_build_chain_task clears _step_instructions when target has none."""
        task = _make_task(_step_instructions="Old instructions from previous step")
        target = WorkflowStep(id="implement", agent="engineer")

        chain = executor._build_chain_task(task, target, "architect")

        assert "_step_instructions" not in chain.context

    def test_replaces_previous_instructions(self, executor):
        """Instructions from previous step are replaced, not accumulated."""
        task = _make_task(_step_instructions="Plan the work")
        target = WorkflowStep(
            id="create_pr",
            agent="architect",
            instructions="Create PR only.",
        )

        chain = executor._build_chain_task(task, target, "engineer")

        assert chain.context["_step_instructions"] == "Create PR only."


# -- prompt_builder.py --

class TestStepInstructionsInPrompt:
    def test_step_instructions_in_legacy_prompt(self, tmp_path):
        """Step instructions appear before YOUR RESPONSIBILITIES in legacy prompt."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task(
            _step_instructions="Create a PR. Do NOT plan.",
            workflow_step="create_pr",
        )

        prompt = builder._build_prompt_legacy(task)

        assert "STEP INSTRUCTIONS" in prompt
        assert "CURRENT STEP: create_pr" in prompt
        assert "Create a PR. Do NOT plan." in prompt
        # Step instructions should appear before the agent prompt section
        step_idx = prompt.index("STEP INSTRUCTIONS")
        resp_idx = prompt.index("YOUR RESPONSIBILITIES:")
        assert step_idx < resp_idx

    def test_step_instructions_in_optimized_prompt(self, tmp_path):
        """Step instructions appear before agent prompt in optimized prompt."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task(
            _step_instructions="Create a PR. Do NOT plan.",
            workflow_step="create_pr",
        )

        prompt = builder._build_prompt_optimized(task)

        assert "STEP INSTRUCTIONS" in prompt
        assert "CURRENT STEP: create_pr" in prompt
        assert "Create a PR. Do NOT plan." in prompt
        # Agent prompt should appear after step instructions
        step_idx = prompt.index("STEP INSTRUCTIONS")
        agent_idx = prompt.index("You are an architect. Plan, route")
        assert step_idx < agent_idx

    def test_no_instructions_no_section(self, tmp_path):
        """Without _step_instructions, no STEP INSTRUCTIONS section appears."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task(workflow_step="implement")

        legacy = builder._build_prompt_legacy(task)
        optimized = builder._build_prompt_optimized(task)

        assert "STEP INSTRUCTIONS" not in legacy
        assert "STEP INSTRUCTIONS" not in optimized

    def test_framing_text_present(self, tmp_path):
        """Step instructions include framing that deprioritizes agent prompt."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task(
            _step_instructions="Do X only.",
            workflow_step="create_pr",
        )

        prompt = builder._build_prompt_legacy(task)

        assert "MUST follow the step instructions" in prompt
        assert "Ignore any conflicting guidance" in prompt

    def test_full_build_pipeline_includes_step_instructions(self, tmp_path):
        """Integration: build() includes step instructions in the final prompt."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task(
            _step_instructions="Create a PR only.",
            workflow_step="create_pr",
            github_repo="owner/repo",
        )

        prompt = builder.build(task)

        assert "STEP INSTRUCTIONS" in prompt
        assert "Create a PR only." in prompt

    def test_standalone_task_no_step_instructions(self, tmp_path):
        """Standalone tasks (no workflow) produce no step instructions section."""
        builder = _make_prompt_builder(tmp_path)
        task = _make_task()

        prompt = builder.build(task)

        assert "STEP INSTRUCTIONS" not in prompt


# -- Round-trip integration --

class TestInstructionsRoundTrip:
    def test_round_trip_through_workflow_definition(self):
        """Instructions survive YAML dict → WorkflowStepDefinition → WorkflowStep → DAG."""
        workflow = WorkflowDefinition(
            description="test",
            start_step="create_pr",
            steps={
                "create_pr": WorkflowStepDefinition(
                    agent="architect",
                    instructions="Only create a PR. Do NOT plan.",
                ),
            },
        )
        dag = workflow.to_dag("test")
        step = dag.steps["create_pr"]

        assert step.instructions == "Only create a PR. Do NOT plan."
        assert step.agent == "architect"


# -- Plan step instructions from YAML --

class TestPlanStepHasInstructions:
    """Verify plan step defines JSON output instructions in workflow definitions."""

    def test_default_workflow_plan_step_has_instructions(self):
        """Plan step in default workflow defines JSON output instructions."""
        defn = WorkflowDefinition(
            description="test",
            start_step="plan",
            steps={
                "plan": WorkflowStepDefinition(
                    agent="architect",
                    instructions="Output your plan as a ```json code block.",
                ),
                "implement": WorkflowStepDefinition(agent="engineer"),
            },
        )
        dag = defn.to_dag("test")
        assert dag.steps["plan"].instructions is not None
        assert "json" in dag.steps["plan"].instructions.lower()

    def test_plan_step_instructions_thread_to_chain_task(self):
        """Plan step instructions survive through _build_chain_task."""
        executor = WorkflowExecutor(MagicMock(), MagicMock())
        task = _make_task()
        target = WorkflowStep(
            id="plan",
            agent="architect",
            instructions="Output your plan as a ```json code block.",
        )

        chain = executor._build_chain_task(task, target, "engineer")

        assert "json" in chain.context["_step_instructions"].lower()
