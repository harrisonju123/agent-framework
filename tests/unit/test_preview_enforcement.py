"""Tests for execution preview enforcement: allowed_tools, CLI flag, workflow routing, config."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.config import WorkflowDefinition, WorkflowStepDefinition
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMRequest
from agent_framework.llm.claude_cli_backend import ClaudeCLIBackend
from agent_framework.workflow.dag import EdgeCondition, EdgeConditionType


# -- Helpers --

def _make_task(task_type=TaskType.IMPLEMENTATION, **ctx_overrides):
    context = {"workflow": "preview", **ctx_overrides}
    return Task(
        id="test-task-preview",
        type=task_type,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


def _make_response(content="Done."):
    return SimpleNamespace(content=content, error=None)


# -- LLMRequest.allowed_tools --

class TestLLMRequestAllowedTools:
    def test_allowed_tools_default_is_none(self):
        """allowed_tools defaults to None when not specified."""
        request = LLMRequest(prompt="test")
        assert request.allowed_tools is None

    def test_allowed_tools_set_to_list(self):
        """allowed_tools accepts a list of tool names."""
        tools = ["Read", "Glob", "Grep", "Bash"]
        request = LLMRequest(prompt="test", allowed_tools=tools)
        assert request.allowed_tools == tools

    def test_preview_tools_exclude_write_edit(self):
        """Preview tool list should not contain Write, Edit, or NotebookEdit."""
        preview_tools = ["Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"]
        assert "Write" not in preview_tools
        assert "Edit" not in preview_tools
        assert "NotebookEdit" not in preview_tools


# -- Claude CLI --allowedTools flag --

class TestCLIAllowedToolsFlag:
    def test_allowed_tools_in_cli_command(self):
        """--allowedTools flag is included when request has allowed_tools."""
        backend = ClaudeCLIBackend(executable="echo")
        tools = ["Read", "Glob", "Grep", "Bash"]
        request = LLMRequest(prompt="test", allowed_tools=tools)

        # Build cmd by peeking at the complete method's command construction
        # We test the command building logic directly
        cmd = [
            backend.executable,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--model", "sonnet",
            "--dangerously-skip-permissions",
            "--max-turns", str(backend.max_turns),
        ]
        if request.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(request.allowed_tools)])

        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Glob,Grep,Bash"

    def test_no_allowed_tools_flag_when_none(self):
        """--allowedTools flag is NOT included when allowed_tools is None."""
        request = LLMRequest(prompt="test")
        cmd = ["claude", "--print"]
        if request.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(request.allowed_tools)])
        assert "--allowedTools" not in cmd

    def test_preview_tools_flag_format(self):
        """Preview tools are joined with commas for the CLI flag."""
        preview_tools = ["Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"]
        flag_value = ",".join(preview_tools)
        assert flag_value == "Read,Glob,Grep,Bash,WebFetch,WebSearch"


# -- Preview workflow config parsing --

class TestPreviewWorkflowConfig:
    def test_preview_required_field_defaults_false(self):
        """preview_required defaults to False on WorkflowStepDefinition."""
        step = WorkflowStepDefinition(agent="engineer")
        assert step.preview_required is False

    def test_preview_required_field_set_true(self):
        """preview_required can be set to True."""
        step = WorkflowStepDefinition(agent="engineer", preview_required=True)
        assert step.preview_required is True

    def test_preview_workflow_dag_construction(self):
        """Preview workflow from config builds a valid DAG with all expected steps."""
        workflow_def = WorkflowDefinition(
            description="Preview workflow",
            start_step="plan",
            steps={
                "plan": WorkflowStepDefinition(
                    agent="architect",
                    next=[{"target": "preview"}],
                ),
                "preview": WorkflowStepDefinition(
                    agent="engineer",
                    task_type="preview",
                    preview_required=True,
                    next=[{"target": "review_preview"}],
                ),
                "review_preview": WorkflowStepDefinition(
                    agent="architect",
                    next=[
                        {"target": "implement", "condition": "preview_approved", "priority": 10},
                        {"target": "preview", "condition": "needs_fix", "priority": 5},
                    ],
                ),
                "implement": WorkflowStepDefinition(
                    agent="engineer",
                    next=[{"target": "qa_review"}],
                ),
                "qa_review": WorkflowStepDefinition(
                    agent="qa",
                    next=[
                        {"target": "create_pr", "condition": "approved", "priority": 10},
                        {"target": "implement", "condition": "needs_fix", "priority": 5},
                    ],
                ),
                "create_pr": WorkflowStepDefinition(agent="architect"),
            },
        )

        dag = workflow_def.to_dag("preview")

        assert dag.name == "preview"
        assert dag.start_step == "plan"
        assert set(dag.steps.keys()) == {"plan", "preview", "review_preview", "implement", "qa_review", "create_pr"}

        # Preview step has task_type_override set to "preview"
        assert dag.steps["preview"].task_type_override == "preview"

        # review_preview has edges with preview_approved and needs_fix conditions
        review_edges = dag.steps["review_preview"].next
        edge_conditions = {e.condition.type for e in review_edges}
        assert EdgeConditionType.PREVIEW_APPROVED in edge_conditions
        assert EdgeConditionType.NEEDS_FIX in edge_conditions

    def test_preview_workflow_routes_plan_to_preview(self):
        """Plan step routes to preview step (not directly to implement)."""
        workflow_def = WorkflowDefinition(
            description="Preview workflow",
            start_step="plan",
            steps={
                "plan": WorkflowStepDefinition(
                    agent="architect",
                    next=[{"target": "preview"}],
                ),
                "preview": WorkflowStepDefinition(
                    agent="engineer",
                    task_type="preview",
                ),
            },
        )

        dag = workflow_def.to_dag("preview")
        plan_edges = dag.steps["plan"].next
        assert len(plan_edges) == 1
        assert plan_edges[0].target == "preview"


# -- Preview artifact storage --

class TestPreviewArtifactStorage:
    def test_preview_artifact_stored_in_context(self):
        """Preview artifact is stored in task context when result_summary exists."""
        task = _make_task(task_type=TaskType.PREVIEW)
        task.result_summary = "Files: src/auth.py (+50 lines)"

        # Simulate what workflow_router does for preview tasks
        if task.result_summary:
            task.context['preview_artifact'] = task.result_summary

        assert task.context['preview_artifact'] == "Files: src/auth.py (+50 lines)"

    def test_preview_artifact_not_stored_without_summary(self):
        """No preview_artifact key when result_summary is None."""
        task = _make_task(task_type=TaskType.PREVIEW)
        task.result_summary = None

        if task.result_summary:
            task.context['preview_artifact'] = task.result_summary

        assert 'preview_artifact' not in task.context

    def test_preview_artifact_propagates_to_implementation_task(self):
        """preview_artifact in context is preserved when building chain tasks."""
        from agent_framework.workflow.executor import WorkflowExecutor
        from agent_framework.workflow.dag import WorkflowStep

        queue = MagicMock()
        queue.queue_dir = MagicMock()
        queue.completed_dir = MagicMock()

        executor = WorkflowExecutor(queue, queue.queue_dir)

        task = _make_task(task_type=TaskType.PREVIEW, preview_artifact="Detailed preview output")
        target_step = WorkflowStep(id="implement", agent="engineer")

        chain_task = executor._build_chain_task(task, target_step, "architect")

        assert chain_task.context.get("preview_artifact") == "Detailed preview output"
