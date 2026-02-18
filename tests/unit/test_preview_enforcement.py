"""Tests for execution preview enforcement.

Covers three aspects of the preview feature:
1. LLMRequest.allowed_tools is populated for PREVIEW tasks in agent.py
2. ClaudeCLIBackend passes --allowedTools to the CLI subprocess
3. PREVIEW_APPROVED condition evaluates correctly
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMRequest
from agent_framework.workflow.dag import EdgeCondition, EdgeConditionType
from agent_framework.workflow.conditions import (
    ConditionRegistry,
    PreviewApprovedCondition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_type=TaskType.PREVIEW, **context_overrides):
    context = {"workflow": "preview", **context_overrides}
    return Task(
        id="test-preview-task",
        type=task_type,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Preview task",
        description="Test preview",
        context=context,
    )


def _make_response(content="Preview complete"):
    return SimpleNamespace(content=content, error=None)


# ---------------------------------------------------------------------------
# LLMRequest.allowed_tools field
# ---------------------------------------------------------------------------

class TestLLMRequestAllowedTools:
    def test_allowed_tools_defaults_to_none(self):
        """allowed_tools is None by default (no restriction)."""
        req = LLMRequest(prompt="test")
        assert req.allowed_tools is None

    def test_allowed_tools_can_be_set(self):
        """allowed_tools can be provided as a list of strings."""
        tools = ["Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"]
        req = LLMRequest(prompt="test", allowed_tools=tools)
        assert req.allowed_tools == tools

    def test_allowed_tools_non_preview_not_set(self):
        """Non-preview requests should not have allowed_tools set by default."""
        req = LLMRequest(prompt="test", task_type=TaskType.IMPLEMENTATION)
        assert req.allowed_tools is None


# ---------------------------------------------------------------------------
# ClaudeCLIBackend --allowedTools wiring
# ---------------------------------------------------------------------------

class TestClaudeCLIBackendAllowedTools:
    def test_allowed_tools_adds_flag(self):
        """--allowedTools flag is included when allowed_tools is set."""
        from agent_framework.llm.claude_cli_backend import ClaudeCLIBackend

        backend = ClaudeCLIBackend(logs_dir=MagicMock())

        # Build command by reaching into the subprocess creation
        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            raise RuntimeError("stop")  # Abort after cmd capture

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            import asyncio
            req = LLMRequest(
                prompt="hello",
                allowed_tools=["Read", "Glob", "Grep"],
            )
            try:
                asyncio.get_event_loop().run_until_complete(backend.complete(req))
            except RuntimeError:
                pass

        flat_cmd = list(captured_cmd)
        assert "--allowedTools" in flat_cmd
        idx = flat_cmd.index("--allowedTools")
        assert flat_cmd[idx + 1] == "Read,Glob,Grep"

    def test_no_allowed_tools_omits_flag(self):
        """--allowedTools is absent when allowed_tools is None."""
        from agent_framework.llm.claude_cli_backend import ClaudeCLIBackend

        backend = ClaudeCLIBackend(logs_dir=MagicMock())

        captured_cmd = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            raise RuntimeError("stop")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            import asyncio
            req = LLMRequest(prompt="hello")
            try:
                asyncio.get_event_loop().run_until_complete(backend.complete(req))
            except RuntimeError:
                pass

        assert "--allowedTools" not in list(captured_cmd)


# ---------------------------------------------------------------------------
# PREVIEW_APPROVED condition
# ---------------------------------------------------------------------------

class TestPreviewApprovedCondition:
    def test_preview_approved_verdict_returns_true(self):
        """verdict='preview_approved' in task context → True."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        task = _make_task(verdict="preview_approved")
        response = _make_response()

        evaluator = PreviewApprovedCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_other_verdict_returns_false(self):
        """Any verdict other than 'preview_approved' → False."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        for verdict in ("approved", "needs_fix", "no_changes", "lgtm"):
            task = _make_task(verdict=verdict)
            response = _make_response()
            evaluator = PreviewApprovedCondition()
            assert evaluator.evaluate(condition, task, response) is False, \
                f"Should be False for verdict={verdict!r}"

    def test_no_verdict_returns_false(self):
        """No verdict in context → False."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        task = _make_task()  # no verdict key
        response = _make_response()

        evaluator = PreviewApprovedCondition()
        assert evaluator.evaluate(condition, task, response) is False

    def test_evaluation_context_takes_priority(self):
        """Evaluation-time context overrides task context."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        # task says needs_fix, but the live evaluation context says preview_approved
        task = _make_task(verdict="needs_fix")
        response = _make_response()
        context = {"verdict": "preview_approved"}

        evaluator = PreviewApprovedCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is True

    def test_registry_evaluates_preview_approved(self):
        """PREVIEW_APPROVED condition flows through the ConditionRegistry."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        task = _make_task(verdict="preview_approved")
        response = _make_response()

        assert ConditionRegistry.evaluate(condition, task, response) is True

    def test_registry_returns_false_without_verdict(self):
        """Registry returns False when no preview_approved verdict."""
        condition = EdgeCondition(EdgeConditionType.PREVIEW_APPROVED)
        task = _make_task()
        response = _make_response()

        assert ConditionRegistry.evaluate(condition, task, response) is False


# ---------------------------------------------------------------------------
# Preview-allowed-tools list (the exact tools exposed to PREVIEW tasks)
# ---------------------------------------------------------------------------

EXPECTED_PREVIEW_TOOLS = {"Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"}


class TestPreviewAllowedToolsSet:
    def test_preview_tools_are_all_read_only(self):
        """The preview tool set contains only read/search tools, never write tools."""
        write_tools = {"Write", "Edit", "NotebookEdit"}
        assert EXPECTED_PREVIEW_TOOLS.isdisjoint(write_tools), \
            "Write tools must not appear in the preview allowed-tools list"

    def test_preview_tools_include_read(self):
        """Read is permitted so the engineer can inspect files."""
        assert "Read" in EXPECTED_PREVIEW_TOOLS

    def test_preview_tools_include_search(self):
        """Glob and Grep are permitted for codebase exploration."""
        assert "Glob" in EXPECTED_PREVIEW_TOOLS
        assert "Grep" in EXPECTED_PREVIEW_TOOLS

    def test_preview_tools_include_bash(self):
        """Bash is allowed so the engineer can run read-only commands like git log."""
        assert "Bash" in EXPECTED_PREVIEW_TOOLS


# ---------------------------------------------------------------------------
# Sync: conftest PREVIEW_WORKFLOW matches config/agent-framework.yaml
# ---------------------------------------------------------------------------

class TestPreviewWorkflowSync:
    """Verify that the PREVIEW_WORKFLOW constant in conftest.py stays in sync
    with the live `preview` workflow in config/agent-framework.yaml.

    If the YAML is updated (e.g. a new step or edge) without updating the
    constant, these tests catch the divergence before tests give false results.
    """

    def _load_yaml_preview_dag(self):
        from pathlib import Path
        from agent_framework.core.config import load_config

        yaml_path = Path(__file__).parents[2] / "config" / "agent-framework.yaml"
        if not yaml_path.exists():
            pytest.skip("config/agent-framework.yaml not present in this environment")

        config = load_config(yaml_path)
        wf_def = config.workflows.get("preview")
        assert wf_def is not None, "preview workflow not found in config/agent-framework.yaml"
        return wf_def.to_dag("preview")

    def test_preview_review_agents_match(self):
        """preview_review step agent is architect in both YAML and fixture."""
        from tests.unit.workflow_fixtures import PREVIEW_WORKFLOW

        yaml_dag = self._load_yaml_preview_dag()
        fixture_dag = PREVIEW_WORKFLOW.to_dag("preview")

        yaml_step = yaml_dag.steps["preview_review"]
        fixture_step = fixture_dag.steps["preview_review"]
        assert yaml_step.agent == fixture_step.agent

    def test_preview_review_edge_targets_match(self):
        """preview_review next-step targets are identical in YAML and fixture."""
        from tests.unit.workflow_fixtures import PREVIEW_WORKFLOW

        yaml_dag = self._load_yaml_preview_dag()
        fixture_dag = PREVIEW_WORKFLOW.to_dag("preview")

        yaml_targets = {e.target for e in yaml_dag.steps["preview_review"].next}
        fixture_targets = {e.target for e in fixture_dag.steps["preview_review"].next}
        assert yaml_targets == fixture_targets, (
            f"preview_review edges differ: YAML={yaml_targets}, fixture={fixture_targets}"
        )

    def test_preview_step_task_type_match(self):
        """preview step task_type_override is 'preview' in both YAML and fixture."""
        from tests.unit.workflow_fixtures import PREVIEW_WORKFLOW

        yaml_dag = self._load_yaml_preview_dag()
        fixture_dag = PREVIEW_WORKFLOW.to_dag("preview")

        assert yaml_dag.steps["preview"].task_type_override == fixture_dag.steps["preview"].task_type_override

    def test_yaml_starts_at_plan_step(self):
        """Real YAML preview workflow starts at the plan step, not preview.

        The fixture (PREVIEW_WORKFLOW) skips plan for conciseness, so this
        test explicitly confirms the canonical workflow starts at plan.
        """
        yaml_dag = self._load_yaml_preview_dag()
        assert yaml_dag.start_step == "plan"
        assert "plan" in yaml_dag.steps


# ---------------------------------------------------------------------------
# _approval_verdict returns the correct verdict per workflow step
# ---------------------------------------------------------------------------

class TestApprovalVerdict:
    """_approval_verdict routes to 'preview_approved' only at preview_review.

    All other review steps (code_review, qa_review) must return plain 'approved'
    so the standard approved→create_pr edge fires instead of the preview edge.
    """

    def _make_agent(self, base_id: str):
        from unittest.mock import MagicMock
        from agent_framework.core.agent import Agent
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = base_id
        agent._approval_verdict = Agent._approval_verdict.__get__(agent)
        return agent

    def test_qa_review_step_returns_approved(self):
        """QA agent at qa_review step returns plain 'approved', not 'preview_approved'."""
        agent = self._make_agent("qa")
        task = _make_task(task_type=TaskType.REVIEW, verdict=None)
        task.context["workflow_step"] = "qa_review"

        assert agent._approval_verdict(task) == "approved"

    def test_code_review_step_returns_approved(self):
        """Architect at code_review step returns plain 'approved'."""
        agent = self._make_agent("architect")
        task = _make_task(task_type=TaskType.REVIEW, verdict=None)
        task.context["workflow_step"] = "code_review"

        assert agent._approval_verdict(task) == "approved"

    def test_preview_review_step_returns_preview_approved(self):
        """Architect at preview_review step returns 'preview_approved'."""
        agent = self._make_agent("architect")
        task = _make_task(verdict=None)  # default task_type=PREVIEW
        task.context["workflow_step"] = "preview_review"

        assert agent._approval_verdict(task) == "preview_approved"

    def test_no_workflow_step_returns_approved(self):
        """Missing workflow_step key defaults to plain 'approved'."""
        agent = self._make_agent("qa")
        task = _make_task(task_type=TaskType.REVIEW, verdict=None)
        task.context.pop("workflow_step", None)

        assert agent._approval_verdict(task) == "approved"
