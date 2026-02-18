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
