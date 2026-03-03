"""Tests for post_completion.py — PostCompletionManager and helper functions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock


from agent_framework.core.post_completion import (
    PostCompletionManager,
    strip_tool_call_markers,
    _IMPLEMENTATION_STEP_IDS,
    _NON_CODE_STEP_IDS,
    _NO_CHANGES_MARKER,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**ctx_overrides):
    ctx = {"github_repo": "org/repo", "workflow": "default", **ctx_overrides}
    return Task(
        id="task-abc",
        title="Test task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        context=ctx,
        created_at=datetime.now(timezone.utc),
        created_by="architect",
        assigned_to="engineer",
        description="Test task description",
        priority=50,
    )


def _make_manager(**overrides):
    defaults = dict(
        config=MagicMock(id="engineer", base_id="engineer"),
        queue=MagicMock(),
        workspace=Path("/tmp/workspace"),
        logger=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
        review_cycle=MagicMock(),
        workflow_router=MagicMock(),
        git_ops=MagicMock(),
        budget=MagicMock(),
        error_recovery=MagicMock(),
        optimization_config={},
        session_logging_enabled=False,
        session_logs_dir=Path("/tmp/logs"),
        agent_definition=None,
    )
    defaults.update(overrides)
    return PostCompletionManager(**defaults)


# ---------------------------------------------------------------------------
# strip_tool_call_markers (module-level function)
# ---------------------------------------------------------------------------

class TestStripToolCallMarkers:
    def test_removes_markers(self):
        content = "Start.\n[Tool Call: Read]\nEnd."
        result = strip_tool_call_markers(content)
        assert "[Tool Call:" not in result
        assert "Start." in result
        assert "End." in result

    def test_empty_string(self):
        assert strip_tool_call_markers("") == ""

    def test_none_returns_empty(self):
        assert strip_tool_call_markers(None) == ""

    def test_no_markers_passthrough(self):
        assert strip_tool_call_markers("clean text") == "clean text"

    def test_compresses_triple_newlines(self):
        content = "A.\n[Tool Call: Read]\n\n\nB."
        result = strip_tool_call_markers(content)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# is_no_changes_response
# ---------------------------------------------------------------------------

class TestIsNoChangesResponse:
    def test_positive(self):
        content = f"{_NO_CHANGES_MARKER} — nothing to do"
        assert PostCompletionManager.is_no_changes_response(content) is True

    def test_marker_too_far(self):
        # Marker after first 200 chars should not match
        content = "x" * 201 + _NO_CHANGES_MARKER
        assert PostCompletionManager.is_no_changes_response(content) is False

    def test_empty(self):
        assert PostCompletionManager.is_no_changes_response("") is False
        assert PostCompletionManager.is_no_changes_response(None) is False


# ---------------------------------------------------------------------------
# is_implementation_step
# ---------------------------------------------------------------------------

class TestIsImplementationStep:
    def test_implement_step(self):
        task = _make_task(workflow_step="implement")
        assert PostCompletionManager.is_implementation_step(task, "engineer") is True

    def test_plan_step_not_implementation(self):
        task = _make_task(workflow_step="plan")
        assert PostCompletionManager.is_implementation_step(task, "engineer") is False

    def test_unknown_step_engineer(self):
        task = _make_task(workflow_step="custom_step")
        assert PostCompletionManager.is_implementation_step(task, "engineer") is True

    def test_unknown_step_architect(self):
        task = _make_task(workflow_step="custom_step")
        assert PostCompletionManager.is_implementation_step(task, "architect") is False

    def test_step_ids_are_frozensets(self):
        assert isinstance(_IMPLEMENTATION_STEP_IDS, frozenset)
        assert isinstance(_NON_CODE_STEP_IDS, frozenset)


# ---------------------------------------------------------------------------
# extract_plan_from_response
# ---------------------------------------------------------------------------

class TestExtractPlanFromResponse:
    def test_extracts_valid_plan(self):
        plan_data = {
            "objectives": ["Implement feature X"],
            "approach": ["Step 1"],
            "success_criteria": ["Tests pass"],
            "files_to_modify": ["src/foo.py"],
        }
        content = f"Here is the plan:\n```json\n{json.dumps(plan_data)}\n```"
        result = PostCompletionManager.extract_plan_from_response(content)
        assert result is not None
        assert result.objectives == ["Implement feature X"]

    def test_returns_none_for_empty(self):
        assert PostCompletionManager.extract_plan_from_response("") is None
        assert PostCompletionManager.extract_plan_from_response(None) is None

    def test_returns_none_for_invalid_json(self):
        content = "```json\n{invalid}\n```"
        assert PostCompletionManager.extract_plan_from_response(content) is None

    def test_returns_none_for_missing_fields(self):
        content = '```json\n{"objectives": ["x"]}\n```'
        assert PostCompletionManager.extract_plan_from_response(content) is None

    def test_unwraps_nested_plan_key(self):
        plan_data = {
            "plan": {
                "objectives": ["Implement feature X"],
                "approach": ["Step 1"],
                "success_criteria": ["Tests pass"],
                "files_to_modify": ["src/foo.py"],
            }
        }
        content = f"```json\n{json.dumps(plan_data)}\n```"
        result = PostCompletionManager.extract_plan_from_response(content)
        assert result is not None


# ---------------------------------------------------------------------------
# extract_design_rationale
# ---------------------------------------------------------------------------

class TestExtractDesignRationale:
    def test_extracts_rationale(self):
        content = "We chose X because it reduces complexity. The tradeoff is speed."
        result = PostCompletionManager.extract_design_rationale(content)
        assert result is not None
        assert "because" in result.lower()

    def test_returns_none_for_empty(self):
        assert PostCompletionManager.extract_design_rationale("") is None
        assert PostCompletionManager.extract_design_rationale(None) is None

    def test_returns_none_when_no_rationale(self):
        assert PostCompletionManager.extract_design_rationale("Just some code.") is None

    def test_truncates_long_rationale(self):
        # Generate a long sentence with a rationale keyword
        content = "This is long because " + "x " * 600 + "."
        result = PostCompletionManager.extract_design_rationale(content)
        if result:
            assert len(result) <= 1000


# ---------------------------------------------------------------------------
# extract_structured_findings_from_content
# ---------------------------------------------------------------------------

class TestExtractStructuredFindings:
    def test_extracts_list(self):
        findings = [{"severity": "high", "message": "Bug found"}]
        content = f"```json\n{json.dumps(findings)}\n```"
        result = PostCompletionManager.extract_structured_findings_from_content(content)
        assert result["findings"] == findings

    def test_extracts_dict_with_findings(self):
        findings = {"findings": [{"severity": "low"}], "summary": "OK"}
        content = f"```json\n{json.dumps(findings)}\n```"
        result = PostCompletionManager.extract_structured_findings_from_content(content)
        assert len(result["findings"]) == 1

    def test_returns_empty_for_no_json(self):
        assert PostCompletionManager.extract_structured_findings_from_content("no json here") == {}

    def test_returns_empty_for_empty(self):
        assert PostCompletionManager.extract_structured_findings_from_content("") == {}


# ---------------------------------------------------------------------------
# approval_verdict
# ---------------------------------------------------------------------------

class TestApprovalVerdict:
    def test_standard_step(self):
        mgr = _make_manager()
        task = _make_task(workflow_step="qa_review")
        assert mgr.approval_verdict(task) == "approved"

    def test_preview_review_step(self):
        mgr = _make_manager()
        task = _make_task(workflow_step="preview_review")
        assert mgr.approval_verdict(task) == "preview_approved"


# ---------------------------------------------------------------------------
# set_session_logger
# ---------------------------------------------------------------------------

class TestSetSessionLogger:
    def test_updates_logger(self):
        mgr = _make_manager()
        new_logger = MagicMock()
        mgr.set_session_logger(new_logger)
        assert mgr.session_logger is new_logger
