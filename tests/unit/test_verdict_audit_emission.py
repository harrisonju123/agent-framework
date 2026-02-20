"""Tests for _set_structured_verdict() emitting verdict_audit events."""

from unittest.mock import MagicMock, patch

from agent_framework.core.agent import Agent
from agent_framework.core.review_cycle import ReviewCycleManager, ReviewOutcome
from agent_framework.core.verdict_audit import VerdictAudit


def _make_agent(**overrides):
    """Build a MagicMock with the real _set_structured_verdict bound to it."""
    agent = MagicMock()
    agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
    agent._is_no_changes_response = Agent._is_no_changes_response
    agent._approval_verdict = Agent._approval_verdict.__get__(agent)

    agent.config.id = overrides.get("agent_id", "qa")
    agent.config.base_id = overrides.get("base_id", "qa")
    agent._session_logger = MagicMock()
    agent.logger = MagicMock()

    # Real ReviewCycleManager so _parse_review_outcome_audited works
    agent._review_cycle = ReviewCycleManager(
        config=agent.config,
        queue=MagicMock(),
        logger=agent.logger,
        agent_definition=MagicMock(),
        session_logger=agent._session_logger,
        activity_manager=MagicMock(),
    )

    return agent


def _make_task(**ctx_overrides):
    task = MagicMock()
    task.id = ctx_overrides.pop("task_id", "task-123")
    task.type = ctx_overrides.pop("task_type", "qa_verification")
    task.context = {"workflow": "default", "workflow_step": "qa_review"}
    task.context.update(ctx_overrides)
    return task


def _make_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


class TestVerdictAuditEmission:
    def test_approved_emits_review_outcome(self):
        agent = _make_agent()
        task = _make_task()
        response = _make_response("Everything APPROVED. No issues.")

        agent._set_structured_verdict(task, response)

        assert task.context["verdict"] in ("approved", "preview_approved")
        assert "verdict_audit" in task.context
        audit = task.context["verdict_audit"]
        assert audit["method"] == "review_outcome"
        assert audit["agent_id"] == "qa"
        assert audit["task_id"] == "task-123"
        agent._session_logger.log.assert_called_once()
        call_args = agent._session_logger.log.call_args
        assert call_args[0][0] == "verdict_audit"

    def test_needs_fix_emits_review_outcome(self):
        agent = _make_agent()
        task = _make_task()
        response = _make_response("CRITICAL: SQL injection in login.py")

        agent._set_structured_verdict(task, response)

        assert task.context["verdict"] == "needs_fix"
        audit = task.context["verdict_audit"]
        assert audit["method"] == "review_outcome"
        assert audit["value"] == "needs_fix"

    def test_ambiguous_at_review_step_emits_halt(self):
        agent = _make_agent()
        task = _make_task(workflow_step="qa_review")
        response = _make_response("I reviewed the code and it looks okay.")

        agent._set_structured_verdict(task, response)

        # Ambiguous at review step: no verdict set
        assert "verdict" not in task.context or task.context.get("verdict") is None
        audit = task.context["verdict_audit"]
        assert audit["method"] == "ambiguous_halt"

    def test_ambiguous_at_non_review_step_emits_default(self):
        agent = _make_agent(base_id="architect", agent_id="architect")
        task = _make_task(workflow_step="plan")
        response = _make_response("I have created a detailed plan for the implementation.")

        agent._set_structured_verdict(task, response)

        audit = task.context["verdict_audit"]
        assert audit["method"] == "ambiguous_default"
        assert audit["value"] in ("approved", "preview_approved")

    def test_no_changes_marker_overrides(self):
        agent = _make_agent(base_id="architect", agent_id="architect")
        task = _make_task(workflow_step="plan")
        response = _make_response("[NO_CHANGES_NEEDED] The feature already exists.")

        agent._set_structured_verdict(task, response)

        assert task.context["verdict"] == "no_changes"
        audit = task.context["verdict_audit"]
        assert audit["method"] == "no_changes_marker"
        assert audit["no_changes_marker_found"] is True
        assert audit["value"] == "no_changes"

    def test_skips_for_engineer(self):
        agent = _make_agent(base_id="engineer", agent_id="engineer")
        task = _make_task()
        response = _make_response("APPROVED all tests pass")

        agent._set_structured_verdict(task, response)

        assert "verdict_audit" not in task.context

    def test_skips_without_workflow(self):
        agent = _make_agent()
        task = _make_task()
        task.context.pop("workflow")
        response = _make_response("APPROVED")

        agent._set_structured_verdict(task, response)

        assert "verdict_audit" not in task.context

    def test_audit_has_matched_patterns(self):
        agent = _make_agent()
        task = _make_task()
        response = _make_response("APPROVED. LGTM.")

        agent._set_structured_verdict(task, response)

        audit = task.context["verdict_audit"]
        assert len(audit["matched_patterns"]) >= 2

    def test_audit_content_snippet(self):
        agent = _make_agent()
        task = _make_task()
        long_content = "APPROVED. " + "x" * 300
        response = _make_response(long_content)

        agent._set_structured_verdict(task, response)

        audit = task.context["verdict_audit"]
        assert len(audit["content_snippet"]) == 200
