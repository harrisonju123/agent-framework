"""Tests for condition_eval_audit emission in ConditionRegistry.evaluate()."""

from unittest.mock import MagicMock

from agent_framework.workflow.conditions import ConditionRegistry
from agent_framework.workflow.dag import EdgeCondition, EdgeConditionType


def _make_task(**ctx):
    task = MagicMock()
    task.id = ctx.pop("task_id", "task-1")
    task.context = ctx
    return task


def _make_response(content=""):
    resp = MagicMock()
    resp.content = content
    return resp


class TestConditionEvalAudit:
    def setup_method(self):
        ConditionRegistry.reset()

    def test_approved_with_verdict_logs_condition_verdict(self):
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.APPROVED)
        task = _make_task(verdict="approved")
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        session_logger.log.assert_called_once()
        call_args = session_logger.log.call_args
        assert call_args[0][0] == "condition_eval_audit"
        assert call_args[1]["condition_type"] == "approved"
        assert call_args[1]["method"] == "condition_verdict"
        assert call_args[1]["result"] is True

    def test_needs_fix_with_verdict_logs_condition_verdict(self):
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.NEEDS_FIX)
        task = _make_task(verdict="needs_fix")
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        call_kwargs = session_logger.log.call_args[1]
        assert call_kwargs["method"] == "condition_verdict"
        assert call_kwargs["verdict_value"] == "needs_fix"

    def test_keyword_fallback_logs_with_snippet(self):
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.APPROVED)
        task = _make_task()  # no verdict in context
        response = _make_response("LGTM everything looks good")

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        call_kwargs = session_logger.log.call_args[1]
        assert call_kwargs["method"] == "keyword_fallback"
        assert "content_snippet" in call_kwargs

    def test_no_changes_condition_logs(self):
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.NO_CHANGES)
        task = _make_task(verdict="no_changes")
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        call_kwargs = session_logger.log.call_args[1]
        assert call_kwargs["condition_type"] == "no_changes"

    def test_preview_approved_condition_logs(self):
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.PREVIEW_APPROVED)
        task = _make_task(verdict="preview_approved")
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        session_logger.log.assert_called_once()

    def test_non_verdict_condition_does_not_log(self):
        """ALWAYS and PR_CREATED are not verdict-sensitive - no audit log."""
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.ALWAYS)
        task = _make_task()
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        assert result is True
        session_logger.log.assert_not_called()

    def test_no_session_logger_skips_audit(self):
        """When session_logger is None, no crash and no audit."""
        condition = EdgeCondition(type=EdgeConditionType.APPROVED)
        task = _make_task(verdict="approved")
        response = _make_response()

        result = ConditionRegistry.evaluate(
            condition, task, response, session_logger=None,
        )
        assert result is True

    def test_condition_verdict_no_content_snippet(self):
        """When using structured verdict, content_snippet is NOT included."""
        session_logger = MagicMock()
        condition = EdgeCondition(type=EdgeConditionType.APPROVED)
        task = _make_task(verdict="approved")
        response = _make_response("LGTM")

        ConditionRegistry.evaluate(
            condition, task, response, session_logger=session_logger,
        )

        call_kwargs = session_logger.log.call_args[1]
        assert "content_snippet" not in call_kwargs
