"""Tests for verdict salvage — recovering valid review verdicts from non-zero exit codes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Helpers --

def _make_task(task_id="salvage-test-001", assigned_to="qa"):
    return Task(
        id=task_id,
        type=TaskType.REVIEW,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="engineer",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Review PR #42",
        description="Review the PR.",
        context={"workflow": "default"},
    )


def _make_response(content="", success=False, error="exit code 1"):
    return SimpleNamespace(
        content=content,
        success=success,
        error=error,
        input_tokens=500,
        output_tokens=200,
        model_used="sonnet",
        latency_ms=5000,
        finish_reason="end_turn",
    )


LONG_APPROVE = (
    "I've reviewed the changes across all files. The implementation is solid, "
    "with proper error handling and good test coverage. No security issues found. "
    "Code follows established patterns and conventions. "
    "All tests pass. VERDICT: APPROVE\n\n"
    "Summary: Clean implementation, well-tested, ready to merge."
)

LONG_LGTM = (
    "Reviewed all changes. The refactoring is clean, maintains backward compatibility, "
    "and improves readability. Test coverage is adequate. "
    "No issues to report. LGTM\n\n"
    "The code is well-structured and follows the project conventions throughout."
)

LONG_REQUEST_CHANGES = (
    "I've reviewed the implementation carefully. There are several issues that need "
    "to be addressed before this can be merged:\n\n"
    "1. CRITICAL: SQL injection vulnerability in query builder\n"
    "2. Missing input validation on the API endpoint\n\n"
    "REQUEST_CHANGES - please fix the above issues and resubmit."
)

SHORT_APPROVE = "APPROVE"

NO_VERDICT_LONG = (
    "I've looked at the code and it seems reasonable. The implementation follows "
    "existing patterns. I don't have strong opinions either way about the approach. "
    "The tests are there but could be more comprehensive. Let me think about this more."
)


def _make_agent(agent_id="qa"):
    """Build a minimal Agent with ReviewCycleManager for testing _can_salvage_verdict."""
    from agent_framework.core.review_cycle import ReviewCycleManager

    config = AgentConfig(
        id=agent_id,
        name=f"Test {agent_id}",
        queue=agent_id.split("-")[0],
        prompt=f"You are {agent_id}.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.logger = MagicMock()
    a._review_cycle = ReviewCycleManager(
        config=config,
        queue=MagicMock(),
        logger=a.logger,
        agent_definition=None,
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )
    return a


# -- _can_salvage_verdict unit tests --

class TestCanSalvageVerdict:
    """Unit tests for _can_salvage_verdict guard logic."""

    def test_qa_approve_long_output(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=LONG_APPROVE)
        assert agent._can_salvage_verdict(task, response) is True

    def test_qa_lgtm_long_output(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=LONG_LGTM)
        assert agent._can_salvage_verdict(task, response) is True

    def test_qa_request_changes(self):
        """REQUEST_CHANGES is needs_fix — still salvageable."""
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=LONG_REQUEST_CHANGES)
        assert agent._can_salvage_verdict(task, response) is True

    def test_short_output_rejected(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=SHORT_APPROVE)
        assert agent._can_salvage_verdict(task, response) is False

    def test_empty_content_rejected(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content="")
        assert agent._can_salvage_verdict(task, response) is False

    def test_none_content_rejected(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=None)
        assert agent._can_salvage_verdict(task, response) is False

    def test_engineer_agent_rejected(self):
        """Engineer failures are real failures, never salvage."""
        agent = _make_agent("engineer")
        task = _make_task(assigned_to="engineer")
        response = _make_response(content=LONG_APPROVE)
        assert agent._can_salvage_verdict(task, response) is False

    def test_no_verdict_in_long_output(self):
        agent = _make_agent("qa")
        task = _make_task()
        response = _make_response(content=NO_VERDICT_LONG)
        assert agent._can_salvage_verdict(task, response) is False

    def test_architect_agent_salvageable(self):
        agent = _make_agent("architect")
        task = _make_task(assigned_to="architect")
        response = _make_response(content=LONG_APPROVE)
        assert agent._can_salvage_verdict(task, response) is True

    def test_replica_architect_salvageable(self):
        """architect-2 has base_id 'architect' — should be salvageable."""
        agent = _make_agent("architect-2")
        task = _make_task(assigned_to="architect-2")
        response = _make_response(content=LONG_APPROVE)
        assert agent._can_salvage_verdict(task, response) is True

    def test_replica_qa_salvageable(self):
        agent = _make_agent("qa-3")
        task = _make_task(assigned_to="qa-3")
        response = _make_response(content=LONG_LGTM)
        assert agent._can_salvage_verdict(task, response) is True

    def test_replica_engineer_rejected(self):
        agent = _make_agent("engineer-2")
        task = _make_task(assigned_to="engineer-2")
        response = _make_response(content=LONG_APPROVE)
        assert agent._can_salvage_verdict(task, response) is False

    def test_missing_content_attr(self):
        """Response object without content attribute."""
        agent = _make_agent("qa")
        task = _make_task()
        response = SimpleNamespace(success=False, error="crash")
        assert agent._can_salvage_verdict(task, response) is False


# -- Routing integration tests --

class TestVerdictSalvageRouting:
    """Verify _process_task routes salvageable failures to the success path."""

    @pytest.mark.asyncio
    async def test_salvageable_routes_to_success_handler(self):
        agent = MagicMock()
        agent._can_salvage_verdict = Agent._can_salvage_verdict.__get__(agent)
        agent.config = AgentConfig(id="qa", name="QA", queue="qa", prompt="QA")
        agent.config.base_id  # property access works via AgentConfig

        # Make salvage return True
        agent._can_salvage_verdict = MagicMock(return_value=True)

        response = _make_response(content=LONG_APPROVE, success=False)

        agent._handle_successful_response = AsyncMock()
        agent._handle_failed_response = AsyncMock()
        agent._populate_read_cache = MagicMock()
        agent._session_logger = MagicMock()
        agent.logger = MagicMock()

        # Simulate the routing logic from _process_task
        if response.success:
            agent._populate_read_cache(_make_task())
            await agent._handle_successful_response(_make_task(), response, None)
        elif agent._can_salvage_verdict(_make_task(), response):
            response.success = True
            response.finish_reason = "stop"
            agent._populate_read_cache(_make_task())
            await agent._handle_successful_response(_make_task(), response, None)
        else:
            await agent._handle_failed_response(_make_task(), response)

        agent._handle_successful_response.assert_called_once()
        agent._handle_failed_response.assert_not_called()
        assert response.success is True

    @pytest.mark.asyncio
    async def test_non_salvageable_routes_to_failure_handler(self):
        agent = MagicMock()
        agent._can_salvage_verdict = MagicMock(return_value=False)
        agent._handle_successful_response = AsyncMock()
        agent._handle_failed_response = AsyncMock()
        agent._populate_read_cache = MagicMock()

        response = _make_response(content="segfault", success=False)

        if response.success:
            await agent._handle_successful_response(_make_task(), response, None)
        elif agent._can_salvage_verdict(_make_task(), response):
            response.success = True
            await agent._handle_successful_response(_make_task(), response, None)
        else:
            await agent._handle_failed_response(_make_task(), response)

        agent._handle_failed_response.assert_called_once()
        agent._handle_successful_response.assert_not_called()
        assert response.success is False

    @pytest.mark.asyncio
    async def test_already_successful_skips_salvage(self):
        agent = MagicMock()
        agent._can_salvage_verdict = MagicMock(return_value=True)
        agent._handle_successful_response = AsyncMock()
        agent._handle_failed_response = AsyncMock()
        agent._populate_read_cache = MagicMock()

        response = _make_response(content=LONG_APPROVE, success=True)

        if response.success:
            agent._populate_read_cache(_make_task())
            await agent._handle_successful_response(_make_task(), response, None)
        elif agent._can_salvage_verdict(_make_task(), response):
            response.success = True
            await agent._handle_successful_response(_make_task(), response, None)
        else:
            await agent._handle_failed_response(_make_task(), response)

        agent._handle_successful_response.assert_called_once()
        agent._handle_failed_response.assert_not_called()
        # Salvage check never reached
        agent._can_salvage_verdict.assert_not_called()
