"""Tests for ErrorRecoveryManager."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides):
    defaults = dict(
        id="test-task-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Add feature X to the system",
        last_error="TypeError: expected str got int",
        retry_count=2,
        replan_history=[],
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_manager():
    config = MagicMock()
    config.id = "test-agent"
    config.base_id = "engineer"
    config.max_retries = 3

    queue = MagicMock()
    llm = AsyncMock()
    logger = MagicMock()
    session_logger = MagicMock()
    retry_handler = MagicMock()
    retry_handler.max_retries = 3
    escalation_handler = MagicMock()
    escalation_handler.categorize_error = MagicMock(return_value="logic_error")
    workspace = MagicMock()
    jira_client = None
    memory_store = None

    manager = ErrorRecoveryManager(
        config=config,
        queue=queue,
        llm=llm,
        logger=logger,
        session_logger=session_logger,
        retry_handler=retry_handler,
        escalation_handler=escalation_handler,
        workspace=workspace,
        jira_client=jira_client,
        memory_store=memory_store,
        replan_config={"enabled": True, "model": "haiku", "min_retry_for_replan": 2},
        self_eval_config={"enabled": True, "model": "haiku", "max_retries": 2},
    )

    return manager


def _llm_response(content: str, success: bool = True):
    return LLMResponse(
        content=content,
        model_used="haiku",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
        latency_ms=200,
        success=success,
    )


class TestHandleFailure:
    @pytest.mark.asyncio
    async def test_resets_task_when_retries_remaining(self):
        manager = _make_manager()
        task = _make_task(retry_count=1)

        manager.queue.find_task.return_value = task

        await manager.handle_failure(task)

        # Should reset task to pending
        assert task.status == TaskStatus.PENDING
        manager.queue.update.assert_called_once_with(task)

    @pytest.mark.asyncio
    async def test_marks_failed_when_max_retries_exceeded(self):
        manager = _make_manager()
        task = _make_task(retry_count=3)

        manager.queue.find_task.return_value = task
        manager.escalation_handler.categorize_error.return_value = "api_error"
        manager.retry_handler.can_create_escalation.return_value = False

        await manager.handle_failure(task)

        # Should mark task as failed
        assert task.status == TaskStatus.FAILED
        manager.queue.mark_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_escalation_when_eligible(self):
        manager = _make_manager()
        task = _make_task(retry_count=3)
        escalation_task = _make_task(id="escalation-001", type=TaskType.ESCALATION)

        manager.queue.find_task.return_value = task
        manager.escalation_handler.categorize_error.return_value = "api_error"
        manager.retry_handler.can_create_escalation.return_value = True
        manager.escalation_handler.create_escalation.return_value = escalation_task

        await manager.handle_failure(task)

        # Should create escalation
        manager.escalation_handler.create_escalation.assert_called_once()
        manager.queue.push.assert_called_once_with(escalation_task, escalation_task.assigned_to)

    @pytest.mark.asyncio
    async def test_triggers_replan_on_retry_2_plus(self):
        manager = _make_manager()
        task = _make_task(retry_count=2)

        manager.queue.find_task.return_value = task
        manager.llm.complete = AsyncMock(
            return_value=_llm_response("Try approach B")
        )

        await manager.handle_failure(task)

        # Should have called replan
        manager.llm.complete.assert_called_once()
        assert "_revised_plan" in task.context

    @pytest.mark.asyncio
    async def test_handles_cancelled_task(self):
        manager = _make_manager()
        task = _make_task(status=TaskStatus.IN_PROGRESS)
        cancelled_task = _make_task(status=TaskStatus.CANCELLED)

        manager.queue.find_task.return_value = cancelled_task

        await manager.handle_failure(task)

        # Should preserve cancelled status
        manager.queue.move_to_completed.assert_called_once_with(cancelled_task)
        manager.queue.update.assert_not_called()


class TestSelfEvaluate:
    @pytest.mark.asyncio
    async def test_passes_when_no_acceptance_criteria(self):
        manager = _make_manager()
        task = _make_task(acceptance_criteria=[])
        response = MagicMock(content="Task completed successfully")

        result = await manager.self_evaluate(task, response)

        assert result is True
        manager.llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_when_eval_limit_reached(self):
        manager = _make_manager()
        task = _make_task(
            acceptance_criteria=["Test 1", "Test 2"],
            context={"_self_eval_count": 2}
        )
        response = MagicMock(content="Task completed")

        result = await manager.self_evaluate(task, response)

        assert result is True
        manager.llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_when_verdict_is_pass(self):
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Test 1", "Test 2"])
        response = MagicMock(content="Task completed successfully")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("PASS - all criteria met")
        )

        result = await manager.self_evaluate(task, response)

        assert result is True
        assert task.status == TaskStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_resets_when_verdict_is_fail(self):
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Test 1", "Test 2"])
        response = MagicMock(content="Task completed")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("FAIL - missing test coverage")
        )

        result = await manager.self_evaluate(task, response)

        assert result is False
        assert task.status == TaskStatus.PENDING
        assert task.context["_self_eval_count"] == 1
        assert "_self_eval_critique" in task.context
        manager.queue.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Test 1"])
        response = MagicMock(content="Task completed")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("", success=False)
        )

        result = await manager.self_evaluate(task, response)

        # Should proceed despite LLM failure
        assert result is True


class TestRequestReplan:
    @pytest.mark.asyncio
    async def test_stores_revised_plan_in_context(self):
        manager = _make_manager()
        task = _make_task()

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("1. Try approach B\n2. Use different API")
        )

        await manager.request_replan(task)

        assert task.context["_revised_plan"] == "1. Try approach B\n2. Use different API"
        assert task.context["_replan_attempt"] == 2

    @pytest.mark.asyncio
    async def test_appends_to_replan_history(self):
        manager = _make_manager()
        task = _make_task()

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("New approach")
        )

        await manager.request_replan(task)

        assert len(task.replan_history) == 1
        entry = task.replan_history[0]
        assert entry["attempt"] == 2
        assert "TypeError" in entry["error"]
        assert entry["revised_plan"] == "New approach"
        # Verify enriched fields are present
        assert "error_type" in entry
        assert "approach_tried" in entry
        assert "files_involved" in entry

    @pytest.mark.asyncio
    async def test_includes_previous_attempts_in_prompt(self):
        manager = _make_manager()
        task = _make_task(
            retry_count=3,
            replan_history=[
                {"attempt": 2, "error": "first error", "revised_plan": "plan A"}
            ],
        )

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("Third approach")
        )

        await manager.request_replan(task)

        # Verify the prompt included previous attempt info
        call_args = manager.llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "first error" in prompt

    @pytest.mark.asyncio
    async def test_llm_failure_non_fatal(self):
        manager = _make_manager()
        task = _make_task()

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("", success=False)
        )

        await manager.request_replan(task)

        # Should not crash, context unchanged
        assert "_revised_plan" not in task.context


class TestInjectReplanContext:
    def test_no_op_without_revised_plan(self):
        manager = _make_manager()
        task = _make_task()

        result = manager.inject_replan_context("original prompt", task)
        assert result == "original prompt"

    def test_appends_revised_approach(self):
        manager = _make_manager()
        task = _make_task(context={"_revised_plan": "Try approach B"})

        result = manager.inject_replan_context("original prompt", task)

        assert "original prompt" in result
        assert "REVISED APPROACH" in result
        assert "Try approach B" in result

    def test_includes_self_eval_critique(self):
        manager = _make_manager()
        task = _make_task(
            context={
                "_revised_plan": "Try approach B",
                "_self_eval_critique": "Missing test coverage",
            }
        )

        result = manager.inject_replan_context("prompt", task)

        assert "Self-Evaluation Feedback" in result
        assert "Missing test coverage" in result

    def test_displays_enriched_attempt_history(self):
        """Test that inject_replan_context shows error_type, approach_tried, and files."""
        manager = _make_manager()
        task = _make_task(
            context={"_revised_plan": "Latest plan"},
            replan_history=[
                {
                    "attempt": 2,
                    "error": "Import error: module not found",
                    "error_type": "dependency",
                    "approach_tried": "Install package directly",
                    "files_involved": ["setup.py", "requirements.txt"],
                    "revised_plan": "Use pip install",
                },
                {
                    "attempt": 3,
                    "error": "Test failure",
                    "error_type": "test_failure",
                    "approach_tried": "Use pip install",
                    "files_involved": ["test_foo.py"],
                    "revised_plan": "Latest plan",
                },
            ],
        )

        result = manager.inject_replan_context("prompt", task)

        # Should show the first attempt (second is current, gets skipped)
        assert "Attempt 2:" in result
        assert "Install package directly" in result  # approach_tried
        assert "dependency error" in result  # error_type
        assert "setup.py" in result or "requirements.txt" in result  # files_involved
        # Should not duplicate current attempt
        assert result.count("Latest plan") == 1


class TestReplanMemory:
    def test_replan_memory_includes_past_failures(self):
        """Verify past_failures is in the priority list for replan memory context."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        memory_store.recall = MagicMock(return_value=[])
        manager.memory_store = memory_store

        task = _make_task(context={"github_repo": "owner/repo"})
        manager._build_replan_memory_context(task)

        # past_failures should be the first category queried
        recall_calls = memory_store.recall.call_args_list
        categories_queried = [c.kwargs["category"] for c in recall_calls]
        assert "past_failures" in categories_queried
        assert categories_queried[0] == "past_failures"

    def test_store_replan_outcome_on_success(self):
        """Verify memory stored with correct category and tags after successful replan."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        task = _make_task(
            status=TaskStatus.COMPLETED,
            replan_history=[
                {
                    "attempt": 2,
                    "error": "TypeError in handler",
                    "error_type": "type_error",
                    "approach_tried": "cast to str",
                    "files_involved": ["src/handler.py"],
                    "revised_plan": "Use explicit type check before assignment\nThen validate",
                }
            ],
        )

        manager.store_replan_outcome(task, "owner/repo")

        memory_store.remember.assert_called_once()
        call_kwargs = memory_store.remember.call_args.kwargs
        assert call_kwargs["category"] == "past_failures"
        assert call_kwargs["repo_slug"] == "owner/repo"
        assert "type_error" in call_kwargs["tags"]
        assert "src/handler.py" in call_kwargs["content"]
        assert "resolved" in call_kwargs["content"]

    def test_store_replan_outcome_skipped_without_history(self):
        """No-op when replan_history is empty."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        task = _make_task(replan_history=[])

        manager.store_replan_outcome(task, "owner/repo")

        memory_store.remember.assert_not_called()

    def test_store_replan_outcome_skipped_without_memory_store(self):
        """No-op when memory store is None."""
        manager = _make_manager()
        assert manager.memory_store is None

        task = _make_task(
            replan_history=[{"attempt": 2, "error": "err", "revised_plan": "fix"}]
        )

        # Should not raise
        manager.store_replan_outcome(task, "owner/repo")
