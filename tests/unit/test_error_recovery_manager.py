"""Tests for ErrorRecoveryManager."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from agent_framework.core.error_recovery import ErrorRecoveryManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse
from agent_framework.memory.memory_store import MemoryEntry


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
        acceptance_criteria=[],
        notes=[],
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


class TestGatherGitEvidence:
    """Test public gather_git_evidence method."""

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_formatted_evidence(self, mock_git):
        """Formats diff stat + diff into markdown sections."""
        mock_git.side_effect = [
            MagicMock(stdout=" src/auth.py | 10 ++++\n 1 file changed"),
            MagicMock(stdout="+def authenticate(token):\n+    return True"),
        ]
        manager = _make_manager()
        result = manager.gather_git_evidence(Path("/tmp/repo"))

        assert "Git Diff" in result
        assert "src/auth.py" in result
        assert "+def authenticate" in result

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_empty_on_clean_worktree(self, mock_git):
        """No changes → empty string."""
        mock_git.return_value = MagicMock(stdout="")
        manager = _make_manager()
        result = manager.gather_git_evidence(Path("/tmp/repo"))
        assert result == ""

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_empty_on_error(self, mock_git):
        """Git errors → empty string (non-fatal)."""
        mock_git.side_effect = RuntimeError("not a git repo")
        manager = _make_manager()
        result = manager.gather_git_evidence(Path("/tmp/bad"))
        assert result == ""


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

    @pytest.mark.asyncio
    async def test_skips_retry_when_task_already_completed(self):
        """Completed task with retries remaining — no queue.update(), status stays COMPLETED."""
        manager = _make_manager()
        task = _make_task(status=TaskStatus.COMPLETED, retry_count=1)

        manager.queue.find_task.return_value = task

        await manager.handle_failure(task)

        assert task.status == TaskStatus.COMPLETED
        manager.queue.update.assert_not_called()
        manager.queue.mark_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_escalation_when_task_already_completed(self):
        """Completed task at max retries — no mark_failed(), no escalation created."""
        manager = _make_manager()
        task = _make_task(status=TaskStatus.COMPLETED, retry_count=3)

        manager.queue.find_task.return_value = task

        await manager.handle_failure(task)

        assert task.status == TaskStatus.COMPLETED
        manager.queue.mark_failed.assert_not_called()
        manager.escalation_handler.create_escalation.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_failure_clears_upstream_context(self):
        """Stale upstream context from a failed attempt is cleared before retry."""
        manager = _make_manager()
        task = _make_task(
            retry_count=1,
            context={
                "upstream_summary": "stale output from previous attempt",
                "upstream_context_file": "/tmp/stale-context.md",
                "other_key": "preserved",
            },
        )

        manager.queue.find_task.return_value = task

        await manager.handle_failure(task)

        assert "upstream_summary" not in task.context
        assert "upstream_context_file" not in task.context
        assert task.context["other_key"] == "preserved"
        assert task.status == TaskStatus.PENDING


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

        result = await manager.self_evaluate(task, response, test_passed=True)

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

    def test_critique_alone_not_injected_by_replan(self):
        """Standalone critique (no _revised_plan) is correctly ignored by replan path.

        Self-eval critique without a revised plan is handled by
        _inject_self_eval_context in prompt_builder, not here.
        """
        manager = _make_manager()
        task = _make_task(
            context={"_self_eval_critique": "Missing test coverage for edge case"}
        )

        result = manager.inject_replan_context("original prompt", task)

        assert result == "original prompt"

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

    def test_replan_memory_filters_past_failures_by_error_type(self):
        """Tag-filtered recall for past_failures is called when error_type is known."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        # Return one match on the first (tag-filtered) call so we don't fall back
        memory_store.recall = MagicMock(
            return_value=[MemoryEntry(category="past_failures", content="fix A")]
        )
        manager.memory_store = memory_store
        manager.escalation_handler.categorize_error = MagicMock(return_value="type_error")

        task = _make_task(
            last_error="TypeError: expected str",
            context={"github_repo": "owner/repo"},
        )
        manager._build_replan_memory_context(task)

        first_call_kwargs = memory_store.recall.call_args_list[0].kwargs
        assert first_call_kwargs["category"] == "past_failures"
        assert first_call_kwargs.get("tags") == ["type_error"]

    def test_replan_memory_tag_filter_falls_back_to_unfiltered(self):
        """Empty tag-filtered result triggers an unfiltered past_failures recall."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True

        def recall_side_effect(**kwargs):
            # Return empty when tags filter is present, one memory otherwise
            if kwargs.get("tags"):
                return []
            if kwargs.get("category") == "past_failures":
                return [MemoryEntry(category="past_failures", content="generic fix")]
            return []

        memory_store.recall = MagicMock(side_effect=recall_side_effect)
        manager.memory_store = memory_store
        manager.escalation_handler.categorize_error = MagicMock(return_value="type_error")

        task = _make_task(
            last_error="TypeError: expected str",
            context={"github_repo": "owner/repo"},
        )
        result = manager._build_replan_memory_context(task)

        assert "generic fix" in result
        # Confirm the filtered call was made and that the fallback was the agent-scoped
        # past_failures call — not one of the always-unfiltered category calls (conventions, etc.)
        calls = memory_store.recall.call_args_list
        filtered = [c for c in calls if c.kwargs.get("tags") == ["type_error"]]
        unfiltered_past_failures = [
            c for c in calls
            if c.kwargs.get("category") == "past_failures"
            and c.kwargs.get("agent_type") != "shared"
            and not c.kwargs.get("tags")
        ]
        assert filtered, "Expected a tag-filtered past_failures call"
        assert unfiltered_past_failures, "Expected an unfiltered agent-scoped past_failures fallback call"

    def test_store_failure_antipattern_on_exhausted_retries(self):
        """Antipattern content records all attempted approaches and unions files across retries."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        task = _make_task(
            last_error="ConnectionError: timeout",
            replan_history=[
                {
                    "attempt": 2,
                    "error_type": "network_error",
                    "approach_tried": "retry with backoff",
                    "files_involved": ["src/client.py"],  # first attempt touches client
                    "revised_plan": "Add exponential backoff",
                },
                {
                    "attempt": 3,
                    "error_type": "network_error",
                    "approach_tried": "switch endpoint",
                    "files_involved": ["src/config.py"],  # second attempt touches config
                    "revised_plan": "Try fallback endpoint",
                },
            ],
        )

        manager.store_failure_antipattern(task, "owner/repo", "network_error")

        memory_store.remember.assert_called_once()
        call_kwargs = memory_store.remember.call_args.kwargs
        assert call_kwargs["category"] == "past_failures"
        assert "network_error" in call_kwargs["tags"]
        assert "retry with backoff" in call_kwargs["content"]
        assert "switch endpoint" in call_kwargs["content"]
        assert "unresolved" in call_kwargs["content"]
        # Files from both retry attempts must appear (union, not just last entry)
        assert "src/client.py" in call_kwargs["content"] and "src/config.py" in call_kwargs["content"]
        # "→ resolved" (without "un") is the success marker — must not appear here
        assert "→ resolved" not in call_kwargs["content"]

    def test_store_failure_antipattern_noop_without_history(self):
        """No memory written when the task never reached replanning."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        task = _make_task(replan_history=[])
        manager.store_failure_antipattern(task, "owner/repo", "logic_error")

        memory_store.remember.assert_not_called()

    def test_store_failure_antipattern_noop_without_memory_store(self):
        """No-op when memory store is None — guard clause exits before any write."""
        manager = _make_manager()
        assert manager.memory_store is None

        # Give the task real replan history so the only thing preventing a write is
        # the missing memory store, not the empty-history guard.
        task = _make_task(
            replan_history=[{"attempt": 2, "approach_tried": "fix", "revised_plan": "plan"}]
        )
        manager.store_failure_antipattern(task, "owner/repo", "logic_error")
        # Nothing to assert on (no mock), but reaching here without AttributeError
        # confirms the None guard short-circuits before touching memory_store.remember

    def test_replan_memory_shared_past_failures_apply_tag_filter(self):
        """Shared past_failures are tag-filtered by error type, same as agent-scoped ones."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        memory_store.recall = MagicMock(return_value=[])
        manager.memory_store = memory_store
        manager.escalation_handler.categorize_error = MagicMock(return_value="type_error")

        task = _make_task(
            last_error="TypeError: expected str",
            context={"github_repo": "owner/repo"},
        )
        manager._build_replan_memory_context(task)

        shared_past_failures_calls = [
            c for c in memory_store.recall.call_args_list
            if c.kwargs.get("agent_type") == "shared"
            and c.kwargs.get("category") == "past_failures"
        ]
        assert shared_past_failures_calls, "Expected at least one shared past_failures recall"
        assert shared_past_failures_calls[0].kwargs.get("tags") == ["type_error"]

    def test_replan_memory_includes_shared_namespace(self):
        """Shared-namespace memories appear in the replan context."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True

        def recall_side_effect(**kwargs):
            if kwargs.get("agent_type") == "shared" and kwargs.get("category") == "conventions":
                return [MemoryEntry(category="conventions", content="shared convention A")]
            return []

        memory_store.recall = MagicMock(side_effect=recall_side_effect)
        manager.memory_store = memory_store

        task = _make_task(context={"github_repo": "owner/repo"})
        result = manager._build_replan_memory_context(task)

        assert "shared convention A" in result

    def test_replan_memory_deduplicates_shared(self):
        """Content present in both agent-scoped and shared namespaces appears once."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True

        shared_content = "Use black for formatting"

        def recall_side_effect(**kwargs):
            if kwargs.get("category") == "conventions":
                return [MemoryEntry(category="conventions", content=shared_content)]
            return []

        memory_store.recall = MagicMock(side_effect=recall_side_effect)
        manager.memory_store = memory_store

        task = _make_task(context={"github_repo": "owner/repo"})
        result = manager._build_replan_memory_context(task)

        # The same content string should appear exactly once in the output
        assert result.count(shared_content) == 1

    def test_replan_memory_no_last_error_skips_tag_filter(self):
        """When task has no last_error, past_failures is fetched without tag filtering."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        memory_store.recall = MagicMock(
            return_value=[MemoryEntry(category="past_failures", content="some fix")]
        )
        manager.memory_store = memory_store

        task = _make_task(last_error=None, context={"github_repo": "owner/repo"})
        result = manager._build_replan_memory_context(task)

        assert "some fix" in result
        # No call should have a tags filter
        tags_used = [c.kwargs.get("tags") for c in memory_store.recall.call_args_list]
        assert all(t is None for t in tags_used)

    @pytest.mark.asyncio
    async def test_handle_failure_calls_store_failure_antipattern(self):
        """Antipattern is stored only when no escalation path exists."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        task = _make_task(
            retry_count=3,
            context={"github_repo": "owner/repo"},
            replan_history=[
                {
                    "attempt": 2,
                    "error_type": "logic_error",
                    "approach_tried": "add null check",
                    "files_involved": ["src/foo.py"],
                    "revised_plan": "Guard against None",
                }
            ],
        )
        manager.queue.find_task.return_value = task
        manager.escalation_handler.categorize_error.return_value = "logic_error"
        manager.retry_handler.can_create_escalation.return_value = False

        await manager.handle_failure(task)

        memory_store.remember.assert_called_once()
        call_kwargs = memory_store.remember.call_args.kwargs
        assert call_kwargs["category"] == "past_failures"
        assert "unresolved" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_handle_failure_no_antipattern_when_escalation_created(self):
        """Antipattern must not be stored when an escalation task is created.

        The escalation may succeed — we shouldn't label the sub-task's approaches
        as dead ends if they might work in the context of an escalated resolution.
        """
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        manager.memory_store = memory_store

        escalation_task = _make_task(id="esc-001", type=TaskType.ESCALATION)
        task = _make_task(
            retry_count=3,
            context={"github_repo": "owner/repo"},
            replan_history=[
                {
                    "attempt": 2,
                    "error_type": "logic_error",
                    "approach_tried": "add null check",
                    "files_involved": ["src/foo.py"],
                    "revised_plan": "Guard against None",
                }
            ],
        )
        manager.queue.find_task.return_value = task
        manager.escalation_handler.categorize_error.return_value = "logic_error"
        manager.retry_handler.can_create_escalation.return_value = True
        manager.escalation_handler.create_escalation.return_value = escalation_task

        await manager.handle_failure(task)

        memory_store.remember.assert_not_called()
        manager.escalation_handler.create_escalation.assert_called_once()
        manager.queue.push.assert_called_once_with(escalation_task, escalation_task.assigned_to)

    @pytest.mark.asyncio
    async def test_handle_failure_antipattern_exception_does_not_block_logging(self):
        """A memory write failure in the terminal path must not prevent failure logging."""
        manager = _make_manager()
        memory_store = MagicMock()
        memory_store.enabled = True
        memory_store.remember.side_effect = OSError("disk full")
        manager.memory_store = memory_store

        task = _make_task(
            retry_count=3,
            context={"github_repo": "owner/repo"},
            replan_history=[{"attempt": 2, "approach_tried": "fix", "revised_plan": "plan"}],
        )
        manager.queue.find_task.return_value = task
        manager.escalation_handler.categorize_error.return_value = "logic_error"
        # No escalation path — antipattern storage runs, then throws
        manager.retry_handler.can_create_escalation.return_value = False

        # Must not raise despite the OSError from memory_store.remember
        await manager.handle_failure(task)

        # Task was still marked failed and logged
        manager.queue.mark_failed.assert_called_once()


class TestTryDiffStrategies:
    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_prefers_uncommitted_changes_over_last_commit(self, mock_git):
        """HEAD strategy returns content — never reaches HEAD~1."""
        mock_git.side_effect = [
            MagicMock(stdout="file.py | 3 +++"),  # diff --stat HEAD
            MagicMock(stdout="+new code"),          # diff HEAD
        ]

        manager = _make_manager()
        stat, diff = manager._try_diff_strategies(Path("/tmp/repo"))

        assert "file.py" in stat
        assert "+new code" in diff
        # Only 2 calls: stat HEAD + diff HEAD — never fell through to staged/HEAD~1
        assert mock_git.call_count == 2
        assert mock_git.call_args_list[0][0][0] == ["diff", "--stat", "HEAD"]

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_falls_back_to_last_commit_when_no_uncommitted(self, mock_git):
        """HEAD and --staged empty, HEAD~1 has content."""
        mock_git.side_effect = [
            MagicMock(stdout=""),                   # diff --stat HEAD (empty)
            MagicMock(stdout=""),                   # diff --stat --staged (empty)
            MagicMock(stdout="app.py | 2 ++"),      # diff --stat HEAD~1
            MagicMock(stdout="+committed change"),   # diff HEAD~1
        ]

        manager = _make_manager()
        stat, diff = manager._try_diff_strategies(Path("/tmp/repo"))

        assert "app.py" in stat
        assert "+committed change" in diff

    @patch("agent_framework.core.error_recovery.run_git_command")
    def test_returns_empty_when_all_strategies_empty(self, mock_git):
        """All strategies return empty — returns empty tuple."""
        mock_git.return_value = MagicMock(stdout="")

        manager = _make_manager()
        stat, diff = manager._try_diff_strategies(Path("/tmp/repo"))

        assert stat == ""
        assert diff == ""


class TestSelfEvalAutoPass:
    @pytest.mark.asyncio
    async def test_auto_passes_without_git_evidence_or_tests(self):
        """No working_dir + test_passed=None → auto-pass, LLM not called."""
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Code compiles"])
        response = MagicMock(content="Did the thing")

        result = await manager.self_evaluate(
            task, response, test_passed=None, working_dir=None,
        )

        assert result is True
        manager.llm.complete.assert_not_called()
        # Should log AUTO_PASS
        manager.session_logger.log.assert_called_once()
        call_kwargs = manager.session_logger.log.call_args
        assert call_kwargs[1]["verdict"] == "AUTO_PASS"
        assert call_kwargs[1]["reason"] == "no_objective_evidence"

    @pytest.mark.asyncio
    async def test_still_evaluates_when_tests_failed(self):
        """test_passed=False + no git → LLM eval still runs (failed tests are evidence)."""
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Tests pass"])
        response = MagicMock(content="Done")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("FAIL — tests did not pass")
        )

        result = await manager.self_evaluate(
            task, response, test_passed=False, working_dir=None,
        )

        assert result is False
        manager.llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_still_evaluates_when_tests_ran(self):
        """test_passed=True + no git → LLM eval still runs."""
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Tests pass"])
        response = MagicMock(content="Done")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("PASS — tests passed")
        )

        result = await manager.self_evaluate(
            task, response, test_passed=True, working_dir=None,
        )

        assert result is True
        manager.llm.complete.assert_called_once()


class TestSelfEvalPrompt:
    @pytest.mark.asyncio
    async def test_prompt_labels_output_as_conversation_summary(self):
        """Eval prompt should label agent output as 'conversation summary — NOT code'."""
        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Feature works"])
        response = MagicMock(content="I implemented the feature")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("PASS")
        )

        await manager.self_evaluate(
            task, response, test_passed=True, working_dir=None,
        )

        prompt = manager.llm.complete.call_args[0][0].prompt
        assert "conversation summary — NOT code" in prompt
        assert "Do NOT fail solely because the conversation summary is vague" in prompt

    @pytest.mark.asyncio
    @patch("agent_framework.core.error_recovery.run_git_command")
    async def test_prompt_prioritizes_diff_over_prose(self, mock_git):
        """When git evidence exists, prompt instructs evaluator to weight diff."""
        mock_git.side_effect = [
            MagicMock(stdout="file.py | 5 +++++"),
            MagicMock(stdout="+code"),
        ]

        manager = _make_manager()
        task = _make_task(acceptance_criteria=["Feature works"])
        response = MagicMock(content="Did stuff")

        manager.llm.complete = AsyncMock(
            return_value=_llm_response("PASS")
        )

        await manager.self_evaluate(
            task, response, test_passed=True, working_dir=Path("/tmp/repo"),
        )

        prompt = manager.llm.complete.call_args[0][0].prompt
        assert "PRIORITIZE the git diff over the conversation summary" in prompt


class TestChecklistReport:
    """Tests for _build_checklist_report() in self-evaluation."""

    def test_report_with_matching_files(self):
        manager = _make_manager()
        task = _make_task(context={
            "requirements_checklist": [
                {"id": 1, "description": "Add dashboard panel", "files": ["src/dashboard.py"], "status": "pending"},
                {"id": 2, "description": "Create config parser", "files": ["src/config.py"], "status": "pending"},
            ],
        })
        git_evidence = "## Git Diff\n### Summary\nsrc/dashboard.py | 50 +++\nsrc/config.py | 30 +++"
        report = manager._build_checklist_report(task, git_evidence)

        assert "2/2 items appear in code changes" in report
        assert "✅" in report
        assert "❌" not in report

    def test_report_with_missing_files(self):
        manager = _make_manager()
        task = _make_task(context={
            "requirements_checklist": [
                {"id": 1, "description": "Add dashboard panel", "files": ["src/dashboard.py"], "status": "pending"},
                {"id": 2, "description": "Create debate metrics", "files": ["src/debate.py"], "status": "pending"},
            ],
        })
        git_evidence = "## Git Diff\n### Summary\nsrc/dashboard.py | 50 +++"
        report = manager._build_checklist_report(task, git_evidence)

        assert "1/2 items appear in code changes" in report
        assert "✅" in report
        assert "❌" in report
        assert "1 deliverable(s) appear to be missing" in report

    def test_no_report_without_checklist(self):
        manager = _make_manager()
        task = _make_task(context={})
        report = manager._build_checklist_report(task, "some diff")
        assert report == ""

    def test_keyword_matching_fallback(self):
        """When no file matches, distinctive words (>6 chars) from description are used."""
        manager = _make_manager()
        task = _make_task(context={
            "requirements_checklist": [
                {"id": 1, "description": "Create workflow routing handler", "files": [], "status": "pending"},
            ],
        })
        # "workflow" (8 chars) and "routing" (7 chars) both appear in diff
        git_evidence = "diff --git a/src/routing.py\n+class WorkflowRoutingHandler:\n+    workflow routing handler"
        report = manager._build_checklist_report(task, git_evidence)

        assert "1/1 items appear in code changes" in report

    def test_short_keywords_not_matched(self):
        """Words <= 6 chars shouldn't trigger keyword fallback."""
        manager = _make_manager()
        task = _make_task(context={
            "requirements_checklist": [
                {"id": 1, "description": "Add new panel to app", "files": [], "status": "pending"},
            ],
        })
        # "panel" (5 chars) and "app" (3 chars) are too short to match
        git_evidence = "diff --git a/panel.py\n+panel = Panel()\n+app.register(panel)"
        report = manager._build_checklist_report(task, git_evidence)

        assert "0/1 items appear in code changes" in report
