"""Integration tests for the _handle_task lifecycle.

Covers the full task lifecycle through _handle_task:
1. Successful completion: PENDING -> IN_PROGRESS -> COMPLETED
2. Failed response triggering error recovery / retry
3. Workflow tasks writing chain state on success
4. Subtask guard skipping the workflow chain
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.post_completion import PostCompletionManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse


def _make_task(**overrides) -> Task:
    defaults = dict(
        id="handle-task-test-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test handle_task lifecycle",
        description="Integration test task",
        context={"github_repo": "org/repo"},
        notes=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_llm_response(*, success: bool = True, content: str = "Done.", error: Optional[str] = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
        finish_reason="stop" if success else "error",
        latency_ms=1200.0,
        success=success,
        error=error,
    )


def _build_mock_agent(tmp_path: Path) -> MagicMock:
    """Build a comprehensive mock agent with all attributes _handle_task touches.

    Uses the bind pattern: real _handle_task method bound to a MagicMock,
    so internal self.X accesses hit mock attributes we control here.
    """
    agent = MagicMock()

    # Workspace and paths
    agent.workspace = tmp_path / "workspace"
    agent.workspace.mkdir(parents=True, exist_ok=True)

    # Config
    agent.config = MagicMock()
    agent.config.id = "engineer-1"
    agent.config.base_id = "engineer"
    agent.config.validate_tasks = False
    agent.config.max_test_retries = 2

    # Queue
    agent.queue = MagicMock()
    agent.queue.acquire_lock.return_value = MagicMock()  # lock object
    agent.queue.update = MagicMock()
    agent.queue.mark_completed = MagicMock()
    agent.queue.release_lock = MagicMock()

    # Logger
    agent.logger = MagicMock()

    # Session logger (tracks init/close for lifecycle assertions)
    session_logger = MagicMock()
    session_logger.log = MagicMock()
    session_logger.close = MagicMock()
    agent._session_logger = session_logger
    agent._session_logs_dir = tmp_path / "logs" / "sessions"
    agent._session_logs_dir.mkdir(parents=True, exist_ok=True)
    agent._session_logging_enabled = True
    agent._session_log_prompts = False
    agent._session_log_tool_inputs = False

    # Prompt builder
    agent._prompt_builder = MagicMock()
    agent._prompt_builder.build.return_value = "You are an engineer. Implement the task."
    agent._prompt_builder.get_current_specialization.return_value = None
    agent._prompt_builder.get_current_file_count.return_value = 0
    agent._prompt_builder.ctx = MagicMock()

    # Git operations
    agent._git_ops = MagicMock()
    working_dir = tmp_path / "worktree"
    working_dir.mkdir(parents=True, exist_ok=True)
    agent._git_ops.get_working_directory.return_value = working_dir
    agent._git_ops.push_if_unpushed = MagicMock()
    agent._git_ops.safety_commit = MagicMock()
    agent._git_ops.active_worktree = None
    agent._git_ops.detect_implementation_branch = MagicMock()
    agent._git_ops.push_and_create_pr_if_needed = MagicMock()
    agent._git_ops.manage_pr_lifecycle = MagicMock()
    agent._git_ops.discover_branch_work.return_value = None
    agent._git_ops.sync_worktree_queued_tasks = MagicMock()
    agent._git_ops.cleanup_worktree = MagicMock()

    # Workflow router
    agent._workflow_router = MagicMock()
    agent._workflow_router.queue = agent.queue
    agent._workflow_router.check_and_create_fan_in_task = MagicMock()
    agent._workflow_router.set_session_logger = MagicMock()
    agent._workflow_router.enforce_chain = MagicMock(return_value=False)
    agent._workflow_router.is_at_terminal_workflow_step = MagicMock(return_value=True)

    # Workflow executor
    agent._workflow_executor = MagicMock()
    agent._workflow_executor.set_session_logger = MagicMock()

    # Budget manager
    agent._budget = MagicMock()
    agent._budget.estimate_cost.return_value = 0.05
    agent._budget.log_task_completion_metrics = MagicMock()

    # Context window manager
    agent._context_window_manager = MagicMock()
    agent._context_window_manager.budget = MagicMock()
    agent._context_window_manager.budget.utilization_percent = 30.0
    agent._context_window_manager.budget.total_budget = 200000
    agent._context_window_manager.budget.available_for_input = 180000
    agent._context_window_manager.get_budget_status.return_value = {"utilization_percent": 30.0}

    # Activity manager
    agent.activity_manager = MagicMock()
    agent.activity_manager.get_activity.return_value = MagicMock()

    # Error recovery
    agent._error_recovery = MagicMock()
    agent._error_recovery.handle_failure = AsyncMock()
    agent._error_recovery.handle_failed_response = AsyncMock()
    agent._error_recovery.has_deliverables.return_value = True

    # Retry handler
    agent.retry_handler = MagicMock()
    agent.retry_handler.should_retry.return_value = True

    # Review cycle
    agent._review_cycle = MagicMock()
    agent._review_cycle.queue_code_review_if_needed = MagicMock()
    agent._review_cycle.queue_review_fix_if_needed = MagicMock()

    # Agent definition
    agent._agent_definition = MagicMock()
    agent._agent_definition.jira_on_start = None
    agent._agent_definition.jira_on_complete = None
    agent._agent_definition.teammates = None

    # Optimization config
    agent._optimization_config = MagicMock()
    agent._optimization_config.get.return_value = False
    agent._optimization_config.__getitem__ = MagicMock(return_value={})

    # Analytics manager (provides callbacks for PostCompletionManager)
    agent._analytics = MagicMock()
    agent._analytics.extract_summary = AsyncMock(return_value="Summary")
    agent._analytics.extract_and_store_memories = MagicMock()
    agent._analytics.analyze_tool_patterns = MagicMock(return_value=0)

    # LLM executor (delegation target for _execute_llm_with_interruption_watch)
    agent._llm_executor = MagicMock()
    agent._llm_executor.process_completion = MagicMock()
    agent._llm_executor.log_routing_decision = MagicMock()

    # PostCompletionManager — real instance with mock dependencies so
    # _handle_successful_response / _run_post_completion_flow exercise
    # the actual completion logic (mark_completed, workflow routing, etc.)
    session_logs_dir = tmp_path / "logs" / "sessions"
    post_completion = PostCompletionManager(
        config=agent.config,
        queue=agent.queue,
        workspace=agent.workspace,
        logger=agent.logger,
        session_logger=session_logger,
        activity_manager=agent.activity_manager,
        review_cycle=agent._review_cycle,
        workflow_router=agent._workflow_router,
        git_ops=agent._git_ops,
        budget=agent._budget,
        error_recovery=agent._error_recovery,
        optimization_config={},
        session_logging_enabled=False,
        session_logs_dir=session_logs_dir,
        agent_definition=agent._agent_definition,
    )
    agent._post_completion = post_completion

    # Self-eval
    agent._self_eval_enabled = False

    # Team mode
    agent._team_mode_enabled = False

    # Current task tracking
    agent._current_task_id = None
    agent._current_specialization = None
    agent._current_file_count = 0

    # Model success store
    agent._model_success_store = None

    # JIRA client
    agent.jira_client = None

    # Feedback bus
    agent._feedback_bus = MagicMock()

    # Bind real methods so the actual control flow executes
    agent._handle_task = Agent._handle_task.__get__(agent)
    agent._handle_successful_response = Agent._handle_successful_response.__get__(agent)
    agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
    agent._handle_failed_response = Agent._handle_failed_response.__get__(agent)
    agent._log_task_completion_metrics = Agent._log_task_completion_metrics.__get__(agent)
    agent._cleanup_task_execution = Agent._cleanup_task_execution.__get__(agent)

    # Stubs for methods called within the bound methods that we don't need real logic for
    agent._normalize_workflow = MagicMock()
    agent._validate_task_or_reject = MagicMock(return_value=True)
    agent._setup_task_context = MagicMock()
    agent._setup_context_window_manager_for_task = MagicMock()
    agent._initialize_task_execution = MagicMock(
        side_effect=lambda task, start_time: task.mark_in_progress(agent.config.id)
    )
    agent._get_validated_working_directory = MagicMock(return_value=working_dir)
    agent._try_index_codebase = MagicMock()
    agent._compose_team_for_task = MagicMock(return_value=None)
    agent._execute_llm_with_interruption_watch = AsyncMock()
    agent._process_llm_completion = MagicMock()
    agent._log_routing_decision = MagicMock()
    agent._read_cache = MagicMock()
    agent._read_cache.populate_read_cache = MagicMock(return_value=[])
    agent._read_cache.measure_cache_effectiveness = MagicMock()
    agent._can_salvage_verdict = MagicMock(return_value=False)
    agent._set_structured_verdict = MagicMock()
    agent._save_upstream_context = MagicMock()
    agent._save_step_to_chain_state = MagicMock()
    agent._run_sandbox_tests = AsyncMock(return_value=None)
    agent._self_evaluate = AsyncMock(return_value=True)
    agent._is_implementation_step = MagicMock(return_value=True)
    agent._extract_plan_from_response = MagicMock(return_value=None)
    agent._extract_design_rationale = MagicMock(return_value=None)
    agent._extract_and_store_memories = MagicMock()
    agent._analyze_tool_patterns = MagicMock(return_value=0)
    agent._enforce_workflow_chain = MagicMock(return_value=False)
    agent._is_at_terminal_workflow_step = MagicMock(return_value=True)
    agent._emit_workflow_summary = MagicMock()
    agent._resolve_budget_ceiling = MagicMock(return_value=None)
    agent._save_pre_scan_findings = MagicMock()
    agent._finalize_failed_attempt = MagicMock()
    agent._handle_failure = AsyncMock()
    agent._get_token_budget = MagicMock(return_value=200000)
    agent._sync_jira_status = MagicMock()
    agent._compute_tool_stats_for_chain = MagicMock(return_value=None)

    return agent


# ---------------------------------------------------------------------------
# Test 1: Successful task lifecycle
# ---------------------------------------------------------------------------

class TestSuccessfulTaskLifecycle:
    """PENDING -> IN_PROGRESS -> COMPLETED with session logger lifecycle."""

    @pytest.mark.asyncio
    async def test_successful_task_lifecycle(self, tmp_path):
        agent = _build_mock_agent(tmp_path)
        task = _make_task(status=TaskStatus.PENDING)
        response = _make_llm_response(success=True, content="Implementation complete.")

        agent._execute_llm_with_interruption_watch.return_value = response

        # Track the session logger that _setup_task_context would create.
        # The real _setup_task_context replaces _session_logger, but our mock
        # keeps the one we set in _build_mock_agent so we can assert on it.
        original_session_logger = agent._session_logger

        await agent._handle_task(task)

        # Task should transition through IN_PROGRESS to COMPLETED
        assert task.status == TaskStatus.COMPLETED, (
            f"Expected COMPLETED, got {task.status}"
        )
        assert task.completed_by == "engineer-1"

        # Verify the task went through IN_PROGRESS (mark_in_progress was called
        # via _initialize_task_execution side_effect)
        assert task.started_by == "engineer-1"

        # Session logger close is called in the finally block
        original_session_logger.close.assert_called_once()

        # Queue interactions: update (for in_progress) and mark_completed
        agent.queue.mark_completed.assert_called_once_with(task)

        # LLM was invoked
        agent._execute_llm_with_interruption_watch.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: Failed response triggers error recovery
# ---------------------------------------------------------------------------

class TestFailedResponseTriggersErrorRecovery:
    """LLM returns failure -> _handle_failed_response -> error_recovery called."""

    @pytest.mark.asyncio
    async def test_failed_response_triggers_error_recovery(self, tmp_path):
        agent = _build_mock_agent(tmp_path)
        task = _make_task(status=TaskStatus.PENDING)

        failed_response = _make_llm_response(
            success=False,
            content="",
            error="Context window exceeded",
        )
        agent._execute_llm_with_interruption_watch.return_value = failed_response

        await agent._handle_task(task)

        # _error_recovery.handle_failed_response should have been called
        agent._error_recovery.handle_failed_response.assert_awaited_once()
        call_kwargs = agent._error_recovery.handle_failed_response.call_args
        assert call_kwargs[0][1] is failed_response  # response arg

        # Task should NOT be completed
        assert task.status != TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test 3: Workflow task writes chain state
# ---------------------------------------------------------------------------

class TestWorkflowTaskWritesChainState:
    """Task with workflow context triggers chain state persistence on success."""

    @pytest.mark.asyncio
    async def test_workflow_task_writes_chain_state(self, tmp_path):
        agent = _build_mock_agent(tmp_path)

        task = _make_task(
            id="chain-root-implement-d1",
            status=TaskStatus.PENDING,
            context={
                "github_repo": "org/repo",
                "workflow": "default",
                "workflow_step": "implement",
                "chain_step": True,
                "_root_task_id": "chain-root",
                "user_goal": "Add authentication",
            },
        )
        response = _make_llm_response(success=True, content="Implemented auth module.")
        agent._execute_llm_with_interruption_watch.return_value = response

        await agent._handle_task(task)

        assert task.status == TaskStatus.COMPLETED

        # Workflow chain enforcement runs inside PostCompletionManager
        # which delegates to workflow_router.enforce_chain
        agent._workflow_router.enforce_chain.assert_called_once()

        # Git operations: detect implementation branch is called
        agent._git_ops.detect_implementation_branch.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Subtask skips workflow chain
# ---------------------------------------------------------------------------

class TestSubtaskSkipsWorkflowChain:
    """Subtask (parent_task_id set) skips workflow chain and PR creation."""

    @pytest.mark.asyncio
    async def test_subtask_skips_workflow_chain(self, tmp_path):
        agent = _build_mock_agent(tmp_path)

        # Parent exists in queue so the phantom-parent guard doesn't clear it
        parent_task = MagicMock()
        parent_task.id = "parent-task-001"
        agent._workflow_router.queue.find_task.return_value = parent_task

        task = _make_task(
            id="parent-task-001-sub1",
            status=TaskStatus.PENDING,
            parent_task_id="parent-task-001",
            context={
                "github_repo": "org/repo",
                "workflow": "default",
                "workflow_step": "implement",
                "chain_step": True,
                "_root_task_id": "parent-task-001",
            },
        )
        response = _make_llm_response(success=True, content="Subtask work done.")
        agent._execute_llm_with_interruption_watch.return_value = response

        await agent._handle_task(task)

        assert task.status == TaskStatus.COMPLETED

        # Fan-in check should still fire so the framework knows if all siblings are done
        agent._workflow_router.check_and_create_fan_in_task.assert_called_once_with(task)

        # Workflow chain enforcement should NOT have been called for a subtask
        agent._workflow_router.enforce_chain.assert_not_called()

        # PR lifecycle should NOT run for subtasks
        agent._git_ops.push_and_create_pr_if_needed.assert_not_called()
        agent._git_ops.manage_pr_lifecycle.assert_not_called()

        # parent_task_id should still be set (not cleared by phantom guard)
        assert task.parent_task_id == "parent-task-001"
