"""Pipeline E2E test: full workflow chain from plan to PR.

Exercises the connected pipeline where completing one task creates the
next chain task in the queue, which is then claimed and processed:

    plan (architect) → implement (engineer) → code_review (architect)
      → qa_review (qa) → create_pr (architect) → PR created

Uses real FileQueue, WorkflowExecutor, WorkflowRouter, and PostCompletionManager.
Only LLM and external services (git, GitHub) stay mocked.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.post_completion import PostCompletionManager
from agent_framework.core.review_cycle import ReviewCycleManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.core.workflow_router import WorkflowRouter
from agent_framework.llm.base import LLMResponse
from agent_framework.queue.file_queue import FileQueue
from agent_framework.workflow.executor import WorkflowExecutor

from tests.unit.workflow_fixtures import PIPELINE_WORKFLOW


# -- Helpers ------------------------------------------------------------------

WORKFLOWS_CONFIG = {"default": PIPELINE_WORKFLOW}

# Agents config stub — WorkflowRouter needs it for routing signal validation
AGENTS_CONFIG: list = []


def _make_llm_response(
    *, success: bool = True, content: str = "Done.", error: Optional[str] = None,
) -> LLMResponse:
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


STEP_RESPONSES = {
    "plan": _make_llm_response(content="Planning complete. Implementation approach defined."),
    "implement": _make_llm_response(content="Implemented the requested changes."),
    "code_review": _make_llm_response(content="Code review APPROVED. Implementation looks good."),
    "qa_review": _make_llm_response(content="QA review APPROVED. All checks pass."),
    "create_pr": _make_llm_response(content="Created PR https://github.com/org/repo/pull/42"),
}


def _build_pipeline_agent(
    tmp_path: Path,
    agent_id: str,
    queue: FileQueue,
    *,
    base_id: Optional[str] = None,
) -> MagicMock:
    """Build a mock agent wired to real queue/workflow infrastructure.

    Adapts _build_mock_agent from test_handle_task_e2e but uses real
    FileQueue, WorkflowExecutor, WorkflowRouter, and PostCompletionManager.
    """
    agent = MagicMock()

    resolved_base_id = base_id or agent_id

    # Workspace and paths — all agents share the same workspace (tmp_path)
    agent.workspace = tmp_path / "workspace"
    agent.workspace.mkdir(parents=True, exist_ok=True)

    # Config
    agent.config = MagicMock(spec=AgentConfig)
    agent.config.id = agent_id
    agent.config.base_id = resolved_base_id
    agent.config.validate_tasks = False
    agent.config.max_test_retries = 2

    # Shared real queue
    agent.queue = queue

    # Logger
    agent.logger = MagicMock()
    agent.logger.phase_change = MagicMock()

    # Session logger
    session_logger = MagicMock()
    session_logger.log = MagicMock()
    session_logger.close = MagicMock()
    agent._session_logger = session_logger
    agent._session_logs_dir = tmp_path / "logs" / "sessions"
    agent._session_logs_dir.mkdir(parents=True, exist_ok=True)
    agent._session_logging_enabled = False
    agent._session_log_prompts = False
    agent._session_log_tool_inputs = False

    # Prompt builder
    agent._prompt_builder = MagicMock()
    agent._prompt_builder.build.return_value = "Execute the task."
    agent._prompt_builder.get_current_specialization.return_value = None
    agent._prompt_builder.get_current_file_count.return_value = 0
    agent._prompt_builder.ctx = MagicMock()

    # Git operations — mocked (no real git)
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

    # Real WorkflowExecutor
    workflow_executor = WorkflowExecutor(
        queue=queue,
        queue_dir=queue.queue_dir,
        agent_logger=agent.logger,
        workspace=agent.workspace,
        activity_manager=None,
        session_logger=session_logger,
    )
    agent._workflow_executor = workflow_executor

    # Real WorkflowRouter
    workflow_router = WorkflowRouter(
        config=agent.config,
        queue=queue,
        workspace=agent.workspace,
        logger=agent.logger,
        session_logger=session_logger,
        workflows_config=WORKFLOWS_CONFIG,
        workflow_executor=workflow_executor,
        agents_config=AGENTS_CONFIG,
    )
    agent._workflow_router = workflow_router

    # Review cycle — real instance for verdict parsing
    agent._review_cycle = ReviewCycleManager(
        config=agent.config,
        queue=queue,
        logger=agent.logger,
        agent_definition=None,
        session_logger=session_logger,
        activity_manager=MagicMock(),
    )

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

    # Agent definition
    agent._agent_definition = MagicMock()
    agent._agent_definition.jira_on_start = None
    agent._agent_definition.jira_on_complete = None
    agent._agent_definition.teammates = None

    # Optimization config
    agent._optimization_config = MagicMock()
    agent._optimization_config.get.return_value = False
    agent._optimization_config.__getitem__ = MagicMock(return_value={})

    # Analytics manager
    agent._analytics = MagicMock()
    agent._analytics.extract_summary = AsyncMock(return_value="Summary")
    agent._analytics.extract_and_store_memories = MagicMock()
    agent._analytics.analyze_tool_patterns = MagicMock(return_value=0)

    # LLM executor
    agent._llm_executor = MagicMock()
    agent._llm_executor.process_completion = MagicMock()
    agent._llm_executor.log_routing_decision = MagicMock()

    # Real PostCompletionManager — wired to real router + review cycle
    post_completion = PostCompletionManager(
        config=agent.config,
        queue=queue,
        workspace=agent.workspace,
        logger=agent.logger,
        session_logger=session_logger,
        activity_manager=agent.activity_manager,
        review_cycle=agent._review_cycle,
        workflow_router=workflow_router,
        git_ops=agent._git_ops,
        budget=agent._budget,
        error_recovery=agent._error_recovery,
        optimization_config={},
        session_logging_enabled=False,
        session_logs_dir=agent._session_logs_dir,
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

    # Bind real Agent methods so the actual control flow executes
    agent._handle_task = Agent._handle_task.__get__(agent)
    agent._handle_successful_response = Agent._handle_successful_response.__get__(agent)
    agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
    agent._handle_failed_response = Agent._handle_failed_response.__get__(agent)
    agent._log_task_completion_metrics = Agent._log_task_completion_metrics.__get__(agent)
    agent._cleanup_task_execution = Agent._cleanup_task_execution.__get__(agent)

    # Stubs for methods called within bound methods that don't need real logic
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
    # Real verdict setter — bound to PostCompletionManager
    agent._set_structured_verdict = Agent._set_structured_verdict.__get__(agent)
    agent._save_upstream_context = Agent._save_upstream_context.__get__(agent)
    agent._save_step_to_chain_state = Agent._save_step_to_chain_state.__get__(agent)
    agent._run_sandbox_tests = AsyncMock(return_value=None)
    agent._self_evaluate = AsyncMock(return_value=True)
    agent._is_implementation_step = Agent._is_implementation_step.__get__(agent)
    agent._extract_plan_from_response = MagicMock(return_value=None)
    agent._extract_design_rationale = MagicMock(return_value=None)
    agent._extract_and_store_memories = MagicMock()
    agent._analyze_tool_patterns = MagicMock(return_value=0)
    agent._enforce_workflow_chain = Agent._enforce_workflow_chain.__get__(agent)
    agent._is_at_terminal_workflow_step = Agent._is_at_terminal_workflow_step.__get__(agent)
    agent._emit_workflow_summary = Agent._emit_workflow_summary.__get__(agent)
    agent._resolve_budget_ceiling = MagicMock(return_value=None)
    agent._save_pre_scan_findings = MagicMock()
    agent._finalize_failed_attempt = MagicMock()
    agent._handle_failure = AsyncMock()
    agent._get_token_budget = MagicMock(return_value=200000)
    agent._sync_jira_status = MagicMock()
    agent._compute_tool_stats_for_chain = MagicMock(return_value=None)

    return agent


def _make_seed_task(task_id: str = "pipeline-test-001") -> Task:
    """Create the initial planning task that kicks off the pipeline."""
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=50,
        created_by="user",
        assigned_to="architect",
        created_at=datetime.now(timezone.utc),
        title="Add authentication module",
        description="Implement user authentication with JWT tokens",
        context={
            "github_repo": "org/repo",
            "workflow": "default",
            "workflow_step": "plan",
            "user_goal": "Implement user authentication with JWT tokens",
            # implementation_branch needed so _has_diff_for_pr doesn't block create_pr
            "implementation_branch": "agent-pipeline-test-001",
        },
        notes=[],
    )


def _claim_chain_task(queue: FileQueue, queue_name: str, agent_id: str) -> Optional[tuple]:
    """Claim the next chain task, skipping fire-and-forget pre-scan tasks.

    The implement→code_review transition queues a QA pre-scan as a side-channel
    task. This helper marks pre-scans as completed and continues claiming until
    a real chain task (or None) is found.
    """
    while True:
        result = queue.claim(queue_name, agent_id)
        if result is None:
            return None
        task, lock = result
        if task.context.get("pre_scan"):
            # Pre-scan is fire-and-forget; complete it so it doesn't block
            task.mark_completed(agent_id)
            queue.mark_completed(task)
            queue.release_lock(lock)
            continue
        return (task, lock)


async def _run_step(
    agent: MagicMock,
    queue: FileQueue,
    queue_name: str,
    agent_id: str,
    step_name: str,
) -> Task:
    """Claim a task from queue, set scripted LLM response, run _handle_task."""
    result = _claim_chain_task(queue, queue_name, agent_id)
    assert result is not None, (
        f"Expected a task in {queue_name!r} queue for step {step_name!r}, but queue was empty"
    )
    task, lock = result

    response = STEP_RESPONSES[step_name]
    agent._execute_llm_with_interruption_watch.return_value = response

    await agent._handle_task(task, lock=lock)
    return task


# -- Tests --------------------------------------------------------------------

class TestHappyPathPlanToPR:
    """Full pipeline: plan → implement → code_review → qa_review → create_pr."""

    @pytest.mark.asyncio
    async def test_happy_path_plan_to_pr(self, tmp_path):
        # -- Setup: shared queue, three agents --
        queue = FileQueue(tmp_path / "workspace")

        architect = _build_pipeline_agent(tmp_path, "architect", queue)
        engineer = _build_pipeline_agent(tmp_path, "engineer", queue)
        qa = _build_pipeline_agent(tmp_path, "qa", queue)

        seed = _make_seed_task()
        queue.push(seed, "architect")

        # -- Step 1: Architect plans --
        plan_task = await _run_step(architect, queue, "architect", "architect", "plan")
        assert plan_task.status == TaskStatus.COMPLETED
        assert plan_task.context.get("verdict") == "approved"

        # Chain task should be in engineer queue
        impl_result = _claim_chain_task(queue, "engineer", "engineer")
        assert impl_result is not None, "Expected chain task in engineer queue after plan"
        impl_task, impl_lock = impl_result
        assert impl_task.context["workflow_step"] == "implement"
        assert impl_task.context["chain_step"] is True
        assert impl_task.context["_chain_depth"] == 1
        assert impl_task.context["_root_task_id"] == seed.id
        # Verdict should be cleared for downstream agent
        assert "verdict" not in impl_task.context

        # -- Step 2: Engineer implements --
        engineer._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["implement"]
        await engineer._handle_task(impl_task, lock=impl_lock)
        assert impl_task.status == TaskStatus.COMPLETED

        # Chain task should be in architect queue for code review
        cr_result = _claim_chain_task(queue, "architect", "architect")
        assert cr_result is not None, "Expected chain task in architect queue after implement"
        cr_task, cr_lock = cr_result
        assert cr_task.context["workflow_step"] == "code_review"
        assert cr_task.context["_chain_depth"] == 2
        assert "verdict" not in cr_task.context

        # -- Step 3: Architect code review (approved) --
        architect._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["code_review"]
        await architect._handle_task(cr_task, lock=cr_lock)
        assert cr_task.status == TaskStatus.COMPLETED
        assert cr_task.context.get("verdict") == "approved"

        # Chain task should be in qa queue
        qa_result = _claim_chain_task(queue, "qa", "qa")
        assert qa_result is not None, "Expected chain task in qa queue after code_review"
        qa_task, qa_lock = qa_result
        assert qa_task.context["workflow_step"] == "qa_review"
        assert qa_task.context["_chain_depth"] == 3
        assert "verdict" not in qa_task.context

        # -- Step 4: QA review (approved) --
        qa._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["qa_review"]
        await qa._handle_task(qa_task, lock=qa_lock)
        assert qa_task.status == TaskStatus.COMPLETED
        assert qa_task.context.get("verdict") == "approved"

        # Chain task should be in architect queue for PR creation
        pr_result = _claim_chain_task(queue, "architect", "architect")
        assert pr_result is not None, "Expected chain task in architect queue after qa_review"
        pr_task, pr_lock = pr_result
        assert pr_task.context["workflow_step"] == "create_pr"
        assert pr_task.context["_chain_depth"] == 4
        assert "verdict" not in pr_task.context

        # -- Step 5: Architect creates PR (terminal step) --
        architect._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["create_pr"]
        await architect._handle_task(pr_task, lock=pr_lock)
        assert pr_task.status == TaskStatus.COMPLETED

        # PR creation should have been triggered (terminal step, no further routing)
        architect._git_ops.push_and_create_pr_if_needed.assert_called()

        # No more tasks in any queue
        assert _claim_chain_task(queue, "architect", "architect") is None
        assert _claim_chain_task(queue, "engineer", "engineer") is None
        assert _claim_chain_task(queue, "qa", "qa") is None

        # Chain state file should exist
        chain_state_dir = (tmp_path / "workspace" / ".agent-communication" / "chain-state")
        chain_files = list(chain_state_dir.glob("*.json")) if chain_state_dir.exists() else []
        assert len(chain_files) >= 1, "Expected chain state file for root task"

    @pytest.mark.asyncio
    async def test_chain_task_ids_follow_format(self, tmp_path):
        """Verify chain task IDs use the stable format: chain-{root}-{step}-d{depth}."""
        queue = FileQueue(tmp_path / "workspace")
        architect = _build_pipeline_agent(tmp_path, "architect", queue)
        engineer = _build_pipeline_agent(tmp_path, "engineer", queue)

        seed = _make_seed_task(task_id="root-001")
        queue.push(seed, "architect")

        # Plan step
        await _run_step(architect, queue, "architect", "architect", "plan")

        # Check implement chain task ID
        impl_result = _claim_chain_task(queue, "engineer", "engineer")
        assert impl_result is not None
        impl_task, impl_lock = impl_result
        assert impl_task.id == "chain-root-001-implement-d1"

        # Implement step
        engineer._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["implement"]
        await engineer._handle_task(impl_task, lock=impl_lock)

        # Check code_review chain task ID
        cr_result = _claim_chain_task(queue, "architect", "architect")
        assert cr_result is not None
        cr_task, _ = cr_result
        assert cr_task.id == "chain-root-001-code_review-d2"


class TestRejectionRoutesBackToEngineer:
    """Code review rejection bounces back to engineer, then re-approves."""

    @pytest.mark.asyncio
    async def test_rejection_routes_back_to_engineer(self, tmp_path):
        queue = FileQueue(tmp_path / "workspace")

        architect = _build_pipeline_agent(tmp_path, "architect", queue)
        engineer = _build_pipeline_agent(tmp_path, "engineer", queue)
        qa = _build_pipeline_agent(tmp_path, "qa", queue)

        seed = _make_seed_task()
        queue.push(seed, "architect")

        # -- Plan --
        await _run_step(architect, queue, "architect", "architect", "plan")

        # -- Implement --
        impl_result = _claim_chain_task(queue, "engineer", "engineer")
        assert impl_result is not None
        impl_task, impl_lock = impl_result
        engineer._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["implement"]
        await engineer._handle_task(impl_task, lock=impl_lock)

        # -- Code review: REJECT --
        rejection_response = _make_llm_response(
            content="Code review: CHANGES REQUESTED. Missing error handling in auth module."
        )
        cr_result = _claim_chain_task(queue, "architect", "architect")
        assert cr_result is not None
        cr_task, cr_lock = cr_result
        assert cr_task.context["workflow_step"] == "code_review"

        architect._execute_llm_with_interruption_watch.return_value = rejection_response
        await architect._handle_task(cr_task, lock=cr_lock)
        assert cr_task.status == TaskStatus.COMPLETED
        assert cr_task.context.get("verdict") == "needs_fix"

        # Should bounce back to engineer (implement step)
        fix_result = _claim_chain_task(queue, "engineer", "engineer")
        assert fix_result is not None, "Expected task in engineer queue after rejection"
        fix_task, fix_lock = fix_result
        assert fix_task.context["workflow_step"] == "implement"
        # Review cycle counter should be incremented
        assert fix_task.context.get("_dag_review_cycles", 0) >= 1

        # -- Engineer re-implements --
        engineer._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["implement"]
        await engineer._handle_task(fix_task, lock=fix_lock)
        assert fix_task.status == TaskStatus.COMPLETED

        # -- Code review: APPROVE --
        cr2_result = _claim_chain_task(queue, "architect", "architect")
        assert cr2_result is not None
        cr2_task, cr2_lock = cr2_result
        assert cr2_task.context["workflow_step"] == "code_review"

        architect._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["code_review"]
        await architect._handle_task(cr2_task, lock=cr2_lock)
        assert cr2_task.status == TaskStatus.COMPLETED
        assert cr2_task.context.get("verdict") == "approved"

        # Should proceed to qa_review
        qa_result = _claim_chain_task(queue, "qa", "qa")
        assert qa_result is not None, "Expected task in qa queue after second code review approval"
        qa_task, qa_lock = qa_result
        assert qa_task.context["workflow_step"] == "qa_review"

        # -- QA approves --
        qa._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["qa_review"]
        await qa._handle_task(qa_task, lock=qa_lock)
        assert qa_task.status == TaskStatus.COMPLETED

        # -- PR creation --
        pr_result = _claim_chain_task(queue, "architect", "architect")
        assert pr_result is not None
        pr_task, pr_lock = pr_result
        assert pr_task.context["workflow_step"] == "create_pr"

        architect._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["create_pr"]
        await architect._handle_task(pr_task, lock=pr_lock)
        assert pr_task.status == TaskStatus.COMPLETED

        architect._git_ops.push_and_create_pr_if_needed.assert_called()

        # No orphaned tasks
        assert _claim_chain_task(queue, "architect", "architect") is None
        assert _claim_chain_task(queue, "engineer", "engineer") is None
        assert _claim_chain_task(queue, "qa", "qa") is None


class TestContextPropagation:
    """Verify key context fields propagate correctly through the chain."""

    @pytest.mark.asyncio
    async def test_root_task_id_propagates(self, tmp_path):
        """_root_task_id and workflow name survive the full chain."""
        queue = FileQueue(tmp_path / "workspace")
        architect = _build_pipeline_agent(tmp_path, "architect", queue)

        seed = _make_seed_task(task_id="ctx-prop-001")
        queue.push(seed, "architect")

        # Plan
        await _run_step(architect, queue, "architect", "architect", "plan")

        # Check implement task
        impl_result = _claim_chain_task(queue, "engineer", "engineer")
        assert impl_result is not None
        impl_task, _ = impl_result
        assert impl_task.context["_root_task_id"] == "ctx-prop-001"
        assert impl_task.context["workflow"] == "default"
        assert impl_task.context["user_goal"] == seed.description

    @pytest.mark.asyncio
    async def test_global_cycle_count_increments(self, tmp_path):
        """_global_cycle_count increments at each non-preview step."""
        queue = FileQueue(tmp_path / "workspace")
        architect = _build_pipeline_agent(tmp_path, "architect", queue)
        engineer = _build_pipeline_agent(tmp_path, "engineer", queue)

        seed = _make_seed_task()
        queue.push(seed, "architect")

        # Plan
        await _run_step(architect, queue, "architect", "architect", "plan")

        impl_result = _claim_chain_task(queue, "engineer", "engineer")
        assert impl_result is not None
        impl_task, impl_lock = impl_result
        # plan→implement is depth 1, global cycle 1
        assert impl_task.context["_global_cycle_count"] == 1

        engineer._execute_llm_with_interruption_watch.return_value = STEP_RESPONSES["implement"]
        await engineer._handle_task(impl_task, lock=impl_lock)

        cr_result = _claim_chain_task(queue, "architect", "architect")
        assert cr_result is not None
        cr_task, _ = cr_result
        # implement→code_review is depth 2, global cycle 2
        assert cr_task.context["_global_cycle_count"] == 2
