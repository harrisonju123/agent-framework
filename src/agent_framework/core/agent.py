"""Agent polling loop implementation (ported from Bash system)."""

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .config import AgentDefinition, RepositoryConfig, WorkflowDefinition

from .task import Task, TaskStatus, TaskType
from .task_validator import validate_task, ValidationResult
from .activity import ActivityManager, AgentActivity, AgentStatus, CurrentTask, ActivityEvent, TaskPhase, ToolActivity
from .routing import read_routing_signal, validate_routing_signal, log_routing_decision, WORKFLOW_COMPLETE
from .team_composer import compose_default_team, compose_team
from .context_window_manager import ContextWindowManager
from .review_cycle import ReviewCycleManager, QAFinding, ReviewOutcome, MAX_REVIEW_CYCLES
from .git_operations import GitOperationsManager
from ..llm.base import LLMBackend, LLMRequest, LLMResponse
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler
from ..workspace.worktree_manager import WorktreeManager, WorktreeConfig
from ..utils.rich_logging import ContextLogger, setup_rich_logging
from ..utils.type_helpers import get_type_str
from ..memory.memory_store import MemoryStore
from ..memory.memory_retriever import MemoryRetriever
from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer
from ..memory.tool_pattern_store import ToolPatternStore
from .session_logger import SessionLogger, noop_logger
from .prompt_builder import PromptBuilder, PromptContext
from .workflow_router import WorkflowRouter
from .error_recovery import ErrorRecoveryManager
from .budget_manager import BudgetManager

# Optional sandbox imports (only used if Docker is available)
try:
    from ..sandbox import DockerExecutor, GoTestRunner, TestResult
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    DockerExecutor = None
    GoTestRunner = None
    TestResult = None


# logger = logging.getLogger(__name__)  # Removed: using self.logger instead

# Pause/resume signal file
PAUSE_SIGNAL_FILE = ".agent-communication/pause"

# Constants for optimization strategies
SUMMARY_CONTEXT_MAX_CHARS = 2000
SUMMARY_MAX_LENGTH = 500
BUDGET_WARNING_THRESHOLD = 1.3  # 30% over budget

# Model pricing (per 1M tokens, as of 2025-01)
MODEL_PRICING = {
    "haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
}

# Downstream agents get the correct task type for model selection
CHAIN_TASK_TYPES = {
    "engineer": TaskType.IMPLEMENTATION,
    "qa": TaskType.QA_VERIFICATION,
    "architect": TaskType.REVIEW,
}

# Review cycle management moved to review_cycle.py
# Importing for backward compatibility
# (QAFinding, ReviewOutcome already imported above)


@dataclass
class AgentConfig:
    """Agent configuration."""
    id: str
    name: str
    queue: str
    prompt: str
    poll_interval: int = 30
    max_retries: int = 5
    timeout: int = 1800
    # Sandbox configuration
    enable_sandbox: bool = False
    sandbox_image: str = "golang:1.22"
    sandbox_test_cmd: str = "go test ./..."
    max_test_retries: int = 2
    # Task validation
    validate_tasks: bool = True
    validation_mode: str = "warn"  # "warn" or "reject"

    @property
    def base_id(self) -> str:
        """Strip replica suffix (e.g., 'engineer-2' -> 'engineer')."""
        parts = self.id.rsplit("-", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else self.id


class Agent:
    """
    Agent with polling loop for processing tasks.

    Ported from scripts/async-agent-runner.sh with the main polling loop
    at lines 254-407.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
        queue: FileQueue,
        workspace: Path,
        jira_client=None,
        github_client=None,
        multi_repo_manager=None,
        jira_config=None,
        github_config=None,
        mcp_enabled: bool = False,
        optimization_config: Optional[dict] = None,
        worktree_manager: Optional[WorktreeManager] = None,
        agents_config: Optional["List[AgentDefinition]"] = None,
        team_mode_enabled: bool = False,
        team_mode_default_model: str = "sonnet",
        agent_definition: Optional["AgentDefinition"] = None,
        workflows_config: Optional[Dict[str, "WorkflowDefinition"]] = None,
        memory_config: Optional[dict] = None,
        self_eval_config: Optional[dict] = None,
        replan_config: Optional[dict] = None,
        session_logging_config: Optional[dict] = None,
        repositories_config: Optional[List["RepositoryConfig"]] = None,
        pr_lifecycle_config: Optional[dict] = None,
    ):
        self.config = config
        self.llm = llm
        self.queue = queue
        self.workspace = Path(workspace)
        self.jira_client = jira_client
        self.github_client = github_client
        self.multi_repo_manager = multi_repo_manager
        self.jira_config = jira_config
        self.github_config = github_config
        self._mcp_enabled = mcp_enabled
        self._running = False
        self._current_task_id: Optional[str] = None
        self.worktree_manager = worktree_manager

        # Team mode: use Claude Agent Teams for multi-agent workflows
        self._agents_config = agents_config or []
        self._agent_definition = agent_definition
        self._team_mode_enabled = team_mode_enabled
        self._team_mode_default_model = team_mode_default_model

        # Workflow chain definitions for automatic next-agent queuing
        self._workflows_config = workflows_config or {}

        # Setup rich logging (log_level passed from CLI via environment)
        import os
        log_level = os.environ.get("AGENT_LOG_LEVEL", "INFO")
        self.logger = setup_rich_logging(
            agent_id=config.id,
            workspace=workspace,
            log_level=log_level,
            use_file=True,
            use_json=False,
        )

        # Workflow DAG executor — uses agent's logger so routing shows in agent logs
        from ..workflow.executor import WorkflowExecutor
        self._workflow_executor = WorkflowExecutor(queue, queue.queue_dir, agent_logger=self.logger)

        # Workflow router: handles workflow chain enforcement and task decomposition
        self._workflow_router = WorkflowRouter(
            config=config,
            queue=queue,
            workspace=self.workspace,
            logger=self.logger,
            session_logger=None,  # Will be set later after session logger initialization
            workflows_config=self._workflows_config,
            workflow_executor=self._workflow_executor,
            agents_config=self._agents_config,
            multi_repo_manager=multi_repo_manager,
        )

        # Optimization configuration (sanitize then make immutable for thread safety)
        sanitized_config = self._sanitize_optimization_config(optimization_config or {})
        self._optimization_config = MappingProxyType(sanitized_config)

        # Log active optimizations on startup
        self.logger.info(f"🔧 Optimization config: {self._get_active_optimizations()}")

        # Initialize safeguards
        self.retry_handler = RetryHandler(max_retries=config.max_retries)
        enable_error_truncation = self._optimization_config.get("enable_error_truncation", False)
        self.escalation_handler = EscalationHandler(enable_error_truncation=enable_error_truncation)

        # Heartbeat file
        self.heartbeat_file = self.workspace / ".agent-communication" / "heartbeats" / config.id

        # Activity tracking
        self.activity_manager = ActivityManager(workspace)

        # Pause state tracking
        self._paused = False

        # Cache for team composition (fixed per agent lifetime / workflow)
        self._default_team_cache: Optional[dict] = None
        self._workflow_team_cache: Dict[str, Optional[dict]] = {}
        # Set per-task by PromptBuilder.build(), consumed by team composition
        self._current_specialization = None
        self._current_file_count = 0

        # Pause signal cache (avoid 2x exists() calls every 2s)
        self._pause_signal_cache: Optional[bool] = None
        self._pause_signal_cache_time: float = 0.0

        # Agent Memory System: persistent cross-task learning
        mem_cfg = memory_config or {}
        self._memory_enabled = mem_cfg.get("enabled", False)
        self._memory_store = MemoryStore(workspace, enabled=self._memory_enabled)
        self._memory_retriever = MemoryRetriever(self._memory_store)

        # Tool pattern analysis: detect and surface inefficient tool usage
        tool_tips_enabled = self._optimization_config.get("enable_tool_pattern_tips", False)
        self._tool_pattern_store = ToolPatternStore(workspace, enabled=tool_tips_enabled)
        self._tool_tips_enabled = tool_tips_enabled

        # Self-Evaluation Loop: review own output before marking done
        eval_cfg = self_eval_config or {}
        self._self_eval_enabled = eval_cfg.get("enabled", False)
        self._self_eval_max_retries = eval_cfg.get("max_retries", 2)
        self._self_eval_model = eval_cfg.get("model", "haiku")

        # Dynamic Replanning: generate revised plans on failure retry 2+
        replan_cfg = replan_config or {}
        self._replan_enabled = replan_cfg.get("enabled", False)
        self._replan_min_retry = replan_cfg.get("min_retry_for_replan", 2)
        self._replan_model = replan_cfg.get("model", "haiku")

        # Session logging: structured per-task JSONL for post-hoc analysis
        sl_cfg = session_logging_config or {}
        self._session_logging_enabled = sl_cfg.get("enabled", False)
        self._session_log_prompts = sl_cfg.get("log_prompts", True)
        self._session_log_tool_inputs = sl_cfg.get("log_tool_inputs", True)
        self._session_logs_dir = self.workspace / "logs"
        self._session_logger: SessionLogger = noop_logger()

        # Cleanup old session logs on startup
        retention_days = sl_cfg.get("retention_days", 30)
        if self._session_logging_enabled and retention_days > 0:
            SessionLogger.cleanup_old_sessions(self._session_logs_dir, retention_days)

        # Context window management: track token budgets and manage message history
        # per task (initialized when task starts)
        self._context_window_manager: Optional[ContextWindowManager] = None

        # Sandbox for isolated test execution
        self._test_runner = None
        if config.enable_sandbox and SANDBOX_AVAILABLE:
            try:
                executor = DockerExecutor(image=config.sandbox_image)
                if executor.health_check():
                    self._test_runner = GoTestRunner(executor=executor)
                    self.logger.info(f"Agent {config.id} sandbox enabled with image {config.sandbox_image}")
                else:
                    self.logger.warning(f"Docker not available, sandbox disabled for {config.id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize sandbox for {config.id}: {e}")

        # PR lifecycle management: autonomous CI poll → fix → merge
        self._pr_lifecycle_manager = None
        if repositories_config:
            from .pr_lifecycle import PRLifecycleManager
            repo_lookup = {rc.github_repo: rc.model_dump() for rc in repositories_config}
            if repo_lookup:
                self._pr_lifecycle_manager = PRLifecycleManager(
                    queue=queue,
                    workspace=workspace,
                    repo_configs=repo_lookup,
                    pr_lifecycle_config=pr_lifecycle_config,
                    logger_instance=self.logger,
                    multi_repo_manager=multi_repo_manager,
                )

        # Review cycle management: QA → Engineer feedback loop
        self._review_cycle = ReviewCycleManager(
            config=config,
            queue=queue,
            logger=self.logger,
            agent_definition=agent_definition,
            session_logger=self._session_logger,
            activity_manager=self.activity_manager,
        )

        # Prompt builder: extracted prompt construction logic
        prompt_ctx = PromptContext(
            config=config,
            workspace=workspace,
            mcp_enabled=mcp_enabled,
            jira_config=jira_config,
            github_config=github_config,
            agent_definition=agent_definition,
            optimization_config=sanitized_config,
            memory_retriever=self._memory_retriever,
            tool_pattern_store=self._tool_pattern_store,
            context_window_manager=None,  # Set per-task
            session_logger=None,  # Set per-task
            logger=self.logger,
            llm=llm,
            queue=queue,
            agent=self,
            workflows_config=workflows_config,
        )
        self._prompt_builder = PromptBuilder(prompt_ctx)

        # Git/PR/Worktree operations manager
        self._git_ops = GitOperationsManager(
            config=config,
            workspace=workspace,
            queue=queue,
            logger=self.logger,
            worktree_manager=worktree_manager,
            multi_repo_manager=multi_repo_manager,
            github_client=github_client,
            jira_client=jira_client,
            session_logger=self._session_logger,
            pr_lifecycle_manager=self._pr_lifecycle_manager,
            agent_definition=agent_definition,
            workflows_config=workflows_config,
        )

    # Backward-compatibility delegation shims for tests that call Agent._* methods
    # These delegate to ReviewCycleManager for cleaner architecture
    def _parse_review_outcome(self, content: str) -> ReviewOutcome:
        """Delegate to ReviewCycleManager.parse_review_outcome."""
        return self._review_cycle.parse_review_outcome(content)

    def _extract_review_findings(self, content: str):
        """Delegate to ReviewCycleManager.extract_review_findings."""
        return self._review_cycle.extract_review_findings(content)

    def _parse_structured_findings(self, content: str):
        """Delegate to ReviewCycleManager.parse_structured_findings."""
        return self._review_cycle.parse_structured_findings(content)

    def _format_findings_checklist(self, findings):
        """Delegate to ReviewCycleManager.format_findings_checklist."""
        return self._review_cycle.format_findings_checklist(findings)

    def _build_review_task(self, task: Task, pr_info: dict) -> Task:
        """Delegate to ReviewCycleManager.build_review_task."""
        return self._review_cycle.build_review_task(task, pr_info)

    def _build_review_fix_task(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> Task:
        """Delegate to ReviewCycleManager.build_review_fix_task."""
        return self._review_cycle.build_review_fix_task(task, outcome, cycle_count)

    def _escalate_review_to_architect(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> None:
        """Delegate to ReviewCycleManager.escalate_review_to_architect."""
        return self._review_cycle.escalate_review_to_architect(task, outcome, cycle_count)

    def _purge_orphaned_review_tasks(self) -> None:
        """Delegate to ReviewCycleManager.purge_orphaned_review_tasks."""
        return self._review_cycle.purge_orphaned_review_tasks()

    def _get_pr_info(self, task: Task, response):
        """Delegate to ReviewCycleManager.get_pr_info."""
        return self._review_cycle.get_pr_info(task, response)

    def _extract_pr_info_from_response(self, response_content: str):
        """Delegate to ReviewCycleManager.extract_pr_info_from_response."""
        return self._review_cycle.extract_pr_info_from_response(response_content)

    def _queue_code_review_if_needed(self, task: Task, response) -> None:
        """Delegate to ReviewCycleManager.queue_code_review_if_needed."""
        return self._review_cycle.queue_code_review_if_needed(task, response)

    def _queue_review_fix_if_needed(self, task: Task, response) -> None:
        """Delegate to ReviewCycleManager.queue_review_fix_if_needed (without sync callback)."""
        # Note: tests don't pass sync_jira_status_callback, so we use a no-op
        return self._review_cycle.queue_review_fix_if_needed(task, response, lambda *args, **kwargs: None)


        # Error recovery and budget management — extracted from Agent
        self._error_recovery = ErrorRecoveryManager(
            config=config,
            queue=queue,
            llm=llm,
            logger=self.logger,
            session_logger=self._session_logger,
            retry_handler=self.retry_handler,
            escalation_handler=self.escalation_handler,
            workspace=workspace,
            jira_client=jira_client,
            memory_store=self._memory_store,
            replan_config=replan_config,
            self_eval_config=self_eval_config,
        )

        self._budget = BudgetManager(
            agent_id=config.id,
            optimization_config=dict(self._optimization_config),
            logger=self.logger,
            session_logger=self._session_logger,
            llm=llm,
            workspace=workspace,
            activity_manager=self.activity_manager,
        )

    async def run(self) -> None:
        """
        Main polling loop.

        Ported from scripts/async-agent-runner.sh lines 254-407.
        """
        self._running = True
        self.logger.info(f"🚀 Starting {self.config.id} runner")

        # Write initial IDLE state when agent starts
        from datetime import datetime
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        ))

        # Drain stale review-chain tasks left over from before the cycle-count guard
        self._review_cycle.purge_orphaned_review_tasks()

        while self._running:
            # Write heartbeat every iteration
            self._write_heartbeat()

            # Check for pause signal before processing tasks
            if self._check_pause_signal():
                if not self._paused:
                    self.logger.info(f"Agent {self.config.id} paused")
                    self._paused = True
                    # Update activity to show paused state
                    self.activity_manager.update_activity(AgentActivity(
                        agent_id=self.config.id,
                        status=AgentStatus.IDLE,
                        last_updated=datetime.now(timezone.utc)
                    ))
                await asyncio.sleep(self.config.poll_interval)
                continue

            if self._paused:
                self.logger.info(f"Agent {self.config.id} resumed")
                self._paused = False

            # Poll for next task
            task = self.queue.pop(self.config.queue)

            if task:
                await self._handle_task(task)
            else:
                self.logger.debug(
                    f"No tasks available for {self.config.id}, "
                    f"sleeping for {self.config.poll_interval}s"
                )

            await asyncio.sleep(self.config.poll_interval)

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self.logger.info(f"Stopping {self.config.id}")
        self._running = False

        # Kill any in-flight LLM subprocess so the agent doesn't block
        self.llm.cancel()

        # Release current task lock if any
        if self._current_task_id:
            self.logger.warning(
                f"Releasing lock for current task: {self._current_task_id}"
            )
            # Lock will be automatically released by FileLock context manager

        # Write final heartbeat
        self._write_heartbeat()

    def _validate_task_or_reject(self, task: Task) -> bool:
        """
        Validate task and reject if invalid.

        Returns:
            True if task is valid and should be processed, False if rejected
        """
        if not self.config.validate_tasks:
            return True

        validation = validate_task(task, mode=self.config.validation_mode)
        if validation.skipped:
            return True

        if validation.warnings:
            for warning in validation.warnings:
                self.logger.warning(f"Task validation warning: {warning}")

        if validation.errors:
            for error in validation.errors:
                self.logger.error(f"Task validation error: {error}")

        if not validation.is_valid:
            self.logger.error(f"Task {task.id} rejected due to validation errors")
            error_msg = f"Task validation failed: {'; '.join(validation.errors)}"
            task.last_error = error_msg
            task.mark_failed(self.config.id, error_message=error_msg, error_type="validation")
            self.queue.mark_failed(task)
            return False

        return True

    def _initialize_task_execution(self, task: Task, task_start_time) -> None:
        """Initialize task execution with activity tracking and events."""
        from datetime import datetime

        task.mark_in_progress(self.config.id)
        self.queue.update(task)

        # Update activity: Started
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.WORKING,
            current_task=CurrentTask(
                id=task.id,
                title=task.title,
                type=get_type_str(task.type),
                started_at=task_start_time
            ),
            current_phase=TaskPhase.ANALYZING,
            last_updated=datetime.now(timezone.utc)
        ))

        # Append start event
        self.activity_manager.append_event(ActivityEvent(
            type="start",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc)
        ))

        # Deterministic JIRA transition on task start
        if self._agent_definition and self._agent_definition.jira_on_start:
            self._sync_jira_status(task, self._agent_definition.jira_on_start)

    async def _handle_successful_response(self, task: Task, response, task_start_time) -> None:
        """Handle successful LLM response including tests, workflow, and completion."""
        from datetime import datetime, timezone

        # Extract summary from response
        if self._optimization_config.get("enable_result_summarization", False):
            summary = await self._extract_summary(response.content, task)
            task.result_summary = summary
            self.logger.debug(f"Task {task.id} summary: {summary}")

        # Run tests in sandbox if enabled
        test_result = await self._run_sandbox_tests(task)
        if test_result and not test_result.success:
            test_retry = task.context.get("_test_retry_count", 0)
            if test_retry < self.config.max_test_retries:
                self.logger.warning(
                    f"Tests failed for {task.id} (attempt {test_retry + 1}/{self.config.max_test_retries})"
                )
                await self._handle_test_failure(task, response, test_result)
                return
            else:
                self.logger.error(f"Tests still failing after {test_retry} retries for {task.id}")
                task.last_error = f"Tests failed: {test_result.error_message}"
                await self._handle_failure(task)
                return

        # Self-evaluation: catch obvious issues before propagating downstream
        if self._self_eval_enabled:
            passed = await self._self_evaluate(task, response)
            if not passed:
                return  # Task was reset for self-eval retry

        # Save upstream context after validation passes, before workflow chain
        if task.context.get("workflow") or task.context.get("chain_step"):
            self._save_upstream_context(task, response)

        # Handle post-LLM workflow
        self.logger.debug(f"Running post-LLM workflow for {task.id}")
        await self._handle_success(task, response)

        # Mark completed
        self.logger.debug(f"Marking task {task.id} as completed")
        task.mark_completed(self.config.id)
        self.queue.mark_completed(task)
        self.logger.info(f"✅ Task {task.id} moved to completed")

        # Deterministic JIRA transition on task completion
        if self._agent_definition and self._agent_definition.jira_on_complete:
            comment = f"Task completed by {self.config.id}"
            pr_url = task.context.get("pr_url")
            if pr_url:
                comment += f"\nPR: {pr_url}"
            self._sync_jira_status(task, self._agent_definition.jira_on_complete, comment=comment)

        # Transition to COMPLETING status
        try:
            self.activity_manager.update_activity(AgentActivity(
                agent_id=self.config.id,
                status=AgentStatus.COMPLETING,
                current_task=CurrentTask(
                    id=task.id,
                    title=task.title,
                    type=get_type_str(task.type),
                    started_at=task_start_time
                ),
                last_updated=datetime.now(timezone.utc)
            ))
            await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            pass

        routing_signal = read_routing_signal(self.workspace, task.id)
        if routing_signal:
            self.logger.info(
                f"Routing signal: target={routing_signal.target_agent}, "
                f"reason={routing_signal.reason}"
            )

        self._run_post_completion_flow(task, response, routing_signal, task_start_time)

    def _run_post_completion_flow(self, task: Task, response, routing_signal, task_start_time) -> None:
        """Route completed task through workflow chain, collect metrics.

        Subtasks (parent_task_id set) skip the workflow chain — the fan-in
        task aggregates results and flows through QA/review/PR instead.
        """
        # Fan-in check: if this is a subtask, check if all siblings are done
        self._workflow_router.check_and_create_fan_in_task(task)

        # Subtasks wait for fan-in — don't route them individually through
        # the workflow chain. The fan-in task handles QA/review/PR creation.
        if task.parent_task_id is not None:
            self.logger.debug(
                f"Subtask {task.id} complete — skipping workflow chain "
                f"(fan-in will handle routing)"
            )
        else:
            # Legacy direct-queue routing only for tasks without a workflow DAG.
            # Workflow-managed tasks route through _enforce_workflow_chain() exclusively.
            has_workflow = bool(task.context.get("workflow"))
            if not has_workflow:
                self.logger.debug(f"Checking if code review needed for {task.id}")
                self._review_cycle.queue_code_review_if_needed(task, response)
                self._review_cycle.queue_review_fix_if_needed(task, response, self._sync_jira_status)
            self._enforce_workflow_chain(task, response, routing_signal=routing_signal)

            # Safety net: create PR if LLM pushed but didn't create one.
            # Runs AFTER workflow chain so pr_url doesn't short-circuit the executor.
            self._git_ops.push_and_create_pr_if_needed(task)

            # Autonomous PR lifecycle: poll CI, fix failures, merge
            self._git_ops.manage_pr_lifecycle(task)

        self._extract_and_store_memories(task, response)
        self._analyze_tool_patterns(task)
        self._log_task_completion_metrics(task, response, task_start_time)

    def _log_task_completion_metrics(self, task: Task, response, task_start_time) -> None:
        """Log token usage, cost, and completion events.

        Delegated to BudgetManager.
        """
        self._budget.log_task_completion_metrics(task, response, task_start_time)

    async def _handle_failed_response(self, task: Task, response) -> None:
        """Handle failed LLM response."""
        from datetime import datetime

        task.last_error = response.error or "Unknown error"

        # Log detailed error information for debugging
        self.logger.error(
            f"Task failed with detailed error:\n"
            f"  Task ID: {task.id}\n"
            f"  Error: {task.last_error}\n"
            f"  Model: {response.model_used}\n"
            f"  Latency: {response.latency_ms:.0f}ms\n"
            f"  Finish reason: {response.finish_reason}"
        )
        self.logger.task_failed(task.last_error, task.retry_count)

        self._session_logger.log(
            "task_failed",
            error=task.last_error,
            retry=task.retry_count,
            model=response.model_used,
            finish_reason=response.finish_reason,
        )

        self.activity_manager.append_event(ActivityEvent(
            type="fail",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc),
            retry_count=task.retry_count,
            error_message=task.last_error
        ))

        await self._handle_failure(task)

    def _cleanup_task_execution(self, task: Task, lock) -> None:
        """Cleanup after task execution."""
        from datetime import datetime

        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        ))

        task_succeeded = task.status == TaskStatus.COMPLETED
        self._git_ops.sync_worktree_queued_tasks()
        self._git_ops.cleanup_worktree(task, success=task_succeeded)

        if lock:
            self.queue.release_lock(lock)
        self._current_task_id = None

    async def _watch_for_interruption(self) -> None:
        """Poll for pause/stop signals during LLM execution.

        Completes (returns) when an interruption is detected, which causes
        the asyncio.wait race in _handle_task to cancel the LLM call.
        """
        while self._running and not self._check_pause_signal():
            await asyncio.sleep(2)

    @staticmethod
    def _normalize_workflow(task: Task) -> None:
        """Map legacy workflow names to 'default'.

        Old tasks in queues may have 'simple', 'standard', or 'full'.
        Normalize them so the rest of the pipeline only sees 'default'.
        """
        if not task.context:
            return
        workflow = task.context.get("workflow")
        if workflow in ("simple", "standard", "full"):
            task.context["workflow"] = "default"

    async def _handle_task(self, task: Task) -> None:
        """Handle task execution with retry/escalation logic."""
        from datetime import datetime

        # Normalize legacy workflow names before anything reads them
        self._normalize_workflow(task)

        # Set task context for logging
        jira_key = task.context.get("jira_key")
        self.logger.task_started(task.id, task.title, jira_key=jira_key)

        # Validate task
        if not self._validate_task_or_reject(task):
            return

        # Acquire lock
        lock = self.queue.acquire_lock(task.id, self.config.id)
        if not lock:
            self.logger.warning(f"⏸️  Could not acquire lock, will retry later")
            return

        self._current_task_id = task.id
        task_start_time = datetime.now(timezone.utc)

        # Session logger: structured JSONL for post-hoc analysis
        self._session_logger = SessionLogger(
            logs_dir=self._session_logs_dir,
            task_id=task.id,
            enabled=self._session_logging_enabled,
            log_prompts=self._session_log_prompts,
            log_tool_inputs=self._session_log_tool_inputs,
        )
        # Update workflow router's session logger for this task
        self._workflow_router.set_session_logger(self._session_logger)
        self._session_logger.log(
            "task_start",
            agent=self.config.id,
            title=task.title,
            retry=task.retry_count,
            task_type=get_type_str(task.type),
        )

        # Initialize context window manager for this task
        task_budget = self._get_token_budget(task.type)
        ctx_cfg = self._optimization_config.get("context_window", {})
        if not isinstance(ctx_cfg, dict):
            ctx_cfg = ctx_cfg.model_dump() if hasattr(ctx_cfg, "model_dump") else {}
        self._context_window_manager = ContextWindowManager(
            total_budget=task_budget,
            output_reserve=ctx_cfg.get("output_reserve", 4096),
            summary_threshold=ctx_cfg.get("summary_threshold", 10),
            min_message_retention=ctx_cfg.get("min_message_retention", 3),
        )
        self.logger.debug(
            f"Context window manager initialized: budget={task_budget}, "
            f"available_for_input={self._context_window_manager.budget.available_for_input}"
        )

        try:
            # Initialize task execution
            self._initialize_task_execution(task, task_start_time)

            # Get working directory for task (worktree, target repo, or framework workspace)
            working_dir = self._git_ops.get_working_directory(task)
            self.logger.info(f"Working directory: {working_dir}")

            # Build prompt and execute LLM
            self.logger.phase_change("analyzing")
            # Update prompt builder with per-task context
            self._prompt_builder.ctx.session_logger = self._session_logger
            self._prompt_builder.ctx.context_window_manager = self._context_window_manager
            prompt = self._prompt_builder.build(task)
            # Get specialization data from prompt builder (set during build)
            self._current_specialization = self._prompt_builder.get_current_specialization()
            self._current_file_count = self._prompt_builder.get_current_file_count()

            # Update activity with specialization (if detected)
            if self._current_specialization:
                activity = self.activity_manager.get_activity(self.config.id)
                if activity:
                    activity.specialization = self._current_specialization.id
                    self.activity_manager.update_activity(activity)

            self._update_phase(TaskPhase.EXECUTING_LLM)
            self.logger.phase_change("executing_llm")
            self.logger.info(
                f"🤖 Calling LLM (model: {task.type}, attempt: {task.retry_count + 1})"
            )

            self._session_logger.log(
                "llm_start",
                task_type=get_type_str(task.type),
                retry=task.retry_count,
            )

            # Throttled callback so tool activity writes hit disk at most once/sec
            _tool_call_count = [0]
            _last_write_time = [0.0]

            def _on_tool_activity(tool_name: str, tool_input_summary: Optional[str]):
                try:
                    _tool_call_count[0] += 1
                    now = time.time()
                    if now - _last_write_time[0] < 1.0:
                        return
                    _last_write_time[0] = now
                    ta = ToolActivity(
                        tool_name=tool_name,
                        tool_input_summary=tool_input_summary,
                        started_at=datetime.now(timezone.utc),
                        tool_call_count=_tool_call_count[0],
                    )
                    self.activity_manager.update_tool_activity(self.config.id, ta)
                except Exception as e:
                    self.logger.debug(f"Tool activity tracking error (non-fatal): {e}")

            # Compose team if team mode is enabled
            team_agents = None
            team_override = task.context.get("team_override")
            if self._team_mode_enabled and team_override is not False:
                team_agents = {}

                # Layer 1: agent's configured teammates
                # For engineer agents, teammates vary by task specialization (no cache)
                # For other agents, teammates are fixed (use cache)
                if self._agent_definition and self._agent_definition.teammates:
                    if self.config.base_id == "engineer":
                        default_team = compose_default_team(
                            self._agent_definition,
                            default_model=self._team_mode_default_model,
                            specialization_profile=self._current_specialization,
                        ) or {}
                        if default_team:
                            team_agents.update(default_team)
                    else:
                        if self._default_team_cache is None:
                            self._default_team_cache = compose_default_team(
                                self._agent_definition,
                                default_model=self._team_mode_default_model,
                            ) or {}
                        if self._default_team_cache:
                            team_agents.update(self._default_team_cache)

                # Layer 2: workflow-required agents (cached per workflow type)
                workflow = task.context.get("workflow", "default")
                if workflow not in self._workflow_team_cache:
                    self._workflow_team_cache[workflow] = compose_team(
                        task.context, workflow, self._agents_config,
                        default_model=self._team_mode_default_model,
                        caller_agent_id=self.config.id,
                    )
                workflow_teammates = self._workflow_team_cache[workflow]
                if workflow_teammates:
                    collisions = sorted(set(team_agents) & set(workflow_teammates))
                    if collisions:
                        self.logger.warning(
                            f"Teammate ID collision: {collisions} - workflow agents take precedence"
                        )
                    team_agents.update(workflow_teammates)

                team_agents = team_agents or None
                if team_agents:
                    self.logger.info(f"Team mode: {list(team_agents.keys())}")
            elif team_override is False:
                self.logger.debug("Team mode skipped via task team_override=False")

            # Race LLM execution against pause/stop signal watcher so we can
            # interrupt mid-task instead of waiting 30+ minutes for completion
            llm_coro = self.llm.complete(
                LLMRequest(
                    prompt=prompt,
                    task_type=task.type,
                    retry_count=task.retry_count,
                    context=task.context,
                    working_dir=str(working_dir),
                    agents=team_agents,
                    specialization_profile=self._current_specialization.id if self._current_specialization else None,
                    file_count=self._current_file_count,
                ),
                task_id=task.id,
                on_tool_activity=_on_tool_activity,
                on_session_tool_call=self._session_logger.log_tool_call,
            )
            llm_task = asyncio.create_task(llm_coro)
            watcher_task = asyncio.create_task(self._watch_for_interruption())

            done, pending = await asyncio.wait(
                [llm_task, watcher_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if watcher_task in done:
                # Pause or stop detected mid-task — kill LLM and reset task
                self.logger.info(f"Interruption detected during task {task.id}, cancelling LLM")
                self.llm.cancel()
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.debug(f"LLM task raised during cancellation: {e}")

                task.reset_to_pending()
                self.queue.update(task)

                self.activity_manager.append_event(ActivityEvent(
                    type="interrupted",
                    agent=self.config.id,
                    task_id=task.id,
                    title=task.title,
                    timestamp=datetime.now(timezone.utc),
                ))
                self.logger.info(f"Task {task.id} reset to pending after interruption")
                return
            else:
                # LLM finished first — cancel the watcher and proceed normally
                watcher_task.cancel()
                try:
                    await watcher_task
                except asyncio.CancelledError:
                    pass
                response = llm_task.result()

            # Log LLM completion to session log
            self._session_logger.log(
                "llm_complete",
                success=response.success,
                model=response.model_used,
                tokens_in=response.input_tokens,
                tokens_out=response.output_tokens,
                cost=response.reported_cost_usd,
                duration_ms=response.latency_ms,
            )

            # Update context window manager with actual token usage
            if self._context_window_manager:
                self._context_window_manager.update_token_usage(
                    response.input_tokens,
                    response.output_tokens
                )
                budget_status = self._context_window_manager.get_budget_status()
                self.logger.debug(
                    f"Context budget: {budget_status['utilization_percent']:.1f}% used "
                    f"({budget_status['used_so_far']}/{budget_status['total_budget']} tokens)"
                )

                # Check if we should trigger a checkpoint due to budget exhaustion
                if self._context_window_manager.should_trigger_checkpoint():
                    self.logger.warning(
                        f"Context budget critically low (>90% used). "
                        f"Consider splitting task into subtasks."
                    )
                    self.activity_manager.append_event(ActivityEvent(
                        type="context_budget_critical",
                        agent=self.config.id,
                        task_id=task.id,
                        title=f"Context budget >90%: consider task splitting",
                        timestamp=datetime.now(timezone.utc)
                    ))

            # Clear tool activity after LLM completes
            try:
                self.activity_manager.update_tool_activity(self.config.id, None)
            except Exception:
                pass

            # Handle response
            if response.success:
                await self._handle_successful_response(task, response, task_start_time)
            else:
                await self._handle_failed_response(task, response)

        except Exception as e:
            task.last_error = str(e)
            self.logger.exception(f"Error processing task {task.id}: {e}")

            self._session_logger.log(
                "task_failed",
                error=str(e),
                retry=task.retry_count,
            )

            self.activity_manager.append_event(ActivityEvent(
                type="fail",
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.now(timezone.utc),
                retry_count=task.retry_count,
                error_message=task.last_error
            ))

            await self._handle_failure(task)

        finally:
            self._context_window_manager = None
            self._session_logger.close()
            self._session_logger = noop_logger()
            self._cleanup_task_execution(task, lock)

    async def _handle_failure(self, task: Task) -> None:
        """
        Handle task failure with retry/escalation logic.

        Delegated to ErrorRecoveryManager.
        """
        await self._error_recovery.handle_failure(task)


    def _sanitize_optimization_config(self, config: dict) -> dict:
        """
        Sanitize optimization config before making immutable.

        Validates and corrects invalid values, warns about issues.
        """
        config = config.copy()  # Don't modify input

        # Clamp canary percentage to valid range
        canary = config.get("canary_percentage", 0)
        if not 0 <= canary <= 100:
            self.logger.warning(f"Invalid canary_percentage: {canary}, clamping to [0, 100]")
            config["canary_percentage"] = max(0, min(100, canary))

        # Warn about incompatible flag combinations
        if config.get("shadow_mode") and config.get("canary_percentage", 0) > 0:
            self.logger.warning(
                "shadow_mode and canary_percentage both enabled. "
                "Shadow mode will use legacy prompts regardless of canary setting."
            )

        return config

    def _get_active_optimizations(self) -> Dict[str, Any]:
        """Get dict of which optimizations are currently active."""
        return {
            "minimal_prompts": self._optimization_config.get("enable_minimal_prompts", False),
            "compact_json": self._optimization_config.get("enable_compact_json", False),
            "context_dedup": self._optimization_config.get("enable_context_deduplication", False),
            "token_tracking": self._optimization_config.get("enable_token_tracking", False),
            "budget_warnings": self._optimization_config.get("enable_token_budget_warnings", False),
            "result_summarization": self._optimization_config.get("enable_result_summarization", False),
            "error_truncation": self._optimization_config.get("enable_error_truncation", False),
            "shadow_mode": self._optimization_config.get("shadow_mode", False),
            "canary_percentage": self._optimization_config.get("canary_percentage", 0),
        }

    def _should_use_optimization(self, task: Task) -> bool:
        """
        Determine if task should use optimizations based on canary percentage.

        Uses deterministic hash-based selection for consistent behavior.
        Task-level overrides take precedence over canary selection.
        """
        # Check for task-level override first
        if hasattr(task, 'optimization_override') and task.optimization_override is not None:
            reason = getattr(task, 'optimization_override_reason', 'no reason given')
            self.logger.info(
                f"Task {task.id} optimization override: {task.optimization_override} ({reason})"
            )
            return task.optimization_override

        canary_pct = self._optimization_config.get("canary_percentage", 0)

        if canary_pct == 0:
            return False
        elif canary_pct >= 100:
            return True
        else:
            # Use deterministic hash for consistent selection
            task_hash = int(hashlib.md5(task.id.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            return (task_hash % 100) < canary_pct

    def _get_token_budget(self, task_type: TaskType) -> int:
        """Get expected token budget for task type.

        Delegated to BudgetManager.
        """
        return self._budget.get_token_budget(task_type)

    async def _extract_summary(self, response: str, task: Task, _recursion_depth: int = 0) -> str:
        """
        Extract key outcomes from agent response.

        Implements Strategy 5 (Result Summarization) from the optimization plan.

        Uses two-tier extraction:
        1. Regex patterns (free, fast) - extracts JIRA keys, PR URLs, file paths
        2. Haiku fallback (cheap) - only if regex insufficient

        Args:
            _recursion_depth: Defensive guard to prevent recursion. Currently
                            not used since we don't recurse, but kept for safety.

        Expected savings: 20-30% on follow-up tasks through context reuse.
        """
        # Guard against empty response
        if not response or not response.strip():
            return f"Task {get_type_str(task.type)} completed (no output)"

        # Defensive guard - prevents recursion even though we don't recurse currently
        if _recursion_depth > 0:
            self.logger.debug("Recursion depth exceeded in summary extraction, using fallback")
            return f"Task {get_type_str(task.type)} completed"

        # Try regex extraction first (fast, no cost)
        extracted = []

        # Extract JIRA keys: project prefix must NOT be a known non-JIRA
        # acronym (HTTP, UTF, ISO, etc.) and must be followed by a digit-only ticket number
        jira_keys = [
            m for m in re.findall(r'\b([A-Z]{2,5}-\d{1,6})\b', response)
            if not re.match(r'^(?:HTTP|UTF|ISO|RFC|TCP|UDP|SSH|SSL|TLS|DNS|API|URL|URI|XML|CSV|PDF)-', m)
        ]
        if jira_keys:
            # Deduplicate and limit
            jira_keys = list(set(jira_keys))[:10]
            extracted.append(f"Created/Updated: {', '.join(jira_keys)}")

        # Extract PR URLs
        pr_urls = re.findall(r'github\.com/[^/]+/[^/]+/pull/(\d+)', response)
        if pr_urls:
            pr_urls = list(set(pr_urls))[:5]
            extracted.append(f"PRs: {', '.join(pr_urls)}")

        # Extract file paths (comprehensive pattern)
        file_paths = re.findall(
            r'\b(?:src|lib|app|tests?|pkg|internal|cmd)/[^\s:;,]+\.(?:py|ts|tsx|js|jsx|go|rb|java|kt)',
            response
        )
        if file_paths:
            # Deduplicate and limit
            file_paths = list(set(file_paths))[:5]
            extracted.append(f"Modified: {', '.join(file_paths)}")

        # If regex extraction found enough, use that
        if len(extracted) >= 2:
            return " | ".join(extracted)

        # Fall back to Haiku for summarization (only if enabled and insufficient data)
        if self._optimization_config.get("enable_result_summarization", False):
            summary_prompt = f"Summarize key outcomes in 3 bullet points:\n{response[:SUMMARY_CONTEXT_MAX_CHARS]}"
            try:
                summary_response = await self.llm.complete(LLMRequest(
                    prompt=summary_prompt,
                    model="haiku"  # Cheap model for summaries
                ))

                # Verify response is valid before returning
                if summary_response.success and summary_response.content:
                    return summary_response.content[:SUMMARY_MAX_LENGTH]
                else:
                    self.logger.warning(f"Haiku summary failed: {summary_response.error}")
            except Exception as e:
                self.logger.warning(f"Failed to extract summary with Haiku: {e}")

        # Guaranteed fallback
        return extracted[0] if extracted else f"Task {get_type_str(task.type)} completed"

    # -- Upstream Context Handoff --

    UPSTREAM_CONTEXT_MAX_CHARS = 15000

    def _save_upstream_context(self, task: Task, response) -> None:
        """Save agent's response to disk so downstream agents can read it.

        Creates .agent-context/summaries/{task_id}-{agent_id}.md with
        the response content (truncated to UPSTREAM_CONTEXT_MAX_CHARS).

        Note: path mirrors FrameworkConfig.context_dir default.
        """
        try:
            from ..utils.atomic_io import atomic_write_text

            summaries_dir = self.workspace / ".agent-context" / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)

            content = response.content or ""
            if len(content) > self.UPSTREAM_CONTEXT_MAX_CHARS:
                content = content[:self.UPSTREAM_CONTEXT_MAX_CHARS] + "\n\n[truncated]"

            context_file = summaries_dir / f"{task.id}-{self.config.base_id}.md"
            atomic_write_text(context_file, content)

            # Store path in task context for chain propagation
            task.context["upstream_context_file"] = str(context_file)
            # Store inline for cross-worktree portability (file path may not resolve)
            task.context["upstream_summary"] = content[:4000]
            self.logger.debug(f"Saved upstream context ({len(content)} chars) to {context_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save upstream context for {task.id}: {e}")

    def _get_repo_slug(self, task: Task) -> Optional[str]:
        """Extract repo slug from task context."""
        return task.context.get("github_repo")

    def _build_replan_memory_context(self, task: Task) -> str:
        """Build memory context specifically for replanning.

        Prioritizes categories that help with task recovery:
        - conventions: coding standards, patterns to follow
        - test_commands: how to run/fix tests
        - repo_structure: where key files live

        Returns empty string if memory disabled or no relevant memories found.
        """
        if not self._memory_enabled:
            return ""

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return ""

        # Prioritize categories useful for recovery
        priority_categories = ["conventions", "test_commands", "repo_structure"]
        memories = []

        for category in priority_categories:
            category_memories = self._memory_store.recall(
                repo_slug=repo_slug,
                agent_type=self.config.base_id,
                category=category,
                limit=5,
            )
            memories.extend(category_memories)

        if not memories:
            return ""

        # Format as a context section
        lines = ["\n## Relevant Context from Previous Work"]
        lines.append("You've worked on this repo before. Here's what you know:\n")

        for mem in memories[:10]:  # Cap at 10 total memories
            lines.append(f"- [{mem.category}] {mem.content}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def _extract_and_store_memories(self, task: Task, response) -> None:
        """Extract learnings from successful response and store as memories."""
        if not self._memory_enabled:
            return

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return

        count = self._memory_retriever.extract_memories_from_response(
            response_content=response.content,
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            task_id=task.id,
        )
        if count > 0:
            self.logger.info(f"Extracted {count} memories from task {task.id}")
            self._session_logger.log(
                "memory_store",
                repo=repo_slug,
                count=count,
            )

    # -- Tool Pattern Analysis --

    def _analyze_tool_patterns(self, task: Task) -> None:
        """Run post-task analysis on session log to detect inefficient tool usage."""
        if not self._tool_tips_enabled or not self._session_logging_enabled:
            return

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return

        session_path = self._session_logs_dir / "sessions" / f"{task.id}.jsonl"
        if not session_path.exists():
            return

        try:
            analyzer = ToolPatternAnalyzer()
            recommendations = analyzer.analyze_session(session_path)
            if recommendations:
                count = self._tool_pattern_store.store_patterns(repo_slug, recommendations)
                self.logger.debug(f"Stored {count} tool pattern recommendations")
                self._session_logger.log(
                    "tool_patterns_analyzed",
                    repo=repo_slug,
                    patterns_found=len(recommendations),
                    patterns_stored=count,
                )
        except Exception as e:
            self.logger.debug(f"Tool pattern analysis failed (non-fatal): {e}")

    async def _self_evaluate(self, task: Task, response) -> bool:
        """Review agent's own output against acceptance criteria.

        Delegated to ErrorRecoveryManager.
        """
        return await self._error_recovery.self_evaluate(task, response)

    # -- Dynamic Replanning --

    async def _request_replan(self, task: Task) -> None:
        """Generate a revised approach based on what failed.

        Delegated to ErrorRecoveryManager.
        """
        await self._error_recovery.request_replan(task)

    def _categorize_error(self, error_message: str) -> Optional[str]:
        """Categorize error message for better diagnostics.

        Delegates to EscalationHandler.categorize_error for consistent
        pattern matching across the codebase.
        """
        return self.escalation_handler.categorize_error(error_message)
    def _inject_replan_context(self, prompt: str, task: Task) -> str:
        """Append revised plan and attempt history to prompt if available.

        Delegated to ErrorRecoveryManager.
        """
        return self._error_recovery.inject_replan_context(prompt, task)

    def _inject_preview_mode(self, prompt: str, task: Task) -> str:
        """Inject preview mode constraints when task is a preview."""
        preview_section = """
## PREVIEW MODE — READ-ONLY EXECUTION
You are in PREVIEW MODE. You must plan your implementation WITHOUT writing any files.

CONSTRAINTS:
- Do NOT use Write, Edit, or NotebookEdit tools
- Do NOT use Bash to create, modify, or delete files
- DO use Read, Glob, Grep, and Bash (read-only commands like git log, git diff, ls) to explore
- DO read every file you plan to modify to understand current state

REQUIRED OUTPUT — Produce a structured execution preview:

### Files to Modify
For each file, list:
- File path
- What changes will be made (specific, not vague)
- Estimated lines added/removed

### New Files to Create
For each new file:
- File path
- Purpose
- Key contents/structure
- Estimated line count

### Implementation Approach
- Step-by-step plan with ordering
- Which patterns from existing code will be followed
- Any dependencies between changes

### Risks and Edge Cases
- What could go wrong
- Edge cases to handle
- Backward compatibility concerns

### Estimated Total Change Size
- Total lines added/removed
- Number of files affected

This preview will be reviewed by the architect before implementation is authorized.
"""
        return preview_section + "\n\n" + prompt


    def _record_optimization_metrics(
        self,
        task: Task,
        legacy_prompt_length: int,
        optimized_prompt_length: int
    ) -> None:
        """
        Record optimization metrics for post-deployment analysis.

        Writes metrics to .agent-communication/metrics/optimization.jsonl
        for later analysis of optimization effectiveness.
        """
        try:
            metrics = {
                "task_id": task.id,
                "task_type": get_type_str(task.type),
                "agent_id": self.config.id,
                "legacy_prompt_chars": legacy_prompt_length,
                "optimized_prompt_chars": optimized_prompt_length,
                "savings_chars": legacy_prompt_length - optimized_prompt_length,
                "savings_percent": (
                    (legacy_prompt_length - optimized_prompt_length) / legacy_prompt_length * 100
                    if legacy_prompt_length > 0 else 0
                ),
                "canary_active": self._should_use_optimization(task),
                "optimizations_enabled": self._get_active_optimizations(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Write to metrics file for later analysis
            metrics_dir = self.workspace / ".agent-communication" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "optimization.jsonl"

            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except PermissionError as e:
            self.logger.warning(f"Permission denied recording optimization metrics: {e}")
        except OSError as e:
            self.logger.warning(f"Failed to record optimization metrics (disk full?): {e}")
        except Exception as e:
            # Don't fail task if metrics recording fails
            self.logger.debug(f"Unexpected error recording optimization metrics: {e}")

    def _estimate_cost(self, response: LLMResponse) -> float:
        """Estimate cost based on model and token usage.

        Delegated to BudgetManager.
        """
        return self._budget.estimate_cost(response)


    def _update_phase(self, phase: TaskPhase):
        """Update current execution phase."""
        from datetime import datetime

        activity = self.activity_manager.get_activity(self.config.id)
        if activity:
            activity.current_phase = phase
            self.activity_manager.update_activity(activity)

            # Append phase event
            if activity.current_task:
                self.activity_manager.append_event(ActivityEvent(
                    type="phase",
                    agent=self.config.id,
                    task_id=activity.current_task.id,
                    title=activity.current_task.title,
                    timestamp=datetime.now(timezone.utc),
                    phase=phase
                ))

    async def _handle_success(self, task: Task, llm_response) -> None:
        """
        Handle post-LLM workflow: git operations, PR creation, JIRA updates.

        DEPRECATED: When MCPs are enabled, agents handle this during execution.
        This method remains for backward compatibility.
        """
        # Skip if MCPs are enabled - agents do this during execution now
        if self._mcp_enabled:
            self.logger.debug("MCPs enabled - skipping post-LLM workflow")
            return

        jira_key = task.context.get("jira_key")
        if not jira_key or not self.github_client or not self.jira_client:
            self.logger.debug("Skipping post-LLM workflow (no JIRA key or clients not configured)")
            return

        try:
            # Get working directory (target repo or framework workspace)
            workspace = self._git_ops.get_working_directory(task)
            self.logger.info(f"Running post-LLM workflow for {jira_key} in {workspace}")

            # Create branch
            slug = task.title.lower().replace(" ", "-")[:30]
            branch = self.github_client.format_branch_name(jira_key, slug)

            self.logger.info(f"Creating branch: {branch}")
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Stage and commit changes
            self._update_phase(TaskPhase.COMMITTING)
            self.logger.info("Committing changes")
            subprocess.run(
                ["git", "add", "-A"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            commit_msg = f"[{jira_key}] {task.title}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Push to remote
            self.logger.info(f"Pushing branch to origin")
            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Create PR
            self._update_phase(TaskPhase.CREATING_PR)
            self.logger.info("Creating pull request")
            pr_title = self.github_client.format_pr_title(jira_key, task.title)
            pr_body = f"Implements {jira_key}\n\n{task.description}"

            pr = self.github_client.create_pull_request(
                title=pr_title,
                body=pr_body,
                head_branch=branch,
            )

            self.logger.info(f"Created PR: {pr.html_url}")

            # Store PR URL in task context for activity events
            task.context["pr_url"] = pr.html_url

            # Update JIRA
            self._update_phase(TaskPhase.UPDATING_JIRA)
            self.logger.info("Updating JIRA ticket")
            self.jira_client.transition_ticket(jira_key, "code_review")
            self.jira_client.add_comment(
                jira_key,
                f"Pull request created: {pr.html_url}"
            )

            self.logger.info(f"Post-LLM workflow complete for {jira_key}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git operation failed: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            self.logger.exception(f"Error in post-LLM workflow: {e}")

    def _sync_jira_status(self, task: Task, target_status: str, comment: Optional[str] = None) -> None:
        """Transition a JIRA ticket to target_status if all preconditions are met.

        Deterministic framework-level JIRA updates — agents don't reliably call
        MCP tools, so the framework ensures tickets reflect actual progress.
        """
        jira_key = task.context.get("jira_key")
        if not jira_key:
            return
        if not self.jira_client:
            return
        if not self._agent_definition or not self._agent_definition.jira_can_update_status:
            return
        if target_status not in (self._agent_definition.jira_allowed_transitions or []):
            self.logger.warning(
                f"Transition '{target_status}' not in allowed transitions for {self.config.id}, skipping"
            )
            return

        try:
            self.jira_client.transition_ticket(jira_key, target_status)
            self.logger.info(f"JIRA {jira_key} → {target_status}")
            if comment:
                self.jira_client.add_comment(jira_key, comment)
        except Exception as e:
            self.logger.warning(f"Failed to transition JIRA {jira_key} to '{target_status}': {e}")

    def _write_heartbeat(self) -> None:
        """Write current Unix timestamp to heartbeat file."""
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file.write_text(str(int(time.time())))

    def _check_pause_signal(self) -> bool:
        """Check if pause signal file exists.

        Checks for two pause signals:
        1. PAUSE_SIGNAL_FILE - manual pause by user
        2. PAUSE_INTAKE - automatic pause by orchestrator due to health issues

        Result is cached for 5 seconds to reduce filesystem I/O.
        """
        now = time.time()
        if self._pause_signal_cache is not None and (now - self._pause_signal_cache_time) < 5.0:
            return self._pause_signal_cache

        pause_file = self.workspace / PAUSE_SIGNAL_FILE
        health_pause_file = self.workspace / ".agent-communication" / "PAUSE_INTAKE"
        result = pause_file.exists() or health_pause_file.exists()
        self._pause_signal_cache = result
        self._pause_signal_cache_time = now
        return result

    @property
    def is_paused(self) -> bool:
        """Check if agent is currently paused."""
        return self._paused

    async def _run_sandbox_tests(self, task: Task) -> Optional[Any]:
        """Run tests in Docker sandbox if enabled and applicable.

        Returns:
            TestResult if tests were run, None if sandbox not enabled/applicable
        """
        # Skip if sandbox not initialized
        if not self._test_runner:
            return None

        # Only run tests for implementation tasks
        if task.type not in (TaskType.IMPLEMENTATION, TaskType.FIX, TaskType.BUGFIX, TaskType.ENHANCEMENT):
            return None

        # Get repository path
        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.debug(f"Skipping sandbox tests for {task.id}: no github_repo in context")
            return None

        repo_path = self._git_ops.get_working_directory(task)
        if not repo_path.exists():
            self.logger.warning(f"Repository path does not exist: {repo_path}")
            return None

        # Update task status
        task.status = TaskStatus.TESTING
        self.queue.update(task)
        self._update_phase(TaskPhase.COMMITTING)  # Reuse committing phase for testing

        self.logger.info(f"Running tests in sandbox for {task.id} at {repo_path}")

        try:
            # Run tests
            test_result = self._test_runner.run_sync(
                repo_path=repo_path,
                packages="./...",
                verbose=True,
            )

            self.logger.info(f"Test result for {task.id}: {test_result.summary}")

            # Log test event
            self.activity_manager.append_event(ActivityEvent(
                type="test_complete" if test_result.success else "test_fail",
                agent=self.config.id,
                task_id=task.id,
                title=test_result.summary,
                timestamp=datetime.now(timezone.utc)
            ))

            return test_result

        except Exception as e:
            self.logger.exception(f"Error running sandbox tests for {task.id}: {e}")
            # Return a failed result
            if TestResult:
                return TestResult(
                    success=False,
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    duration_seconds=0,
                    error_message=str(e),
                )
            return None

    async def _handle_test_failure(self, task: Task, llm_response, test_result) -> None:
        """Handle test failure by feeding results back to agent for fixing.

        Args:
            task: The task being processed
            llm_response: Original LLM response
            test_result: TestResult with failure details
        """
        # Increment test retry count
        test_retry = task.context.get("_test_retry_count", 0)
        task.context["_test_retry_count"] = test_retry + 1

        # Build prompt with test failure context
        failure_report = self._test_runner.format_failure_report(test_result)

        # Store failure context for next attempt
        task.context["_test_failure_report"] = failure_report
        task.notes.append(f"Test failure (attempt {test_retry + 1}): {test_result.error_message}")

        # Reset task to pending for retry
        task.reset_to_pending()
        self.queue.update(task)

        self.logger.info(f"Task {task.id} reset for test fix retry (attempt {test_retry + 1})")

    # Review cycle methods moved to ReviewCycleManager

    # -- Workflow routing methods moved to WorkflowRouter --
    # Backwards compatibility shims that delegate to WorkflowRouter

    def _check_and_create_fan_in_task(self, task: Task) -> None:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.check_and_create_fan_in_task(task)

    def _should_decompose_task(self, task: Task) -> bool:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.should_decompose_task(task)

    def _decompose_and_queue_subtasks(self, task: Task) -> None:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.decompose_and_queue_subtasks(task)

    def _enforce_workflow_chain(self, task: Task, response, routing_signal=None) -> None:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.enforce_chain(task, response, routing_signal)

    def _is_at_terminal_workflow_step(self, task: Task) -> bool:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.is_at_terminal_workflow_step(task)

    def _build_workflow_context(self, task: Task) -> Dict[str, Any]:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.build_workflow_context(task)

    def _route_to_agent(self, task: Task, target_agent: str, reason: str) -> None:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.route_to_agent(task, target_agent, reason)

    def _queue_pr_creation_if_needed(self, task: Task, workflow) -> None:
        """Delegate to WorkflowRouter for backwards compatibility."""
        return self._workflow_router.queue_pr_creation_if_needed(task, workflow)
