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
ERROR_HEAD_LINES = 20
ERROR_TAIL_LINES = 10
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

# Cap review cycles to prevent infinite QA ↔ Engineer loops
MAX_REVIEW_CYCLES = 3

REVIEW_OUTCOME_PATTERNS = {
    "request_changes": [r'\bREQUEST_CHANGES\b', r'\bCHANGES REQUESTED\b'],
    "critical_issues": [r'\bCRITICAL\b.*?:', r'severity:\s*CRITICAL'],
    "major_issues": [r'\bMAJOR\b.*?:', r'\bHIGH\b.*?:'],
    "test_failures": [r'tests?\s+fail', r'[1-9]\d*\s+failed'],
    "approve": [r'\bAPPROVE[D]?\b', r'\bLGTM\b'],
}

# Severity patterns matched case-sensitively — uppercase tags only, avoids prose false positives
_CASE_SENSITIVE_KEYS = frozenset({"critical_issues", "major_issues"})

# Line-anchored severity tags for default-deny detection
_SEVERITY_TAG_RE = re.compile(r'^(CRITICAL|HIGH|MAJOR|MEDIUM|MINOR|LOW|SUGGESTION)\b', re.MULTILINE)


@dataclass
class QAFinding:
    """Structured QA finding with file location, severity, and details."""
    file: str
    line_number: Optional[int]
    severity: str  # CRITICAL|HIGH|MAJOR|MEDIUM|LOW|MINOR|SUGGESTION
    description: str
    suggested_fix: Optional[str]
    category: str  # security|performance|correctness|readability|testing|best_practices


@dataclass
class ReviewOutcome:
    """Parsed result of a QA review."""
    approved: bool
    has_critical_issues: bool
    has_test_failures: bool
    has_change_requests: bool
    findings_summary: str
    has_major_issues: bool = False
    structured_findings: List['QAFinding'] = None

    def __post_init__(self):
        if self.structured_findings is None:
            self.structured_findings = []

    @property
    def needs_fix(self) -> bool:
        return self.has_critical_issues or self.has_test_failures or self.has_change_requests or self.has_major_issues


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
        self._active_worktree: Optional[Path] = None  # Track active worktree for cleanup

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

        # Caches for prompt guidance (rebuilt identically per task)
        self._error_handling_guidance: Optional[str] = None
        self._guidance_cache: Dict[str, str] = {}

        # Cache for team composition (fixed per agent lifetime / workflow)
        self._default_team_cache: Optional[dict] = None
        self._workflow_team_cache: Dict[str, Optional[dict]] = {}
        # Set per-task by _build_prompt, consumed by team composition
        self._current_specialization = None

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
        self._purge_orphaned_review_tasks()

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
        self._check_and_create_fan_in_task(task)

        # Subtasks wait for fan-in — don't route them individually through
        # the workflow chain. The fan-in task handles QA/review/PR creation.
        if task.parent_task_id is not None:
            self.logger.debug(
                f"Subtask {task.id} complete — skipping workflow chain "
                f"(fan-in will handle routing)"
            )
        else:
            self.logger.debug(f"Checking if code review needed for {task.id}")
            self._queue_code_review_if_needed(task, response)
            self._queue_review_fix_if_needed(task, response)
            self._enforce_workflow_chain(task, response, routing_signal=routing_signal)

            # Safety net: create PR if LLM pushed but didn't create one.
            # Runs AFTER workflow chain so pr_url doesn't short-circuit the executor.
            self._push_and_create_pr_if_needed(task)

            # Autonomous PR lifecycle: poll CI, fix failures, merge
            self._manage_pr_lifecycle(task)

        self._extract_and_store_memories(task, response)
        self._analyze_tool_patterns(task)
        self._log_task_completion_metrics(task, response, task_start_time)

    def _log_task_completion_metrics(self, task: Task, response, task_start_time) -> None:
        """Log token usage, cost, and completion events."""
        from datetime import datetime

        total_tokens = response.input_tokens + response.output_tokens
        budget = self._get_token_budget(task.type)
        cost = self._estimate_cost(response)

        duration = (datetime.now(timezone.utc) - task_start_time).total_seconds()
        self.logger.token_usage(response.input_tokens, response.output_tokens, cost)
        self.logger.task_completed(duration, tokens_used=total_tokens)

        # Budget warning
        if self._optimization_config.get("enable_token_budget_warnings", False):
            threshold = self._optimization_config.get("budget_warning_threshold", BUDGET_WARNING_THRESHOLD)
            if total_tokens > budget * threshold:
                self.logger.warning(
                    f"Task {task.id} EXCEEDED TOKEN BUDGET: "
                    f"{total_tokens} tokens (budget: {budget}, "
                    f"{int(threshold * 100)}% threshold: {budget * threshold:.0f})"
                )
                self.activity_manager.append_event(ActivityEvent(
                    type="token_budget_exceeded",
                    agent=self.config.id,
                    task_id=task.id,
                    title=f"Token budget exceeded: {total_tokens} > {budget}",
                    timestamp=datetime.now(timezone.utc)
                ))

        # Append complete event
        duration_ms = int((datetime.now(timezone.utc) - task_start_time).total_seconds() * 1000)
        pr_url = task.context.get("pr_url")
        self.activity_manager.append_event(ActivityEvent(
            type="complete",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            pr_url=pr_url,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost
        ))

        self._session_logger.log(
            "task_complete",
            status="completed",
            duration_ms=duration_ms,
        )

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
        self._sync_worktree_queued_tasks()
        self._cleanup_worktree(task, success=task_succeeded)

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
            working_dir = self._get_working_directory(task)
            self.logger.info(f"Working directory: {working_dir}")

            # Build prompt and execute LLM
            self.logger.phase_change("analyzing")
            prompt = self._build_prompt(task)

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

        Ported from scripts/async-agent-runner.sh lines 374-394.
        """
        # Re-read from disk to detect external status changes (e.g. `agent cancel`)
        refreshed = self.queue.find_task(task.id)
        if refreshed and refreshed.status == TaskStatus.CANCELLED:
            self.logger.info(f"Task {task.id} was cancelled, skipping retry")
            # Preserve CANCELLED status — mark_completed would overwrite it
            self.queue.move_to_completed(refreshed)
            return

        if task.retry_count >= self.retry_handler.max_retries:
            # Max retries exceeded - mark as failed
            self.logger.error(
                f"Task {task.id} has failed {task.retry_count} times "
                f"(max: {self.retry_handler.max_retries})"
            )
            error_type = self._categorize_error(task.last_error or "")
            task.mark_failed(self.config.id, error_message=task.last_error, error_type=error_type)
            self.queue.mark_failed(task)

            # Notify JIRA about permanent failure (no status change, just a comment)
            jira_key = task.context.get("jira_key")
            if jira_key and self.jira_client:
                try:
                    self.jira_client.add_comment(
                        jira_key,
                        f"Agent {self.config.id} failed after {task.retry_count} retries: {task.last_error}",
                    )
                except Exception:
                    pass

            # CRITICAL: Prevent infinite loop - escalations should NOT create more escalations
            if self.retry_handler.can_create_escalation(task):
                escalation = self.escalation_handler.create_escalation(
                    task, self.config.id
                )
                self.queue.push(escalation, escalation.assigned_to)
                self.logger.warning(
                    f"Created escalation task {escalation.id} for failed task {task.id}"
                )
            else:
                # Escalation failed - log to escalations directory for human review
                self.logger.error(
                    f"Escalation task {task.id} failed after {task.retry_count} retries - "
                    "NOT creating another escalation (would cause infinite loop). "
                    "Logging to escalations directory for human intervention."
                )
                self._log_failed_escalation(task)
        else:
            # Dynamic replanning: generate revised approach on retry 2+
            if self._replan_enabled and task.retry_count >= self._replan_min_retry:
                await self._request_replan(task)

            # Reset task to pending so it can be retried
            self.logger.warning(
                f"Resetting task {task.id} to pending status "
                f"(retry {task.retry_count + 1}/{self.retry_handler.max_retries})"
            )
            task.reset_to_pending()
            self.queue.update(task)

    def _log_failed_escalation(self, task: Task) -> None:
        """
        Log a failed escalation to the escalations directory for human review.

        When an escalation task itself fails, we cannot create another escalation
        (infinite loop). Instead, write it to a dedicated directory where humans
        can review and resolve it.
        """
        escalations_dir = self.workspace / ".agent-communication" / "escalations"
        escalations_dir.mkdir(parents=True, exist_ok=True)

        escalation_file = escalations_dir / f"{task.id}.json"

        # Add metadata for human review
        task_dict = task.model_dump()
        task_dict["logged_at"] = datetime.now(timezone.utc).isoformat()
        task_dict["logged_by"] = self.config.id
        task_dict["requires_human_intervention"] = True
        task_dict["escalation_failed"] = True

        try:
            escalation_file.write_text(json.dumps(task_dict, indent=2))
            self.logger.info(
                f"Logged failed escalation to {escalation_file}. "
                f"Run 'bash scripts/review-escalations.sh' to review."
            )
        except Exception as e:
            self.logger.error(f"Failed to log escalation to file: {e}")
            # Last resort: at least log the full task details to the log file
            self.logger.error(f"Failed escalation details: {json.dumps(task_dict, indent=2)}")

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
            task_hash = int(hashlib.md5(task.id.encode()).hexdigest()[:8], 16)
            return (task_hash % 100) < canary_pct

    def _get_minimal_task_dict(self, task: Task) -> Dict[str, Any]:
        """
        Extract only prompt-relevant task fields.

        Implements Strategy 1 (Minimal Task Prompts) from the optimization plan.
        See OPTIMIZATION_IMPLEMENTATION_SUMMARY.md for details.

        Omits metadata fields that don't contribute to task execution:
        - Timestamps (created_at, started_at, completed_at)
        - Internal tracking (created_by, assigned_to, retry_count)
        - Dependency arrays (depends_on, blocks)

        Expected savings: 3-8KB per task (40-50% reduction).
        """
        # Validate essential fields
        if not task.title or not task.description:
            self.logger.warning(
                f"Task {task.id} missing essential fields: "
                f"title={bool(task.title)}, description={bool(task.description)}. "
                f"Falling back to full task dict."
            )
            return task.model_dump()

        minimal = {
            "title": task.title.strip(),
            "description": task.description.strip(),
            "type": get_type_str(task.type),
        }

        # Include acceptance criteria and deliverables if present
        if task.acceptance_criteria:
            minimal["acceptance_criteria"] = task.acceptance_criteria
        if task.deliverables:
            minimal["deliverables"] = task.deliverables

        # Include notes if non-empty (can contain important context)
        if task.notes:
            minimal["notes"] = task.notes

        # Include only relevant context keys
        relevant_context = {}
        for key in ["jira_key", "jira_project", "github_repo", "mode", "user_goal", "repository_name", "epic_key"]:
            if key in task.context:
                relevant_context[key] = task.context[key]

        if relevant_context:
            minimal["context"] = relevant_context

        return minimal

    def _get_token_budget(self, task_type: TaskType) -> int:
        """
        Get expected token budget for task type.

        Implements Strategy 6 (Token Tracking) from the optimization plan.

        Budgets can be configured via optimization.token_budgets in config,
        otherwise uses sensible defaults.
        """
        # Default budgets
        default_budgets = {
            "planning": 30000,
            "implementation": 50000,
            "testing": 20000,
            "escalation": 80000,
            "review": 25000,
            "architecture": 40000,
            "coordination": 15000,
            "documentation": 15000,
            "fix": 30000,
            "bugfix": 30000,
            "bug-fix": 30000,
            "verification": 20000,
            "status_report": 10000,
            "enhancement": 40000,
        }

        # Get budget from config or use default
        budget_key = get_type_str(task_type).lower().replace("-", "_")
        configured_budgets = self._optimization_config.get("token_budgets", {})

        return configured_budgets.get(budget_key, default_budgets.get(budget_key, 40000))

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
            self.logger.debug(f"Saved upstream context ({len(content)} chars) to {context_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save upstream context for {task.id}: {e}")

    def _load_upstream_context(self, task: Task) -> str:
        """Load upstream agent's findings from disk if available.

        Returns formatted section string or empty string.
        """
        context_file = task.context.get("upstream_context_file")
        if not context_file:
            return ""

        try:
            context_path = Path(context_file).resolve()
            summaries_dir = (self.workspace / ".agent-context" / "summaries").resolve()

            # Only read files inside our summaries directory
            if not str(context_path).startswith(str(summaries_dir)):
                self.logger.warning(f"Upstream context path outside summaries dir: {context_file}")
                return ""

            if not context_path.exists():
                return ""

            content = context_path.read_text()
            if not content.strip():
                return ""

            return f"\n## UPSTREAM AGENT FINDINGS\n{content}\n"
        except Exception as e:
            self.logger.debug(f"Failed to load upstream context: {e}")
            return ""

    # -- Agent Memory Integration --

    def _get_repo_slug(self, task: Task) -> Optional[str]:
        """Extract repo slug from task context."""
        return task.context.get("github_repo")

    def _inject_memories(self, prompt: str, task: Task) -> str:
        """Append relevant memories from previous tasks to the prompt."""
        if not self._memory_enabled:
            return prompt

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return prompt

        # Build tag hints from task context
        task_tags = []
        if task.type:
            task_tags.append(get_type_str(task.type))
        jira_project = task.context.get("jira_project")
        if jira_project:
            task_tags.append(jira_project)

        memory_section = self._memory_retriever.format_for_prompt(
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            task_tags=task_tags,
        )

        if memory_section:
            self.logger.debug(f"Injected {len(memory_section)} chars of memory context")
            self._session_logger.log(
                "memory_recall",
                repo=repo_slug,
                chars_injected=len(memory_section),
                categories=task_tags,
            )
            return prompt + "\n" + memory_section

        return prompt

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

    def _inject_tool_tips(self, prompt: str, task: Task) -> str:
        """Append tool efficiency tips from previous session analysis."""
        if not self._tool_tips_enabled:
            return prompt

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return prompt

        max_count = self._optimization_config.get("tool_tips_max_count", 5)
        max_chars = self._optimization_config.get("tool_tips_max_chars", 1500)
        patterns = self._tool_pattern_store.get_top_patterns(
            repo_slug, limit=max_count, max_chars=max_chars,
        )
        if not patterns:
            return prompt

        tips_lines = [f"- {p.tip}" for p in patterns]
        tips_section = "## Tool Efficiency Tips\n\n" + "\n".join(tips_lines)

        self.logger.debug(f"Injected {len(patterns)} tool tips ({len(tips_section)} chars)")
        self._session_logger.log(
            "tool_tips_injected",
            repo=repo_slug,
            count=len(patterns),
            chars=len(tips_section),
        )
        return prompt + "\n\n" + tips_section

    # -- Self-Evaluation Loop --

    async def _self_evaluate(self, task: Task, response) -> bool:
        """Review agent's own output against acceptance criteria.

        Uses a cheap model to check for obvious gaps. If gaps found,
        resets task with critique context for retry — without consuming
        a queue-level retry.

        Returns True if evaluation passed (or disabled/skipped).
        Returns False if task was reset for retry.
        """
        eval_retries = task.context.get("_self_eval_count", 0)
        if eval_retries >= self._self_eval_max_retries:
            self.logger.debug(
                f"Self-eval retry limit reached ({eval_retries}), proceeding"
            )
            return True

        # Build evaluation prompt from acceptance criteria
        criteria = task.acceptance_criteria
        if not criteria:
            return True

        criteria_text = "\n".join(f"- {c}" for c in criteria)
        response_preview = response.content[:4000] if response.content else ""

        eval_prompt = f"""Review this agent's output against the acceptance criteria.
Reply with PASS if all criteria are met, or FAIL followed by specific gaps.

## Acceptance Criteria
{criteria_text}

## Agent Output (preview)
{response_preview}

Verdict:"""

        try:
            eval_response = await self.llm.complete(LLMRequest(
                prompt=eval_prompt,
                model=self._self_eval_model,
            ))

            if not eval_response.success or not eval_response.content:
                self.logger.warning("Self-eval LLM call failed, proceeding without eval")
                return True

            verdict = eval_response.content.strip()
            passed = verdict.upper().startswith("PASS")

            self._session_logger.log(
                "self_eval",
                verdict="PASS" if passed else "FAIL",
                model=self._self_eval_model,
                criteria_count=len(criteria),
                eval_attempt=eval_retries + 1,
            )

            if passed:
                self.logger.info(f"Self-evaluation PASSED for task {task.id}")
                return True

            # Failed self-eval — reset for retry with critique
            self.logger.warning(
                f"Self-evaluation FAILED for task {task.id} "
                f"(attempt {eval_retries + 1}/{self._self_eval_max_retries}): "
                f"{verdict[:200]}"
            )

            task.context["_self_eval_count"] = eval_retries + 1
            task.context["_self_eval_critique"] = verdict[:1000]
            task.notes.append(f"Self-eval failed (attempt {eval_retries + 1}): {verdict[:200]}")

            # Reset without consuming queue retry
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.started_by = None
            self.queue.update(task)

            return False

        except Exception as e:
            self.logger.warning(f"Self-evaluation error (non-fatal): {e}")
            return True

    # -- Dynamic Replanning --

    async def _request_replan(self, task: Task) -> None:
        """Generate a revised approach based on what failed.

        Called on retry 2+ to avoid repeating the same failing approach.
        Stores the revised plan in task.replan_history and task.context
        so the next prompt attempt sees what was tried and the new approach.
        """
        error = task.last_error or "Unknown error"
        previous_attempts = task.replan_history or []

        attempts_text = ""
        if previous_attempts:
            for attempt in previous_attempts:
                attempts_text += (
                    f"\n- Attempt {attempt.get('attempt', '?')}: "
                    f"{attempt.get('error', 'no error recorded')}"
                )

        # Inject relevant memories for context
        memory_section = ""
        if self._memory_enabled:
            repo_slug = self._get_repo_slug(task)
            if repo_slug:
                task_tags = []
                if task.type:
                    task_tags.append(get_type_str(task.type))
                jira_project = task.context.get("jira_project")
                if jira_project:
                    task_tags.append(jira_project)

                memory_section = self._memory_retriever.format_for_replan(
                    repo_slug=repo_slug,
                    agent_type=self.config.base_id,
                    task_tags=task_tags,
                )

                if memory_section:
                    self.logger.debug(f"Injected {len(memory_section)} chars of memory context into replan")
                    self._session_logger.log(
                        "memory_replan_recall",
                        repo=repo_slug,
                        chars_injected=len(memory_section),
                    )

        replan_prompt = f"""A task has failed {task.retry_count} times. Generate a REVISED approach.

## Task
{task.title}: {task.description[:1000]}

## Latest Error
{error[:500]}

## Previous Attempts{attempts_text if attempts_text else ' (first replan)'}

{memory_section}
## Instructions
Provide a revised approach in 3-5 bullet points. Focus on what to do DIFFERENTLY.
Do NOT repeat the same approach. Consider: different implementation strategy,
breaking the task into smaller steps, or working around the root cause."""

        try:
            replan_response = await self.llm.complete(LLMRequest(
                prompt=replan_prompt,
                model=self._replan_model,
            ))

            if replan_response.success and replan_response.content:
                revised_plan = replan_response.content.strip()[:2000]

                # Store in replan history
                history_entry = {
                    "attempt": task.retry_count,
                    "error": error[:500],
                    "revised_plan": revised_plan,
                }
                task.replan_history.append(history_entry)

                # Store in context for prompt injection
                task.context["_revised_plan"] = revised_plan
                task.context["_replan_attempt"] = task.retry_count

                self._session_logger.log(
                    "replan",
                    retry=task.retry_count,
                    previous_error=error[:500],
                    revised_plan=revised_plan,
                    model=self._replan_model,
                )

                self.logger.info(
                    f"Generated revised plan for task {task.id} "
                    f"(retry {task.retry_count}): {revised_plan[:100]}..."
                )
            else:
                self.logger.warning(
                    f"Replan LLM call failed for task {task.id}: "
                    f"{replan_response.error}"
                )

        except Exception as e:
            self.logger.warning(f"Replanning error (non-fatal): {e}")

    def _inject_replan_context(self, prompt: str, task: Task) -> str:
        """Append revised plan and attempt history to prompt if available."""
        revised_plan = task.context.get("_revised_plan")
        if not revised_plan:
            return prompt

        self_eval_critique = task.context.get("_self_eval_critique", "")

        replan_section = f"""

## REVISED APPROACH (retry {task.retry_count})

Previous attempts failed. Use this revised approach:

{revised_plan}
"""
        if self_eval_critique:
            replan_section += f"""
## Self-Evaluation Feedback
{self_eval_critique}
"""

        if task.replan_history:
            replan_section += "\n## Previous Attempt History\n"
            for entry in task.replan_history[:-1]:  # Skip current, already shown above
                replan_section += (
                    f"- Attempt {entry.get('attempt', '?')}: "
                    f"Failed with: {entry.get('error', 'unknown')[:100]}\n"
                )

        return prompt + replan_section

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

    def _categorize_error(self, error_message: str) -> Optional[str]:
        """Categorize error message for better diagnostics.

        Delegates to EscalationHandler.categorize_error for consistent
        pattern matching across the codebase.
        """
        return self.escalation_handler.categorize_error(error_message)

    def _inject_human_guidance(self, prompt: str, task: Task) -> str:
        """Inject human guidance from escalation report if available."""
        # Check for human guidance in escalation report
        if task.escalation_report and task.escalation_report.human_guidance:
            guidance = task.escalation_report.human_guidance
            guidance_section = f"""

## CRITICAL: Human Guidance Provided

A human expert has reviewed this task and provided the following guidance to help you succeed:

{guidance}

Please carefully consider this guidance when approaching the task. This information may help you avoid the previous failures.

## Previous Failure Context

{task.escalation_report.root_cause_hypothesis}

Suggested interventions:
"""
            for i, intervention in enumerate(task.escalation_report.suggested_interventions, 1):
                guidance_section += f"{i}. {intervention}\n"

            return prompt + guidance_section

        # Fall back to context-based guidance (legacy support)
        context_guidance = task.context.get("human_guidance")
        if context_guidance:
            return prompt + f"""

## CRITICAL: Human Guidance Provided

A human expert has provided guidance for this task:

{context_guidance}

Please carefully consider this guidance when approaching the task.
"""

        return prompt

    def _handle_shadow_mode_comparison(self, task: Task, prompt_override: str = None) -> str:
        """Generate and compare both prompts in shadow mode, return legacy prompt."""
        legacy_prompt = self._build_prompt_legacy(task, prompt_override=prompt_override)
        optimized_prompt = self._build_prompt_optimized(task, prompt_override=prompt_override)

        # Log comparison
        legacy_len = len(legacy_prompt)
        optimized_len = len(optimized_prompt)
        savings = legacy_len - optimized_len
        savings_pct = (savings / legacy_len * 100) if legacy_len > 0 else 0

        # Truncate task ID for security
        task_id_short = task.id[:8] + "..." if len(task.id) > 8 else task.id

        self.logger.debug(
            f"[SHADOW MODE] Task {task_id_short} prompt comparison: "
            f"legacy={legacy_len} chars, optimized={optimized_len} chars, "
            f"savings={savings} chars ({savings_pct:.1f}%)"
        )

        # Record metrics for analysis
        self._record_optimization_metrics(task, legacy_len, optimized_len)

        # Return legacy prompt (no behavioral change in shadow mode)
        return legacy_prompt

    def _append_test_failure_context(self, prompt: str, task: Task) -> str:
        """Append test failure report to prompt if present."""
        test_failure_report = task.context.get("_test_failure_report")
        if not test_failure_report:
            return prompt

        return prompt + f"""

## IMPORTANT: Previous Tests Failed

Your previous implementation had test failures. Please fix the issues below:

{test_failure_report}

Fix the failing tests and ensure all tests pass.
"""

    def _detect_engineer_specialization(self, task: Task) -> Optional['SpecializationProfile']:
        """Detect engineer specialization profile for this task.

        Returns the profile if this is an engineer agent with a clear match, else None.
        Respects both per-agent and global enable/disable toggles.
        Falls back to LLM-generated profiles when static detection returns None.
        """
        if self.config.base_id != "engineer":
            return None

        # Per-agent toggle from agents.yaml
        if self._agent_definition and not self._agent_definition.specialization_enabled:
            self.logger.debug("Specialization disabled for this agent via agents.yaml")
            return None

        # Global toggle from specializations.yaml
        from .engineer_specialization import (
            detect_specialization,
            get_specialization_enabled,
            get_auto_profile_config,
            detect_file_patterns,
            _load_profiles,
        )

        if not get_specialization_enabled():
            self.logger.debug("Specialization disabled globally via specializations.yaml")
            return None

        # Extract files once — reused for both static detection and auto-profile fallback
        files = detect_file_patterns(task)

        profile = detect_specialization(task, files=files)
        if profile:
            return profile

        # Auto-profile fallback: check registry, then generate
        auto_config = get_auto_profile_config()
        if auto_config is None or not auto_config.enabled:
            return None

        if not files:
            return None

        import time
        from .profile_registry import ProfileRegistry, GeneratedProfileEntry
        from .profile_generator import ProfileGenerator

        registry = ProfileRegistry(
            self.workspace,
            max_profiles=auto_config.max_cached_profiles,
            staleness_days=auto_config.staleness_days,
        )

        # Try cache first
        cached = registry.find_matching_profile(
            files,
            f"{task.title} {task.description}",
            min_score=auto_config.min_match_score,
        )
        if cached:
            self.logger.info("Matched cached generated profile '%s'", cached.id)
            return cached

        # Generate new profile
        generator = ProfileGenerator(self.workspace, model=auto_config.model)
        existing_ids = [p.id for p in _load_profiles()]
        generated = generator.generate_profile(task, files, existing_ids)
        if generated:
            now = time.time()
            registry.store_profile(GeneratedProfileEntry(
                profile=generated.profile,
                created_at=now,
                last_matched_at=now,
                match_count=1,
                source_task_id=task.id,
                tags=generated.tags,
                file_extensions=generated.file_extensions,
            ))
            self.logger.info("Generated new profile '%s'", generated.profile.id)
            return generated.profile

        return None

    def _build_prompt(self, task: Task) -> str:
        """
        Build prompt from task.

        Supports optimization strategies:
        - Strategy 1: Minimal task prompts
        - Strategy 3: Context deduplication
        - Strategy 4: Compact JSON
        - Strategy 5: Result summarization
        - Shadow mode: Generate both for comparison

        Ported from scripts/async-agent-runner.sh lines 268-294.
        """
        shadow_mode = self._optimization_config.get("shadow_mode", False)
        use_optimizations = self._should_use_optimization(task)

        # Detect specialization once — reused for both prompt and team composition
        from .engineer_specialization import apply_specialization_to_prompt
        profile = self._detect_engineer_specialization(task)
        self._current_specialization = profile
        prompt_text = apply_specialization_to_prompt(self.config.prompt, profile)

        # Update activity with specialization (if detected)
        if profile:
            activity = self.activity_manager.get_activity(self.config.id)
            if activity:
                activity.specialization = profile.id
                self.activity_manager.update_activity(activity)

        # Determine which prompt to use
        if shadow_mode:
            prompt = self._handle_shadow_mode_comparison(task, prompt_override=prompt_text)
        elif use_optimizations:
            prompt = self._build_prompt_optimized(task, prompt_override=prompt_text)
        else:
            prompt = self._build_prompt_legacy(task, prompt_override=prompt_text)

        # Inject preview mode constraints when task is a preview
        if task.type == TaskType.PREVIEW:
            prompt = self._inject_preview_mode(prompt, task)

        # Log prompt preview for debugging (sanitized)
        if self.logger.isEnabledFor(logging.DEBUG):
            prompt_preview = prompt[:500].replace(task.id, "TASK_ID")
            if hasattr(task, 'context') and task.context.get('jira_key'):
                prompt_preview = prompt_preview.replace(task.context['jira_key'], "JIRA-XXX")
            self.logger.debug(f"Built prompt preview (first 500 chars): {prompt_preview}...")

        # Append test failure context if present
        prompt = self._append_test_failure_context(prompt, task)

        # Inject relevant memories from previous tasks
        prompt = self._inject_memories(prompt, task)

        # Inject tool efficiency tips from session analysis
        prompt = self._inject_tool_tips(prompt, task)

        # Inject replan history if retrying with revised approach
        prompt = self._inject_replan_context(prompt, task)

        # Inject human guidance if provided via `agent guide` command
        prompt = self._inject_human_guidance(prompt, task)

        # Session log: capture what was sent to the LLM
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        has_replan = "_revised_plan" in task.context
        self._session_logger.log(
            "prompt_built",
            prompt_length=len(prompt),
            prompt_hash=prompt_hash,
            replan_injected=has_replan,
            retry=task.retry_count,
        )
        self._session_logger.log_prompt(prompt)

        return prompt

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
        """
        Estimate cost based on model and token usage.

        Prefers CLI-reported cost when available (accounts for prompt caching
        discounts). Falls back to static MODEL_PRICING calculation for other
        backends or when reported cost is unavailable.
        """
        if response.reported_cost_usd is not None:
            return response.reported_cost_usd

        model_name_lower = response.model_used.lower()

        # Detect model family
        if "haiku" in model_name_lower:
            model_type = "haiku"
        elif "opus" in model_name_lower:
            model_type = "opus"
        elif "sonnet" in model_name_lower:
            model_type = "sonnet"
        else:
            # Unknown model, assume sonnet pricing as conservative estimate
            self.logger.warning(
                f"Unknown model '{response.model_used}', assuming Sonnet pricing for cost estimate"
            )
            model_type = "sonnet"

        prices = MODEL_PRICING.get(model_type, MODEL_PRICING["sonnet"])
        cost = (
            response.input_tokens / 1_000_000 * prices["input"] +
            response.output_tokens / 1_000_000 * prices["output"]
        )
        return cost

    def _build_jira_guidance(self, jira_key: str, jira_project: str) -> str:
        """Build JIRA integration guidance for MCP."""
        cache_key = f"jira:{jira_key}:{jira_project}"
        if cache_key in self._guidance_cache:
            return self._guidance_cache[cache_key]

        jira_server = self.jira_config.server if self.jira_config else "jira.example.com"

        result = f"""
JIRA INTEGRATION (via MCP):
You have access to JIRA via MCP tools:
- Search issues: jira_search_issues(jql="project = {jira_project or 'PROJ'}")
- Get issue: jira_get_issue(issueKey="{jira_key or 'PROJ-123'}")
- Create ticket: jira_create_issue(project="{jira_project or 'PROJ'}", summary="...", description="...", issueType="Story")
- Create epic: jira_create_epic(project="{jira_project or 'PROJ'}", title="...", description="...")
- Create subtask: jira_create_subtask(parentKey="{jira_key or 'PROJ-123'}", summary="...", description="...")
- Update status: jira_transition_issue(issueKey="{jira_key or 'PROJ-123'}", transitionName="In Progress")
- Add comment: jira_add_comment(issueKey="{jira_key or 'PROJ-123'}", comment="...")

Current context:
- JIRA Server: {jira_server}
- Ticket: {jira_key or 'N/A'}
- Project: {jira_project or 'N/A'}

"""
        self._guidance_cache[cache_key] = result
        return result

    def _build_github_guidance(self, github_repo: str, jira_key: str) -> str:
        """Build GitHub integration guidance for MCP."""
        cache_key = f"github:{github_repo}:{jira_key}"
        if cache_key in self._guidance_cache:
            return self._guidance_cache[cache_key]

        owner, repo = github_repo.split("/")

        # Get formatting patterns from config
        branch_pattern = "{type}/{ticket_id}-{slug}"
        pr_title_pattern = "[{ticket_id}] {title}"
        if self.github_config:
            branch_pattern = self.github_config.branch_pattern
            pr_title_pattern = self.github_config.pr_title_pattern

        result = f"""
GITHUB INTEGRATION (via MCP):
Repository: {github_repo}
Branch naming: Use pattern "{branch_pattern}"
  Example: feature/{jira_key or 'PROJ-123'}-add-authentication
PR title: Use pattern "{pr_title_pattern}"
  Example: [{jira_key or 'PROJ-123'}] Add authentication feature

Available tools:
- github_create_pr(owner="{owner}", repo="{repo}",
                   title="[{jira_key or 'PROJ-123'}] Your Title",
                   body="...",
                   head="feature/{jira_key or 'PROJ-123'}-slug")
- github_add_pr_comment(owner="{owner}", repo="{repo}", prNumber=123, body="...")
- github_link_pr_to_jira(owner="{owner}", repo="{repo}", prNumber=123, jiraKey="{jira_key or 'PROJ-123'}")

NOTE: You are responsible for committing and pushing your changes.

Workflow coordination:
1. Make your code changes
2. Commit changes: git add <files> && git commit -m "[TICKET] description"
3. Push to feature branch: git push
4. Create a PR using github_create_pr (if your workflow requires it)
5. Update JIRA using jira_transition_issue and jira_add_comment

"""
        self._guidance_cache[cache_key] = result
        return result

    def _build_error_handling_guidance(self) -> str:
        """Build error handling guidance for MCP tools."""
        if self._error_handling_guidance is not None:
            return self._error_handling_guidance
        self._error_handling_guidance = """
ERROR HANDLING:
If a tool call fails:
1. Read the error message carefully
2. If rate limited, wait and retry
3. If authentication failed, report failure
4. If invalid input, correct and retry
5. For partial failures (e.g., PR created but JIRA update failed):
   - Retry the failed operation
   - If still fails, leave completed operations and report the failure
   - Do NOT try to undo successful operations

"""
        return self._error_handling_guidance

    def _build_prompt_legacy(self, task: Task, prompt_override: str = None) -> str:
        """Build prompt using legacy format (original implementation)."""
        task_json = task.model_dump_json(indent=2)
        agent_prompt = prompt_override or self.config.prompt

        # Extract integration context
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        mcp_guidance = ""

        # Build MCP guidance sections
        if self._mcp_enabled:
            if jira_key or jira_project:
                mcp_guidance += self._build_jira_guidance(jira_key, jira_project)
            if github_repo:
                mcp_guidance += self._build_github_guidance(github_repo, jira_key)
            mcp_guidance += self._build_error_handling_guidance()

        # Intermediate chain steps must not create PRs — the terminal step handles that
        if task.context.get("chain_step") and not self._is_at_terminal_workflow_step(task):
            mcp_guidance += """
IMPORTANT: You are an intermediate step in the workflow chain.
Push your commits but do NOT create a pull request.
The PR will be created by a downstream agent after all steps complete.

"""

        # Subtasks must not create PRs — the fan-in task creates a single PR
        if task.parent_task_id is not None:
            mcp_guidance += """
IMPORTANT: You are a SUBTASK of a decomposed task.
Commit and push your changes, but do NOT create a pull request.
A fan-in task will aggregate all subtask results and create a single PR.

"""

        # Load upstream context from previous agent if available
        upstream_context = self._load_upstream_context(task)

        return f"""You are {self.config.id}.

TASK DETAILS:
{task_json}

{mcp_guidance}{upstream_context}
YOUR RESPONSIBILITIES:
{agent_prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _build_prompt_optimized(self, task: Task, prompt_override: str = None) -> str:
        """
        Build prompt using optimized format.

        This method should only be called when optimizations are enabled.
        All enable checks happen in _build_prompt(), not here.

        Applies:
        - Strategy 1: Minimal task prompts (only essential fields)
        - Strategy 3: Context deduplication (no redundant info)
        - Strategy 4: Compact JSON (no whitespace)
        - Strategy 5: Result summarization (include dep summaries)
        """
        # Always use minimal fields in optimized prompts (Strategy 1)
        task_dict = self._get_minimal_task_dict(task)

        # Always use compact JSON in optimized prompts (Strategy 4)
        task_json = json.dumps(task_dict, separators=(',', ':'))

        # Extract integration context (only dynamic values, not boilerplate)
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        # Build minimal MCP context (just dynamic values, no boilerplate)
        context_note = ""
        if self._mcp_enabled:
            if jira_key:
                context_note += f"JIRA Ticket: {jira_key}\n"
            if jira_project:
                context_note += f"JIRA Project: {jira_project}\n"
            if github_repo:
                context_note += f"GitHub Repository: {github_repo}\n"

        # Include dependency results (Strategy 5: Result Summarization)
        dep_context = ""
        enable_summarization = self._optimization_config.get("enable_result_summarization", False)
        if enable_summarization and task.depends_on:
            dep_context = "\nPREVIOUS WORK:\n"
            for dep_id in task.depends_on:
                dep_task = self.queue.get_completed(dep_id)
                if dep_task and dep_task.result_summary:
                    dep_context += f"- {dep_task.title}: {dep_task.result_summary}\n"

        # Intermediate chain steps must not create PRs
        chain_note = ""
        if task.context.get("chain_step") and not self._is_at_terminal_workflow_step(task):
            chain_note = "\nIMPORTANT: You are an intermediate step in the workflow chain.\nPush your commits but do NOT create a pull request.\n"

        # Subtasks must not create PRs — fan-in handles it
        if task.parent_task_id is not None:
            chain_note += "\nIMPORTANT: You are a SUBTASK of a decomposed task.\nCommit and push your changes, but do NOT create a pull request.\nA fan-in task will aggregate all subtask results and create a single PR.\n"

        # Load upstream context from previous agent if available
        upstream_context = self._load_upstream_context(task)

        # Build optimized prompt (shorter, focused on essentials)
        agent_prompt = prompt_override or self.config.prompt
        return f"""You are {self.config.id}.

TASK:
{task_json}

{context_note}{dep_context}{chain_note}{upstream_context}
{agent_prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _get_working_directory(self, task: Task) -> Path:
        """Get working directory for task (worktree, target repo, or framework workspace).

        Priority:
        0. PR creation tasks with an implementation_branch skip worktree entirely
        1. If worktree mode enabled (config or task override), create isolated worktree
        2. If multi_repo_manager available, use shared clone
        3. Fall back to framework workspace
        """
        github_repo = task.context.get("github_repo")

        # PR creation tasks that reference an upstream implementation branch
        # don't need their own worktree — `gh pr create` works from the shared clone
        if task.context.get("pr_creation_step") and task.context.get("implementation_branch"):
            if github_repo and self.multi_repo_manager:
                repo_path = self.multi_repo_manager.ensure_repo(github_repo)
                self.logger.info("PR creation task — using shared clone (no worktree needed)")
                return repo_path

        # Check if worktree mode should be used
        use_worktree = self._should_use_worktree(task)

        if use_worktree and github_repo and self.worktree_manager:
            # Get base repo path (shared clone or explicit override)
            base_repo = self._get_base_repo_for_worktree(task, github_repo)

            if base_repo:
                # Fix-cycle reuse: if a prior step already established a branch, reuse it
                branch_name = task.context.get("worktree_branch") or task.context.get("implementation_branch")

                if not branch_name:
                    jira_key = task.context.get("jira_key", "task")
                    task_hash = hashlib.sha256(task.id.encode()).hexdigest()[:8]
                    branch_name = f"agent/{self.config.id}/{jira_key}-{task_hash}"

                # Check registry for existing worktree on this branch (reuse or retry)
                existing = self.worktree_manager.find_worktree_by_branch(branch_name)
                if existing:
                    self._active_worktree = existing
                    task.context["worktree_branch"] = branch_name
                    self.logger.info(f"Reusing worktree for branch {branch_name}: {existing}")
                    return existing

                try:
                    worktree_path = self.worktree_manager.create_worktree(
                        base_repo=base_repo,
                        branch_name=branch_name,
                        agent_id=self.config.id,
                        task_id=task.id,
                        owner_repo=github_repo,
                    )
                    self._active_worktree = worktree_path
                    task.context["worktree_branch"] = branch_name
                    self.logger.info(f"Using worktree: {github_repo} at {worktree_path}")
                    return worktree_path
                except Exception as e:
                    self.logger.warning(f"Failed to create worktree, falling back to shared clone: {e}")
                    # Fall through to shared clone

        if github_repo and self.multi_repo_manager:
            # Ensure repo is cloned/updated
            repo_path = self.multi_repo_manager.ensure_repo(github_repo)
            self.logger.info(f"Using repository: {github_repo} at {repo_path}")
            return repo_path
        else:
            # No repo context, use framework workspace
            return self.workspace

    def _should_use_worktree(self, task: Task) -> bool:
        """Determine if worktree mode should be used for this task.

        Task context can override config:
        - task.context["use_worktree"] = True/False
        """
        # Check task-level override first
        task_override = task.context.get("use_worktree")
        if task_override is not None:
            return bool(task_override)

        # Check if worktree manager is available and enabled
        if not self.worktree_manager:
            return False

        return True  # Worktree manager exists, so worktree mode is enabled

    def _get_base_repo_for_worktree(self, task: Task, github_repo: str) -> Optional[Path]:
        """Get base repository path for worktree creation.

        Priority:
        1. Explicit path in task.context["worktree_base_repo"]
        2. Shared clone from multi_repo_manager
        """
        # Check for explicit base repo override
        explicit_base = task.context.get("worktree_base_repo")
        if explicit_base:
            base_path = Path(explicit_base).expanduser().resolve()
            if base_path.exists() and (base_path / ".git").exists():
                self.logger.debug(f"Using explicit base repo: {base_path}")
                return base_path
            else:
                self.logger.warning(f"Explicit worktree_base_repo not valid: {explicit_base}")

        # Use shared clone from multi_repo_manager
        if self.multi_repo_manager:
            try:
                return self.multi_repo_manager.ensure_repo(github_repo)
            except Exception as e:
                self.logger.error(f"Failed to get base repo from multi_repo_manager: {e}")

        return None

    def _sync_worktree_queued_tasks(self) -> None:
        """Move any task files the LLM wrote to the worktree's queues back to the main queue.

        When the Claude CLI subprocess runs in a worktree, the LLM may create
        subtask JSON files via the Write tool at .agent-communication/queues/<agent>/.
        These land in the worktree instead of the main repo's queue that agent
        workers actually poll.  This method finds those orphaned files and
        re-queues them through the canonical FileQueue.push() path.
        """
        if not self._active_worktree:
            return

        worktree_queue_dir = self._active_worktree / ".agent-communication" / "queues"
        if not worktree_queue_dir.exists():
            return

        # Don't sync from the main workspace back into itself
        main_queue_dir = self.queue.queue_dir
        try:
            if worktree_queue_dir.resolve() == main_queue_dir.resolve():
                return
        except OSError:
            return

        synced = 0
        for agent_dir in worktree_queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            # Skip non-agent directories (checkpoints, completed, etc.)
            queue_id = agent_dir.name
            if queue_id in ("checkpoints", "completed", "failed", "locks", "heartbeats", "malformed"):
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    data = json.loads(task_file.read_text())
                    synced_task = Task(**data)
                    self.queue.push(synced_task, synced_task.assigned_to)
                    task_file.unlink()
                    synced += 1
                    self.logger.info(
                        f"Synced worktree queue task {synced_task.id} → {synced_task.assigned_to}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to sync worktree task {task_file.name}: {e}")

        if synced:
            self.logger.info(f"Synced {synced} task(s) from worktree to main queue")

    def _cleanup_worktree(self, task: Task, success: bool) -> None:
        """Cleanup worktree after task completion based on config.

        Safety checks:
        - Skip cleanup if there are unpushed commits (data loss prevention)
        - Skip cleanup if there are uncommitted changes
        - Log warnings when skipping to help with debugging
        """
        if not self._active_worktree or not self.worktree_manager:
            return

        worktree_config = self.worktree_manager.config
        should_cleanup = (
            (success and worktree_config.cleanup_on_complete) or
            (not success and worktree_config.cleanup_on_failure)
        )

        if should_cleanup:
            # Safety check: don't delete worktrees with unpushed work
            has_unpushed = self.worktree_manager.has_unpushed_commits(self._active_worktree)
            has_uncommitted = self.worktree_manager.has_uncommitted_changes(self._active_worktree)

            if has_unpushed:
                self.logger.warning(
                    f"Skipping worktree cleanup - unpushed commits detected: {self._active_worktree}. "
                    f"Manual cleanup required after pushing changes."
                )
            elif has_uncommitted:
                self.logger.warning(
                    f"Skipping worktree cleanup - uncommitted changes detected: {self._active_worktree}. "
                    f"Manual cleanup required after committing/discarding changes."
                )
            else:
                try:
                    self.worktree_manager.remove_worktree(self._active_worktree, force=not success)
                    self.logger.info(f"Cleaned up worktree: {self._active_worktree}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup worktree: {e}")

        self._active_worktree = None

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
            workspace = self._get_working_directory(task)
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

    def _push_and_create_pr_if_needed(self, task: Task) -> None:
        """Push branch and create PR if the agent produced unpushed commits.

        Runs after the LLM finishes but before the task is marked completed,
        so the PR URL is available in task.context for downstream chain steps.
        Only acts when working in a worktree with actual unpushed commits.

        Intermediate workflow steps push their branch but skip PR creation —
        the terminal step (or pr_creator) handles that.
        """
        from ..utils.subprocess_utils import run_git_command, run_command, SubprocessError

        # Already has a PR (created by the LLM via MCP or _handle_success)
        if task.context.get("pr_url"):
            self.logger.debug(f"PR already exists for {task.id}: {task.context['pr_url']}")
            return

        # PR creation task with an implementation branch from upstream — create
        # the PR from that branch without needing a worktree
        impl_branch = task.context.get("implementation_branch")
        if task.context.get("pr_creation_step") and impl_branch:
            self._create_pr_from_branch(task, impl_branch)
            return

        # Only act if we have an active worktree with changes
        if not self._active_worktree or not self.worktree_manager:
            self.logger.debug(f"No active worktree for {task.id}, skipping PR creation")
            return

        has_unpushed = self.worktree_manager.has_unpushed_commits(self._active_worktree)
        branch_already_pushed = False
        if not has_unpushed:
            # LLM may have pushed the branch itself — check if it exists on the remote
            branch_already_pushed = self._remote_branch_exists(self._active_worktree)
            if not branch_already_pushed:
                self.logger.debug(f"No unpushed commits and no remote branch for {task.id}")
                return
            self.logger.debug(f"Branch already pushed to remote for {task.id}, will create PR only")

        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.debug(f"No github_repo in task context for {task.id}, skipping PR creation")
            return

        try:
            worktree = self._active_worktree

            # Get the current branch name
            result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree, check=False, timeout=10,
            )
            if result.returncode != 0:
                self.logger.warning("Could not determine branch name, skipping PR creation")
                return
            branch = result.stdout.strip()

            # Don't create PRs from main/master
            if branch in ("main", "master"):
                self.logger.debug(f"On {branch} branch for {task.id}, skipping PR creation")
                return

            # Push the branch (skip if LLM already pushed it)
            if not branch_already_pushed:
                self.logger.info(f"Pushing branch {branch} to origin")
                push_result = run_git_command(
                    ["push", "-u", "origin", branch],
                    cwd=worktree, check=False, timeout=60,
                )
                if push_result.returncode != 0:
                    self.logger.error(f"Failed to push branch: {push_result.stderr}")
                    return

            # Intermediate workflow steps: push code but skip PR creation.
            # Store the branch so downstream agents can create the PR later.
            if not self._is_at_terminal_workflow_step(task):
                task.context["implementation_branch"] = branch
                self.logger.info(
                    f"Intermediate step — pushed {branch} but skipped PR creation"
                )
                return

            self._create_pr_via_gh(task, github_repo, branch, cwd=worktree)

        except SubprocessError as e:
            self.logger.error(f"Subprocess error during PR creation: {e}")
        except Exception as e:
            self.logger.error(f"Error creating PR: {e}")

    def _manage_pr_lifecycle(self, task: Task) -> None:
        """Autonomously monitor CI, fix failures, and merge PR if repo opts in."""
        if not self._pr_lifecycle_manager:
            return
        if not self._pr_lifecycle_manager.should_manage(task):
            return

        try:
            merged = self._pr_lifecycle_manager.manage(task, self.config.id)
            if merged:
                self._sync_jira_status(
                    task, "Done",
                    comment=f"PR merged automatically: {task.context.get('pr_url')}",
                )
        except Exception as e:
            self.logger.error(f"PR lifecycle error for {task.id}: {e}")

    def _create_pr_from_branch(self, task: Task, branch: str) -> None:
        """Create a PR from an existing pushed branch (used by pr_creation_step tasks).

        Called when the terminal PR creation agent receives a task with an
        implementation_branch set by an upstream agent. No worktree needed —
        just runs `gh pr create --head <branch>` against the repo.
        """
        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.warning("No github_repo in context, cannot create PR from branch")
            return

        # Determine cwd — use shared clone if available, otherwise workspace
        cwd = self.workspace
        if self.multi_repo_manager:
            try:
                cwd = self.multi_repo_manager.ensure_repo(github_repo)
            except Exception:
                pass

        self._create_pr_via_gh(task, github_repo, branch, cwd=cwd)

    def _create_pr_via_gh(self, task: Task, github_repo: str, branch: str, *, cwd) -> None:
        """Create a PR via gh CLI. Shared by worktree and branch-based flows."""
        from ..utils.subprocess_utils import run_command, SubprocessError

        # Build a clean PR title — strip workflow prefixes
        pr_title = self._strip_chain_prefixes(task.title)[:70]

        pr_body = f"## Summary\n\n{task.context.get('user_goal', task.description)}"

        self.logger.info(f"Creating PR for {github_repo} from branch {branch}")
        try:
            pr_result = run_command(
                ["gh", "pr", "create",
                 "--repo", github_repo,
                 "--title", pr_title,
                 "--body", pr_body,
                 "--head", branch],
                cwd=cwd, check=False, timeout=30,
            )

            if pr_result.returncode == 0:
                pr_url = pr_result.stdout.strip()
                task.context["pr_url"] = pr_url
                self.logger.info(f"Created PR: {pr_url}")
                # Clean up orphaned subtask PRs/branches for fan-in tasks
                self._close_subtask_prs(task, pr_url)
                self._cleanup_subtask_branches(task)
            else:
                if "already exists" in pr_result.stderr:
                    self.logger.info("PR already exists for this branch")
                else:
                    self.logger.error(f"Failed to create PR: {pr_result.stderr}")
        except SubprocessError as e:
            self.logger.error(f"Failed to create PR: {e}")

    def _close_subtask_prs(self, task: Task, fan_in_pr_url: str) -> None:
        """Close orphaned PRs created by subtask LLMs. Best-effort.

        Subtask LLMs may create PRs via MCP despite prompt suppression.
        After the fan-in PR is created, close those orphans so they don't
        linger as duplicates.
        """
        if not task.context.get("fan_in"):
            return

        from ..utils.subprocess_utils import run_command

        parent_task_id = task.context.get("parent_task_id")
        if not parent_task_id:
            return

        parent = self.queue.find_task(parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        github_repo = task.context.get("github_repo")
        if not github_repo:
            return

        for sid in parent.subtask_ids:
            subtask = self.queue.get_completed(sid)
            if not subtask:
                continue
            pr_url = subtask.context.get("pr_url")
            if not pr_url:
                continue
            # Extract PR number from URL (e.g. .../pull/18 → 18)
            pr_number = pr_url.rstrip("/").split("/")[-1]
            self.logger.info(f"Closing orphaned subtask PR #{pr_number}")
            run_command(
                ["gh", "pr", "close", pr_number,
                 "--repo", github_repo,
                 "--comment", f"Superseded by fan-in PR {fan_in_pr_url}"],
                check=False, timeout=30,
            )

    def _cleanup_subtask_branches(self, task: Task) -> None:
        """Delete remote branches created by subtask LLMs. Best-effort.

        After the fan-in PR lands, subtask branches are stale. Clean them up
        so they don't clutter the remote.
        """
        if not task.context.get("fan_in"):
            return

        from ..utils.subprocess_utils import run_command

        parent_task_id = task.context.get("parent_task_id")
        if not parent_task_id:
            return

        parent = self.queue.find_task(parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        github_repo = task.context.get("github_repo")
        if not github_repo:
            return

        for sid in parent.subtask_ids:
            subtask = self.queue.get_completed(sid)
            if not subtask:
                continue
            branch = (
                subtask.context.get("implementation_branch")
                or subtask.context.get("worktree_branch")
            )
            if not branch:
                continue
            self.logger.info(f"Deleting subtask branch: {branch}")
            run_command(
                ["git", "push", "origin", "--delete", branch],
                check=False, timeout=30,
            )

    def _remote_branch_exists(self, worktree_path) -> bool:
        """Check if current branch exists on the remote (origin)."""
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        try:
            branch_result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree_path, check=False, timeout=10,
            )
            if branch_result.returncode != 0:
                return False
            branch = branch_result.stdout.strip()
            if branch in ("main", "master", "HEAD"):
                return False
            result = run_git_command(
                ["ls-remote", "--heads", "origin", branch],
                cwd=worktree_path, check=False, timeout=10,
            )
            return bool(result.stdout.strip())
        except SubprocessError:
            return False

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

        repo_path = self._get_working_directory(task)
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

    def _extract_pr_info_from_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Extract PR information from LLM response content.

        Parses the response for GitHub PR URLs created via MCP tools.
        Returns dict with pr_url, pr_number, owner, repo if found.
        """
        # Pattern for GitHub PR URLs: https://github.com/{owner}/{repo}/pull/{number}
        pr_pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'

        match = re.search(pr_pattern, response_content)
        if match:
            owner, repo, pr_number = match.groups()
            return {
                "pr_url": match.group(0),
                "pr_number": int(pr_number),
                "owner": owner,
                "repo": repo,
                "github_repo": f"{owner}/{repo}",
            }
        return None

    def _get_pr_info(self, task: Task, response) -> Optional[dict]:
        """Extract PR information from response or task context."""
        # Try extracting from response content
        pr_info = self._extract_pr_info_from_response(response.content)
        if pr_info:
            return pr_info

        # Check task context
        pr_url = task.context.get("pr_url")
        if not pr_url:
            return None

        match = re.search(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_url)
        if not match:
            return None

        owner, repo, pr_number = match.groups()
        return {
            "pr_url": pr_url,
            "pr_number": int(pr_number),
            "owner": owner,
            "repo": repo,
            "github_repo": f"{owner}/{repo}",
        }

    def _build_review_task(self, task: Task, pr_info: dict) -> Task:
        """Build code review task for a PR."""
        from datetime import datetime

        jira_key = task.context.get("jira_key", "UNKNOWN")
        pr_number = pr_info["pr_number"]

        return Task(
            id=f"review-{task.id}-{pr_number}",
            type=TaskType.REVIEW,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title=f"Review PR #{pr_number} - [{jira_key}] {task.title[:50]}",
            description=f"""Automated code review request for PR #{pr_number}.

## PR Information
- **PR URL**: {pr_info['pr_url']}
- **Repository**: {pr_info['github_repo']}
- **JIRA Ticket**: {jira_key}
- **Created by**: {self.config.id} agent

## Review Instructions
1. Fetch PR details and diff using `github_get_pr` and `github_get_pr_diff` MCP tools
2. Check CI status using `github_get_check_runs` with the PR branch
3. Review the diff against standard review criteria:
   - Correctness: Logic errors, edge cases, error handling
   - Security: Vulnerabilities, input validation, secrets
   - Performance: Inefficient patterns, N+1 queries
   - Readability: Code clarity, naming, documentation
   - Best Practices: Language conventions, test coverage
4. If CI checks are failing, include CI failures in your findings as CRITICAL
5. Post review comments on PR
6. Update JIRA with review summary
7. Transition JIRA status if appropriate (Approved/Changes Requested)
""",
            context={
                "jira_key": jira_key,
                "jira_url": task.context.get("jira_url"),
                "pr_number": pr_number,
                "pr_url": pr_info["pr_url"],
                "github_repo": pr_info["github_repo"],
                "branch_name": task.context.get("branch_name"),
                "workflow": task.context.get("workflow", "default"),
                "review_mode": True,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "implementation_branch": task.context.get("implementation_branch"),
                # Carry review cycle count so QA → Engineer loop is capped
                "_review_cycle_count": task.context.get("_review_cycle_count", 0),
            },
        )

    def _purge_orphaned_review_tasks(self) -> None:
        """Remove REVIEW/FIX tasks for PRs that already have an ESCALATION.

        On restart, stale review-chain tasks from before the cycle-count guard
        may still be queued.  If an escalation already exists for a PR, every
        REVIEW/FIX task for that PR is orphaned — processing them would restart
        a parallel chain the architect already owns.
        """
        queue_dir = self.queue.queue_dir
        completed_dir = self.queue.completed_dir

        # Step 1: collect PR URLs that have been escalated
        escalated_prs: set[str] = set()
        for search_dir in (queue_dir / "architect", completed_dir):
            if not search_dir.is_dir():
                continue
            for f in search_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("type") == TaskType.ESCALATION.value:
                    pr_url = data.get("context", {}).get("pr_url")
                    if pr_url:
                        escalated_prs.add(pr_url)

        if not escalated_prs:
            return

        # Step 2: remove REVIEW/FIX tasks whose pr_url matches an escalated PR
        purged = 0
        for sub in queue_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                task_type = data.get("type")
                if task_type not in (TaskType.REVIEW.value, TaskType.FIX.value):
                    continue
                pr_url = data.get("context", {}).get("pr_url")
                if pr_url in escalated_prs:
                    f.unlink()
                    purged += 1

        if purged:
            self.logger.info(
                f"Purged {purged} orphaned review-chain task(s) for "
                f"{len(escalated_prs)} escalated PR(s)"
            )

    def _queue_code_review_if_needed(self, task: Task, response) -> None:
        """
        Automatically queue a code review task if a PR was created.

        This ensures every PR gets reviewed by the QA agent,
        regardless of whether the creating agent remembered to queue the review.
        """
        # Skip if this agent IS the QA (avoid infinite loop); use base_id for replica support
        if self.config.base_id == "qa":
            return

        # Skip if task type is already a review or escalation
        if task.type in (TaskType.REVIEW, TaskType.ESCALATION):
            return

        # Chain tasks are routed by the workflow DAG which already includes
        # the QA step — creating a separate review task would duplicate it.
        if task.context.get("chain_step"):
            self.logger.debug(f"Skipping review for chain task {task.id}: DAG handles QA routing")
            return

        # Skip if this task already hit the escalation threshold — the review
        # loop escalated to the architect, so spawning another review would
        # restart a parallel chain (fork bomb).
        if task.context.get("_review_cycle_count", 0) >= MAX_REVIEW_CYCLES:
            self.logger.debug(
                f"Skipping review for {task.id}: cycle count "
                f"{task.context['_review_cycle_count']} >= {MAX_REVIEW_CYCLES} (already escalated)"
            )
            return

        # Get PR information
        pr_info = self._get_pr_info(task, response)
        if not pr_info:
            self.logger.debug(f"No PR found in task {task.id} - skipping code review queue")
            return

        # Build and queue review task
        review_task = self._build_review_task(task, pr_info)
        pr_number = pr_info["pr_number"]

        # Deduplicate: skip if this exact review task is already queued
        review_path = self.queue.queue_dir / "qa" / f"{review_task.id}.json"
        if review_path.exists():
            self.logger.debug(f"Review task {review_task.id} already queued, skipping")
            return

        try:
            self.queue.push(review_task, "qa")
            self.logger.info(
                f"🔍 Queued code review for PR #{pr_number} ({pr_info['github_repo']}) -> qa"
            )

            # Store PR URL in original task context for tracking
            task.context["pr_url"] = pr_info["pr_url"]
            task.context["pr_number"] = pr_number
            task.context["code_review_task_id"] = review_task.id

        except Exception as e:
            self.logger.error(f"Failed to queue code review for PR #{pr_number}: {e}")

    # -- QA → Engineer review feedback loop --

    def _queue_review_fix_if_needed(self, task: Task, response) -> None:
        """Deterministically queue a fix task to engineer when QA finds issues.

        Mirrors _queue_code_review_if_needed: that method hard-codes the
        Engineer → QA direction; this method hard-codes QA → Engineer.
        """
        if self.config.base_id != "qa":
            return
        if task.type != TaskType.REVIEW:
            return

        outcome = self._parse_review_outcome(response.content)
        if outcome.approved and not outcome.needs_fix:
            self._sync_jira_status(task, "Approved", comment=f"QA approved by {self.config.id}")
            return
        # Ambiguous (neither approved nor flagged) → treat as needs_fix
        if not outcome.needs_fix and not outcome.approved:
            self.logger.info("Ambiguous QA verdict (no APPROVE/issues) — treating as needs_fix")
            outcome = replace(
                outcome,
                has_major_issues=True,
                findings_summary=outcome.findings_summary or response.content[:500],
            )

        cycle_count = task.context.get("_review_cycle_count", 0) + 1

        if cycle_count > MAX_REVIEW_CYCLES:
            self._escalate_review_to_architect(task, outcome, cycle_count)
            self._sync_jira_status(
                task, "Changes Requested",
                comment=f"Escalated to architect after {cycle_count} review cycles",
            )
            return

        fix_task = self._build_review_fix_task(task, outcome, cycle_count)

        # Deduplicate: skip if fix task file already exists in engineer queue
        fix_path = self.queue.queue_dir / "engineer" / f"{fix_task.id}.json"
        if fix_path.exists():
            self.logger.debug(f"Review fix task {fix_task.id} already queued, skipping")
            return

        try:
            self.queue.push(fix_task, "engineer")
            self.logger.info(
                f"🔧 Queued review fix (cycle {cycle_count}/{MAX_REVIEW_CYCLES}) -> engineer"
            )
            self._sync_jira_status(
                task, "Changes Requested",
                comment=f"Review cycle {cycle_count}: {outcome.findings_summary[:200]}",
            )
        except Exception as e:
            self.logger.error(f"Failed to queue review fix task: {e}")

    def _parse_review_outcome(self, content: str) -> ReviewOutcome:
        """Parse QA response for review verdict using regex patterns."""
        if not content:
            return ReviewOutcome(
                approved=False, has_critical_issues=False,
                has_test_failures=False, has_change_requests=False,
                findings_summary="",
            )

        _NEGATIONS = ('no ', 'zero ', '0 ', 'without ', 'not ')

        def _matches(key: str) -> bool:
            flags = 0 if key in _CASE_SENSITIVE_KEYS else re.IGNORECASE
            for p in REVIEW_OUTCOME_PATTERNS[key]:
                m = re.search(p, content, flags)
                if m:
                    prefix = content[max(0, m.start() - 20):m.start()].lower()
                    if any(neg in prefix for neg in _NEGATIONS):
                        continue
                    return True
            return False

        approved = _matches("approve")
        has_critical = _matches("critical_issues")
        has_major = _matches("major_issues")
        has_test_fail = _matches("test_failures")
        has_changes = _matches("request_changes")

        # CRITICAL/MAJOR/HIGH override explicit APPROVE
        if has_critical or has_major or has_test_fail or has_changes:
            approved = False

        # Default-deny: severity-tagged findings without explicit APPROVE → needs fix.
        # Only exact APPROVE/LGTM keywords count as approval.
        if not approved and not (has_critical or has_major or has_test_fail or has_changes):
            if _SEVERITY_TAG_RE.search(content):
                has_major = True

        findings_summary, structured_findings = self._extract_review_findings(content)

        return ReviewOutcome(
            approved=approved,
            has_critical_issues=has_critical,
            has_test_failures=has_test_fail,
            has_change_requests=has_changes,
            has_major_issues=has_major,
            findings_summary=findings_summary,
            structured_findings=structured_findings,
        )

    def _extract_review_findings(self, content: str) -> tuple[str, List[QAFinding]]:
        """Extract findings from QA review output.

        Returns:
            tuple: (findings_summary: str, structured_findings: List[QAFinding])
            Tries new structured parser first, falls back to legacy regex extraction.
        """
        # Try new structured parsing first (handles code fences and inline JSON)
        structured_findings = self._parse_structured_findings(content)

        # If new parser didn't find anything, try legacy JSON block parsing
        if not structured_findings:
            structured_findings = []
            json_pattern = r'```json\s*\n(.*?)\n```'
            json_matches = re.findall(json_pattern, content, re.DOTALL)

            for json_block in json_matches:
                try:
                    findings_data = json.loads(json_block)
                    if isinstance(findings_data, list):
                        for item in findings_data:
                            if isinstance(item, dict):
                                # Create QAFinding from JSON object
                                finding = QAFinding(
                                    file=item.get('file', ''),
                                    line_number=item.get('line_number'),
                                    severity=item.get('severity', 'UNKNOWN'),
                                    description=item.get('description', ''),
                                    suggested_fix=item.get('suggested_fix'),
                                    category=item.get('category', 'general')
                                )
                                structured_findings.append(finding)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    self.logger.debug(f"Failed to parse JSON finding: {e}")
                    continue

        # Extract severity-tagged lines for text summary
        findings_text = []
        for line in content.splitlines():
            stripped = line.strip()
            # Case-sensitive: we want structured output tags (CRITICAL, HIGH, …),
            # not prose that happens to contain the word. _parse_review_outcome
            # uses IGNORECASE for leniency; here we want precision.
            if re.match(r'^(CRITICAL|HIGH|MAJOR|MEDIUM|MINOR|LOW|SUGGESTION)\b', stripped):
                findings_text.append(stripped)

        # If we have structured findings, build summary from them
        if structured_findings:
            summary_lines = []
            for finding in structured_findings:
                location = f"{finding.file}"
                if finding.line_number:
                    location += f":{finding.line_number}"
                summary_lines.append(f"{finding.severity}: {finding.description} ({location})")
            findings_summary = "\n".join(summary_lines)
        elif findings_text:
            findings_summary = "\n".join(findings_text)
        else:
            # Fall back to first 500 chars if no tagged lines found
            findings_summary = content[:500]

        return findings_summary, structured_findings

    def _parse_structured_findings(self, content: str) -> Optional[List[QAFinding]]:
        """Extract structured JSON findings from QA response.

        Looks for JSON blocks in code fences (```json...```) or inline.
        Supports both array format and object format with 'findings' key.
        Falls back to None if no structured findings found.
        """
        import json
        import re

        # Try to find JSON block in code fence first
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                findings_data = json.loads(json_match.group(1))

                # Handle object wrapper with 'findings' array
                if isinstance(findings_data, dict) and 'findings' in findings_data:
                    findings_list = findings_data['findings']
                # Handle direct array format
                elif isinstance(findings_data, list):
                    findings_list = findings_data
                else:
                    self.logger.debug("JSON block doesn't contain findings array")
                    return None

                # Convert to QAFinding objects
                parsed_findings = []
                for item in findings_list:
                    if isinstance(item, dict):
                        finding = QAFinding(
                            file=item.get('file', ''),
                            line_number=item.get('line_number') or item.get('line'),
                            severity=item.get('severity', 'UNKNOWN'),
                            description=item.get('description', ''),
                            suggested_fix=item.get('suggested_fix'),
                            category=item.get('category', 'general')
                        )
                        parsed_findings.append(finding)

                return parsed_findings if parsed_findings else None
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.warning(f"Failed to parse code fence JSON findings: {e}")

        # Try to find inline JSON object with "findings" key
        try:
            # Look for complete JSON object with findings array
            json_pattern = r'\{[^{}]*"findings"\s*:\s*\[[^\]]*\][^{}]*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                findings_data = json.loads(json_match.group(0))
                if 'findings' in findings_data:
                    findings_list = findings_data['findings']
                    parsed_findings = []
                    for item in findings_list:
                        if isinstance(item, dict):
                            finding = QAFinding(
                                file=item.get('file', ''),
                                line_number=item.get('line_number') or item.get('line'),
                                severity=item.get('severity', 'UNKNOWN'),
                                description=item.get('description', ''),
                                suggested_fix=item.get('suggested_fix'),
                                category=item.get('category', 'general')
                            )
                            parsed_findings.append(finding)
                    return parsed_findings if parsed_findings else None
        except Exception as e:
            self.logger.debug(f"Inline JSON parse failed: {e}")

        return None  # No structured findings found

    def _format_findings_checklist(self, findings: List[QAFinding]) -> str:
        """Format structured findings as numbered checklist."""
        lines = []

        for i, finding in enumerate(findings, 1):
            # Format location
            location = ""
            if finding.file and finding.line_number:
                location = f" ({finding.file}:{finding.line_number})"
            elif finding.file:
                location = f" ({finding.file})"

            # Format severity with emoji
            severity_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MAJOR": "🟡",
                "MEDIUM": "🔵",
                "MINOR": "⚪",
                "LOW": "⚪",
                "SUGGESTION": "💡",
            }
            emoji = severity_emoji.get(finding.severity, "")

            lines.append(f"### {i}. {emoji} {finding.severity}: {finding.category.title()}{location}")
            lines.append(f"**Issue**: {finding.description}")

            if finding.suggested_fix:
                lines.append(f"**Suggested Fix**: {finding.suggested_fix}")

            lines.append("")  # blank line

        return "\n".join(lines)

    def _build_review_fix_task(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> Task:
        """Build fix task with structured checklist."""
        from datetime import datetime

        jira_key = task.context.get("jira_key", "UNKNOWN")
        pr_url = task.context.get("pr_url", "")
        pr_number = task.context.get("pr_number", "")

        # Check if we have structured findings
        has_structured_findings = outcome.structured_findings and len(outcome.structured_findings) > 0

        # Build description with numbered checklist or legacy format
        if has_structured_findings:
            # Generate checklist using existing formatter
            checklist = self._format_findings_checklist(outcome.structured_findings)
            total_count = len(outcome.structured_findings)

            description = f"""QA review found {total_count} issue(s) that need fixing.

## Summary
{outcome.findings_summary}

## Issues to Address

{checklist}

## Instructions
1. Review each finding above
2. Fix the issues in the specified files/lines
3. Run tests to verify fixes: `pytest tests/`
4. Run linting: `pylint src/` or appropriate linter
5. Commit and push your changes
6. The review will be automatically re-queued

## Context
- **PR**: {pr_url}
- **JIRA**: {jira_key}
- **Review Cycle**: {cycle_count} of {MAX_REVIEW_CYCLES}
"""
        else:
            # Legacy format (backward compatible)
            description = f"""QA review found issues that need fixing.

## Review Findings
{outcome.findings_summary}

## Instructions
1. Review the findings above
2. Fix the identified issues
3. Run tests to verify fixes
4. Commit and push your changes

## Context
- **PR**: {pr_url}
- **JIRA**: {jira_key}
- **Review Cycle**: {cycle_count} of {MAX_REVIEW_CYCLES}
"""

        # Build context (strip review_* keys, preserve essential context)
        fix_context = {
            k: v for k, v in task.context.items()
            if not k.startswith("review_")
        }
        fix_context["_review_cycle_count"] = cycle_count
        fix_context["pr_url"] = pr_url
        fix_context["pr_number"] = pr_number
        fix_context["github_repo"] = task.context.get("github_repo")
        fix_context["jira_key"] = jira_key
        fix_context["workflow"] = task.context.get("workflow", "full")

        # Preserve engineer's branch so fix cycle reuses the same worktree
        if task.context.get("implementation_branch"):
            fix_context["worktree_branch"] = task.context["implementation_branch"]

        # Store structured findings in context for programmatic access
        if has_structured_findings:
            # Serialize findings to dict for context storage
            findings_dicts = []
            for finding in outcome.structured_findings:
                findings_dicts.append({
                    "file": finding.file,
                    "line_number": finding.line_number,
                    "severity": finding.severity,
                    "category": finding.category,
                    "description": finding.description,
                    "suggested_fix": finding.suggested_fix,
                })

            fix_context["structured_findings"] = {
                "findings": findings_dicts,
                "summary": outcome.findings_summary,
                "total_count": len(outcome.structured_findings),
                "critical_count": sum(1 for f in outcome.structured_findings if f.severity == "CRITICAL"),
                "high_count": sum(1 for f in outcome.structured_findings if f.severity == "HIGH"),
                "major_count": sum(1 for f in outcome.structured_findings if f.severity == "MAJOR"),
            }

        # Build acceptance criteria
        if has_structured_findings:
            total_count = len(outcome.structured_findings)
            acceptance_criteria = [
                f"All {total_count} issues addressed",
                "Tests pass",
                "Linting passes",
            ]
        else:
            acceptance_criteria = [
                "All identified issues addressed",
                "Tests pass",
            ]

        return Task(
            id=f"review-fix-{task.id[:12]}-c{cycle_count}",
            type=TaskType.FIX,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title=f"Fix review issues (cycle {cycle_count}) - [{jira_key}]",
            description=description,
            context=fix_context,
            acceptance_criteria=acceptance_criteria,
        )

    def _escalate_review_to_architect(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> None:
        """Escalate to architect after too many failed review cycles."""
        from datetime import datetime

        jira_key = task.context.get("jira_key", "UNKNOWN")

        escalation_task = Task(
            id=f"review-escalation-{task.id[:12]}",
            type=TaskType.ESCALATION,
            status=TaskStatus.PENDING,
            priority=max(1, task.priority - 1),  # Lower number = higher priority
            created_by=self.config.id,
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title=f"Review escalation ({cycle_count} cycles) - [{jira_key}]",
            description=f"""QA and Engineer failed to resolve review issues after {cycle_count} cycles.

## Last Review Findings
{outcome.findings_summary}

## Action Required
- Replan the implementation approach
- Consider breaking the task into smaller pieces
- Provide more detailed architectural guidance
""",
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "_review_cycle_count": cycle_count,
                "escalation_reason": f"Review loop exceeded {MAX_REVIEW_CYCLES} cycles",
            },
        )

        try:
            self.queue.push(escalation_task, "architect")
            self.logger.warning(
                f"⚠️  Review escalated to architect after {cycle_count} cycles"
            )
        except Exception as e:
            self.logger.error(f"Failed to escalate review to architect: {e}")

    # -- Fan-in task creation --

    def _check_and_create_fan_in_task(self, task: Task) -> None:
        """Check if this subtask completion triggers fan-in task creation.

        When a subtask completes, checks if all siblings are also complete.
        If so, creates a fan-in task that aggregates results and continues workflow.
        """
        if not task.parent_task_id:
            return

        # This is a subtask - check if all siblings are done
        parent = self.queue.find_task(task.parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        if self.queue.check_subtasks_complete(parent.id, parent.subtask_ids):
            # All subtasks done - create fan-in task
            if not self.queue._fan_in_already_created(parent.id):
                completed_subtasks = [
                    self.queue.get_completed(sid) for sid in parent.subtask_ids
                ]
                completed_subtasks = [s for s in completed_subtasks if s is not None]
                fan_in_task = self.queue.create_fan_in_task(parent, completed_subtasks)
                self.queue.push(fan_in_task, fan_in_task.assigned_to)
                self.logger.info(
                    f"🔀 All subtasks complete - created fan-in task {fan_in_task.id}"
                )
        else:
            self.logger.info(
                f"Subtask {task.id} complete, waiting for siblings"
            )

    # -- Task decomposition --

    def _should_decompose_task(self, task: Task) -> bool:
        """Check if task should be decomposed into subtasks.

        Only applies to architect-created tasks with plans.
        Uses TaskDecomposer heuristics (estimated lines > threshold).
        """
        if not task.plan:
            return False

        # Don't decompose subtasks (max depth = 1)
        if task.parent_task_id:
            return False

        # Estimate lines: files_to_modify count * 15 lines/file (rough heuristic)
        estimated_lines = len(task.plan.files_to_modify) * 15 if task.plan.files_to_modify else 0

        from .task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        return decomposer.should_decompose(task.plan, estimated_lines)

    def _decompose_and_queue_subtasks(self, task: Task) -> None:
        """Decompose task into subtasks and queue them to engineer.

        Replaces normal workflow routing - subtasks will each flow through
        the workflow individually, and fan-in will aggregate them at completion.
        """
        from .task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()
        estimated_lines = len(task.plan.files_to_modify) * 15 if task.plan.files_to_modify else 0

        self.logger.info(
            f"Decomposing task {task.id} into parallel subtasks "
            f"(estimated {estimated_lines} lines across {len(task.plan.files_to_modify)} files)"
        )

        subtasks = decomposer.decompose(task, task.plan, estimated_lines)

        # Queue each subtask to engineer
        for subtask in subtasks:
            self.queue.push(subtask, "engineer")
            self.logger.info(f"  ✅ Queued subtask: {subtask.id} ({subtask.title})")

        # Update parent task with subtask IDs and save to completed
        # (parent is now just a container for subtasks)
        task.subtask_ids = [st.id for st in subtasks]
        task.result_summary = f"Decomposed into {len(subtasks)} subtasks"
        self.queue.update(task)

        self.logger.info(
            f"🔀 Task {task.id} decomposed into {len(subtasks)} parallel subtasks"
        )

    # -- Workflow chain enforcement --

    def _enforce_workflow_chain(self, task: Task, response, routing_signal=None) -> None:
        """Queue next agent in workflow using DAG executor.

        Supports both legacy linear workflows and new DAG workflows with conditions.
        """
        # Task decomposition: architect auto-decomposes large tasks before routing to engineer
        if self.config.base_id == "architect" and task.plan:
            if self._should_decompose_task(task):
                self._decompose_and_queue_subtasks(task)
                return

        # Preview tasks route back to architect for review, not to QA
        if task.type == TaskType.PREVIEW and self.config.base_id == 'engineer':
            self._route_to_agent(task, 'architect', 'preview_review')
            return

        # REVIEW/FIX tasks are routed by _queue_code_review_if_needed and
        # _queue_review_fix_if_needed respectively — letting them also route
        # through the DAG creates a duplicate-routing feedback loop.
        if task.type in (TaskType.REVIEW, TaskType.FIX):
            self.logger.debug(
                f"Skipping workflow chain for {task.id}: "
                f"task type {task.type.value} handled by dedicated review routing"
            )
            return

        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            self.logger.debug(
                f"No workflow chain for {task.id}: "
                f"workflow={workflow_name!r}, available={list(self._workflows_config.keys())}"
            )
            # No workflow defined - handle routing signal if present
            if routing_signal:
                validated = validate_routing_signal(
                    routing_signal, self.config.base_id, get_type_str(task.type), self._agents_config,
                )
                if validated and validated != WORKFLOW_COMPLETE:
                    self._route_to_agent(task, validated, routing_signal.reason)
                log_routing_decision(
                    self.workspace, task.id, self.config.id,
                    routing_signal, validated, used_fallback=False,
                )
            return

        # Get workflow definition and convert to DAG
        workflow_def = self._workflows_config[workflow_name]
        try:
            workflow_dag = workflow_def.to_dag(workflow_name)
        except Exception as e:
            self.logger.error(f"Failed to build workflow DAG for {workflow_name}: {e}")
            return

        # Single-agent workflows don't need routing
        if len(workflow_dag.get_all_agents()) <= 1:
            if routing_signal:
                self.logger.debug("Routing signal discarded: single-agent workflow")
            return

        # Execute workflow step using DAG executor
        try:
            routed = self._workflow_executor.execute_step(
                workflow=workflow_dag,
                task=task,
                response=response,
                current_agent_id=self.config.base_id,
                routing_signal=routing_signal,
                context=self._build_workflow_context(task),
            )

            # Terminal step with no routing — check if pr_creator should take over
            if not routed:
                self._queue_pr_creation_if_needed(task, workflow_def)

            # Log routing decision
            if routing_signal:
                log_routing_decision(
                    self.workspace, task.id, self.config.id,
                    routing_signal, None, used_fallback=not routed,
                )

            # Session logging
            if routed:
                self._session_logger.log(
                    "workflow_routing",
                    workflow=workflow_name,
                    signal=routing_signal.target_agent if routing_signal else None,
                )
        except Exception as e:
            self.logger.error(f"Workflow execution failed for task {task.id}: {e}")

    def _is_at_terminal_workflow_step(self, task: Task) -> bool:
        """Check if the current agent is at the last step in the workflow DAG.

        Returns True for standalone tasks (no workflow) to preserve backward
        compatibility — standalone agents should always be allowed to create PRs.
        """
        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            return True

        workflow_def = self._workflows_config[workflow_name]
        try:
            dag = workflow_def.to_dag(workflow_name)
        except Exception:
            return True

        # Prefer explicit workflow_step from chain context
        step_id = task.context.get("workflow_step")
        if step_id and step_id in dag.steps:
            return dag.is_terminal_step(step_id)

        # Fallback: find the step for this agent's base_id
        for step in dag.steps.values():
            if step.agent == self.config.base_id:
                return dag.is_terminal_step(step.id)

        return True

    def _get_changed_files(self) -> List[str]:
        """Get list of changed files from git diff (staged and unstaged)."""
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        try:
            result = run_git_command(
                ["diff", "--name-only", "HEAD"],
                cwd=self.workspace,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                self.logger.debug(f"git diff failed: {result.stderr}")
                return []
            return [f.strip() for f in result.stdout.split("\n") if f.strip()]
        except SubprocessError:
            self.logger.warning("git diff timed out")
            return []
        except Exception as e:
            self.logger.debug(f"Failed to get changed files: {e}")
            return []

    def _build_workflow_context(self, task: Task) -> Dict[str, Any]:
        """Build context dict for workflow condition evaluation."""
        context = {}

        # Prefer task context, fallback to git diff
        if task.context and "changed_files" in task.context:
            context["changed_files"] = task.context["changed_files"]
        else:
            changed_files = self._get_changed_files()
            if changed_files:
                context["changed_files"] = changed_files

        # Add test results if available
        if task.context and "test_result" in task.context:
            context["test_result"] = task.context["test_result"]

        return context

    def _route_to_agent(self, task: Task, target_agent: str, reason: str) -> None:
        if self._is_chain_task_already_queued(target_agent, task.id):
            self.logger.debug(f"Chain task for {target_agent} already queued from {task.id}")
            return

        chain_task = self._build_chain_task(task, target_agent)
        try:
            self.queue.push(chain_task, target_agent)
            self.logger.info(f"🔗 Routed to {target_agent} (signal): {reason}")
            self._session_logger.log(
                "workflow_chain",
                next_agent=target_agent,
                reason=reason,
            )
        except Exception as e:
            self.logger.error(f"Failed to route task to {target_agent}: {e}")

    def _is_chain_task_already_queued(self, next_agent: str, source_task_id: str) -> bool:
        """O(1) file existence check using deterministic chain task ID.

        Only checks the pending queue directory. If the task was already picked
        up (moved to completed/in-progress), this won't detect it — acceptable
        because _handle_successful_response only runs once per task lifecycle.
        """
        chain_id = f"chain-{source_task_id}-{next_agent}"
        queue_path = self.queue.queue_dir / next_agent / f"{chain_id}.json"
        return queue_path.exists()

    @staticmethod
    def _strip_chain_prefixes(title: str) -> str:
        """Remove accumulated [chain]/[pr] prefixes so re-wrapping adds exactly one."""
        while title.startswith(("[chain] ", "[pr] ")):
            title = title[len("[chain] "):] if title.startswith("[chain] ") else title[len("[pr] "):]
        return title

    def _build_chain_task(self, task: Task, next_agent: str) -> Task:
        """Create a continuation task for the next agent in the chain."""
        from datetime import datetime

        chain_id = f"chain-{task.id}-{next_agent}"
        task_type = CHAIN_TASK_TYPES.get(next_agent, task.type)

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to=next_agent,
            created_at=datetime.now(timezone.utc),
            title=f"[chain] {self._strip_chain_prefixes(task.title)}",
            description=task.description,
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "chain_step": True,
            },
        )

    def _queue_pr_creation_if_needed(self, task: Task, workflow) -> None:
        """Queue a PR creation task when the last agent in the chain completes.

        The workflow's pr_creator field designates which agent should open the PR.
        Without this, the chain ends silently after the last agent finishes.
        """
        pr_creator = getattr(workflow, "pr_creator", None)
        if not pr_creator:
            return

        if task.context.get("pr_creation_step"):
            return

        if task.context.get("pr_url"):
            return

        # Deterministic ID with -pr suffix to avoid collision with normal chain tasks
        pr_task_id = f"chain-{task.id}-{pr_creator}-pr"
        queue_path = self.queue.queue_dir / pr_creator / f"{pr_task_id}.json"
        if queue_path.exists():
            self.logger.debug(f"PR creation task {pr_task_id} already queued, skipping")
            return

        from datetime import datetime

        pr_task = Task(
            id=pr_task_id,
            type=TaskType.PR_REQUEST,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to=pr_creator,
            created_at=datetime.now(timezone.utc),
            title=f"[pr] {self._strip_chain_prefixes(task.title)}",
            description=task.description,
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "pr_creation_step": True,
            },
        )

        try:
            self.queue.push(pr_task, pr_creator)
            self.logger.info(f"📦 Queued PR creation for {pr_creator} from task {task.id}")
            self._session_logger.log(
                "pr_creation_queued",
                pr_creator=pr_creator,
                source_task=task.id,
            )
        except Exception as e:
            self.logger.error(f"Failed to queue PR creation task for {pr_creator}: {e}")
