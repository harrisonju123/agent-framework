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
    from .config import AgentDefinition, WorkflowDefinition

from .task import Task, TaskStatus, TaskType
from .task_validator import validate_task, ValidationResult
from .activity import ActivityManager, AgentActivity, AgentStatus, CurrentTask, ActivityEvent, TaskPhase, ToolActivity
from .team_composer import compose_default_team, compose_team
from ..llm.base import LLMBackend, LLMRequest, LLMResponse
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler
from ..workspace.worktree_manager import WorktreeManager, WorktreeConfig
from ..utils.rich_logging import ContextLogger, setup_rich_logging
from ..utils.type_helpers import get_type_str

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
        team_mode_min_workflow: str = "standard",
        team_mode_default_model: str = "sonnet",
        agent_definition: Optional["AgentDefinition"] = None,
        workflows_config: Optional[Dict[str, "WorkflowDefinition"]] = None,
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
        self._team_mode_min_workflow = team_mode_min_workflow
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
            last_updated=datetime.utcnow()
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
                        last_updated=datetime.utcnow()
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
            task.last_error = f"Task validation failed: {'; '.join(validation.errors)}"
            task.mark_failed(self.config.id)
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
            last_updated=datetime.utcnow()
        ))

        # Append start event
        self.activity_manager.append_event(ActivityEvent(
            type="start",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.utcnow()
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

        # Code review runs first: if a PR is found, it writes pr_url into
        # task.context, which chain enforcement then sees and correctly skips.
        self.logger.debug(f"Checking if code review needed for {task.id}")
        self._queue_code_review_if_needed(task, response)

        # QA review feedback: deterministically queue fix task to engineer
        # when QA finds issues, mirroring the hard-coded code review above
        self._queue_review_fix_if_needed(task, response)

        # Enforce workflow chain: queue next agent if no PR was created
        self._enforce_workflow_chain(task, response)

        # Log metrics and events
        self._log_task_completion_metrics(task, response, task_start_time)

    def _log_task_completion_metrics(self, task: Task, response, task_start_time) -> None:
        """Log token usage, cost, and completion events."""
        from datetime import datetime

        total_tokens = response.input_tokens + response.output_tokens
        budget = self._get_token_budget(task.type)
        cost = self._estimate_cost(response)

        duration = (datetime.utcnow() - task_start_time).total_seconds()
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
                    timestamp=datetime.utcnow()
                ))

        # Append complete event
        duration_ms = int((datetime.utcnow() - task_start_time).total_seconds() * 1000)
        pr_url = task.context.get("pr_url")
        self.activity_manager.append_event(ActivityEvent(
            type="complete",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            pr_url=pr_url,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost
        ))

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

        self.activity_manager.append_event(ActivityEvent(
            type="fail",
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.utcnow(),
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
            last_updated=datetime.utcnow()
        ))

        task_succeeded = task.status == TaskStatus.COMPLETED
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

    async def _handle_task(self, task: Task) -> None:
        """Handle task execution with retry/escalation logic."""
        from datetime import datetime

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
        task_start_time = datetime.utcnow()

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
                        started_at=datetime.utcnow(),
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

                # Layer 1: agent's configured teammates (peer-engineer, test-runner, etc.)
                if self._agent_definition and self._agent_definition.teammates:
                    configured = compose_default_team(
                        self._agent_definition,
                        default_model=self._team_mode_default_model,
                    )
                    if configured:
                        team_agents.update(configured)

                # Layer 2: workflow-required agents (QA for standard, engineer+QA for full)
                workflow = task.context.get("workflow", "full")
                workflow_teammates = compose_team(
                    task.context, workflow, self._agents_config,
                    min_workflow=self._team_mode_min_workflow,
                    default_model=self._team_mode_default_model,
                    caller_agent_id=self.config.id,
                )
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
                    timestamp=datetime.utcnow(),
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

            self.activity_manager.append_event(ActivityEvent(
                type="fail",
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.utcnow(),
                retry_count=task.retry_count,
                error_message=task.last_error
            ))

            await self._handle_failure(task)

        finally:
            self._cleanup_task_execution(task, lock)

    async def _handle_failure(self, task: Task) -> None:
        """
        Handle task failure with retry/escalation logic.

        Ported from scripts/async-agent-runner.sh lines 374-394.
        """
        if task.retry_count >= self.retry_handler.max_retries:
            # Max retries exceeded - mark as failed
            self.logger.error(
                f"Task {task.id} has failed {task.retry_count} times "
                f"(max: {self.retry_handler.max_retries})"
            )
            task.mark_failed(self.config.id)
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
        task_dict["logged_at"] = datetime.utcnow().isoformat()
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

    def _handle_shadow_mode_comparison(self, task: Task) -> str:
        """Generate and compare both prompts in shadow mode, return legacy prompt."""
        legacy_prompt = self._build_prompt_legacy(task)
        optimized_prompt = self._build_prompt_optimized(task)

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

        # Determine which prompt to use
        if shadow_mode:
            prompt = self._handle_shadow_mode_comparison(task)
        elif use_optimizations:
            prompt = self._build_prompt_optimized(task)
        else:
            prompt = self._build_prompt_legacy(task)

        # Log prompt preview for debugging (sanitized)
        if self.logger.isEnabledFor(logging.DEBUG):
            prompt_preview = prompt[:500].replace(task.id, "TASK_ID")
            if hasattr(task, 'context') and task.context.get('jira_key'):
                prompt_preview = prompt_preview.replace(task.context['jira_key'], "JIRA-XXX")
            self.logger.debug(f"Built prompt preview (first 500 chars): {prompt_preview}...")

        # Append test failure context if present
        prompt = self._append_test_failure_context(prompt, task)

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
                "timestamp": datetime.utcnow().isoformat(),
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
        jira_server = self.jira_config.server if self.jira_config else "jira.example.com"

        return f"""
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

    def _build_github_guidance(self, github_repo: str, jira_key: str) -> str:
        """Build GitHub integration guidance for MCP."""
        owner, repo = github_repo.split("/")

        # Get formatting patterns from config
        branch_pattern = "{type}/{ticket_id}-{slug}"
        pr_title_pattern = "[{ticket_id}] {title}"
        if self.github_config:
            branch_pattern = self.github_config.branch_pattern
            pr_title_pattern = self.github_config.pr_title_pattern

        return f"""
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

    def _build_error_handling_guidance(self) -> str:
        """Build error handling guidance for MCP tools."""
        return """
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

    def _build_prompt_legacy(self, task: Task) -> str:
        """Build prompt using legacy format (original implementation)."""
        task_json = task.model_dump_json(indent=2)

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

        return f"""You are {self.config.id} working on an asynchronous task.

TASK DETAILS:
{task_json}

{mcp_guidance}
YOUR RESPONSIBILITIES:
{self.config.prompt}

IMPORTANT:
- Complete the task described above
- Create any follow-up tasks by writing JSON files to other agents' queues
- Use unique task IDs (timestamp or UUID)
- Set depends_on array for tasks that depend on this one completing
- This task will be automatically marked as completed when you're done
"""

    def _build_prompt_optimized(self, task: Task) -> str:
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

        # Build optimized prompt (shorter, focused on essentials)
        return f"""You are {self.config.id} working on an asynchronous task.

TASK:
{task_json}

{context_note}{dep_context}
{self.config.prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _get_working_directory(self, task: Task) -> Path:
        """Get working directory for task (worktree, target repo, or framework workspace).

        Priority:
        1. If worktree mode enabled (config or task override), create isolated worktree
        2. If multi_repo_manager available, use shared clone
        3. Fall back to framework workspace
        """
        github_repo = task.context.get("github_repo")

        # Check if worktree mode should be used
        use_worktree = self._should_use_worktree(task)

        if use_worktree and github_repo and self.worktree_manager:
            # Get base repo path (shared clone or explicit override)
            base_repo = self._get_base_repo_for_worktree(task, github_repo)

            if base_repo:
                # Create worktree for isolated work
                # Include task_id[:8] for uniqueness to avoid branch collisions on retries
                jira_key = task.context.get("jira_key", "task")
                task_short = task.id[:8]
                branch_name = f"agent/{self.config.id}/{jira_key}-{task_short}"

                try:
                    worktree_path = self.worktree_manager.create_worktree(
                        base_repo=base_repo,
                        branch_name=branch_name,
                        agent_id=self.config.id,
                        task_id=task.id,
                        owner_repo=github_repo,
                    )
                    self._active_worktree = worktree_path
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
                    timestamp=datetime.utcnow(),
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
        """
        pause_file = self.workspace / PAUSE_SIGNAL_FILE
        health_pause_file = self.workspace / ".agent-communication" / "PAUSE_INTAKE"
        return pause_file.exists() or health_pause_file.exists()

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
                timestamp=datetime.utcnow()
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
            created_at=datetime.utcnow(),
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
                "workflow": task.context.get("workflow", "standard"),
                "review_mode": True,
                "source_task_id": task.id,
                "source_agent": self.config.id,
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
        """Build a fix task for the engineer with QA review findings."""
        from datetime import datetime

        jira_key = task.context.get("jira_key", "UNKNOWN")
        pr_url = task.context.get("pr_url", "")
        pr_number = task.context.get("pr_number", "")

        # Build numbered checklist from structured findings
        checklist_items = []
        if outcome.structured_findings:
            for idx, finding in enumerate(outcome.structured_findings, 1):
                # Format: [ ] **SEVERITY** (category): file:line
                location = finding.file
                if finding.line_number:
                    location += f":{finding.line_number}"

                item_lines = [f"{idx}. [ ] **{finding.severity}** ({finding.category}): {location}"]
                item_lines.append(f"    Issue: {finding.description}")
                if finding.suggested_fix:
                    item_lines.append(f"    Fix: {finding.suggested_fix}")

                checklist_items.append("\n".join(item_lines))

            checklist = "\n\n".join(checklist_items)
        else:
            # Fallback to legacy text format if no structured findings
            checklist = outcome.findings_summary

        return Task(
            id=f"review-fix-{task.id[:12]}-c{cycle_count}",
            type=TaskType.FIX,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to="engineer",
            created_at=datetime.utcnow(),
            title=f"Fix review issues (cycle {cycle_count}) - [{jira_key}]",
            description=f"""QA review found issues that need fixing.

## Review Findings

{checklist}

## Context
- **PR**: {pr_url}
- **JIRA**: {jira_key}
- **Review cycle**: {cycle_count}/{MAX_REVIEW_CYCLES}
- **Critical issues**: {outcome.has_critical_issues}
- **Major issues**: {outcome.has_major_issues}
- **Test failures**: {outcome.has_test_failures}
- **Changes requested**: {outcome.has_change_requests}

## Instructions
1. Fetch and read ALL review comments on the PR using `github_get_pr_comments`
2. Check CI status using `github_get_check_runs` — fix any CI failures
3. Address every review comment and all findings listed above
4. Fix any failing tests
5. Commit and push to the existing branch
6. The system will automatically re-queue a review to QA
""",
            context={
                **{k: v for k, v in task.context.items() if not k.startswith("review_")},
                "pr_url": pr_url,
                "pr_number": pr_number,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "_review_cycle_count": cycle_count,
            },
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
            created_at=datetime.utcnow(),
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

    # -- Workflow chain enforcement --

    def _enforce_workflow_chain(self, task: Task, response) -> None:
        """Queue next agent in the workflow chain when no PR was created.

        Complements _queue_code_review_if_needed: if a PR exists, review
        gets queued (existing path) and the chain stops. If no PR, this
        method forwards the task to the next agent in the configured chain.
        """
        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            return

        workflow = self._workflows_config[workflow_name]
        agents = workflow.agents
        if len(agents) <= 1:
            return

        # Skip when team mode already handled this workflow
        if self._team_mode_handled_workflow(task):
            return

        # Skip if a PR was created — workflow reached its natural endpoint
        pr_info = self._get_pr_info(task, response)
        if pr_info:
            return

        # Find current agent position, determine next
        current = self.config.base_id
        try:
            idx = agents.index(current)
        except ValueError:
            return
        if idx >= len(agents) - 1:
            return

        next_agent = agents[idx + 1]

        # O(1) duplicate check via deterministic task ID
        if self._is_chain_task_already_queued(next_agent, task.id):
            self.logger.debug(
                f"Chain task for {next_agent} already queued from {task.id}"
            )
            return

        chain_task = self._build_chain_task(task, next_agent)
        try:
            self.queue.push(chain_task, next_agent)
            self.logger.info(
                f"🔗 Workflow chain: queued {next_agent} for task {task.id}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to queue chain task for {next_agent}: {e}"
            )

    def _team_mode_handled_workflow(self, task: Task) -> bool:
        """Return True if team mode already handled this workflow's agents.

        Mirrors the check in _handle_task (and compose_team): team_override=True
        forces teams on regardless of rank, team_override=False skips teams.
        """
        from .team_composer import WORKFLOW_RANK

        if not self._team_mode_enabled:
            return False

        team_override = task.context.get("team_override")
        if team_override is False:
            return False

        # team_override=True forces team mode regardless of workflow rank
        if team_override is True:
            return True

        workflow = task.context.get("workflow", "full")
        workflow_rank = WORKFLOW_RANK.get(workflow, 0)
        min_rank = WORKFLOW_RANK.get(self._team_mode_min_workflow, 1)
        return workflow_rank >= min_rank

    def _is_chain_task_already_queued(self, next_agent: str, source_task_id: str) -> bool:
        """O(1) file existence check using deterministic chain task ID.

        Only checks the pending queue directory. If the task was already picked
        up (moved to completed/in-progress), this won't detect it — acceptable
        because _handle_successful_response only runs once per task lifecycle.
        """
        chain_id = f"chain-{source_task_id[:12]}-{next_agent}"
        queue_path = self.queue.queue_dir / next_agent / f"{chain_id}.json"
        return queue_path.exists()

    def _build_chain_task(self, task: Task, next_agent: str) -> Task:
        """Create a continuation task for the next agent in the chain."""
        from datetime import datetime

        chain_id = f"chain-{task.id[:12]}-{next_agent}"
        task_type = CHAIN_TASK_TYPES.get(next_agent, task.type)

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to=next_agent,
            created_at=datetime.utcnow(),
            title=f"[chain] {task.title}",
            description=task.description,
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "chain_step": True,
            },
        )
