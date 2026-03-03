"""Agent polling loop implementation (ported from Bash system)."""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # noqa: F401 (Dict/Any used in shims)

if TYPE_CHECKING:
    from .config import AgentDefinition, RepositoryConfig, WorkflowDefinition
    from ..queue.locks import FileLock

from .task import Task, TaskStatus, TaskType
from .task_validator import validate_task
from .activity import ActivityManager, AgentActivity, AgentStatus, CurrentTask, ActivityEvent, TaskPhase
from .team_composer import compose_default_team, compose_team
from .context_window_manager import ContextWindowManager
from .review_cycle import ReviewCycleManager, ReviewOutcome
from .git_operations import GitOperationsManager
from .task_manifest import load_manifest
from ..llm.base import LLMBackend, LLMResponse
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler
from ..workspace.worktree_manager import WorktreeManager
from ..utils.rich_logging import setup_rich_logging
from ..utils.type_helpers import get_type_str
from ..memory.memory_store import MemoryStore
from ..memory.memory_retriever import MemoryRetriever
from ..memory.tool_pattern_store import ToolPatternStore
from .session_logger import SessionLogger, noop_logger
from .prompt_builder import PromptBuilder, PromptContext
from .workflow_router import WorkflowRouter
from .error_recovery import ErrorRecoveryManager
from .budget_manager import BudgetManager
from .feedback_bus import FeedbackBus
from .post_completion import PostCompletionManager
from .llm_executor import LLMExecutionManager
from .task_analytics import TaskAnalyticsManager

from .sandbox_runner import SandboxRunner

# Pause/resume signal file
PAUSE_SIGNAL_FILE = ".agent-communication/pause"

# Re-exported for checkpoint_manager compatibility
_DIAGNOSTIC_PREFIXES = frozenset({
    "pwd", "echo", "test", "[", "cd", "ls", "find", "stat", "file",
    "readlink", "realpath", "which", "type", "env", "printenv",
    "whoami", "id", "uname", "hostname", "dirname", "basename",
})

# Downstream agents get the correct task type for model selection
CHAIN_TASK_TYPES = {
    "engineer": TaskType.IMPLEMENTATION,
    "qa": TaskType.QA_VERIFICATION,
    "architect": TaskType.REVIEW,
}


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
    """Agent with polling loop for processing tasks."""

    _SOLO_WORKFLOW_STEPS = frozenset({"plan", "code_review", "preview_review", "create_pr"})

    def _init_code_indexing(self, code_indexing_config):
        """Initialize codebase indexing for structural code context in prompts."""
        cfg = code_indexing_config or {}
        self._code_indexing_enabled = cfg.get("enabled", True)
        self._code_indexing_inject_for = cfg.get("inject_for_agents", ["architect", "engineer", "qa"])
        self._code_indexer = None
        self._code_index_query = None

        if not self._code_indexing_enabled:
            return

        embedder = None
        emb_cfg = cfg.get("embeddings", {})
        if emb_cfg.get("enabled", False):
            from ..indexing.embeddings import EMBEDDINGS_AVAILABLE
            if EMBEDDINGS_AVAILABLE:
                from ..indexing.embeddings.embedder import Embedder
                embedder = Embedder(
                    model_name=emb_cfg.get("model", "nomic-ai/nomic-embed-text-v1.5"),
                    dimensions=emb_cfg.get("dimensions", 256),
                )

        from ..indexing import IndexStore, CodebaseIndexer, IndexQuery
        store = IndexStore(self.workspace)
        self._code_indexer = CodebaseIndexer(
            store=store,
            max_symbols=cfg.get("max_symbols", 500),
            exclude_patterns=cfg.get("exclude_patterns", []),
            embedder=embedder,
        )
        self._code_index_query = IndexQuery(
            store,
            embedder=embedder,
            n_semantic_results=emb_cfg.get("n_results", 15),
        )

    def _try_index_codebase(self, task: Task, repo_path: Path) -> None:
        """Trigger indexing after repo checkout, before prompt building."""
        if not self._code_indexer:
            return
        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return
        if self.config.base_id not in self._code_indexing_inject_for:
            return
        try:
            self._code_indexer.ensure_indexed(repo_slug, str(repo_path))
        except Exception:
            self.logger.debug("Codebase indexing failed, continuing without index", exc_info=True)

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
        code_indexing_config: Optional[dict] = None,
        heartbeat_interval: int = 15,
        max_consecutive_tool_calls: int = 15,
        max_consecutive_diagnostic_calls: int = 10,
        exploration_alert_threshold: int = 50,
        exploration_alert_thresholds: Optional[Dict[str, int]] = None,
    ):
        """Initialize Agent with modular subsystem setup."""
        self._heartbeat_interval = heartbeat_interval
        self._max_consecutive_tool_calls = max_consecutive_tool_calls
        self._max_consecutive_diagnostic_calls = max_consecutive_diagnostic_calls
        self._exploration_alert_threshold = exploration_alert_threshold
        self._exploration_alert_thresholds = exploration_alert_thresholds or {}

        # Core dependencies and basic state
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
        self._active_worktree: Optional[Path] = None
        self._last_worktree_cleanup: float = time.time()

        # Team mode and workflow configuration
        self._agents_config = agents_config or []
        self._agent_definition = agent_definition
        self._team_mode_enabled = team_mode_enabled
        self._team_mode_default_model = team_mode_default_model
        self._workflows_config = workflows_config or {}

        # Logging and workflow executor
        import os
        self.logger = setup_rich_logging(
            agent_id=self.config.id, workspace=workspace,
            log_level=os.environ.get("AGENT_LOG_LEVEL", "INFO"),
            use_file=True, use_json=False,
        )
        from ..workflow.executor import WorkflowExecutor
        self._workflow_executor = WorkflowExecutor(
            self.queue, self.queue.queue_dir, agent_logger=self.logger,
            workspace=workspace, activity_manager=None,
        )

        # Optimization config and safeguards
        sanitized_config = BudgetManager.sanitize_optimization_config(optimization_config or {}, logger=self.logger)
        self._optimization_config = MappingProxyType(sanitized_config)
        self.logger.info(f"Optimization config: {BudgetManager.get_active_optimizations(self._optimization_config)}")
        self._model_success_store = None
        ir_cfg = sanitized_config.get("intelligent_routing", {})
        if ir_cfg.get("enabled", False):
            from ..llm.model_success_store import ModelSuccessStore
            self._model_success_store = ModelSuccessStore(self.workspace, enabled=True)
        self.retry_handler = RetryHandler(max_retries=self.config.max_retries)
        self.escalation_handler = EscalationHandler(
            enable_error_truncation=self._optimization_config.get("enable_error_truncation", False)
        )

        # State tracking: heartbeat, activity, caches
        self.heartbeat_file = self.workspace / ".agent-communication" / "heartbeats" / self.config.id
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.activity_manager = ActivityManager(self.workspace)
        self._workflow_executor.activity_manager = self.activity_manager
        self._paused = False
        self._error_handling_guidance: Optional[str] = None
        self._guidance_cache: Dict[str, str] = {}
        self._default_team_cache: Optional[dict] = None
        self._workflow_team_cache: Dict[str, Optional[dict]] = {}
        self._current_specialization = None
        self._current_file_count = 0
        self._pause_signal_cache: Optional[bool] = None
        self._pause_signal_cache_time: float = 0.0

        # Session logging (must precede agentic features — FeedbackBus needs _session_logger)
        sl_cfg = session_logging_config or {}
        self._session_logging_enabled = sl_cfg.get("enabled", False)
        self._session_log_prompts = sl_cfg.get("log_prompts", True)
        self._session_log_tool_inputs = sl_cfg.get("log_tool_inputs", True)
        self._session_logs_dir = self.workspace / "logs"
        self._session_logger: SessionLogger = noop_logger()
        retention_days = sl_cfg.get("retention_days", 30)
        if self._session_logging_enabled and retention_days > 0:
            SessionLogger.cleanup_old_sessions(self._session_logs_dir, retention_days)

        # Agentic features: memory, tool patterns, self-eval, replanning
        mem_cfg = memory_config or {}
        self._memory_enabled = mem_cfg.get("enabled", False)
        self._memory_store = MemoryStore(self.workspace, enabled=self._memory_enabled)
        self._memory_retriever = MemoryRetriever(self._memory_store)
        tool_tips_enabled = self._optimization_config.get("enable_tool_pattern_tips", False)
        self._tool_pattern_store = ToolPatternStore(self.workspace, enabled=tool_tips_enabled)
        self._tool_tips_enabled = tool_tips_enabled
        eval_cfg = self_eval_config or {}
        self._self_eval_enabled = eval_cfg.get("enabled", False)
        self._self_eval_max_retries = eval_cfg.get("max_retries", 2)
        self._self_eval_model = eval_cfg.get("model", "haiku")
        replan_cfg = replan_config or {}
        self._replan_enabled = replan_cfg.get("enabled", False)
        self._replan_min_retry = replan_cfg.get("min_retry_for_replan", 2)
        self._replan_model = replan_cfg.get("model", "haiku")

        # Context window manager — initialized per task, starts as None
        self._context_window_manager: Optional[ContextWindowManager] = None

        # Sandbox for test execution
        self._test_runner = SandboxRunner.create_test_runner(self.config, self.logger)

        # PR lifecycle management
        self._pr_lifecycle_manager = None
        if repositories_config:
            from .pr_lifecycle import PRLifecycleManager
            repo_lookup = {rc.github_repo: rc.model_dump() for rc in repositories_config}
            if repo_lookup:
                self._pr_lifecycle_manager = PRLifecycleManager(
                    queue=queue, workspace=self.workspace, repo_configs=repo_lookup,
                    pr_lifecycle_config=pr_lifecycle_config, logger_instance=self.logger,
                    multi_repo_manager=multi_repo_manager,
                )

        # Codebase indexing for structural code context in prompts
        self._init_code_indexing(code_indexing_config)

        # Workflow routing: chain enforcement, task decomposition, agent handoffs
        self._workflow_router = WorkflowRouter(
            config=config,
            queue=queue,
            workspace=self.workspace,
            logger=self.logger,
            session_logger=self._session_logger,
            workflows_config=self._workflows_config,
            workflow_executor=self._workflow_executor,
            agents_config=self._agents_config,
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
            optimization_config=self._optimization_config,
            memory_retriever=self._memory_retriever,
            memory_store=self._memory_store,
            tool_pattern_store=self._tool_pattern_store,
            context_window_manager=None,  # Set per-task
            session_logger=None,  # Set per-task
            logger=self.logger,
            llm=llm,
            queue=queue,
            agent=self,
            workflows_config=workflows_config,
            code_index_query=self._code_index_query,
            code_indexing_config=code_indexing_config,
        )
        self._prompt_builder = PromptBuilder(prompt_ctx)

        # Read cache manager — cross-step file read dedup
        from .read_cache_manager import ReadCacheManager
        self._read_cache = ReadCacheManager(
            workspace=workspace,
            session_logger=self._session_logger,
            logger=self.logger,
            config_base_id=config.base_id,
            prompt_builder=self._prompt_builder,
        )

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

        # Sandbox test runner
        self._sandbox = SandboxRunner(
            config=config,
            logger=self.logger,
            queue=queue,
            activity_manager=self.activity_manager,
            git_ops=self._git_ops,
            test_runner=self._test_runner,
        )

        # Error recovery and budget management
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
            activity_manager=self.activity_manager,
            review_cycle=self._review_cycle,
            model_success_store=self._model_success_store,
        )

        # Cross-feature learning loop — depends on _error_recovery + _memory_store
        self._feedback_bus = FeedbackBus(
            memory_store=self._memory_store,
            session_logger=self._session_logger,
            error_recovery=self._error_recovery,
        )

        self._budget = BudgetManager(
            agent_id=config.id,
            optimization_config=dict(self._optimization_config),
            logger=self.logger,
            session_logger=self._session_logger,
            llm=llm,
            workspace=workspace,
            activity_manager=self.activity_manager,
            model_success_store=self._model_success_store,
        )
        # Wire budget_manager into error_recovery (created before BudgetManager)
        self._error_recovery._budget_manager = self._budget

        # Post-completion flow: verdicts, context handoff, chain routing
        self._post_completion = PostCompletionManager(
            config=config,
            queue=queue,
            workspace=workspace,
            logger=self.logger,
            session_logger=self._session_logger,
            activity_manager=self.activity_manager,
            review_cycle=self._review_cycle,
            workflow_router=self._workflow_router,
            git_ops=self._git_ops,
            budget=self._budget,
            error_recovery=self._error_recovery,
            optimization_config=dict(self._optimization_config),
            session_logging_enabled=self._session_logging_enabled,
            session_logs_dir=self._session_logs_dir,
            agent_definition=agent_definition,
        )

        # LLM execution: interruption watching, circuit breaker, completion logging
        self._llm_executor = LLMExecutionManager(
            config=config,
            llm=llm,
            git_ops=self._git_ops,
            logger=self.logger,
            session_logger=self._session_logger,
            activity_manager=self.activity_manager,
        )

        # Task analytics: memories, tool patterns, summaries, metrics
        self._analytics = TaskAnalyticsManager(
            config=config,
            logger=self.logger,
            session_logger=self._session_logger,
            llm=llm,
            memory_retriever=self._memory_retriever,
            tool_pattern_store=self._tool_pattern_store,
            optimization_config=dict(self._optimization_config),
            memory_enabled=self._memory_enabled,
            tool_tips_enabled=self._tool_tips_enabled,
            session_logging_enabled=self._session_logging_enabled,
            session_logs_dir=self._session_logs_dir,
            workspace=workspace,
            feedback_bus=self._feedback_bus,
            code_index_query=self._code_index_query,
        )

    # -- Backward-compat shims: ReviewCycleManager --

    def _parse_review_outcome(self, content: str) -> ReviewOutcome:
        return self._review_cycle.parse_review_outcome(content)

    def _extract_review_findings(self, content: str):
        return self._review_cycle.extract_review_findings(content)

    def _parse_structured_findings(self, content: str):
        return self._review_cycle.parse_structured_findings(content)

    def _format_findings_checklist(self, findings):
        return self._review_cycle.format_findings_checklist(findings)

    def _build_review_task(self, task: Task, pr_info: dict) -> Task:
        return self._review_cycle.build_review_task(task, pr_info)

    def _build_review_fix_task(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> Task:
        return self._review_cycle.build_review_fix_task(task, outcome, cycle_count)

    def _escalate_review_to_architect(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> None:
        return self._review_cycle.escalate_review_to_architect(task, outcome, cycle_count)

    def _purge_orphaned_review_tasks(self) -> None:
        return self._review_cycle.purge_orphaned_review_tasks()

    def _get_pr_info(self, task: Task, response):
        return self._review_cycle.get_pr_info(task, response)

    def _extract_pr_info_from_response(self, response_content: str):
        return self._review_cycle.extract_pr_info_from_response(response_content)

    def _queue_code_review_if_needed(self, task: Task, response) -> None:
        return self._review_cycle.queue_code_review_if_needed(task, response)

    def _queue_review_fix_if_needed(self, task: Task, response) -> None:
        return self._review_cycle.queue_review_fix_if_needed(task, response, lambda *a, **kw: None)

    # --- Hot-reload: restart agent process when source code changes ---

    def _get_source_code_version(self) -> Optional[str]:
        """Hash of *.py file mtimes — detects source edits without depending on git state."""
        source_dir = Path(__file__).parent.parent  # agent_framework/
        try:
            mtimes = []
            for py_file in source_dir.rglob("*.py"):
                mtimes.append(f"{py_file}:{py_file.stat().st_mtime}")
            mtimes.sort()
            return hashlib.sha256("\n".join(mtimes).encode()).hexdigest()[:16]
        except OSError:
            return None

    def _should_hot_restart(self) -> bool:
        """Check if source code changed since startup (rate-limited to 60s)."""
        if not hasattr(self, "_startup_code_version") or self._startup_code_version is None:
            return False
        now = time.time()
        if now - getattr(self, "_last_version_check", 0.0) < 60:
            return False
        self._last_version_check = now
        current = self._get_source_code_version()
        if current is None:
            return False
        return current != self._startup_code_version

    def _hot_restart(self) -> None:
        """Replace process with fresh one to pick up code changes (only between tasks)."""
        import sys
        import os as _os

        new_version = self._get_source_code_version()
        self.logger.info(
            f"Source code changed ({self._startup_code_version[:8]}... -> "
            f"{new_version[:8] if new_version else '?'}...), "
            f"restarting agent process"
        )

        # Clean up mkdir-based locks owned by this process (directories survive exec)
        try:
            locks_dir = self.workspace / ".agent-communication" / "locks"
            if locks_dir.exists():
                pid = _os.getpid()
                for lock_dir in locks_dir.iterdir():
                    if lock_dir.is_dir():
                        pid_file = lock_dir / "pid"
                        if pid_file.exists():
                            try:
                                if int(pid_file.read_text().strip()) == pid:
                                    import shutil
                                    shutil.rmtree(lock_dir, ignore_errors=True)
                            except (ValueError, OSError):
                                pass
        except Exception:
            pass

        # Write IDLE so watchdog doesn't think we crashed
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc),
        ))
        self._write_heartbeat()

        # Always use -m invocation — sys.argv[0] is the resolved file path
        # which breaks relative imports when replayed as a script.
        _os.execv(sys.executable, [
            sys.executable, "-m", "agent_framework.run_agent",
        ] + sys.argv[1:])

    async def run(self) -> None:
        """Main polling loop."""
        self._running = True
        self._startup_code_version = self._get_source_code_version()
        self._last_version_check: float = 0.0
        self.logger.info(
            f"🚀 Starting {self.config.id} runner "
            f"(code version: {(self._startup_code_version or 'unknown')[:12]})"
        )

        # Write initial IDLE state when agent starts
        from datetime import datetime
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        ))

        # Drain stale review-chain tasks left over from before the cycle-count guard
        self._review_cycle.purge_orphaned_review_tasks()

        # Background heartbeat keeps the watchdog informed during long LLM calls
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        while self._running:
            # Belt-and-suspenders: also write heartbeat at top of each iteration
            self._write_heartbeat()

            # Hot-reload: pick up code changes between tasks
            if self._should_hot_restart():
                self._hot_restart()

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

            # Poll for next task (atomic claim = pop + lock in one step)
            claimed = self.queue.claim(self.config.queue, self.config.id)

            if claimed:
                task, lock = claimed
                try:
                    await self._handle_task(task, lock=lock)
                except Exception as e:
                    # Should not happen — _handle_task has its own finally guard — but if it does,
                    # log and continue polling rather than crashing the agent process.
                    self.logger.error(f"Unhandled exception in _handle_task for {task.id}: {e}", exc_info=True)
                    # The finally guard in _handle_task also failed, so the lock was never released.
                    try:
                        self.queue.release_lock(lock)
                    except Exception:
                        pass
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

        # Stop background heartbeat before anything else
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

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

    async def run_single_task(self, task_id: str) -> bool:
        """Execute a single task by ID and exit — no polling loop.

        Entry point for parallel subtask workers. Claims the specific task,
        runs _handle_task(), and returns success/failure.

        Args:
            task_id: ID of the task to execute.

        Returns:
            True if task completed successfully, False otherwise.
        """
        self._running = True
        self.logger.info(f"run_single_task: claiming {task_id}")

        # Find and claim the task directly
        task = self.queue.find_task(task_id)
        if not task:
            self.logger.error(f"run_single_task: task {task_id} not found")
            return False

        lock = self.queue.acquire_lock(task.id, self.config.id)
        if not lock:
            self.logger.error(f"run_single_task: could not acquire lock for {task_id}")
            return False

        try:
            await self._handle_task(task, lock=lock)
            return task.status == TaskStatus.COMPLETED
        except Exception as e:
            self.logger.error(f"run_single_task: unhandled exception for {task_id}: {e}", exc_info=True)
            try:
                self.queue.release_lock(lock)
            except Exception:
                pass
            return False

    def _validate_task_or_reject(self, task: Task) -> bool:
        """Validate task and reject if invalid. Returns True if task should proceed."""
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

        # Read before mark_in_progress to avoid any future mutation ordering surprises
        self_eval_count = task.context.get("_self_eval_count", 0)

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

        # On self-eval retry, emit "retry" so the stream doesn't show a
        # duplicate "start" for what is logically the same task execution.
        event_type = "retry" if self_eval_count > 0 else "start"
        self.activity_manager.append_event(ActivityEvent(
            type=event_type,
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc),
            retry_count=self_eval_count if event_type == "retry" else None,
            root_task_id=task.root_id,
        ))

        # Deterministic JIRA transition on task start
        if self._agent_definition and self._agent_definition.jira_on_start:
            self._sync_jira_status(task, self._agent_definition.jira_on_start)

    async def _handle_successful_response(self, task: Task, response, task_start_time, *, working_dir: Optional[Path] = None) -> None:
        """Handle successful LLM response including tests, workflow, and completion."""
        # Extract summary from response
        if self._optimization_config.get("enable_result_summarization", False):
            summary = await self._analytics.extract_summary(response.content, task)
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
            passed = await self._self_evaluate(
                task, response,
                test_passed=test_result.success if test_result else None,
                working_dir=working_dir,
            )
            if not passed:
                return  # Task was reset for self-eval retry

        # Deliverable gate: context-exhausted sessions exit 0 with no code changes
        if working_dir is not None and self._is_implementation_step(task):
            if not self._error_recovery.has_deliverables(task, working_dir):
                task.last_error = (
                    "No code changes detected — likely context window exhaustion. "
                    "Retrying with a fresh context."
                )
                self._session_logger.log(
                    "context_exhaustion",
                    task_id=task.id,
                    utilization_percent=self._context_window_manager.budget.utilization_percent if self._context_window_manager else None,
                )
                await self._handle_failure(task)
                return

        # Delegate the rest to PostCompletionManager
        self._post_completion.finalize_successful_response(
            task, response, task_start_time,
            working_dir=working_dir,
            context_window_manager=self._context_window_manager,
            extract_and_store_memories_cb=self._analytics.extract_and_store_memories,
            analyze_tool_patterns_cb=self._analytics.analyze_tool_patterns,
            sync_jira_status_cb=self._sync_jira_status,
        )

    def _run_post_completion_flow(self, task: Task, response, routing_signal, task_start_time) -> None:
        """Delegate to PostCompletionManager."""
        self._post_completion.run_post_completion_flow(
            task, response, routing_signal, task_start_time,
            context_window_manager=self._context_window_manager,
            extract_and_store_memories_cb=self._analytics.extract_and_store_memories,
            analyze_tool_patterns_cb=self._analytics.analyze_tool_patterns,
            sync_jira_status_cb=self._sync_jira_status,
        )

    # -- Backward-compat shims: PostCompletionManager --

    def _log_task_completion_metrics(self, task: Task, response, task_start_time, *, tool_call_count=None) -> None:
        self._post_completion.log_task_completion_metrics(
            task, response, task_start_time,
            tool_call_count=tool_call_count,
            context_window_manager=self._context_window_manager,
        )

    def _approval_verdict(self, task: Task) -> str:
        return self._post_completion.approval_verdict(task)

    def _set_structured_verdict(self, task: Task, response) -> None:
        self._post_completion.set_structured_verdict(task, response)

    @staticmethod
    def _is_no_changes_response(content: str) -> bool:
        return PostCompletionManager.is_no_changes_response(content)

    def _is_implementation_step(self, task: Task) -> bool:
        return PostCompletionManager.is_implementation_step(task, self.config.base_id)

    def _resolve_budget_ceiling(self, task: Task) -> Optional[float]:
        return self._post_completion.resolve_budget_ceiling(task)

    # -- Backward-compat shims: ErrorRecoveryManager --

    @staticmethod
    def _extract_partial_progress(content: str, max_bytes: int = 2048) -> str:
        return ErrorRecoveryManager.extract_partial_progress(content, max_bytes)

    def _finalize_failed_attempt(
        self, task: Task, working_dir: Optional[Path], *,
        content: Optional[str] = None, error: Optional[str] = None,
        input_tokens: int = 0, output_tokens: int = 0, cost_usd: Optional[float] = None,
    ) -> None:
        self._error_recovery.finalize_failed_attempt(
            task, working_dir, content=content, error=error,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd,
        )

    def _can_salvage_verdict(self, task: Task, response) -> bool:
        return self._error_recovery.can_salvage_verdict(task, response)

    async def _handle_failed_response(self, task: Task, response, *, working_dir: Optional[Path] = None) -> None:
        ctx_budget = self._context_window_manager.budget if self._context_window_manager else None
        await self._error_recovery.handle_failed_response(
            task, response, working_dir=working_dir, context_budget=ctx_budget,
        )

    def _cleanup_task_execution(self, task: Task, lock) -> None:
        """Cleanup after task: safety commit, worktree cleanup, set IDLE, release lock."""
        from datetime import datetime

        task_succeeded = task.status == TaskStatus.COMPLETED
        # Last-chance safety commit before worktree is cleaned up
        if self._git_ops.active_worktree:
            self._git_ops.safety_commit(
                self._git_ops.active_worktree,
                f"WIP: uncommitted changes at cleanup ({task.id})",
            )
        self._git_ops.sync_worktree_queued_tasks()
        self._git_ops.cleanup_worktree(task, success=task_succeeded)

        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        ))

        if lock:
            self.queue.release_lock(lock)
        self._current_task_id = None

    def _get_validated_working_directory(self, task: Task) -> Path:
        """Get working directory with one retry if path vanishes."""
        # On retries, restore the manifest branch so get_working_directory()
        # checks out the same branch the original attempt used
        if task.retry_count and task.retry_count > 0:
            manifest = load_manifest(self.workspace, task.root_id)
            if manifest:
                task.context.setdefault("implementation_branch", manifest.branch)
                task.context.setdefault("worktree_branch", manifest.branch)

        working_dir = self._git_ops.get_working_directory(task)
        branch = task.context.get("worktree_branch") or task.context.get("implementation_branch")
        if not working_dir.exists():
            self.logger.error(
                f"Working directory vanished after creation: {working_dir}. "
                f"branch={branch}, root_id={task.root_id}. "
                f"Possible cause: sibling worktree removal destroyed parent directory."
            )
            working_dir = self._git_ops.get_working_directory(task)
            if not working_dir.exists():
                raise RuntimeError(f"Working directory does not exist after retry: {working_dir}")

        try:
            file_count = sum(1 for _ in working_dir.iterdir())
        except OSError:
            file_count = -1
        self._session_logger.log(
            "worktree_validated",
            path=str(working_dir),
            branch=branch,
            file_count=file_count,
        )
        return working_dir

    def _maybe_run_periodic_worktree_cleanup(self) -> None:
        """No-op — worktrees cleaned via CLI only."""

    async def _watch_for_interruption(self) -> None:
        """Poll for pause/stop/worktree-vanish during LLM execution."""
        while self._running and not self._check_pause_signal():
            self._write_heartbeat()
            wt = self._git_ops.active_worktree
            if wt and not wt.exists():
                self.logger.critical(
                    f"WORKTREE VANISHED during LLM execution: {wt}"
                )
                return
            await asyncio.sleep(2)

    @staticmethod
    def _normalize_workflow(task: Task) -> None:
        """Map legacy workflow names to 'default' and assign default when missing.

        Old tasks in queues may have 'simple', 'standard', or 'full'.
        Tasks from issue_to_task() may have no workflow at all.
        Normalize so the rest of the pipeline only sees 'default' (or a known name).
        """
        if task.context is None:
            return
        workflow = task.context.get("workflow")
        if not workflow or workflow in ("simple", "standard", "full"):
            task.context["workflow"] = "default"

    def _setup_task_context(self, task: Task, task_start_time) -> None:
        """Setup task context: logging, session logger, validation."""

        # Set task context for logging
        jira_key = task.context.get("jira_key")
        self.logger.task_started(task.id, task.title, jira_key=jira_key)

        self._current_task_id = task.id

        # Session logger: structured JSONL for post-hoc analysis
        self._session_logger = SessionLogger(
            logs_dir=self._session_logs_dir,
            task_id=task.id,
            enabled=self._session_logging_enabled,
            log_prompts=self._session_log_prompts,
            log_tool_inputs=self._session_log_tool_inputs,
        )
        # Update all managers' session loggers for this task
        self._workflow_router.set_session_logger(self._session_logger)
        self._workflow_executor.set_session_logger(self._session_logger)
        self._read_cache.set_session_logger(self._session_logger)
        self._post_completion.set_session_logger(self._session_logger)
        self._llm_executor.set_session_logger(self._session_logger)
        self._analytics.set_session_logger(self._session_logger)
        self._session_logger.log(
            "task_start",
            agent=self.config.id,
            title=task.title,
            retry=task.retry_count,
            task_type=get_type_str(task.type),
        )

    def _setup_context_window_manager_for_task(self, task: Task) -> None:
        """Initialize context window manager for this task."""
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

    def _compose_team_for_task(self, task: Task) -> Optional[dict]:
        """Compose team agents for task execution if team mode is enabled."""
        team_override = task.context.get("team_override")
        if not self._team_mode_enabled or team_override is False:
            if team_override is False:
                self.logger.debug("Team mode skipped via task team_override=False")
            return None

        team_agents = {}

        # Layer 1: agent's configured teammates
        if self._agent_definition and self._agent_definition.teammates:
            if self.config.base_id == "engineer":
                # Engineer teammates vary by specialization (no cache)
                default_team = compose_default_team(
                    self._agent_definition,
                    default_model=self._team_mode_default_model,
                    specialization_profile=self._current_specialization,
                ) or {}
                if default_team:
                    team_agents.update(default_team)
            else:
                # Other agents have fixed teammates (use cache)
                if self._default_team_cache is None:
                    self._default_team_cache = compose_default_team(
                        self._agent_definition,
                        default_model=self._team_mode_default_model,
                    ) or {}
                if self._default_team_cache:
                    team_agents.update(self._default_team_cache)

        # Layer 2: workflow-required agents (cached per workflow type)
        # Solo steps (plan, code_review, etc.) skip Layer 2 — the configured
        # teammates from Layer 1 are sufficient and workflow agents just
        # duplicate the lead's exploration.
        workflow_step = task.context.get("workflow_step")
        is_solo_step = (
            workflow_step in self._SOLO_WORKFLOW_STEPS
            or (task.context.get("workflow") and workflow_step is None)
        )
        if not is_solo_step:
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
        return team_agents

    async def _execute_llm_with_interruption_watch(self, task: Task, prompt: str, working_dir: Path, team_agents: Optional[dict]) -> Optional[LLMResponse]:
        """Delegate to LLMExecutionManager."""
        response = await self._llm_executor.execute(
            task, prompt, working_dir, team_agents,
            context_window_manager=self._context_window_manager,
            is_implementation_step=self._is_implementation_step(task),
            max_consecutive_tool_calls=self._max_consecutive_tool_calls,
            max_consecutive_diagnostic_calls=self._max_consecutive_diagnostic_calls,
            exploration_alert_threshold=self._exploration_alert_threshold,
            exploration_alert_thresholds=self._exploration_alert_thresholds,
            watch_for_interruption_coro=self._watch_for_interruption,
            update_phase_cb=self._update_phase,
            current_specialization=self._current_specialization,
            current_file_count=self._current_file_count,
            optimization_config=dict(self._optimization_config),
            finalize_failed_attempt_cb=self._finalize_failed_attempt,
        )
        # LLMExecutionManager._handle_interruption sets task state but
        # needs queue.update — handle it here for the interruption case
        if response is None and task.status == TaskStatus.PENDING:
            self.queue.update(task)
        return response

    # -- Backward-compat shims: LLMExecutionManager --

    async def _auto_commit_wip(self, task: Task, working_dir: Path, bash_count: int) -> None:
        await self._llm_executor.auto_commit_wip(task, working_dir, bash_count)

    def _log_routing_decision(self, task: Task, response) -> None:
        self._llm_executor.log_routing_decision(task, response)

    def _process_llm_completion(self, response, task: Task) -> None:
        self._llm_executor.process_completion(response, task, context_window_manager=self._context_window_manager)

    async def _handle_task(self, task: Task, *, lock: Optional["FileLock"] = None) -> None:
        """Handle task execution with retry/escalation logic."""
        from datetime import datetime

        # Normalize legacy workflow names and validate task
        self._normalize_workflow(task)
        if not self._validate_task_or_reject(task):
            if lock:
                lock.release()
            return

        if lock is None:
            lock = self.queue.acquire_lock(task.id, self.config.id)
            if not lock:
                self.logger.warning("⏸️  Could not acquire lock, will retry later")
                return

        task_start_time = datetime.now(timezone.utc)

        self._setup_task_context(task, task_start_time)
        self._setup_context_window_manager_for_task(task)

        working_dir = None
        _cost_before_try = task.context.get("_cumulative_cost", 0.0)
        try:
            self._initialize_task_execution(task, task_start_time)
            working_dir = self._get_validated_working_directory(task)
            self.logger.info(f"Working directory: {working_dir}")

            if task.retry_count > 0:
                branch_work = self._git_ops.discover_branch_work(working_dir)

                # Fallback: check attempt history for a pushed branch
                if not branch_work:
                    try:
                        from .attempt_tracker import get_last_pushed_branch
                        from ..utils.subprocess_utils import run_git_command
                        pushed_branch = get_last_pushed_branch(self.workspace, task.id)
                        if pushed_branch:
                            self.logger.info(f"Fetching pushed branch {pushed_branch} from attempt history")
                            run_git_command(
                                ["fetch", "origin", pushed_branch],
                                cwd=working_dir, check=False, timeout=30,
                            )
                            branch_work = self._git_ops.discover_branch_work(working_dir)
                    except Exception as e:
                        self.logger.debug(f"Attempt history branch recovery failed (non-fatal): {e}")

                if branch_work:
                    task.context["_previous_attempt_branch_work"] = branch_work
                    self.logger.info(
                        f"Discovered {branch_work['commit_count']} commits "
                        f"({branch_work['insertions']}+/{branch_work['deletions']}-) "
                        f"from previous attempt(s)"
                    )
                    self._session_logger.log(
                        "branch_work_discovered",
                        commit_count=branch_work["commit_count"],
                        insertions=branch_work["insertions"],
                        file_count=len(branch_work["file_list"]),
                    )

            self._try_index_codebase(task, working_dir)
            self.logger.phase_change("analyzing")
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

            # Compose team for task execution
            team_agents = self._compose_team_for_task(task)

            # Second validation: catch deletions during prompt build/indexing
            if not working_dir.exists():
                raise RuntimeError(
                    f"Working directory vanished before LLM start: {working_dir}. "
                    f"Likely deleted by sibling agent cleanup during prompt build."
                )

            # Hard budget check — fail fast before spending on another LLM call
            _ceiling = task.context.get("_budget_ceiling")
            if _ceiling is not None and task.context.get("_cumulative_cost", 0.0) >= _ceiling:
                raise RuntimeError(
                    f"Budget ceiling exceeded: cumulative cost "
                    f"${task.context['_cumulative_cost']:.2f} >= ceiling ${_ceiling:.2f}"
                )

            # Execute LLM with interruption watching
            response = await self._execute_llm_with_interruption_watch(task, prompt, working_dir, team_agents)
            if response is None:
                # Task was interrupted and reset to pending
                return

            # Process LLM completion (logging, context window updates)
            self._process_llm_completion(response, task)
            self._log_routing_decision(task, response)

            # Push committed work to remote immediately — protects against
            # worktree corruption destroying unpushed code
            self._git_ops.push_if_unpushed()

            # Handle response
            if response.success:
                # Populate shared read cache for downstream chain steps
                file_reads = self._read_cache.populate_read_cache(task, working_dir=working_dir)
                self._read_cache.measure_cache_effectiveness(task, file_reads, working_dir=working_dir)
                await self._handle_successful_response(task, response, task_start_time, working_dir=working_dir)
            elif self._can_salvage_verdict(task, response):
                self.logger.warning(
                    f"Salvaging task {task.id}: non-zero exit code but output "
                    f"contains valid review verdict"
                )
                self._session_logger.log(
                    "verdict_salvaged",
                    task_id=task.id,
                    original_error=response.error,
                    content_length=len(response.content or ""),
                )
                response.success = True
                response.finish_reason = "stop"
                file_reads = self._read_cache.populate_read_cache(task, working_dir=working_dir)
                self._read_cache.measure_cache_effectiveness(task, file_reads, working_dir=working_dir)
                await self._handle_successful_response(task, response, task_start_time, working_dir=working_dir)
            else:
                await self._handle_failed_response(task, response, working_dir=working_dir)

        except Exception as e:
            task.last_error = str(e)
            self.logger.exception(f"Error processing task {task.id}: {e}")

            # Salvage cost from response if LLM completed before the exception.
            # Skip if _handle_failed_response / _run_post_completion_flow already
            # accumulated cost (avoids double-counting when the exception originates
            # from within those methods).
            _fail_cost = None
            _fail_in = 0
            _fail_out = 0
            _already_accumulated = task.context.get("_cumulative_cost", 0.0) != _cost_before_try
            if not _already_accumulated and 'response' in locals() and response is not None:
                _fail_cost = self._budget.estimate_cost(response)
                _fail_in = response.input_tokens
                _fail_out = response.output_tokens
                prev = task.context.get("_cumulative_cost", 0.0)
                task.context["_cumulative_cost"] = prev + _fail_cost

            self._session_logger.log(
                "task_failed",
                error=str(e),
                retry=task.retry_count,
                tokens_in=_fail_in,
                tokens_out=_fail_out,
                cost=_fail_cost,
            )

            ctx_budget = self._context_window_manager.budget if self._context_window_manager else None
            self.activity_manager.append_event(ActivityEvent(
                type="fail",
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.now(timezone.utc),
                retry_count=task.retry_count,
                error_message=task.last_error,
                root_task_id=task.root_id,
                context_utilization_percent=ctx_budget.utilization_percent if ctx_budget else None,
                context_budget_tokens=ctx_budget.total_budget if ctx_budget else None,
                input_tokens=_fail_in,
                output_tokens=_fail_out,
                cost=_fail_cost,
            ))

            # Push whatever was committed before the error — prevents worktree
            # corruption from destroying partial work during retry setup
            self._git_ops.push_if_unpushed()

            if task.status == TaskStatus.COMPLETED:
                self.logger.warning(
                    f"Post-completion error for task {task.id} (already completed): {e}"
                )
            else:
                self._finalize_failed_attempt(
                    task, working_dir, error=str(e),
                    input_tokens=_fail_in, output_tokens=_fail_out, cost_usd=_fail_cost,
                )
                await self._handle_failure(task)

        finally:
            self._context_window_manager = None
            try:
                self._cleanup_task_execution(task, lock)
            except Exception as e:
                # Cleanup failed mid-way. The worktree and activity state may be inconsistent,
                # but release the lock so the task doesn't stay locked forever.
                self.logger.error(f"Cleanup failed for task {task.id}, releasing lock directly: {e}")
                # IDLE may not have been set if cleanup raised before reaching it
                try:
                    self.activity_manager.update_activity(AgentActivity(
                        agent_id=self.config.id,
                        status=AgentStatus.IDLE,
                        last_updated=datetime.now(timezone.utc),
                    ))
                except Exception:
                    pass
                if lock:
                    try:
                        self.queue.release_lock(lock)
                    except Exception as lock_err:
                        self.logger.debug(f"Also failed to release lock for {task.id}: {lock_err}")
                self._current_task_id = None
            finally:
                # Close session logger AFTER cleanup so cleanup operations are logged
                self._session_logger.close()
                self._session_logger = noop_logger()

    async def _handle_failure(self, task: Task) -> None:
        """Delegate retry/escalation to ErrorRecoveryManager."""
        await self._error_recovery.handle_failure(task)

    # -- Backward-compat shims: BudgetManager --

    def _get_active_optimizations(self) -> Dict[str, Any]:
        return BudgetManager.get_active_optimizations(self._optimization_config)

    def _should_use_optimization(self, task: Task) -> bool:
        return BudgetManager.should_use_optimization(task, self._optimization_config, logger=self.logger)

    def _get_token_budget(self, task_type: TaskType) -> int:
        return self._budget.get_token_budget(task_type)

    # -- Backward-compat shims: TaskAnalyticsManager --

    async def _extract_summary(self, response: str, task: Task, _recursion_depth: int = 0) -> str:
        return await self._analytics.extract_summary(response, task, _recursion_depth)

    def _get_repo_slug(self, task: Task) -> Optional[str]:
        return TaskAnalyticsManager.get_repo_slug(task)

    def _extract_and_store_memories(self, task: Task, response) -> None:
        self._analytics.extract_and_store_memories(task, response)

    def _analyze_tool_patterns(self, task: Task) -> Optional[int]:
        return self._analytics.analyze_tool_patterns(task)

    def _record_optimization_metrics(self, task: Task, legacy_prompt_length: int, optimized_prompt_length: int) -> None:
        self._analytics.record_optimization_metrics(
            task, legacy_prompt_length, optimized_prompt_length,
            should_use_optimization_cb=self._should_use_optimization,
            get_active_optimizations_cb=self._get_active_optimizations,
        )

    # -- Backward-compat shims: PostCompletionManager (additional) --

    UPSTREAM_CONTEXT_MAX_CHARS = 15000
    UPSTREAM_INLINE_MAX_CHARS = 15000

    def _save_upstream_context(self, task: Task, response) -> None:
        self._post_completion.save_upstream_context(task, response)

    def _save_step_to_chain_state(self, task: Task, response, **kwargs) -> None:
        self._post_completion.save_step_to_chain_state(task, response, **kwargs)

    def _emit_workflow_summary(self, task: Task) -> None:
        self._post_completion.emit_workflow_summary(task)

    def _compute_tool_stats_for_chain(self, task: Task) -> Optional[Dict]:
        return self._post_completion.compute_tool_stats_for_chain(task)

    def _save_pre_scan_findings(self, task: Task, response) -> None:
        self._post_completion.save_pre_scan_findings(task, response)

    @staticmethod
    def _extract_structured_findings_from_content(content: str) -> dict:
        return PostCompletionManager.extract_structured_findings_from_content(content)

    @staticmethod
    def _extract_plan_from_response(content: str):
        return PostCompletionManager.extract_plan_from_response(content)

    @staticmethod
    def _extract_design_rationale(content: str) -> Optional[str]:
        return PostCompletionManager.extract_design_rationale(content)

    # -- Backward-compat shims: ErrorRecoveryManager (additional) --

    async def _self_evaluate(self, task: Task, response, *, test_passed=None, working_dir=None) -> bool:
        return await self._error_recovery.self_evaluate(
            task, response, test_passed=test_passed, working_dir=working_dir
        )

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

    def _sync_jira_status(self, task: Task, target_status: str, comment: Optional[str] = None) -> None:
        """Framework-level JIRA status transition if all preconditions are met."""
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
        self.heartbeat_file.write_text(str(int(time.time())))

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat independent of main loop progress."""
        while self._running:
            try:
                self._write_heartbeat()
            except OSError:
                pass  # Transient FS error; next iteration will retry
            await asyncio.sleep(self._heartbeat_interval)

    def _check_pause_signal(self) -> bool:
        """Check pause signal files (manual or orchestrator). Cached 5s."""
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

    # -- Backward-compat shims: SandboxRunner --

    async def _run_sandbox_tests(self, task: Task) -> Optional[Any]:
        return await self._sandbox.run_sandbox_tests(task)

    async def _handle_test_failure(self, task: Task, llm_response, test_result) -> None:
        await self._sandbox.handle_test_failure(task, llm_response, test_result)

    # -- Backward-compat shims: WorkflowRouter --

    def _check_and_create_fan_in_task(self, task: Task) -> None:
        return self._workflow_router.check_and_create_fan_in_task(task)

    def _should_decompose_task(self, task: Task) -> bool:
        return self._workflow_router.should_decompose_task(task)

    def _decompose_and_queue_subtasks(self, task: Task) -> None:
        return self._workflow_router.decompose_and_queue_subtasks(task)

    def _enforce_workflow_chain(self, task: Task, response, routing_signal=None) -> bool:
        return self._workflow_router.enforce_chain(task, response, routing_signal)

    def _is_at_terminal_workflow_step(self, task: Task) -> bool:
        return self._workflow_router.is_at_terminal_workflow_step(task)

    def _build_workflow_context(self, task: Task) -> Dict[str, Any]:
        return self._workflow_router.build_workflow_context(task)

    def _route_to_agent(self, task: Task, target_agent: str, reason: str) -> None:
        return self._workflow_router.route_to_agent(task, target_agent, reason)

    def _queue_pr_creation_if_needed(self, task: Task, workflow) -> None:
        return self._workflow_router.queue_pr_creation_if_needed(task, workflow)
