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

from .task import PlanDocument, Task, TaskStatus, TaskType
from .task_validator import validate_task, ValidationResult
from .activity import ActivityManager, AgentActivity, AgentStatus, CurrentTask, ActivityEvent, TaskPhase, ToolActivity
from .routing import read_routing_signal, validate_routing_signal, log_routing_decision, WORKFLOW_COMPLETE
from .team_composer import compose_default_team, compose_team
from .context_window_manager import ContextWindowManager
from .review_cycle import ReviewCycleManager, QAFinding, ReviewOutcome, MAX_REVIEW_CYCLES
from .git_operations import GitOperationsManager
from .task_manifest import load_manifest
from ..llm.base import LLMBackend, LLMRequest, LLMResponse
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler
from ..workspace.worktree_manager import WorktreeManager, WorktreeConfig
from ..utils.rich_logging import ContextLogger, setup_rich_logging
from ..utils.type_helpers import get_type_str, strip_chain_prefixes
from ..memory.memory_store import MemoryStore
from ..memory.memory_retriever import MemoryRetriever
from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer
from ..memory.tool_pattern_store import ToolPatternStore
from .session_logger import SessionLogger, noop_logger
from .prompt_builder import PromptBuilder, PromptContext
from .workflow_router import WorkflowRouter
from .error_recovery import ErrorRecoveryManager
from .budget_manager import BudgetManager
from ..workflow.executor import PREVIEW_REVIEW_STEPS


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
_CONVERSATIONAL_PREFIXES = ("i ", "i'", "thank", "sure", "certainly", "of course", "let me")
BUDGET_WARNING_THRESHOLD = 1.3  # 30% over budget
_MAX_REPO_CACHE_ENTRIES = 200

# Circuit breaker: commands that indicate productive work (test/build/lint/git)
# rather than stuck-agent flailing (ls, pwd, echo). When most commands are
# productive, we use a higher threshold to avoid killing legitimate workflows.
_PRODUCTIVE_PREFIXES = frozenset({
    "pytest", "python -m pytest", "python -m unittest",
    "pip", "pip3", "uv ",
    "npm", "yarn", "pnpm", "bun ",
    "make", "cmake",
    "cargo", "go test", "go build", "go vet",
    "mvn", "gradle",
    "docker", "docker-compose",
    "bandit", "pylint", "mypy", "ruff", "flake8", "black", "isort", "tox",
    "git commit", "git push", "git add", "git diff", "git log", "git stash",
    "python", "node ",
})
_PRODUCTIVE_THRESHOLD_MULTIPLIER = 3
_PRODUCTIVE_RATIO_THRESHOLD = 0.7

# Matches synthetic [Tool Call: Read], [Tool Call: Bash] etc. markers injected
# by the Claude CLI backend into response.content for logging visibility.
# Harmless in session logs but pure noise for downstream agents reading upstream_summary.
_TOOL_CALL_MARKER_RE = re.compile(r'\n?\[Tool Call: [^\]]+\]\n?')


def _strip_tool_call_markers(content: str) -> str:
    """Remove [Tool Call: ...] markers and compress resulting whitespace."""
    if not content:
        return ""
    cleaned = _TOOL_CALL_MARKER_RE.sub('\n', content)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# Diagnostic commands: environment-probing commands that indicate the agent is
# stuck (usually after its working directory is deleted). 5+ consecutive
# diagnostic commands trips a dedicated circuit breaker that the diversity
# heuristic misses because pwd/echo/ls/which/find are inherently diverse.
_DIAGNOSTIC_PREFIXES = frozenset({
    "pwd", "echo", "test", "[", "cd",     # shell builtins
    "ls", "find", "stat", "file",         # filesystem probes
    "readlink", "realpath",               # path resolution
    "which", "type", "env", "printenv",   # environment probes
    "whoami", "id", "uname", "hostname",  # identity probes
    "dirname", "basename",                # path utilities
})


def _is_productive_command(cmd: str) -> bool:
    """Check if a bash command is a productive tool (test/build/lint/git)."""
    cmd_stripped = cmd.strip().lower()
    return any(cmd_stripped.startswith(p) for p in _PRODUCTIVE_PREFIXES)


def _is_diagnostic_command(cmd: str) -> bool:
    """Check if a bash command is an environment-probing diagnostic."""
    stripped = cmd.strip()
    token = stripped.split()[0] if stripped else ""
    # Strip path prefix: /usr/bin/echo → echo
    bare = token.split("/")[-1].lower()
    return bare in _DIAGNOSTIC_PREFIXES


def _repo_cache_slug(github_repo: str) -> str:
    """Convert 'owner/repo' to 'owner-repo' for cache file naming."""
    return github_repo.replace("/", "-")


def _to_relative_path(file_path: str, working_dir: Optional[Path]) -> str:
    """Strip worktree prefix for cache portability across chain steps."""
    if not working_dir or not file_path.startswith("/"):
        return file_path
    prefix = str(working_dir).rstrip("/") + "/"
    if file_path.startswith(prefix):
        return file_path[len(prefix):]
    return file_path

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


_JSON_FENCE_PATTERN = re.compile(r'```json\s*\n(.*?)\n?\s*```', re.DOTALL)

_NO_CHANGES_MARKER = "[NO_CHANGES_NEEDED]"

# Step classification for the deliverable gate — implementation steps must
# produce git-visible code changes; prose-only steps (plan, review) are exempt.
_IMPLEMENTATION_STEP_IDS = frozenset({"implement", "implementation"})
_NON_CODE_STEP_IDS = frozenset({
    "plan", "planning", "code_review", "qa_review", "create_pr",
    "preview_review", "preview",
})


class Agent:
    """
    Agent with polling loop for processing tasks.

    Ported from scripts/async-agent-runner.sh with the main polling loop
    at lines 254-407.
    """

    # Steps where only Layer 1 (configured) teammates are needed —
    # workflow-level teammates (engineer/qa) add redundant exploration
    _SOLO_WORKFLOW_STEPS = frozenset({"plan", "code_review", "preview_review", "create_pr"})

    def _init_core_dependencies(
        self,
        config,
        llm,
        queue,
        workspace,
        jira_client,
        github_client,
        multi_repo_manager,
        jira_config,
        github_config,
        mcp_enabled,
        worktree_manager,
    ):
        """Initialize core dependencies and basic state."""
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

    def _init_team_mode(self, agents_config, agent_definition, team_mode_enabled, team_mode_default_model, workflows_config):
        """Initialize team mode and workflow configuration."""
        self._agents_config = agents_config or []
        self._agent_definition = agent_definition
        self._team_mode_enabled = team_mode_enabled
        self._team_mode_default_model = team_mode_default_model
        self._workflows_config = workflows_config or {}

    def _init_logging(self, workspace):
        """Setup rich logging and workflow executor."""
        import os
        log_level = os.environ.get("AGENT_LOG_LEVEL", "INFO")
        self.logger = setup_rich_logging(
            agent_id=self.config.id,
            workspace=workspace,
            log_level=log_level,
            use_file=True,
            use_json=False,
        )
        from ..workflow.executor import WorkflowExecutor
        self._workflow_executor = WorkflowExecutor(
            self.queue, self.queue.queue_dir, agent_logger=self.logger,
            workspace=workspace,
            activity_manager=None,  # wired in _init_state_tracking after ActivityManager exists
        )

    def _init_optimization_and_safeguards(self, optimization_config):
        """Initialize optimization config and safeguards (retry/escalation)."""
        sanitized_config = self._sanitize_optimization_config(optimization_config or {})
        self._optimization_config = MappingProxyType(sanitized_config)
        self.logger.info(f"🔧 Optimization config: {self._get_active_optimizations()}")

        # Initialize model success store for intelligent routing
        self._model_success_store = None
        ir_cfg = sanitized_config.get("intelligent_routing", {})
        if ir_cfg.get("enabled", False):
            from ..llm.model_success_store import ModelSuccessStore
            self._model_success_store = ModelSuccessStore(self.workspace, enabled=True)

        self.retry_handler = RetryHandler(max_retries=self.config.max_retries)
        enable_error_truncation = self._optimization_config.get("enable_error_truncation", False)
        self.escalation_handler = EscalationHandler(enable_error_truncation=enable_error_truncation)

    def _init_state_tracking(self):
        """Initialize heartbeat, activity tracking, and caches."""
        self.heartbeat_file = self.workspace / ".agent-communication" / "heartbeats" / self.config.id
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.activity_manager = ActivityManager(self.workspace)
        # Wire activity_manager into executor (created before activity_manager exists)
        self._workflow_executor.activity_manager = self.activity_manager
        self._paused = False

        # Caches for prompt guidance
        self._error_handling_guidance: Optional[str] = None
        self._guidance_cache: Dict[str, str] = {}

        # Team composition caches
        self._default_team_cache: Optional[dict] = None
        self._workflow_team_cache: Dict[str, Optional[dict]] = {}
        self._current_specialization = None
        self._current_file_count = 0

        # Pause signal cache
        self._pause_signal_cache: Optional[bool] = None
        self._pause_signal_cache_time: float = 0.0

    def _init_agentic_features(self, memory_config, self_eval_config, replan_config):
        """Initialize agentic features: memory, tool patterns, self-eval, replanning."""
        # Memory system
        mem_cfg = memory_config or {}
        self._memory_enabled = mem_cfg.get("enabled", False)
        self._memory_store = MemoryStore(self.workspace, enabled=self._memory_enabled)
        self._memory_retriever = MemoryRetriever(self._memory_store)

        # Feedback bus for cross-feature learning (self-eval → memory, QA → memory)
        from .feedback_bus import FeedbackBus
        self._feedback_bus = FeedbackBus(self._memory_store)

        # Tool pattern analysis
        tool_tips_enabled = self._optimization_config.get("enable_tool_pattern_tips", False)
        self._tool_pattern_store = ToolPatternStore(self.workspace, enabled=tool_tips_enabled)
        self._tool_tips_enabled = tool_tips_enabled

        # Self-evaluation
        eval_cfg = self_eval_config or {}
        self._self_eval_enabled = eval_cfg.get("enabled", False)
        self._self_eval_max_retries = eval_cfg.get("max_retries", 2)
        self._self_eval_model = eval_cfg.get("model", "haiku")

        # Dynamic replanning
        replan_cfg = replan_config or {}
        self._replan_enabled = replan_cfg.get("enabled", False)
        self._replan_min_retry = replan_cfg.get("min_retry_for_replan", 2)
        self._replan_model = replan_cfg.get("model", "haiku")

    def _init_session_logging(self, session_logging_config):
        """Initialize session logging configuration."""
        sl_cfg = session_logging_config or {}
        self._session_logging_enabled = sl_cfg.get("enabled", False)
        self._session_log_prompts = sl_cfg.get("log_prompts", True)
        self._session_log_tool_inputs = sl_cfg.get("log_tool_inputs", True)
        self._session_logs_dir = self.workspace / "logs"
        self._session_logger: SessionLogger = noop_logger()

        retention_days = sl_cfg.get("retention_days", 30)
        if self._session_logging_enabled and retention_days > 0:
            SessionLogger.cleanup_old_sessions(self._session_logs_dir, retention_days)

    def _init_context_window_manager(self):
        """Initialize per-task context window manager (set to None, initialized per task)."""
        self._context_window_manager: Optional[ContextWindowManager] = None

    def _init_sandbox(self):
        """Initialize sandbox for isolated test execution."""
        self._test_runner = None
        if self.config.enable_sandbox and SANDBOX_AVAILABLE:
            try:
                executor = DockerExecutor(image=self.config.sandbox_image)
                if executor.health_check():
                    self._test_runner = GoTestRunner(executor=executor)
                    self.logger.info(f"Agent {self.config.id} sandbox enabled with image {self.config.sandbox_image}")
                else:
                    self.logger.warning(f"Docker not available, sandbox disabled for {self.config.id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize sandbox for {self.config.id}: {e}")

    def _init_pr_lifecycle(self, repositories_config, pr_lifecycle_config):
        """Initialize PR lifecycle manager for autonomous CI poll → fix → merge."""
        self._pr_lifecycle_manager = None
        if repositories_config:
            from .pr_lifecycle import PRLifecycleManager
            repo_lookup = {rc.github_repo: rc.model_dump() for rc in repositories_config}
            if repo_lookup:
                self._pr_lifecycle_manager = PRLifecycleManager(
                    queue=self.queue,
                    workspace=self.workspace,
                    repo_configs=repo_lookup,
                    pr_lifecycle_config=pr_lifecycle_config,
                    logger_instance=self.logger,
                    multi_repo_manager=self.multi_repo_manager,
                )

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
        max_consecutive_diagnostic_calls: int = 5,
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
        self._init_core_dependencies(
            config, llm, queue, workspace, jira_client, github_client,
            multi_repo_manager, jira_config, github_config, mcp_enabled, worktree_manager
        )

        # Team mode and workflow configuration
        self._init_team_mode(agents_config, agent_definition, team_mode_enabled, team_mode_default_model, workflows_config)

        # Logging and workflow executor
        self._init_logging(workspace)

        # Optimization config and safeguards
        self._init_optimization_and_safeguards(optimization_config)

        # State tracking: heartbeat, activity, caches
        self._init_state_tracking()

        # Agentic features: memory, tool patterns, self-eval, replanning
        self._init_agentic_features(memory_config, self_eval_config, replan_config)

        # Session logging
        self._init_session_logging(session_logging_config)

        # Context window manager (initialized per task)
        self._init_context_window_manager()

        # Sandbox for test execution
        self._init_sandbox()

        # PR lifecycle management
        self._init_pr_lifecycle(repositories_config, pr_lifecycle_config)

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
            feedback_bus=self._feedback_bus,
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
            feedback_bus=self._feedback_bus,
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


    # --- Hot-reload: restart agent process when source code changes ---

    def _get_source_code_version(self) -> Optional[str]:
        """Get current source code version (git HEAD of the agent-framework repo).

        Falls back to a hash of *.py mtimes if not in a git repo.
        """
        source_dir = Path(__file__).parent.parent  # agent_framework/
        repo_dir = source_dir.parent  # src/ parent -> project root

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Fallback: hash of py file mtimes
        try:
            mtimes = []
            for py_file in source_dir.rglob("*.py"):
                mtimes.append(f"{py_file}:{py_file.stat().st_mtime}")
            mtimes.sort()
            return hashlib.sha256("\n".join(mtimes).encode()).hexdigest()[:16]
        except OSError:
            return None

    def _should_hot_restart(self) -> bool:
        """Check if source code has changed since startup.

        Rate-limited to once per 60s to avoid spawning git subprocesses every poll cycle.
        """
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
        """Replace the current process with a fresh one to pick up code changes.

        Safe: only called between tasks (never mid-task). os.execv is atomic —
        it replaces the process image without running finally/atexit handlers.
        File descriptors (locks, logs) are released by the OS on process replacement.
        If execv fails (e.g. interpreter gone), OSError propagates and the agent
        loop continues with the old code.
        """
        import sys
        import os as _os

        new_version = self._get_source_code_version()
        self.logger.info(
            f"Source code changed ({self._startup_code_version[:8]}... -> "
            f"{new_version[:8] if new_version else '?'}...), "
            f"restarting agent process"
        )

        # Write IDLE so watchdog doesn't think we crashed
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc),
        ))
        self._write_heartbeat()

        _os.execv(sys.executable, [sys.executable] + sys.argv)

    async def run(self) -> None:
        """
        Main polling loop.

        Ported from scripts/async-agent-runner.sh lines 254-407.
        """
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

        # Extract structured plan from architect's planning response
        if (task.plan is None
                and self.config.base_id == "architect"
                and task.context.get("workflow_step", task.type) in ("plan", "planning")):
            content = getattr(response, "content", "") or ""
            extracted = self._extract_plan_from_response(content)
            if extracted:
                task.plan = extracted
                self.logger.info(
                    f"Extracted plan from response: {len(extracted.files_to_modify)} files, "
                    f"{len(extracted.approach)} steps"
                )
                # Build requirements checklist so downstream agents have a concrete contract
                from .task_decomposer import extract_requirements_checklist
                checklist = extract_requirements_checklist(extracted)
                if checklist:
                    task.context["requirements_checklist"] = checklist
                    self.logger.info(f"Extracted {len(checklist)} requirements checklist items")
            else:
                self.logger.warning("Architect plan step completed but no PlanDocument found in response")

        # Verdict must be set before serialization so it persists to disk
        self._set_structured_verdict(task, response)

        # Save upstream context AFTER plan extraction + verdict so task.context
        # has structured data when _build_chain_task copies it. The raw
        # upstream_summary is superseded by chain state in prompt rendering.
        if task.context.get("workflow") or task.context.get("chain_step"):
            self._save_upstream_context(task, response)

        # Append step to chain state file — structured data for step-aware rendering
        if task.context.get("workflow") or task.context.get("chain_step"):
            self._save_step_to_chain_state(task, response, working_dir=working_dir, task_start_time=task_start_time)

        # Safety commit: catch any uncommitted work before marking done
        if working_dir and working_dir.exists() and self._is_implementation_step(task):
            self._git_ops.safety_commit(working_dir, f"WIP: uncommitted changes at task completion ({task.id})")

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
        # Accumulate cost before chain routing so _build_chain_task
        # copies the up-to-date total into the next task's context
        if response is not None:
            this_cost = self._budget.estimate_cost(response)
            prev = task.context.get("_cumulative_cost", 0.0)
            task.context["_cumulative_cost"] = prev + this_cost

        # Stamp ceiling once on root task; propagated via context copy
        if "_budget_ceiling" not in task.context:
            ceiling = self._resolve_budget_ceiling(task)
            if ceiling is not None:
                task.context["_budget_ceiling"] = ceiling

        # Pre-scan tasks are fire-and-forget — save findings and skip all
        # workflow routing (no enforce_chain, no legacy review, no PR creation)
        if task.context.get("pre_scan"):
            self._save_pre_scan_findings(task, response)
            self._extract_and_store_memories(task, response)
            self._analyze_tool_patterns(task)
            self._log_task_completion_metrics(task, response, task_start_time)
            return

        # Validate parent actually exists before fan-in — LLMs can
        # fabricate parent references when they directly write task JSON.
        # A phantom parent_task_id causes the guard below to skip the
        # workflow chain with no fan-in ever firing.
        if task.parent_task_id is not None:
            parent = self._workflow_router.queue.find_task(task.parent_task_id)
            if parent is None:
                self.logger.warning(
                    f"Task {task.id} has phantom parent_task_id "
                    f"{task.parent_task_id!r} — not found in queue/completed. "
                    f"Clearing to allow normal workflow routing."
                )
                task.parent_task_id = None

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
            if not has_workflow and not task.context.get("chain_step"):
                self.logger.warning(
                    f"Task {task.id} has no workflow in context — "
                    f"expected for CLI/web-created tasks"
                )
            if not has_workflow:
                self.logger.debug(f"Checking if code review needed for {task.id}")
                self._review_cycle.queue_code_review_if_needed(task, response)
                self._review_cycle.queue_review_fix_if_needed(task, response, self._sync_jira_status)

            # Verdict was already set in _handle_successful_completion() before
            # serialization. Check it here to decide whether to skip chain enforcement.
            skip_chain = task.context.get("verdict") == "no_changes"
            if skip_chain:
                self.logger.info(
                    f"No-changes verdict at plan step for task {task.id} — "
                    f"terminating workflow (nothing to implement or PR)"
                )

            self._git_ops.detect_implementation_branch(task)

            if not skip_chain:
                self._enforce_workflow_chain(task, response, routing_signal=routing_signal)

            # Emit waterfall summary when the chain terminates (either at
            # terminal step or via no_changes early exit)
            if skip_chain or self._is_at_terminal_workflow_step(task):
                self._emit_workflow_summary(task)

            # Push + PR lifecycle after chain routing. Downstream agents pick up
            # from queue asynchronously, so push completes before they fetch the branch.
            self._git_ops.push_and_create_pr_if_needed(task)
            self._git_ops.manage_pr_lifecycle(task)

        self._extract_and_store_memories(task, response)
        tool_call_count = self._analyze_tool_patterns(task)
        self._log_task_completion_metrics(task, response, task_start_time, tool_call_count=tool_call_count)

    def _log_task_completion_metrics(self, task: Task, response, task_start_time, *, tool_call_count=None) -> None:
        """Log token usage, cost, and completion events.

        Delegated to BudgetManager.
        """
        ctx_status = self._context_window_manager.get_budget_status() if self._context_window_manager else None
        self._budget.log_task_completion_metrics(
            task, response, task_start_time,
            tool_call_count=tool_call_count,
            root_task_id=task.root_id,
            context_budget_status=ctx_status,
        )

    def _approval_verdict(self, task: Task) -> str:
        """Return the appropriate approval verdict for the current workflow step.

        preview_review uses "preview_approved" so the preview_approved DAG edge
        fires instead of the generic "approved" edge, which routes to qa_review
        rather than implement.
        """
        if task.context.get("workflow_step") in PREVIEW_REVIEW_STEPS:
            return "preview_approved"
        return "approved"

    def _set_structured_verdict(self, task: Task, response) -> None:
        """Parse review outcome and store verdict + audit trail before task serialization.

        Only qa/architect agents with a workflow produce verdicts.
        Engineer output may contain stray keywords that cause false verdicts.
        """
        if not task.context.get("workflow"):
            return
        if self.config.base_id not in ("qa", "architect"):
            return

        content = getattr(response, "content", "") or ""
        outcome, audit = self._review_cycle._parse_review_outcome_audited(content)

        workflow_step = task.context.get("workflow_step", "")

        if outcome.approved:
            task.context["verdict"] = self._approval_verdict(task)
            audit.method = "review_outcome"
        elif outcome.needs_fix:
            task.context["verdict"] = "needs_fix"
            audit.method = "review_outcome"
        else:
            _REVIEW_STEP_IDS = PREVIEW_REVIEW_STEPS | {"code_review", "qa_review"}
            if workflow_step in _REVIEW_STEP_IDS:
                self.logger.warning(
                    f"Ambiguous review outcome at step {workflow_step!r} for "
                    f"task {task.id} — not setting verdict (chain will halt)"
                )
                audit.method = "ambiguous_halt"
            else:
                task.context["verdict"] = self._approval_verdict(task)
                audit.method = "ambiguous_default"

        # no_changes marker overrides any previous verdict at plan step
        if (self.config.base_id == "architect"
                and task.context.get("workflow_step", task.type) in ("plan", "planning")):
            if self._is_no_changes_response(content):
                task.context["verdict"] = "no_changes"
                audit.method = "no_changes_marker"
                audit.no_changes_marker_found = True

        audit.agent_id = self.config.id
        audit.workflow_step = workflow_step
        audit.task_id = task.id
        audit.value = task.context.get("verdict")

        task.context["verdict_audit"] = audit.to_dict()
        self._session_logger.log("verdict_audit", **audit.to_dict())

    @staticmethod
    def _is_no_changes_response(content: str) -> bool:
        """Detect if response indicates no code changes are needed.

        Requires the LLM to emit an explicit marker rather than relying
        on regex over free-text, which is prone to false positives when
        the planner describes existing state.
        """
        if not content:
            return False
        # Marker must appear in the first 200 chars (before any plan body)
        return _NO_CHANGES_MARKER in content[:200]

    def _is_implementation_step(self, task: Task) -> bool:
        """Whether this task is an implementation step that must produce code.

        Returns True for workflow "implement" steps and for engineer agents
        running without an explicit workflow step (direct-queue work).
        Returns False for prose-only steps (plan, review, QA, PR creation).
        """
        step = task.context.get("workflow_step")
        if step in _IMPLEMENTATION_STEP_IDS:
            return True
        if step in _NON_CODE_STEP_IDS:
            return False
        # No explicit step — engineer agents doing direct-queue work
        return self.config.base_id == "engineer"

    def _resolve_budget_ceiling(self, task: Task) -> Optional[float]:
        """Resolve USD budget ceiling from task effort and/or absolute cap.

        Returns the tighter of the two when both are set, or whichever is
        available when only one is configured.  Returns None when neither applies.
        """
        effort_ceiling = None
        if self._optimization_config.get("enable_effort_budget_ceilings", False):
            effort = task.estimated_effort
            if not effort:
                effort = self._budget.derive_effort_from_plan(task.plan)
            effort_ceiling = self._budget.get_effort_ceiling(effort.upper())

        absolute = self._optimization_config.get("absolute_budget_ceiling_usd")

        if effort_ceiling is not None and absolute is not None:
            return min(effort_ceiling, absolute)
        return effort_ceiling if effort_ceiling is not None else absolute

    @staticmethod
    def _extract_partial_progress(content: str, max_bytes: int = 2048) -> str:
        """Extract meaningful text blocks from a partial LLM response.

        Filters out [Tool Call: ...] noise and keeps the last few
        substantive text blocks so retries can pick up where we left off.
        """
        if not content:
            return ""

        # Split on tool-call markers and keep non-marker blocks
        blocks = re.split(r'\[Tool Call:[^\]]*\]', content)
        meaningful = [b.strip() for b in blocks if b.strip()]

        # Keep last 5 meaningful blocks
        meaningful = meaningful[-5:]
        joined = "\n\n".join(meaningful)

        # Enforce size cap
        if len(joined.encode("utf-8", errors="replace")) > max_bytes:
            joined = joined[-max_bytes:]

        return joined

    def _finalize_failed_attempt(
        self, task: Task, working_dir: Optional[Path], *,
        content: Optional[str] = None,
        error: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Consolidate all attempt preservation for retry awareness.

        Called from all failure/interruption paths before task reset.
        Captures partial progress, records attempt to disk, preserves code.
        """
        # Extract partial progress from LLM content
        if content:
            summary = self._extract_partial_progress(content)
            if summary:
                task.context["_previous_attempt_summary"] = summary

        # Record attempt: commit WIP, push, persist to disk
        try:
            from .attempt_tracker import record_attempt
            record = record_attempt(
                workspace=self.workspace,
                task=task,
                agent_id=self.config.id,
                working_dir=working_dir,
                error=error or task.last_error,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                logger=self.logger,
            )
            if record and record.branch:
                task.context["_previous_attempt_branch"] = record.branch
                task.context["_previous_attempt_commit_sha"] = record.commit_sha
        except Exception as e:
            self.logger.debug(f"Attempt recording failed (non-fatal): {e}")

    def _can_salvage_verdict(self, task: Task, response) -> bool:
        """Check if a failed response contains a valid review verdict worth salvaging.

        Claude CLI occasionally exits 1 after producing complete output
        (e.g., cleanup failure after verdict). Salvage rather than retry.
        """
        if self.config.base_id not in ("qa", "architect"):
            return False

        content = getattr(response, "content", "") or ""
        if len(content) < 200:
            return False

        outcome = self._review_cycle.parse_review_outcome(content)
        return outcome.approved or outcome.needs_fix

    async def _handle_failed_response(self, task: Task, response, *, working_dir: Optional[Path] = None) -> None:
        """Handle failed LLM response."""
        from datetime import datetime

        task.last_error = response.error or "Unknown error"

        # Accumulate cost so budget ceilings see failed-attempt spend
        this_cost = self._budget.estimate_cost(response)
        prev = task.context.get("_cumulative_cost", 0.0)
        task.context["_cumulative_cost"] = prev + this_cost

        # Record attempt: partial progress + commit WIP + push + persist
        self._finalize_failed_attempt(
            task, working_dir,
            content=response.content,
            error=response.error,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=this_cost,
        )

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
            tokens_in=response.input_tokens,
            tokens_out=response.output_tokens,
            cost=this_cost,
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
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=this_cost,
        ))

        # Record failure outcome for intelligent routing
        if self._model_success_store is not None and response.model_used:
            repo_slug = task.context.get("github_repo", "")
            task_type_str = task.type if isinstance(task.type, str) else task.type.value
            self._model_success_store.record_outcome(
                repo_slug=repo_slug,
                model_tier=response.model_used,
                task_type=task_type_str,
                success=False,
                cost=this_cost,
            )

        await self._handle_failure(task)

    def _cleanup_task_execution(self, task: Task, lock) -> None:
        """Cleanup after task execution.

        IDLE is set AFTER worktree cleanup so the agent stays in the
        protected set during periodic cleanup (prevents race where our
        worktree gets evicted mid-cleanup).
        """
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
        """Get working directory with one retry if the path vanishes between creation and use."""
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
        """No-op. Automatic worktree deletion is disabled.

        Worktrees are only cleaned up via the explicit CLI command
        `agent cleanup-worktrees`. This prevents race conditions where
        active worktrees get deleted while agents are still using them.
        """
        return

    async def _watch_for_interruption(self) -> None:
        """Poll for pause/stop signals during LLM execution.

        Completes (returns) when an interruption is detected, which causes
        the asyncio.wait race in _handle_task to cancel the LLM call.
        Also monitors worktree existence — returns (triggering LLM cancellation)
        when directory vanishes instead of letting the LLM burn tool calls.
        """
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
        from datetime import datetime

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
        # Update workflow router and executor session loggers for this task
        self._workflow_router.set_session_logger(self._session_logger)
        self._workflow_executor.set_session_logger(self._session_logger)
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
        """Execute LLM with interruption watching, return response or None if interrupted."""
        from datetime import datetime

        # Setup tool activity callback
        _tool_call_count = [0]
        _last_write_time = [0.0]
        _consecutive_bash = [0]
        _consecutive_diagnostic = [0]
        _bash_commands: list[list[str]] = [[]]  # mutable list for closure access
        _soft_threshold_logged = [False]
        _circuit_breaker_event = asyncio.Event()
        _diagnostic_trip = [False]  # tracks which trigger fired
        _exploration_alerted = [False]
        # Resolve once — workflow_step is immutable during a session
        _workflow_step = task.context.get("workflow_step")
        _exploration_threshold = (
            self._exploration_alert_thresholds.get(_workflow_step, self._exploration_alert_threshold)
            if _workflow_step else self._exploration_alert_threshold
        )
        _subagent_spawns: list[list[dict]] = [[]]

        def _emit_subagent_summary(outcome: str, orphan_risk: bool):
            if _subagent_spawns[0]:
                self._session_logger.log(
                    "subagent_summary",
                    total_spawned=len(_subagent_spawns[0]),
                    session_outcome=outcome,
                    orphan_risk=orphan_risk,
                    spawns=_subagent_spawns[0],
                )

        _DIVERSITY_THRESHOLD = 0.5
        # Tighter interval for implementation steps to reduce max work loss
        _COMMIT_CHECKPOINT_INTERVAL = 15 if self._is_implementation_step(task) else 25

        def _on_tool_activity(tool_name: str, tool_input_summary: Optional[str]):
            try:
                _tool_call_count[0] += 1

                if tool_name == "Task":
                    _subagent_spawns[0].append({
                        "summary": tool_input_summary,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    self._session_logger.log(
                        "subagent_spawned",
                        spawn_index=len(_subagent_spawns[0]),
                        tool_input_summary=tool_input_summary,
                    )

                # Circuit breaker: track consecutive Bash calls with command diversity
                if tool_name == "Bash":
                    _consecutive_bash[0] += 1
                    _bash_commands[0].append(tool_input_summary or "")
                    count = _consecutive_bash[0]
                    threshold = self._max_consecutive_tool_calls

                    # Diagnostic sub-breaker: catches stuck agents probing their
                    # environment after worktree deletion. These commands are
                    # inherently diverse so the main diversity heuristic never fires.
                    if _is_diagnostic_command(tool_input_summary or ""):
                        _consecutive_diagnostic[0] += 1
                        if _consecutive_diagnostic[0] >= self._max_consecutive_diagnostic_calls:
                            _diagnostic_trip[0] = True
                            _circuit_breaker_event.set()
                            return
                    else:
                        _consecutive_diagnostic[0] = 0

                    if count >= threshold:
                        unique = len(set(_bash_commands[0]))
                        diversity = unique / count if count > 0 else 0.0

                        if diversity <= _DIVERSITY_THRESHOLD:
                            productive = sum(1 for c in _bash_commands[0] if _is_productive_command(c))
                            productive_ratio = productive / count

                            if productive_ratio > _PRODUCTIVE_RATIO_THRESHOLD:
                                # Productive workflow — use higher threshold before tripping
                                effective = threshold * _PRODUCTIVE_THRESHOLD_MULTIPLIER
                                if count >= effective:
                                    self.logger.warning(
                                        f"Circuit breaker: {count} consecutive Bash calls, "
                                        f"low diversity={diversity:.2f} (unique_commands={unique}), "
                                        f"productive_ratio={productive_ratio:.2f} exceeded hard ceiling={effective}"
                                    )
                                    _circuit_breaker_event.set()
                                elif not _soft_threshold_logged[0]:
                                    self.logger.info(
                                        f"Circuit breaker deferred: {count} consecutive Bash calls, "
                                        f"productive_ratio={productive_ratio:.2f} (effective threshold={effective})"
                                    )
                                    _soft_threshold_logged[0] = True
                            else:
                                self.logger.warning(
                                    f"Circuit breaker: {count} consecutive Bash calls, "
                                    f"low diversity={diversity:.2f} (unique_commands={unique}), "
                                    f"productive_ratio={productive_ratio:.2f}"
                                )
                                _circuit_breaker_event.set()
                        elif not _soft_threshold_logged[0]:
                            # Diverse commands — log once and let them continue
                            self.logger.info(
                                f"Circuit breaker deferred: {count} consecutive Bash calls "
                                f"but diversity={diversity:.2f} (unique_commands={unique})"
                            )
                            _soft_threshold_logged[0] = True
                else:
                    _consecutive_bash[0] = 0
                    _consecutive_diagnostic[0] = 0
                    _bash_commands[0] = []
                    _soft_threshold_logged[0] = False

                # Exploration metric: one-time alert when total calls exceed threshold
                total = _tool_call_count[0]
                if total >= _exploration_threshold and not _exploration_alerted[0]:
                    _exploration_alerted[0] = True
                    self.logger.info(
                        f"Exploration alert: {total} tool calls in session "
                        f"(threshold={_exploration_threshold}, step={_workflow_step or 'standalone'})"
                    )
                    self._session_logger.log(
                        "exploration_alert",
                        total_tool_calls=total,
                        threshold=_exploration_threshold,
                        workflow_step=_workflow_step,
                        agent_type=self.config.base_id,
                    )
                    self.activity_manager.append_event(ActivityEvent(
                        type="exploration_alert",
                        agent=self.config.id,
                        task_id=task.id,
                        title=(
                            f"Exploration: {total} calls "
                            f"(threshold={_exploration_threshold}, step={_workflow_step or 'standalone'})"
                        ),
                        timestamp=datetime.now(timezone.utc),
                    ))

                # Periodic checkpoint: commit + push so work survives worktree deletion
                if (_tool_call_count[0] % _COMMIT_CHECKPOINT_INTERVAL == 0
                        and working_dir and working_dir.exists()):
                    committed = self._git_ops.safety_commit(
                        working_dir,
                        f"WIP: periodic checkpoint (tool call {_tool_call_count[0]})",
                    )
                    if committed:
                        self._git_ops.push_if_unpushed()

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

        # Log LLM start
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

        # PREVIEW tasks get tool-level read-only enforcement so the model cannot write
        # files even if it ignores the prompt instructions. This supplements (not replaces)
        # the prompt injection in _inject_preview_mode().
        preview_allowed_tools: list[str] | None = None
        if task.type == TaskType.PREVIEW:
            preview_allowed_tools = [
                "Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch",
            ]

        # Behavioral directive to reduce redundant file reads within a session
        efficiency_parts = [
            "EFFICIENCY: Track which files you have already read in this session. "
            "When you read a file, read it ONCE in full — never chunk through the same "
            "file with repeated Read calls at different offsets. After one full read, "
            "the contents are in your context; use your conversation history to recall "
            "them. Use Grep for targeted searches instead of reading files at all.",
        ]
        efficiency_parts.append(
            "COMMIT DISCIPLINE: After completing each major deliverable, immediately "
            "commit AND push your work (git add + git commit + git push origin HEAD). "
            "This preserves progress if you run out of context or lose your working "
            "directory. Prioritize shipping all deliverables over perfecting any single one."
        )
        efficiency_parts.append(
            "FAILURE CIRCUIT BREAKER: If 3+ consecutive bash/shell commands fail with errors, "
            "STOP. Do not attempt more diagnostic commands. Instead, report what went wrong "
            "and what you were trying to do."
        )
        if self.config.id == "architect":
            efficiency_parts.append(
                "EXPLORATION: Do not delegate file exploration to subagents — "
                "explore the codebase directly with Grep and Read."
            )
        efficiency_directive = " ".join(efficiency_parts)

        # Compute routing signals for intelligent model selection
        _estimated_lines = 0
        if task.plan:
            from .task_decomposer import estimate_plan_lines
            _estimated_lines = estimate_plan_lines(task.plan)

        _budget_remaining_usd = None
        _budget_ceiling = task.context.get("_budget_ceiling")
        _cumulative_cost = task.context.get("_cumulative_cost", 0.0)
        if _budget_ceiling is not None:
            _budget_remaining_usd = max(0.0, _budget_ceiling - _cumulative_cost)

        # Race LLM execution against pause/stop signal watcher
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
                estimated_lines=_estimated_lines,
                budget_remaining_usd=_budget_remaining_usd,
                allowed_tools=preview_allowed_tools,
                append_system_prompt=efficiency_directive,
                env_vars=self._git_ops.worktree_env_vars,
            ),
            task_id=task.id,
            on_tool_activity=_on_tool_activity,
            on_session_tool_call=self._session_logger.log_tool_call,
            on_session_tool_result=self._session_logger.log_tool_result,
        )
        llm_task = asyncio.create_task(llm_coro)
        watcher_task = asyncio.create_task(self._watch_for_interruption())
        circuit_breaker_task = asyncio.create_task(_circuit_breaker_event.wait())

        done, pending = await asyncio.wait(
            [llm_task, watcher_task, circuit_breaker_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if circuit_breaker_task in done:
            # Stuck agent detected — kill subprocess and return synthetic failure
            count = _consecutive_bash[0]
            diag_count = _consecutive_diagnostic[0]
            is_diagnostic = _diagnostic_trip[0]
            commands = _bash_commands[0]
            unique = len(set(commands))
            diversity = unique / count if count > 0 else 0.0
            productive = sum(1 for c in commands if _is_productive_command(c))
            productive_ratio = productive / count if count > 0 else 0.0
            trigger = "diagnostic" if is_diagnostic else "volume"

            wd_str = str(working_dir) if working_dir else "N/A"
            wd_exists = working_dir.exists() if working_dir else None

            if is_diagnostic:
                self.logger.warning(
                    f"Diagnostic circuit breaker tripped for task {task.id}: "
                    f"{diag_count} consecutive diagnostic commands "
                    f"(threshold={self._max_consecutive_diagnostic_calls})"
                    f" (working_dir={wd_str}, exists={wd_exists})"
                )
                event_title = (
                    f"Diagnostic circuit breaker: {diag_count} consecutive diagnostic commands"
                    f" — {wd_str}"
                )
                error_msg = (
                    f"Stuck agent detected: {diag_count} consecutive diagnostic commands — "
                    f"working directory {wd_str} (exists={wd_exists})"
                )
            else:
                self.logger.warning(
                    f"Circuit breaker tripped for task {task.id}: "
                    f"{count} consecutive Bash calls (threshold={self._max_consecutive_tool_calls}, "
                    f"diversity={diversity:.2f}, unique_commands={unique}, "
                    f"productive_ratio={productive_ratio:.2f})"
                    f" (working_dir={wd_str}, exists={wd_exists})"
                )
                event_title = (
                    f"Circuit breaker: {count} consecutive Bash calls "
                    f"(diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f})"
                    f" — {wd_str}"
                )
                error_msg = (
                    f"Circuit breaker tripped: {count} consecutive Bash calls without other tool types "
                    f"(diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f}). "
                    f"Working directory {wd_str} (exists={wd_exists})."
                )

            # Auto-commit WIP before killing — prevents code loss
            await self._auto_commit_wip(task, working_dir, count)

            self.llm.cancel()

            # Record attempt for retry awareness (partial progress + disk persistence)
            try:
                partial = self.llm.get_partial_output()
            except Exception:
                partial = None
            self._finalize_failed_attempt(task, working_dir, content=partial, error=error_msg)

            llm_task.cancel()
            watcher_task.cancel()
            for t in [llm_task, watcher_task]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            self._session_logger.log(
                "circuit_breaker",
                trigger=trigger,
                consecutive_bash=count,
                consecutive_diagnostic=diag_count,
                threshold=self._max_consecutive_diagnostic_calls if is_diagnostic else self._max_consecutive_tool_calls,
                unique_commands=unique,
                diversity=round(diversity, 2),
                productive_ratio=round(productive_ratio, 2),
                working_dir=wd_str,
                working_dir_exists=wd_exists,
                last_commands=[c[:200] for c in commands[-5:]],
            )

            self.activity_manager.append_event(ActivityEvent(
                type="circuit_breaker",
                agent=self.config.id,
                task_id=task.id,
                title=event_title,
                timestamp=datetime.now(timezone.utc),
            ))
            _emit_subagent_summary("circuit_breaker", True)
            return LLMResponse(
                content="",
                model_used="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="circuit_breaker",
                latency_ms=0,
                success=False,
                error=error_msg,
            )

        if watcher_task in done:
            # Determine cause: worktree vanished vs pause/stop signal
            wt = self._git_ops.active_worktree
            worktree_gone = wt is not None and not wt.exists()

            if worktree_gone:
                error_msg = f"Worktree vanished during LLM execution: {wt}"
                event_type = "worktree_vanished"
            else:
                error_msg = "Interrupted during LLM execution"
                event_type = "interrupted"

            self.logger.info(f"{error_msg} for task {task.id}, cancelling LLM")
            self.llm.cancel()
            llm_task.cancel()
            circuit_breaker_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.debug(f"LLM task raised during cancellation: {e}")

            # Harvest partial output and record attempt for retry awareness
            partial = self.llm.get_partial_output()
            self._finalize_failed_attempt(
                task, working_dir,
                content=partial,
                error=error_msg,
            )

            if worktree_gone:
                self._session_logger.log(
                    "worktree_vanished",
                    path=str(wt),
                    task_id=task.id,
                )

            task.last_error = error_msg
            task.reset_to_pending()
            self.queue.update(task)
            self.activity_manager.append_event(ActivityEvent(
                type=event_type,
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.now(timezone.utc),
            ))
            self.logger.info(f"Task {task.id} reset to pending after {event_type}")
            _emit_subagent_summary(event_type, True)
            return None
        else:
            # LLM finished first — cancel watcher and circuit breaker, return response
            watcher_task.cancel()
            circuit_breaker_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass
            result = llm_task.result()
            outcome = "success" if result and result.success else "failed"
            _emit_subagent_summary(outcome, not (result and result.success))
            return result

    async def _auto_commit_wip(self, task: Task, working_dir: Path, bash_count: int) -> None:
        """Best-effort WIP commit so code isn't lost when circuit breaker trips."""
        try:
            committed = self._git_ops.safety_commit(
                working_dir,
                f"WIP: auto-save before circuit breaker ({bash_count} consecutive Bash calls)",
            )
            if committed:
                self._session_logger.log("wip_auto_commit", task_id=task.id, bash_count=bash_count)
        except Exception:
            pass

    def _log_routing_decision(self, task: Task, response) -> None:
        """Log intelligent routing decision if one was made."""
        backend = self.llm
        selector = getattr(backend, 'model_selector', None)
        decision = getattr(selector, '_last_routing_decision', None) if selector else None
        if decision is None:
            return
        # Clear stashed decision so it doesn't leak to the next call
        selector._last_routing_decision = None
        self._session_logger.log(
            "model_routing_decision",
            chosen_tier=decision.chosen_tier,
            scores=decision.scores,
            signals=decision.signals,
            fallback=decision.fallback,
            model_used=response.model_used,
        )
        self.activity_manager.append_event(ActivityEvent(
            type="model_routing_decision",
            agent=self.config.id,
            task_id=task.id,
            title=f"Intelligent routing: {decision.chosen_tier} (scores: {decision.scores})",
            timestamp=datetime.now(timezone.utc),
        ))

    def _process_llm_completion(self, response, task: Task) -> None:
        """Log LLM completion and update context window manager."""
        from datetime import datetime

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

            self._session_logger.log(
                "context_budget_update",
                utilization_percent=budget_status["utilization_percent"],
                used_tokens=budget_status["used_so_far"],
                total_budget=budget_status["total_budget"],
                remaining=budget_status["remaining"],
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
                self._session_logger.log(
                    "context_budget_critical",
                    task_id=task.id,
                    utilization_percent=budget_status["utilization_percent"],
                )

        # Clear tool activity after LLM completes
        try:
            self.activity_manager.update_tool_activity(self.config.id, None)
        except Exception:
            pass

    async def _handle_task(self, task: Task, *, lock: Optional["FileLock"] = None) -> None:
        """Handle task execution with retry/escalation logic."""
        from datetime import datetime
        from ..queue.locks import FileLock

        # Normalize legacy workflow names and validate task
        self._normalize_workflow(task)
        if not self._validate_task_or_reject(task):
            if lock:
                lock.release()
            return

        # Use pre-acquired lock from claim(), or fall back to separate acquire
        if lock is None:
            lock = self.queue.acquire_lock(task.id, self.config.id)
            if not lock:
                self.logger.warning(f"⏸️  Could not acquire lock, will retry later")
                return

        task_start_time = datetime.now(timezone.utc)

        # Setup task context (logging, session logger)
        self._setup_task_context(task, task_start_time)

        # Initialize context window manager
        self._setup_context_window_manager_for_task(task)

        working_dir = None
        _cost_before_try = task.context.get("_cumulative_cost", 0.0)
        try:
            # Initialize task execution state
            self._initialize_task_execution(task, task_start_time)

            # Get working directory for task (worktree, target repo, or framework workspace)
            working_dir = self._get_validated_working_directory(task)
            self.logger.info(f"Working directory: {working_dir}")

            # Discover committed work from previous attempts so retry prompt knows what exists
            if task.retry_count > 0:
                branch_work = self._git_ops.discover_branch_work(working_dir)

                # Fallback: if discover found nothing, check attempt history for a pushed branch
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

            # Index codebase for structural context (cached by commit SHA)
            self._try_index_codebase(task, working_dir)

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

            # Compose team for task execution
            team_agents = self._compose_team_for_task(task)

            # Second validation: catch deletions during prompt build/indexing
            if not working_dir.exists():
                raise RuntimeError(
                    f"Working directory vanished before LLM start: {working_dir}. "
                    f"Likely deleted by sibling agent cleanup during prompt build."
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
                file_reads = self._populate_read_cache(task, working_dir=working_dir)
                self._measure_cache_effectiveness(task, file_reads, working_dir=working_dir)
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
                file_reads = self._populate_read_cache(task, working_dir=working_dir)
                self._measure_cache_effectiveness(task, file_reads, working_dir=working_dir)
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
            _already_accumulated = task.context.get("_cumulative_cost", 0.0) > _cost_before_try
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
            self._session_logger.close()
            self._session_logger = noop_logger()
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
            summary_prompt = (
                "Extract the key outcomes from the following agent output. "
                "Return exactly 3 bullet points, each on its own line starting with '- '. "
                "Focus on: what was done, what files/PRs were created, and the final status.\n\n"
                "<agent_output>\n"
                f"{response[:SUMMARY_CONTEXT_MAX_CHARS]}\n"
                "</agent_output>"
            )
            try:
                summary_response = await self.llm.complete(LLMRequest(
                    prompt=summary_prompt,
                    system_prompt="You are a summarization tool. Output only bullet points. Never converse, ask questions, or add commentary.",
                    model="haiku",
                    temperature=0.0,
                    max_tokens=512,
                ))

                if summary_response.success and summary_response.content:
                    content = summary_response.content.strip()
                    # Guard against conversational garbage from the LLM
                    if content.lower().startswith(_CONVERSATIONAL_PREFIXES):
                        self.logger.warning("Haiku returned conversational response, using fallback")
                    else:
                        return content[:SUMMARY_MAX_LENGTH]
                else:
                    self.logger.warning(f"Haiku summary failed: {summary_response.error}")
            except Exception as e:
                self.logger.warning(f"Failed to extract summary with Haiku: {e}")

        # Guaranteed fallback
        return extracted[0] if extracted else f"Task {get_type_str(task.type)} completed"

    # -- Upstream Context Handoff --

    UPSTREAM_CONTEXT_MAX_CHARS = 15000
    UPSTREAM_INLINE_MAX_CHARS = 15000

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

            content = _strip_tool_call_markers(response.content or "")
            if len(content) > self.UPSTREAM_CONTEXT_MAX_CHARS:
                content = content[:self.UPSTREAM_CONTEXT_MAX_CHARS] + "\n\n[truncated]"

            context_file = summaries_dir / f"{task.id}-{self.config.base_id}.md"
            atomic_write_text(context_file, content)

            # Store path in task context for chain propagation
            task.context["upstream_context_file"] = str(context_file)
            # Store inline for cross-worktree portability (file path may not resolve)
            task.context["upstream_summary"] = content[:self.UPSTREAM_INLINE_MAX_CHARS]
            task.context["upstream_source_agent"] = self.config.base_id
            step = task.context.get("workflow_step")
            if step:
                task.context["upstream_source_step"] = step
            self.logger.debug(f"Saved upstream context ({len(content)} chars) to {context_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save upstream context for {task.id}: {e}")

    def _save_step_to_chain_state(
        self, task: Task, response, *,
        working_dir: Optional[Path] = None,
        task_start_time: Optional[datetime] = None,
    ) -> None:
        """Append a structured step record to the chain state file.

        Called AFTER plan extraction + verdict setting so all structured
        data is available. Non-fatal — workflow continues on failure.
        """
        try:
            from .chain_state import append_step

            # Compute tool stats from session log (session is fully flushed by this point)
            tool_stats_dict = None
            if self._session_logging_enabled:
                tool_stats_dict = self._compute_tool_stats_for_chain(task)

            content = getattr(response, "content", "") or ""
            state = append_step(
                workspace=self.workspace,
                task=task,
                agent_id=self.config.base_id,
                response_content=content,
                working_dir=working_dir,
                tool_stats=tool_stats_dict,
                started_at=task_start_time,
            )

            # Store files_modified in task context for chain propagation
            if state.steps:
                last_step = state.steps[-1]
                if last_step.files_modified:
                    task.context["files_modified"] = last_step.files_modified

            self.logger.debug(
                f"Chain state: appended step {task.context.get('workflow_step', 'unknown')} "
                f"({len(state.steps)} total steps)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save chain state for {task.id}: {e}")

    def _emit_workflow_summary(self, task: Task) -> None:
        """Emit a waterfall summary event capturing the full workflow timeline.

        Non-fatal — failure to emit doesn't block task completion.
        """
        try:
            from .chain_state import load_chain_state, build_workflow_summary

            state = load_chain_state(self.workspace, task.root_id)
            if state is None or not state.steps:
                return

            summary = build_workflow_summary(state)

            # Enrich with PR URL if available
            pr_url = task.context.get("pr_url")
            if pr_url:
                summary["pr_url"] = pr_url

            if self._session_logging_enabled:
                self._session_logger.log("workflow_summary", **summary)

            self.activity_manager.append_event(ActivityEvent(
                type="workflow_summary",
                agent=self.config.id,
                task_id=task.id,
                title=f"Workflow {summary.get('outcome', 'unknown')}: {len(summary.get('steps', []))} steps",
                timestamp=datetime.now(timezone.utc),
                root_task_id=task.root_id,
                duration_ms=int(summary["total_duration_seconds"] * 1000) if summary.get("total_duration_seconds") else None,
            ))

            self.logger.info(
                f"Workflow summary: {summary.get('outcome')} in {len(summary.get('steps', []))} steps, "
                f"{summary.get('total_duration_seconds')}s"
            )
        except Exception as e:
            self.logger.warning(f"Failed to emit workflow summary for {task.id}: {e}")

    def _compute_tool_stats_for_chain(self, task: Task) -> Optional[Dict]:
        """Compute tool usage stats dict for embedding in chain state.

        Returns a plain dict suitable for JSON serialization, or None
        if the session log doesn't exist or has no tool calls.

        Caches on task.context["_tool_stats_cache"] so _analyze_tool_patterns
        can reuse the parsed result without re-reading the session log.
        """
        session_path = self._session_logs_dir / "sessions" / f"{task.id}.jsonl"
        if not session_path.exists():
            return None

        try:
            from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer, compute_tool_usage_stats
            from dataclasses import asdict

            analyzer = ToolPatternAnalyzer()
            tool_calls = analyzer.extract_tool_calls(session_path)
            if not tool_calls:
                return None

            stats = compute_tool_usage_stats(tool_calls)
            stats_dict = asdict(stats)
            # Cache for reuse by _analyze_tool_patterns (avoids double-parse)
            task.context["_tool_stats_cache"] = stats_dict
            return stats_dict
        except Exception as e:
            self.logger.debug(f"Tool stats computation failed (non-fatal): {e}")
            return None

    def _populate_read_cache(self, task: Task, working_dir: Optional[Path] = None) -> list[str]:
        """Populate shared read cache with files read during this session.

        Appends to .agent-communication/read-cache/{root_task_id}.json so
        downstream chain steps can skip re-reading the same files.
        Cache keys are repo-relative so they match across worktrees.
        Non-fatal — workflow continues even if this fails.

        Returns the raw file_reads list so callers can reuse it without
        re-parsing the session log.
        """
        try:
            from ..utils.atomic_io import atomic_write_text
            from ..utils.file_summarizer import summarize_file

            file_reads = self._session_logger.extract_file_reads()
            if not file_reads:
                return file_reads

            root_task_id = task.root_id
            cache_dir = self.workspace / ".agent-communication" / "read-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{root_task_id}.json"

            # Load existing cache (may have entries from MCP tool calls with summaries)
            existing: dict = {}
            if cache_file.exists():
                try:
                    existing = json.loads(cache_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = {}

            entries = existing.get("entries", {})
            pre_existing_keys = set(entries.keys())
            step = task.context.get("workflow_step", "unknown")
            now_iso = datetime.now(timezone.utc).isoformat()

            # Detect files modified in this step so stale summaries get refreshed
            from .chain_state import _collect_files_modified
            modified_in_step = set(_collect_files_modified(working_dir))

            added = 0
            refreshed = 0
            for file_path in file_reads:
                # Store repo-relative key for cross-worktree portability
                cache_key = _to_relative_path(file_path, working_dir)
                is_modified = cache_key in modified_in_step
                if cache_key not in entries or is_modified:
                    # Preserve original reader lineage when refreshing modified entries
                    old_entry = entries.get(cache_key, {})
                    entry = {
                        "summary": summarize_file(file_path),
                        "read_by": old_entry.get("read_by", self.config.base_id),
                        "read_at": old_entry.get("read_at", now_iso),
                        "workflow_step": old_entry.get("workflow_step", step),
                    }
                    if is_modified:
                        entry["modified_by"] = self.config.base_id
                        entry["modified_at"] = now_iso
                        refreshed += 1
                    else:
                        # New entry -- attribute to current agent
                        entry["read_by"] = self.config.base_id
                        entry["read_at"] = now_iso
                        entry["workflow_step"] = step
                        added += 1
                    entries[cache_key] = entry

            # Measure cache bypass rate at the storage layer (complements
            # _measure_cache_effectiveness which measures from prompt-injected paths)
            if pre_existing_keys:
                read_keys = {_to_relative_path(fp, working_dir) for fp in file_reads}
                re_read = read_keys & pre_existing_keys
                # Partition: re-reads of modified files are justified, others are wasteful
                modified_keys = {k for k, v in entries.items() if v.get("modified_by")}
                justified = sorted(re_read & modified_keys)
                wasteful = sorted(re_read - modified_keys)
                wasteful_rate = len(wasteful) / len(pre_existing_keys)
                self._session_logger.log(
                    "read_cache_bypass",
                    cached_files=len(pre_existing_keys),
                    re_read_count=len(re_read),
                    bypass_rate=round(len(re_read) / len(pre_existing_keys), 3),
                    justified_rereads=len(justified),
                    wasteful_rereads=len(wasteful),
                    wasteful_rate=round(wasteful_rate, 3),
                    new_reads=added,
                    total_reads=len(file_reads),
                    re_read_files=sorted(re_read)[:20],
                )

            if added + refreshed == 0:
                return file_reads

            cache_data = {
                "root_task_id": root_task_id,
                "entries": entries,
            }
            atomic_write_text(cache_file, json.dumps(cache_data))
            self.logger.debug(
                f"Read cache: added {added}, refreshed {refreshed} "
                f"({len(entries)} total) for {root_task_id}"
            )

            # Merge into repo-scoped cache for cross-attempt persistence
            github_repo = task.context.get("github_repo")
            if github_repo:
                self._update_repo_cache(cache_dir, github_repo, entries)
            return file_reads
        except Exception as e:
            self.logger.warning(f"Failed to populate read cache for {task.id}: {e}")
            return []

    def _measure_cache_effectiveness(
        self, task: Task, file_reads: list[str], *, working_dir: Optional[Path] = None,
    ) -> None:
        """Log how well the read cache prevented redundant file reads.

        Compares the set of paths injected into the prompt (from previous steps)
        against the files actually read during this session. Non-fatal.
        """
        try:
            injected = self._prompt_builder._injected_cache_paths
            if not injected:
                return

            session_paths = {_to_relative_path(p, working_dir) for p in file_reads}

            cache_hits = injected - session_paths   # cached and NOT re-read
            cache_misses = injected & session_paths  # cached but re-read anyway
            new_reads = session_paths - injected     # not cached, first time

            # Load cache to identify modified files for justified/wasteful split
            modified_keys: set[str] = set()
            root_task_id = task.root_id
            cache_file = self.workspace / ".agent-communication" / "read-cache" / f"{root_task_id}.json"
            if cache_file.exists():
                try:
                    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                    modified_keys = {
                        k for k, v in cache_data.get("entries", {}).items()
                        if v.get("modified_by")
                    }
                except (json.JSONDecodeError, OSError):
                    pass

            justified = cache_misses & modified_keys   # re-read because file changed
            wasteful = cache_misses - modified_keys    # avoidable re-reads

            total_cached = len(injected)
            hit_rate = len(cache_hits) / total_cached if total_cached else 0.0
            wasteful_rate = len(wasteful) / total_cached if total_cached else 0.0

            self._session_logger.log(
                "read_cache_effectiveness",
                total_cached=total_cached,
                cache_hits=len(cache_hits),
                cache_misses=len(cache_misses),
                justified_rereads=len(justified),
                wasteful_rereads=len(wasteful),
                wasteful_rate=round(wasteful_rate, 3),
                new_reads=len(new_reads),
                hit_rate=round(hit_rate, 3),
                missed_files=sorted(cache_misses)[:20],
                new_files=sorted(new_reads)[:20],
            )
            self.logger.debug(
                f"Read cache effectiveness: {len(cache_hits)}/{total_cached} hits "
                f"({hit_rate:.0%}), {len(cache_misses)} re-reads "
                f"({len(justified)} justified, {len(wasteful)} wasteful), "
                f"{len(new_reads)} new"
            )
        except Exception as e:
            self.logger.debug(f"Cache effectiveness measurement failed (non-fatal): {e}")

    def _update_repo_cache(self, cache_dir: Path, github_repo: str, entries: dict) -> None:
        """Merge entries into repo-scoped cache for cross-attempt persistence.

        Accumulates file-read knowledge across independent task attempts on the
        same repo. New tasks can seed from this when no task-specific cache exists.
        Non-fatal — exceptions are logged and swallowed.
        """
        try:
            from ..utils.atomic_io import atomic_write_text

            slug = _repo_cache_slug(github_repo)
            repo_cache_file = cache_dir / f"_repo-{slug}.json"

            existing: dict = {}
            if repo_cache_file.exists():
                try:
                    existing = json.loads(repo_cache_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = {}

            repo_entries = existing.get("entries", {})

            # Merge: newer wins by read_at timestamp
            for path, entry in entries.items():
                if path not in repo_entries:
                    repo_entries[path] = entry
                else:
                    existing_at = repo_entries[path].get("read_at", "")
                    new_at = entry.get("read_at", "")
                    if new_at > existing_at:
                        repo_entries[path] = entry

            # Evict oldest entries if over limit
            if len(repo_entries) > _MAX_REPO_CACHE_ENTRIES:
                sorted_paths = sorted(
                    repo_entries.keys(),
                    key=lambda p: repo_entries[p].get("read_at", ""),
                )
                for path in sorted_paths[:len(repo_entries) - _MAX_REPO_CACHE_ENTRIES]:
                    del repo_entries[path]

            repo_data = {"github_repo": github_repo, "entries": repo_entries}
            atomic_write_text(repo_cache_file, json.dumps(repo_data))
            self.logger.debug(f"Repo cache: {len(repo_entries)} entries for {github_repo}")
        except Exception as e:
            self.logger.warning(f"Failed to update repo cache for {github_repo}: {e}")

    def _save_pre_scan_findings(self, task: Task, response) -> None:
        """Persist QA pre-scan results so downstream agents can load them.

        Writes to .agent-communication/pre-scans/{root_task_id}.json.
        Non-fatal — workflow continues even if this fails.
        """
        try:
            from ..utils.atomic_io import atomic_write_text

            root_task_id = task.root_id
            pre_scans_dir = self.workspace / ".agent-communication" / "pre-scans"
            pre_scans_dir.mkdir(parents=True, exist_ok=True)

            content = getattr(response, "content", "") or ""
            structured = self._extract_structured_findings_from_content(content)

            findings_data = {
                "root_task_id": root_task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_summary": content[:4000],
                "structured_findings": structured,
            }

            findings_file = pre_scans_dir / f"{root_task_id}.json"
            atomic_write_text(findings_file, json.dumps(findings_data, indent=2))
            self.logger.info(f"Saved pre-scan findings to {findings_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save pre-scan findings for {task.id}: {e}")

    @staticmethod
    def _extract_structured_findings_from_content(content: str) -> dict:
        """Parse structured findings JSON from LLM response content.

        Looks for ```json ... ``` blocks and parses the last one as findings.
        Returns {"findings": [...], "summary": "..."} or empty dict.
        """
        if not content:
            return {}

        matches = _JSON_FENCE_PATTERN.findall(content)
        if not matches:
            return {}

        # Parse the last JSON block (most likely the findings output)
        try:
            parsed = json.loads(matches[-1])
            if isinstance(parsed, list):
                return {"findings": parsed, "summary": ""}
            if isinstance(parsed, dict):
                if "findings" not in parsed:
                    parsed["findings"] = []
                return parsed
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _extract_plan_from_response(content: str) -> Optional[PlanDocument]:
        """Parse PlanDocument JSON from architect's planning response.

        Looks for ```json blocks containing PlanDocument fields (objectives,
        approach, success_criteria). Returns the first valid match, or None.
        """
        if not content:
            return None

        matches = _JSON_FENCE_PATTERN.findall(content)
        if not matches:
            return None

        for raw in matches:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            # Unwrap {"plan": {...}} wrapper if present
            if "plan" in parsed and isinstance(parsed["plan"], dict):
                parsed = parsed["plan"]
            # Discriminate: must have the 3 required PlanDocument fields
            if not all(k in parsed for k in ("objectives", "approach", "success_criteria")):
                continue
            try:
                return PlanDocument.model_validate(parsed)
            except Exception:
                continue
        return None

    def _get_repo_slug(self, task: Task) -> Optional[str]:
        """Extract repo slug from task context."""
        return task.context.get("github_repo")

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

        # Store successful recovery pattern so future replans can reference it
        if task.replan_history and task.status == TaskStatus.COMPLETED:
            self._error_recovery.store_replan_outcome(task, repo_slug)

        # Store recurring QA finding patterns via feedback bus
        structured_findings = task.context.get("structured_findings")
        if structured_findings and isinstance(structured_findings, dict):
            try:
                qa_count = self._feedback_bus.store_qa_pattern(
                    repo_slug=repo_slug,
                    agent_type=self.config.base_id,
                    task_id=task.id,
                    structured_findings=structured_findings,
                )
                if qa_count > 0:
                    self._session_logger.log(
                        "feedback_bus_qa_pattern",
                        repo=repo_slug,
                        memories_stored=qa_count,
                    )
            except Exception as e:
                self.logger.debug(f"Feedback bus QA pattern store failed (non-fatal): {e}")

    # -- Tool Pattern Analysis --

    def _analyze_tool_patterns(self, task: Task) -> Optional[int]:
        """Run post-task analysis on session log to detect inefficient tool usage.

        Returns total tool call count (or None if analysis was skipped).
        """
        if not self._session_logging_enabled:
            return None

        session_path = self._session_logs_dir / "sessions" / f"{task.id}.jsonl"
        if not session_path.exists():
            return None

        tool_call_count = None
        tool_calls = None
        try:
            from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer, compute_tool_usage_stats

            analyzer = ToolPatternAnalyzer()

            # Reuse stats from _compute_tool_stats_for_chain if available
            cached = task.context.pop("_tool_stats_cache", None)
            if cached:
                tool_call_count = cached.get("total_calls")
                self._session_logger.log("tool_usage_stats", agent_id=self.config.id, **cached)
            else:
                tool_calls = analyzer.extract_tool_calls(session_path)
                if tool_calls:
                    stats = compute_tool_usage_stats(tool_calls)
                    tool_call_count = stats.total_calls
                    self._session_logger.log(
                        "tool_usage_stats",
                        agent_id=self.config.id,
                        total_calls=stats.total_calls,
                        tool_distribution=stats.tool_distribution,
                        duplicate_reads=stats.duplicate_reads,
                        read_before_write_ratio=stats.read_before_write_ratio,
                        edit_write_count=stats.edit_write_count,
                        exploration_count=stats.exploration_count,
                        edit_density=stats.edit_density,
                        files_read=stats.files_read,
                        files_written=stats.files_written,
                    )

            # Language mismatch: flag searches for extensions belonging to a different language
            repo_slug = self._get_repo_slug(task)
            if repo_slug and self._code_index_query:
                try:
                    from ..memory.tool_pattern_analyzer import detect_language_mismatches
                    from dataclasses import asdict
                    project_language = self._code_index_query.get_project_language(repo_slug)
                    if project_language:
                        if tool_calls is None:
                            tool_calls = analyzer.extract_tool_calls(session_path)
                        mismatches = detect_language_mismatches(tool_calls, project_language)
                        if mismatches:
                            self._session_logger.log(
                                "language_mismatch",
                                agent_id=self.config.id,
                                project_language=project_language,
                                repo=repo_slug,
                                mismatch_count=len(mismatches),
                                mismatches=[asdict(m) for m in mismatches],
                            )
                except Exception as e:
                    self.logger.debug(f"Language mismatch detection failed (non-fatal): {e}")

            # Qualitative anti-pattern detection (gated by tool_tips config)
            if self._tool_tips_enabled:
                if not repo_slug:
                    repo_slug = self._get_repo_slug(task)
                if repo_slug:
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

        return tool_call_count

    async def _self_evaluate(self, task: Task, response, *, test_passed=None, working_dir=None) -> bool:
        """Review agent's own output against acceptance criteria.

        Delegated to ErrorRecoveryManager.
        """
        return await self._error_recovery.self_evaluate(
            task, response, test_passed=test_passed, working_dir=working_dir
        )

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

    def _push_and_create_pr_if_needed(self, task: Task) -> None:
        """Push branch and create PR if the agent produced unpushed commits.

        Runs after the LLM finishes but before the task is marked completed,
        so the PR URL is available in task.context for downstream chain steps.
        Only acts when working in a worktree with actual unpushed commits.

        Intermediate workflow steps push their branch but skip PR creation —
        the terminal step (or pr_creator) handles that.
        """
        from ..utils.subprocess_utils import run_git_command, run_command, SubprocessError

        # Already has a PR (created by the LLM via MCP)
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
        pr_title = strip_chain_prefixes(task.title)[:70]

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
        self.heartbeat_file.write_text(str(int(time.time())))

    async def _heartbeat_loop(self) -> None:
        """Background loop that writes heartbeats independent of main loop progress.

        Decouples heartbeat freshness from LLM call duration so the watchdog
        can use a tight timeout (90s) without false-positive kills.
        """
        while self._running:
            try:
                self._write_heartbeat()
            except OSError:
                pass  # Transient FS error; next iteration will retry
            await asyncio.sleep(self._heartbeat_interval)

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
