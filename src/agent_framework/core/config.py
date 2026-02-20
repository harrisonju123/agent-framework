"""Configuration loading and validation."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """LLM configuration."""
    mode: Literal["claude_cli", "litellm"] = "claude_cli"

    # Use Claude Max OAuth account instead of API key billing
    use_max_account: bool = False

    # Proxy passthrough: route Claude CLI API calls through LiteLLM proxy
    proxy_url: Optional[str] = None
    proxy_auth_token: Optional[str] = None

    @field_validator('proxy_url')
    @classmethod
    def validate_proxy_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError(
                f"proxy_url must start with http:// or https://, got '{v}'"
            )
        return v

    # LiteLLM direct settings
    litellm_api_key: Optional[str] = None
    litellm_api_base: Optional[str] = None
    litellm_cheap_model: str = "claude-haiku-4-5-20251001"
    litellm_default_model: str = "claude-sonnet-4-5-20250929"
    litellm_premium_model: str = "claude-sonnet-4-5-20250929"

    # Claude CLI settings
    claude_cli_executable: str = "claude"
    claude_cli_max_turns: int = 999
    claude_cli_timeout: int = 3600  # Default fallback timeout (1 hour)
    claude_cli_cheap_model: str = "haiku"
    claude_cli_default_model: str = "sonnet"
    claude_cli_premium_model: str = "opus"  # YAML overrides with full model ID

    # Task-type-specific timeouts (used by ModelSelector.select_timeout)
    claude_cli_timeout_large: int = 3600   # 1 hour - IMPLEMENTATION, ARCHITECTURE, ANALYSIS, PLANNING, REVIEW
    claude_cli_timeout_bounded: int = 1800  # 30 min - TESTING, VERIFICATION, FIX, BUGFIX, PR_REQUEST
    claude_cli_timeout_simple: int = 900    # 15 min - DOCUMENTATION, COORDINATION, STATUS_REPORT

    # MCP settings
    mcp_config_path: Optional[str] = None
    use_mcp: bool = False

    def get_proxy_env(self) -> Dict[str, str]:
        """Build env vars dict for proxy passthrough to Claude CLI subprocesses."""
        env: Dict[str, str] = {}
        if self.proxy_url:
            env["ANTHROPIC_BASE_URL"] = self.proxy_url
        if self.proxy_auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self.proxy_auth_token
        return env


class TaskConfig(BaseModel):
    """Task processing configuration."""
    poll_interval: int = 30
    timeout: int = 1800
    max_retries: int = 5
    backoff_initial: int = 30
    backoff_max: int = 240
    backoff_multiplier: int = 2

    # Task validation settings
    validate_tasks: bool = True
    validation_mode: str = "warn"  # "warn" logs warning, "reject" fails the task


class SafeguardsConfig(BaseModel):
    """Safeguards configuration."""
    max_queue_size: int = 100
    max_escalations: int = 50
    max_task_age_days: int = 7
    heartbeat_interval: int = 15  # Background heartbeat write interval (seconds)
    heartbeat_timeout: int = 90  # Agent considered dead after N seconds without heartbeat
    watchdog_interval: int = 60
    max_consecutive_tool_calls: int = 15  # Circuit breaker: kill subprocess after N consecutive Bash calls
    max_consecutive_diagnostic_calls: int = 5  # Diagnostic circuit breaker: kill after N consecutive env-probing commands


class ContextWindowConfig(BaseModel):
    """Context window management settings."""
    output_reserve: int = 4096
    summary_threshold: int = 10
    min_message_retention: int = 3


class IntelligentRoutingConfig(BaseModel):
    """Configuration for multi-signal intelligent model routing."""
    enabled: bool = False
    complexity_weight: float = 0.3
    historical_weight: float = 0.25
    specialization_weight: float = 0.2
    budget_weight: float = 0.15
    retry_weight: float = 0.1
    min_historical_samples: int = 5  # Minimum outcomes before historical signal is trusted

    @field_validator('min_historical_samples')
    @classmethod
    def validate_min_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_historical_samples must be >= 1, got {v}")
        return v


class OptimizationConfig(BaseModel):
    """Optimization configuration for token usage reduction."""
    # Intelligent model routing
    intelligent_routing: IntelligentRoutingConfig = Field(default_factory=IntelligentRoutingConfig)

    # Quick wins (Phase 1)
    enable_minimal_prompts: bool = False
    enable_compact_json: bool = False
    enable_context_deduplication: bool = False

    # Structural changes (Phase 2)
    enable_token_tracking: bool = False
    enable_token_budget_warnings: bool = False

    # Advanced features (Phase 3)
    enable_result_summarization: bool = False
    enable_error_truncation: bool = False

    # Tool pattern tips (Phase 4) — feed session analysis back into prompts
    enable_tool_pattern_tips: bool = False
    tool_tips_max_chars: int = 1500
    tool_tips_max_count: int = 5

    # Rollout settings
    canary_percentage: int = 0  # 0-100
    shadow_mode: bool = False

    # Budget warning threshold (e.g., 1.3 = warn at 130% of budget)
    budget_warning_threshold: float = 1.3

    # Exploration alert: activity event when a session exceeds N tool calls
    exploration_alert_threshold: int = 50

    # Per-task-tree USD budget ceilings by estimated effort (t-shirt size)
    enable_effort_budget_ceilings: bool = False
    effort_budget_ceilings: Dict[str, float] = Field(default_factory=lambda: {
        "XS": 3.0,
        "S": 5.0,
        "M": 15.0,
        "L": 30.0,
        "XL": 50.0,
    })

    # Hard cap on all tasks regardless of t-shirt size — safety net when effort estimation is imprecise
    absolute_budget_ceiling_usd: Optional[float] = None

    context_window: ContextWindowConfig = Field(default_factory=ContextWindowConfig)

    # Token budgets by task type (configurable)
    token_budgets: Dict[str, int] = Field(default_factory=lambda: {
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
        "bug_fix": 30000,
        "verification": 20000,
        "status_report": 10000,
        "enhancement": 40000,
    })

    @field_validator('token_budgets')
    @classmethod
    def validate_budgets(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate token budgets are reasonable."""
        for task_type, budget in v.items():
            if budget < 1000:
                raise ValueError(
                    f"Token budget for '{task_type}' must be at least 1000, got {budget}"
                )
            if budget > 1_000_000:
                logger.warning(
                    f"Token budget for '{task_type}' is very high: {budget}. "
                    "This may indicate a configuration error."
                )
        return v

    @field_validator('effort_budget_ceilings')
    @classmethod
    def validate_effort_ceilings(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate effort budget ceilings are positive."""
        for size, ceiling in v.items():
            if ceiling <= 0:
                raise ValueError(
                    f"Effort budget ceiling for '{size}' must be positive, got {ceiling}"
                )
        return v

    @field_validator('absolute_budget_ceiling_usd')
    @classmethod
    def validate_absolute_ceiling(cls, v: Optional[float]) -> Optional[float]:
        """Reject zero/negative values — None means no absolute cap."""
        if v is not None and v <= 0:
            raise ValueError(
                f"absolute_budget_ceiling_usd must be positive, got {v}"
            )
        return v

    @field_validator('canary_percentage')
    @classmethod
    def validate_canary(cls, v: int) -> int:
        """Validate canary percentage is in valid range."""
        if not 0 <= v <= 100:
            raise ValueError(f"canary_percentage must be 0-100, got {v}")
        return v

    @field_validator('budget_warning_threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate budget warning threshold is reasonable."""
        if v < 1.0:
            raise ValueError(
                f"budget_warning_threshold must be >= 1.0 (100%), got {v}"
            )
        if v > 5.0:
            logger.warning(
                f"budget_warning_threshold is very high ({v}). "
                "You may not get warnings for over-budget tasks."
            )
        return v

    @model_validator(mode='after')
    def validate_config(self) -> 'OptimizationConfig':
        """Validate optimization config after loading."""
        if self.shadow_mode and self.canary_percentage > 0:
            logger.warning(
                "shadow_mode=true will override canary_percentage setting. "
                "Shadow mode always uses legacy prompts."
            )

        return self


class WorktreeConfig(BaseModel):
    """Git worktree configuration for isolated agent workspaces."""
    enabled: bool = False
    root: Path = Field(default=Path("~/.agent-workspaces/worktrees"))
    cleanup_on_complete: bool = False
    cleanup_on_failure: bool = False  # Keep failed worktrees for debugging
    max_age_hours: int = 24
    max_worktrees: int = 200

    def to_manager_config(self):
        """Convert to WorktreeManager's WorktreeConfig dataclass.

        This bridges the Pydantic config model with the manager's dataclass.
        """
        from ..workspace.worktree_manager import WorktreeConfig as ManagerWorktreeConfig
        return ManagerWorktreeConfig(
            enabled=self.enabled,
            root=self.root,
            cleanup_on_complete=self.cleanup_on_complete,
            cleanup_on_failure=self.cleanup_on_failure,
            max_age_hours=self.max_age_hours,
            max_worktrees=self.max_worktrees,
        )


class TeamModeConfig(BaseModel):
    """Team mode configuration for Claude Agent Teams."""
    enabled: bool = True


class WorkflowStepDefinition(BaseModel):
    """Defines a single step in a DAG workflow."""
    agent: str
    next: Optional[List[Dict[str, Any]]] = None  # List of edge definitions
    task_type: Optional[str] = None  # Override default task type
    instructions: Optional[str] = None  # Per-step instructions injected into prompts


class WorkflowDefinition(BaseModel):
    """Defines a workflow's agent chain and behaviour.

    Supports both legacy linear format (agents list) and new DAG format (steps dict).
    """
    description: str = ""

    # Legacy format (backward compatible)
    agents: Optional[List[str]] = None

    # DAG format
    steps: Optional[Dict[str, WorkflowStepDefinition]] = None
    start_step: Optional[str] = None

    # Common metadata
    pr_creator: Optional[str] = None
    auto_review: bool = True
    require_tests: bool = True
    output: Optional[str] = None

    @model_validator(mode='after')
    def validate_workflow(self) -> 'WorkflowDefinition':
        """Validate that either agents or steps is provided, but not both."""
        if self.agents is None and self.steps is None:
            raise ValueError("Workflow must have either 'agents' (legacy) or 'steps' (DAG)")

        if self.agents is not None and self.steps is not None:
            raise ValueError("Workflow cannot have both 'agents' and 'steps' (choose one format)")

        if self.steps is not None and self.start_step is None:
            raise ValueError("DAG workflow must specify 'start_step'")

        if self.steps is not None and self.start_step not in self.steps:
            raise ValueError(f"start_step '{self.start_step}' not found in steps")

        return self

    @property
    def is_legacy_format(self) -> bool:
        """Check if this is a legacy linear workflow."""
        return self.agents is not None

    @property
    def agent_list(self) -> List[str]:
        """Safe accessor — returns agents list or empty list for DAG workflows."""
        return self.agents or []

    def to_dag(self, name: str):
        """Convert to WorkflowDAG instance (cached after first call).

        For legacy format, creates a linear DAG.
        For DAG format, builds the full DAG from step definitions.
        """
        # Return cached DAG if available
        cached = getattr(self, "_cached_dag", None)
        if cached is not None:
            return cached

        from ..workflow.dag import (
            WorkflowDAG,
            WorkflowStep,
            WorkflowEdge,
            EdgeCondition,
            EdgeConditionType,
        )

        if self.is_legacy_format:
            dag = WorkflowDAG.from_linear_chain(name, self.agents, self.description)
            object.__setattr__(self, "_cached_dag", dag)
            return dag

        # Build DAG from step definitions
        dag_steps = {}
        for step_id, step_def in self.steps.items():
            edges = []
            if step_def.next:
                for edge_def in step_def.next:
                    if isinstance(edge_def, str):
                        edges.append(WorkflowEdge(
                            target=edge_def,
                            condition=EdgeCondition(EdgeConditionType.ALWAYS)
                        ))
                    elif isinstance(edge_def, dict):
                        target = edge_def.get("target")
                        condition_type = edge_def.get("condition", "always")
                        condition_params = edge_def.get("params", {})
                        priority = edge_def.get("priority", 0)

                        if not target:
                            raise ValueError(f"Edge in step '{step_id}' missing 'target'")

                        edges.append(WorkflowEdge(
                            target=target,
                            condition=EdgeCondition(
                                EdgeConditionType(condition_type),
                                condition_params
                            ),
                            priority=priority
                        ))

            dag_steps[step_id] = WorkflowStep(
                id=step_id,
                agent=step_def.agent,
                next=edges,
                task_type_override=step_def.task_type,
                instructions=step_def.instructions,
            )

        dag = WorkflowDAG(
            name=name,
            description=self.description,
            steps=dag_steps,
            start_step=self.start_step,
            metadata={
                "pr_creator": self.pr_creator,
                "auto_review": self.auto_review,
                "require_tests": self.require_tests,
                "output": self.output,
            }
        )
        object.__setattr__(self, "_cached_dag", dag)
        return dag


class PRLifecycleConfig(BaseModel):
    """Configuration for autonomous PR lifecycle management (CI polling, merge)."""
    ci_poll_interval: int = 30  # Seconds between CI status checks
    ci_poll_max_wait: int = 1200  # Max seconds to wait for CI (20 min)
    max_ci_fix_attempts: int = 3  # Default cap on engineer CI fix loops
    auto_approve: bool = True  # QA-approve via gh pr review --approve
    delete_branch_on_merge: bool = True  # Pass --delete-branch to gh pr merge


class MultiRepoConfig(BaseModel):
    """Multi-repository configuration."""
    workspace_root: Path = Field(default=Path("~/.agent-workspaces"))
    worktree: WorktreeConfig = Field(default_factory=WorktreeConfig)


class RepositoryConfig(BaseModel):
    """Configuration for a registered repository."""
    github_repo: str  # owner/repo format (e.g., "justworkshr/pto")
    jira_project: Optional[str] = None  # JIRA project key (e.g., "PTO"), None for local-only
    display_name: Optional[str] = None  # Display name (defaults to repo name)
    auto_merge: bool = False  # Opt-in: autonomously merge PRs after CI passes
    merge_strategy: Literal["squash", "merge", "rebase"] = "squash"
    max_ci_fix_attempts: int = 3  # Per-repo override for CI fix retry limit

    @property
    def name(self) -> str:
        """Display name for UI."""
        return self.display_name or self.github_repo.split("/")[1]


class TeammateDefinition(BaseModel):
    """Teammate definition for always-on Agent Teams."""
    description: str
    prompt: str
    model: Optional[str] = None  # None = use team_mode default model

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("haiku", "sonnet", "opus"):
            raise ValueError(
                f"model must be 'haiku', 'sonnet', or 'opus', got '{v}'"
            )
        return v


class AgentDefinition(BaseModel):
    """Agent definition from agents.yaml."""
    id: str
    name: str
    queue: str
    prompt: str
    enabled: bool = True

    # Always-on teammates for Claude Agent Teams
    teammates: Dict[str, TeammateDefinition] = Field(default_factory=dict)

    # JIRA settings
    jira_can_create_tickets: bool = False
    jira_can_update_status: bool = False
    jira_allowed_transitions: List[str] = Field(default_factory=list)
    jira_on_start: Optional[str] = None
    jira_on_complete: Optional[str] = None

    # Permissions
    can_commit: bool = False
    can_create_pr: bool = False

    # Per-agent toggle for engineer specialization (only applies to engineer agents)
    specialization_enabled: bool = True


class JIRAConfig(BaseModel):
    """JIRA configuration."""
    server: str
    email: Optional[str] = None
    api_token: Optional[str] = None
    project: str
    backlog_filter: str
    transitions: Dict[str, str] = Field(default_factory=dict)
    poll_interval: int = 300
    batch_size: int = 10


class GitHubConfig(BaseModel):
    """GitHub configuration."""
    token: Optional[str] = None
    owner: str
    repo: str
    branch_pattern: str = "feature/{ticket_id}-{slug}"
    pr_title_pattern: str = "[{ticket_id}] {title}"
    labels: List[str] = Field(default_factory=lambda: ["agent-pr"])


class EmbeddingsConfig(BaseModel):
    """Embedding-based semantic search configuration."""
    enabled: bool = False
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 256
    n_results: int = 15


class CodeIndexingConfig(BaseModel):
    """Codebase structural indexing configuration."""
    enabled: bool = True
    max_symbols: int = 500
    max_prompt_chars: int = 4000
    inject_for_agents: List[str] = Field(
        default_factory=lambda: ["architect", "engineer", "qa"]
    )
    exclude_patterns: List[str] = Field(default_factory=lambda: [
        "vendor/", "node_modules/", "third_party/", ".git/",
        "dist/", "build/", "__pycache__/", ".tox/",
        "*.min.js", "*.min.css", "*.generated.*",
        "*.pb.go", "*_generated.go", "*.pb.rb",
    ])
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)


class FrameworkConfig(BaseSettings):
    """Main framework configuration."""
    workspace: Path = Field(default=Path("."))
    communication_dir: str = ".agent-communication"
    context_dir: Path = Field(default=Path(".agent-context"))
    # Subdirectories created automatically:
    # - .agent-context/plans/
    # - .agent-context/summaries/
    # - .agent-context/archives/

    llm: LLMConfig = Field(default_factory=LLMConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    safeguards: SafeguardsConfig = Field(default_factory=SafeguardsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    team_mode: TeamModeConfig = Field(default_factory=TeamModeConfig)
    workflows: Dict[str, WorkflowDefinition] = Field(default_factory=dict)
    multi_repo: MultiRepoConfig = Field(default_factory=MultiRepoConfig)
    repositories: List[RepositoryConfig] = Field(default_factory=list)
    pr_lifecycle: PRLifecycleConfig = Field(default_factory=PRLifecycleConfig)
    indexing: CodeIndexingConfig = Field(default_factory=CodeIndexingConfig)

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"
        extra = "allow"


# Module-level mtime-based config cache: path -> (parsed_config, file_mtime)
_config_cache: Dict[str, tuple] = {}


def _get_cached_or_load(resolved_path: Path, loader):
    """Return cached config if file mtime unchanged, else reload.

    Works for any config loader that takes a Path and returns a parsed object.
    """
    key = str(resolved_path)
    try:
        current_mtime = resolved_path.stat().st_mtime
    except FileNotFoundError:
        _config_cache.pop(key, None)
        return None

    cached = _config_cache.get(key)
    if cached is not None:
        cached_result, cached_mtime = cached
        if cached_mtime == current_mtime:
            return cached_result

    result = loader(resolved_path)
    _config_cache[key] = (result, current_mtime)
    return result


def _load_config_from_file(config_path: Path) -> FrameworkConfig:
    """Internal loader for framework config (no caching)."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    data = _expand_env_vars(data)
    config = FrameworkConfig(**data)

    if config.llm.use_mcp and config.llm.mode != "claude_cli":
        raise ValueError(
            "MCP integration requires Claude CLI mode. "
            "Set llm.mode to 'claude_cli' or disable MCPs with use_mcp: false"
        )

    return config


def load_config(config_path: Path = Path("agent-framework.yaml")) -> FrameworkConfig:
    """Load framework configuration from YAML file.

    Uses mtime-based caching — returns cached config if the file hasn't changed.
    """
    if not config_path.exists():
        logger.warning(
            f"Config file not found: {config_path}. Using default configuration. "
            "To customize settings, create a config file at this path."
        )
        return FrameworkConfig()

    resolved = config_path.resolve()
    result = _get_cached_or_load(resolved, _load_config_from_file)
    return result if result is not None else FrameworkConfig()


def _load_agents_from_file(agents_path: Path) -> List[AgentDefinition]:
    """Internal loader for agent definitions (no caching)."""
    with open(agents_path) as f:
        data = yaml.safe_load(f) or {}
    agents_data = data.get("agents", [])
    return [AgentDefinition(**agent) for agent in agents_data]


def load_agents(agents_path: Path = Path("config/agents.yaml")) -> List[AgentDefinition]:
    """Load agent definitions from YAML file.

    Uses mtime-based caching — returns cached agents if the file hasn't changed.
    """
    if not agents_path.exists():
        raise FileNotFoundError(f"Agents config not found: {agents_path}")

    resolved = agents_path.resolve()
    result = _get_cached_or_load(resolved, _load_agents_from_file)
    if result is None:
        raise FileNotFoundError(f"Agents config not found: {agents_path}")
    return result


def _load_jira_config_from_file(jira_path: Path) -> Optional[JIRAConfig]:
    """Internal loader for JIRA config (no caching)."""
    with open(jira_path) as f:
        data = yaml.safe_load(f) or {}
    data = _expand_env_vars(data)
    jira_data = data.get("jira", {})
    return JIRAConfig(**jira_data)


def load_jira_config(jira_path: Path = Path("config/jira.yaml")) -> Optional[JIRAConfig]:
    """Load JIRA configuration from YAML file.

    Uses mtime-based caching — returns cached config if the file hasn't changed.
    """
    if not jira_path.exists():
        return None
    return _get_cached_or_load(jira_path.resolve(), _load_jira_config_from_file)


def _load_github_config_from_file(github_path: Path) -> Optional[GitHubConfig]:
    """Internal loader for GitHub config (no caching)."""
    with open(github_path) as f:
        data = yaml.safe_load(f) or {}
    data = _expand_env_vars(data)
    github_data = data.get("github", {})
    return GitHubConfig(**github_data)


def load_github_config(github_path: Path = Path("config/github.yaml")) -> Optional[GitHubConfig]:
    """Load GitHub configuration from YAML file.

    Uses mtime-based caching — returns cached config if the file hasn't changed.
    """
    if not github_path.exists():
        return None
    return _get_cached_or_load(github_path.resolve(), _load_github_config_from_file)


# --- Specialization config models and loader ---


class SpecializationTeammateConfig(BaseModel):
    """Teammate definition within a specialization profile."""
    description: str
    prompt: str


class SpecializationProfileConfig(BaseModel):
    """A single specialization profile from YAML."""
    id: str
    name: str
    description: str
    file_patterns: List[str]
    prompt_suffix: str
    tool_guidance: str
    teammates: Dict[str, SpecializationTeammateConfig] = Field(default_factory=dict)


class AutoProfileConfig(BaseModel):
    """Configuration for autonomous LLM-based profile generation."""
    enabled: bool = False  # Opt-in: makes LLM calls from engineer agent
    max_cached_profiles: int = 50
    staleness_days: int = 90
    model: str = "haiku"
    min_match_score: float = 0.4

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ("haiku", "sonnet", "opus"):
            raise ValueError(
                f"model must be 'haiku', 'sonnet', or 'opus', got '{v}'"
            )
        return v


class SpecializationConfig(BaseModel):
    """Top-level specialization configuration."""
    enabled: bool = True
    auto_profile_generation: AutoProfileConfig = Field(default_factory=AutoProfileConfig)
    profiles: List[SpecializationProfileConfig] = Field(default_factory=list)


def _load_specializations_from_file(spec_path: Path) -> SpecializationConfig:
    """Internal loader for specialization config (no caching)."""
    with open(spec_path) as f:
        data = yaml.safe_load(f) or {}
    return SpecializationConfig(**data)


SPECIALIZATIONS_CONFIG_PATH = Path("config/specializations.yaml")


def clear_config_cache() -> None:
    """Clear the module-level config cache. Useful for tests."""
    _config_cache.clear()


def load_specializations(
    spec_path: Path = SPECIALIZATIONS_CONFIG_PATH,
) -> Optional[SpecializationConfig]:
    """Load specialization configuration from YAML file.

    Uses mtime-based caching — returns cached config if the file hasn't changed.
    Returns None when the file doesn't exist (callers fall back to hardcoded defaults).
    """
    if not spec_path.exists():
        return None
    return _get_cached_or_load(spec_path.resolve(), _load_specializations_from_file)


def _expand_env_vars(data: Any, _path: str = "") -> Any:
    """Recursively expand environment variables in config data.

    Args:
        data: Config data to process
        _path: Internal tracking for error messages (e.g., "llm.api_token")
    """
    if isinstance(data, dict):
        return {k: _expand_env_vars(v, f"{_path}.{k}" if _path else k) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item, f"{_path}[{i}]") for i, item in enumerate(data)]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        value = os.environ.get(env_var)
        if value is None:
            logger.warning(
                f"Environment variable '{env_var}' not set (referenced at config path: {_path or 'root'}). "
                f"The literal string '{data}' will be used, which may cause errors."
            )
            return data
        return value
    return data
