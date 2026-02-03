"""Configuration loading and validation."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """LLM configuration."""
    mode: str = "litellm"  # "litellm" or "claude_cli"

    # LiteLLM settings
    litellm_api_key: Optional[str] = None
    litellm_cheap_model: str = "claude-3-5-haiku-20241022"
    litellm_default_model: str = "claude-sonnet-4-20250514"
    litellm_premium_model: str = "claude-opus-4-20250514"

    # Claude CLI settings
    claude_cli_executable: str = "claude"
    claude_cli_max_turns: int = 999
    claude_cli_cheap_model: str = "haiku"
    claude_cli_default_model: str = "sonnet"
    claude_cli_premium_model: str = "opus"

    # MCP settings
    mcp_config_path: Optional[str] = None
    use_mcp: bool = False


class TaskConfig(BaseModel):
    """Task processing configuration."""
    poll_interval: int = 30
    timeout: int = 1800
    max_retries: int = 5
    backoff_initial: int = 30
    backoff_max: int = 240
    backoff_multiplier: int = 2


class SafeguardsConfig(BaseModel):
    """Safeguards configuration."""
    max_queue_size: int = 100
    max_escalations: int = 50
    max_task_age_days: int = 7
    heartbeat_timeout: int = 90
    watchdog_interval: int = 60


class OptimizationConfig(BaseModel):
    """Optimization configuration for token usage reduction."""
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

    # Rollout settings
    canary_percentage: int = 0  # 0-100
    shadow_mode: bool = False

    # Budget warning threshold (e.g., 1.3 = warn at 130% of budget)
    budget_warning_threshold: float = 1.3

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


class MultiRepoConfig(BaseModel):
    """Multi-repository configuration."""
    workspace_root: Path = Field(default=Path("~/.agent-workspaces"))


class RepositoryConfig(BaseModel):
    """Configuration for a registered repository."""
    github_repo: str  # owner/repo format (e.g., "justworkshr/pto")
    jira_project: str  # JIRA project key (e.g., "PTO")
    display_name: Optional[str] = None  # Display name (defaults to repo name)

    @property
    def name(self) -> str:
        """Display name for UI."""
        return self.display_name or self.github_repo.split("/")[1]


class AgentDefinition(BaseModel):
    """Agent definition from agents.yaml."""
    id: str
    name: str
    queue: str
    prompt: str
    enabled: bool = True

    # JIRA settings
    jira_can_create_tickets: bool = False
    jira_can_update_status: bool = False
    jira_allowed_transitions: List[str] = Field(default_factory=list)

    # Permissions
    can_commit: bool = False
    can_create_pr: bool = False


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


class FrameworkConfig(BaseSettings):
    """Main framework configuration."""
    workspace: Path = Field(default=Path("."))
    communication_dir: str = ".agent-communication"

    llm: LLMConfig = Field(default_factory=LLMConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    safeguards: SafeguardsConfig = Field(default_factory=SafeguardsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    multi_repo: MultiRepoConfig = Field(default_factory=MultiRepoConfig)
    repositories: List[RepositoryConfig] = Field(default_factory=list)

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"
        extra = "allow"


def load_config(config_path: Path = Path("agent-framework.yaml")) -> FrameworkConfig:
    """Load framework configuration from YAML file."""
    if not config_path.exists():
        # Return default config
        return FrameworkConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Expand environment variables
    data = _expand_env_vars(data)

    config = FrameworkConfig(**data)

    # Validate MCP requirements
    if config.llm.use_mcp and config.llm.mode != "claude_cli":
        raise ValueError(
            "MCP integration requires Claude CLI mode. "
            "Set llm.mode to 'claude_cli' or disable MCPs with use_mcp: false"
        )

    return config


def load_agents(agents_path: Path = Path("config/agents.yaml")) -> List[AgentDefinition]:
    """Load agent definitions from YAML file."""
    if not agents_path.exists():
        raise FileNotFoundError(f"Agents config not found: {agents_path}")

    with open(agents_path) as f:
        data = yaml.safe_load(f) or {}

    agents_data = data.get("agents", [])
    return [AgentDefinition(**agent) for agent in agents_data]


def load_jira_config(jira_path: Path = Path("config/jira.yaml")) -> Optional[JIRAConfig]:
    """Load JIRA configuration from YAML file."""
    if not jira_path.exists():
        return None

    with open(jira_path) as f:
        data = yaml.safe_load(f) or {}

    # Expand environment variables
    data = _expand_env_vars(data)

    jira_data = data.get("jira", {})
    return JIRAConfig(**jira_data)


def load_github_config(github_path: Path = Path("config/github.yaml")) -> Optional[GitHubConfig]:
    """Load GitHub configuration from YAML file."""
    if not github_path.exists():
        return None

    with open(github_path) as f:
        data = yaml.safe_load(f) or {}

    # Expand environment variables
    data = _expand_env_vars(data)

    github_data = data.get("github", {})
    return GitHubConfig(**github_data)


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in config data."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.environ.get(env_var, data)
    return data
