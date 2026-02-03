"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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

    return FrameworkConfig(**data)


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
