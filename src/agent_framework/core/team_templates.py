"""Team template loading and prompt building for Claude Agent Teams."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TeamTemplate:
    """Pre-configured team structure for interactive sessions."""

    name: str
    team_name_prefix: str
    lead_role: str
    spawn_prompt: str
    plan_approval: bool = True
    delegate_mode: bool = False


def load_team_templates(config_dir: Path) -> dict[str, TeamTemplate]:
    """Load team templates from config/team-templates.yaml.

    Args:
        config_dir: Path to the config directory

    Returns:
        Dict mapping template name to TeamTemplate
    """
    templates_path = config_dir / "team-templates.yaml"
    if not templates_path.exists():
        logger.warning(f"Team templates not found at {templates_path}")
        return {}

    with open(templates_path) as f:
        data = yaml.safe_load(f) or {}

    templates = {}
    for name, config in data.get("templates", {}).items():
        templates[name] = TeamTemplate(
            name=name,
            team_name_prefix=config.get("team_name_prefix", name),
            lead_role=config.get("lead_role", "Lead"),
            spawn_prompt=config.get("spawn_prompt", ""),
            plan_approval=config.get("plan_approval", True),
            delegate_mode=config.get("delegate_mode", False),
        )

    return templates


def build_spawn_prompt(
    template: TeamTemplate,
    task_context: Optional[str] = None,
    repo_info: Optional[str] = None,
    team_context_doc: Optional[str] = None,
) -> str:
    """Build the full spawn prompt by combining template with runtime context.

    Args:
        template: The team template to use
        task_context: Optional context from a failed task or escalation
        repo_info: Optional target repository information
        team_context_doc: Content of config/docs/team_context.md

    Returns:
        Complete spawn prompt string for the team session
    """
    parts = [template.spawn_prompt.strip()]

    if repo_info:
        parts.append(f"\n## Target Repository\n{repo_info}")

    if task_context:
        parts.append(f"\n## Task Context\n{task_context}")

    if team_context_doc:
        parts.append(f"\n## Pipeline Integration\n{team_context_doc}")

    return "\n".join(parts)
