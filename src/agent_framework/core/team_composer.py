"""Dynamic team composition for Claude Agent Teams.

Selects teammates based on workflow complexity. The calling agent
is the lead — teammates returned here become --agents subagents.
"""

import logging
from typing import Dict, List, Optional

from .config import AgentDefinition

logger = logging.getLogger(__name__)

# Workflow → teammate roles (lead is the caller, not listed here).
# Keep in sync with the workflows section in config/agent-framework.yaml.
WORKFLOW_TEAMMATES: Dict[str, List[str]] = {
    "simple": [],
    "standard": ["qa"],
    "full": ["engineer", "qa"],
    "quality-focused": ["engineer", "qa", "static-analysis"],
}

# Minimum workflow rank for ordering comparisons
WORKFLOW_RANK = {
    "simple": 0,
    "standard": 1,
    "full": 2,
    "quality-focused": 3,
}


def compose_team(
    task_context: dict,
    workflow: str,
    agents_config: List[AgentDefinition],
    min_workflow: str = "standard",
    default_model: str = "sonnet",
    caller_agent_id: Optional[str] = None,
) -> Optional[dict]:
    """Pick teammates based on workflow complexity.

    Returns None for single-agent execution, or a dict suitable
    for the --agents CLI flag when team mode applies.

    The calling agent IS the lead — teammates returned here are
    the subagents that assist within the same session.

    Args:
        task_context: Task context dict, checked for team_override
        workflow: Workflow name (simple/standard/full/quality-focused)
        agents_config: Agent definitions from agents.yaml
        min_workflow: Minimum workflow complexity to trigger teams
        default_model: Model name for teammates (from framework config)
        caller_agent_id: ID of the lead agent, excluded from teammates
    """
    # Task-level override: force or skip team mode
    override = task_context.get("team_override")
    if override is not None:
        if not override:
            logger.debug("Team mode skipped via task team_override=False")
            return None
        # override=True means force team mode even below min_workflow

    # Check workflow meets minimum threshold (unless overridden)
    if override is None:
        workflow_rank = WORKFLOW_RANK.get(workflow, 0)
        min_rank = WORKFLOW_RANK.get(min_workflow, 1)
        if workflow_rank < min_rank:
            return None

    teammate_roles = WORKFLOW_TEAMMATES.get(workflow, [])
    if not teammate_roles:
        return None

    # Build lookup of agent configs by ID
    config_by_id = {a.id: a for a in agents_config}

    agents_dict = {}
    for role_id in teammate_roles:
        # Don't add the lead agent as its own teammate
        if role_id == caller_agent_id:
            continue

        agent_def = config_by_id.get(role_id)
        if not agent_def:
            logger.warning(f"Teammate '{role_id}' not found in agents config, skipping")
            continue

        agents_dict[role_id] = {
            "model": default_model,
            "description": f"{agent_def.name} - handles {role_id} responsibilities",
            "prompt": agent_def.prompt,
        }

    if not agents_dict:
        return None

    return agents_dict
