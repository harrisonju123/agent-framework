"""Dynamic team composition for Claude Agent Teams.

Selects teammates based on workflow. The calling agent
is the lead — teammates returned here become --agents subagents.
"""

import logging
from typing import Dict, List, Optional

from .config import AgentDefinition

logger = logging.getLogger(__name__)

# Workflow → teammate roles (lead is the caller, not listed here).
# Keep in sync with the workflows section in config/agent-framework.yaml.
WORKFLOW_TEAMMATES: Dict[str, List[str]] = {
    "default": ["engineer", "qa"],
}


def compose_default_team(
    agent_def: AgentDefinition,
    default_model: str = "sonnet",
    specialization_profile=None,
) -> Optional[dict]:
    """Build team from agent's configured teammates (always-on mode).

    Each agent defines its own teammates in agents.yaml. This function
    converts those definitions into the dict format expected by --agents.

    Args:
        agent_def: Agent definition from config
        default_model: Default model for teammates
        specialization_profile: Optional SpecializationProfile for engineer agents.
            When provided, merges specialized teammates into the team.
    """
    if not agent_def.teammates:
        return None

    teammates_dict = dict(agent_def.teammates)

    # Merge specialized teammates when a profile is provided
    if specialization_profile:
        from .engineer_specialization import get_specialized_teammates

        logger.info(f"Applying {specialization_profile.name} teammates for engineer")
        teammates_dict = get_specialized_teammates(teammates_dict, specialization_profile)

    # Convert to --agents format
    agents_dict = {}
    for teammate_id, teammate_data in teammates_dict.items():
        # Handle both TeammateDefinition objects and dict (from specialization)
        if isinstance(teammate_data, dict):
            agents_dict[teammate_id] = {
                "model": default_model,
                "description": teammate_data["description"],
                "prompt": teammate_data["prompt"],
            }
        else:
            agents_dict[teammate_id] = {
                "model": teammate_data.model or default_model,
                "description": teammate_data.description,
                "prompt": teammate_data.prompt,
            }

    return agents_dict


def compose_team(
    task_context: dict,
    workflow: str,
    agents_config: List[AgentDefinition],
    default_model: str = "sonnet",
    caller_agent_id: Optional[str] = None,
) -> Optional[dict]:
    """Pick teammates for the given workflow.

    Returns None for single-agent execution, or a dict suitable
    for the --agents CLI flag when team mode applies.

    The calling agent IS the lead — teammates returned here are
    the subagents that assist within the same session.

    Args:
        task_context: Task context dict, checked for team_override
        workflow: Workflow name ("default" or "analysis")
        agents_config: Agent definitions from agents.yaml
        default_model: Model name for teammates (from framework config)
        caller_agent_id: ID of the lead agent, excluded from teammates
    """
    # Task-level override: force or skip team mode
    override = task_context.get("team_override")
    if override is not None:
        if not override:
            logger.debug("Team mode skipped via task team_override=False")
            return None
        # override=True means force team mode

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
