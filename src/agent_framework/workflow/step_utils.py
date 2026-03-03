"""Shared workflow step utilities.

Pure functions that inspect workflow DAGs without requiring class state.
Extracted to break circular imports between workflow_router, git_operations,
and prompt_builder — all three previously carried identical copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..core.task import Task
    from ..core.config import WorkflowDefinition


def is_at_terminal_workflow_step(
    task: "Task",
    workflows_config: Dict[str, "WorkflowDefinition"],
    agent_base_id: str,
) -> bool:
    """Check if the current agent is at the last step in the workflow DAG.

    Returns True for standalone tasks (no workflow) to preserve backward
    compatibility — standalone agents should always be allowed to create PRs.

    Args:
        task: The task to check.
        workflows_config: Mapping of workflow name to WorkflowDefinition.
        agent_base_id: The base_id of the agent (e.g. "engineer").
    """
    workflow_name = task.context.get("workflow")
    if not workflow_name or not workflows_config or workflow_name not in workflows_config:
        return True

    workflow_def = workflows_config[workflow_name]
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
        if step.agent == agent_base_id:
            return dag.is_terminal_step(step.id)

    return True
