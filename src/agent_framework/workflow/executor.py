"""Workflow DAG executor for task routing and orchestration."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.task import Task
    from ..core.routing import RoutingSignal
    from ..queue.file_queue import FileQueue

from .dag import WorkflowDAG, WorkflowStep, WorkflowEdge, EdgeConditionType
from .conditions import ConditionRegistry
from ..core.task import TaskStatus, TaskType
from ..core.routing import WORKFLOW_COMPLETE

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutionState:
    """Tracks the current state of workflow execution for a task."""
    task_id: str
    workflow_name: str
    current_step: str
    completed_steps: List[str]
    timestamp: str


# Aligned with CHAIN_TASK_TYPES in agent.py
AGENT_TASK_TYPES = {
    "architect": TaskType.REVIEW,
    "engineer": TaskType.IMPLEMENTATION,
    "qa": TaskType.QA_VERIFICATION,
}

_PR_URL_PATTERN = re.compile(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)')


class WorkflowExecutor:
    """Executes workflow DAGs and routes tasks between agents."""

    def __init__(self, queue: "FileQueue", queue_dir: Path, agent_logger=None):
        self.queue = queue
        self.queue_dir = queue_dir
        self.logger = agent_logger or logger

    def execute_step(
        self,
        workflow: WorkflowDAG,
        task: "Task",
        response: Any,
        current_agent_id: str,
        routing_signal: Optional["RoutingSignal"] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Execute workflow routing for a completed task.

        Evaluates edge conditions and routes the task to the next appropriate agent(s).

        Returns:
            True if task was routed, False if workflow is complete.
        """
        current_step = self._find_current_step(workflow, task, current_agent_id)
        if not current_step:
            self.logger.warning(
                f"Agent {current_agent_id} not found in workflow {workflow.name}"
            )
            return False

        # PR creation terminates the workflow and stores info for downstream review queue
        pr_info = self._extract_pr_info(task, response)
        if pr_info:
            task.context["pr_url"] = pr_info["pr_url"]
            task.context["pr_number"] = pr_info["pr_number"]
            self.logger.info(f"PR created for task {task.id}, workflow complete")
            return False

        # pr_creation_step tasks should not continue the chain
        if task.context.get("pr_creation_step"):
            return False

        if routing_signal and routing_signal.target_agent == WORKFLOW_COMPLETE:
            self.logger.info(
                f"Workflow marked complete by routing signal: {routing_signal.reason}"
            )
            return False

        # Routing signal override to a specific agent
        if routing_signal and routing_signal.target_agent != WORKFLOW_COMPLETE:
            target_step = self._find_step_for_agent(workflow, routing_signal.target_agent)
            if target_step and target_step.agent != current_agent_id:
                self._route_to_step(task, target_step, workflow, current_agent_id, routing_signal)
                return True

        next_edges = workflow.get_next_steps(current_step.id)
        if not next_edges:
            self.logger.info(f"Terminal step reached for task {task.id}")
            return False

        matched_edge = self._evaluate_edges(
            next_edges, task, response, workflow, routing_signal, context
        )

        if not matched_edge:
            self.logger.warning(
                f"No matching edge found for task {task.id} at step {current_step.id}"
            )
            return False

        target_step = workflow.steps.get(matched_edge.target)
        if not target_step:
            self.logger.error(
                f"Invalid edge target: {matched_edge.target} not found in workflow"
            )
            return False

        self._route_to_step(task, target_step, workflow, current_agent_id, routing_signal)
        return True

    def _find_current_step(
        self, workflow: WorkflowDAG, task: "Task", current_agent_id: str
    ) -> Optional[WorkflowStep]:
        """Find the workflow step the current agent is executing.

        Prefers the explicit workflow_step stored in task context (set by
        _build_chain_task) so agents appearing at multiple steps are unambiguous.
        Falls back to agent-name scan for the first task in a chain.
        """
        step_id = task.context.get("workflow_step") if task.context else None
        if step_id and step_id in workflow.steps:
            return workflow.steps[step_id]
        return self._find_step_for_agent(workflow, current_agent_id)

    def _find_step_for_agent(
        self, workflow: WorkflowDAG, agent_id: str
    ) -> Optional[WorkflowStep]:
        """Find the first workflow step assigned to an agent (by base ID)."""
        base_id = (
            agent_id.rsplit("-", 1)[0]
            if "-" in agent_id and agent_id.split("-")[-1].isdigit()
            else agent_id
        )
        for step in workflow.steps.values():
            if step.agent == base_id:
                return step
        return None

    def _extract_pr_info(self, task: "Task", response: Any) -> Optional[Dict[str, Any]]:
        """Extract PR information using regex validation (matches agent._get_pr_info)."""
        if task.context and "pr_url" in task.context:
            match = _PR_URL_PATTERN.search(task.context["pr_url"])
            if match:
                owner, repo, pr_number = match.groups()
                return {
                    "pr_url": task.context["pr_url"],
                    "pr_number": int(pr_number),
                    "owner": owner,
                    "repo": repo,
                    "github_repo": f"{owner}/{repo}",
                }

        if hasattr(response, "content") and response.content:
            match = _PR_URL_PATTERN.search(str(response.content))
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

    def _evaluate_edges(
        self,
        edges: List[WorkflowEdge],
        task: "Task",
        response: Any,
        workflow: WorkflowDAG,
        routing_signal: Optional["RoutingSignal"],
        context: Optional[Dict[str, Any]],
    ) -> Optional[WorkflowEdge]:
        """Evaluate edge conditions and return first matching edge.

        Edges are evaluated in priority order (highest first).
        """
        # Routing signal can prioritise an edge whose target agent matches
        if routing_signal and routing_signal.target_agent != WORKFLOW_COMPLETE:
            for edge in edges:
                target_step = workflow.steps.get(edge.target)
                if target_step and target_step.agent == routing_signal.target_agent:
                    if ConditionRegistry.evaluate(
                        edge.condition, task, response, routing_signal, context
                    ):
                        self.logger.info(
                            f"Routing signal matched edge to {target_step.agent}: "
                            f"{routing_signal.reason}"
                        )
                        return edge

        for edge in edges:
            if ConditionRegistry.evaluate(
                edge.condition, task, response, routing_signal, context
            ):
                self.logger.debug(
                    f"Edge condition {edge.condition.type} matched for target {edge.target}"
                )
                return edge

        return None

    def _route_to_step(
        self,
        task: "Task",
        target_step: WorkflowStep,
        workflow: WorkflowDAG,
        current_agent_id: str,
        routing_signal: Optional["RoutingSignal"],
    ):
        """Route task to the target workflow step."""
        next_agent = target_step.agent

        if self._is_chain_task_already_queued(next_agent, task.id):
            self.logger.debug(f"Chain task for {next_agent} already queued from {task.id}")
            return

        chain_task = self._build_chain_task(task, target_step, current_agent_id)

        try:
            self.queue.push(chain_task, next_agent)
            reason = routing_signal.reason if routing_signal else "workflow DAG"
            self.logger.info(f"🔗 Workflow: queued {next_agent} for task {task.id} ({reason})")
        except Exception as e:
            self.logger.error(f"Failed to queue chain task for {next_agent}: {e}")

    def _is_chain_task_already_queued(self, next_agent: str, source_task_id: str) -> bool:
        """Check if chain task already exists in target queue."""
        chain_id = f"chain-{source_task_id[:12]}-{next_agent}"
        queue_path = self.queue_dir / next_agent / f"{chain_id}.json"
        return queue_path.exists()

    def _build_chain_task(
        self, task: "Task", target_step: WorkflowStep, current_agent_id: str
    ) -> "Task":
        """Create a continuation task for the target step."""
        from ..core.task import Task

        next_agent = target_step.agent
        chain_id = f"chain-{task.id[:12]}-{next_agent}"

        if target_step.task_type_override:
            task_type = TaskType[target_step.task_type_override.upper()]
        else:
            task_type = AGENT_TASK_TYPES.get(next_agent, task.type)

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=current_agent_id,
            assigned_to=next_agent,
            created_at=datetime.utcnow(),
            title=f"[chain] {task.title}",
            description=task.description,
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": current_agent_id,
                "chain_step": True,
                "workflow_step": target_step.id,
            },
        )

    def get_execution_state(self, task: "Task", current_step: str) -> WorkflowExecutionState:
        """Get current execution state for a task."""
        workflow_name = task.context.get("workflow", "unknown")
        completed = task.context.get("completed_steps", [])

        return WorkflowExecutionState(
            task_id=task.id,
            workflow_name=workflow_name,
            current_step=current_step,
            completed_steps=completed,
            timestamp=datetime.utcnow().isoformat(),
        )
