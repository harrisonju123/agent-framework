"""Workflow DAG executor for task routing and orchestration."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.task import Task
    from ..core.routing import RoutingSignal
    from ..queue.file_queue import FileQueue

from .dag import WorkflowDAG, WorkflowEdge, EdgeConditionType
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


# Mapping from agent names to task types (same as legacy CHAIN_TASK_TYPES)
AGENT_TASK_TYPES = {
    "architect": TaskType.PLANNING,
    "engineer": TaskType.IMPLEMENTATION,
    "qa": TaskType.QA_VERIFICATION,
}


class WorkflowExecutor:
    """Executes workflow DAGs and routes tasks between agents."""

    def __init__(self, queue: "FileQueue", queue_dir: Path):
        """Initialize workflow executor.

        Args:
            queue: FileQueue instance for pushing tasks
            queue_dir: Path to queue directory (for duplicate checking)
        """
        self.queue = queue
        self.queue_dir = queue_dir
        self.logger = logger

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

        Args:
            workflow: The workflow DAG to execute
            task: The completed task
            response: The LLM response
            current_agent_id: ID of the agent that just completed the task
            routing_signal: Optional routing signal from the agent
            context: Additional context for condition evaluation

        Returns:
            True if task was routed, False if workflow is complete
        """
        # Find current step in workflow
        current_step = self._find_step_for_agent(workflow, current_agent_id)
        if not current_step:
            self.logger.warning(
                f"Agent {current_agent_id} not found in workflow {workflow.name}"
            )
            return False

        # Check for PR creation (legacy behavior - terminates workflow)
        if self._has_pr_created(task, response):
            self.logger.info(f"PR created for task {task.id}, workflow complete")
            return False

        # Handle routing signal for WORKFLOW_COMPLETE
        if routing_signal and routing_signal.target_agent == WORKFLOW_COMPLETE:
            self.logger.info(
                f"Workflow marked complete by routing signal: {routing_signal.reason}"
            )
            return False

        # Handle routing signal to specific agent (even if not next in default path)
        if routing_signal and routing_signal.target_agent != WORKFLOW_COMPLETE:
            # Check if routing signal targets a valid agent in the workflow
            target_step = self._find_step_for_agent(workflow, routing_signal.target_agent)
            if target_step and target_step.agent != current_agent_id:
                # Route to the signal target
                self._route_to_step(task, target_step, workflow, routing_signal)
                return True

        # Get possible next steps from default path
        next_edges = workflow.get_next_steps(current_step.id)
        if not next_edges:
            self.logger.info(f"Terminal step reached for task {task.id}")
            return False

        # Evaluate conditions and find first matching edge
        matched_edge = self._evaluate_edges(
            next_edges, task, response, routing_signal, context
        )

        if not matched_edge:
            self.logger.warning(
                f"No matching edge found for task {task.id} at step {current_step.id}"
            )
            return False

        # Route to target step
        target_step = workflow.steps.get(matched_edge.target)
        if not target_step:
            self.logger.error(
                f"Invalid edge target: {matched_edge.target} not found in workflow"
            )
            return False

        # Queue task to next agent
        self._route_to_step(task, target_step, workflow, routing_signal)
        return True

    def _find_step_for_agent(self, workflow: WorkflowDAG, agent_id: str) -> Optional[Any]:
        """Find the workflow step assigned to a specific agent.

        Handles agent replicas (e.g., engineer-2 -> engineer).
        """
        # Strip replica suffix (-N) from agent ID
        base_agent_id = agent_id.rsplit("-", 1)[0] if "-" in agent_id and agent_id.split("-")[-1].isdigit() else agent_id

        for step in workflow.steps.values():
            if step.agent == base_agent_id:
                return step
        return None

    def _has_pr_created(self, task: "Task", response: Any) -> bool:
        """Check if a PR was created (legacy termination condition)."""
        if task.context and "pr_url" in task.context:
            return True

        if hasattr(response, "content") and response.content:
            content = str(response.content)
            return "github.com/" in content and "/pull/" in content

        return False

    def _evaluate_edges(
        self,
        edges: List[WorkflowEdge],
        task: "Task",
        response: Any,
        routing_signal: Optional["RoutingSignal"],
        context: Optional[Dict[str, Any]],
    ) -> Optional[WorkflowEdge]:
        """Evaluate edge conditions and return first matching edge.

        Edges are evaluated in priority order (highest first).
        """
        # Special handling for routing signals with ALWAYS condition
        # This allows routing signals to override default paths
        if routing_signal and routing_signal.target_agent != WORKFLOW_COMPLETE:
            for edge in edges:
                target_step_agent = self._get_edge_target_agent(edge)
                if target_step_agent == routing_signal.target_agent:
                    # Check if this edge's condition passes
                    if ConditionRegistry.evaluate(
                        edge.condition, task, response, routing_signal, context
                    ):
                        self.logger.info(
                            f"Routing signal matched edge to {target_step_agent}: {routing_signal.reason}"
                        )
                        return edge

        # Evaluate edges in priority order
        for edge in edges:
            if ConditionRegistry.evaluate(
                edge.condition, task, response, routing_signal, context
            ):
                self.logger.debug(
                    f"Edge condition {edge.condition.type} matched for target {edge.target}"
                )
                return edge

        return None

    def _get_edge_target_agent(self, edge: WorkflowEdge) -> Optional[str]:
        """Get the agent assigned to an edge's target step."""
        # This needs to be set by the caller or stored in edge metadata
        # For now, we'll use the target step ID as agent name
        return edge.target

    def _route_to_step(
        self,
        task: "Task",
        target_step: Any,
        workflow: WorkflowDAG,
        routing_signal: Optional["RoutingSignal"],
    ):
        """Route task to the target workflow step."""
        next_agent = target_step.agent

        # Check if task already queued (prevent duplicates)
        if self._is_chain_task_already_queued(next_agent, task.id):
            self.logger.debug(f"Chain task for {next_agent} already queued from {task.id}")
            return

        # Build chain task
        chain_task = self._build_chain_task(task, target_step)

        # Push to queue
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

    def _build_chain_task(self, task: "Task", target_step: Any) -> "Task":
        """Create a continuation task for the target step."""
        from ..core.task import Task

        next_agent = target_step.agent
        chain_id = f"chain-{task.id[:12]}-{next_agent}"

        # Determine task type
        if target_step.task_type_override:
            task_type = TaskType[target_step.task_type_override.upper()]
        else:
            task_type = AGENT_TASK_TYPES.get(next_agent, task.type)

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=task.assigned_to,  # Current agent
            assigned_to=next_agent,
            created_at=datetime.utcnow(),
            title=f"[chain] {task.title}",
            description=task.description,
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": task.assigned_to,
                "chain_step": True,
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
