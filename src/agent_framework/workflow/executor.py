"""Workflow DAG executor for task routing and orchestration."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Hard ceiling on chain depth to prevent runaway loops regardless of routing logic
MAX_CHAIN_DEPTH = 10

# Cap QA→engineer fix cycles. Each increment = one fix round.
# Value of 2: initial review + 2 fix rounds + final review = 6 chain hops max.
MAX_DAG_REVIEW_CYCLES = 2

# Absolute ceiling on total chain hops — survives escalation and re-planning resets
MAX_GLOBAL_CYCLES = 15


@dataclass
class WorkflowExecutionState:
    """Tracks the current state of workflow execution for a task."""
    task_id: str
    workflow_name: str
    current_step: str
    completed_steps: List[str]
    timestamp: str


# Maps agent IDs to their default task types for workflow routing
AGENT_TASK_TYPES = {
    "architect": TaskType.REVIEW,
    "engineer": TaskType.IMPLEMENTATION,
    "qa": TaskType.QA_VERIFICATION,
}

_PR_URL_PATTERN = re.compile(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)')


def _strip_chain_prefixes(title: str) -> str:
    """Remove accumulated [chain]/[pr] prefixes so re-wrapping adds exactly one."""
    while title.startswith(("[chain] ", "[pr] ")):
        title = title[len("[chain] "):] if title.startswith("[chain] ") else title[len("[pr] "):]
    return title


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
            True if task was routed to the next step.
            False if workflow is complete, terminal, or paused at a checkpoint.

        Note:
            Routing signal overrides are evaluated before checkpoints, so a
            routing signal (e.g. QA sending back to engineer) can bypass a
            checkpoint. This is intentional — programmatic re-routing reflects
            an agent decision, not a human-reviewable transition.
        """
        current_step = self._find_current_step(workflow, task, current_agent_id)
        if not current_step:
            self.logger.warning(
                f"Agent {current_agent_id} not found in workflow {workflow.name}"
            )
            return False

        # PR detection only terminates at terminal steps (no outgoing edges).
        # Intermediate agents (e.g. engineer) may have a pr_url from the safety
        # net or LLM, but the chain must continue to QA/architect.
        pr_info = self._extract_pr_info(task, response)
        if pr_info and workflow.is_terminal_step(current_step.id):
            task.context["pr_url"] = pr_info["pr_url"]
            task.context["pr_number"] = pr_info["pr_number"]
            self.logger.info(f"PR created at terminal step for task {task.id}, workflow complete")
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

        # Checkpoint gate — pause for human approval before routing to next step.
        # Compare checkpoint_id so each checkpoint requires its own approval;
        # a prior approval at a different step won't bypass this one.
        if current_step.checkpoint:
            checkpoint_id = f"{workflow.name}-{current_step.id}"
            already_approved = (
                task.approved_at is not None
                and task.checkpoint_reached == checkpoint_id
            )
            if not already_approved:
                message = current_step.checkpoint.message
                task.mark_awaiting_approval(checkpoint_id, message)
                self._save_task_checkpoint(task)
                self.logger.info(
                    f"Task {task.id} paused at checkpoint '{checkpoint_id}': {message}"
                )
                self.logger.info(f"   Run 'agent approve {task.id}' to continue workflow")
                return False

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

        # Hard ceiling: refuse to create chain tasks beyond max depth
        chain_depth = task.context.get("_chain_depth", 0)
        if chain_depth >= MAX_CHAIN_DEPTH:
            self.logger.warning(
                f"Chain depth {chain_depth} reached max ({MAX_CHAIN_DEPTH}) "
                f"for task {task.id} — halting workflow to prevent runaway loop"
            )
            return

        # Absolute ceiling that survives escalation/re-planning resets
        global_cycles = task.context.get("_global_cycle_count", 0)
        if global_cycles >= MAX_GLOBAL_CYCLES:
            self.logger.warning(
                f"Global cycle count {global_cycles} reached max ({MAX_GLOBAL_CYCLES}) "
                f"for task {task.id} — halting workflow to prevent runaway loop"
            )
            return

        # Cap QA→engineer review cycles to prevent infinite bounce loops.
        # QA routing back to engineer (needs_fix) increments the counter.
        qa_cycles = task.context.get("_dag_review_cycles", 0)
        is_qa_to_engineer = (
            current_agent_id in ("qa", "qa-1", "qa-2")
            and target_step.agent == "engineer"
        )
        if is_qa_to_engineer:
            qa_cycles += 1
            if qa_cycles > MAX_DAG_REVIEW_CYCLES:
                self.logger.warning(
                    f"QA→engineer review cycle {qa_cycles} exceeds max ({MAX_DAG_REVIEW_CYCLES}) "
                    f"for task {task.id} — routing to PR creation instead of another fix cycle"
                )
                # Try to route to create_pr step instead of looping back
                pr_step = workflow.steps.get("create_pr")
                if pr_step and pr_step != target_step:
                    target_step = pr_step
                    next_agent = pr_step.agent
                else:
                    return

        chain_task = self._build_chain_task(
            task, target_step, current_agent_id, is_qa_to_engineer=is_qa_to_engineer,
        )

        if self._is_chain_task_already_queued(
            next_agent, task.id, chain_id=chain_task.id, title=chain_task.title,
        ):
            self.logger.debug(f"Chain task for {next_agent} already queued from {task.id}")
            return

        if is_qa_to_engineer:
            chain_task.context["_dag_review_cycles"] = qa_cycles

        try:
            self.queue.push(chain_task, next_agent)
            reason = routing_signal.reason if routing_signal else "workflow DAG"
            self.logger.info(f"🔗 Workflow: queued {next_agent} for task {task.id} ({reason})")
        except Exception as e:
            self.logger.error(f"Failed to queue chain task for {next_agent}: {e}")

    def _is_chain_task_already_queued(
        self, next_agent: str, source_task_id: str, *,
        chain_id: Optional[str] = None, title: Optional[str] = None,
    ) -> bool:
        """Check if chain task already exists in target queue or completed."""
        import json

        cid = chain_id or f"chain-{source_task_id}-{next_agent}"
        queue_path = self.queue_dir / next_agent / f"{cid}.json"
        if queue_path.exists():
            return True
        # Also check completed to prevent re-queuing tasks that already ran
        completed_path = self.queue.completed_dir / f"{cid}.json"
        if completed_path.exists():
            return True

        # Title-based dedup: catch same work queued under different IDs
        if title:
            normalized = _strip_chain_prefixes(title).lower().strip()
            queue_dir = self.queue_dir / next_agent
            if queue_dir.exists():
                for f in queue_dir.glob("*.json"):
                    try:
                        data = json.loads(f.read_text())
                        existing = _strip_chain_prefixes(data.get("title", "")).lower().strip()
                        if existing == normalized:
                            self.logger.debug(
                                f"Title dedup: '{title}' matches existing {f.name}"
                            )
                            return True
                    except (json.JSONDecodeError, OSError):
                        continue
        return False

    def _build_chain_task(
        self, task: "Task", target_step: WorkflowStep, current_agent_id: str,
        *, is_qa_to_engineer: bool = False,
    ) -> "Task":
        """Create a continuation task for the target step."""
        from ..core.task import Task

        next_agent = target_step.agent
        chain_depth = task.context.get("_chain_depth", 0) + 1

        # Stable root identity — stamped on first hop, propagated forever
        root_task_id = task.context.get("_root_task_id", task.id)
        chain_id = f"chain-{root_task_id}-{target_step.id}-d{chain_depth}"

        if target_step.task_type_override:
            task_type = TaskType[target_step.task_type_override.upper()]
        else:
            task_type = AGENT_TASK_TYPES.get(next_agent, task.type)

        # Preserve fan-in metadata through the chain
        context = {
            **task.context,
            "source_task_id": task.id,
            "source_agent": current_agent_id,
            "chain_step": True,
            "workflow_step": target_step.id,
            "_chain_depth": chain_depth,
            "_root_task_id": root_task_id,
            "_global_cycle_count": task.context.get("_global_cycle_count", 0) + 1,
        }
        # Clear stale verdict so the next agent's output is evaluated fresh
        context.pop("verdict", None)
        # Prevent same-agent self-referential upstream context — an agent
        # should never see its own output labeled as "UPSTREAM AGENT FINDINGS"
        upstream_source = context.get("upstream_source_agent")
        if upstream_source and upstream_source == target_step.agent:
            context.pop("upstream_summary", None)
            context.pop("upstream_context_file", None)
            context.pop("upstream_source_agent", None)
        if task.context.get("fan_in"):
            context["fan_in"] = True
            context["parent_task_id"] = task.context.get("parent_task_id")
            context["subtask_count"] = task.context.get("subtask_count")

        # Prepend QA findings so the fix-cycle engineer sees what to address
        description = task.description
        if is_qa_to_engineer:
            upstream = task.context.get("upstream_summary", "")
            if upstream:
                description = f"## QA FINDINGS TO ADDRESS\n{upstream}\n\n## ORIGINAL TASK\n{description}"

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=current_agent_id,
            assigned_to=next_agent,
            created_at=datetime.now(timezone.utc),
            title=f"[chain] {_strip_chain_prefixes(task.title)}",
            description=description,
            context=context,
        )

    def _save_task_checkpoint(self, task: "Task") -> None:
        """Save task to checkpoint queue for human approval."""
        from ..utils.atomic_io import atomic_write_model

        checkpoint_queue_dir = self.queue_dir / "checkpoints"
        checkpoint_queue_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_file = checkpoint_queue_dir / f"{task.id}.json"
        atomic_write_model(checkpoint_file, task)
        self.logger.debug(f"Saved checkpoint state for task {task.id}")

    def get_execution_state(self, task: "Task", current_step: str) -> WorkflowExecutionState:
        """Get current execution state for a task."""
        workflow_name = task.context.get("workflow", "unknown")
        completed = task.context.get("completed_steps", [])

        return WorkflowExecutionState(
            task_id=task.id,
            workflow_name=workflow_name,
            current_step=current_step,
            completed_steps=completed,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
