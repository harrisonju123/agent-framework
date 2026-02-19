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
from ..utils.type_helpers import strip_chain_prefixes

logger = logging.getLogger(__name__)


# Hard ceiling on chain depth to prevent runaway loops regardless of routing logic
MAX_CHAIN_DEPTH = 10

# Cap review→engineer fix cycles. The counter is shared across all review steps
# (code_review, qa_review, preview_review) so the total budget is global per task.
# Value of 2: initial review + 1 fix round + final review = 4 chain hops max.
# Consequence: preview_review loops eat into the code_review/qa_review budget.
MAX_DAG_REVIEW_CYCLES = 2

# Absolute ceiling on total chain hops — survives escalation and re-planning resets
MAX_GLOBAL_CYCLES = 15

# Workflow steps where the architect evaluates a plan rather than live code.
# Used by both WorkflowExecutor (verdict routing) and PromptBuilder (guidance injection)
# so that both consumers stay in sync when new review-type steps are added.
PREVIEW_REVIEW_STEPS: frozenset[str] = frozenset({"preview_review"})


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



class WorkflowExecutor:
    """Executes workflow DAGs and routes tasks between agents."""

    def __init__(self, queue: "FileQueue", queue_dir: Path, agent_logger=None, workspace=None):
        self.queue = queue
        self.queue_dir = queue_dir
        self.logger = agent_logger or logger
        self.workspace = workspace

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
            False if workflow is complete or terminal.
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

        # Per-task budget ceiling: terminate chain cleanly (branch is in git, work not lost)
        budget_action = self._check_budget_ceiling(task)
        if budget_action == "halt":
            self.logger.warning(
                f"Cumulative cost ${task.context.get('_cumulative_cost', 0):.2f} "
                f"exceeds ceiling ${task.context.get('_budget_ceiling', 0):.2f} "
                f"for task {task.id} — halting workflow (budget exceeded)"
            )
            task.context["budget_halted"] = True
            return

        # Cap review→engineer fix cycles to prevent infinite bounce loops.
        # code_review, qa_review, and preview_review share a single counter.
        review_cycles = task.context.get("_dag_review_cycles", 0)
        cap_redirect = False
        is_review_to_engineer = (
            target_step.agent == "engineer"
            and task.context.get("workflow_step") in PREVIEW_REVIEW_STEPS | {"code_review", "qa_review"}
        )
        # preview_review → implement is a phase boundary, not a fix cycle.
        # Reset the counter so implementation review gets a full budget, and
        # don't treat this as a review→fix hand-off (no "QA FINDINGS" prefix).
        if (task.context.get("workflow_step") in PREVIEW_REVIEW_STEPS
                and target_step.id == "implement"):
            is_review_to_engineer = False
            review_cycles = 0
        elif is_review_to_engineer:
            review_cycles += 1
            if review_cycles >= MAX_DAG_REVIEW_CYCLES:
                self.logger.warning(
                    f"Review→engineer cycle {review_cycles} exceeds max ({MAX_DAG_REVIEW_CYCLES}) "
                    f"for task {task.id} — routing to PR creation instead of another fix cycle"
                )
                # Try to route to create_pr step instead of looping back
                pr_step = workflow.steps.get("create_pr")
                if pr_step and pr_step != target_step:
                    target_step = pr_step
                    next_agent = pr_step.agent
                    cap_redirect = True
                else:
                    return

        # Cap redirect bypasses the diff check — an infinite review loop is
        # worse than a no-diff PR attempt.
        if not cap_redirect and not self._has_diff_for_pr(task, target_step):
            return

        chain_task = self._build_chain_task(
            task, target_step, current_agent_id, is_review_to_engineer=is_review_to_engineer,
        )

        if self._is_chain_task_already_queued(
            next_agent, task.id, chain_id=chain_task.id, title=chain_task.title,
            root_task_id=task.root_id,
        ):
            self.logger.debug(f"Chain task for {next_agent} already queued from {task.id}")
            return

        if is_review_to_engineer:
            chain_task.context["_dag_review_cycles"] = review_cycles
        elif task.context.get("workflow_step") in PREVIEW_REVIEW_STEPS and target_step.id == "implement":
            # Phase transition resets the counter — write 0 explicitly so the chain
            # task doesn't inherit a non-zero value from the previous preview round.
            chain_task.context["_dag_review_cycles"] = 0

        try:
            self.queue.push(chain_task, next_agent)
            reason = routing_signal.reason if routing_signal else "workflow DAG"
            self.logger.info(f"🔗 Workflow: queued {next_agent} for task {task.id} ({reason})")
        except Exception as e:
            self.logger.error(f"Failed to queue chain task for {next_agent}: {e}")
            return

        # Side-channel: queue QA pre-scan in parallel with code review
        if (task.context.get("workflow_step") == "implement"
                and target_step.id == "code_review"):
            self._queue_qa_pre_scan(task)

    def _is_chain_task_already_queued(
        self, next_agent: str, source_task_id: str, *,
        chain_id: Optional[str] = None, title: Optional[str] = None,
        root_task_id: Optional[str] = None,
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
            normalized = strip_chain_prefixes(title).lower().strip()
            queue_dir = self.queue_dir / next_agent
            if queue_dir.exists():
                for f in queue_dir.glob("*.json"):
                    try:
                        data = json.loads(f.read_text())
                        existing = strip_chain_prefixes(data.get("title", "")).lower().strip()
                        if existing == normalized:
                            self.logger.debug(
                                f"Title dedup: '{title}' matches existing {f.name}"
                            )
                            return True
                    except (json.JSONDecodeError, OSError):
                        continue

        # Cross-type dedup: block chain task when subtasks already exist for the
        # same root task — subtasks handle the work, fan-in aggregates results
        if root_task_id:
            for agent_dir in self.queue_dir.iterdir() if self.queue_dir.exists() else []:
                if not agent_dir.is_dir():
                    continue
                for f in agent_dir.glob("*.json"):
                    try:
                        data = json.loads(f.read_text())
                        if (data.get("parent_task_id")
                                and data.get("context", {}).get("_root_task_id") == root_task_id):
                            self.logger.debug(
                                f"Cross-type dedup: subtask {f.name} shares root {root_task_id}"
                            )
                            return True
                    except (json.JSONDecodeError, OSError):
                        continue

        return False

    def _build_chain_task(
        self, task: "Task", target_step: WorkflowStep, current_agent_id: str,
        *, is_review_to_engineer: bool = False,
    ) -> "Task":
        """Create a continuation task for the target step."""
        from ..core.task import Task

        next_agent = target_step.agent
        chain_depth = task.context.get("_chain_depth", 0) + 1

        # Stable root identity — stamped on first hop, propagated forever
        root_task_id = task.root_id
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
            "_dag_review_cycles": task.context.get("_dag_review_cycles", 0),
        }
        # Clear stale verdict so the next agent's output is evaluated fresh
        context.pop("verdict", None)
        # Thread step-specific instructions into the chain task for prompt injection
        if target_step.instructions is not None:
            context["_step_instructions"] = target_step.instructions
        else:
            context.pop("_step_instructions", None)
        # worktree_branch is ephemeral per-agent — each agent creates its own worktree
        context.pop("worktree_branch", None)
        # Prevent same-step self-referential upstream context — only clear when
        # the exact same workflow step produced the upstream (e.g. engineer→engineer).
        # Different steps on the same agent (plan→create_pr, both architect) keep context.
        upstream_source_step = context.get("upstream_source_step")
        if upstream_source_step and upstream_source_step == target_step.id:
            context.pop("upstream_summary", None)
            context.pop("upstream_context_file", None)
            context.pop("upstream_source_agent", None)
            context.pop("upstream_source_step", None)
        if task.context.get("fan_in"):
            context["fan_in"] = True
            context["parent_task_id"] = task.context.get("parent_task_id")
            context["subtask_count"] = task.context.get("subtask_count")

        # Prepend review findings so the fix-cycle engineer sees what to address
        description = task.description
        user_goal = task.context.get("user_goal", "")

        if is_review_to_engineer:
            upstream = task.context.get("upstream_summary", "")
            if upstream:
                workflow_step = task.context.get("workflow_step", "")
                header = "CODE REVIEW FINDINGS TO ADDRESS" if workflow_step == "code_review" else "QA FINDINGS TO ADDRESS"
                description = f"## {header}\n{upstream}\n\n## ORIGINAL TASK\n{user_goal or description}"
        elif user_goal and target_step.id != task.context.get("workflow_step"):
            step_directives = {
                "implement": "Implement the following changes",
                "code_review": "Review the implementation for the following task",
                "qa_review": "Test and verify the implementation for the following task",
                "create_pr": "Create a pull request for the following completed work",
            }
            directive = step_directives.get(target_step.id)
            if directive:
                description = f"## {directive}\n\n{user_goal}"

        return Task(
            id=chain_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=current_agent_id,
            assigned_to=next_agent,
            created_at=datetime.now(timezone.utc),
            title=f"[chain] {strip_chain_prefixes(task.title)}",
            description=description,
            context=context,
            plan=task.plan,
        )

    def _check_budget_ceiling(self, task: "Task") -> str:
        """Check cumulative cost against budget ceiling.

        Returns "ok", "warn", or "halt".
        """
        ceiling = task.context.get("_budget_ceiling")
        if ceiling is None:
            return "ok"
        cumulative = task.context.get("_cumulative_cost", 0.0)
        if cumulative >= ceiling:
            return "halt"
        if cumulative >= ceiling * 0.8:
            self.logger.info(
                f"Budget warning: ${cumulative:.2f} is ≥80% of "
                f"${ceiling:.2f} ceiling for task {task.id}"
            )
            return "warn"
        return "ok"

    def _has_diff_for_pr(self, task: "Task", target_step: WorkflowStep) -> bool:
        """Check whether there are actual code changes before routing to create_pr.

        Returns True (proceed) unless we can definitively prove there's nothing to PR.
        Only gates the create_pr step — all other steps pass through unconditionally.
        """
        if target_step.id != "create_pr":
            return True

        impl_branch = task.context.get("implementation_branch") if task.context else None
        pr_number = task.context.get("pr_number") if task.context else None

        # Layer 1: no branch and no existing PR → nothing to PR
        if not impl_branch and not pr_number:
            self.logger.info(
                f"No-diff guard: skipping create_pr for task {task.id} — "
                f"no implementation_branch or pr_number in context"
            )
            return False

        # Layer 2: branch exists, verify it has commits vs main
        if impl_branch and self.workspace:
            from ..utils.subprocess_utils import run_git_command
            try:
                result = run_git_command(
                    ["log", "--oneline", f"origin/main..origin/{impl_branch}"],
                    cwd=self.workspace,
                    check=False,
                    timeout=10,
                )
                if result.returncode == 0 and not result.stdout.strip():
                    self.logger.info(
                        f"No-diff guard: skipping create_pr for task {task.id} — "
                        f"branch {impl_branch} has no commits vs main"
                    )
                    return False
            except Exception as e:
                self.logger.warning(
                    f"No-diff guard: git check failed ({e}) for task {task.id} — "
                    f"blocking create_pr (fail closed)"
                )
                return False

        return True

    def _queue_qa_pre_scan(self, task: "Task") -> None:
        """Queue a lightweight QA pre-scan in parallel with code review.

        Fire-and-forget side task — the workflow chain continues regardless.
        Deduped by root_task_id so only the first implement→code_review
        transition triggers a scan.
        """
        from ..core.task import Task

        root_task_id = task.root_id
        impl_branch = task.context.get("implementation_branch")
        if not impl_branch:
            self.logger.debug(f"Skipping QA pre-scan: no implementation_branch for task {task.id}")
            return

        if self._is_prescan_already_queued(root_task_id):
            self.logger.debug(f"QA pre-scan already queued for root {root_task_id}")
            return

        prescan_id = f"prescan-{root_task_id}"
        prescan_context = {
            k: v for k, v in task.context.items()
            if k in (
                "github_repo", "jira_key", "jira_project", "workflow",
                "implementation_branch", "_root_task_id", "_chain_depth",
            )
        }
        prescan_context.update({
            "pre_scan": True,
            "_root_task_id": root_task_id,
            "source_task_id": task.id,
        })

        prescan_task = Task(
            id=prescan_id,
            type=TaskType.QA_VERIFICATION,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=task.assigned_to or "executor",
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title=f"[pre-scan] {task.title}",
            description=(
                "QA PRE-SCAN: Run lightweight checks in parallel with architect code review.\n\n"
                "Run ONLY:\n"
                "1. Linting (language-appropriate linter)\n"
                "2. Test suite execution\n"
                "3. Basic security scan (if available)\n\n"
                "Do NOT:\n"
                "- Do deep code review (architect handles that)\n"
                "- Create PRs or modify files\n"
                "- Route to other agents or create follow-up tasks\n\n"
                "Output structured findings in JSON format.\n\n"
                f"Branch: {impl_branch}\n"
                f"Original task: {task.title}"
            ),
            context=prescan_context,
        )

        try:
            self.queue.push(prescan_task, "qa")
            self.logger.info(f"🔍 Pre-scan: queued QA pre-scan for root {root_task_id}")
        except Exception as e:
            self.logger.warning(f"Failed to queue QA pre-scan (non-fatal): {e}")

    def _is_prescan_already_queued(self, root_task_id: str) -> bool:
        """Check if a pre-scan task already exists for this root task."""
        prescan_id = f"prescan-{root_task_id}"
        queue_path = self.queue_dir / "qa" / f"{prescan_id}.json"
        if queue_path.exists():
            return True
        completed_path = self.queue.completed_dir / f"{prescan_id}.json"
        if completed_path.exists():
            return True
        return False

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
