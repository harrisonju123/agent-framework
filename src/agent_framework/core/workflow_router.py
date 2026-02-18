"""Workflow routing and task decomposition logic extracted from Agent."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentDefinition, WorkflowDefinition
    from ..queue.file_queue import FileQueue
    from ..workflow.executor import WorkflowExecutor

from .task import Task, TaskStatus, TaskType
from .routing import validate_routing_signal, log_routing_decision, WORKFLOW_COMPLETE
from ..utils.type_helpers import get_type_str


class WorkflowRouter:
    """Handles workflow chain enforcement, task decomposition, and agent routing."""

    def __init__(
        self,
        config,
        queue: "FileQueue",
        workspace: Path,
        logger,
        session_logger,
        workflows_config: Dict[str, "WorkflowDefinition"],
        workflow_executor: "WorkflowExecutor",
        agents_config: List["AgentDefinition"],
        multi_repo_manager=None,
    ):
        """Initialize WorkflowRouter with dependencies from Agent.

        Args:
            config: Agent configuration
            queue: FileQueue for task management
            workspace: Path to workspace
            logger: Logger instance
            session_logger: Session logger for metrics
            workflows_config: Available workflow definitions
            workflow_executor: WorkflowExecutor for DAG execution
            agents_config: List of agent definitions
            multi_repo_manager: Optional multi-repo manager
        """
        self.config = config
        self.queue = queue
        self.workspace = workspace
        self.logger = logger
        self._session_logger = session_logger
        self._workflows_config = workflows_config
        self._workflow_executor = workflow_executor
        self._agents_config = agents_config
        self.multi_repo_manager = multi_repo_manager

    def set_session_logger(self, session_logger) -> None:
        """Update the session logger for this router.

        Args:
            session_logger: SessionLogger instance to use for metrics logging
        """
        self._session_logger = session_logger

    def check_and_create_fan_in_task(self, task: Task) -> None:
        """Check if this subtask completion triggers fan-in task creation.

        When a subtask completes, checks if all siblings are also complete.
        If so, creates a fan-in task that aggregates results and continues workflow.
        """
        if not task.parent_task_id:
            return

        # This is a subtask - check if all siblings are done
        parent = self.queue.find_task(task.parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        if self.queue.check_subtasks_complete(parent.id, parent.subtask_ids):
            # All subtasks done - create fan-in task
            if not self.queue._fan_in_already_created(parent.id):
                completed_subtasks = [
                    self.queue.get_completed(sid) for sid in parent.subtask_ids
                ]
                completed_subtasks = [s for s in completed_subtasks if s is not None]
                fan_in_task = self.queue.create_fan_in_task(parent, completed_subtasks)
                self.queue.push(fan_in_task, fan_in_task.assigned_to)
                self.logger.info(
                    f"🔀 All subtasks complete - created fan-in task {fan_in_task.id}"
                )
        else:
            self.logger.info(
                f"Subtask {task.id} complete, waiting for siblings"
            )

    def should_decompose_task(self, task: Task) -> bool:
        """Check if task should be decomposed into subtasks.

        Only applies to architect-created tasks with plans.
        Uses TaskDecomposer heuristics (estimated lines > threshold).
        """
        if not task.plan:
            return False

        # Don't decompose subtasks (max depth = 1)
        if task.parent_task_id:
            return False

        # Estimate lines: files_to_modify count * 15 lines/file (rough heuristic)
        estimated_lines = len(task.plan.files_to_modify) * 15 if task.plan.files_to_modify else 0

        from .task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        return decomposer.should_decompose(task.plan, estimated_lines)

    def decompose_and_queue_subtasks(self, task: Task) -> None:
        """Decompose task into subtasks and queue them to engineer.

        Replaces normal workflow routing - subtasks will each flow through
        the workflow individually, and fan-in will aggregate them at completion.
        """
        from .task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()
        estimated_lines = len(task.plan.files_to_modify) * 15 if task.plan.files_to_modify else 0

        self.logger.info(
            f"Decomposing task {task.id} into parallel subtasks "
            f"(estimated {estimated_lines} lines across {len(task.plan.files_to_modify)} files)"
        )

        subtasks = decomposer.decompose(task, task.plan, estimated_lines)

        # Queue each subtask to engineer (skip duplicates)
        queued_count = 0
        for subtask in subtasks:
            task_file = self.queue.queue_dir / "engineer" / f"{subtask.id}.json"
            completed_file = self.queue.completed_dir / f"{subtask.id}.json"
            if task_file.exists() or completed_file.exists():
                self.logger.info(f"  ⏭️  Subtask {subtask.id} already exists, skipping")
                continue
            self.queue.push(subtask, "engineer")
            queued_count += 1
            self.logger.info(f"  ✅ Queued subtask: {subtask.id} ({subtask.title})")

        # Update parent task with subtask IDs and persist
        task.subtask_ids = [st.id for st in subtasks]
        task.result_summary = f"Decomposed into {len(subtasks)} subtasks ({queued_count} newly queued)"

        # mark_completed() already moved the task file to completed_dir before
        # enforce_chain runs, so queue.update() would silently fail (file not in
        # queue_dir). Write directly to the completed copy instead.
        completed_file = self.queue.completed_dir / f"{task.id}.json"
        if completed_file.exists():
            from ..utils.atomic_io import atomic_write_model
            atomic_write_model(completed_file, task)
        else:
            self.queue.update(task)

        self.logger.info(
            f"🔀 Task {task.id} decomposed into {len(subtasks)} subtasks ({queued_count} newly queued)"
        )

    def enforce_chain(self, task: Task, response, routing_signal=None) -> None:
        """Queue next agent in workflow using DAG executor.

        Supports both legacy linear workflows and new DAG workflows with conditions.
        """
        # Task decomposition: architect auto-decomposes large tasks before routing to engineer
        if self.config.base_id == "architect" and task.plan:
            if self.should_decompose_task(task):
                self.decompose_and_queue_subtasks(task)
                return

        # If subtasks were already created (by a prior run), skip chain routing —
        # subtasks handle the work individually, fan-in aggregates at completion
        if task.subtask_ids:
            self.logger.info(
                f"Task {task.id} already decomposed into {len(task.subtask_ids)} subtasks, "
                f"skipping chain routing"
            )
            return

        # Preview tasks route back to architect for review, not to QA.
        # Store the preview output as an artifact so it can be referenced
        # by the subsequent implementation task.
        if task.type == TaskType.PREVIEW and self.config.base_id == 'engineer':
            if task.result_summary:
                task.context['preview_artifact'] = task.result_summary
            self.route_to_agent(task, 'architect', 'preview_review')
            return

        # Legacy REVIEW/FIX tasks are routed by _queue_code_review_if_needed
        # and _queue_review_fix_if_needed — letting them also route through
        # the DAG creates a duplicate-routing feedback loop.
        # Chain tasks (chain_step=True) with REVIEW type are DAG-routed
        # (e.g. code_review step) and must pass through.
        if task.type in (TaskType.REVIEW, TaskType.FIX) and not task.context.get("chain_step"):
            self.logger.debug(
                f"Skipping workflow chain for {task.id}: "
                f"task type {task.type} handled by dedicated review routing"
            )
            return

        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            self.logger.debug(
                f"No workflow chain for {task.id}: "
                f"workflow={workflow_name!r}, available={list(self._workflows_config.keys())}"
            )
            # No workflow defined - handle routing signal if present
            if routing_signal:
                validated = validate_routing_signal(
                    routing_signal, self.config.base_id, get_type_str(task.type), self._agents_config,
                )
                if validated and validated != WORKFLOW_COMPLETE:
                    self.route_to_agent(task, validated, routing_signal.reason)
                log_routing_decision(
                    self.workspace, task.id, self.config.id,
                    routing_signal, validated, used_fallback=False,
                )
            return

        # Get workflow definition and convert to DAG
        workflow_def = self._workflows_config[workflow_name]
        try:
            workflow_dag = workflow_def.to_dag(workflow_name)
        except Exception as e:
            self.logger.error(f"Failed to build workflow DAG for {workflow_name}: {e}")
            return

        # Single-agent workflows don't need routing
        if len(workflow_dag.get_all_agents()) <= 1:
            if routing_signal:
                self.logger.debug("Routing signal discarded: single-agent workflow")
            return

        # Execute workflow step using DAG executor
        try:
            routed = self._workflow_executor.execute_step(
                workflow=workflow_dag,
                task=task,
                response=response,
                current_agent_id=self.config.base_id,
                routing_signal=routing_signal,
                context=self.build_workflow_context(task),
            )

            # Terminal step with no routing — check if pr_creator should take over
            if not routed:
                self.queue_pr_creation_if_needed(task, workflow_def)

            # Log routing decision
            if routing_signal:
                log_routing_decision(
                    self.workspace, task.id, self.config.id,
                    routing_signal, None, used_fallback=not routed,
                )

            # Session logging
            if routed and self._session_logger:
                self._session_logger.log(
                    "workflow_routing",
                    workflow=workflow_name,
                    signal=routing_signal.target_agent if routing_signal else None,
                )
        except Exception as e:
            self.logger.error(f"Workflow execution failed for task {task.id}: {e}")

    def is_at_terminal_workflow_step(self, task: Task) -> bool:
        """Check if the current agent is at the last step in the workflow DAG.

        Returns True for standalone tasks (no workflow) to preserve backward
        compatibility — standalone agents should always be allowed to create PRs.
        """
        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            return True

        workflow_def = self._workflows_config[workflow_name]
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
            if step.agent == self.config.base_id:
                return dag.is_terminal_step(step.id)

        return True

    def build_workflow_context(self, task: Task) -> Dict[str, Any]:
        """Build context dict for workflow condition evaluation."""
        context = {}

        if task.context and "verdict" in task.context:
            context["verdict"] = task.context["verdict"]

        # Prefer task context, fallback to git diff
        if task.context and "changed_files" in task.context:
            context["changed_files"] = task.context["changed_files"]
        else:
            changed_files = self._get_changed_files()
            if changed_files:
                context["changed_files"] = changed_files

        # Add test results if available
        if task.context and "test_result" in task.context:
            context["test_result"] = task.context["test_result"]

        return context

    def _get_changed_files(self) -> List[str]:
        """Get list of changed files from git diff (staged and unstaged)."""
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        try:
            result = run_git_command(
                ["diff", "--name-only", "HEAD"],
                cwd=self.workspace,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                self.logger.debug(f"git diff failed: {result.stderr}")
                return []
            return [f.strip() for f in result.stdout.split("\n") if f.strip()]
        except SubprocessError:
            self.logger.warning("git diff timed out")
            return []
        except Exception as e:
            self.logger.debug(f"Failed to get changed files: {e}")
            return []

    def route_to_agent(self, task: Task, target_agent: str, reason: str) -> None:
        """Route task to another agent using the workflow executor's chain builder."""
        from ..workflow.dag import WorkflowStep
        from ..workflow.executor import AGENT_TASK_TYPES

        # Build a synthetic WorkflowStep so executor._build_chain_task works
        task_type_override = None
        if target_agent in AGENT_TASK_TYPES:
            task_type_override = AGENT_TASK_TYPES[target_agent].name.lower()

        step = WorkflowStep(
            id=target_agent,
            agent=target_agent,
            task_type_override=task_type_override,
        )

        chain_task = self._workflow_executor._build_chain_task(
            task, step, self.config.id
        )

        if self._workflow_executor._is_chain_task_already_queued(
            target_agent, task.id, chain_id=chain_task.id
        ):
            self.logger.debug(f"Chain task for {target_agent} already queued from {task.id}")
            return

        try:
            self.queue.push(chain_task, target_agent)
            self.logger.info(f"🔗 Routed to {target_agent} (signal): {reason}")
            if self._session_logger:
                self._session_logger.log(
                    "workflow_chain",
                    next_agent=target_agent,
                    reason=reason,
                )
        except Exception as e:
            self.logger.error(f"Failed to route task to {target_agent}: {e}")

    def queue_pr_creation_if_needed(self, task: Task, workflow) -> None:
        """Queue a PR creation task when the last agent in the chain completes.

        The workflow's pr_creator field designates which agent should open the PR.
        Without this, the chain ends silently after the last agent finishes.
        """
        pr_creator = getattr(workflow, "pr_creator", None)
        if not pr_creator:
            return

        if task.context.get("pr_creation_step"):
            return

        if task.context.get("pr_url"):
            return

        # No code changes = no PR needed (e.g., architect planning-only tasks)
        if not task.context.get("implementation_branch") and not task.context.get("pr_number"):
            self.logger.debug(f"No implementation branch or PR for {task.id}, skipping PR creation")
            return

        # Deterministic ID with -pr suffix to avoid collision with normal chain tasks
        pr_task_id = f"chain-{task.id}-{pr_creator}-pr"
        queue_path = self.queue.queue_dir / pr_creator / f"{pr_task_id}.json"
        if queue_path.exists():
            self.logger.debug(f"PR creation task {pr_task_id} already queued, skipping")
            return

        from ..workflow.executor import _strip_chain_prefixes

        pr_context = {
            **task.context,
            "source_task_id": task.id,
            "source_agent": self.config.id,
            "pr_creation_step": True,
        }
        # worktree_branch is ephemeral per-agent — PR creator gets its own working dir
        pr_context.pop("worktree_branch", None)

        pr_task = Task(
            id=pr_task_id,
            type=TaskType.PR_REQUEST,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to=pr_creator,
            created_at=datetime.now(timezone.utc),
            title=f"[pr] {_strip_chain_prefixes(task.title)}",
            description=task.description,
            context=pr_context,
        )

        try:
            self.queue.push(pr_task, pr_creator)
            self.logger.info(f"📦 Queued PR creation for {pr_creator} from task {task.id}")
            if self._session_logger:
                self._session_logger.log(
                    "pr_creation_queued",
                    pr_creator=pr_creator,
                    source_task=task.id,
                )
        except Exception as e:
            self.logger.error(f"Failed to queue PR creation: {e}")
