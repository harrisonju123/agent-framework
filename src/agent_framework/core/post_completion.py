"""Post-completion flow manager.

Handles verdict setting, plan extraction, upstream context persistence,
chain state recording, workflow chain dispatch, and PR creation gating.
Extracted from agent.py to reduce its size.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .config import AgentConfig, AgentDefinition
    from .review_cycle import ReviewCycleManager
    from .workflow_router import WorkflowRouter
    from .git_operations import GitOperationsManager
    from .budget_manager import BudgetManager
    from .error_recovery import ErrorRecoveryManager
    from .session_logger import SessionLogger
    from .activity import ActivityManager
    from ..queue.file_queue import FileQueue
    from ..utils.rich_logging import ContextLogger

from .task import PlanDocument, Task
from .activity import ActivityEvent, AgentActivity, AgentStatus, CurrentTask
from ..utils.type_helpers import get_type_str
from ..workflow.constants import WorkflowStepConstants as Steps

# Matches synthetic [Tool Call: Read], [Tool Call: Bash] etc. markers injected
# by the Claude CLI backend into response.content for logging visibility.
_TOOL_CALL_MARKER_RE = re.compile(r'\n?\[Tool Call: [^\]]+\]\n?')

_JSON_FENCE_PATTERN = re.compile(r'```json\s*\n(.*?)\n?\s*```', re.DOTALL)

_NO_CHANGES_MARKER = "[NO_CHANGES_NEEDED]"

# Step classification for the deliverable gate — canonical source is WorkflowStepConstants
_IMPLEMENTATION_STEP_IDS = Steps.IMPLEMENTATION_STEPS
_NON_CODE_STEP_IDS = Steps.NON_CODE_STEPS

_RATIONALE_RE = re.compile(
    r'[^.]*\b(?:because|tradeoff|trade-off|instead of|constraint|reason)\b[^.]*\.',
    re.IGNORECASE,
)


def strip_tool_call_markers(content: str) -> str:
    """Remove [Tool Call: ...] markers and compress resulting whitespace."""
    if not content:
        return ""
    cleaned = _TOOL_CALL_MARKER_RE.sub('\n', content)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


class PostCompletionManager:
    """Manages post-LLM-completion flow: verdicts, context handoff, chain routing."""

    UPSTREAM_CONTEXT_MAX_CHARS = 15000
    UPSTREAM_INLINE_MAX_CHARS = 15000

    def __init__(
        self,
        config: "AgentConfig",
        queue: "FileQueue",
        workspace: Path,
        logger: "ContextLogger",
        session_logger: "SessionLogger",
        activity_manager: "ActivityManager",
        review_cycle: "ReviewCycleManager",
        workflow_router: "WorkflowRouter",
        git_ops: "GitOperationsManager",
        budget: "BudgetManager",
        error_recovery: "ErrorRecoveryManager",
        optimization_config: dict,
        session_logging_enabled: bool,
        session_logs_dir: Path,
        agent_definition: Optional["AgentDefinition"] = None,
    ):
        self.config = config
        self.queue = queue
        self.workspace = workspace
        self.logger = logger
        self.session_logger = session_logger
        self.activity_manager = activity_manager
        self.review_cycle = review_cycle
        self.workflow_router = workflow_router
        self.git_ops = git_ops
        self.budget = budget
        self.error_recovery = error_recovery
        self.optimization_config = optimization_config
        self.session_logging_enabled = session_logging_enabled
        self.session_logs_dir = session_logs_dir
        self._agent_definition = agent_definition

    def set_session_logger(self, session_logger: "SessionLogger") -> None:
        """Update session logger for new task."""
        self.session_logger = session_logger

    # -- Verdict Logic --

    def approval_verdict(self, task: Task) -> str:
        """Return the appropriate approval verdict for the current workflow step.

        preview_review uses "preview_approved" so the preview_approved DAG edge
        fires instead of the generic "approved" edge.
        """
        if task.context.get("workflow_step") in Steps.PREVIEW_REVIEW_STEPS:
            return "preview_approved"
        return "approved"

    def set_structured_verdict(self, task: Task, response) -> None:
        """Parse review outcome and store verdict + audit trail before serialization.

        Only qa/architect agents with a workflow produce verdicts.
        """
        if not task.context.get("workflow"):
            return
        if self.config.base_id not in ("qa", "architect"):
            return

        content = getattr(response, "content", "") or ""
        outcome, audit = self.review_cycle._parse_review_outcome_audited(content)

        workflow_step = task.context.get("workflow_step", "")

        if outcome.approved:
            task.context["verdict"] = self.approval_verdict(task)
            audit.method = "review_outcome"
        elif outcome.needs_fix:
            task.context["verdict"] = "needs_fix"
            audit.method = "review_outcome"
        else:
            if workflow_step in Steps.PREVIEW_REVIEW_STEPS:
                # Preview review is a pre-implementation safety gate — ambiguous
                # output should halt, not auto-approve past the gate
                self.logger.warning(
                    f"Ambiguous review outcome at step {workflow_step!r} for "
                    f"task {task.id} — not setting verdict (preview gate halts)"
                )
                audit.method = "ambiguous_halt"
            else:
                # Default to approved so the chain keeps moving — downstream
                # reviewers will catch real issues
                task.context["verdict"] = self.approval_verdict(task)
                if workflow_step in Steps.REVIEW_STEPS:
                    self.logger.warning(
                        f"Ambiguous review outcome at step {workflow_step!r} for "
                        f"task {task.id} — defaulting to approved to keep chain moving"
                    )
                    audit.method = "ambiguous_review_default"
                else:
                    audit.method = "ambiguous_default"

        # no_changes marker overrides any previous verdict at plan step
        if (self.config.base_id == "architect"
                and task.context.get("workflow_step", get_type_str(task.type)) in (Steps.PLAN, "planning")):
            if self.is_no_changes_response(content):
                task.context["verdict"] = "no_changes"
                audit.method = "no_changes_marker"
                audit.no_changes_marker_found = True

        audit.agent_id = self.config.id
        audit.workflow_step = workflow_step
        audit.task_id = task.id
        audit.value = task.context.get("verdict")

        task.context["verdict_audit"] = audit.to_dict()
        self.session_logger.log("verdict_audit", **audit.to_dict())

    @staticmethod
    def is_no_changes_response(content: str) -> bool:
        """Detect if response indicates no code changes are needed."""
        if not content:
            return False
        return _NO_CHANGES_MARKER in content[:200]

    @staticmethod
    def is_implementation_step(task: Task, agent_base_id: str) -> bool:
        """Whether this task is an implementation step that must produce code."""
        step = task.context.get("workflow_step")
        if step in _IMPLEMENTATION_STEP_IDS:
            return True
        if step in _NON_CODE_STEP_IDS:
            return False
        return agent_base_id == "engineer"

    # -- Plan & Rationale Extraction --

    @staticmethod
    def extract_plan_from_response(content: str) -> Optional[PlanDocument]:
        """Parse PlanDocument JSON from architect's planning response."""
        if not content:
            return None

        matches = _JSON_FENCE_PATTERN.findall(content)
        if not matches:
            return None

        for raw in matches:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            if "plan" in parsed and isinstance(parsed["plan"], dict):
                parsed = parsed["plan"]
            if not all(k in parsed for k in ("objectives", "approach", "success_criteria")):
                continue
            try:
                return PlanDocument.model_validate(parsed)
            except Exception:
                continue
        return None

    @staticmethod
    def extract_design_rationale(content: str) -> Optional[str]:
        """Extract design rationale sentences from planning response."""
        if not content:
            return None

        sentences = _RATIONALE_RE.findall(content)
        if not sentences:
            return None

        seen: set[str] = set()
        unique = []
        for s in sentences:
            s = s.strip()
            if s not in seen and len(s) > 20:
                seen.add(s)
                unique.append(s)

        if not unique:
            return None

        result = " ".join(unique)
        return result[:1000] if len(result) > 1000 else result

    @staticmethod
    def extract_structured_findings_from_content(content: str) -> dict:
        """Parse structured findings JSON from LLM response content."""
        if not content:
            return {}

        matches = _JSON_FENCE_PATTERN.findall(content)
        if not matches:
            return {}

        try:
            parsed = json.loads(matches[-1])
            if isinstance(parsed, list):
                return {"findings": parsed, "summary": ""}
            if isinstance(parsed, dict):
                if "findings" not in parsed:
                    parsed["findings"] = []
                return parsed
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}

    # -- Context Persistence --

    def save_upstream_context(self, task: Task, response) -> None:
        """Save agent's response to disk so downstream agents can read it."""
        try:
            from ..utils.atomic_io import atomic_write_text

            summaries_dir = self.workspace / ".agent-context" / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)

            content = strip_tool_call_markers(response.content or "")
            if len(content) > self.UPSTREAM_CONTEXT_MAX_CHARS:
                content = content[:self.UPSTREAM_CONTEXT_MAX_CHARS] + "\n\n[truncated]"

            context_file = summaries_dir / f"{task.id}-{self.config.base_id}.md"
            atomic_write_text(context_file, content)

            task.context["upstream_context_file"] = str(context_file)
            task.context["upstream_summary"] = content[:self.UPSTREAM_INLINE_MAX_CHARS]
            task.context["upstream_source_agent"] = self.config.base_id
            step = task.context.get("workflow_step")
            if step:
                task.context["upstream_source_step"] = step
            self.logger.debug(f"Saved upstream context ({len(content)} chars) to {context_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save upstream context for {task.id}: {e}")

    def save_step_to_chain_state(
        self, task: Task, response, *,
        working_dir: Optional[Path] = None,
        task_start_time: Optional[datetime] = None,
    ) -> None:
        """Append a structured step record to the chain state file."""
        try:
            from .chain_state import append_step

            tool_stats_dict = None
            if self.session_logging_enabled:
                tool_stats_dict = self.compute_tool_stats_for_chain(task)

            content = getattr(response, "content", "") or ""
            state = append_step(
                workspace=self.workspace,
                task=task,
                agent_id=self.config.base_id,
                response_content=content,
                working_dir=working_dir,
                tool_stats=tool_stats_dict,
                started_at=task_start_time,
            )

            if state.steps:
                last_step = state.steps[-1]
                if last_step.files_modified:
                    task.context["files_modified"] = last_step.files_modified

            self.logger.debug(
                f"Chain state: appended step {task.context.get('workflow_step', 'unknown')} "
                f"({len(state.steps)} total steps)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save chain state for {task.id}: {e}")

    def emit_workflow_summary(self, task: Task) -> None:
        """Emit a waterfall summary event capturing the full workflow timeline."""
        try:
            from .chain_state import load_chain_state, build_workflow_summary

            state = load_chain_state(self.workspace, task.root_id)
            if state is None or not state.steps:
                return

            summary = build_workflow_summary(state)

            pr_url = task.context.get("pr_url")
            if pr_url:
                summary["pr_url"] = pr_url

            if self.session_logging_enabled:
                self.session_logger.log("workflow_summary", **summary)

            self.activity_manager.append_event(ActivityEvent(
                type="workflow_summary",
                agent=self.config.id,
                task_id=task.id,
                title=f"Workflow {summary.get('outcome', 'unknown')}: {len(summary.get('steps', []))} steps",
                timestamp=datetime.now(timezone.utc),
                root_task_id=task.root_id,
                duration_ms=int(summary["total_duration_seconds"] * 1000) if summary.get("total_duration_seconds") else None,
            ))

            self.logger.info(
                f"Workflow summary: {summary.get('outcome')} in {len(summary.get('steps', []))} steps, "
                f"{summary.get('total_duration_seconds')}s"
            )
        except Exception as e:
            self.logger.warning(f"Failed to emit workflow summary for {task.id}: {e}")

    def compute_tool_stats_for_chain(self, task: Task) -> Optional[Dict]:
        """Compute tool usage stats dict for embedding in chain state."""
        session_path = self.session_logs_dir / "sessions" / f"{task.id}.jsonl"
        if not session_path.exists():
            return None

        try:
            from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer, compute_tool_usage_stats
            from dataclasses import asdict

            analyzer = ToolPatternAnalyzer()
            tool_calls = analyzer.extract_tool_calls(session_path)
            if not tool_calls:
                return None

            stats = compute_tool_usage_stats(tool_calls)
            stats_dict = asdict(stats)
            task.context["_tool_stats_cache"] = stats_dict
            return stats_dict
        except Exception as e:
            self.logger.debug(f"Tool stats computation failed (non-fatal): {e}")
            return None

    def save_pre_scan_findings(self, task: Task, response) -> None:
        """Persist QA pre-scan results so downstream agents can load them."""
        try:
            from ..utils.atomic_io import atomic_write_text

            root_task_id = task.root_id
            pre_scans_dir = self.workspace / ".agent-communication" / "pre-scans"
            pre_scans_dir.mkdir(parents=True, exist_ok=True)

            content = getattr(response, "content", "") or ""
            structured = self.extract_structured_findings_from_content(content)

            findings_data = {
                "root_task_id": root_task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_summary": content[:4000],
                "structured_findings": structured,
            }

            findings_file = pre_scans_dir / f"{root_task_id}.json"
            atomic_write_text(findings_file, json.dumps(findings_data, indent=2))
            self.logger.info(f"Saved pre-scan findings to {findings_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save pre-scan findings for {task.id}: {e}")

    # -- Completion Metrics --

    def log_task_completion_metrics(
        self, task: Task, response, task_start_time, *,
        tool_call_count=None, context_window_manager=None,
    ) -> None:
        """Log token usage, cost, and completion events. Delegates to BudgetManager."""
        ctx_status = context_window_manager.get_budget_status() if context_window_manager else None
        self.budget.log_task_completion_metrics(
            task, response, task_start_time,
            tool_call_count=tool_call_count,
            root_task_id=task.root_id,
            context_budget_status=ctx_status,
        )

    # -- Main Post-Completion Flow --

    def run_post_completion_flow(
        self, task: Task, response, routing_signal, task_start_time, *,
        context_window_manager=None,
        extract_and_store_memories_cb=None,
        analyze_tool_patterns_cb=None,
        sync_jira_status_cb=None,
    ) -> None:
        """Route completed task through workflow chain, collect metrics.

        Subtasks (parent_task_id set) skip the workflow chain — the fan-in
        task aggregates results and flows through QA/review/PR instead.
        """
        # Accumulate cost before chain routing
        if response is not None:
            this_cost = self.budget.estimate_cost(response)
            prev = task.context.get("_cumulative_cost", 0.0)
            task.context["_cumulative_cost"] = prev + this_cost

        # Stamp ceiling once on root task
        if "_budget_ceiling" not in task.context:
            ceiling = self.resolve_budget_ceiling(task)
            if ceiling is not None:
                task.context["_budget_ceiling"] = ceiling

        # Pre-scan tasks are fire-and-forget
        if task.context.get("pre_scan"):
            self.save_pre_scan_findings(task, response)
            if extract_and_store_memories_cb:
                extract_and_store_memories_cb(task, response)
            tool_call_count = analyze_tool_patterns_cb(task) if analyze_tool_patterns_cb else None
            self.log_task_completion_metrics(
                task, response, task_start_time,
                tool_call_count=tool_call_count,
                context_window_manager=context_window_manager,
            )
            return

        # Validate parent actually exists before fan-in
        if task.parent_task_id is not None:
            parent = self.workflow_router.queue.find_task(task.parent_task_id)
            if parent is None:
                self.logger.warning(
                    f"Task {task.id} has phantom parent_task_id "
                    f"{task.parent_task_id!r} — not found in queue/completed. "
                    f"Clearing to allow normal workflow routing."
                )
                task.parent_task_id = None

        # Fan-in check
        self.workflow_router.check_and_create_fan_in_task(task)

        if task.parent_task_id is not None:
            self.logger.debug(
                f"Subtask {task.id} complete — skipping workflow chain "
                f"(fan-in will handle routing)"
            )
        else:
            has_workflow = bool(task.context.get("workflow"))
            if not has_workflow and not task.context.get("chain_step"):
                self.logger.warning(
                    f"Task {task.id} has no workflow in context — "
                    f"expected for CLI/web-created tasks"
                )
            if not has_workflow:
                self.logger.debug(f"Checking if code review needed for {task.id}")
                self.review_cycle.queue_code_review_if_needed(task, response)
                self.review_cycle.queue_review_fix_if_needed(task, response, sync_jira_status_cb)

            skip_chain = task.context.get("verdict") == "no_changes"
            if skip_chain:
                self.logger.info(
                    f"No-changes verdict at plan step for task {task.id} — "
                    f"terminating workflow (nothing to implement or PR)"
                )

            self.git_ops.detect_implementation_branch(task)

            chain_routed = False
            if not skip_chain:
                chain_routed = self.workflow_router.enforce_chain(task, response, routing_signal)

            if skip_chain or self.workflow_router.is_at_terminal_workflow_step(task):
                self.emit_workflow_summary(task)

            if not chain_routed:
                terminal_verdict = task.context.get("verdict")
                if terminal_verdict == "needs_fix" and self.workflow_router.is_at_terminal_workflow_step(task):
                    self.logger.warning(
                        f"Skipping PR creation: terminal step verdict is needs_fix for {task.id}"
                    )
                else:
                    self.git_ops.push_and_create_pr_if_needed(task)
                    self.git_ops.manage_pr_lifecycle(task)

        if extract_and_store_memories_cb:
            extract_and_store_memories_cb(task, response)
        tool_call_count = analyze_tool_patterns_cb(task) if analyze_tool_patterns_cb else None
        self.log_task_completion_metrics(
            task, response, task_start_time,
            tool_call_count=tool_call_count,
            context_window_manager=context_window_manager,
        )

    def resolve_budget_ceiling(self, task: Task) -> Optional[float]:
        """Resolve USD budget ceiling from task effort and/or absolute cap."""
        effort_ceiling = None
        if self.optimization_config.get("enable_effort_budget_ceilings", False):
            effort = task.estimated_effort
            if not effort:
                effort = self.budget.derive_effort_from_plan(task.plan)
            effort_ceiling = self.budget.get_effort_ceiling(effort.upper())

        absolute = self.optimization_config.get("absolute_budget_ceiling_usd")

        if effort_ceiling is not None and absolute is not None:
            return min(effort_ceiling, absolute)
        return effort_ceiling if effort_ceiling is not None else absolute

    def finalize_successful_response(
        self, task: Task, response, task_start_time, *,
        working_dir: Optional[Path] = None,
        routing_signal=None,
        context_window_manager=None,
        extract_and_store_memories_cb=None,
        analyze_tool_patterns_cb=None,
        sync_jira_status_cb=None,
    ) -> None:
        """Handle the post-eval/post-gate phase of a successful response.

        Called by agent._handle_successful_response after sandbox tests,
        self-eval, and deliverable gate have passed.
        """
        content = getattr(response, "content", "") or ""

        # Extract structured plan from architect's planning response
        if (task.plan is None
                and self.config.base_id == "architect"
                and task.context.get("workflow_step", get_type_str(task.type)) in (Steps.PLAN, "planning")):
            extracted = self.extract_plan_from_response(content)
            if extracted:
                task.plan = extracted
                self.logger.info(
                    f"Extracted plan from response: {len(extracted.files_to_modify)} files, "
                    f"{len(extracted.approach)} steps"
                )
                from .task_decomposer import extract_requirements_checklist
                checklist = extract_requirements_checklist(extracted)
                if checklist:
                    task.context["requirements_checklist"] = checklist
                    self.logger.info(f"Extracted {len(checklist)} requirements checklist items")
            else:
                self.logger.warning("Architect plan step completed but no PlanDocument found in response")

            rationale = self.extract_design_rationale(content)
            if rationale:
                task.context["_design_rationale"] = rationale

        # Verdict must be set before serialization
        self.set_structured_verdict(task, response)

        # Save upstream context AFTER plan extraction + verdict
        if task.context.get("workflow") or task.context.get("chain_step"):
            self.save_upstream_context(task, response)

        # Append step to chain state file
        if task.context.get("workflow") or task.context.get("chain_step"):
            self.save_step_to_chain_state(task, response, working_dir=working_dir, task_start_time=task_start_time)

        # Safety commit before marking done
        if working_dir and working_dir.exists() and self.is_implementation_step(task, self.config.base_id):
            self.git_ops.safety_commit(working_dir, f"WIP: uncommitted changes at task completion ({task.id})")

        # Mark completed
        self.logger.debug(f"Marking task {task.id} as completed")
        task.mark_completed(self.config.id)
        self.queue.mark_completed(task)
        self.logger.info(f"✅ Task {task.id} moved to completed")

        # Deterministic JIRA transition on task completion
        if self._agent_definition and self._agent_definition.jira_on_complete:
            comment = f"Task completed by {self.config.id}"
            pr_url = task.context.get("pr_url")
            if pr_url:
                comment += f"\nPR: {pr_url}"
            if sync_jira_status_cb:
                sync_jira_status_cb(task, self._agent_definition.jira_on_complete, comment=comment)

        # Transition to COMPLETING status
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.COMPLETING,
            current_task=CurrentTask(
                id=task.id,
                title=task.title,
                type=get_type_str(task.type),
                started_at=task_start_time
            ),
            last_updated=datetime.now(timezone.utc)
        ))

        # Handle routing signal
        from .routing import read_routing_signal, WORKFLOW_COMPLETE
        if routing_signal is None:
            routing_signal = read_routing_signal(self.workspace, task.id)

        if routing_signal:
            self.logger.info(
                f"Routing signal: target={routing_signal.target_agent}, "
                f"reason={routing_signal.reason}"
            )

        # __complete__ at plan step: distinguish "done planning" from "no work needed"
        if (routing_signal
                and routing_signal.target_agent == WORKFLOW_COMPLETE
                and self.config.base_id == "architect"
                and task.context.get("workflow_step", get_type_str(task.type)) in (Steps.PLAN, "planning")):
            if task.plan is not None:
                self.logger.info(
                    f"Clearing __complete__ routing signal at plan step — "
                    f"plan was extracted ({len(task.plan.files_to_modify)} files), "
                    f"proceeding to implement"
                )
                self.session_logger.log(
                    "routing_signal_cleared", task_id=task.id,
                    reason="plan_extracted_at_plan_step",
                    plan_files=len(task.plan.files_to_modify),
                )
                routing_signal = None
            else:
                self.logger.info(
                    "No plan extracted at plan step — passing __complete__ signal "
                    "to workflow router for PR lifecycle handling"
                )
                self.session_logger.log(
                    "routing_signal_passthrough", task_id=task.id,
                    reason="no_plan_at_plan_step",
                    routing_signal_reason=routing_signal.reason,
                )

        self.run_post_completion_flow(
            task, response, routing_signal, task_start_time,
            context_window_manager=context_window_manager,
            extract_and_store_memories_cb=extract_and_store_memories_cb,
            analyze_tool_patterns_cb=analyze_tool_patterns_cb,
            sync_jira_status_cb=sync_jira_status_cb,
        )
