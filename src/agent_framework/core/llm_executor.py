"""LLM execution manager.

Handles the LLM call lifecycle: interruption watching, circuit breaker
handling, completion processing, and routing decision logging.
Extracted from agent.py to reduce its size.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AgentConfig
    from .git_operations import GitOperationsManager
    from .session_logger import SessionLogger
    from .activity import ActivityManager
    from .context_window_manager import ContextWindowManager
    from ..llm.base import LLMBackend, LLMResponse
    from ..utils.rich_logging import ContextLogger

from .task import Task, TaskType
from .activity import ActivityEvent, TaskPhase
from ..llm.base import LLMRequest, LLMResponse
from ..utils.type_helpers import get_type_str
from ..workflow.constants import WorkflowStepConstants as Steps

# Circuit breaker: commands that indicate productive work
_PRODUCTIVE_PREFIXES = frozenset({
    "pytest", "python -m pytest", "python -m unittest",
    "pip", "pip3", "uv ",
    "npm", "yarn", "pnpm", "bun ",
    "make", "cmake",
    "cargo", "go test", "go build", "go vet",
    "mvn", "gradle",
    "docker", "docker-compose",
    "bandit", "pylint", "mypy", "ruff", "flake8", "black", "isort", "tox",
    "git commit", "git push", "git add", "git diff", "git log", "git stash",
    "python", "node ",
})
_PRODUCTIVE_THRESHOLD_MULTIPLIER = 3
_PRODUCTIVE_RATIO_THRESHOLD = 0.7


def is_productive_command(cmd: str) -> bool:
    """Check if a bash command is a productive tool (test/build/lint/git)."""
    cmd_stripped = cmd.strip().lower()
    return any(cmd_stripped.startswith(p) for p in _PRODUCTIVE_PREFIXES)


class LLMExecutionManager:
    """Manages LLM call execution with interruption watching and circuit breaking."""

    def __init__(
        self,
        config: "AgentConfig",
        llm: "LLMBackend",
        git_ops: "GitOperationsManager",
        logger: "ContextLogger",
        session_logger: "SessionLogger",
        activity_manager: "ActivityManager",
    ):
        self.config = config
        self.llm = llm
        self.git_ops = git_ops
        self.logger = logger
        self.session_logger = session_logger
        self.activity_manager = activity_manager

    def set_session_logger(self, session_logger: "SessionLogger") -> None:
        """Update session logger for new task."""
        self.session_logger = session_logger

    async def execute(
        self,
        task: Task,
        prompt: str,
        working_dir: Path,
        team_agents: Optional[dict],
        *,
        context_window_manager: Optional["ContextWindowManager"] = None,
        is_implementation_step: bool = False,
        max_consecutive_tool_calls: int = 15,
        max_consecutive_diagnostic_calls: int = 10,
        exploration_alert_threshold: int = 50,
        exploration_alert_thresholds: Optional[dict] = None,
        watch_for_interruption_coro=None,
        update_phase_cb=None,
        current_specialization=None,
        current_file_count: int = 0,
        optimization_config: Optional[dict] = None,
        finalize_failed_attempt_cb=None,
        read_cache_cb=None,
        cached_paths: frozenset[str] = frozenset(),
    ) -> Optional[LLMResponse]:
        """Execute LLM with interruption watching, return response or None if interrupted."""

        # Setup tool activity tracking via CheckpointManager
        from .checkpoint_manager import CheckpointManager
        _circuit_breaker_event = asyncio.Event()
        _workflow_step = task.context.get("workflow_step")
        _exploration_threshold = (
            (exploration_alert_thresholds or {}).get(_workflow_step, exploration_alert_threshold)
            if _workflow_step else exploration_alert_threshold
        )
        # Step-aware re-read threshold: implementation steps legitimately
        # re-read files during read→edit→verify cycles across many files
        _default_reread = 8 if (is_implementation_step or _workflow_step in Steps.IMPLEMENTATION_STEPS) else 3
        _reread_threshold = (optimization_config or {}).get(
            "reread_interrupt_threshold", _default_reread,
        )

        _checkpoint_mgr = CheckpointManager(
            task=task,
            working_dir=working_dir,
            is_implementation_step=is_implementation_step,
            max_consecutive_tool_calls=max_consecutive_tool_calls,
            max_consecutive_diagnostic_calls=max_consecutive_diagnostic_calls,
            exploration_threshold=_exploration_threshold,
            workflow_step=_workflow_step,
            git_ops=self.git_ops,
            session_logger=self.session_logger,
            activity_manager=self.activity_manager,
            context_window_manager=context_window_manager,
            logger=self.logger,
            circuit_breaker_event=_circuit_breaker_event,
            agent_id=self.config.id,
            agent_base_id=self.config.base_id,
            reread_threshold=_reread_threshold,
            cached_paths=cached_paths,
        )

        def _on_tool_activity(tool_name: str, tool_input_summary: Optional[str]):
            _checkpoint_mgr.on_tool_activity(tool_name, tool_input_summary)

        # Log LLM start
        if update_phase_cb:
            update_phase_cb(TaskPhase.EXECUTING_LLM)
        self.logger.phase_change("executing_llm")
        self.logger.info(
            f"🤖 Calling LLM (model: {task.type}, attempt: {task.retry_count + 1})"
        )
        self.session_logger.log(
            "llm_start",
            task_type=get_type_str(task.type),
            retry=task.retry_count,
        )

        # PREVIEW tasks get tool-level read-only enforcement
        preview_allowed_tools: list[str] | None = None
        if task.type == TaskType.PREVIEW:
            preview_allowed_tools = [
                "Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch",
            ]

        # Behavioral directives not covered by agent prompts in agents.yaml.
        # Kept concise — each rule is unique runtime guidance.
        efficiency_parts = [
            "FILE READS: (1) NEVER use offset or limit parameters on Read — always read the full file. "
            "(2) If you need a specific section, use Grep with -C context lines instead. "
            "(3) Files listed in 'FILES ANALYZED BY PREVIOUS AGENTS' are already summarized — prefer Grep over re-reading them. "
            "(4) NEVER read the same file twice — after reading, the contents are in your context. "
            "(5) If you have read 3+ files and need to recall one, use Grep to find the section. "
            "(6) Reading a file you already read WILL trigger an automatic session interrupt.",
            "COMMITS: After each deliverable, git add + commit + push immediately.",
            "CIRCUIT BREAKER: 3+ consecutive failed shell commands → stop and report.",
        ]
        if self.config.id == "architect":
            efficiency_parts.append(
                "EXPLORATION: Explore the codebase yourself with Grep/Read — do not delegate."
            )
        efficiency_directive = " ".join(efficiency_parts)

        # Compute routing signals for intelligent model selection
        _estimated_lines = 0
        if task.plan:
            from .task_decomposer import estimate_plan_lines
            _estimated_lines = estimate_plan_lines(task.plan)

        if task.context.get("_escalate_model"):
            _estimated_lines = max(_estimated_lines, 600)

        _budget_remaining_usd = None
        _budget_ceiling = task.context.get("_budget_ceiling")
        _cumulative_cost = task.context.get("_cumulative_cost", 0.0)
        if _budget_ceiling is not None:
            _budget_remaining_usd = max(0.0, _budget_ceiling - _cumulative_cost)

        # Race LLM execution against pause/stop signal watcher
        llm_coro = self.llm.complete(
            LLMRequest(
                prompt=prompt,
                task_type=task.type,
                retry_count=task.retry_count,
                context=task.context,
                working_dir=str(working_dir),
                agents=team_agents,
                specialization_profile=current_specialization.id if current_specialization else None,
                file_count=current_file_count,
                estimated_lines=_estimated_lines,
                budget_remaining_usd=_budget_remaining_usd,
                allowed_tools=preview_allowed_tools,
                append_system_prompt=efficiency_directive,
                env_vars=self.git_ops.worktree_env_vars,
            ),
            task_id=task.id,
            on_tool_activity=_on_tool_activity,
            on_session_tool_call=self.session_logger.log_tool_call,
            on_session_tool_result=self.session_logger.log_tool_result,
        )
        llm_task = asyncio.create_task(llm_coro)
        watcher_task = asyncio.create_task(watch_for_interruption_coro())
        circuit_breaker_task = asyncio.create_task(_circuit_breaker_event.wait())

        done, pending = await asyncio.wait(
            [llm_task, watcher_task, circuit_breaker_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if circuit_breaker_task in done:
            return await self._handle_circuit_breaker(
                task, working_dir, _checkpoint_mgr,
                llm_task, watcher_task,
                finalize_failed_attempt_cb=finalize_failed_attempt_cb,
                read_cache_cb=read_cache_cb,
            )

        if watcher_task in done:
            return await self._handle_interruption(
                task, working_dir, _checkpoint_mgr,
                llm_task, circuit_breaker_task,
                finalize_failed_attempt_cb=finalize_failed_attempt_cb,
                read_cache_cb=read_cache_cb,
            )

        # LLM finished first — cancel watcher and circuit breaker
        watcher_task.cancel()
        circuit_breaker_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        result = llm_task.result()
        outcome = "success" if result and result.success else "failed"
        _checkpoint_mgr.emit_subagent_summary(outcome, not (result and result.success))
        return result

    async def _handle_circuit_breaker(
        self, task, working_dir, checkpoint_mgr,
        llm_task, watcher_task, *,
        finalize_failed_attempt_cb=None,
        read_cache_cb=None,
    ) -> LLMResponse:
        """Handle circuit breaker trip — kill subprocess and return synthetic failure."""
        count = checkpoint_mgr.consecutive_bash
        diag_count = checkpoint_mgr.consecutive_diagnostic
        is_diagnostic = checkpoint_mgr.diagnostic_tripped
        is_reread = checkpoint_mgr._reread_interrupted
        commands = checkpoint_mgr.bash_commands
        unique = len(set(commands))
        diversity = unique / count if count > 0 else 0.0
        productive = sum(1 for c in commands if is_productive_command(c))
        productive_ratio = productive / count if count > 0 else 0.0
        trigger = "reread" if is_reread else ("diagnostic" if is_diagnostic else "volume")

        wd_str = str(working_dir) if working_dir else "N/A"
        wd_exists = working_dir.exists() if working_dir else None

        # Log distinct event for re-read interruptions so analytics can track them
        if is_reread:
            worst = checkpoint_mgr.get_worst_reread()
            worst_file = worst[0] if worst else "unknown"
            worst_count = worst[1] if worst else 0
            read_stats = checkpoint_mgr.get_read_stats()
            self.session_logger.log(
                "reread_interrupt",
                worst_file=worst_file,
                worst_count=worst_count,
                read_stats=read_stats,
            )
            self.logger.warning(
                f"Re-read circuit breaker tripped for task {task.id}: "
                f"{worst_file} read {worst_count} times"
            )
            event_title = (
                f"Re-read circuit breaker: {worst_file} read {worst_count} times"
            )
            error_msg = (
                f"Session interrupted: file '{worst_file}' was read {worst_count} times. "
                f"Re-read files: {read_stats}. "
                f"Read each file once; use Grep to locate sections."
            )
        elif is_diagnostic:
            self.logger.warning(
                f"Diagnostic circuit breaker tripped for task {task.id}: "
                f"{diag_count} consecutive diagnostic commands "
                f"(threshold={checkpoint_mgr._max_diagnostic})"
                f" (working_dir={wd_str}, exists={wd_exists})"
            )
            event_title = (
                f"Diagnostic circuit breaker: {diag_count} consecutive diagnostic commands"
                f" — {wd_str}"
            )
            error_msg = (
                f"Stuck agent detected: {diag_count} consecutive diagnostic commands — "
                f"working directory {wd_str} (exists={wd_exists})"
            )
        else:
            self.logger.warning(
                f"Circuit breaker tripped for task {task.id}: "
                f"{count} consecutive Bash calls (threshold={checkpoint_mgr._max_consecutive}, "
                f"diversity={diversity:.2f}, unique_commands={unique}, "
                f"productive_ratio={productive_ratio:.2f})"
                f" (working_dir={wd_str}, exists={wd_exists})"
            )
            event_title = (
                f"Circuit breaker: {count} consecutive Bash calls "
                f"(diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f})"
                f" — {wd_str}"
            )
            error_msg = (
                f"Circuit breaker tripped: {count} consecutive Bash calls without other tool types "
                f"(diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f}). "
                f"Working directory {wd_str} (exists={wd_exists})."
            )

        # Auto-commit WIP before killing
        await self.auto_commit_wip(task, working_dir, count)

        self.llm.cancel()

        # Record attempt for retry awareness
        try:
            partial = self.llm.get_partial_output()
        except Exception:
            partial = None
        if finalize_failed_attempt_cb:
            finalize_failed_attempt_cb(task, working_dir, content=partial, error=error_msg)

        # Persist reads from this failed session so retries start with context
        if read_cache_cb:
            try:
                read_cache_cb(task, working_dir)
            except Exception:
                self.logger.debug("read_cache_cb failed on circuit breaker path", exc_info=True)

        llm_task.cancel()
        watcher_task.cancel()
        for t in [llm_task, watcher_task]:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        self.session_logger.log(
            "circuit_breaker",
            trigger=trigger,
            consecutive_bash=count,
            consecutive_diagnostic=diag_count,
            threshold=checkpoint_mgr._max_diagnostic if is_diagnostic else checkpoint_mgr._max_consecutive,
            unique_commands=unique,
            diversity=round(diversity, 2),
            productive_ratio=round(productive_ratio, 2),
            working_dir=wd_str,
            working_dir_exists=wd_exists,
            last_commands=[c[:200] for c in commands[-5:]],
        )

        self.activity_manager.append_event(ActivityEvent(
            type="circuit_breaker",
            agent=self.config.id,
            task_id=task.id,
            title=event_title,
            timestamp=datetime.now(timezone.utc),
        ))
        checkpoint_mgr.emit_subagent_summary("circuit_breaker", True)
        return LLMResponse(
            content="",
            model_used="",
            input_tokens=0,
            output_tokens=0,
            finish_reason="circuit_breaker",
            latency_ms=0,
            success=False,
            error=error_msg,
        )

    async def _handle_interruption(
        self, task, working_dir, checkpoint_mgr,
        llm_task, circuit_breaker_task, *,
        finalize_failed_attempt_cb=None,
        read_cache_cb=None,
    ) -> None:
        """Handle pause/stop signal or worktree vanishing during LLM execution."""
        wt = self.git_ops.active_worktree
        worktree_gone = wt is not None and not wt.exists()

        if worktree_gone:
            error_msg = f"Worktree vanished during LLM execution: {wt}"
            event_type = "worktree_vanished"
        else:
            error_msg = "Interrupted during LLM execution"
            event_type = "interrupted"

        self.logger.info(f"{error_msg} for task {task.id}, cancelling LLM")
        self.llm.cancel()
        llm_task.cancel()
        circuit_breaker_task.cancel()
        try:
            await llm_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.debug(f"LLM task raised during cancellation: {e}")

        # Harvest partial output and record attempt
        partial = self.llm.get_partial_output()
        if finalize_failed_attempt_cb:
            finalize_failed_attempt_cb(task, working_dir, content=partial, error=error_msg)

        # Persist reads so the retry starts with context (skip if worktree gone)
        if read_cache_cb and not worktree_gone:
            try:
                read_cache_cb(task, working_dir)
            except Exception:
                self.logger.debug("read_cache_cb failed on interruption path", exc_info=True)

        if worktree_gone:
            self.session_logger.log(
                "worktree_vanished",
                path=str(wt),
                task_id=task.id,
            )

        task.last_error = error_msg
        task.reset_to_pending()
        # Caller (agent.py) handles queue.update(task) after we return None
        self.activity_manager.append_event(ActivityEvent(
            type=event_type,
            agent=self.config.id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc),
        ))
        self.logger.info(f"Task {task.id} reset to pending after {event_type}")
        checkpoint_mgr.emit_subagent_summary(event_type, True)
        return None

    async def auto_commit_wip(self, task: Task, working_dir: Path, bash_count: int) -> None:
        """Best-effort WIP commit so code isn't lost when circuit breaker trips."""
        try:
            committed = self.git_ops.safety_commit(
                working_dir,
                f"WIP: auto-save before circuit breaker ({bash_count} consecutive Bash calls)",
            )
            if committed:
                self.session_logger.log("wip_auto_commit", task_id=task.id, bash_count=bash_count)
        except Exception:
            pass

    def process_completion(self, response: "LLMResponse", task: Task, *, context_window_manager=None) -> None:
        """Log LLM completion and update context window manager."""
        self.session_logger.log(
            "llm_complete",
            success=response.success,
            model=response.model_used,
            tokens_in=response.input_tokens,
            tokens_out=response.output_tokens,
            cost=response.reported_cost_usd,
            duration_ms=response.latency_ms,
        )

        if context_window_manager:
            context_window_manager.update_token_usage(
                response.input_tokens,
                response.output_tokens
            )
            budget_status = context_window_manager.get_budget_status()
            self.logger.debug(
                f"Context budget: {budget_status['utilization_percent']:.1f}% used "
                f"({budget_status['used_so_far']}/{budget_status['total_budget']} tokens)"
            )

            self.session_logger.log(
                "context_budget_update",
                utilization_percent=budget_status["utilization_percent"],
                used_tokens=budget_status["used_so_far"],
                total_budget=budget_status["total_budget"],
                remaining=budget_status["remaining"],
            )

            if context_window_manager.should_trigger_checkpoint():
                self.logger.warning(
                    "Context budget critically low (>90% used). "
                    "Consider splitting task into subtasks."
                )
                self.activity_manager.append_event(ActivityEvent(
                    type="context_budget_critical",
                    agent=self.config.id,
                    task_id=task.id,
                    title="Context budget >90%: consider task splitting",
                    timestamp=datetime.now(timezone.utc)
                ))
                self.session_logger.log(
                    "context_budget_critical",
                    task_id=task.id,
                    utilization_percent=budget_status["utilization_percent"],
                )

        # Clear tool activity after LLM completes
        try:
            self.activity_manager.update_tool_activity(self.config.id, None)
        except Exception:
            pass

    def log_routing_decision(self, task: Task, response: "LLMResponse") -> None:
        """Log intelligent routing decision if one was made."""
        backend = self.llm
        selector = getattr(backend, 'model_selector', None)
        decision = getattr(selector, '_last_routing_decision', None) if selector else None
        if decision is None:
            return
        selector._last_routing_decision = None
        self.session_logger.log(
            "model_routing_decision",
            chosen_tier=decision.chosen_tier,
            scores=decision.scores,
            signals=decision.signals,
            fallback=decision.fallback,
            model_used=response.model_used,
        )
        self.activity_manager.append_event(ActivityEvent(
            type="model_routing_decision",
            agent=self.config.id,
            task_id=task.id,
            title=f"Intelligent routing: {decision.chosen_tier} (scores: {decision.scores})",
            timestamp=datetime.now(timezone.utc),
        ))
