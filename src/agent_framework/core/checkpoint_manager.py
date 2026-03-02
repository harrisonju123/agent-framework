"""Checkpoint and circuit breaker management during LLM execution.

Extracted from the _on_tool_activity closure in agent.py. Tracks tool calls,
detects stuck agents via diversity heuristics, triggers periodic commits,
and fires a circuit breaker event when the agent is unproductive.
"""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .activity import ActivityEvent, ToolActivity


def _load_diagnostic_commands():
    """Import diagnostic command set from agent.py to stay in sync."""
    try:
        from .agent import _DIAGNOSTIC_PREFIXES
        return _DIAGNOSTIC_PREFIXES
    except ImportError:
        # Fallback if circular import
        return frozenset({
            "pwd", "echo", "test", "[", "cd",
            "ls", "find", "stat", "file",
            "readlink", "realpath",
            "which", "type", "env", "printenv",
            "whoami", "id", "uname", "hostname",
        })


_DIAGNOSTIC_COMMANDS = None  # lazy-loaded

# Productive threshold: if this fraction of commands are productive, use a
# higher ceiling before tripping the breaker
_PRODUCTIVE_RATIO_THRESHOLD = 0.7
_PRODUCTIVE_THRESHOLD_MULTIPLIER = 3


def _is_diagnostic_command(summary: str) -> bool:
    """Check if a Bash command summary looks like a diagnostic probe."""
    global _DIAGNOSTIC_COMMANDS
    if _DIAGNOSTIC_COMMANDS is None:
        _DIAGNOSTIC_COMMANDS = _load_diagnostic_commands()
    if not summary:
        return False
    stripped = summary.strip()
    token = stripped.split()[0] if stripped else ""
    bare = token.split("/")[-1].lower()
    return bare in _DIAGNOSTIC_COMMANDS


def _is_productive_command(summary: str) -> bool:
    """Check if a Bash command summary looks like productive work."""
    if not summary:
        return False
    lower = summary.lower()
    return any(kw in lower for kw in (
        "git commit", "git push", "git add", "pytest", "npm", "make",
        "pip install", "cargo", "go build", "go test",
    ))


class CheckpointManager:
    """Tracks tool activity during LLM execution and fires circuit breaker when stuck.

    Encapsulates the mutable state that was previously in closure variables
    (_tool_call_count[0], _consecutive_bash[0], etc.).
    """

    def __init__(
        self,
        *,
        task,
        working_dir: Optional[Path],
        is_implementation_step: bool,
        max_consecutive_tool_calls: int,
        max_consecutive_diagnostic_calls: int,
        exploration_threshold: int,
        workflow_step: Optional[str],
        git_ops,
        session_logger,
        activity_manager,
        context_window_manager,
        logger,
        circuit_breaker_event,
        agent_id: str = "",
        agent_base_id: str = "",
        diversity_threshold: float = 0.5,
    ):
        self.task = task
        self.working_dir = working_dir
        self._is_impl = is_implementation_step
        self._max_consecutive = max_consecutive_tool_calls
        self._max_diagnostic = max_consecutive_diagnostic_calls
        self._exploration_threshold = exploration_threshold
        self._workflow_step = workflow_step
        self._git_ops = git_ops
        self._session_logger = session_logger
        self._activity_manager = activity_manager
        self._cwm = context_window_manager
        self._logger = logger
        self._circuit_breaker_event = circuit_breaker_event
        self._agent_id = agent_id
        self._agent_base_id = agent_base_id
        self._diversity_threshold = diversity_threshold

        # Checkpoint interval: tighter for impl steps to reduce max work loss
        self.checkpoint_interval = 8 if is_implementation_step else 25

        # Mutable state
        self.tool_call_count = 0
        self.consecutive_bash = 0
        self.consecutive_diagnostic = 0
        self.diagnostic_tripped = False
        self.bash_commands: list[str] = []
        self.soft_threshold_logged = False
        self.exploration_alerted = False
        self.subagent_spawns: list[dict] = []
        self._last_write_time = 0.0

    def on_tool_activity(self, tool_name: str, tool_input_summary: Optional[str]) -> None:
        """Called on each tool invocation during LLM execution."""
        try:
            self.tool_call_count += 1

            if tool_name == "Task":
                self.subagent_spawns.append({
                    "summary": tool_input_summary,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                self._session_logger.log(
                    "subagent_spawned",
                    spawn_index=len(self.subagent_spawns),
                    tool_input_summary=tool_input_summary,
                )

            self._check_circuit_breaker(tool_name, tool_input_summary)
            self._check_exploration_alert()
            self._check_periodic_checkpoint()
            self._update_activity_display(tool_name, tool_input_summary)

        except Exception as e:
            self._logger.debug(f"Tool activity tracking error (non-fatal): {e}")

    def _check_circuit_breaker(self, tool_name: str, summary: Optional[str]) -> None:
        """Track consecutive Bash calls and trip breaker on stuck patterns."""
        if tool_name != "Bash":
            self.consecutive_bash = 0
            self.consecutive_diagnostic = 0
            self.bash_commands = []
            self.soft_threshold_logged = False
            return

        self.consecutive_bash += 1
        self.bash_commands.append(summary or "")
        count = self.consecutive_bash

        # Diagnostic sub-breaker: catches stuck agents probing environment
        if _is_diagnostic_command(summary or ""):
            self.consecutive_diagnostic += 1
            if self.consecutive_diagnostic >= self._max_diagnostic:
                self.diagnostic_tripped = True
                self._circuit_breaker_event.set()
                return
        else:
            self.consecutive_diagnostic = 0

        if count < self._max_consecutive:
            return

        unique = len(set(self.bash_commands))
        diversity = unique / count if count > 0 else 0.0

        if diversity <= self._diversity_threshold:
            productive = sum(1 for c in self.bash_commands if _is_productive_command(c))
            productive_ratio = productive / count

            if productive_ratio > _PRODUCTIVE_RATIO_THRESHOLD:
                effective = self._max_consecutive * _PRODUCTIVE_THRESHOLD_MULTIPLIER
                if count >= effective:
                    if self._has_uncommitted_work():
                        self._logger.warning(
                            f"Circuit breaker suppressed: {count} consecutive Bash calls "
                            f"exceeded hard ceiling={effective}, but uncommitted work exists"
                        )
                    else:
                        self._logger.warning(
                            f"Circuit breaker: {count} consecutive Bash calls, "
                            f"low diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f} "
                            f"exceeded hard ceiling={effective}"
                        )
                        self._circuit_breaker_event.set()
                elif not self.soft_threshold_logged:
                    self._logger.info(
                        f"Circuit breaker deferred: {count} consecutive Bash calls, "
                        f"productive_ratio={productive_ratio:.2f} (effective threshold={effective})"
                    )
                    self.soft_threshold_logged = True
            else:
                if self._has_uncommitted_work():
                    self._logger.warning(
                        f"Circuit breaker suppressed: {count} consecutive Bash calls, "
                        f"low diversity={diversity:.2f}, "
                        f"productive_ratio={productive_ratio:.2f}, but uncommitted work exists"
                    )
                else:
                    self._logger.warning(
                        f"Circuit breaker: {count} consecutive Bash calls, "
                        f"low diversity={diversity:.2f}, productive_ratio={productive_ratio:.2f}"
                    )
                    self._circuit_breaker_event.set()
        elif not self.soft_threshold_logged:
            self._logger.info(
                f"Circuit breaker deferred: {count} consecutive Bash calls "
                f"but diversity={diversity:.2f} (unique_commands={unique})"
            )
            self.soft_threshold_logged = True

    def _check_exploration_alert(self) -> None:
        """One-time alert when total tool calls exceed threshold."""
        total = self.tool_call_count
        if total >= self._exploration_threshold and not self.exploration_alerted:
            self.exploration_alerted = True
            self._logger.info(
                f"Exploration alert: {total} tool calls in session "
                f"(threshold={self._exploration_threshold}, step={self._workflow_step or 'standalone'})"
            )
            self._session_logger.log(
                "exploration_alert",
                total_tool_calls=total,
                threshold=self._exploration_threshold,
                workflow_step=self._workflow_step,
                agent_type=self._agent_base_id,
            )
            self._activity_manager.append_event(ActivityEvent(
                type="exploration_alert",
                agent=self._agent_id,
                task_id=self.task.id,
                title=(
                    f"Exploration: {total} calls "
                    f"(threshold={self._exploration_threshold}, step={self._workflow_step or 'standalone'})"
                ),
                timestamp=datetime.now(timezone.utc),
            ))

    def _check_periodic_checkpoint(self) -> None:
        """Commit + push at regular intervals. Detect progress stalls."""
        if (self.tool_call_count % self.checkpoint_interval != 0
                or not self.working_dir or not self.working_dir.exists()):
            return

        committed = self._git_ops.safety_commit(
            self.working_dir,
            f"WIP: periodic checkpoint (tool call {self.tool_call_count})",
        )
        if committed:
            self._git_ops.push_if_unpushed()

        # Progress stall: high context usage + no commits = stuck agent
        if (self._is_impl
                and self.tool_call_count >= self.checkpoint_interval * 3
                and self._cwm):
            util = self._cwm.budget.utilization_percent
            if util > 75 and not committed and not self._git_ops._has_unpushed_commits(self.working_dir):
                self._logger.warning(
                    f"Progress stall: {util:.0f}% context, no commits after "
                    f"{self.tool_call_count} tool calls"
                )
                self._session_logger.log(
                    "progress_stall",
                    context_utilization=util,
                    tool_calls=self.tool_call_count,
                )
                self._circuit_breaker_event.set()

    def _update_activity_display(self, tool_name: str, summary: Optional[str]) -> None:
        """Throttled activity update for the status display."""
        now = time.time()
        if now - self._last_write_time < 1.0:
            return
        self._last_write_time = now
        ta = ToolActivity(
            tool_name=tool_name,
            tool_input_summary=summary,
            started_at=datetime.now(timezone.utc),
            tool_call_count=self.tool_call_count,
        )
        self._activity_manager.update_tool_activity(self._agent_id, ta)

    def _has_uncommitted_work(self) -> bool:
        """Check if working directory has uncommitted changes."""
        if not self.working_dir:
            return False
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.working_dir),
                capture_output=True, text=True, timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def emit_subagent_summary(self, outcome: str, orphan_risk: bool) -> None:
        """Log subagent spawn summary at end of LLM execution."""
        if self.subagent_spawns:
            self._session_logger.log(
                "subagent_summary",
                total_spawned=len(self.subagent_spawns),
                session_outcome=outcome,
                orphan_risk=orphan_risk,
                spawns=self.subagent_spawns,
            )
