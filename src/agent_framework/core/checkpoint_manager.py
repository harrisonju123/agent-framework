"""Checkpoint and circuit breaker management during LLM execution.

Extracted from the _on_tool_activity closure in agent.py. Tracks tool calls,
detects stuck agents via diversity heuristics, triggers periodic commits,
and fires a circuit breaker event when the agent is unproductive.
"""

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .activity import ActivityEvent, ToolActivity


_DIAGNOSTIC_COMMANDS = None  # lazy-loaded from agent.py

# Window size for chunked read detection — if the same file was read within
# this many tool calls, the second read is likely an offset continuation
_CHUNKED_READ_WINDOW = 5


@dataclass
class FileReadInfo:
    """Per-file read tracking state."""
    count: int = 0
    first_seq: int = 0
    last_seq: int = 0
    was_full_read: bool = False  # True if read without offset/limit
    chunked_sequences: list[int] = field(default_factory=list)  # sequential offset reads

# Productive threshold: if this fraction of commands are productive, use a
# higher ceiling before tripping the breaker
_PRODUCTIVE_RATIO_THRESHOLD = 0.7
_PRODUCTIVE_THRESHOLD_MULTIPLIER = 3


def _is_diagnostic_command(summary: str) -> bool:
    """Check if a Bash command summary looks like a diagnostic probe."""
    global _DIAGNOSTIC_COMMANDS
    if _DIAGNOSTIC_COMMANDS is None:
        from .agent import _DIAGNOSTIC_PREFIXES
        _DIAGNOSTIC_COMMANDS = _DIAGNOSTIC_PREFIXES
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
        reread_threshold: int = 3,
        escalation_multipliers: Optional[list[float]] = None,
        cached_paths: frozenset[str] = frozenset(),
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
        self.subagent_spawns: list[dict] = []
        self._last_write_time = 0.0

        # Read tracking — detects redundant file re-reads so we can intervene
        self._file_reads: dict[str, FileReadInfo] = {}
        self._reread_threshold: int = reread_threshold
        self._reread_interrupted: bool = False
        # Cached files get a free first read (count starts at 0 instead of 1)
        # so the circuit breaker doesn't penalize files already in the prompt.
        # Store last-3 segments for suffix matching against abbreviated tool summaries.
        self._cached_suffixes: tuple[str, ...] = tuple(
            "/".join(p.replace("\\", "/").split("/")[-3:])
            for p in cached_paths if p
        )

        # Escalating exploration levels replace the old boolean flag.
        # Level 0 = normal, 1 = alert, 2 = wrap-up warning, 3 = force halt
        self._exploration_level: int = 0
        self._escalation_multipliers: list[float] = list(
            escalation_multipliers if escalation_multipliers is not None else [1.0, 2.0, 3.0]
        )

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

            if tool_name == "Read":
                self._track_file_read(tool_input_summary)

            self._check_circuit_breaker(tool_name, tool_input_summary)
            self._check_exploration_alert()
            self._check_periodic_checkpoint()
            self._update_activity_display(tool_name, tool_input_summary)

        except Exception as e:
            self._logger.debug(f"Tool activity tracking error (non-fatal): {e}")

    def _track_file_read(self, tool_input_summary: Optional[str]) -> None:
        """Track per-file read counts and detect re-read anti-patterns.

        tool_input_summary for Read tools is the abbreviated file path
        (last 3 segments), e.g. "agent_framework/core/checkpoint_manager.py".
        """
        if not tool_input_summary:
            return

        path = tool_input_summary.strip()
        if not path:
            return
        seq = self.tool_call_count
        info = self._file_reads.get(path)

        if info is None:
            # Tool summaries and cache paths may differ in prefix length;
            # require match at a "/" boundary to avoid "foobar.py" matching "bar.py"
            is_cached = any(
                s == path or s.endswith("/" + path) or path.endswith("/" + s)
                for s in self._cached_suffixes
            )
            initial_count = 0 if is_cached else 1
            info = FileReadInfo(count=initial_count, first_seq=seq, last_seq=seq, was_full_read=True)
            self._file_reads[path] = info
            return

        info.count += 1
        prev_seq = info.last_seq
        info.last_seq = seq

        # Detect chunked reads: same file read again within a small window
        if (seq - prev_seq) <= _CHUNKED_READ_WINDOW:
            info.chunked_sequences.append(seq)

        # Trigger interrupt flag when a single file hits the re-read threshold
        if info.count >= self._reread_threshold and not self._reread_interrupted:
            self._reread_interrupted = True
            self._session_logger.log(
                "reread_threshold_exceeded",
                file=path,
                count=info.count,
                threshold=self._reread_threshold,
            )
            self._logger.info(
                f"Re-read threshold exceeded: {path} read {info.count} times "
                f"(threshold={self._reread_threshold})"
            )
            # Interrupt the LLM session — the flag alone isn't enough
            self._circuit_breaker_event.set()

    def get_read_stats(self) -> dict[str, int]:
        """Return {file_path: read_count} for all files read 2+ times."""
        return {
            path: info.count
            for path, info in self._file_reads.items()
            if info.count >= 2
        }

    def get_worst_reread(self) -> tuple[str, int] | None:
        """Return (file, count) of the most re-read file, or None if no reads."""
        if not self._file_reads:
            return None
        worst_path = max(self._file_reads, key=lambda p: self._file_reads[p].count)
        worst_count = self._file_reads[worst_path].count
        if worst_count < 1:
            return None
        return (worst_path, worst_count)

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
        """Progressive escalation when tool calls exceed multiples of the threshold.

        Level 1 (1x threshold): Log alert + emit activity event.
        Level 2 (2x threshold): Stronger wrap-up warning for prompt injection by T2.
        Level 3 (3x threshold): Force commit + trip circuit breaker to halt execution.
        """
        total = self.tool_call_count

        # Walk through escalation levels we haven't fired yet
        for level_idx, multiplier in enumerate(self._escalation_multipliers):
            target_level = level_idx + 1
            if target_level <= self._exploration_level:
                continue  # already fired this level
            trigger_at = int(self._exploration_threshold * multiplier)
            if total < trigger_at:
                break  # not yet at this level

            self._exploration_level = target_level

            if target_level == 1:
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
                    level=1,
                )
                self._activity_manager.append_event(ActivityEvent(
                    type="exploration_alert",
                    agent=self._agent_id,
                    task_id=self.task.id,
                    title=(
                        f"Exploration: {total} calls "
                        f"(threshold={self._exploration_threshold}, "
                        f"step={self._workflow_step or 'standalone'})"
                    ),
                    timestamp=datetime.now(timezone.utc),
                ))

            elif target_level == 2:
                self._logger.warning(
                    f"WRAP UP: {total} tool calls exceeded 2x threshold "
                    f"({trigger_at}), step={self._workflow_step or 'standalone'}"
                )
                self._session_logger.log(
                    "exploration_escalation",
                    total_tool_calls=total,
                    trigger_at=trigger_at,
                    level=2,
                    workflow_step=self._workflow_step,
                    agent_type=self._agent_base_id,
                )
                # Reuse exploration_alert event type — ActivityEvent Literal
                # doesn't include escalation types; differentiate by title
                self._activity_manager.append_event(ActivityEvent(
                    type="exploration_alert",
                    agent=self._agent_id,
                    task_id=self.task.id,
                    title=f"WRAP UP: {total} calls (2x threshold={trigger_at})",
                    timestamp=datetime.now(timezone.utc),
                ))

            elif target_level >= 3:
                self._logger.warning(
                    f"Exploration force halt: {total} tool calls exceeded 3x threshold "
                    f"({trigger_at}), forcing commit and halting"
                )
                self._session_logger.log(
                    "exploration_force_halt",
                    total_tool_calls=total,
                    trigger_at=trigger_at,
                    level=3,
                    workflow_step=self._workflow_step,
                    agent_type=self._agent_base_id,
                )
                self._activity_manager.append_event(ActivityEvent(
                    type="circuit_breaker",
                    agent=self._agent_id,
                    task_id=self.task.id,
                    title=f"FORCE HALT: {total} calls (3x threshold={trigger_at})",
                    timestamp=datetime.now(timezone.utc),
                ))
                # Commit whatever we have before halting
                if self.working_dir and self.working_dir.exists():
                    self._git_ops.safety_commit(
                        self.working_dir,
                        f"WIP: force halt at {total} tool calls (exploration limit)",
                    )
                self._circuit_breaker_event.set()
                break  # no point checking further levels

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
