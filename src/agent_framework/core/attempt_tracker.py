"""Attempt tracker — persists what each retry attempt accomplished.

When a task fails and retries, this module records the attempt's git state
(branch, commit SHA, files modified, push status) to disk so the next attempt
can recover code and context. Follows the chain_state.py pattern: dataclass
models, free functions, atomic writes.

Persistence path: `.agent-communication/attempt-history/{task_id}.json`
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.atomic_io import atomic_write_text
from ..utils.subprocess_utils import run_git_command

logger = logging.getLogger(__name__)

# Cap rendered prompt context to avoid bloating retry prompts
ATTEMPT_HISTORY_MAX_PROMPT_CHARS = 4000


@dataclass
class AttemptRecord:
    """A single failed attempt's outcome."""

    attempt_number: int
    started_at: str                   # ISO timestamp
    ended_at: Optional[str] = None
    agent_id: str = ""
    workflow_step: Optional[str] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None  # HEAD SHA at end of attempt
    pushed: bool = False
    files_modified: List[str] = field(default_factory=list)
    commit_count: int = 0
    insertions: int = 0
    deletions: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    # Per-attempt cost visibility
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass
class AttemptHistory:
    """Accumulated attempt records for a task."""

    task_id: str
    attempts: List[AttemptRecord] = field(default_factory=list)


# --- Disk I/O ---

def _history_dir(workspace: Path) -> Path:
    return workspace / ".agent-communication" / "attempt-history"


def _history_path(workspace: Path, task_id: str) -> Path:
    return _history_dir(workspace) / f"{task_id}.json"


def load_attempt_history(workspace: Path, task_id: str) -> Optional[AttemptHistory]:
    """Load attempt history from disk. Returns None if missing or corrupt."""
    path = _history_path(workspace, task_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load attempt history for {task_id}: {e}")
        return None

    if not isinstance(data, dict) or "task_id" not in data:
        return None

    # Tolerate schema evolution — filter to known fields
    known_fields = set(AttemptRecord.__dataclass_fields__)
    attempts = [
        AttemptRecord(**{k: v for k, v in a.items() if k in known_fields})
        for a in data.get("attempts", [])
    ]

    return AttemptHistory(task_id=data["task_id"], attempts=attempts)


def save_attempt_history(workspace: Path, history: AttemptHistory) -> None:
    """Atomically write attempt history to disk."""
    hist_dir = _history_dir(workspace)
    hist_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "task_id": history.task_id,
        "attempts": [asdict(a) for a in history.attempts],
    }

    path = _history_path(workspace, history.task_id)
    atomic_write_text(path, json.dumps(data, indent=2, default=str))


# --- Core entry point ---

def record_attempt(
    workspace: Path,
    task: "Task",
    agent_id: str,
    working_dir: Optional[Path],
    *,
    error: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[AttemptRecord]:
    """Record attempt outcome: commit WIP, push, collect stats, persist to disk.

    Single entry point for all failure paths — replaces scattered git evidence
    capture across _handle_failed_response, exception handler, and interruption
    handler.

    Non-fatal: returns None on any error so callers don't need try/except.
    """
    _log = logger or logging.getLogger(__name__)

    try:
        if not working_dir or not working_dir.exists():
            _log.debug("record_attempt: no working directory, skipping")
            return None

        # Guard: don't record on main/master or detached HEAD
        branch = _get_current_branch(working_dir)
        if branch is None or branch in ("main", "master", "HEAD"):
            _log.debug(f"record_attempt: on {branch}, skipping")
            return None

        # Commit uncommitted work (idempotent with safety_commit)
        had_uncommitted = _commit_wip(working_dir, task.id, _log)

        # Push to origin so code survives worktree deletion
        pushed = _try_push_branch(working_dir, _log)

        # Collect stats from committed work
        commit_sha = _get_head_sha(working_dir)
        stats = _collect_branch_stats(working_dir)

        record = AttemptRecord(
            attempt_number=task.retry_count + 1,
            started_at=(task.started_at or datetime.now(timezone.utc)).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id,
            workflow_step=task.context.get("workflow_step"),
            branch=branch,
            commit_sha=commit_sha,
            pushed=pushed,
            files_modified=stats.get("files", []),
            commit_count=stats.get("commit_count", 0),
            insertions=stats.get("insertions", 0),
            deletions=stats.get("deletions", 0),
            error=_truncate_error(error),
            error_type=_classify_error(error),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

        # Persist
        history = load_attempt_history(workspace, task.id) or AttemptHistory(task_id=task.id)
        history.attempts.append(record)
        save_attempt_history(workspace, history)

        _log.info(
            f"Recorded attempt {record.attempt_number}: "
            f"branch={branch}, sha={commit_sha or 'none'}, "
            f"pushed={pushed}, commits={record.commit_count}, "
            f"lines={record.insertions}+/{record.deletions}-"
        )

        return record

    except Exception as e:
        _log.debug(f"record_attempt failed (non-fatal): {e}")
        return None


# --- Prompt rendering ---

def render_for_retry(workspace: Path, task_id: str) -> str:
    """Render attempt history as markdown for the retry prompt.

    Returns empty string if no history exists.
    """
    history = load_attempt_history(workspace, task_id)
    if not history or not history.attempts:
        return ""

    lines = ["### Previous Attempt History"]

    for a in history.attempts:
        parts = [f"Attempt {a.attempt_number}:"]
        if a.branch:
            parts.append(f"branch={a.branch}")
        if a.commit_count:
            parts.append(f"{a.commit_count} commits ({a.insertions}+/{a.deletions}-)")
        parts.append(f"pushed={a.pushed}")
        if a.cost_usd is not None:
            parts.append(f"cost=${a.cost_usd:.2f}")
        lines.append("  ".join(parts))

        if a.error:
            lines.append(f"  Error: {a.error}")

        if a.files_modified:
            display = a.files_modified[:20]
            lines.append(f"  Files: {', '.join(display)}")
            if len(a.files_modified) > 20:
                lines.append(f"  ... and {len(a.files_modified) - 20} more")

    # Directive: point LLM at the most recent pushed branch
    last_pushed = _find_last_pushed(history)
    if last_pushed and last_pushed.branch:
        lines.append("")
        lines.append(
            f"Your previous code is on branch `{last_pushed.branch}`."
        )
        lines.append(
            "Run `git log --oneline` and `git diff origin/main..HEAD` "
            "to review before writing new code."
        )

    rendered = "\n".join(lines)
    if len(rendered) > ATTEMPT_HISTORY_MAX_PROMPT_CHARS:
        rendered = rendered[:ATTEMPT_HISTORY_MAX_PROMPT_CHARS] + "\n[attempt history truncated]"
    return rendered


def get_last_pushed_branch(workspace: Path, task_id: str) -> Optional[str]:
    """Return the branch name from the most recent pushed attempt, or None."""
    history = load_attempt_history(workspace, task_id)
    if not history:
        return None
    last = _find_last_pushed(history)
    return last.branch if last else None


# --- Internal helpers ---

def _find_last_pushed(history: AttemptHistory) -> Optional[AttemptRecord]:
    """Find the most recent attempt that was successfully pushed."""
    for a in reversed(history.attempts):
        if a.pushed and a.branch:
            return a
    return None


def _get_current_branch(working_dir: Path) -> Optional[str]:
    """Get current branch name, or None on error."""
    try:
        result = run_git_command(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=working_dir, check=False, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_head_sha(working_dir: Path) -> Optional[str]:
    """Get short HEAD SHA."""
    try:
        result = run_git_command(
            ["rev-parse", "--short", "HEAD"],
            cwd=working_dir, check=False, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _commit_wip(working_dir: Path, task_id: str, log: logging.Logger) -> bool:
    """Commit any uncommitted changes. Returns True if committed."""
    try:
        status = run_git_command(
            ["status", "--porcelain"],
            cwd=working_dir, check=False, timeout=10,
        )
        if status.returncode != 0 or not status.stdout.strip():
            return False

        run_git_command(["add", "-A"], cwd=working_dir, check=False, timeout=10)
        result = run_git_command(
            ["commit", "-m", f"[auto-commit] WIP: auto-save on failure ({task_id})"],
            cwd=working_dir, check=False, timeout=10,
        )
        if result.returncode == 0:
            log.info("Auto-committed uncommitted changes before recording attempt")
            return True
    except Exception as e:
        log.debug(f"WIP commit failed (non-fatal): {e}")
    return False


def _try_push_branch(working_dir: Path, log: logging.Logger) -> bool:
    """Push current branch to origin. Returns True on success."""
    try:
        branch = _get_current_branch(working_dir)
        if not branch or branch in ("main", "master", "HEAD"):
            return False

        result = run_git_command(
            ["push", "origin", branch, "--force-with-lease"],
            cwd=working_dir, check=False, timeout=30,
        )
        if result.returncode == 0:
            log.info(f"Pushed branch {branch} to origin")
            return True
        else:
            log.debug(f"Push failed: {result.stderr}")
    except Exception as e:
        log.debug(f"Push failed (non-fatal): {e}")
    return False


def _collect_branch_stats(working_dir: Path) -> Dict:
    """Collect commit count, insertions, deletions, and file list vs origin default."""
    result: Dict = {"commit_count": 0, "insertions": 0, "deletions": 0, "files": []}

    try:
        # Detect default branch
        default = "main"
        for candidate in ["main", "master"]:
            probe = run_git_command(
                ["rev-parse", "--verify", f"origin/{candidate}"],
                cwd=working_dir, check=False, timeout=10,
            )
            if probe.returncode == 0:
                default = candidate
                break

        range_spec = f"origin/{default}..HEAD"

        # Commit count
        log_result = run_git_command(
            ["rev-list", "--count", range_spec],
            cwd=working_dir, check=False, timeout=10,
        )
        if log_result.returncode == 0:
            result["commit_count"] = int(log_result.stdout.strip())

        # Diffstat for insertions/deletions
        stat_result = run_git_command(
            ["diff", "--stat", range_spec],
            cwd=working_dir, check=False, timeout=10,
        )
        if stat_result.returncode == 0 and stat_result.stdout.strip():
            summary_line = stat_result.stdout.strip().split("\n")[-1]
            ins_match = re.search(r"(\d+) insertion", summary_line)
            del_match = re.search(r"(\d+) deletion", summary_line)
            if ins_match:
                result["insertions"] = int(ins_match.group(1))
            if del_match:
                result["deletions"] = int(del_match.group(1))

        # File list
        names_result = run_git_command(
            ["diff", "--name-only", range_spec],
            cwd=working_dir, check=False, timeout=10,
        )
        if names_result.returncode == 0 and names_result.stdout.strip():
            result["files"] = sorted(set(
                f for f in names_result.stdout.strip().split("\n") if f
            ))[:50]

    except Exception:
        pass

    return result


def _truncate_error(error: Optional[str], max_len: int = 500) -> Optional[str]:
    """Truncate error message for storage."""
    if not error:
        return None
    if len(error) <= max_len:
        return error
    return error[:max_len] + "..."


def _classify_error(error: Optional[str]) -> Optional[str]:
    """Classify error into a category for analytics."""
    if not error:
        return None
    lower = error.lower()
    if "circuit breaker" in lower:
        return "circuit_breaker"
    if "context" in lower and ("exhausted" in lower or "window" in lower):
        return "context_exhausted"
    if "interrupt" in lower:
        return "interrupted"
    if "timeout" in lower:
        return "timeout"
    return "other"
