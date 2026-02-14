"""Structured session logging for post-hoc analysis of agent decisions.

Writes append-only JSONL files to logs/sessions/{task_id}.jsonl.
Each line is a timestamped event capturing the framework's decisions
around LLM calls — prompts sent, tool parameters, retries, memory ops, etc.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionLogger:
    """Append-only JSONL session log per task.

    One SessionLogger instance per task execution. Events are flushed
    immediately so logs survive crashes.
    """

    def __init__(
        self,
        logs_dir: Path,
        task_id: str,
        enabled: bool = True,
        log_prompts: bool = True,
        log_tool_inputs: bool = True,
    ):
        self._enabled = enabled
        self._log_prompts = log_prompts
        self._log_tool_inputs = log_tool_inputs
        self._task_id = task_id
        self._path = logs_dir / "sessions" / f"{task_id}.jsonl"
        self._file = None
        self._sequence = 0

    def log(self, event: str, **data: Any) -> None:
        """Append a single event line."""
        if not self._enabled:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "task_id": self._task_id,
            **data,
        }
        try:
            self._ensure_open()
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()
        except Exception as e:
            logger.debug(f"Session log write failed (non-fatal): {e}")

    def log_tool_call(self, tool_name: str, tool_input: Optional[dict] = None) -> None:
        """Log a tool call with optional input parameters."""
        if not self._enabled:
            return
        self._sequence += 1
        data = {"tool": tool_name, "sequence": self._sequence}
        if self._log_tool_inputs and tool_input:
            data["input"] = tool_input
        self.log("tool_call", **data)

    def log_prompt(self, prompt: str, **extra: Any) -> None:
        """Log prompt content (respects log_prompts config flag)."""
        if not self._enabled or not self._log_prompts:
            return
        self.log("prompt_content", prompt=prompt, **extra)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a")

    @staticmethod
    def cleanup_old_sessions(logs_dir: Path, retention_days: int) -> int:
        """Delete session logs older than retention_days. Returns count removed."""
        sessions_dir = logs_dir / "sessions"
        if not sessions_dir.exists():
            return 0

        cutoff = time.time() - (retention_days * 86400)
        removed = 0
        for f in sessions_dir.glob("*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Cleaned up {removed} session logs older than {retention_days} days")
        return removed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Singleton no-op instance to avoid None checks everywhere
_NOOP = None


def noop_logger() -> "SessionLogger":
    """Return a disabled SessionLogger (avoids None checks at call sites)."""
    global _NOOP
    if _NOOP is None:
        _NOOP = SessionLogger(Path("/dev/null"), "noop", enabled=False)
    return _NOOP
