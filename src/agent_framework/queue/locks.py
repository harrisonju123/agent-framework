"""Atomic file locking using mkdir (ported from Bash system)."""

import os
import signal
from pathlib import Path
from typing import Optional


class FileLock:
    """
    Atomic file lock using mkdir.

    Ported from scripts/async-agent-runner.sh:
    - Uses mkdir which is atomic on most filesystems
    - Stores PID for stale lock detection
    - Checks if process is still alive before claiming stale locks
    """

    def __init__(self, lock_dir: Path, task_id: str):
        self.lock_dir = lock_dir
        self.task_id = task_id
        self.lock_path = lock_dir / f"{task_id}.lock"
        self.pid_file = self.lock_path / "pid"
        self._acquired = False

    def acquire(self) -> bool:
        """
        Attempt to acquire the lock.

        Returns True if lock acquired, False otherwise.
        """
        # Check if lock exists and is stale
        if self.lock_path.exists():
            if self._is_stale_lock():
                self._remove_lock()
            else:
                return False

        # Try to acquire lock (mkdir is atomic)
        try:
            self.lock_path.mkdir(parents=True, exist_ok=False)
            self.pid_file.write_text(str(os.getpid()))
            self._acquired = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._acquired and self.lock_path.exists():
            self._remove_lock()
            self._acquired = False

    def _is_stale_lock(self) -> bool:
        """Check if lock is stale (process no longer exists)."""
        if not self.pid_file.exists():
            return True

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process is still running
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            return True

    def _remove_lock(self) -> None:
        """Remove the lock directory."""
        if self.lock_path.exists():
            if self.pid_file.exists():
                self.pid_file.unlink()
            self.lock_path.rmdir()

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock for {self.task_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
