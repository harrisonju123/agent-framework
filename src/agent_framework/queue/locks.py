"""Atomic file locking using mkdir (ported from Bash system)."""

import logging
import os
import shutil
import signal
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
                logger.info(f"Removing stale lock for {self.task_id}")
                self._remove_lock()
            else:
                logger.debug(f"Lock for {self.task_id} is held by another process")
                return False

        # Try to acquire lock (mkdir is atomic)
        try:
            self.lock_path.mkdir(parents=True, exist_ok=False)
            self.pid_file.write_text(str(os.getpid()))
            self._acquired = True
            logger.debug(f"Acquired lock for {self.task_id} (PID: {os.getpid()})")
            return True
        except FileExistsError:
            logger.debug(f"Lock for {self.task_id} already exists (race condition)")
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._acquired and self.lock_path.exists():
            self._remove_lock()
            self._acquired = False

    def _is_stale_lock(self) -> bool:
        """Check if lock is stale (process no longer exists)."""
        if not self.pid_file.exists():
            logger.warning(f"Lock for {self.task_id} has no PID file (stale)")
            return True

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process is still running
            os.kill(pid, 0)
            logger.debug(f"Lock for {self.task_id} is held by active PID {pid}")
            return False
        except ValueError:
            # Invalid PID in file - treat as stale
            logger.warning(f"Lock for {self.task_id} has invalid PID (stale)")
            return True
        except ProcessLookupError:
            # Process doesn't exist - definitely stale
            logger.warning(f"Lock for {self.task_id} held by dead PID {pid} (stale)")
            return True
        except PermissionError:
            # Can't signal the process (different user/permissions) but it may still be alive
            # Conservatively assume the lock is held to avoid data corruption
            logger.debug(f"Lock for {self.task_id} held by PID {pid} (cannot signal, assuming live)")
            return False

    def _remove_lock(self) -> None:
        """Remove the lock directory."""
        if self.lock_path.exists():
            try:
                shutil.rmtree(self.lock_path)
            except OSError as e:
                logger.warning(f"Failed to remove lock directory {self.lock_path}: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock for {self.task_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
