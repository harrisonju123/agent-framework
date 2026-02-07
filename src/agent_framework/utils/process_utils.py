"""Process management utilities for killing process trees."""

import os
import signal


def kill_process_tree(pid: int, sig: int) -> None:
    """Send signal to entire process group, falling back to single process.

    Agents spawned with start_new_session=True get their own process group,
    so killpg reaches all child processes (e.g. claude CLI subprocesses).
    Falls back to os.kill for agents spawned before this fix.
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    except OSError:
        # Fallback: process may not be a group leader (pre-fix agents)
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
