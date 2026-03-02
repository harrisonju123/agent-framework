"""Type conversion helper utilities."""

import os
import time


def unique_id_suffix() -> str:
    """Collision-resistant suffix for task/team IDs.

    Combines second-granularity timestamp with PID and random hex
    so concurrent processes in the same second get distinct IDs.
    """
    return f"{int(time.time())}-{os.getpid() % 10000:04d}{os.urandom(2).hex()}"


def strip_chain_prefixes(title: str) -> str:
    """Remove accumulated [chain]/[pr] prefixes so re-wrapping adds exactly one."""
    while title.startswith(("[chain] ", "[pr] ")):
        title = title[len("[chain] "):] if title.startswith("[chain] ") else title[len("[pr] "):]
    return title


def get_type_str(task_type) -> str:
    """
    Get string value from task type.

    Handles both enum values (with .value attribute) and plain strings.

    Args:
        task_type: Either a TaskType enum or string

    Returns:
        String representation of the task type
    """
    return task_type.value if hasattr(task_type, 'value') else str(task_type)
