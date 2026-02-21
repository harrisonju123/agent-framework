"""Task manifest — immutable shared context for a workflow chain.

Write-once JSON file at `.agent-communication/manifests/{root_task_id}.json`
that stores the canonical branch and key metadata for a task chain. All agents
read it; only the framework writes it. The core invariant: once a manifest
exists for a root_task_id, get_or_create_manifest() never overwrites it.

This prevents branch drift when external git checkouts or worktree races
change HEAD out from under an agent mid-task.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..utils.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass
class TaskManifest:
    """Immutable metadata for a workflow chain."""

    root_task_id: str
    branch: str
    github_repo: Optional[str] = None
    user_goal: str = ""
    workflow: str = "default"
    working_directory: Optional[str] = None
    created_at: str = ""
    created_by: str = ""


def _manifest_dir(workspace: Path) -> Path:
    return workspace / ".agent-communication" / "manifests"


def _manifest_path(workspace: Path, root_task_id: str) -> Path:
    return _manifest_dir(workspace) / f"{root_task_id}.json"


def load_manifest(workspace: Path, root_task_id: str) -> Optional[TaskManifest]:
    """Load manifest from disk. Returns None if missing or corrupt."""
    path = _manifest_path(workspace, root_task_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load manifest for %s: %s", root_task_id, e)
        return None

    if not isinstance(data, dict) or "root_task_id" not in data or "branch" not in data:
        logger.warning("Malformed manifest for %s: missing required fields", root_task_id)
        return None

    known_fields = set(TaskManifest.__dataclass_fields__)
    return TaskManifest(**{k: v for k, v in data.items() if k in known_fields})


def save_manifest(workspace: Path, manifest: TaskManifest) -> None:
    """Atomically write manifest to disk."""
    manifest_dir = _manifest_dir(workspace)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    path = _manifest_path(workspace, manifest.root_task_id)
    atomic_write_text(path, json.dumps(asdict(manifest), indent=2))


def get_or_create_manifest(
    workspace: Path,
    root_task_id: str,
    branch: str,
    **kwargs,
) -> TaskManifest:
    """Return existing manifest or create a new one.

    If a manifest already exists for this root_task_id, it is returned as-is —
    the passed-in branch and kwargs are ignored. This enforces the write-once
    invariant: the first agent to call this sets the canonical branch forever.
    """
    existing = load_manifest(workspace, root_task_id)
    if existing is not None:
        return existing

    manifest = TaskManifest(
        root_task_id=root_task_id,
        branch=branch,
        created_at=datetime.now(timezone.utc).isoformat(),
        **kwargs,
    )
    save_manifest(workspace, manifest)
    logger.info("Created task manifest for %s on branch %s", root_task_id, branch)

    # Re-read to handle the (benign) race where two agents both see None
    # and write simultaneously — return whichever one landed on disk
    return load_manifest(workspace, root_task_id) or manifest
