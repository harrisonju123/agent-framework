"""Git worktree manager for isolated agent workspaces.

Provides git worktree support so agents work in isolated directories while
the user keeps their working directory untouched. This follows industry patterns
used by Cursor, Claude Squad, and other AI coding tools.
"""

import fcntl
import json
import logging
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.validators import validate_branch_name, validate_identifier, validate_owner_repo
from ..utils.subprocess_utils import run_git_command, SubprocessError

logger = logging.getLogger(__name__)

# Registry file for tracking worktrees
REGISTRY_FILENAME = ".worktree-registry.json"

# Default limits
DEFAULT_MAX_WORKTREES = 20
DEFAULT_MAX_AGE_HOURS = 24

# Active worktrees older than this are considered stale (crashed agent) and eligible for eviction
STALE_ACTIVE_THRESHOLD_SECONDS = 2 * 3600


@dataclass
class WorktreeInfo:
    """Information about a worktree."""
    path: str
    branch: str
    agent_id: str
    task_id: str
    created_at: str
    last_accessed: str
    base_repo: str
    active: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorktreeInfo":
        """Create from dictionary."""
        # Filter to known fields for backwards compatibility with registries
        # that don't yet have the 'active' field
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class WorktreeConfig:
    """Configuration for worktree management."""
    enabled: bool = False
    root: Path = field(default_factory=lambda: Path("~/.agent-workspaces/worktrees"))
    cleanup_on_complete: bool = True
    cleanup_on_failure: bool = False
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS
    max_worktrees: int = DEFAULT_MAX_WORKTREES

    def __post_init__(self):
        """Expand paths after initialization."""
        if isinstance(self.root, str):
            self.root = Path(self.root)
        self.root = self.root.expanduser().resolve()


class WorktreeManager:
    """Manages git worktrees for isolated agent workspaces."""

    # Keeps worktree registry keys and filesystem path segments from blowing up
    # when tasks accumulate nested "chain-" prefixes across review cycles.
    _MAX_WORKTREE_KEY_LENGTH = 60

    def __init__(
        self,
        config: WorktreeConfig,
        github_token: Optional[str] = None,
    ):
        """
        Initialize worktree manager.

        Args:
            config: Worktree configuration
            github_token: GitHub token for authenticated operations
        """
        self.config = config
        self.token = github_token
        self._registry: Dict[str, WorktreeInfo] = {}
        self._registry_dirty = False
        self._last_registry_save: float = 0.0

        # Ensure worktree root exists
        self.config.root.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def _get_registry_path(self) -> Path:
        """Get path to registry file."""
        return self.config.root / REGISTRY_FILENAME

    def _get_lock_path(self) -> Path:
        """Get path to lock file for registry operations."""
        return self.config.root / ".worktree-registry.lock"

    @contextmanager
    def _registry_lock(self):
        """Context manager for file-based locking of registry operations.

        Uses fcntl.flock for cross-process synchronization.
        """
        lock_path = self._get_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def _load_registry(self) -> None:
        """Load worktree registry from disk."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with self._registry_lock():
                    with open(registry_path) as f:
                        data = json.load(f)
                    self._registry = {
                        key: WorktreeInfo.from_dict(val)
                        for key, val in data.items()
                    }
                logger.debug(f"Loaded {len(self._registry)} worktrees from registry")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load worktree registry: {e}")
                self._registry = {}
        else:
            self._registry = {}

    def reload_registry(self) -> None:
        """Reload registry from disk for cross-process visibility."""
        self._load_registry()

    def _save_registry(self) -> None:
        """Save worktree registry to disk with file locking."""
        registry_path = self._get_registry_path()
        try:
            with self._registry_lock():
                with open(registry_path, "w") as f:
                    json.dump(
                        {key: val.to_dict() for key, val in self._registry.items()},
                        f,
                        indent=2,
                    )
            self._registry_dirty = False
            self._last_registry_save = time.time()
        except OSError as e:
            logger.error(f"Failed to save worktree registry: {e}")

    def _validate_branch_name(self, branch_name: str) -> str:
        """Validate and sanitize git branch name."""
        return validate_branch_name(branch_name)

    def _validate_identifier(self, value: str, name: str) -> str:
        """Validate agent_id or task_id to prevent path traversal."""
        return validate_identifier(value, name)

    def _validate_owner_repo(self, owner_repo: str) -> str:
        """Validate repository name format (owner/repo)."""
        return validate_owner_repo(owner_repo)

    def _get_worktree_key(self, agent_id: str, task_id: str) -> str:
        """Generate registry key for a worktree."""
        agent_id = self._validate_identifier(agent_id, "agent_id")
        task_id = self._validate_identifier(task_id, "task_id")
        # Extract JIRA key (e.g., "ME-429") from task IDs like "jira-ME-429-1770446158"
        # so retries reuse the same worktree and each ticket gets its own
        ticket_key = task_id
        if task_id.startswith("jira-"):
            parts = task_id.split("-")
            if len(parts) >= 3:
                ticket_key = f"{parts[1]}-{parts[2]}"
        if len(ticket_key) > self._MAX_WORKTREE_KEY_LENGTH:
            ticket_key = ticket_key[:self._MAX_WORKTREE_KEY_LENGTH].rstrip("-")
        return f"{agent_id}-{ticket_key}"

    def _get_worktree_path(self, owner_repo: str, agent_id: str, task_id: str) -> Path:
        """Generate worktree path."""
        owner_repo = self._validate_owner_repo(owner_repo)
        owner, repo = owner_repo.split("/")
        key = self._get_worktree_key(agent_id, task_id)
        return self.config.root / owner / repo / key

    def create_worktree(
        self,
        base_repo: Path,
        branch_name: str,
        agent_id: str,
        task_id: str,
        owner_repo: str,
        start_point: Optional[str] = None,
    ) -> Path:
        """
        Create an isolated worktree for agent work.

        Args:
            base_repo: Path to the base repository (user's local clone or shared clone)
            branch_name: Branch name for the worktree
            agent_id: Agent identifier
            task_id: Task identifier
            owner_repo: Repository in "owner/repo" format
            start_point: Optional branch to base the new worktree on (e.g. an
                upstream engineer's branch). Falls back to default branch if
                the start_point isn't available on the remote.

        Returns:
            Path to the created worktree

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If worktree creation fails
        """
        branch_name = self._validate_branch_name(branch_name)
        base_repo = Path(base_repo).resolve()

        if not base_repo.exists():
            raise ValueError(f"Base repository does not exist: {base_repo}")

        # Check capacity and cleanup if needed
        self._enforce_capacity_limit()

        # Generate worktree path
        worktree_path = self._get_worktree_path(owner_repo, agent_id, task_id)
        worktree_key = self._get_worktree_key(agent_id, task_id)

        # If worktree already exists, return it
        if worktree_path.exists():
            if worktree_key in self._registry:
                logger.info(f"Reusing existing worktree: {worktree_path}")
                self._registry[worktree_key].active = True
                self._registry[worktree_key].last_accessed = datetime.now(timezone.utc).isoformat()
                self._save_registry()
                return worktree_path
            else:
                # Orphaned worktree, remove it first
                logger.warning(f"Removing orphaned worktree: {worktree_path}")
                self._remove_worktree_directory(worktree_path, base_repo)

        # Ensure parent directory exists
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        # Prune stale worktree refs — handles force-removal leftovers where
        # shutil.rmtree deleted the directory but git still tracks the worktree
        try:
            self._run_git(["worktree", "prune"], cwd=base_repo, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        try:
            # Fetch latest from origin
            self._run_git(["fetch", "origin"], cwd=base_repo, timeout=60)

            # Check if branch exists remotely
            branch_exists = self._branch_exists(base_repo, branch_name)

            if branch_exists:
                # Create worktree from existing branch
                self._run_git(
                    ["worktree", "add", str(worktree_path), branch_name],
                    cwd=base_repo,
                    timeout=60,
                )
            else:
                # Create worktree with new branch, preferring start_point if available
                base_ref = None
                if start_point:
                    # Only trust remote refs — local branches in shared clones
                    # may belong to another agent's worktree and haven't been pushed
                    if self._remote_branch_exists(base_repo, start_point):
                        base_ref = f"origin/{start_point}"
                    else:
                        logger.debug(
                            f"start_point {start_point} not on remote, falling back to default branch"
                        )
                if not base_ref:
                    default_branch = self._get_default_branch(base_repo)
                    base_ref = f"origin/{default_branch}"

                self._run_git(
                    ["worktree", "add", "-b", branch_name, str(worktree_path), base_ref],
                    cwd=base_repo,
                    timeout=60,
                )

            logger.info(f"Created worktree: {worktree_path} (branch: {branch_name})")

            # Copy CLAUDE.md from base repo if present but not in worktree
            # (handles uncommitted CLAUDE.md that won't be in git worktree checkouts)
            claude_md_src = base_repo / "CLAUDE.md"
            claude_md_dst = worktree_path / "CLAUDE.md"
            if claude_md_src.exists() and not claude_md_dst.exists():
                import shutil
                shutil.copy2(str(claude_md_src), str(claude_md_dst))
                logger.info(f"Copied CLAUDE.md from base repo to worktree")

            # Register worktree
            now = datetime.now(timezone.utc).isoformat()
            self._registry[worktree_key] = WorktreeInfo(
                path=str(worktree_path),
                branch=branch_name,
                agent_id=agent_id,
                task_id=task_id,
                created_at=now,
                last_accessed=now,
                base_repo=str(base_repo),
                active=True,
            )
            self._save_registry()

            return worktree_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)

            # Branch already checked out in another worktree — reuse if same agent
            conflict_path = self._parse_branch_conflict_path(error_msg)
            if conflict_path and conflict_path.exists():
                owning_agent = self._find_worktree_agent(conflict_path)
                if owning_agent and owning_agent != agent_id:
                    logger.warning(
                        f"Branch {branch_name} checked out by agent '{owning_agent}' "
                        f"at {conflict_path} — refusing cross-agent reuse"
                    )
                    raise RuntimeError(
                        f"Branch {branch_name} locked by agent '{owning_agent}' "
                        f"at {conflict_path}"
                    )
                logger.info(f"Branch {branch_name} already checked out at {conflict_path}, reusing")
                now = datetime.now(timezone.utc).isoformat()
                self._registry[worktree_key] = WorktreeInfo(
                    path=str(conflict_path),
                    branch=branch_name,
                    agent_id=agent_id,
                    task_id=task_id,
                    created_at=now,
                    last_accessed=now,
                    base_repo=str(base_repo),
                    active=True,
                )
                self._save_registry()
                return conflict_path

            logger.error(f"Failed to create worktree: {error_msg}")
            raise RuntimeError(f"Failed to create worktree: {error_msg}")

    def _parse_branch_conflict_path(self, error_msg: str) -> Optional[Path]:
        """Extract worktree path from git 'already checked out' error."""
        match = re.search(r"already checked out at '([^']+)'", error_msg)
        if match:
            return Path(match.group(1))
        return None

    def _find_worktree_agent(self, path: Path) -> Optional[str]:
        """Find the agent_id that owns a worktree path, from registry."""
        resolved = path.resolve()
        for info in self._registry.values():
            if Path(info.path).resolve() == resolved:
                return info.agent_id
        return None

    def remove_worktree(self, path: Path, force: bool = False) -> bool:
        """
        Remove a worktree.

        Args:
            path: Path to the worktree
            force: Force removal even if dirty

        Returns:
            True if removed successfully
        """
        path = Path(path).resolve()

        # Find registry entry
        worktree_key = None
        base_repo = None
        for key, info in self._registry.items():
            if Path(info.path).resolve() == path:
                worktree_key = key
                base_repo = Path(info.base_repo)
                break

        if not worktree_key:
            # Not in registry, try to remove directly
            if path.exists():
                logger.warning(f"Removing unregistered worktree: {path}")
                # Try to find base repo from git
                base_repo = self._find_base_repo(path)
                if base_repo:
                    return self._remove_worktree_directory(path, base_repo, force)
            return False

        # Remove worktree
        success = self._remove_worktree_directory(path, base_repo, force)

        if success:
            del self._registry[worktree_key]
            self._save_registry()

        return success

    def _remove_worktree_directory(
        self,
        path: Path,
        base_repo: Optional[Path],
        force: bool = False,
    ) -> bool:
        """Remove worktree directory using git worktree remove."""
        try:
            # Path already gone — just clean up git's stale worktree refs
            if not path.exists():
                if base_repo and base_repo.exists():
                    try:
                        self._run_git(["worktree", "prune"], cwd=base_repo, timeout=30)
                    except subprocess.CalledProcessError:
                        pass  # Another process may have pruned already
                logger.debug(f"Worktree already removed: {path}")
                return True

            if base_repo and base_repo.exists():
                # Use git worktree remove for proper cleanup
                cmd = ["worktree", "remove", str(path)]
                if force:
                    cmd.append("--force")
                self._run_git(cmd, cwd=base_repo, timeout=30)
            elif path.exists():
                # Fallback: direct removal if no base repo
                import shutil
                shutil.rmtree(path)

            logger.info(f"Removed worktree: {path}")
            return True

        except subprocess.CalledProcessError as e:
            if force:
                # Last resort: direct removal
                try:
                    import shutil
                    shutil.rmtree(path)
                    # Clean up git's stale worktree tracking left behind by rmtree
                    if base_repo and base_repo.exists():
                        try:
                            self._run_git(["worktree", "prune"], cwd=base_repo, timeout=30)
                        except subprocess.CalledProcessError:
                            pass
                    logger.info(f"Force removed worktree: {path}")
                    return True
                except Exception:
                    pass
            logger.error(f"Failed to remove worktree {path}: {e}")
            return False

    def get_worktree_for_task(self, task_id: str) -> Optional[Path]:
        """
        Find existing worktree for a task.

        Args:
            task_id: Task identifier

        Returns:
            Path to worktree if found, None otherwise
        """
        task_id_short = task_id[:8]
        for key, info in self._registry.items():
            if info.task_id.startswith(task_id_short):
                path = Path(info.path)
                if path.exists():
                    info.active = True
                    info.last_accessed = datetime.now(timezone.utc).isoformat()
                    self._save_registry()
                    return path
        return None

    def find_worktree_by_branch(self, branch_name: str) -> Optional[Path]:
        """Find an existing worktree that has the given branch checked out."""
        for key, info in self._registry.items():
            if info.branch == branch_name:
                path = Path(info.path)
                if path.exists():
                    info.active = True
                    info.last_accessed = datetime.now(timezone.utc).isoformat()
                    self._save_registry()
                    return path
        return None

    def cleanup_orphaned_worktrees(self) -> Dict[str, int]:
        """
        Remove stale worktrees older than max_age.

        Returns:
            Dict with counts: {"registered": N, "unregistered": M, "total": N+M}
        """
        registered_removed = 0
        now = datetime.now(timezone.utc)
        max_age_seconds = self.config.max_age_hours * 3600

        keys_to_remove = []
        for key, info in self._registry.items():
            try:
                # Handle both timezone-aware and naive datetimes
                last_accessed = datetime.fromisoformat(info.last_accessed)
                if last_accessed.tzinfo is None:
                    # Assume UTC for naive datetimes
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                age_seconds = (now - last_accessed).total_seconds()

                if age_seconds > max_age_seconds:
                    # Don't evict worktrees actively in use (unless stale)
                    if info.active and not self._is_stale_active(info):
                        continue

                    path = Path(info.path)
                    base_repo = Path(info.base_repo) if info.base_repo else None

                    # Worktree path gone — just purge from registry
                    if not path.exists():
                        keys_to_remove.append(key)
                        registered_removed += 1
                        logger.debug(f"Purged stale worktree from registry: {key}")
                        continue

                    # Don't destroy worktrees with uncommitted or unpushed work
                    if self.has_uncommitted_changes(path) or self.has_unpushed_commits(path):
                        logger.warning(
                            f"Skipping stale worktree with unsaved work: {path}"
                        )
                        continue

                    if self._remove_worktree_directory(path, base_repo, force=False):
                        keys_to_remove.append(key)
                        registered_removed += 1
                        logger.info(f"Cleaned up stale registered worktree: {path} (age: {age_seconds/3600:.1f}h)")
            except (ValueError, OSError) as e:
                logger.warning(f"Error processing worktree {key}: {e}")

        # Remove from registry
        for key in keys_to_remove:
            del self._registry[key]

        if registered_removed:
            self._save_registry()

        # Also clean up worktrees not in registry
        unregistered_removed = self._cleanup_unregistered_worktrees()

        if unregistered_removed:
            logger.info(f"Cleaned up {unregistered_removed} unregistered worktrees")

        return {
            "registered": registered_removed,
            "unregistered": unregistered_removed,
            "total": registered_removed + unregistered_removed,
        }

    def _cleanup_unregistered_worktrees(self) -> int:
        """Clean up worktrees that exist on disk but not in registry."""
        removed = 0

        # Check if root exists and is accessible
        if not self.config.root.exists():
            return 0

        try:
            registered_paths = {Path(info.path).resolve() for info in self._registry.values()}

            # Walk worktree root looking for .git files (worktree marker)
            for owner_dir in self.config.root.iterdir():
                if not owner_dir.is_dir() or owner_dir.name.startswith('.'):
                    continue
                try:
                    for repo_dir in owner_dir.iterdir():
                        if not repo_dir.is_dir():
                            continue
                        try:
                            for worktree_dir in repo_dir.iterdir():
                                if not worktree_dir.is_dir():
                                    continue
                                git_marker = worktree_dir / ".git"
                                if git_marker.exists() and worktree_dir.resolve() not in registered_paths:
                                    # Don't destroy worktrees with uncommitted or unpushed work
                                    if self.has_uncommitted_changes(worktree_dir) or self.has_unpushed_commits(worktree_dir):
                                        logger.warning(
                                            f"Skipping unregistered worktree with unsaved work: {worktree_dir}"
                                        )
                                        continue

                                    base_repo = self._find_base_repo(worktree_dir)
                                    if self._remove_worktree_directory(worktree_dir, base_repo, force=False):
                                        removed += 1
                                        logger.debug(f"Removed unregistered worktree: {worktree_dir}")
                        except PermissionError as e:
                            logger.warning(f"Permission denied accessing {repo_dir}: {e}")
                except PermissionError as e:
                    logger.warning(f"Permission denied accessing {owner_dir}: {e}")
        except PermissionError as e:
            logger.warning(f"Permission denied accessing worktree root {self.config.root}: {e}")
        except OSError as e:
            logger.error(f"Error cleaning unregistered worktrees: {e}")

        return removed

    def _enforce_capacity_limit(self) -> None:
        """Remove oldest worktrees if over capacity, skipping active ones."""
        # Reload from disk so we see active flags set by other agent processes
        self._load_registry()

        if len(self._registry) < self.config.max_worktrees:
            return

        # Only consider inactive or stale-active worktrees for eviction
        evictable = [
            (key, info) for key, info in self._registry.items()
            if not info.active or self._is_stale_active(info)
        ]

        # Sort by last_accessed (oldest first)
        evictable.sort(key=lambda x: x[1].last_accessed)

        # Remove oldest until under limit
        to_remove = len(self._registry) - self.config.max_worktrees + 1
        removed = 0
        for key, info in evictable[:to_remove]:
            path = Path(info.path)
            base_repo = Path(info.base_repo) if info.base_repo else None
            if self._remove_worktree_directory(path, base_repo, force=True):
                del self._registry[key]
                logger.info(f"Removed oldest worktree (LRU): {path}")
                removed += 1

        if removed:
            self._save_registry()

    def mark_worktree_inactive(self, path: Path) -> None:
        """Mark a worktree as no longer actively used by an agent.

        Called when the agent finishes (success or failure) so the worktree
        becomes eligible for eviction again.
        """
        path = Path(path).resolve()
        for key, info in self._registry.items():
            if Path(info.path).resolve() == path:
                info.active = False
                self._save_registry()
                logger.debug(f"Marked worktree inactive: {path}")
                return

    def touch_worktree(self, path: Path) -> None:
        """Update last_accessed without changing active status.

        Used by intermediate chain steps to prevent stale-active eviction
        during the gap before the next step reuses this worktree.
        """
        path = Path(path).resolve()
        for key, info in self._registry.items():
            if Path(info.path).resolve() == path:
                info.last_accessed = datetime.now(timezone.utc).isoformat()
                self._save_registry()
                return

    def _is_stale_active(self, info: WorktreeInfo) -> bool:
        """Check if an active worktree is stale (likely from a crashed agent).

        Returns True if the worktree is marked active but hasn't been accessed
        within STALE_ACTIVE_THRESHOLD_SECONDS.
        """
        if not info.active:
            return False
        try:
            last_accessed = datetime.fromisoformat(info.last_accessed)
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - last_accessed).total_seconds()
            return age > STALE_ACTIVE_THRESHOLD_SECONDS
        except (ValueError, OSError):
            return False

    def _update_last_accessed(self, key: str) -> None:
        """Update last_accessed timestamp for a worktree.

        Debounces saves — only writes to disk if 30s elapsed since last save.
        Structural changes (create/remove) still save immediately via _save_registry().
        """
        if key in self._registry:
            self._registry[key].last_accessed = datetime.now(timezone.utc).isoformat()
            self._registry_dirty = True
            self._maybe_flush_registry()

    def _maybe_flush_registry(self) -> None:
        """Save registry if dirty and 30s since last save."""
        if self._registry_dirty and (time.time() - self._last_registry_save > 30):
            self._save_registry()

    def flush(self) -> None:
        """Explicitly save registry if dirty."""
        if self._registry_dirty:
            self._save_registry()

    def _branch_exists(self, repo_path: Path, branch_name: str) -> bool:
        """Check if branch exists locally or remotely."""
        try:
            run_git_command(
                ["rev-parse", "--verify", branch_name],
                cwd=repo_path,
                check=True,
                timeout=10,
            )
            return True
        except (SubprocessError, subprocess.TimeoutExpired):
            pass

        return self._remote_branch_exists(repo_path, branch_name)

    def _remote_branch_exists(self, repo_path: Path, branch_name: str) -> bool:
        """Check if branch exists on the remote (origin) only.

        Unlike _branch_exists(), this ignores local branches. Worktrees share
        refs with the base repo, so a local-only branch created by one agent
        appears to exist for all agents — but 'origin/{branch}' won't resolve
        until it's been pushed.
        """
        try:
            run_git_command(
                ["rev-parse", "--verify", f"origin/{branch_name}"],
                cwd=repo_path,
                check=True,
                timeout=10,
            )
            return True
        except (SubprocessError, subprocess.TimeoutExpired):
            return False

    def _get_default_branch(self, repo_path: Path) -> str:
        """Get default branch name."""
        try:
            result = run_git_command(
                ["symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=repo_path,
                check=True,
                timeout=10,
            )
            return result.stdout.strip().split('/')[-1]
        except (SubprocessError, subprocess.TimeoutExpired):
            # Fallback to common names
            for branch in ["main", "master"]:
                try:
                    run_git_command(
                        ["rev-parse", "--verify", f"origin/{branch}"],
                        cwd=repo_path,
                        check=True,
                        timeout=10,
                    )
                    return branch
                except (SubprocessError, subprocess.TimeoutExpired):
                    continue
            return "main"

    def _find_base_repo(self, worktree_path: Path) -> Optional[Path]:
        """Find base repository for a worktree."""
        git_file = worktree_path / ".git"
        if not git_file.exists():
            return None

        try:
            if git_file.is_file():
                # .git file contains path to actual git dir
                content = git_file.read_text().strip()
                if content.startswith("gitdir:"):
                    git_dir = Path(content[7:].strip())
                    # git_dir is like /path/to/base/.git/worktrees/name
                    # Base repo .git is two levels up
                    base_git = git_dir.parent.parent
                    if base_git.name == ".git":
                        return base_git.parent
            return None
        except Exception:
            return None

    def _run_git(
        self,
        args: List[str],
        cwd: Path,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        """Run git command with proper error handling."""
        cmd = ["git"] + args
        env = os.environ.copy()

        # Set up credentials if token available
        if self.token:
            env['GIT_ASKPASS'] = 'echo'
            env['GIT_USERNAME'] = 'x-access-token'
            env['GIT_PASSWORD'] = self.token

        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )

    def has_unpushed_commits(self, worktree_path: Path) -> bool:
        """Check if worktree has commits not pushed to remote.

        Returns:
            True if there are unpushed commits, False otherwise
        """
        if not worktree_path.exists():
            return False

        try:
            # Check for commits ahead of origin
            result = subprocess.run(
                ["git", "rev-list", "--count", "@{u}..HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                count = int(result.stdout.strip())
                return count > 0

            # No tracking branch — check for local commits not reachable
            # from any remote ref (catches committed-but-never-pushed work)
            log_result = subprocess.run(
                ["git", "log", "--oneline", "--not", "--remotes", "-1"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if log_result.returncode == 0 and log_result.stdout.strip():
                return True

            # Fall through to uncommitted changes check
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            # If we can't determine, assume there might be unpushed work
            return True

    def has_uncommitted_changes(self, worktree_path: Path) -> bool:
        """Check if worktree has uncommitted changes.

        Returns:
            True if there are uncommitted changes, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            # If we can't determine, assume there might be changes
            return True

    def list_worktrees(self) -> List[WorktreeInfo]:
        """List all registered worktrees."""
        return list(self._registry.values())

    def get_stats(self) -> Dict:
        """Get worktree statistics."""
        total = len(self._registry)
        active = sum(1 for info in self._registry.values() if Path(info.path).exists())
        in_use = sum(1 for info in self._registry.values() if info.active)

        return {
            "total_registered": total,
            "active": active,
            "in_use": in_use,
            "orphaned": total - active,
            "max_worktrees": self.config.max_worktrees,
            "max_age_hours": self.config.max_age_hours,
        }
