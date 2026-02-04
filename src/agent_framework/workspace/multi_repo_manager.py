"""Multi-repository manager for cloning and managing multiple repos."""

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.validators import validate_branch_name, validate_owner_repo

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PROMPT_LENGTH = 100000
MAX_COMMIT_MESSAGE_LENGTH = 10000


class MultiRepoManager:
    """Manages multiple repositories in a central workspace."""

    def __init__(self, workspace_root: Path, github_token: str):
        """
        Initialize multi-repo manager.

        Args:
            workspace_root: Root directory where repos will be cloned
            github_token: GitHub personal access token for cloning

        Raises:
            ValueError: If token is invalid or authentication fails
        """
        self.workspace_root = Path(workspace_root).expanduser().resolve()

        # Validate token
        if not github_token or len(github_token) < 20:
            raise ValueError("Invalid GitHub token")

        self.token = github_token
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Verify token and get authenticated user
        self._verify_authentication()

    def _verify_authentication(self):
        """Verify GitHub token is valid."""
        try:
            from github import Github
            gh = Github(self.token)
            self.gh_user = gh.get_user().login
            logger.info(f"Authenticated as GitHub user: {self.gh_user}")
        except Exception as e:
            raise ValueError(f"GitHub authentication failed: {e}")

    def _validate_owner_repo(self, owner_repo: str) -> str:
        """
        Validate repository name format.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            Validated repository name

        Raises:
            ValueError: If format is invalid
        """
        return validate_owner_repo(owner_repo)

    def _validate_branch_name(self, branch_name: str) -> str:
        """
        Validate and sanitize git branch name.

        Args:
            branch_name: Branch name to validate

        Returns:
            Validated branch name

        Raises:
            ValueError: If branch name is invalid
        """
        result = validate_branch_name(branch_name)

        # Additional check specific to multi-repo manager
        if branch_name.endswith('.lock'):
            raise ValueError("Branch name cannot end with .lock")

        return result

    def _validate_file_path(self, file_path: str) -> str:
        """
        Validate file path to prevent traversal attacks.

        Args:
            file_path: File path to validate

        Returns:
            Validated file path

        Raises:
            ValueError: If path is invalid
        """
        # Prevent traversal
        if '..' in file_path or file_path.startswith('/'):
            raise ValueError(f"Invalid file path (traversal attempt): {file_path}")

        # No absolute paths
        if os.path.isabs(file_path):
            raise ValueError(f"Absolute paths not allowed: {file_path}")

        return file_path

    def _validate_commit_message(self, message: str) -> str:
        """
        Validate and sanitize commit message.

        Args:
            message: Commit message

        Returns:
            Sanitized message

        Raises:
            ValueError: If message is invalid
        """
        if not message or not message.strip():
            raise ValueError("Commit message cannot be empty")

        if len(message) > MAX_COMMIT_MESSAGE_LENGTH:
            raise ValueError("Commit message too long")

        # Remove control characters except newlines
        sanitized = ''.join(
            c for c in message
            if c == '\n' or (32 <= ord(c) < 127) or ord(c) >= 128
        )

        return sanitized

    def _verify_repo_access(self, owner_repo: str) -> bool:
        """
        Verify user has access to repository.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            True if access is available
        """
        try:
            from github import Github
            gh = Github(self.token)
            repo = gh.get_repo(owner_repo)
            # If we can get repo metadata, we have at least read access
            _ = repo.name
            return True
        except Exception as e:
            logger.error(f"No access to {owner_repo}: {e}")
            return False

    def ensure_repo(self, owner_repo: str) -> Path:
        """
        Ensure repository is cloned and up to date.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            Path to local repository

        Raises:
            ValueError: If repository name is invalid
            PermissionError: If no access to repository

        Example:
            >>> manager.ensure_repo("harrisonju123/service-a")
            Path("/Users/user/.agent-workspaces/harrisonju123/service-a")
        """
        owner_repo = self._validate_owner_repo(owner_repo)

        # Verify access before cloning
        if not self._verify_repo_access(owner_repo):
            raise PermissionError(f"No access to repository: {owner_repo}")

        # Resolve path and verify it's within workspace
        local_path = (self.workspace_root / owner_repo).resolve()

        # Security check: ensure resolved path is within workspace
        if not str(local_path).startswith(str(self.workspace_root)):
            raise ValueError(f"Path traversal attempt detected: {owner_repo}")

        if not local_path.exists():
            logger.info(f"Cloning {owner_repo}")
            self._clone(owner_repo, local_path)
        else:
            logger.info(f"Updating {owner_repo}")
            self._pull(local_path)

        return local_path

    def get_path(self, owner_repo: str) -> Path:
        """
        Get local path for a repository.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            Path to repository (may not exist yet)
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        return self.workspace_root / owner_repo

    def read_files(self, owner_repo: str, file_paths: List[str]) -> Dict[str, str]:
        """
        Read files from a repository.

        Args:
            owner_repo: Repository in format "owner/repo"
            file_paths: List of file paths relative to repo root

        Returns:
            Dict mapping file path to file content

        Raises:
            ValueError: If file paths contain invalid characters

        Example:
            >>> manager.read_files("org/repo", ["pkg/auth.go", "README.md"])
            {"pkg/auth.go": "package auth...", "README.md": "# Repo..."}
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        repo_path = self.get_path(owner_repo).resolve()
        result = {}

        for file_path in file_paths:
            try:
                # Validate file path
                file_path = self._validate_file_path(file_path)
            except ValueError as e:
                logger.error(str(e))
                continue

            full_path = (repo_path / file_path).resolve()

            # Verify path is within repository
            if not str(full_path).startswith(str(repo_path)):
                logger.error(f"Path traversal attempt: {file_path}")
                continue

            if not full_path.exists():
                logger.warning(f"File not found: {file_path} in {owner_repo}")
                continue

            if not full_path.is_file():
                logger.warning(f"Not a file: {file_path} in {owner_repo}")
                continue

            # Check file size
            if full_path.stat().st_size > MAX_FILE_SIZE:
                logger.error(f"File too large (>10MB): {file_path}")
                continue

            try:
                # Try to read as text with explicit encoding
                result[file_path] = full_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logger.error(f"Cannot read binary file: {file_path}")
                continue
            except PermissionError:
                logger.error(f"Permission denied: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        return result

    def get_default_branch(self, owner_repo: str) -> str:
        """
        Get the default branch name for a repository.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            Default branch name
        """
        repo_path = self.get_path(owner_repo)

        try:
            # Get default branch from remote
            result = subprocess.run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            # Output is like "refs/remotes/origin/main"
            branch = result.stdout.strip().split('/')[-1]
            return branch
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Fallback to common names
            for branch_name in ["main", "master", "develop"]:
                result = subprocess.run(
                    ["git", "rev-parse", "--verify", f"origin/{branch_name}"],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return branch_name

            # Last resort
            return "main"

    def create_branch(
        self,
        owner_repo: str,
        branch_name: str,
        base_branch: Optional[str] = None
    ):
        """
        Create a new branch from base branch.

        Args:
            owner_repo: Repository in format "owner/repo"
            branch_name: Name of new branch
            base_branch: Base branch (auto-detected if None)

        Raises:
            ValueError: If inputs are invalid
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        branch_name = self._validate_branch_name(branch_name)

        if base_branch is None:
            base_branch = self.get_default_branch(owner_repo)
        else:
            base_branch = self._validate_branch_name(base_branch)

        repo_path = self.get_path(owner_repo)

        try:
            # Fetch latest changes
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=60,
            )

            # Check if branch exists locally
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                cwd=repo_path,
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Branch exists, check it out and reset to base
                logger.info(f"Branch {branch_name} exists, resetting to {base_branch}")
                subprocess.run(
                    ["git", "checkout", branch_name],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{base_branch}"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
            else:
                # Create new branch
                subprocess.run(
                    ["git", "checkout", base_branch],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )
                subprocess.run(
                    ["git", "pull", "origin", base_branch],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    timeout=10,
                )

            logger.info(f"Branch {branch_name} ready in {owner_repo}")

        except subprocess.TimeoutExpired as e:
            logger.error(f"Git operation timed out: {e}")
            raise
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Failed to create branch: {error_msg}")
            raise

    def commit_and_push(self, owner_repo: str, branch_name: str, message: str):
        """
        Stage all changes, commit, and push to remote.

        Args:
            owner_repo: Repository in format "owner/repo"
            branch_name: Branch to push
            message: Commit message

        Raises:
            ValueError: If inputs are invalid
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        branch_name = self._validate_branch_name(branch_name)
        message = self._validate_commit_message(message)

        repo_path = self.get_path(owner_repo)

        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Check if there are changes to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if not status.stdout.strip():
                logger.info(f"No changes to commit in {owner_repo}")
                return

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Push with authentication via environment
            env = os.environ.copy()
            env['GIT_ASKPASS'] = 'echo'
            env['GIT_USERNAME'] = 'x-access-token'
            env['GIT_PASSWORD'] = self.token

            subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True,
                env=env,
                timeout=120,
            )

            logger.info(f"Committed and pushed changes in {owner_repo}")

        except subprocess.TimeoutExpired as e:
            logger.error(f"Git operation timed out: {e}")
            raise
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Failed to commit/push: {error_msg}")
            raise

    def _clone(self, owner_repo: str, local_path: Path):
        """
        Clone repository using HTTPS with secure credential handling.

        Args:
            owner_repo: Repository in format "owner/repo"
            local_path: Local path to clone to
        """
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Use HTTPS without token in URL (more secure)
        clone_url = f"https://github.com/{owner_repo}.git"

        # Set up credential helper environment
        env = os.environ.copy()
        env['GIT_ASKPASS'] = 'echo'
        env['GIT_USERNAME'] = 'x-access-token'
        env['GIT_PASSWORD'] = self.token

        try:
            subprocess.run(
                ["git", "clone", clone_url, str(local_path)],
                check=True,
                capture_output=True,
                env=env,
                timeout=300,
            )
            logger.info(f"Successfully cloned {owner_repo}")
        except subprocess.TimeoutExpired as e:
            logger.error(f"Clone timed out for {owner_repo}")
            raise
        except subprocess.CalledProcessError as e:
            # Don't log stderr as it might contain sensitive info
            logger.error(f"Failed to clone {owner_repo}")
            raise

    def _pull(self, repo_path: Path):
        """
        Pull latest changes from origin.

        Args:
            repo_path: Path to repository
        """
        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            current_branch = result.stdout.strip()

            # Pull latest
            env = os.environ.copy()
            env['GIT_ASKPASS'] = 'echo'
            env['GIT_USERNAME'] = 'x-access-token'
            env['GIT_PASSWORD'] = self.token

            subprocess.run(
                ["git", "pull", "origin", current_branch],
                cwd=repo_path,
                check=True,
                capture_output=True,
                env=env,
                timeout=120,
            )
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Pull timed out: {e}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to pull (non-fatal)")

    def get_current_branch(self, owner_repo: str) -> Optional[str]:
        """
        Get current branch name for a repository.

        Args:
            owner_repo: Repository in format "owner/repo"

        Returns:
            Current branch name or None if error
        """
        repo_path = self.get_path(owner_repo)

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    def delete_branch(self, owner_repo: str, branch_name: str, force: bool = False):
        """
        Delete a local branch.

        Args:
            owner_repo: Repository in format "owner/repo"
            branch_name: Branch to delete
            force: Force delete even if not merged
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        branch_name = self._validate_branch_name(branch_name)
        repo_path = self.get_path(owner_repo)

        try:
            flag = "-D" if force else "-d"
            subprocess.run(
                ["git", "branch", flag, branch_name],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=10,
            )
            logger.info(f"Deleted branch {branch_name} in {owner_repo}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to delete branch {branch_name}: {e}")

    def reset_hard(self, owner_repo: str):
        """
        Reset repository to clean state (discard all changes).

        Args:
            owner_repo: Repository in format "owner/repo"
        """
        owner_repo = self._validate_owner_repo(owner_repo)
        repo_path = self.get_path(owner_repo)

        try:
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info(f"Reset {owner_repo} to clean state")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Failed to reset {owner_repo}: {e}")
            raise
