"""Workspace management utilities."""

import logging
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspace setup and git operations."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)

    def ensure_git_repo(self, repo_url: Optional[str] = None) -> bool:
        """
        Ensure workspace is a git repository.

        Args:
            repo_url: Optional URL to clone from if .git doesn't exist

        Returns:
            True if git repo exists or was created
        """
        git_dir = self.workspace / ".git"

        if git_dir.exists():
            logger.info(f"Git repository exists at {self.workspace}")
            return True

        if repo_url:
            logger.info(f"Cloning repository from {repo_url}")
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(self.workspace)],
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repository: {e.stderr.decode()}")
                return False
        else:
            # Initialize new git repo
            logger.info(f"Initializing new git repository at {self.workspace}")
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=self.workspace,
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to init repository: {e.stderr.decode()}")
                return False

    def checkout_clean_branch(self, base_branch: str = "main") -> bool:
        """
        Checkout base branch and pull latest.

        Args:
            base_branch: Base branch name (default: main)

        Returns:
            True if successful
        """
        try:
            # Fetch latest
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Checkout base branch
            subprocess.run(
                ["git", "checkout", base_branch],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Pull latest
            subprocess.run(
                ["git", "pull", "origin", base_branch],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            logger.info(f"Checked out and updated {base_branch}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout clean branch: {e.stderr.decode() if e.stderr else str(e)}")
            return False

    def get_current_branch(self) -> Optional[str]:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
                text=True,
            )
            return len(result.stdout.strip()) > 0
        except subprocess.CalledProcessError:
            return False

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """
        Create a new branch from base.

        Args:
            branch_name: New branch name
            from_branch: Branch to create from

        Returns:
            True if successful
        """
        try:
            # Ensure we're on the base branch
            subprocess.run(
                ["git", "checkout", from_branch],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            logger.info(f"Created branch {branch_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e.stderr.decode() if e.stderr else str(e)}")
            return False

    def commit_changes(self, message: str) -> bool:
        """
        Stage and commit all changes.

        Args:
            message: Commit message

        Returns:
            True if successful
        """
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            logger.info(f"Committed changes: {message}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit: {e.stderr.decode() if e.stderr else str(e)}")
            return False

    def push_branch(self, branch_name: str, set_upstream: bool = True) -> bool:
        """
        Push branch to remote.

        Args:
            branch_name: Branch to push
            set_upstream: Set upstream tracking

        Returns:
            True if successful
        """
        try:
            cmd = ["git", "push"]
            if set_upstream:
                cmd.extend(["-u", "origin", branch_name])
            else:
                cmd.extend(["origin", branch_name])

            subprocess.run(
                cmd,
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            logger.info(f"Pushed branch {branch_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push: {e.stderr.decode() if e.stderr else str(e)}")
            return False
