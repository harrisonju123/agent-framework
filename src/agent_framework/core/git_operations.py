"""Git/PR/Worktree operations extracted from Agent class.

This module contains all git, pull request, and worktree lifecycle operations
that were previously embedded in the Agent class. By extracting these operations,
we achieve better separation of concerns and make the Agent class more focused.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace.worktree_manager import WorktreeManager
    from ..queue.file_queue import FileQueue
    from .pr_lifecycle import PRLifecycleManager

from .task import Task


class GitOperationsManager:
    """Manages git operations, worktree lifecycle, and PR creation."""

    def __init__(
        self,
        config,
        workspace: Path,
        queue: "FileQueue",
        logger,
        worktree_manager: Optional["WorktreeManager"] = None,
        multi_repo_manager=None,
        github_client=None,
        jira_client=None,
        session_logger=None,
        pr_lifecycle_manager: Optional["PRLifecycleManager"] = None,
        agent_definition=None,
        workflows_config=None,
    ):
        """Initialize GitOperationsManager.

        Args:
            config: Agent configuration
            workspace: Base workspace path
            queue: Task queue for managing tasks
            logger: Logger instance
            worktree_manager: Optional worktree manager
            multi_repo_manager: Optional multi-repo manager
            github_client: Optional GitHub client
            jira_client: Optional JIRA client
            session_logger: Optional session logger
            pr_lifecycle_manager: Optional PR lifecycle manager
            agent_definition: Optional agent definition for JIRA permissions
            workflows_config: Optional workflow configuration dict
        """
        self.config = config
        self.workspace = Path(workspace)
        self.queue = queue
        self.logger = logger
        self.worktree_manager = worktree_manager
        self.multi_repo_manager = multi_repo_manager
        self.github_client = github_client
        self.jira_client = jira_client
        self.session_logger = session_logger
        self._pr_lifecycle_manager = pr_lifecycle_manager
        self._agent_definition = agent_definition
        self._workflows_config = workflows_config or {}

        # Track active worktree for cleanup (mutable state)
        self._active_worktree: Optional[Path] = None

    def get_working_directory(self, task: Task) -> Path:
        """Get working directory for task (worktree, target repo, or framework workspace).

        Priority:
        0. PR creation tasks with an implementation_branch skip worktree entirely
        1. If worktree mode enabled (config or task override), create isolated worktree
        2. If multi_repo_manager available, use shared clone
        3. Fall back to framework workspace
        """
        github_repo = task.context.get("github_repo")

        # PR creation tasks that reference an upstream implementation branch
        # don't need their own worktree — `gh pr create` works from the shared clone
        if task.context.get("pr_creation_step") and task.context.get("implementation_branch"):
            if github_repo and self.multi_repo_manager:
                repo_path = self.multi_repo_manager.ensure_repo(github_repo)
                self.logger.info("PR creation task — using shared clone (no worktree needed)")
                return repo_path

        # Check if worktree mode should be used
        use_worktree = self._should_use_worktree(task)

        if use_worktree and github_repo and self.worktree_manager:
            # Get base repo path (shared clone or explicit override)
            base_repo = self._get_base_repo_for_worktree(task, github_repo)

            if base_repo:
                # Fix-cycle reuse: if a prior step already established a branch, reuse it
                branch_name = task.context.get("worktree_branch") or task.context.get("implementation_branch")

                if not branch_name:
                    jira_key = task.context.get("jira_key", "task")
                    task_hash = hashlib.sha256(task.id.encode()).hexdigest()[:8]
                    branch_name = f"agent/{self.config.id}/{jira_key}-{task_hash}"

                # Check registry for existing worktree on this branch (reuse or retry)
                existing = self.worktree_manager.find_worktree_by_branch(branch_name)
                if existing:
                    self._active_worktree = existing
                    task.context["worktree_branch"] = branch_name
                    self.logger.info(f"Reusing worktree for branch {branch_name}: {existing}")
                    return existing

                try:
                    worktree_path = self.worktree_manager.create_worktree(
                        base_repo=base_repo,
                        branch_name=branch_name,
                        agent_id=self.config.id,
                        task_id=task.id,
                        owner_repo=github_repo,
                    )
                    self._active_worktree = worktree_path
                    task.context["worktree_branch"] = branch_name
                    self.logger.info(f"Using worktree: {github_repo} at {worktree_path}")
                    return worktree_path
                except Exception as e:
                    self.logger.warning(f"Failed to create worktree, falling back to shared clone: {e}")
                    # Fall through to shared clone

        if github_repo and self.multi_repo_manager:
            # Ensure repo is cloned/updated
            repo_path = self.multi_repo_manager.ensure_repo(github_repo)
            self.logger.info(f"Using repository: {github_repo} at {repo_path}")
            return repo_path
        else:
            # No repo context, use framework workspace
            return self.workspace

    def _should_use_worktree(self, task: Task) -> bool:
        """Determine if worktree mode should be used for this task.

        Task context can override config:
        - task.context["use_worktree"] = True/False
        """
        # Check task-level override first
        task_override = task.context.get("use_worktree")
        if task_override is not None:
            return bool(task_override)

        # Check if worktree manager is available and enabled
        if not self.worktree_manager:
            return False

        return True  # Worktree manager exists, so worktree mode is enabled

    def _get_base_repo_for_worktree(self, task: Task, github_repo: str) -> Optional[Path]:
        """Get base repository path for worktree creation.

        Priority:
        1. Explicit path in task.context["worktree_base_repo"]
        2. Shared clone from multi_repo_manager
        """
        # Check for explicit base repo override
        explicit_base = task.context.get("worktree_base_repo")
        if explicit_base:
            base_path = Path(explicit_base).expanduser().resolve()
            if base_path.exists() and (base_path / ".git").exists():
                self.logger.debug(f"Using explicit base repo: {base_path}")
                return base_path
            else:
                self.logger.warning(f"Explicit worktree_base_repo not valid: {explicit_base}")

        # Use shared clone from multi_repo_manager
        if self.multi_repo_manager:
            try:
                return self.multi_repo_manager.ensure_repo(github_repo)
            except Exception as e:
                self.logger.error(f"Failed to get base repo from multi_repo_manager: {e}")

        return None

    def sync_worktree_queued_tasks(self) -> None:
        """Move any task files the LLM wrote to the worktree's queues back to the main queue.

        When the Claude CLI subprocess runs in a worktree, the LLM may create
        subtask JSON files via the Write tool at .agent-communication/queues/<agent>/.
        These land in the worktree instead of the main repo's queue that agent
        workers actually poll.  This method finds those orphaned files and
        re-queues them through the canonical FileQueue.push() path.
        """
        if not self._active_worktree:
            return

        worktree_queue_dir = self._active_worktree / ".agent-communication" / "queues"
        if not worktree_queue_dir.exists():
            return

        # Don't sync from the main workspace back into itself
        main_queue_dir = self.queue.queue_dir
        try:
            if worktree_queue_dir.resolve() == main_queue_dir.resolve():
                return
        except OSError:
            return

        synced = 0
        for agent_dir in worktree_queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            # Skip non-agent directories (checkpoints, completed, etc.)
            queue_id = agent_dir.name
            if queue_id in ("checkpoints", "completed", "failed", "locks", "heartbeats", "malformed"):
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    data = json.loads(task_file.read_text())
                    synced_task = Task(**data)
                    self.queue.push(synced_task, synced_task.assigned_to)
                    task_file.unlink()
                    synced += 1
                    self.logger.info(
                        f"Synced worktree queue task {synced_task.id} → {synced_task.assigned_to}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to sync worktree task {task_file.name}: {e}")

        if synced:
            self.logger.info(f"Synced {synced} task(s) from worktree to main queue")

    def cleanup_worktree(self, task: Task, success: bool) -> None:
        """Cleanup worktree after task completion based on config.

        Safety checks:
        - Skip cleanup if there are unpushed commits (data loss prevention)
        - Skip cleanup if there are uncommitted changes
        - Log warnings when skipping to help with debugging
        """
        if not self._active_worktree or not self.worktree_manager:
            return

        worktree_config = self.worktree_manager.config
        should_cleanup = (
            (success and worktree_config.cleanup_on_complete) or
            (not success and worktree_config.cleanup_on_failure)
        )

        if should_cleanup:
            # Safety check: don't delete worktrees with unpushed work
            has_unpushed = self.worktree_manager.has_unpushed_commits(self._active_worktree)
            has_uncommitted = self.worktree_manager.has_uncommitted_changes(self._active_worktree)

            if has_unpushed:
                self.logger.warning(
                    f"Skipping worktree cleanup - unpushed commits detected: {self._active_worktree}. "
                    f"Manual cleanup required after pushing changes."
                )
            elif has_uncommitted:
                self.logger.warning(
                    f"Skipping worktree cleanup - uncommitted changes detected: {self._active_worktree}. "
                    f"Manual cleanup required after committing/discarding changes."
                )
            else:
                try:
                    self.worktree_manager.remove_worktree(self._active_worktree, force=not success)
                    self.logger.info(f"Cleaned up worktree: {self._active_worktree}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup worktree: {e}")

        self._active_worktree = None

    def push_and_create_pr_if_needed(self, task: Task) -> None:
        """Push branch and create PR if the agent produced unpushed commits.

        Runs after the LLM finishes but before the task is marked completed,
        so the PR URL is available in task.context for downstream chain steps.
        Only acts when working in a worktree with actual unpushed commits.

        Intermediate workflow steps push their branch but skip PR creation —
        the terminal step (or pr_creator) handles that.
        """
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        # Already has a PR (created by the LLM via MCP or _handle_success)
        if task.context.get("pr_url"):
            self.logger.debug(f"PR already exists for {task.id}: {task.context['pr_url']}")
            return

        # PR creation task with an implementation branch from upstream — create
        # the PR from that branch without needing a worktree
        impl_branch = task.context.get("implementation_branch")
        if task.context.get("pr_creation_step") and impl_branch:
            self._create_pr_from_branch(task, impl_branch)
            return

        # Only act if we have an active worktree with changes
        if not self._active_worktree or not self.worktree_manager:
            self.logger.debug(f"No active worktree for {task.id}, skipping PR creation")
            return

        has_unpushed = self.worktree_manager.has_unpushed_commits(self._active_worktree)
        branch_already_pushed = False
        if not has_unpushed:
            # LLM may have pushed the branch itself — check if it exists on the remote
            branch_already_pushed = self._remote_branch_exists(self._active_worktree)
            if not branch_already_pushed:
                self.logger.debug(f"No unpushed commits and no remote branch for {task.id}")
                return
            self.logger.debug(f"Branch already pushed to remote for {task.id}, will create PR only")

        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.debug(f"No github_repo in task context for {task.id}, skipping PR creation")
            return

        try:
            worktree = self._active_worktree

            # Get the current branch name
            result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree, check=False, timeout=10,
            )
            if result.returncode != 0:
                self.logger.warning("Could not determine branch name, skipping PR creation")
                return
            branch = result.stdout.strip()

            # Don't create PRs from main/master
            if branch in ("main", "master"):
                self.logger.debug(f"On {branch} branch for {task.id}, skipping PR creation")
                return

            # Push the branch (skip if LLM already pushed it)
            if not branch_already_pushed:
                self.logger.info(f"Pushing branch {branch} to origin")
                push_result = run_git_command(
                    ["push", "-u", "origin", branch],
                    cwd=worktree, check=False, timeout=60,
                )
                if push_result.returncode != 0:
                    self.logger.error(f"Failed to push branch: {push_result.stderr}")
                    return

            # Intermediate workflow steps: push code but skip PR creation.
            # Store the branch so downstream agents can create the PR later.
            if not self._is_at_terminal_workflow_step(task):
                task.context["implementation_branch"] = branch
                self.logger.info(
                    f"Intermediate step — pushed {branch} but skipped PR creation"
                )
                return

            self._create_pr_via_gh(task, github_repo, branch, cwd=worktree)

        except SubprocessError as e:
            self.logger.error(f"Subprocess error during PR creation: {e}")
        except Exception as e:
            self.logger.error(f"Error creating PR: {e}")

    def manage_pr_lifecycle(self, task: Task) -> None:
        """Autonomously monitor CI, fix failures, and merge PR if repo opts in."""
        if not self._pr_lifecycle_manager:
            return
        if not self._pr_lifecycle_manager.should_manage(task):
            return

        try:
            merged = self._pr_lifecycle_manager.manage(task, self.config.id)
            if merged:
                self._sync_jira_status(
                    task, "Done",
                    comment=f"PR merged automatically: {task.context.get('pr_url')}",
                )
        except Exception as e:
            self.logger.error(f"PR lifecycle error for {task.id}: {e}")

    def _create_pr_from_branch(self, task: Task, branch: str) -> None:
        """Create a PR from an existing pushed branch (used by pr_creation_step tasks).

        Called when the terminal PR creation agent receives a task with an
        implementation_branch set by an upstream agent. No worktree needed —
        just runs `gh pr create --head <branch>` against the repo.
        """
        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.warning("No github_repo in context, cannot create PR from branch")
            return

        # Determine cwd — use shared clone if available, otherwise workspace
        cwd = self.workspace
        if self.multi_repo_manager:
            try:
                cwd = self.multi_repo_manager.ensure_repo(github_repo)
            except Exception as e:
                self.logger.debug(f"Failed to ensure shared repo for {github_repo}, falling back to workspace: {e}")

        self._create_pr_via_gh(task, github_repo, branch, cwd=cwd)

    def _create_pr_via_gh(self, task: Task, github_repo: str, branch: str, *, cwd) -> None:
        """Create a PR via gh CLI. Shared by worktree and branch-based flows."""
        from ..utils.subprocess_utils import run_command, SubprocessError

        # Build a clean PR title — strip workflow prefixes
        from ..workflow.executor import _strip_chain_prefixes
        pr_title = _strip_chain_prefixes(task.title)[:70]

        pr_body = f"## Summary\n\n{task.context.get('user_goal', task.description)}"

        self.logger.info(f"Creating PR for {github_repo} from branch {branch}")
        try:
            pr_result = run_command(
                ["gh", "pr", "create",
                 "--repo", github_repo,
                 "--title", pr_title,
                 "--body", pr_body,
                 "--head", branch],
                cwd=cwd, check=False, timeout=30,
            )

            if pr_result.returncode == 0:
                pr_url = pr_result.stdout.strip()
                task.context["pr_url"] = pr_url
                self.logger.info(f"Created PR: {pr_url}")
                # Clean up orphaned subtask PRs/branches for fan-in tasks
                self._close_subtask_prs(task, pr_url)
                self._cleanup_subtask_branches(task)
            else:
                if "already exists" in pr_result.stderr:
                    self.logger.info("PR already exists for this branch")
                else:
                    self.logger.error(f"Failed to create PR: {pr_result.stderr}")
        except SubprocessError as e:
            self.logger.error(f"Failed to create PR: {e}")

    def _close_subtask_prs(self, task: Task, fan_in_pr_url: str) -> None:
        """Close orphaned PRs created by subtask LLMs. Best-effort.

        Subtask LLMs may create PRs via MCP despite prompt suppression.
        After the fan-in PR is created, close those orphans so they don't
        linger as duplicates.
        """
        if not task.context.get("fan_in"):
            return

        from ..utils.subprocess_utils import run_command

        parent_task_id = task.context.get("parent_task_id")
        if not parent_task_id:
            return

        parent = self.queue.find_task(parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        github_repo = task.context.get("github_repo")
        if not github_repo:
            return

        for sid in parent.subtask_ids:
            subtask = self.queue.get_completed(sid)
            if not subtask:
                continue
            pr_url = subtask.context.get("pr_url")
            if not pr_url:
                continue
            # Extract PR number from URL (e.g. .../pull/18 → 18)
            pr_number = pr_url.rstrip("/").split("/")[-1]
            self.logger.info(f"Closing orphaned subtask PR #{pr_number}")
            run_command(
                ["gh", "pr", "close", pr_number,
                 "--repo", github_repo,
                 "--comment", f"Superseded by fan-in PR {fan_in_pr_url}"],
                check=False, timeout=30,
            )

    def _cleanup_subtask_branches(self, task: Task) -> None:
        """Delete remote branches created by subtask LLMs. Best-effort.

        After the fan-in PR lands, subtask branches are stale. Clean them up
        so they don't clutter the remote.
        """
        if not task.context.get("fan_in"):
            return

        from ..utils.subprocess_utils import run_command

        parent_task_id = task.context.get("parent_task_id")
        if not parent_task_id:
            return

        parent = self.queue.find_task(parent_task_id)
        if not parent or not parent.subtask_ids:
            return

        github_repo = task.context.get("github_repo")
        if not github_repo:
            return

        # Get cwd for git command - use shared clone
        cwd = self.workspace
        if self.multi_repo_manager:
            try:
                cwd = self.multi_repo_manager.ensure_repo(github_repo)
            except Exception as e:
                self.logger.warning(f"Failed to get repo path for branch cleanup: {e}")
                return

        for sid in parent.subtask_ids:
            subtask = self.queue.get_completed(sid)
            if not subtask:
                continue
            branch = (
                subtask.context.get("implementation_branch")
                or subtask.context.get("worktree_branch")
            )
            if not branch:
                continue
            self.logger.info(f"Deleting subtask branch: {branch}")
            run_command(
                ["git", "push", "origin", "--delete", branch],
                cwd=cwd,
                check=False, timeout=30,
            )

    def _remote_branch_exists(self, worktree_path) -> bool:
        """Check if current branch exists on the remote (origin)."""
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        try:
            branch_result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree_path, check=False, timeout=10,
            )
            if branch_result.returncode != 0:
                return False
            branch = branch_result.stdout.strip()
            if branch in ("main", "master", "HEAD"):
                return False
            result = run_git_command(
                ["ls-remote", "--heads", "origin", branch],
                cwd=worktree_path, check=False, timeout=10,
            )
            return bool(result.stdout.strip())
        except SubprocessError:
            return False

    def get_changed_files(self) -> List[str]:
        """Get list of changed files from git diff (staged and unstaged)."""
        from ..utils.subprocess_utils import run_git_command, SubprocessError

        try:
            result = run_git_command(
                ["diff", "--name-only", "HEAD"],
                cwd=self.workspace,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                self.logger.debug(f"git diff failed: {result.stderr}")
                return []
            return [f.strip() for f in result.stdout.split("\n") if f.strip()]
        except SubprocessError:
            self.logger.warning("git diff timed out")
            return []
        except Exception as e:
            self.logger.debug(f"Failed to get changed files: {e}")
            return []

    def _is_at_terminal_workflow_step(self, task: Task) -> bool:
        """Check if the current agent is at the last step in the workflow DAG.

        Returns True for standalone tasks (no workflow) to preserve backward
        compatibility — standalone agents should always be allowed to create PRs.
        """
        workflow_name = task.context.get("workflow")
        if not workflow_name or workflow_name not in self._workflows_config:
            return True

        workflow_def = self._workflows_config[workflow_name]
        try:
            dag = workflow_def.to_dag(workflow_name)
        except Exception:
            return True

        # Prefer explicit workflow_step from chain context
        step_id = task.context.get("workflow_step")
        if step_id and step_id in dag.steps:
            return dag.is_terminal_step(step_id)

        # Fallback: find the step for this agent's base_id
        for step in dag.steps.values():
            if step.agent == self.config.base_id:
                return dag.is_terminal_step(step.id)

        return True

    def _sync_jira_status(self, task: Task, target_status: str, comment: Optional[str] = None) -> None:
        """Transition a JIRA ticket to target_status if all preconditions are met.

        Deterministic framework-level JIRA updates — agents don't reliably call
        MCP tools, so the framework ensures tickets reflect actual progress.
        """
        jira_key = task.context.get("jira_key")
        if not jira_key:
            return
        if not self.jira_client:
            return
        if not self._agent_definition or not self._agent_definition.jira_can_update_status:
            return
        if target_status not in (self._agent_definition.jira_allowed_transitions or []):
            self.logger.warning(
                f"Transition '{target_status}' not in allowed transitions for {self.config.id}, skipping"
            )
            return

        try:
            self.jira_client.transition_ticket(jira_key, target_status)
            self.logger.info(f"JIRA {jira_key} → {target_status}")
            if comment:
                self.jira_client.add_comment(jira_key, comment)
        except Exception as e:
            self.logger.warning(f"Failed to transition JIRA {jira_key} to '{target_status}': {e}")

    @property
    def active_worktree(self) -> Optional[Path]:
        """Get the currently active worktree path."""
        return self._active_worktree

    @active_worktree.setter
    def active_worktree(self, path: Optional[Path]) -> None:
        """Set the active worktree path."""
        self._active_worktree = path
