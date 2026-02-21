"""Git/PR/Worktree operations extracted from Agent class.

This module contains all git, pull request, and worktree lifecycle operations
that were previously embedded in the Agent class. By extracting these operations,
we achieve better separation of concerns and make the Agent class more focused.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..workspace.worktree_manager import WorktreeManager
    from ..queue.file_queue import FileQueue
    from .pr_lifecycle import PRLifecycleManager

from .task import Task
from .task_manifest import load_manifest, get_or_create_manifest
from ..utils.type_helpers import strip_chain_prefixes


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
        self._worktree_env_vars: Optional[Dict[str, str]] = None
        self._active_root_task_id: Optional[str] = None

    def safety_commit(self, working_dir: Path, reason: str) -> bool:
        """Best-effort commit of uncommitted changes. Returns True if committed.

        Never raises — safe for finally blocks and error paths.
        Uses [auto-commit] prefix to distinguish framework commits from LLM commits.
        """
        from ..utils.subprocess_utils import run_git_command

        try:
            if not working_dir.exists():
                return False

            status = run_git_command(
                ["status", "--porcelain"],
                cwd=working_dir, check=False, timeout=10,
            )
            if status.returncode != 0 or not status.stdout.strip():
                return False

            self._verify_manifest_branch(working_dir)

            add_result = run_git_command(
                ["add", "-A"],
                cwd=working_dir, check=False, timeout=10,
            )
            if add_result.returncode != 0:
                self.logger.debug(f"safety_commit: git add failed: {add_result.stderr}")
                return False

            commit_result = run_git_command(
                ["commit", "-m", f"[auto-commit] {reason}"],
                cwd=working_dir, check=False, timeout=10,
            )
            if commit_result.returncode != 0:
                self.logger.debug(f"safety_commit: git commit failed: {commit_result.stderr}")
                return False

            self.logger.info(f"safety_commit: {reason}")
            return True
        except Exception as e:
            self.logger.debug(f"safety_commit failed (non-fatal): {e}")
            return False

    @property
    def worktree_env_vars(self) -> Optional[Dict[str, str]]:
        """Env vars for the active worktree's virtualenv (PATH, VIRTUAL_ENV)."""
        return self._worktree_env_vars

    def _setup_worktree_venv(self, worktree_path: Path) -> None:
        """Create a per-worktree virtualenv for Python projects.

        Non-fatal: failure is logged but doesn't block the worktree.
        """
        try:
            from ..workspace.venv_manager import VenvManager
            self._worktree_env_vars = VenvManager().setup_venv(worktree_path)
        except Exception as e:
            self.logger.warning(f"Venv setup skipped: {e}")
            self._worktree_env_vars = None

    def _write_manifest_if_needed(self, task: Task, branch_name: str, working_dir: Path) -> None:
        """Write a task manifest if one doesn't exist yet for this root task.

        Called after branch checkout to record the canonical branch. The manifest
        is write-once: if another agent already created one, this is a no-op.
        """
        try:
            root_task_id = task.root_id
            self._active_root_task_id = root_task_id
            get_or_create_manifest(
                self.workspace,
                root_task_id=root_task_id,
                branch=branch_name,
                github_repo=task.context.get("github_repo"),
                user_goal=task.context.get("user_goal", task.description),
                workflow=task.context.get("workflow", "default"),
                working_directory=str(working_dir),
                created_by=self.config.id,
            )
        except Exception as e:
            self.logger.debug("Failed to write task manifest (non-fatal): %s", e)

    def _verify_manifest_branch(self, working_dir: Path) -> None:
        """Verify HEAD matches the manifest branch; correct if mismatched.

        Called from safety_commit() to prevent committing to the wrong branch
        when an external checkout changed HEAD. Never raises.
        """
        from ..utils.subprocess_utils import run_git_command

        if not self._active_root_task_id:
            return

        try:
            manifest = load_manifest(self.workspace, self._active_root_task_id)
            if not manifest or not manifest.branch:
                return

            result = run_git_command(
                ["branch", "--show-current"],
                cwd=working_dir, check=False, timeout=10,
            )
            if result.returncode != 0:
                return

            current = result.stdout.strip()
            if current == manifest.branch:
                return

            self.logger.warning(
                "Branch mismatch: HEAD=%s, manifest=%s — correcting",
                current, manifest.branch,
            )
            checkout = run_git_command(
                ["checkout", manifest.branch],
                cwd=working_dir, check=False, timeout=10,
            )
            if checkout.returncode != 0:
                self.logger.warning(
                    "Failed to checkout manifest branch %s: %s",
                    manifest.branch, checkout.stderr,
                )
                return
            if self.session_logger:
                self.session_logger.log(
                    "manifest_branch_correction",
                    expected=manifest.branch,
                    actual=current,
                    root_task_id=self._active_root_task_id,
                )
        except Exception as e:
            self.logger.debug("Manifest branch verification failed (non-fatal): %s", e)

    def get_working_directory(self, task: Task) -> Path:
        """Get working directory for task (worktree, target repo, or framework workspace).

        Priority:
        0. PR creation tasks with an implementation_branch skip worktree entirely
        1. If worktree mode enabled (config or task override), create isolated worktree
        2. If multi_repo_manager available, use shared clone
        3. Fall back to framework workspace
        """
        # Reset from previous task — non-worktree paths don't call _setup_worktree_venv
        self._worktree_env_vars = None

        github_repo = task.context.get("github_repo")

        # PR creation tasks that reference an upstream implementation branch
        # work directly from the workspace — no worktree or clone needed
        if task.context.get("pr_creation_step") and task.context.get("implementation_branch"):
            self.logger.info("PR creation task — using workspace (no worktree needed)")
            return self.workspace

        # Check if worktree mode should be used
        use_worktree = self._should_use_worktree(task)

        if use_worktree and github_repo and self.worktree_manager:
            # Get base repo path (shared clone or explicit override)
            base_repo = self._get_base_repo_for_worktree(task, github_repo)

            if base_repo:
                # Reuse branch only if explicitly set for this step, or if
                # the upstream implementation_branch belongs to the same agent role
                branch_name = task.context.get("worktree_branch")
                start_point = None
                is_chain = task.context.get("chain_step", False)

                if not branch_name:
                    impl_branch = task.context.get("implementation_branch")
                    if impl_branch and (self._is_own_branch(impl_branch) or is_chain):
                        branch_name = impl_branch
                    else:
                        jira_key = task.context.get("jira_key", "task")
                        task_hash = hashlib.sha256(task.id.encode()).hexdigest()[:8]
                        branch_name = f"agent/{self.config.id}/{jira_key}-{task_hash}"
                        # Base new branch on upstream engineer's code
                        if impl_branch:
                            start_point = impl_branch

                # Reload registry so we see worktrees created by other agent processes
                self.worktree_manager.reload_registry()

                # Check registry for existing worktree on this branch (reuse or retry)
                existing = self.worktree_manager.find_worktree_by_branch(branch_name)
                if existing:
                    if existing.exists():
                        self._active_worktree = existing
                        self.worktree_manager.acquire_worktree(existing, self.config.id)
                        self._setup_worktree_venv(existing)
                        task.context["worktree_branch"] = branch_name
                        self._write_manifest_if_needed(task, branch_name, existing)
                        self.logger.info(f"Reusing worktree for branch {branch_name}: {existing}")
                        return existing
                    else:
                        # Worktree deleted by another process — clear registry only.
                        # Don't call remove_worktree() which runs git worktree prune
                        # and could damage tracking for other active worktrees.
                        self.worktree_manager.remove_registry_entry_by_path(existing)
                        self.logger.warning(f"Worktree path missing, will recreate: {existing}")

                try:
                    # Chain steps share one worktree; use generic key so the
                    # path reads "chain-{root_id}" instead of "architect-{root_id}"
                    effective_agent_id = "chain" if is_chain else self.config.id

                    worktree_path = self.worktree_manager.create_worktree(
                        base_repo=base_repo,
                        branch_name=branch_name,
                        agent_id=effective_agent_id,
                        # Root ID keeps chain hops on the same worktree key
                        task_id=task.root_id,
                        owner_repo=github_repo,
                        start_point=start_point,
                        allow_cross_agent=is_chain,
                    )
                    self._active_worktree = worktree_path
                    # create_worktree already registered effective_agent_id;
                    # also acquire with our real config.id for proper ref counting
                    if effective_agent_id != self.config.id:
                        self.worktree_manager.acquire_worktree(worktree_path, self.config.id)
                    self._setup_worktree_venv(worktree_path)
                    task.context["worktree_branch"] = branch_name
                    self._write_manifest_if_needed(task, branch_name, worktree_path)
                    self.logger.info(f"Using worktree: {github_repo} at {worktree_path}")
                    return worktree_path
                except Exception as e:
                    self.logger.warning(f"Failed to create worktree, falling back to shared clone: {e}")
                    # Fall through to shared clone

        # Direct repo mode: work in the main workspace on a feature branch.
        # Avoids worktree creation entirely — the P0 worktree-vanishing issue
        # has blocked delivery in 8 consecutive observations.
        if github_repo:
            branch_name = task.context.get("worktree_branch") or task.context.get("implementation_branch")
            if not branch_name:
                impl_branch = task.context.get("implementation_branch")
                if impl_branch and (self._is_own_branch(impl_branch) or task.context.get("chain_step")):
                    branch_name = impl_branch
                else:
                    jira_key = task.context.get("jira_key", "task")
                    task_hash = hashlib.sha256(task.id.encode()).hexdigest()[:8]
                    branch_name = f"agent/{self.config.id}/{jira_key}-{task_hash}"

            self._checkout_or_create_branch(self.workspace, branch_name)
            task.context["worktree_branch"] = branch_name
            self._active_worktree = self.workspace
            self._write_manifest_if_needed(task, branch_name, self.workspace)
            self.logger.info(f"Working directly in {self.workspace} on branch {branch_name}")
            return self.workspace

        # No repo context, use framework workspace
        return self.workspace

    def detect_implementation_branch(self, task: Task) -> None:
        """Snapshot the current worktree branch into task context.

        Prefers the manifest branch (immutable) over git HEAD (mutable).
        Downstream chain steps use `implementation_branch` to check out the
        upstream branch instead of creating a fresh worktree from main.
        Skips if no active worktree or HEAD is main/master.
        """
        # Manifest takes precedence — immune to external git checkouts
        manifest = load_manifest(self.workspace, task.root_id)
        if manifest and manifest.branch and manifest.branch not in ("main", "master"):
            task.context["implementation_branch"] = manifest.branch
            return

        old_branch = task.context.get("implementation_branch")

        if not self._active_worktree:
            return

        from ..utils.subprocess_utils import run_git_command

        try:
            result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self._active_worktree,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                return
            branch = result.stdout.strip()
            if branch in ("main", "master", "HEAD"):
                return
            task.context["implementation_branch"] = branch
            if old_branch and old_branch != branch:
                self.logger.info(
                    f"Updated implementation branch: {old_branch} → {branch}"
                )
            else:
                self.logger.info(f"Detected implementation branch: {branch}")
        except Exception as e:
            self.logger.debug(f"Failed to detect implementation branch: {e}")

    def _detect_default_branch(self, working_dir) -> str:
        """Detect the default branch (main/master) for origin."""
        from ..utils.subprocess_utils import run_git_command

        try:
            result = run_git_command(
                ["symbolic-ref", "refs/remotes/origin/HEAD"],
                cwd=working_dir, check=False, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("/")[-1]
        except Exception:
            pass

        # Probe common names
        for branch in ["main", "master"]:
            try:
                probe = run_git_command(
                    ["rev-parse", "--verify", f"origin/{branch}"],
                    cwd=working_dir, check=False, timeout=10,
                )
                if probe.returncode == 0:
                    return branch
            except Exception:
                continue
        return "main"

    def discover_branch_work(self, working_dir) -> Optional[Dict]:
        """Discover committed work on the current branch beyond origin's default.

        Used at retry startup to give the LLM awareness of code from
        previous attempts that was committed but not captured by the
        truncated diff/summary in retry context.

        Returns a dict with commit_count, insertions, deletions, commit_log,
        file_list, and diffstat — or None if there's nothing to report.
        """
        from ..utils.subprocess_utils import run_git_command

        try:
            # What branch are we on?
            head_result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=working_dir, check=False, timeout=10,
            )
            if head_result.returncode != 0:
                return None
            branch = head_result.stdout.strip()
            if branch in ("main", "master", "HEAD"):
                return None

            default_branch = self._detect_default_branch(working_dir)
            range_spec = f"origin/{default_branch}..HEAD"

            # Commit log (capped at 20 entries, 2KB)
            log_result = run_git_command(
                ["log", range_spec, "--oneline", "-20"],
                cwd=working_dir, check=False, timeout=10,
            )
            if log_result.returncode != 0 or not log_result.stdout.strip():
                return None
            commit_log = log_result.stdout.strip()[:2048]

            # Diffstat (capped at 2KB)
            stat_result = run_git_command(
                ["diff", "--stat", range_spec],
                cwd=working_dir, check=False, timeout=10,
            )
            diffstat = (stat_result.stdout.strip()[:2048]
                        if stat_result.returncode == 0 else "")

            # Parse insertions/deletions from the summary line
            insertions = 0
            deletions = 0
            if diffstat:
                summary_line = diffstat.split("\n")[-1]
                import re
                ins_match = re.search(r"(\d+) insertion", summary_line)
                del_match = re.search(r"(\d+) deletion", summary_line)
                if ins_match:
                    insertions = int(ins_match.group(1))
                if del_match:
                    deletions = int(del_match.group(1))

            # File list (capped at 50 entries)
            names_result = run_git_command(
                ["diff", "--name-only", range_spec],
                cwd=working_dir, check=False, timeout=10,
            )
            file_list = []
            if names_result.returncode == 0 and names_result.stdout.strip():
                file_list = [
                    f for f in names_result.stdout.strip().split("\n") if f
                ][:50]

            commit_count = len(commit_log.split("\n"))

            return {
                "commit_count": commit_count,
                "insertions": insertions,
                "deletions": deletions,
                "commit_log": commit_log,
                "file_list": file_list,
                "diffstat": diffstat,
            }

        except Exception as e:
            self.logger.debug(f"Failed to discover branch work: {e}")
            return None

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
            # Skip non-agent directories (completed, failed, locks, etc.)
            queue_id = agent_dir.name
            if queue_id in ("completed", "failed", "locks", "heartbeats", "malformed"):
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    data = json.loads(task_file.read_text())
                    synced_task = Task(**data)

                    # Reject LLM-created orphans — framework tasks always have chain_step or parent_task_id
                    if not synced_task.context.get("chain_step") and synced_task.parent_task_id is None:
                        self.logger.warning(
                            f"Rejecting orphaned task {synced_task.id} ({synced_task.title}) "
                            f"from worktree sync (missing chain_step and parent_task_id)"
                        )
                        task_file.unlink(missing_ok=True)
                        continue

                    # Dedup: skip tasks already queued or completed
                    task_file_in_queue = self.queue.queue_dir / synced_task.assigned_to / f"{synced_task.id}.json"
                    completed_file = self.queue.completed_dir / f"{synced_task.id}.json"
                    if task_file_in_queue.exists() or completed_file.exists():
                        self.logger.info(
                            f"Skipping already-existing task {synced_task.id} during worktree sync"
                        )
                        task_file.unlink(missing_ok=True)
                        continue

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

        # Intermediate steps keep worktree active/protected so it survives
        # eviction during the gap before the next step picks it up.
        # Root plan tasks have workflow= but no chain_step= — they still
        # need the worktree for downstream implement/review/qa steps.
        has_downstream_steps = not self._is_at_terminal_workflow_step(task)
        # Subtasks that inherit a workflow should keep the worktree active
        # at non-terminal steps — the fan-in task handles terminal cleanup
        is_intermediate = (
            has_downstream_steps
            and (task.context.get("chain_step") or task.context.get("workflow"))
        )
        if is_intermediate:
            self.worktree_manager.touch_worktree(self._active_worktree)
            self.logger.debug(f"Intermediate chain step — kept worktree active: {self._active_worktree}")
            self._active_worktree = None
            return

        # Terminal or standalone — push any unpushed commits, then release
        # our ref. Physical deletion is deferred to cleanup_orphaned_worktrees()
        # which runs at startup and periodically with full safety checks.
        if self.worktree_manager.has_unpushed_commits(self._active_worktree):
            if self._try_push_worktree_branch(self._active_worktree):
                self.logger.info("Pushed unpushed commits during cleanup")

        self.worktree_manager.mark_worktree_inactive(self._active_worktree, user_id=self.config.id)
        self.logger.debug(f"Released worktree (deferred cleanup): {self._active_worktree}")

        self._active_worktree = None

    def push_and_create_pr_if_needed(self, task: Task) -> None:
        """Push branch and create PR if the agent produced unpushed commits.

        Runs after workflow chain routing. Downstream agents poll asynchronously,
        so the push completes before they fetch the branch.

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

        # Only act if we have an active working directory with changes
        if not self._active_worktree:
            self.logger.debug(f"No active working directory for {task.id}, skipping PR creation")
            return

        has_unpushed = self._has_unpushed_commits(self._active_worktree)
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
                    self._log_push_event(branch, success=False, error=push_result.stderr)
                    self.logger.error(f"Failed to push branch: {push_result.stderr}")
                    return
                self._log_push_event(branch, success=True)

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
        pr_title = strip_chain_prefixes(task.title)[:70]

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

    def _checkout_or_create_branch(self, repo_dir: Path, branch: str) -> None:
        """Switch to branch, creating it if it doesn't exist.

        Handles dirty working tree by stashing before switch and popping after.
        """
        from ..utils.subprocess_utils import run_git_command

        # Check if already on the target branch
        head_result = run_git_command(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir, check=False, timeout=10,
        )
        if head_result.returncode == 0 and head_result.stdout.strip() == branch:
            return

        # Stash dirty state before switching
        status = run_git_command(
            ["status", "--porcelain"],
            cwd=repo_dir, check=False, timeout=10,
        )
        stashed = False
        if status.returncode == 0 and status.stdout.strip():
            stash_result = run_git_command(
                ["stash", "push", "-m", f"auto-stash before switching to {branch}"],
                cwd=repo_dir, check=False, timeout=10,
            )
            stashed = stash_result.returncode == 0

        # Try checkout (existing branch), else create
        checkout = run_git_command(
            ["checkout", branch],
            cwd=repo_dir, check=False, timeout=10,
        )
        if checkout.returncode != 0:
            create = run_git_command(
                ["checkout", "-b", branch],
                cwd=repo_dir, check=False, timeout=10,
            )
            if create.returncode != 0:
                self.logger.error(f"Failed to create branch {branch}: {create.stderr}")
                # Pop stash even on failure so we don't lose work
                if stashed:
                    run_git_command(
                        ["stash", "pop"], cwd=repo_dir, check=False, timeout=10,
                    )
                return

        # Restore stashed changes
        if stashed:
            run_git_command(
                ["stash", "pop"], cwd=repo_dir, check=False, timeout=10,
            )

    def _has_unpushed_commits(self, working_dir: Path) -> bool:
        """Check if current branch has commits not pushed to origin."""
        from ..utils.subprocess_utils import run_git_command

        try:
            # Fast path: tracking branch exists → compare directly
            result = run_git_command(
                ["rev-list", "--count", "@{u}..HEAD"],
                cwd=working_dir, check=False, timeout=10,
            )
            if result.returncode == 0:
                count = int(result.stdout.strip() or "0")
                return count > 0
        except (ValueError, Exception):
            pass

        # Fallback: no tracking branch or parse error
        branch_result = run_git_command(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=working_dir, check=False, timeout=10,
        )
        if branch_result.returncode != 0:
            return False
        branch = branch_result.stdout.strip()
        if branch in ("main", "master", "HEAD"):
            return False

        # Check if branch exists on remote
        remote_check = run_git_command(
            ["ls-remote", "--heads", "origin", branch],
            cwd=working_dir, check=False, timeout=10,
        )
        if not remote_check.stdout.strip():
            # Branch not on remote — any local commits count as unpushed
            log_result = run_git_command(
                ["log", "--oneline", "-1"],
                cwd=working_dir, check=False, timeout=10,
            )
            return log_result.returncode == 0 and bool(log_result.stdout.strip())

        # Branch exists on remote but no tracking — compare local vs remote HEAD
        local_rev = run_git_command(
            ["rev-parse", "HEAD"],
            cwd=working_dir, check=False, timeout=10,
        )
        remote_rev = run_git_command(
            ["rev-parse", f"origin/{branch}"],
            cwd=working_dir, check=False, timeout=10,
        )
        if local_rev.returncode != 0 or remote_rev.returncode != 0:
            return True  # Assume unpushed if we can't determine
        return local_rev.stdout.strip() != remote_rev.stdout.strip()

    def _is_own_branch(self, branch_name: str) -> bool:
        """Check if a branch was created by this agent (or a replica of the same role).

        Matches patterns like:
          - agent/engineer/PROJ-123-abc  (same agent)
          - agent/engineer-2/PROJ-123    (replica of same base role)
        Does NOT match:
          - agent/architect/PROJ-123     (different agent role)
        """
        base = f"agent/{self.config.base_id}"
        if not branch_name.startswith(base):
            return False
        rest = branch_name[len(base):]
        # Exact base_id: rest starts with "/"
        # Replica (e.g., engineer-2): rest starts with "-<digit>/"
        return rest.startswith("/") or (
            rest.startswith("-") and "/" in rest and rest[1:rest.index("/")].isdigit()
        )

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

    def _try_push_worktree_branch(self, worktree_path: Path) -> bool:
        """Best-effort push of the worktree's branch to origin.

        Prevents worktree accumulation by pushing local-only commits so the
        worktree can be safely removed afterward.

        Returns True if push succeeded, False on any failure.
        """
        from ..utils.subprocess_utils import run_git_command

        try:
            result = run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                cwd=worktree_path, check=False, timeout=10,
            )
            if result.returncode != 0:
                return False
            branch = result.stdout.strip()
            if branch in ("main", "master", "HEAD"):
                return False

            push_result = run_git_command(
                ["push", "origin", branch],
                cwd=worktree_path, check=False, timeout=60,
            )
            return push_result.returncode == 0
        except Exception as e:
            self.logger.debug(f"Best-effort push failed for {worktree_path}: {e}")
            return False

    def push_if_unpushed(self) -> bool:
        """Push the active working directory's branch if it has unpushed commits.

        Called immediately after LLM returns to ensure committed work
        reaches the remote before any corruption can destroy it.
        Returns True if push succeeded, False otherwise (including no-op).
        """
        if not self._active_worktree:
            return False
        if not self._active_worktree.exists():
            return False
        if not self._has_unpushed_commits(self._active_worktree):
            return False
        pushed = self._try_push_worktree_branch(self._active_worktree)
        if pushed:
            self.logger.info("Pushed unpushed commits to remote (post-LLM safety push)")
        return pushed

    def _log_push_event(self, branch: str, success: bool, error: Optional[str] = None) -> None:
        """Log a git push event to the session logger for metrics collection."""
        if not self.session_logger:
            return
        data = {"branch": branch, "success": success}
        if error:
            data["error"] = error[:500]
        self.session_logger.log("git_push", **data)

    @property
    def active_worktree(self) -> Optional[Path]:
        """Get the currently active worktree path."""
        return self._active_worktree

    @active_worktree.setter
    def active_worktree(self, path: Optional[Path]) -> None:
        """Set the active worktree path."""
        self._active_worktree = path
