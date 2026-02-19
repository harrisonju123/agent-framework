"""Autonomous PR lifecycle management: CI polling, fix loops, merge.

After a PR is created, PRLifecycleManager monitors CI, queues fix tasks
for the engineer on failure, rebases on merge conflicts, and squash-merges
when everything is green.  Per-repo opt-in via `auto_merge: true`.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.subprocess_utils import run_command, run_git_command, SubprocessError
from .task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)

VALID_MERGE_STRATEGIES = frozenset({"squash", "merge", "rebase"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class CIStatus(str, Enum):
    PASSING = "passing"
    FAILING = "failing"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class CICheckResult:
    status: CIStatus
    failed_checks: List[str]
    failure_logs: str  # Truncated to max_log_chars


# ---------------------------------------------------------------------------
# PRLifecycleManager
# ---------------------------------------------------------------------------

class PRLifecycleManager:
    """Orchestrates the post-PR-creation lifecycle for auto_merge repos.

    Collaborator of Agent — not a subclass.  Receives the queue and config
    it needs at construction time, called from Agent._manage_pr_lifecycle().
    """

    MAX_LOG_CHARS = 3000

    def __init__(
        self,
        queue,
        workspace: Path,
        repo_configs: Dict[str, dict],
        pr_lifecycle_config: Optional[dict] = None,
        logger_instance=None,
        multi_repo_manager=None,
    ):
        self._queue = queue
        self._workspace = Path(workspace)
        # repo_configs: {"owner/repo": {config dict}} — auto_merge, merge_strategy, etc.
        self._repo_configs = repo_configs
        self._log = logger_instance or logger
        self._multi_repo_manager = multi_repo_manager

        cfg = pr_lifecycle_config or {}
        self._ci_poll_interval = cfg.get("ci_poll_interval", 30)
        self._ci_poll_max_wait = cfg.get("ci_poll_max_wait", 1200)
        self._default_max_ci_fix_attempts = cfg.get("max_ci_fix_attempts", 3)
        self._auto_approve = cfg.get("auto_approve", True)
        self._delete_branch_on_merge = cfg.get("delete_branch_on_merge", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_manage(self, task: Task) -> bool:
        """Decide whether this task's PR should be autonomously managed."""
        if task.parent_task_id is not None:
            return False

        pr_url = task.context.get("pr_url")
        if not pr_url:
            return False

        github_repo = task.context.get("github_repo")
        if not github_repo:
            return False

        repo_cfg = self._repo_configs.get(github_repo, {})
        return bool(repo_cfg.get("auto_merge", False))

    def manage(self, task: Task, agent_id: str) -> bool:
        """Run the full PR lifecycle.  Returns True if merged, False otherwise."""
        pr_url = task.context["pr_url"]
        github_repo = task.context["github_repo"]
        pr_number = self._extract_pr_number(pr_url)
        repo_cfg = self._repo_configs.get(github_repo, {})
        max_ci_fixes = repo_cfg.get(
            "max_ci_fix_attempts", self._default_max_ci_fix_attempts
        )
        merge_strategy = repo_cfg.get("merge_strategy", "squash")

        ci_fix_count = task.context.get("ci_fix_count", 0)

        self._log.info(
            f"PR lifecycle: monitoring {pr_url} "
            f"(ci_fix_count={ci_fix_count}/{max_ci_fixes})"
        )

        # 1. Poll CI
        ci_result = self._poll_ci_checks(github_repo, pr_number)

        if ci_result.status in (CIStatus.FAILING, CIStatus.ERROR):
            if ci_fix_count < max_ci_fixes:
                self._create_ci_fix_task(
                    task, ci_result, ci_fix_count + 1, agent_id
                )
                return False
            else:
                self._escalate_ci_failure(
                    task, ci_result, ci_fix_count, agent_id
                )
                return False

        if ci_result.status == CIStatus.PENDING:
            self._log.warning(
                f"CI timed out (still pending after {self._ci_poll_max_wait}s) "
                f"for {pr_url}, leaving PR open for manual review"
            )
            return False

        # 2. Check merge conflicts
        if self._has_merge_conflicts(github_repo, pr_number):
            rebase_ok = self._rebase_on_main(task, github_repo, pr_number)
            if not rebase_ok:
                self._log.warning(
                    f"Rebase failed for {pr_url}, leaving PR open for manual resolution"
                )
                return False
            # Re-poll CI after rebase (force-push triggers new CI run)
            ci_result = self._poll_ci_checks(github_repo, pr_number)
            if ci_result.status != CIStatus.PASSING:
                self._log.warning(
                    f"CI not passing after rebase ({ci_result.status.value}), "
                    "leaving PR open"
                )
                return False

        # 3. Approve
        if self._auto_approve:
            self._approve_pr(github_repo, pr_number)

        # 4. Merge
        merged = self._merge_pr(
            github_repo, pr_number, strategy=merge_strategy
        )
        if merged:
            self._log.info(f"PR {pr_url} merged successfully")
        return merged

    # ------------------------------------------------------------------
    # CI polling
    # ------------------------------------------------------------------

    def _poll_ci_checks(
        self, github_repo: str, pr_number: str
    ) -> CICheckResult:
        """Poll CI checks until terminal state or timeout."""
        deadline = time.monotonic() + self._ci_poll_max_wait

        while time.monotonic() < deadline:
            result = self._fetch_ci_status(github_repo, pr_number)
            if result.status in (CIStatus.PASSING, CIStatus.FAILING, CIStatus.ERROR):
                return result
            time.sleep(self._ci_poll_interval)

        # Timed out while still pending
        return CICheckResult(
            status=CIStatus.PENDING,
            failed_checks=[],
            failure_logs="CI checks did not complete within timeout",
        )

    def _fetch_ci_status(
        self, github_repo: str, pr_number: str
    ) -> CICheckResult:
        """Single fetch of current CI check status via gh CLI."""
        try:
            result = run_command(
                [
                    "gh", "pr", "checks", pr_number,
                    "--repo", github_repo,
                    "--json", "name,state,bucket",
                ],
                check=False,
                timeout=30,
            )
        except SubprocessError:
            return CICheckResult(
                status=CIStatus.ERROR, failed_checks=[], failure_logs=""
            )

        # gh exits non-zero when checks fail but still writes valid JSON to stdout.
        # Parse first; only treat as infrastructure error when we have no usable output.
        try:
            checks = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return CICheckResult(
                status=CIStatus.ERROR,
                failed_checks=[],
                failure_logs=result.stderr[:self.MAX_LOG_CHARS] if result.returncode != 0 else "",
            )

        if not checks:
            # No CI checks configured — treat as passing
            return CICheckResult(
                status=CIStatus.PASSING, failed_checks=[], failure_logs=""
            )

        failed = []
        any_pending = False
        for check in checks:
            bucket = (check.get("bucket") or "").lower()

            if bucket == "pending":
                any_pending = True
            elif bucket == "fail":
                failed.append(check.get("name", "unknown"))

        if failed:
            logs = self._fetch_failure_logs(github_repo, pr_number, failed)
            return CICheckResult(
                status=CIStatus.FAILING,
                failed_checks=failed,
                failure_logs=logs[:self.MAX_LOG_CHARS],
            )

        if any_pending:
            return CICheckResult(
                status=CIStatus.PENDING, failed_checks=[], failure_logs=""
            )

        return CICheckResult(
            status=CIStatus.PASSING, failed_checks=[], failure_logs=""
        )

    def _fetch_failure_logs(
        self, github_repo: str, pr_number: str, failed_names: List[str]
    ) -> str:
        """Best-effort fetch of failure details from gh pr checks output."""
        try:
            result = run_command(
                ["gh", "pr", "checks", pr_number, "--repo", github_repo],
                check=False,
                timeout=30,
            )
            return result.stdout[:self.MAX_LOG_CHARS] if result.stdout else result.stderr[:self.MAX_LOG_CHARS]
        except SubprocessError:
            return f"Failed checks: {', '.join(failed_names)}"

    # ------------------------------------------------------------------
    # CI fix task creation
    # ------------------------------------------------------------------

    def _create_ci_fix_task(
        self,
        task: Task,
        ci_result: CICheckResult,
        count: int,
        agent_id: str,
    ) -> None:
        """Queue an engineer task to fix CI failures."""
        fix_task_id = f"ci-fix-{task.id[:12]}-c{count}"

        failed_names = ", ".join(ci_result.failed_checks) or "unknown checks"
        description = (
            f"CI checks failing on PR {task.context['pr_url']}.\n\n"
            f"Failed checks: {failed_names}\n\n"
            f"Failure logs (truncated):\n```\n{ci_result.failure_logs}\n```\n\n"
            f"Fix attempt {count}. Fix the failures and push to the same branch."
        )

        # Propagate branch/repo/PR context so the engineer works on the same branch
        fix_context = {
            "github_repo": task.context.get("github_repo"),
            "pr_url": task.context.get("pr_url"),
            "implementation_branch": task.context.get("implementation_branch")
            or task.context.get("worktree_branch"),
            "workflow": task.context.get("workflow"),
            "jira_key": task.context.get("jira_key"),
            "ci_fix_count": count,
            "ci_fix_parent_task_id": task.id,
        }

        fix_task = Task(
            id=fix_task_id,
            type=TaskType.FIX,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=agent_id,
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title=f"Fix CI failures (attempt {count}): {failed_names[:50]}",
            description=description,
            context=fix_context,
        )

        self._queue.push(fix_task, "engineer")
        self._log.info(
            f"Queued CI fix task {fix_task_id} (attempt {count}) "
            f"for {task.context['pr_url']}"
        )

    def _escalate_ci_failure(
        self,
        task: Task,
        ci_result: CICheckResult,
        count: int,
        agent_id: str,
    ) -> None:
        """Escalate to architect after max CI fix attempts exhausted."""
        escalation_id = f"ci-escalation-{task.id[:12]}"
        failed_names = ", ".join(ci_result.failed_checks) or "unknown checks"
        description = (
            f"CI failures on PR {task.context['pr_url']} could not be "
            f"resolved after {count} fix attempts.\n\n"
            f"Failed checks: {failed_names}\n\n"
            f"Last failure logs:\n```\n{ci_result.failure_logs}\n```\n\n"
            "Needs architectural review or manual intervention."
        )

        escalation_task = Task(
            id=escalation_id,
            type=TaskType.ESCALATION,
            status=TaskStatus.PENDING,
            # Lower number = higher priority in this codebase
            priority=max(1, task.priority - 1),
            created_by=agent_id,
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title=f"CI failure escalation: {task.title[:50]}",
            description=description,
            context={
                "github_repo": task.context.get("github_repo"),
                "pr_url": task.context.get("pr_url"),
                "jira_key": task.context.get("jira_key"),
                "ci_fix_count": count,
                "ci_fix_parent_task_id": task.id,
            },
        )

        self._queue.push(escalation_task, "architect")
        self._log.warning(
            f"CI fix attempts exhausted ({count}), escalated {escalation_id}"
        )

    # ------------------------------------------------------------------
    # Merge conflict handling
    # ------------------------------------------------------------------

    def _has_merge_conflicts(self, github_repo: str, pr_number: str) -> bool:
        """Check if the PR has merge conflicts via gh pr view."""
        try:
            result = run_command(
                [
                    "gh", "pr", "view", pr_number,
                    "--repo", github_repo,
                    "--json", "mergeable",
                ],
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                return False
            data = json.loads(result.stdout)
            return data.get("mergeable") == "CONFLICTING"
        except (SubprocessError, json.JSONDecodeError):
            return False

    def _resolve_repo_cwd(self, github_repo: str) -> Path:
        """Get the local checkout path for a repo via multi_repo_manager."""
        if self._multi_repo_manager:
            try:
                return Path(self._multi_repo_manager.ensure_repo(github_repo))
            except Exception:
                pass
        return self._workspace

    def _rebase_on_main(
        self, task: Task, github_repo: str, pr_number: str
    ) -> bool:
        """Rebase the PR branch on main.  Returns True on success."""
        branch = (
            task.context.get("implementation_branch")
            or task.context.get("worktree_branch")
        )
        if not branch:
            self._log.warning("No branch in task context, cannot rebase")
            return False

        cwd = self._resolve_repo_cwd(github_repo)

        try:
            run_git_command(["fetch", "origin", "main"], cwd=cwd, timeout=60)
            run_git_command(
                ["checkout", branch], cwd=cwd, check=False, timeout=30
            )
            result = run_git_command(
                ["rebase", "origin/main"], cwd=cwd, check=False, timeout=120
            )
            if result.returncode != 0:
                # Abort the failed rebase
                run_git_command(
                    ["rebase", "--abort"], cwd=cwd, check=False, timeout=30
                )
                self._log.warning(f"Rebase failed for branch {branch}, aborted")
                return False

            run_git_command(
                ["push", "--force-with-lease"], cwd=cwd, timeout=60
            )
            self._log.info(f"Rebased {branch} on main and force-pushed")
            return True
        except SubprocessError as e:
            self._log.error(f"Rebase error: {e}")
            # Best-effort abort
            try:
                run_git_command(
                    ["rebase", "--abort"], cwd=cwd, check=False, timeout=30
                )
            except SubprocessError:
                pass
            return False

    # ------------------------------------------------------------------
    # PR approval and merge
    # ------------------------------------------------------------------

    def _approve_pr(self, github_repo: str, pr_number: str) -> None:
        """Approve the PR via gh CLI (satisfies branch protection reviewer req)."""
        try:
            run_command(
                [
                    "gh", "pr", "review", pr_number,
                    "--repo", github_repo,
                    "--approve",
                    "--body", "Approved by agent framework (CI passing, QA verified)",
                ],
                check=False,
                timeout=30,
            )
            self._log.info(f"Approved PR #{pr_number}")
        except SubprocessError as e:
            self._log.warning(f"Failed to approve PR #{pr_number}: {e}")

    def _merge_pr(
        self, github_repo: str, pr_number: str, *, strategy: str = "squash"
    ) -> bool:
        """Merge the PR.  Returns True on success."""
        if strategy not in VALID_MERGE_STRATEGIES:
            self._log.error(
                f"Invalid merge strategy '{strategy}', expected one of {sorted(VALID_MERGE_STRATEGIES)}"
            )
            return False

        cmd = [
            "gh", "pr", "merge", pr_number,
            "--repo", github_repo,
            f"--{strategy}",
            "--auto",
        ]
        if self._delete_branch_on_merge:
            cmd.append("--delete-branch")

        try:
            result = run_command(cmd, check=False, timeout=30)
            if result.returncode == 0:
                self._log.info(f"Merged PR #{pr_number} ({strategy})")
                return True
            self._log.error(
                f"Failed to merge PR #{pr_number}: {result.stderr}"
            )
            return False
        except SubprocessError as e:
            self._log.error(f"Merge error for PR #{pr_number}: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pr_number(pr_url: str) -> str:
        """Extract PR number from GitHub URL (e.g. .../pull/42 → '42')."""
        return pr_url.rstrip("/").split("/")[-1]
