"""Parallel subtask execution manager.

Orchestrates concurrent execution of independent subtasks in isolated
worktrees. Uses dependency-tier partitioning (topological sort on
depends_on_subtasks) to run independent subtasks in parallel while
respecting ordering constraints.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..workspace.worktree_manager import WorktreeManager
    from .config import ParallelSubtasksConfig
    from .session_logger import SessionLogger
    from .task import Task

logger = logging.getLogger(__name__)


@dataclass
class SubtaskResult:
    """Result of a single parallel subtask execution."""
    task_id: str
    success: bool
    branch_name: str
    worktree_path: Optional[Path] = None
    return_code: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class TierResult:
    """Result of executing all subtasks in a single dependency tier."""
    tier: int
    results: List[SubtaskResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def failed_tasks(self) -> List[SubtaskResult]:
        return [r for r in self.results if not r.success]


@dataclass
class MergeResult:
    """Result of merging parallel subtask branches."""
    success: bool
    merged_branches: List[str] = field(default_factory=list)
    conflict_files: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ParallelExecutionManager:
    """Manages concurrent execution of decomposed subtasks.

    Groups subtasks into dependency tiers via topological sort, then
    executes each tier in parallel using isolated worktrees and
    subprocess-based agent invocations.
    """

    def __init__(
        self,
        config: "ParallelSubtasksConfig",
        workspace: Path,
        worktree_manager: "WorktreeManager",
        session_logger: Optional["SessionLogger"] = None,
    ):
        self._config = config
        self._workspace = workspace
        self._worktree_manager = worktree_manager
        self._session_logger = session_logger
        self._child_processes: List[subprocess.Popen] = []

    @staticmethod
    def partition_into_tiers(subtasks: List["Task"]) -> List[List["Task"]]:
        """Group subtasks into dependency tiers via topological sort.

        Tier 0 contains subtasks with no dependencies. Tier N contains
        subtasks whose dependencies are all in tiers < N.

        Args:
            subtasks: List of subtasks with depends_on fields.

        Returns:
            List of tiers, each tier is a list of tasks that can run concurrently.
        """
        if not subtasks:
            return []

        # Build ID -> task mapping
        task_map: Dict[str, "Task"] = {t.id: t for t in subtasks}
        # Track which tier each task belongs to
        assigned: Dict[str, int] = {}

        # Iteratively assign tiers
        remaining = set(task_map.keys())
        tier_num = 0
        tiers: List[List["Task"]] = []

        while remaining:
            # Find tasks whose deps are all assigned (or have no deps)
            ready = []
            for task_id in remaining:
                task = task_map[task_id]
                deps = task.depends_on or []
                # Only consider deps that are within our subtask set
                relevant_deps = [d for d in deps if d in task_map]
                if all(d in assigned for d in relevant_deps):
                    ready.append(task_id)

            if not ready:
                # Circular dependency — force-assign remaining to current tier
                logger.warning(
                    f"Circular dependency detected among subtasks: {remaining}. "
                    f"Force-assigning to tier {tier_num}."
                )
                ready = list(remaining)

            tier_tasks = [task_map[tid] for tid in ready]
            tiers.append(tier_tasks)
            for tid in ready:
                assigned[tid] = tier_num
                remaining.discard(tid)
            tier_num += 1

        return tiers

    async def execute_parallel_subtasks(
        self,
        parent_task: "Task",
        subtasks: List["Task"],
        owner_repo: str,
    ) -> List[TierResult]:
        """Execute subtasks in parallel, tier by tier.

        Args:
            parent_task: The parent task that was decomposed.
            subtasks: All subtasks from decomposition.
            owner_repo: Repository in 'owner/repo' format.

        Returns:
            List of TierResults, one per dependency tier.
        """
        tiers = self.partition_into_tiers(subtasks)

        if self._session_logger:
            self._session_logger.log(
                "parallel_execution_started",
                parent_task_id=parent_task.id,
                subtask_count=len(subtasks),
                tier_count=len(tiers),
                tier_sizes=[len(t) for t in tiers],
            )

        execution_start = time.monotonic()
        tier_results: List[TierResult] = []

        for tier_idx, tier_tasks in enumerate(tiers):
            tier_start = time.monotonic()

            # Cap concurrency to max_workers
            semaphore = asyncio.Semaphore(self._config.max_workers)
            tier_coros = [
                self._execute_subtask_with_semaphore(
                    semaphore, task, parent_task, owner_repo
                )
                for task in tier_tasks
            ]

            results = await asyncio.gather(*tier_coros, return_exceptions=True)

            tier_result = TierResult(
                tier=tier_idx,
                duration_ms=(time.monotonic() - tier_start) * 1000,
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    tier_result.results.append(SubtaskResult(
                        task_id=tier_tasks[i].id,
                        success=False,
                        branch_name=f"agent-parallel-{tier_tasks[i].id}",
                        error=str(result),
                    ))
                else:
                    tier_result.results.append(result)

            tier_results.append(tier_result)

            if self._session_logger:
                self._session_logger.log(
                    "parallel_tier_completed",
                    parent_task_id=parent_task.id,
                    tier=tier_idx,
                    duration_ms=tier_result.duration_ms,
                    subtask_ids=[t.id for t in tier_tasks],
                    success=tier_result.all_succeeded,
                )

            # Stop if tier failed — downstream tiers may depend on it
            if not tier_result.all_succeeded:
                logger.warning(
                    f"Tier {tier_idx} had failures: "
                    f"{[r.task_id for r in tier_result.failed_tasks]}. "
                    f"Stopping parallel execution."
                )
                break

        total_ms = (time.monotonic() - execution_start) * 1000
        serial_ms = sum(
            r.duration_ms
            for tr in tier_results
            for r in tr.results
        )
        speedup = serial_ms / total_ms if total_ms > 0 else 1.0

        if self._session_logger:
            self._session_logger.log(
                "parallel_execution_completed",
                parent_task_id=parent_task.id,
                total_duration_ms=total_ms,
                serial_duration_ms=serial_ms,
                speedup_ratio=round(speedup, 2),
                tiers_completed=len(tier_results),
                all_succeeded=all(tr.all_succeeded for tr in tier_results),
            )

        return tier_results

    async def _execute_subtask_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        task: "Task",
        parent_task: "Task",
        owner_repo: str,
    ) -> SubtaskResult:
        """Execute a single subtask within a concurrency-limited semaphore."""
        async with semaphore:
            return await self._execute_single_subtask(task, parent_task, owner_repo)

    async def _execute_single_subtask(
        self,
        task: "Task",
        parent_task: "Task",
        owner_repo: str,
    ) -> SubtaskResult:
        """Spawn a subprocess to execute a single subtask in an isolated worktree.

        Creates a worktree, launches `python -m agent_framework.run_agent
        <agent_id> --task <task_id>`, waits for completion, and returns results.
        """
        branch_name = f"agent-parallel-{task.id}"
        worktree_path = None
        start_time = time.monotonic()

        try:
            # Create isolated worktree for this subtask
            worktree_path = self._worktree_manager.create_worktree(
                base_repo=self._workspace,
                branch_name=branch_name,
                agent_id=task.assigned_to or "engineer",
                task_id=task.id,
                owner_repo=owner_repo,
            )

            # Spawn subprocess
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            env["AGENT_PARALLEL_MODE"] = "1"
            env["AGENT_TASK_ID"] = task.id
            env["AGENT_WORKTREE_PATH"] = str(worktree_path)

            agent_id = task.assigned_to or "engineer"
            cmd = [
                sys.executable, "-m", "agent_framework.run_agent",
                agent_id, "--task", task.id,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=worktree_path,
                env=env,
            )
            self._child_processes.append(proc)

            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._config.subprocess_timeout,
                )
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                return SubtaskResult(
                    task_id=task.id,
                    success=False,
                    branch_name=branch_name,
                    worktree_path=worktree_path,
                    return_code=-1,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    error="Subprocess timed out",
                )

            duration_ms = (time.monotonic() - start_time) * 1000
            success = proc.returncode == 0

            return SubtaskResult(
                task_id=task.id,
                success=success,
                branch_name=branch_name,
                worktree_path=worktree_path,
                return_code=proc.returncode or 0,
                duration_ms=duration_ms,
                error=stderr.decode(errors="replace")[:500] if not success else None,
            )

        except Exception as e:
            return SubtaskResult(
                task_id=task.id,
                success=False,
                branch_name=branch_name,
                worktree_path=worktree_path,
                duration_ms=(time.monotonic() - start_time) * 1000,
                error=str(e),
            )

    def merge_subtask_branches(
        self,
        target_branch: str,
        tier_results: List[TierResult],
        working_dir: Optional[Path] = None,
    ) -> MergeResult:
        """Merge completed subtask branches into the target branch.

        Uses sequential merge (--no-ff) by default. Falls back to
        reporting conflicts rather than silently dropping changes.

        Args:
            target_branch: Branch to merge into (usually parent's implementation branch).
            tier_results: Results from execute_parallel_subtasks.
            working_dir: Directory to run git commands in.

        Returns:
            MergeResult with success status and any conflicts.
        """
        from ..utils.subprocess_utils import run_git_command

        cwd = working_dir or self._workspace
        merged: List[str] = []
        conflicts: List[str] = []

        if self._session_logger:
            self._session_logger.log(
                "parallel_merge_started",
                target_branch=target_branch,
                branch_count=sum(
                    len(tr.results) for tr in tier_results
                    if tr.all_succeeded
                ),
            )

        # Ensure we're on the target branch
        run_git_command(
            ["checkout", target_branch],
            cwd=cwd, check=True, timeout=30,
        )

        for tier_result in tier_results:
            for result in tier_result.results:
                if not result.success:
                    continue

                branch = result.branch_name
                try:
                    merge_result = run_git_command(
                        ["merge", "--no-ff", "-m",
                         f"Merge parallel subtask {result.task_id}",
                         branch],
                        cwd=cwd, check=False, timeout=60,
                    )

                    if merge_result.returncode != 0:
                        # Conflict detected — abort and record
                        conflict_output = merge_result.stdout + merge_result.stderr
                        run_git_command(
                            ["merge", "--abort"],
                            cwd=cwd, check=False, timeout=10,
                        )
                        conflicts.append(branch)

                        if self._session_logger:
                            self._session_logger.log(
                                "parallel_merge_conflict",
                                branch=branch,
                                task_id=result.task_id,
                                conflict_output=conflict_output[:500],
                            )
                    else:
                        merged.append(branch)

                except Exception as e:
                    logger.error(f"Failed to merge branch {branch}: {e}")
                    conflicts.append(branch)

        success = len(conflicts) == 0
        return MergeResult(
            success=success,
            merged_branches=merged,
            conflict_files=conflicts,
            error=f"{len(conflicts)} branch(es) had merge conflicts" if conflicts else None,
        )

    def cleanup_worktrees(self, tier_results: List[TierResult]) -> None:
        """Remove worktrees created for parallel execution."""
        for tier_result in tier_results:
            for result in tier_result.results:
                if result.worktree_path and result.worktree_path.exists():
                    try:
                        self._worktree_manager.remove_worktree(result.worktree_path)
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up worktree {result.worktree_path}: {e}"
                        )

    def terminate_children(self) -> None:
        """Terminate all spawned child processes (for cleanup on crash)."""
        for proc in self._child_processes:
            try:
                if proc.returncode is None:
                    proc.terminate()
            except Exception:
                pass
        self._child_processes.clear()
