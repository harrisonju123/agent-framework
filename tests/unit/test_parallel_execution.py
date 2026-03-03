"""Tests for ParallelExecutionManager."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.parallel_execution import (
    MergeResult,
    ParallelExecutionManager,
    SubtaskResult,
    TierResult,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Fixtures --


def _make_task(task_id="task-1", depends_on=None, **ctx):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title=f"Subtask {task_id}",
        description="Do the thing.",
        depends_on=depends_on or [],
        context=ctx,
    )


def _make_config(enabled=True, max_workers=3, merge_strategy="sequential-merge", timeout=3600):
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.max_workers = max_workers
    cfg.merge_strategy = merge_strategy
    cfg.subprocess_timeout = timeout
    return cfg


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def worktree_manager():
    wm = MagicMock()
    wm.create_worktree.return_value = Path("/tmp/worktree-test")
    return wm


@pytest.fixture
def session_logger():
    return MagicMock()


@pytest.fixture
def manager(config, worktree_manager, session_logger, tmp_path):
    return ParallelExecutionManager(
        config=config,
        workspace=tmp_path,
        worktree_manager=worktree_manager,
        session_logger=session_logger,
    )


# -- Tier Partitioning --


class TestPartitionIntoTiers:
    def test_empty_subtasks(self):
        assert ParallelExecutionManager.partition_into_tiers([]) == []

    def test_all_independent(self):
        """All independent subtasks land in tier 0."""
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        tiers = ParallelExecutionManager.partition_into_tiers(tasks)
        assert len(tiers) == 1
        assert len(tiers[0]) == 3

    def test_linear_dependencies(self):
        """A -> B -> C creates 3 tiers."""
        a = _make_task("a")
        b = _make_task("b", depends_on=["a"])
        c = _make_task("c", depends_on=["b"])
        tiers = ParallelExecutionManager.partition_into_tiers([a, b, c])
        assert len(tiers) == 3
        assert [t.id for t in tiers[0]] == ["a"]
        assert [t.id for t in tiers[1]] == ["b"]
        assert [t.id for t in tiers[2]] == ["c"]

    def test_diamond_dependency(self):
        """Diamond: A -> B, A -> C, B+C -> D creates 3 tiers."""
        a = _make_task("a")
        b = _make_task("b", depends_on=["a"])
        c = _make_task("c", depends_on=["a"])
        d = _make_task("d", depends_on=["b", "c"])
        tiers = ParallelExecutionManager.partition_into_tiers([a, b, c, d])
        assert len(tiers) == 3
        assert [t.id for t in tiers[0]] == ["a"]
        # B and C should be in the same tier (order may vary)
        tier1_ids = sorted(t.id for t in tiers[1])
        assert tier1_ids == ["b", "c"]
        assert [t.id for t in tiers[2]] == ["d"]

    def test_external_dependencies_ignored(self):
        """Dependencies on tasks outside the subtask set are ignored."""
        a = _make_task("a", depends_on=["external-task-99"])
        b = _make_task("b")
        tiers = ParallelExecutionManager.partition_into_tiers([a, b])
        assert len(tiers) == 1
        assert len(tiers[0]) == 2

    def test_circular_dependency_forced_to_same_tier(self):
        """Circular deps are force-assigned to same tier with a warning."""
        a = _make_task("a", depends_on=["b"])
        b = _make_task("b", depends_on=["a"])
        tiers = ParallelExecutionManager.partition_into_tiers([a, b])
        # Both should be in the same tier (forced)
        assert len(tiers) == 1
        assert len(tiers[0]) == 2

    def test_mixed_independent_and_dependent(self):
        """Mix of independent and dependent subtasks."""
        a = _make_task("a")
        b = _make_task("b")
        c = _make_task("c", depends_on=["a"])
        tiers = ParallelExecutionManager.partition_into_tiers([a, b, c])
        assert len(tiers) == 2
        tier0_ids = sorted(t.id for t in tiers[0])
        assert tier0_ids == ["a", "b"]
        assert [t.id for t in tiers[1]] == ["c"]


# -- Execute Parallel Subtasks --


class TestExecuteParallelSubtasks:
    @pytest.mark.asyncio
    async def test_executes_all_independent_subtasks(self, manager):
        """All independent subtasks execute in a single tier."""
        tasks = [_make_task("a"), _make_task("b")]
        parent = _make_task("parent")

        with patch.object(manager, "_execute_single_subtask") as mock_exec:
            mock_exec.return_value = SubtaskResult(
                task_id="x", success=True, branch_name="branch-x",
                duration_ms=100,
            )

            results = await manager.execute_parallel_subtasks(parent, tasks, "owner/repo")

        assert len(results) == 1  # Single tier
        assert len(results[0].results) == 2
        assert results[0].all_succeeded

    @pytest.mark.asyncio
    async def test_stops_on_tier_failure(self, manager):
        """Stops executing subsequent tiers when a tier fails."""
        a = _make_task("a")
        b = _make_task("b", depends_on=["a"])
        parent = _make_task("parent")

        with patch.object(manager, "_execute_single_subtask") as mock_exec:
            mock_exec.return_value = SubtaskResult(
                task_id="a", success=False, branch_name="branch-a",
                error="Build failed",
            )

            results = await manager.execute_parallel_subtasks(parent, [a, b], "owner/repo")

        # Only tier 0 executed, tier 1 (b) was skipped
        assert len(results) == 1
        assert not results[0].all_succeeded

    @pytest.mark.asyncio
    async def test_handles_exception_in_subtask(self, manager):
        """Exceptions from gather are captured as failed results."""
        tasks = [_make_task("a")]
        parent = _make_task("parent")

        with patch.object(manager, "_execute_single_subtask") as mock_exec:
            mock_exec.side_effect = RuntimeError("Boom")

            results = await manager.execute_parallel_subtasks(parent, tasks, "owner/repo")

        assert len(results) == 1
        assert not results[0].results[0].success
        assert "Boom" in results[0].results[0].error

    @pytest.mark.asyncio
    async def test_emits_session_events(self, manager, session_logger):
        """Emits parallel_execution_started and parallel_execution_completed."""
        tasks = [_make_task("a")]
        parent = _make_task("parent")

        with patch.object(manager, "_execute_single_subtask") as mock_exec:
            mock_exec.return_value = SubtaskResult(
                task_id="a", success=True, branch_name="branch-a",
                duration_ms=100,
            )
            await manager.execute_parallel_subtasks(parent, tasks, "owner/repo")

        event_names = [c[0][0] for c in session_logger.log.call_args_list]
        assert "parallel_execution_started" in event_names
        assert "parallel_tier_completed" in event_names
        assert "parallel_execution_completed" in event_names

    @pytest.mark.asyncio
    async def test_respects_max_workers(self, manager):
        """Concurrency is capped by max_workers."""
        manager._config.max_workers = 2
        concurrent_count = 0
        max_concurrent = 0

        async def counting_exec(task, parent_task, owner_repo):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return SubtaskResult(
                task_id=task.id, success=True, branch_name=f"branch-{task.id}",
                duration_ms=10,
            )

        tasks = [_make_task(f"t{i}") for i in range(5)]
        parent = _make_task("parent")

        with patch.object(manager, "_execute_single_subtask", side_effect=counting_exec):
            await manager.execute_parallel_subtasks(parent, tasks, "owner/repo")

        assert max_concurrent <= 2


# -- Merge --


class TestMergeSubtaskBranches:
    def test_merges_successful_branches(self, manager):
        tier_results = [TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=True, branch_name="branch-a"),
                SubtaskResult(task_id="b", success=True, branch_name="branch-b"),
            ],
        )]

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = manager.merge_subtask_branches("main", tier_results)

        assert result.success
        assert len(result.merged_branches) == 2
        assert len(result.conflict_files) == 0

    def test_detects_merge_conflicts(self, manager):
        tier_results = [TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=True, branch_name="branch-a"),
            ],
        )]

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            # checkout succeeds, merge fails, abort succeeds
            mock_git.side_effect = [
                MagicMock(returncode=0),  # checkout
                MagicMock(returncode=1, stdout="CONFLICT", stderr=""),  # merge
                MagicMock(returncode=0),  # merge --abort
            ]
            result = manager.merge_subtask_branches("main", tier_results)

        assert not result.success
        assert "branch-a" in result.conflict_files

    def test_skips_failed_subtasks(self, manager):
        tier_results = [TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=False, branch_name="branch-a"),
                SubtaskResult(task_id="b", success=True, branch_name="branch-b"),
            ],
        )]

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = manager.merge_subtask_branches("main", tier_results)

        # Only branch-b should be merged
        assert result.success
        assert result.merged_branches == ["branch-b"]

    def test_emits_merge_events(self, manager, session_logger):
        tier_results = [TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=True, branch_name="branch-a"),
            ],
        )]

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
            manager.merge_subtask_branches("main", tier_results)

        event_names = [c[0][0] for c in session_logger.log.call_args_list]
        assert "parallel_merge_started" in event_names


# -- Cleanup --


class TestCleanup:
    def test_cleanup_worktrees(self, manager, worktree_manager):
        tier_results = [TierResult(
            tier=0,
            results=[
                SubtaskResult(
                    task_id="a", success=True, branch_name="branch-a",
                    worktree_path=Path("/tmp/wt-a"),
                ),
                SubtaskResult(
                    task_id="b", success=True, branch_name="branch-b",
                    worktree_path=Path("/tmp/wt-b"),
                ),
            ],
        )]

        # Mock Path.exists() for the worktree paths
        with patch.object(Path, "exists", return_value=True):
            manager.cleanup_worktrees(tier_results)

        assert worktree_manager.remove_worktree.call_count == 2

    def test_terminate_children(self, manager):
        proc = MagicMock()
        proc.returncode = None
        manager._child_processes = [proc]

        manager.terminate_children()

        proc.terminate.assert_called_once()
        assert manager._child_processes == []


# -- Config Model --


class TestParallelSubtasksConfig:
    def test_default_config(self):
        from agent_framework.core.config import ParallelSubtasksConfig
        cfg = ParallelSubtasksConfig()
        assert cfg.enabled is False
        assert cfg.max_workers == 3
        assert cfg.merge_strategy == "sequential-merge"
        assert cfg.subprocess_timeout == 3600

    def test_max_workers_validation(self):
        from agent_framework.core.config import ParallelSubtasksConfig
        with pytest.raises(ValueError, match="max_workers must be 1-10"):
            ParallelSubtasksConfig(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be 1-10"):
            ParallelSubtasksConfig(max_workers=11)

    def test_valid_merge_strategies(self):
        from agent_framework.core.config import ParallelSubtasksConfig
        cfg = ParallelSubtasksConfig(merge_strategy="octopus")
        assert cfg.merge_strategy == "octopus"

    def test_framework_config_has_parallel_subtasks(self):
        from agent_framework.core.config import FrameworkConfig
        cfg = FrameworkConfig()
        assert cfg.parallel_subtasks.enabled is False
        assert cfg.parallel_subtasks.max_workers == 3


# -- SubtaskResult / TierResult dataclasses --


class TestDataclasses:
    def test_tier_result_all_succeeded(self):
        tr = TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=True, branch_name="b-a"),
                SubtaskResult(task_id="b", success=True, branch_name="b-b"),
            ],
        )
        assert tr.all_succeeded is True

    def test_tier_result_has_failures(self):
        tr = TierResult(
            tier=0,
            results=[
                SubtaskResult(task_id="a", success=True, branch_name="b-a"),
                SubtaskResult(task_id="b", success=False, branch_name="b-b"),
            ],
        )
        assert tr.all_succeeded is False
        assert len(tr.failed_tasks) == 1
        assert tr.failed_tasks[0].task_id == "b"

    def test_merge_result_success(self):
        mr = MergeResult(success=True, merged_branches=["a", "b"])
        assert mr.success
        assert mr.error is None

    def test_merge_result_with_conflicts(self):
        mr = MergeResult(
            success=False,
            conflict_files=["branch-a"],
            error="1 branch(es) had merge conflicts",
        )
        assert not mr.success
