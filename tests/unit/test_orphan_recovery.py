"""Tests for orphan task recovery — 3-tier smart detection."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.queue.file_queue import FileQueue


def _make_task(task_id="task-1", status=TaskStatus.PENDING, assigned_to="engineer", **kwargs):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=1,
        created_by="architect",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Implement feature",
        description="Build the thing.",
        context=kwargs.get("context", {}),
    )


@pytest.fixture
def queue(tmp_path):
    return FileQueue(tmp_path)


class TestRecoverOrphanedTasks:
    """Tests for FileQueue.recover_orphaned_tasks()."""

    def test_auto_completes_task_already_in_completed(self, queue):
        """Task in both queue and completed/ → queue copy removed."""
        task = _make_task(status=TaskStatus.IN_PROGRESS)
        task.started_by = "dead-agent"
        queue.push(task, "engineer")

        # Simulate crash between move_to_completed write and queue unlink
        completed_file = queue.completed_dir / f"{task.id}.json"
        completed_file.write_text(task.model_dump_json(indent=2))

        result = queue.recover_orphaned_tasks()

        assert task.id in result["auto_completed"]
        assert not result["reset_to_pending"]
        # Queue copy should be gone
        assert not (queue.queue_dir / "engineer" / f"{task.id}.json").exists()
        # Completed copy should still exist
        assert completed_file.exists()

    def test_auto_completes_chain_task_with_successor(self, queue):
        """Chain task + successor exists → auto-completed to completed/."""
        # Source chain task stuck in IN_PROGRESS
        source = _make_task(
            task_id="chain-abc-implement-d1",
            status=TaskStatus.IN_PROGRESS,
            context={"chain_step": True, "source_task_id": "root-task"},
        )
        source.started_by = "dead-agent"
        queue.push(source, "engineer")

        # Successor chain task already queued
        successor = _make_task(
            task_id="chain-abc-code_review-d2",
            status=TaskStatus.PENDING,
            assigned_to="architect",
            context={
                "chain_step": True,
                "source_task_id": "chain-abc-implement-d1",
            },
        )
        queue.push(successor, "architect")

        result = queue.recover_orphaned_tasks()

        assert "chain-abc-implement-d1" in result["auto_completed"]
        assert not result["reset_to_pending"]
        # Source should be in completed/
        assert (queue.completed_dir / "chain-abc-implement-d1.json").exists()
        # Source should not be in queue
        assert not (queue.queue_dir / "engineer" / "chain-abc-implement-d1.json").exists()

    def test_resets_genuine_orphan_to_pending(self, queue):
        """No successor, no completed copy → reset to PENDING."""
        task = _make_task(status=TaskStatus.IN_PROGRESS)
        task.started_by = "dead-agent"
        queue.push(task, "engineer")

        result = queue.recover_orphaned_tasks()

        assert task.id in result["reset_to_pending"]
        assert not result["auto_completed"]
        # Task should still be in queue but PENDING
        task_file = queue.queue_dir / "engineer" / f"{task.id}.json"
        recovered = json.loads(task_file.read_text())
        assert recovered["status"] == "pending"
        assert recovered["started_at"] is None
        assert recovered["started_by"] is None

    def test_skips_non_in_progress_tasks(self, queue):
        """PENDING/COMPLETED tasks are untouched."""
        pending = _make_task(task_id="pending-1", status=TaskStatus.PENDING)
        queue.push(pending, "engineer")

        completed = _make_task(task_id="done-1", status=TaskStatus.COMPLETED)
        queue.push(completed, "engineer")

        result = queue.recover_orphaned_tasks()

        assert not result["auto_completed"]
        assert not result["reset_to_pending"]
        # Both still in queue
        assert (queue.queue_dir / "engineer" / "pending-1.json").exists()
        assert (queue.queue_dir / "engineer" / "done-1.json").exists()

    def test_filters_by_queue_id(self, queue):
        """Passing queue_ids only scans those queues."""
        eng_task = _make_task(
            task_id="eng-orphan", status=TaskStatus.IN_PROGRESS, assigned_to="engineer"
        )
        eng_task.started_by = "dead-agent"
        queue.push(eng_task, "engineer")

        qa_task = _make_task(
            task_id="qa-orphan", status=TaskStatus.IN_PROGRESS, assigned_to="qa"
        )
        qa_task.started_by = "dead-agent"
        queue.push(qa_task, "qa")

        result = queue.recover_orphaned_tasks(queue_ids=["engineer"])

        assert "eng-orphan" in result["reset_to_pending"]
        # QA queue was not scanned
        assert "qa-orphan" not in result["reset_to_pending"]
        assert "qa-orphan" not in result["auto_completed"]

    def test_has_successor_ignores_non_chain_source(self, queue):
        """Task with matching source_task_id but no chain_step flag is ignored."""
        source = _make_task(
            task_id="source-task",
            status=TaskStatus.IN_PROGRESS,
            context={"chain_step": True},
        )
        source.started_by = "dead-agent"
        queue.push(source, "engineer")

        # Non-chain task that happens to reference source_task_id
        non_chain = _make_task(
            task_id="pr-task",
            status=TaskStatus.PENDING,
            assigned_to="architect",
            context={"source_task_id": "source-task"},  # no chain_step
        )
        queue.push(non_chain, "architect")

        result = queue.recover_orphaned_tasks()

        # Should be reset, not auto-completed — the non-chain ref doesn't count
        assert "source-task" in result["reset_to_pending"]
        assert "source-task" not in result["auto_completed"]

    def test_recovery_result_summary(self, queue):
        """Result dict has the expected structure."""
        result = queue.recover_orphaned_tasks()
        assert "auto_completed" in result
        assert "reset_to_pending" in result
        assert "errors" in result
        assert isinstance(result["auto_completed"], list)
        assert isinstance(result["reset_to_pending"], list)
        assert isinstance(result["errors"], list)

    def test_tier1_takes_priority_over_tier2_for_chain_task(self, queue):
        """Chain task in both queue and completed/ hits tier 1 (cheap unlink),
        not tier 2 (expensive successor scan)."""
        source = _make_task(
            task_id="chain-dup-implement-d1",
            status=TaskStatus.IN_PROGRESS,
            context={"chain_step": True},
        )
        source.started_by = "dead-agent"
        queue.push(source, "engineer")

        # Already in completed/ (tier 1 match)
        completed_file = queue.completed_dir / f"{source.id}.json"
        completed_file.write_text(source.model_dump_json(indent=2))

        # Successor also exists (would match tier 2 if tier 1 didn't fire first)
        successor = _make_task(
            task_id="chain-dup-review-d2",
            status=TaskStatus.PENDING,
            assigned_to="architect",
            context={"chain_step": True, "source_task_id": "chain-dup-implement-d1"},
        )
        queue.push(successor, "architect")

        result = queue.recover_orphaned_tasks()

        assert "chain-dup-implement-d1" in result["auto_completed"]
        # Queue copy removed, completed copy untouched (tier 1 just unlinks)
        assert not (queue.queue_dir / "engineer" / "chain-dup-implement-d1.json").exists()
        assert completed_file.exists()
        # No recovery_reason stamped — tier 1 doesn't rewrite the completed copy
        completed_data = json.loads(completed_file.read_text())
        assert "recovery_reason" not in completed_data.get("context", {})


class TestPopOrphanRecovery:
    """Integration: 3-tier recovery via pop()."""

    def test_pop_auto_completes_orphan_with_successor(self, queue):
        """pop() auto-completes a chain orphan that has a successor."""
        source = _make_task(
            task_id="chain-x-implement-d1",
            status=TaskStatus.IN_PROGRESS,
            context={"chain_step": True},
        )
        source.started_by = "dead-agent"
        queue.push(source, "engineer")

        successor = _make_task(
            task_id="chain-x-review-d2",
            status=TaskStatus.PENDING,
            context={
                "chain_step": True,
                "source_task_id": "chain-x-implement-d1",
            },
        )
        queue.push(successor, "engineer")

        # pop should skip the auto-completed orphan and return the successor
        task = queue.pop("engineer")

        assert task is not None
        assert task.id == "chain-x-review-d2"
        # Source should be in completed/
        assert (queue.completed_dir / "chain-x-implement-d1.json").exists()
        assert not (queue.queue_dir / "engineer" / "chain-x-implement-d1.json").exists()


class TestClaimOrphanRecovery:
    """Integration: 3-tier recovery via claim()."""

    def test_claim_auto_completes_orphan_with_successor(self, queue):
        """claim() auto-completes a chain orphan that has a successor."""
        source = _make_task(
            task_id="chain-y-implement-d1",
            status=TaskStatus.IN_PROGRESS,
            context={"chain_step": True},
        )
        source.started_by = "dead-agent"
        queue.push(source, "engineer")

        successor = _make_task(
            task_id="chain-y-review-d2",
            status=TaskStatus.PENDING,
            context={
                "chain_step": True,
                "source_task_id": "chain-y-implement-d1",
            },
        )
        queue.push(successor, "engineer")

        result = queue.claim("engineer", "engineer-1")

        assert result is not None
        claimed_task, lock = result
        assert claimed_task.id == "chain-y-review-d2"
        lock.release()
        # Source should be in completed/
        assert (queue.completed_dir / "chain-y-implement-d1.json").exists()
