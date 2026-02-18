"""Tests for DashboardDataProvider checkpoint methods."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.web.data_provider import DashboardDataProvider


def _make_workspace(tmpdir: str) -> Path:
    """Create a workspace with agents.yaml defining architect/engineer/qa."""
    workspace = Path(tmpdir)
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "agents.yaml").write_text(
        """
agents:
  - id: architect
    name: Architect
    queue: architect
    enabled: true
    prompt: "test"
  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "test"
  - id: qa
    name: QA
    queue: qa
    enabled: true
    prompt: "test"
"""
    )
    return workspace


def _make_checkpoint_task(
    task_id: str = "chain-abc123-engineer",
    title: str = "Implement feature X",
    checkpoint_id: str = "default-plan",
    checkpoint_message: str = "Review the implementation plan before proceeding",
    assigned_to: str = "engineer",
    status: TaskStatus = TaskStatus.AWAITING_APPROVAL,
) -> Task:
    """Create a task in AWAITING_APPROVAL status for checkpoint testing."""
    task = Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=5,
        created_by="test",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title=title,
        description="Test task description",
    )
    task.checkpoint_reached = checkpoint_id
    task.checkpoint_message = checkpoint_message
    return task


def _write_checkpoint(workspace: Path, task: Task) -> Path:
    """Write a task to the checkpoints directory."""
    checkpoint_dir = workspace / ".agent-communication" / "queues" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{task.id}.json"
    checkpoint_file.write_text(task.model_dump_json())
    return checkpoint_file


class TestGetPendingCheckpoints:
    def test_empty_when_no_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()
            assert result == []

    def test_empty_when_checkpoint_dir_exists_but_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            (workspace / ".agent-communication" / "queues" / "checkpoints").mkdir(
                parents=True
            )
            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()
            assert result == []

    def test_returns_awaiting_approval_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_checkpoint_task()
            _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()

            assert len(result) == 1
            assert result[0].id == task.id
            assert result[0].title == task.title
            assert result[0].checkpoint_id == "default-plan"
            assert result[0].checkpoint_message == "Review the implementation plan before proceeding"
            assert result[0].assigned_to == "engineer"
            assert result[0].paused_at is not None

    def test_skips_non_awaiting_approval_tasks(self):
        """Tasks with other statuses in checkpoint dir are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            # Write a task that's already been approved (IN_PROGRESS)
            task = _make_checkpoint_task(status=TaskStatus.IN_PROGRESS)
            _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()
            assert result == []

    def test_returns_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task1 = _make_checkpoint_task(task_id="task-1", title="Task 1")
            task2 = _make_checkpoint_task(task_id="task-2", title="Task 2")
            _write_checkpoint(workspace, task1)
            _write_checkpoint(workspace, task2)

            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()
            assert len(result) == 2

    def test_handles_corrupt_json(self):
        """Corrupt checkpoint files are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            checkpoint_dir = workspace / ".agent-communication" / "queues" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "corrupt.json").write_text("{not valid json")

            # Also write a valid one
            task = _make_checkpoint_task()
            _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_pending_checkpoints()
            assert len(result) == 1


class TestApproveCheckpoint:
    @patch("agent_framework.workflow.executor.resume_after_checkpoint", return_value=True)
    def test_approve_success(self, mock_resume):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)

            task = _make_checkpoint_task()
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.approve_checkpoint(task.id)

            assert result is True
            # Checkpoint file deleted on successful routing
            assert not checkpoint_file.exists()
            mock_resume.assert_called_once()

    @patch("agent_framework.workflow.executor.resume_after_checkpoint", return_value=False)
    def test_approve_preserves_checkpoint_on_routing_failure(self, mock_resume):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)

            task = _make_checkpoint_task()
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.approve_checkpoint(task.id)

            # Approval itself succeeds even if routing failed
            assert result is True
            # Checkpoint preserved for retry
            assert checkpoint_file.exists()

    @patch("agent_framework.workflow.executor.resume_after_checkpoint", return_value=True)
    def test_approve_with_message(self, mock_resume):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)

            task = _make_checkpoint_task()
            _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.approve_checkpoint(task.id, message="Looks good")

            assert result is True

    def test_approve_nonexistent_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            result = provider.approve_checkpoint("nonexistent-task")
            assert result is False

    def test_approve_non_awaiting_task(self):
        """Cannot approve a task that isn't AWAITING_APPROVAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_checkpoint_task(status=TaskStatus.IN_PROGRESS)
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.approve_checkpoint(task.id)

            assert result is False
            # File should NOT be deleted
            assert checkpoint_file.exists()


class TestRejectCheckpoint:
    def test_reject_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            queue_dir = workspace / ".agent-communication" / "queues" / "engineer"
            queue_dir.mkdir(parents=True)

            task = _make_checkpoint_task()
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.reject_checkpoint(task.id, "Use a different approach")

            assert result is True
            assert not checkpoint_file.exists()

            # Task should be re-queued to the same agent
            queued_files = list(queue_dir.glob("*.json"))
            assert len(queued_files) == 1

            requeued = json.loads(queued_files[0].read_text())
            assert requeued["status"] == TaskStatus.PENDING.value
            assert requeued["context"]["rejection_feedback"] == "Use a different approach"
            assert any("Checkpoint rejected" in n for n in requeued["notes"])
            assert requeued["checkpoint_reached"] is None
            assert requeued["checkpoint_message"] is None
            assert requeued["retry_count"] == 1

    def test_reject_nonexistent_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            result = provider.reject_checkpoint("nonexistent-task", "feedback")
            assert result is False

    def test_reject_non_awaiting_task(self):
        """Cannot reject a task that isn't AWAITING_APPROVAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_checkpoint_task(status=TaskStatus.IN_PROGRESS)
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.reject_checkpoint(task.id, "feedback")

            assert result is False
            assert checkpoint_file.exists()

    def test_reject_preserves_existing_context(self):
        """Rejection feedback is added to context without clobbering other keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            queue_dir = workspace / ".agent-communication" / "queues" / "engineer"
            queue_dir.mkdir(parents=True)

            task = _make_checkpoint_task()
            task.context["github_repo"] = "org/repo"
            checkpoint_file = _write_checkpoint(workspace, task)

            provider = DashboardDataProvider(workspace)
            provider.reject_checkpoint(task.id, "Needs tests")

            queued_files = list(queue_dir.glob("*.json"))
            requeued = json.loads(queued_files[0].read_text())
            assert requeued["context"]["github_repo"] == "org/repo"
            assert requeued["context"]["rejection_feedback"] == "Needs tests"
