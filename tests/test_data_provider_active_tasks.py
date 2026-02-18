"""Tests for DashboardDataProvider active task and cancel methods."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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


def _make_task(
    task_id: str = "task-001",
    title: str = "Test task",
    assigned_to: str = "engineer",
    status: TaskStatus = TaskStatus.PENDING,
    task_type: TaskType = TaskType.IMPLEMENTATION,
    jira_key: str | None = None,
    parent_task_id: str | None = None,
    started_at: datetime | None = None,
) -> Task:
    task = Task(
        id=task_id,
        type=task_type,
        status=status,
        priority=5,
        created_by="test",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title=title,
        description="Test description",
        parent_task_id=parent_task_id,
        started_at=started_at,
    )
    if jira_key:
        task.context["jira_key"] = jira_key
    return task


def _queue_task(workspace: Path, task: Task) -> Path:
    """Write a task to its queue directory."""
    queue_dir = workspace / ".agent-communication" / "queues" / task.assigned_to
    queue_dir.mkdir(parents=True, exist_ok=True)
    task_file = queue_dir / f"{task.id}.json"
    task_file.write_text(task.model_dump_json())
    return task_file


class TestGetActiveTasks:
    def test_empty_when_no_queues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider.get_active_tasks() == []

    def test_returns_pending_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.PENDING)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()

            assert len(result) == 1
            assert result[0].id == "task-001"
            assert result[0].status == "pending"

    def test_returns_in_progress_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
            )
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()

            assert len(result) == 1
            assert result[0].status == "in_progress"
            assert result[0].started_at is not None

    def test_excludes_completed_and_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _queue_task(workspace, _make_task(task_id="t1", status=TaskStatus.PENDING))
            _queue_task(workspace, _make_task(task_id="t2", status=TaskStatus.COMPLETED))
            _queue_task(workspace, _make_task(task_id="t3", status=TaskStatus.FAILED))

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()

            assert len(result) == 1
            assert result[0].id == "t1"

    def test_sorts_in_progress_before_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _queue_task(workspace, _make_task(task_id="pending-1", status=TaskStatus.PENDING))
            _queue_task(workspace, _make_task(
                task_id="running-1",
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
            ))

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()

            assert len(result) == 2
            assert result[0].id == "running-1"
            assert result[1].id == "pending-1"

    def test_includes_jira_key_and_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(jira_key="PROJ-42", parent_task_id="parent-001")
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()

            assert result[0].jira_key == "PROJ-42"
            assert result[0].parent_task_id == "parent-001"

    def test_skips_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            queue_dir = workspace / ".agent-communication" / "queues" / "engineer"
            queue_dir.mkdir(parents=True, exist_ok=True)
            (queue_dir / "bad.json").write_text("{invalid json")

            _queue_task(workspace, _make_task(task_id="good"))

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()
            assert len(result) == 1
            assert result[0].id == "good"

    def test_spans_multiple_queues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _queue_task(workspace, _make_task(task_id="t1", assigned_to="engineer"))
            _queue_task(workspace, _make_task(task_id="t2", assigned_to="architect"))

            provider = DashboardDataProvider(workspace)
            result = provider.get_active_tasks()
            ids = {t.id for t in result}
            assert ids == {"t1", "t2"}


class TestCancelTask:
    def test_cancel_pending_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.PENDING)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is True

            # Verify task is now cancelled on disk
            task_file = workspace / ".agent-communication" / "queues" / "engineer" / "task-001.json"
            reloaded = Task.model_validate_json(task_file.read_text())
            assert reloaded.status == TaskStatus.CANCELLED

    def test_cancel_in_progress_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(
                status=TaskStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
            )
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is True

    def test_cancel_with_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.PENDING)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            provider.cancel_task("task-001", reason="No longer needed")

            task_file = workspace / ".agent-communication" / "queues" / "engineer" / "task-001.json"
            reloaded = Task.model_validate_json(task_file.read_text())
            assert "No longer needed" in (reloaded.last_error or "")

    def test_cancel_nonexistent_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("nonexistent") is False

    def test_cancel_completed_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.COMPLETED)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is False

    def test_cancel_already_cancelled_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.CANCELLED)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is False

    def test_cancel_failed_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.FAILED)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is False

    def test_cancel_testing_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.TESTING)
            _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.cancel_task("task-001") is False

    def test_cancelled_task_disappears_from_active(self):
        """After cancelling, the task should no longer appear in active tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _queue_task(workspace, _make_task(task_id="t1", status=TaskStatus.PENDING))
            _queue_task(workspace, _make_task(task_id="t2", status=TaskStatus.PENDING))

            provider = DashboardDataProvider(workspace)
            assert len(provider.get_active_tasks()) == 2

            provider.cancel_task("t1")
            result = provider.get_active_tasks()
            assert len(result) == 1
            assert result[0].id == "t2"
