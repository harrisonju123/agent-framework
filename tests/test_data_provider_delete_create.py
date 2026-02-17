"""Tests for DashboardDataProvider.delete_task() and create_task()."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.web.data_provider import DashboardDataProvider


def _make_workspace(tmpdir: str) -> Path:
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
    status: TaskStatus = TaskStatus.PENDING,
    assigned_to: str = "engineer",
) -> Task:
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=5,
        created_by="test",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test description",
    )


def _queue_task(workspace: Path, task: Task) -> Path:
    queue_dir = workspace / ".agent-communication" / "queues" / task.assigned_to
    queue_dir.mkdir(parents=True, exist_ok=True)
    task_file = queue_dir / f"{task.id}.json"
    task_file.write_text(task.model_dump_json())
    return task_file


def _complete_task(workspace: Path, task: Task) -> Path:
    completed_dir = workspace / ".agent-communication" / "completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    task_file = completed_dir / f"{task.id}.json"
    task_file.write_text(task.model_dump_json())
    return task_file


class TestDeleteTask:
    def test_delete_pending_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.PENDING)
            task_file = _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("task-001") is None
            assert not task_file.exists()

    def test_delete_failed_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.FAILED)
            task_file = _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("task-001") is None
            assert not task_file.exists()

    def test_delete_cancelled_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.CANCELLED)
            task_file = _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("task-001") is None
            assert not task_file.exists()

    def test_reject_in_progress_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.IN_PROGRESS)
            task.started_at = datetime.now(timezone.utc)
            task_file = _queue_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("task-001") == "not_deletable"
            assert task_file.exists()

    def test_reject_completed_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            task = _make_task(status=TaskStatus.COMPLETED)
            task_file = _complete_task(workspace, task)

            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("task-001") == "not_deletable"
            assert task_file.exists()

    def test_nonexistent_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider.delete_task("nonexistent") == "not_found"

    def test_deleted_task_disappears_from_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _queue_task(workspace, _make_task(task_id="t1", status=TaskStatus.PENDING))
            _queue_task(workspace, _make_task(task_id="t2", status=TaskStatus.PENDING))

            provider = DashboardDataProvider(workspace)
            assert len(provider.get_active_tasks()) == 2

            assert provider.delete_task("t1") is None
            result = provider.get_active_tasks()
            assert len(result) == 1
            assert result[0].id == "t2"


class TestCreateTask:
    def test_creates_task_in_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            task = provider.create_task(
                title="Build feature X",
                description="Implement the feature",
                task_type="implementation",
                assigned_to="engineer",
            )

            assert task.id.startswith("manual-")
            assert task.title == "Build feature X"
            assert task.status == TaskStatus.PENDING
            assert task.created_by == "web-dashboard"
            assert task.assigned_to == "engineer"
            assert task.type == TaskType.IMPLEMENTATION

            # Verify file exists on disk
            task_file = workspace / ".agent-communication" / "queues" / "engineer" / f"{task.id}.json"
            assert task_file.exists()

    def test_creates_task_with_repository(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            task = provider.create_task(
                title="Fix bug",
                description="Fix the bug",
                task_type="fix",
                assigned_to="engineer",
                repository="owner/repo",
            )

            assert task.context["github_repo"] == "owner/repo"

    def test_creates_task_with_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            task = provider.create_task(
                title="Urgent fix",
                description="Fix it now",
                task_type="fix",
                assigned_to="engineer",
                priority=8,
            )

            assert task.priority == 8

    def test_routes_to_correct_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            task = provider.create_task(
                title="Architect work",
                description="Plan the thing",
                task_type="planning",
                assigned_to="architect",
            )

            task_file = workspace / ".agent-communication" / "queues" / "architect" / f"{task.id}.json"
            assert task_file.exists()

    def test_created_task_appears_in_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            task = provider.create_task(
                title="New work",
                description="Do things",
                task_type="implementation",
                assigned_to="engineer",
            )

            active = provider.get_active_tasks()
            assert len(active) == 1
            assert active[0].id == task.id
            assert active[0].status == "pending"
