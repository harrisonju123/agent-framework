"""Basic import and instantiation tests for Task model."""

from datetime import datetime, timezone

from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(**overrides) -> Task:
    """Create a Task with sensible defaults for testing."""
    defaults = dict(
        id="test-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        title="Test task",
        description="Test description",
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Task(**defaults)


def test_task_creation():
    """Test basic task creation."""
    task = _make_task()

    assert task.id == "test-123"
    assert task.type == "implementation"
    assert task.status == TaskStatus.PENDING
    assert task.retry_count == 0
    assert task.title == "Test task"
    assert task.description == "Test description"
    assert task.created_by == "test"
    assert task.assigned_to == "engineer"


def test_task_with_dependencies():
    """Test task creation with dependencies."""
    task = _make_task(
        id="test-456",
        depends_on=["task-1", "task-2"],
    )

    assert len(task.depends_on) == 2
    assert "task-1" in task.depends_on
    assert "task-2" in task.depends_on


def test_task_mark_in_progress():
    """Test marking task as in progress."""
    task = _make_task(id="test-789")

    task.mark_in_progress("engineer")

    assert task.status == TaskStatus.IN_PROGRESS
    assert task.started_at is not None
    assert task.started_by == "engineer"


def test_task_mark_completed():
    """Test marking task as completed."""
    task = _make_task(id="test-abc")

    task.mark_in_progress("engineer")
    task.mark_completed("engineer")

    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
    assert task.completed_by == "engineer"


def test_task_mark_failed():
    """Test marking task as failed."""
    task = _make_task(id="test-def")

    task.mark_failed("engineer")

    assert task.status == TaskStatus.FAILED
    assert task.failed_at is not None
    assert task.failed_by == "engineer"


def test_task_reset_to_pending():
    """Test resetting task to pending after failure."""
    task = _make_task(id="test-ghi")

    task.mark_in_progress("engineer")
    task.reset_to_pending()

    assert task.status == TaskStatus.PENDING
    assert task.started_at is None
    assert task.started_by is None


def test_root_id_uses_context_when_set():
    """root_id returns _root_task_id from context for chain tasks."""
    task = _make_task(id="chain-chain-original-engineer-1", context={"_root_task_id": "original"})
    assert task.root_id == "original"


def test_root_id_falls_back_to_own_id():
    """root_id returns task.id when no parent chain has been established."""
    task = _make_task(id="my-task", context={})
    assert task.root_id == "my-task"


def test_preview_task_type():
    """PREVIEW task type exists and serializes correctly."""
    assert TaskType.PREVIEW == "preview"
    assert TaskType.PREVIEW.value == "preview"

    task = _make_task(type=TaskType.PREVIEW)
    assert task.type == TaskType.PREVIEW
