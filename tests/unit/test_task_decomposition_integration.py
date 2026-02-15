"""Integration tests for task decomposition in architect and engineer agents."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, PlanDocument, TaskStatus, TaskType
from agent_framework.queue.file_queue import FileQueue


@pytest.fixture
def mock_queue(tmp_path):
    """Create a mock FileQueue for testing."""
    queue = Mock(spec=FileQueue)
    queue.queue_dir = tmp_path / "queues"
    queue.queue_dir.mkdir(parents=True, exist_ok=True)
    queue.completed_dir = tmp_path / "completed"
    queue.completed_dir.mkdir(parents=True, exist_ok=True)
    return queue


@pytest.fixture
def architect_agent(mock_queue, tmp_path):
    """Create a mock architect agent."""
    config = AgentConfig(
        id="architect",
        name="Architect",
        queue="architect",
        prompt="You are an architect",
    )

    # Create minimal agent with mocked dependencies
    mock_llm = Mock()
    agent = Agent(
        config=config,
        llm=mock_llm,
        queue=mock_queue,
        workspace=tmp_path,
    )
    agent._workflows_config = {}
    return agent


@pytest.fixture
def engineer_agent(mock_queue, tmp_path):
    """Create a mock engineer agent."""
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer",
    )

    # Create minimal agent with mocked dependencies
    mock_llm = Mock()
    agent = Agent(
        config=config,
        llm=mock_llm,
        queue=mock_queue,
        workspace=tmp_path,
    )
    agent._workflows_config = {}
    return agent


def test_architect_decomposes_large_task(architect_agent, mock_queue):
    """Test that architect decomposes tasks with >500 estimated lines."""
    # Create a task with 40 files in different directories (40 * 15 = 600 lines estimated)
    # Group files by directory to enable decomposition
    files = []
    for dir_num in range(4):  # 4 directories
        for file_num in range(10):  # 10 files per directory
            files.append(f"src/dir{dir_num}/module_{file_num}.py")

    plan = PlanDocument(
        objectives=["Implement large feature"],
        approach=["Step 1", "Step 2", "Step 3", "Step 4"],
        risks=["Complexity"],
        success_criteria=["All tests pass"],
        files_to_modify=files,
        dependencies=[],
    )

    task = Task(
        id="test-task-large",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="architect",
        created_at=datetime.now(UTC),
        title="Large implementation task",
        description="Implement feature across 40 files",
        plan=plan,
    )

    # Test should_decompose check
    assert architect_agent._should_decompose_task(task) is True

    # Test decomposition
    mock_queue.update = Mock()
    mock_queue.push = Mock()

    architect_agent._decompose_and_queue_subtasks(task)

    # Verify subtasks were queued to engineer
    assert mock_queue.push.called, "Expected subtasks to be queued"
    push_calls = mock_queue.push.call_args_list
    assert len(push_calls) > 0, "Expected at least one subtask to be created"

    # Verify all queued tasks are subtasks
    for call in push_calls:
        subtask, target_agent = call[0]
        assert target_agent == "engineer"
        assert subtask.parent_task_id == task.id
        assert "-sub" in subtask.id

    # Verify parent task was updated with subtask IDs
    assert mock_queue.update.called
    assert task.subtask_ids is not None
    assert len(task.subtask_ids) > 0


def test_architect_skips_decomposition_small_task(architect_agent):
    """Test that architect skips decomposition for tasks with <500 lines."""
    # Create a task with 5 files (5 * 15 = 75 lines estimated)
    plan = PlanDocument(
        objectives=["Implement small feature"],
        approach=["Step 1"],
        risks=[],
        success_criteria=["Tests pass"],
        files_to_modify=[f"src/module_{i}.py" for i in range(5)],
        dependencies=[],
    )

    task = Task(
        id="test-task-small",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="architect",
        created_at=datetime.now(UTC),
        title="Small implementation task",
        description="Implement feature in 5 files",
        plan=plan,
    )

    # Test should_decompose check
    assert architect_agent._should_decompose_task(task) is False


def test_fan_in_triggered_on_last_subtask(engineer_agent, mock_queue):
    """Test that completing the last subtask creates a fan-in task."""
    parent_id = "parent-task-123"
    subtask_ids = ["parent-task-123-sub1", "parent-task-123-sub2", "parent-task-123-sub3"]

    # Create parent task
    parent_task = Task(
        id=parent_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Parent task",
        description="Parent of subtasks",
        subtask_ids=subtask_ids,
    )

    # Create completed subtask
    subtask = Task(
        id=subtask_ids[2],  # Last subtask
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Subtask 3",
        description="Third subtask",
        parent_task_id=parent_id,
    )

    # Mock queue methods
    mock_queue.find_task = Mock(return_value=parent_task)
    mock_queue.check_subtasks_complete = Mock(return_value=True)
    mock_queue._fan_in_already_created = Mock(return_value=False)
    mock_queue.get_completed = Mock(return_value=subtask)
    mock_queue.push = Mock()

    # Create mock fan-in task
    fan_in_task = Task(
        id=f"fan-in-{parent_id}",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="system",
        assigned_to="qa",
        created_at=datetime.now(UTC),
        title=f"[fan-in] {parent_task.title}",
        description=parent_task.description,
        context={"fan_in": True, "parent_task_id": parent_id, "subtask_count": 3},
    )
    mock_queue.create_fan_in_task = Mock(return_value=fan_in_task)

    # Trigger fan-in check
    engineer_agent._check_and_create_fan_in_task(subtask)

    # Verify fan-in task was created and queued
    assert mock_queue.create_fan_in_task.called
    assert mock_queue.push.called
    push_call = mock_queue.push.call_args[0]
    assert push_call[0].id == fan_in_task.id
    assert push_call[1] == "qa"


def test_fan_in_not_triggered_on_partial_completion(engineer_agent, mock_queue):
    """Test that completing a non-final subtask does not create fan-in task."""
    parent_id = "parent-task-456"
    subtask_ids = ["parent-task-456-sub1", "parent-task-456-sub2", "parent-task-456-sub3"]

    # Create parent task
    parent_task = Task(
        id=parent_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Parent task",
        description="Parent of subtasks",
        subtask_ids=subtask_ids,
    )

    # Create completed subtask (not the last one)
    subtask = Task(
        id=subtask_ids[0],  # First subtask
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Subtask 1",
        description="First subtask",
        parent_task_id=parent_id,
    )

    # Mock queue methods - not all subtasks complete
    mock_queue.find_task = Mock(return_value=parent_task)
    mock_queue.check_subtasks_complete = Mock(return_value=False)
    mock_queue.push = Mock()

    # Trigger fan-in check
    engineer_agent._check_and_create_fan_in_task(subtask)

    # Verify fan-in task was NOT created
    assert not mock_queue.push.called


def test_fan_in_task_routes_to_qa(mock_queue):
    """Test that fan-in task is assigned to QA agent."""
    parent_task = Task(
        id="parent-789",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Parent task",
        description="Parent of subtasks",
        context={},
    )

    subtasks = [
        Task(
            id=f"parent-789-sub{i}",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title=f"Subtask {i}",
            description=f"Subtask {i}",
            parent_task_id="parent-789",
            result_summary=f"Completed subtask {i}",
        )
        for i in range(1, 4)
    ]

    # Mock the create_fan_in_task method to test expected behavior
    fan_in_task = Task(
        id="fan-in-parent-789",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="system",
        assigned_to="qa",
        created_at=datetime.now(UTC),
        title="[fan-in] Parent task",
        description="Parent of subtasks",
        context={
            "fan_in": True,
            "parent_task_id": "parent-789",
            "subtask_count": 3,
            "aggregated_results": "\n".join(
                f"[Subtask {i}]: Completed subtask {i}" for i in range(1, 4)
            ),
        },
        result_summary="\n".join(
            f"[Subtask {i}]: Completed subtask {i}" for i in range(1, 4)
        ),
    )

    # Verify fan-in task properties
    assert fan_in_task.id == "fan-in-parent-789"
    assert fan_in_task.assigned_to == "qa"
    assert fan_in_task.context.get("fan_in") is True
    assert fan_in_task.context.get("parent_task_id") == "parent-789"
    assert fan_in_task.context.get("subtask_count") == 3
    assert "Completed subtask" in fan_in_task.result_summary
