"""Tests for fan-in logic in FileQueue and task decomposition."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.core.task_builder import build_decomposed_subtask
from agent_framework.queue.file_queue import FileQueue


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace for testing."""
    return tmp_path


@pytest.fixture
def queue(workspace):
    """Create a FileQueue instance for testing."""
    return FileQueue(workspace)


@pytest.fixture
def parent_task():
    """Create a parent task for testing."""
    return Task(
        id="parent-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Parent task for decomposition",
        description="This is a parent task that will be decomposed",
        context={
            "workflow": "default",
            "github_repo": "test/repo",
        },
        subtask_ids=["parent-123-sub-0", "parent-123-sub-1", "parent-123-sub-2"],
    )


@pytest.fixture
def completed_subtasks(parent_task):
    """Create completed subtasks for testing."""
    subtasks = []
    for i in range(3):
        subtask = Task(
            id=f"parent-123-sub-{i}",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="decomposer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title=f"Subtask {i}",
            description=f"Description for subtask {i}",
            parent_task_id="parent-123",
            result_summary=f"Completed work for subtask {i}",
            context={"subtask_index": i},
        )
        subtasks.append(subtask)
    return subtasks


class TestCheckSubtasksComplete:
    """Tests for check_subtasks_complete method."""

    def test_all_subtasks_completed(self, queue, completed_subtasks):
        """Returns True when all subtasks are completed successfully."""
        # Write all subtasks to completed directory
        for subtask in completed_subtasks:
            queue.mark_completed(subtask)

        subtask_ids = [s.id for s in completed_subtasks]
        result = queue.check_subtasks_complete("parent-123", subtask_ids)

        assert result is True

    def test_some_subtasks_pending(self, queue, completed_subtasks):
        """Returns False when some subtasks are still pending."""
        # Only complete first two subtasks
        queue.mark_completed(completed_subtasks[0])
        queue.mark_completed(completed_subtasks[1])

        # Put third subtask in queue as pending
        pending_subtask = completed_subtasks[2]
        pending_subtask.status = TaskStatus.PENDING
        queue.push(pending_subtask, "engineer")

        subtask_ids = [s.id for s in completed_subtasks]
        result = queue.check_subtasks_complete("parent-123", subtask_ids)

        assert result is False

    def test_some_subtasks_failed(self, queue, completed_subtasks):
        """Returns False when any subtask has failed."""
        # Complete first two subtasks
        queue.mark_completed(completed_subtasks[0])
        queue.mark_completed(completed_subtasks[1])

        # Mark third subtask as failed and move to completed
        failed_subtask = completed_subtasks[2]
        failed_subtask.status = TaskStatus.FAILED
        queue.mark_completed(failed_subtask)

        subtask_ids = [s.id for s in completed_subtasks]
        result = queue.check_subtasks_complete("parent-123", subtask_ids)

        assert result is False

    def test_empty_subtask_list(self, queue):
        """Returns True for empty subtask_ids (edge case)."""
        result = queue.check_subtasks_complete("parent-123", [])
        assert result is True

    def test_missing_subtask_file(self, queue, completed_subtasks):
        """Returns False when a subtask file is missing."""
        # Only complete first subtask
        queue.mark_completed(completed_subtasks[0])

        # Check for all three subtasks (two are missing)
        subtask_ids = [s.id for s in completed_subtasks]
        result = queue.check_subtasks_complete("parent-123", subtask_ids)

        assert result is False


class TestGetSubtasks:
    """Tests for get_subtasks method."""

    def test_finds_subtasks_in_completed(self, queue, completed_subtasks):
        """Finds subtasks in completed directory."""
        for subtask in completed_subtasks:
            queue.mark_completed(subtask)

        result = queue.get_subtasks("parent-123")

        assert len(result) == 3
        assert all(t.parent_task_id == "parent-123" for t in result)

    def test_finds_subtasks_in_active_queues(self, queue, completed_subtasks):
        """Finds subtasks in active queues."""
        for subtask in completed_subtasks:
            subtask.status = TaskStatus.PENDING
            queue.push(subtask, "engineer")

        result = queue.get_subtasks("parent-123")

        assert len(result) == 3
        assert all(t.parent_task_id == "parent-123" for t in result)

    def test_finds_subtasks_across_both(self, queue, completed_subtasks):
        """Finds subtasks in both active queues and completed directory."""
        # Put first subtask in completed
        queue.mark_completed(completed_subtasks[0])

        # Put remaining subtasks in queue
        for subtask in completed_subtasks[1:]:
            subtask.status = TaskStatus.PENDING
            queue.push(subtask, "engineer")

        result = queue.get_subtasks("parent-123")

        assert len(result) == 3
        assert all(t.parent_task_id == "parent-123" for t in result)

    def test_returns_empty_when_no_subtasks(self, queue):
        """Returns empty list when no subtasks found."""
        result = queue.get_subtasks("nonexistent-parent")
        assert result == []

    def test_ignores_unrelated_tasks(self, queue, completed_subtasks):
        """Ignores tasks with different parent_task_id."""
        # Add subtasks for parent-123
        for subtask in completed_subtasks:
            queue.mark_completed(subtask)

        # Add task with different parent
        other_task = Task(
            id="other-task-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Other task",
            description="Task with different parent",
            parent_task_id="parent-456",
        )
        queue.mark_completed(other_task)

        result = queue.get_subtasks("parent-123")

        assert len(result) == 3
        assert all(t.parent_task_id == "parent-123" for t in result)


class TestGetParentTask:
    """Tests for get_parent_task method."""

    def test_finds_parent_in_completed(self, queue, parent_task, completed_subtasks):
        """Finds parent task in completed directory."""
        queue.mark_completed(parent_task)

        result = queue.get_parent_task(completed_subtasks[0])

        assert result is not None
        assert result.id == "parent-123"

    def test_finds_parent_in_queue(self, queue, parent_task, completed_subtasks):
        """Finds parent task in active queue."""
        queue.push(parent_task, "engineer")

        result = queue.get_parent_task(completed_subtasks[0])

        assert result is not None
        assert result.id == "parent-123"

    def test_returns_none_when_no_parent_id(self, queue):
        """Returns None when task has no parent_task_id."""
        task = Task(
            id="standalone-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Standalone task",
            description="Task without parent",
        )

        result = queue.get_parent_task(task)

        assert result is None

    def test_returns_none_when_parent_not_found(self, queue, completed_subtasks):
        """Returns None when parent task not found."""
        result = queue.get_parent_task(completed_subtasks[0])
        assert result is None


class TestCreateFanInTask:
    """Tests for create_fan_in_task method."""

    def test_aggregates_results(self, queue, parent_task, completed_subtasks):
        """Fan-in task has combined result_summary from all subtasks."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.result_summary is not None
        assert "Completed work for subtask 0" in fan_in.result_summary
        assert "Completed work for subtask 1" in fan_in.result_summary
        assert "Completed work for subtask 2" in fan_in.result_summary

    def test_inherits_context(self, queue, parent_task, completed_subtasks):
        """Fan-in task has parent's context plus fan_in flag."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.context["fan_in"] is True
        assert fan_in.context["parent_task_id"] == "parent-123"
        assert fan_in.context["subtask_count"] == 3
        assert fan_in.context["workflow"] == "default"
        assert fan_in.context["github_repo"] == "test/repo"
        assert "aggregated_results" in fan_in.context

    def test_fan_in_id_pattern(self, queue, parent_task, completed_subtasks):
        """Fan-in task ID follows fan-in-{parent_id} pattern."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.id == "fan-in-parent-123"

    def test_fan_in_title(self, queue, parent_task, completed_subtasks):
        """Fan-in task has [fan-in] prefix in title."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.title.startswith("[fan-in]")
        assert "Parent task for decomposition" in fan_in.title

    def test_fan_in_assigned_to_qa(self, queue, parent_task, completed_subtasks):
        """Fan-in task is assigned to QA for next workflow step."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.assigned_to == "qa"

    def test_fan_in_status_pending(self, queue, parent_task, completed_subtasks):
        """Fan-in task starts with PENDING status."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.status == TaskStatus.PENDING

    def test_fan_in_inherits_type(self, queue, parent_task, completed_subtasks):
        """Fan-in task inherits parent's type."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.type == TaskType.IMPLEMENTATION

    def test_fan_in_inherits_priority(self, queue, parent_task, completed_subtasks):
        """Fan-in task inherits parent's priority."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.priority == parent_task.priority

    def test_fan_in_created_by_system(self, queue, parent_task, completed_subtasks):
        """Fan-in task is created by system."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert fan_in.created_by == "system"

    def test_propagates_single_subtask_implementation_branch(self, queue, parent_task):
        """Fan-in picks up implementation_branch from a single subtask."""
        subtask = Task(
            id="parent-123-sub-0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="decomposer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Subtask 0",
            description="Description",
            parent_task_id="parent-123",
            result_summary="Done",
            context={"implementation_branch": "feature/parent-123-sub-0"},
        )

        fan_in = queue.create_fan_in_task(parent_task, [subtask])

        assert fan_in.context["implementation_branch"] == "feature/parent-123-sub-0"
        assert "subtask_branches" not in fan_in.context

    def test_propagates_multiple_subtask_branches(self, queue, parent_task):
        """Fan-in collects all subtask branches when multiple exist."""
        subtasks = [
            Task(
                id=f"parent-123-sub-{i}",
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.COMPLETED,
                priority=1,
                created_by="decomposer",
                assigned_to="engineer",
                created_at=datetime.now(timezone.utc),
                title=f"Subtask {i}",
                description=f"Description {i}",
                parent_task_id="parent-123",
                result_summary=f"Done {i}",
                context={"implementation_branch": f"feature/parent-123-sub-{i}"},
            )
            for i in range(3)
        ]

        fan_in = queue.create_fan_in_task(parent_task, subtasks)

        assert fan_in.context["implementation_branch"] == "feature/parent-123-sub-0"
        assert fan_in.context["subtask_branches"] == [
            "feature/parent-123-sub-0",
            "feature/parent-123-sub-1",
            "feature/parent-123-sub-2",
        ]

    def test_no_branches_when_subtasks_lack_them(self, queue, parent_task, completed_subtasks):
        """Fan-in has no branch keys when subtasks don't have branches."""
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)

        assert "implementation_branch" not in fan_in.context
        assert "subtask_branches" not in fan_in.context

    def test_falls_back_to_worktree_branch(self, queue, parent_task):
        """Fan-in falls back to worktree_branch when implementation_branch is absent."""
        subtask = Task(
            id="parent-123-sub-0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="decomposer",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Subtask 0",
            description="Description",
            parent_task_id="parent-123",
            result_summary="Done",
            context={"worktree_branch": "worktree/parent-123-sub-0"},
        )

        fan_in = queue.create_fan_in_task(parent_task, [subtask])

        assert fan_in.context["implementation_branch"] == "worktree/parent-123-sub-0"

    def test_handles_subtasks_without_result_summary(self, queue, parent_task):
        """Fan-in task handles subtasks without result_summary."""
        subtasks_no_summary = [
            Task(
                id=f"parent-123-sub-{i}",
                type=TaskType.IMPLEMENTATION,
                status=TaskStatus.COMPLETED,
                priority=1,
                created_by="decomposer",
                assigned_to="engineer",
                created_at=datetime.now(timezone.utc),
                title=f"Subtask {i}",
                description=f"Description for subtask {i}",
                parent_task_id="parent-123",
                # No result_summary
            )
            for i in range(2)
        ]

        fan_in = queue.create_fan_in_task(parent_task, subtasks_no_summary)

        # Should create fan-in with empty or None result_summary
        assert fan_in.result_summary is None or fan_in.result_summary == ""


class TestFanInIdempotent:
    """Tests for _fan_in_already_created method."""

    def test_detects_existing_fan_in(self, queue, parent_task, completed_subtasks):
        """Duplicate fan-in creation is detected."""
        # Create and save first fan-in
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)
        queue.push(fan_in, "qa")

        # Check if fan-in already exists
        result = queue._fan_in_already_created("parent-123")

        assert result is True

    def test_returns_false_when_no_fan_in(self, queue):
        """Returns False when fan-in doesn't exist."""
        result = queue._fan_in_already_created("parent-123")

        assert result is False

    def test_detects_fan_in_in_completed(self, queue, parent_task, completed_subtasks):
        """Detects fan-in task in completed directory."""
        # Create and complete fan-in
        fan_in = queue.create_fan_in_task(parent_task, completed_subtasks)
        fan_in.status = TaskStatus.COMPLETED
        queue.mark_completed(fan_in)

        # Check if fan-in already exists
        result = queue._fan_in_already_created("parent-123")

        assert result is True


class TestBuildDecomposedSubtask:
    """Tests for build_decomposed_subtask helper."""

    def test_subtask_id_pattern(self, parent_task):
        """IDs follow {parent_id}-sub-{index} pattern."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Detailed description",
            files_to_modify=["file1.py", "file2.py"],
            approach_steps=["Step 1", "Step 2"],
            index=0,
        )

        assert subtask.id == "parent-123-sub-0"

    def test_subtask_has_parent_id(self, parent_task):
        """Subtask has parent_task_id set."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Detailed description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=0,
        )

        assert subtask.parent_task_id == "parent-123"

    def test_subtask_inherits_context(self, parent_task):
        """Subtask inherits parent's context."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Detailed description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=0,
        )

        assert subtask.context["workflow"] == "default"
        assert subtask.context["github_repo"] == "test/repo"
        assert subtask.context["parent_task_id"] == "parent-123"
        assert subtask.context["subtask_index"] == 0

    def test_subtask_inherits_priority(self, parent_task):
        """Subtask inherits parent's priority."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Detailed description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=1,
        )

        assert subtask.priority == parent_task.priority

    def test_subtask_has_plan(self, parent_task):
        """Subtask has a plan document."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Detailed description",
            files_to_modify=["file1.py", "file2.py"],
            approach_steps=["Step 1", "Step 2"],
            index=0,
        )

        assert subtask.plan is not None
        assert "Implement feature X" in subtask.plan.objectives
        assert "Step 1" in subtask.plan.approach
        assert "Step 2" in subtask.plan.approach
        assert "file1.py" in subtask.plan.files_to_modify
        assert "file2.py" in subtask.plan.files_to_modify

    def test_subtask_with_dependencies(self, parent_task):
        """Subtask can have dependencies."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature Y",
            description="Depends on feature X",
            files_to_modify=["file3.py"],
            approach_steps=["Step 1"],
            index=1,
            depends_on=["parent-123-sub-0"],
        )

        assert "parent-123-sub-0" in subtask.depends_on

    def test_subtask_status_pending(self, parent_task):
        """Subtask starts with PENDING status."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=0,
        )

        assert subtask.status == TaskStatus.PENDING

    def test_subtask_created_by_decomposer(self, parent_task):
        """Subtask is created by decomposer."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=0,
        )

        assert subtask.created_by == "decomposer"

    def test_subtask_inherits_assigned_to(self, parent_task):
        """Subtask inherits parent's assigned_to."""
        subtask = build_decomposed_subtask(
            parent_task=parent_task,
            name="Implement feature X",
            description="Description",
            files_to_modify=["file1.py"],
            approach_steps=["Step 1"],
            index=0,
        )

        assert subtask.assigned_to == parent_task.assigned_to
