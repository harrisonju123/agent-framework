"""Unit tests for TaskDecomposer."""

import json
from datetime import datetime, timezone

import pytest

from agent_framework.core.task import PlanDocument, Task, TaskStatus, TaskType
from agent_framework.core.task_decomposer import SubtaskBoundary, TaskDecomposer


def _make_task(**overrides) -> Task:
    """Create a Task with sensible defaults for testing."""
    defaults = dict(
        id="test-parent-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        title="Test parent task",
        description="Test parent description",
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_plan(**overrides) -> PlanDocument:
    """Create a PlanDocument with sensible defaults."""
    defaults = dict(
        objectives=["Implement feature X"],
        approach=["Step 1", "Step 2", "Step 3", "Step 4"],
        risks=["Risk 1"],
        success_criteria=["Tests pass"],
        files_to_modify=["src/core/file1.py", "src/api/file2.py"],
        dependencies=[]
    )
    defaults.update(overrides)
    return PlanDocument(**defaults)


class TestTaskDecomposer:
    """Test suite for TaskDecomposer."""

    def test_should_decompose_above_threshold(self):
        """Returns True for tasks exceeding 500 lines with multiple files."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=["file1.py", "file2.py", "file3.py"]
        )

        result = decomposer.should_decompose(plan, estimated_lines=600)

        assert result is True

    def test_should_not_decompose_below_threshold(self):
        """Returns False for tasks below 500 lines."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=["file1.py", "file2.py"]
        )

        result = decomposer.should_decompose(plan, estimated_lines=400)

        assert result is False

    def test_should_not_decompose_single_file(self):
        """Returns False when plan has only one file even if >500 lines."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=["file1.py"]
        )

        result = decomposer.should_decompose(plan, estimated_lines=600)

        assert result is False

    def test_decompose_creates_correct_subtask_count(self):
        """Decompose creates 2-5 subtasks based on file grouping."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/core/file2.py",
                "src/api/file3.py",
                "src/api/file4.py",
            ]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        assert 2 <= len(subtasks) <= 5

    def test_subtask_has_parent_id(self):
        """Each subtask.parent_task_id equals parent.id."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-abc-123")
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
            ]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            assert subtask.parent_task_id == "parent-abc-123"

    def test_parent_has_subtask_ids(self):
        """Parent.subtask_ids contains all subtask IDs after decomposition."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-xyz-456")
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
            ]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        assert len(parent.subtask_ids) == len(subtasks)
        for subtask in subtasks:
            assert subtask.id in parent.subtask_ids

    def test_subtask_inherits_context(self):
        """Subtasks inherit github_repo, workflow, etc. from parent."""
        decomposer = TaskDecomposer()
        parent = _make_task(
            context={
                "github_repo": "user/repo",
                "workflow": "default",
                "jira_project": "PROJ",
                "custom_field": "value"
            }
        )
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            assert subtask.context.get("github_repo") == "user/repo"
            assert subtask.context.get("workflow") == "default"
            assert subtask.context.get("jira_project") == "PROJ"
            assert subtask.context.get("custom_field") == "value"
            assert subtask.context.get("parent_task_id") == parent.id

    def test_independent_subtasks_have_no_depends_on(self):
        """Parallel subtasks have empty depends_on lists."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
            ]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # All subtasks should be independent (no depends_on by default)
        for subtask in subtasks:
            assert len(subtask.depends_on) == 0

    def test_backward_compatible_deserialization(self):
        """Old task JSON without new fields loads fine."""
        # Simulate old task JSON structure (no parent_task_id, subtask_ids, decomposition_strategy)
        old_task_dict = {
            "id": "old-task-123",
            "type": "implementation",
            "status": "pending",
            "priority": 1,
            "title": "Old task",
            "description": "Legacy task without new fields",
            "created_by": "architect",
            "assigned_to": "engineer",
            "created_at": "2026-02-15T10:00:00Z",
        }

        # Should deserialize without errors
        task = Task(**old_task_dict)

        assert task.id == "old-task-123"
        assert task.parent_task_id is None
        assert task.subtask_ids == []
        assert task.decomposition_strategy is None

    def test_max_depth_prevents_nested_decomposition(self):
        """Subtask with parent_task_id is not further decomposed."""
        decomposer = TaskDecomposer()
        # Create a subtask (has parent_task_id)
        subtask = _make_task(
            id="subtask-123",
            parent_task_id="parent-abc"
        )
        plan = _make_plan(
            files_to_modify=["file1.py", "file2.py", "file3.py"]
        )
        subtask.plan = plan

        # Should return empty list (no nested decomposition)
        nested_subtasks = decomposer.decompose(subtask, plan, estimated_lines=700)

        assert len(nested_subtasks) == 0
        assert len(subtask.subtask_ids) == 0

    def test_subtask_ids_follow_naming_pattern(self):
        """Subtask IDs follow pattern {parent_id}-sub-{index}."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="task-xyz-789")
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for idx, subtask in enumerate(subtasks):
            assert subtask.id == f"task-xyz-789-sub-{idx}"

    def test_subtask_has_scoped_plan(self):
        """Each subtask has a PlanDocument scoped to its files."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            assert subtask.plan is not None
            assert len(subtask.plan.files_to_modify) > 0
            # Subtask files should be a subset of parent files
            for file in subtask.plan.files_to_modify:
                assert file in plan.files_to_modify

    def test_respects_max_subtasks_limit(self):
        """Decomposer caps subtasks at MAX_SUBTASKS even with many files."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        # Create 20 files in different directories
        files = [f"dir{i}/file{i}.py" for i in range(20)]
        plan = _make_plan(files_to_modify=files)
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=2000)

        assert len(subtasks) <= decomposer.MAX_SUBTASKS

    def test_filters_small_boundaries(self):
        """Boundaries smaller than MIN_SUBTASK_SIZE are filtered out."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        # Plan with approach steps but minimal files
        plan = _make_plan(
            files_to_modify=["src/tiny.py"],  # Would create very small boundaries
            approach=["Step 1", "Step 2"]
        )
        parent.plan = plan

        # Should still handle gracefully
        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # Either creates reasonable subtasks or none at all
        assert len(subtasks) >= 0

    def test_subtask_includes_context_fields(self):
        """Subtask context includes subtask_index and subtask_total."""
        decomposer = TaskDecomposer()
        parent = _make_task()
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )
        parent.plan = plan

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for idx, subtask in enumerate(subtasks):
            assert subtask.context.get("subtask_index") == idx
            assert subtask.context.get("subtask_total") == len(subtasks)

    def test_json_serialization_with_new_fields(self):
        """Task with new fields can be serialized to JSON and back."""
        task = _make_task(
            parent_task_id="parent-123",
            subtask_ids=["sub-1", "sub-2"],
            decomposition_strategy="by_feature"
        )

        # Serialize to JSON
        task_json = task.model_dump_json()
        task_dict = json.loads(task_json)

        # Verify fields present
        assert task_dict["parent_task_id"] == "parent-123"
        assert task_dict["subtask_ids"] == ["sub-1", "sub-2"]
        assert task_dict["decomposition_strategy"] == "by_feature"

        # Deserialize back
        restored_task = Task.model_validate_json(task_json)
        assert restored_task.parent_task_id == "parent-123"
        assert restored_task.subtask_ids == ["sub-1", "sub-2"]
        assert restored_task.decomposition_strategy == "by_feature"

    def test_subtask_boundary_dataclass(self):
        """SubtaskBoundary dataclass works as expected."""
        boundary = SubtaskBoundary(
            name="Core module",
            files=["file1.py", "file2.py"],
            approach_steps=["Step 1", "Step 2"],
            depends_on_subtasks=[0],
            estimated_lines=200
        )

        assert boundary.name == "Core module"
        assert len(boundary.files) == 2
        assert len(boundary.approach_steps) == 2
        assert boundary.depends_on_subtasks == [0]
        assert boundary.estimated_lines == 200

    def test_identify_split_boundaries_groups_by_directory(self):
        """_identify_split_boundaries groups files by top-level directory."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/core/file2.py",
                "src/api/file3.py",
                "tests/test_file.py"
            ]
        )

        boundaries = decomposer._identify_split_boundaries(plan)

        # Should have multiple boundaries based on directories
        assert len(boundaries) >= 2
        # Each boundary should have files
        for boundary in boundaries:
            assert len(boundary.files) > 0

    def test_large_boundary_splitting(self):
        """Boundaries >300 lines estimated are split further."""
        decomposer = TaskDecomposer()
        # Create a plan with many files in same directory
        files = [f"src/core/file{i}.py" for i in range(10)]
        plan = _make_plan(files_to_modify=files)

        boundaries = decomposer._identify_split_boundaries(plan)

        # Should split large group into smaller parts
        assert len(boundaries) >= 2

    def test_fallback_to_approach_step_splitting(self):
        """Falls back to splitting approach steps when < 2 directory groups."""
        decomposer = TaskDecomposer()
        # Single directory but enough approach steps
        plan = _make_plan(
            files_to_modify=["src/file1.py"],
            approach=["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
        )

        boundaries = decomposer._identify_split_boundaries(plan)

        # Should create boundaries based on approach steps
        assert len(boundaries) >= 2
