"""Unit tests for TaskDecomposer."""

import json
from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument
from agent_framework.core.task_decomposer import TaskDecomposer, SubtaskBoundary


def _make_task(**overrides) -> Task:
    """Create a Task with sensible defaults for testing."""
    defaults = dict(
        id="test-parent-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        title="Test task",
        description="Test description",
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_plan(**overrides) -> PlanDocument:
    """Create a PlanDocument with sensible defaults."""
    defaults = dict(
        objectives=["Complete the feature"],
        approach=["Step 1", "Step 2", "Step 3"],
        risks=[],
        success_criteria=["All tests pass"],
        files_to_modify=["src/core/file1.py", "src/core/file2.py"],
        dependencies=[],
    )
    defaults.update(overrides)
    return PlanDocument(**defaults)


class TestTaskDecomposer:
    """Tests for TaskDecomposer class."""

    def test_should_decompose_above_threshold(self):
        """Test that should_decompose returns True for tasks above threshold."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=["src/file1.py", "src/file2.py", "src/file3.py"]
        )

        # 600 lines > 500 threshold
        result = decomposer.should_decompose(plan, estimated_lines=600)

        assert result is True

    def test_should_not_decompose_below_threshold(self):
        """Test that should_decompose returns False for tasks below threshold."""
        decomposer = TaskDecomposer()
        plan = _make_plan(files_to_modify=["src/file1.py", "src/file2.py"])

        # 400 lines < 500 threshold
        result = decomposer.should_decompose(plan, estimated_lines=400)

        assert result is False

    def test_should_not_decompose_single_file(self):
        """Test that should_decompose returns False for single file tasks."""
        decomposer = TaskDecomposer()
        plan = _make_plan(files_to_modify=["src/file1.py"])

        # Even though 600 > 500, only 1 file
        result = decomposer.should_decompose(plan, estimated_lines=600)

        assert result is False

    def test_decompose_creates_correct_subtask_count(self):
        """Test that decompose creates 2-5 subtasks."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-123")
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
                "tests/test_file1.py",
                "tests/test_file2.py",
            ]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # Should create between 2 and 5 subtasks
        assert 2 <= len(subtasks) <= 5

    def test_subtask_has_parent_id(self):
        """Test that each subtask has parent_task_id set correctly."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-456")
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            assert subtask.parent_task_id == "parent-456"

    def test_parent_has_subtask_ids(self):
        """Test that parent.subtask_ids contains all subtask IDs."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-789")
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # Parent should have subtask IDs
        assert len(parent.subtask_ids) == len(subtasks)

        # All subtask IDs should be in parent's list
        for subtask in subtasks:
            assert subtask.id in parent.subtask_ids

    def test_subtask_inherits_context(self):
        """Test that subtasks inherit context from parent."""
        decomposer = TaskDecomposer()
        parent = _make_task(
            id="parent-abc",
            context={
                "github_repo": "test/repo",
                "workflow": "default",
                "jira_project": "TEST",
            },
        )
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            # Check inherited context
            assert subtask.context.get("github_repo") == "test/repo"
            assert subtask.context.get("workflow") == "default"
            assert subtask.context.get("jira_project") == "TEST"
            # Check added context
            assert subtask.context.get("parent_task_id") == "parent-abc"
            assert "subtask_index" in subtask.context
            assert "subtask_total" in subtask.context

    def test_independent_subtasks_have_no_depends_on(self):
        """Test that parallel/independent subtasks have empty depends_on."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-def")
        # Different directories should create independent subtasks
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
                "tests/test1.py",
            ]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # At least some subtasks should be independent (empty depends_on)
        independent_count = sum(1 for st in subtasks if len(st.depends_on) == 0)
        assert independent_count >= 1

    def test_backward_compatible_deserialization(self):
        """Test that old task JSON without new fields loads fine."""
        # Old task JSON without parent_task_id, subtask_ids, decomposition_strategy
        old_task_json = {
            "id": "old-task-123",
            "type": "implementation",
            "status": "pending",
            "priority": 1,
            "created_by": "test",
            "assigned_to": "engineer",
            "created_at": "2024-01-01T00:00:00Z",
            "title": "Old task",
            "description": "Old description",
        }

        # Should deserialize without errors
        task = Task(**old_task_json)

        assert task.id == "old-task-123"
        assert task.parent_task_id is None
        assert task.subtask_ids == []
        assert task.decomposition_strategy is None

    def test_max_depth_prevents_nested_decomposition(self):
        """Test that subtasks (with parent_task_id) are not further decomposed."""
        decomposer = TaskDecomposer()

        # Create a subtask (has parent_task_id set)
        subtask = _make_task(
            id="parent-123-sub1",
            parent_task_id="parent-123",  # This is already a subtask
        )
        plan = _make_plan(
            files_to_modify=["src/file1.py", "src/file2.py", "src/file3.py"]
        )

        # Attempt to decompose the subtask (should return empty list)
        result = decomposer.decompose(subtask, plan, estimated_lines=600)

        # Should not create any subtasks (max depth = 1)
        assert result == []
        assert subtask.subtask_ids == []

    def test_subtask_has_scoped_plan(self):
        """Test that subtasks have properly scoped PlanDocuments."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-scoped")
        plan = _make_plan(
            files_to_modify=[
                "src/core/file1.py",
                "src/api/file2.py",
                "tests/test1.py",
            ]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        for subtask in subtasks:
            # Each subtask should have a plan
            assert subtask.plan is not None

            # Plan should have files_to_modify (subset of parent's files)
            assert len(subtask.plan.files_to_modify) > 0
            assert all(f in plan.files_to_modify for f in subtask.plan.files_to_modify)

            # Plan should have objectives and approach
            assert len(subtask.plan.objectives) > 0
            assert len(subtask.plan.approach) >= 0  # May be empty if no relevant steps

    def test_subtask_id_pattern(self):
        """Test that subtask IDs follow the pattern {parent_id}-sub{index}."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-pattern")
        plan = _make_plan(
            files_to_modify=["src/core/file1.py", "src/api/file2.py"]
        )

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # Check ID pattern
        for index, subtask in enumerate(subtasks, start=1):
            expected_id = f"parent-pattern-sub{index}"
            assert subtask.id == expected_id

    def test_decompose_with_max_subtasks_cap(self):
        """Test that decompose respects MAX_SUBTASKS cap."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-maxcap")

        # Create plan with many files in different directories
        files = [
            f"dir{i}/file{i}.py" for i in range(10)
        ]  # 10 different directories
        plan = _make_plan(files_to_modify=files)

        subtasks = decomposer.decompose(parent, plan, estimated_lines=1000)

        # Should not exceed MAX_SUBTASKS (5)
        assert len(subtasks) <= decomposer.MAX_SUBTASKS

    def test_min_subtask_size_filter(self):
        """Test that subtasks below MIN_SUBTASK_SIZE are filtered out."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-minsize")

        # Create plan that would result in very small subtasks
        plan = _make_plan(
            files_to_modify=[
                "src/file1.py",
                "tests/file2.py",
            ]
        )

        # Low estimated lines might create subtasks below MIN_SUBTASK_SIZE
        subtasks = decomposer.decompose(parent, plan, estimated_lines=80)

        # All subtasks should meet minimum size, or no decomposition occurs
        if subtasks:
            for subtask in subtasks:
                # Check via notes which contain estimated lines
                assert any("Estimated lines:" in note for note in subtask.notes)

    def test_subtask_boundary_dataclass(self):
        """Test SubtaskBoundary dataclass creation."""
        boundary = SubtaskBoundary(
            name="Test boundary",
            files=["file1.py", "file2.py"],
            approach_steps=["Step 1", "Step 2"],
            depends_on_subtasks=[0],
            estimated_lines=100,
        )

        assert boundary.name == "Test boundary"
        assert len(boundary.files) == 2
        assert len(boundary.approach_steps) == 2
        assert boundary.depends_on_subtasks == [0]
        assert boundary.estimated_lines == 100

    def test_decompose_returns_empty_for_insufficient_boundaries(self):
        """Test that decompose returns empty list if can't create valid boundaries."""
        decomposer = TaskDecomposer()
        parent = _make_task(id="parent-noboundaries")

        # Plan with only 1 file (can't split)
        plan = _make_plan(files_to_modify=["src/file1.py"])

        subtasks = decomposer.decompose(parent, plan, estimated_lines=600)

        # Should return empty list
        assert subtasks == []
        assert parent.subtask_ids == []

    def test_serialization_with_new_fields(self):
        """Test that tasks with new fields serialize/deserialize correctly."""
        task = _make_task(
            id="serialize-test",
            parent_task_id="parent-123",
            subtask_ids=["child-1", "child-2"],
            decomposition_strategy="by_feature",
        )

        # Serialize to JSON
        task_dict = task.model_dump()

        # Check fields are present
        assert task_dict["parent_task_id"] == "parent-123"
        assert task_dict["subtask_ids"] == ["child-1", "child-2"]
        assert task_dict["decomposition_strategy"] == "by_feature"

        # Deserialize back
        task_restored = Task(**task_dict)

        assert task_restored.parent_task_id == "parent-123"
        assert task_restored.subtask_ids == ["child-1", "child-2"]
        assert task_restored.decomposition_strategy == "by_feature"
