"""Unit tests for TaskDecomposer."""

import json
from datetime import datetime, timezone

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument
from agent_framework.core.task_decomposer import (
    TaskDecomposer,
    SubtaskBoundary,
    estimate_plan_lines,
    extract_requirements_checklist,
)


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


class TestEstimatePlanLines:
    """Tests for the estimate_plan_lines() shared estimator."""

    def test_files_only(self):
        plan = _make_plan(files_to_modify=["a.py", "b.py", "c.py"], approach=[])
        assert estimate_plan_lines(plan) == 150  # 3 * 50

    def test_steps_only(self):
        plan = _make_plan(files_to_modify=[], approach=["s1", "s2", "s3", "s4"])
        assert estimate_plan_lines(plan) == 100  # 4 * 25

    def test_combined(self):
        plan = _make_plan(
            files_to_modify=["a.py", "b.py"],
            approach=["s1", "s2", "s3"],
        )
        assert estimate_plan_lines(plan) == 175  # 2*50 + 3*25

    def test_empty_plan(self):
        plan = _make_plan(files_to_modify=[], approach=[])
        assert estimate_plan_lines(plan) == 0

    def test_threshold_integration(self):
        """7 files + 6 steps = 500, exactly at threshold."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=[f"f{i}.py" for i in range(7)],
            approach=[f"step {i}" for i in range(6)],
        )
        estimated = estimate_plan_lines(plan)
        assert estimated == 500  # 7*50 + 6*25
        # Threshold is >= 500, need at least 2 files
        assert decomposer.should_decompose(plan, estimated) is True

    def test_at_threshold(self):
        """5 files + 4 steps = 350, exactly at threshold."""
        plan = _make_plan(
            files_to_modify=[f"f{i}.py" for i in range(5)],
            approach=[f"step {i}" for i in range(4)],
        )
        estimated = estimate_plan_lines(plan)
        assert estimated == 350  # 5*50 + 4*25
        decomposer = TaskDecomposer()
        assert decomposer.should_decompose(plan, estimated) is True


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

        # 300 lines < 350 threshold
        result = decomposer.should_decompose(plan, estimated_lines=300)

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


class TestExtractRequirementsChecklist:
    """Tests for extract_requirements_checklist()."""

    def test_action_steps_become_checklist_items(self):
        plan = _make_plan(
            approach=[
                "Add memory hit rate panel with category breakdown",
                "Create self-evaluation retry rate panel",
                "Implement debate usage tracking panel",
            ],
            files_to_modify=["src/dashboard.py"],
        )
        checklist = extract_requirements_checklist(plan)

        assert len(checklist) == 3
        assert checklist[0]["id"] == 1
        assert "memory hit rate" in checklist[0]["description"].lower()
        assert checklist[0]["status"] == "pending"

    def test_prep_steps_excluded(self):
        plan = _make_plan(
            approach=[
                "Read existing dashboard code to understand patterns",
                "Analyze the metrics schema",
                "Add new metrics panel",
                "Understand how routing works",
            ],
        )
        checklist = extract_requirements_checklist(plan)

        assert len(checklist) == 1
        assert "metrics panel" in checklist[0]["description"].lower()

    def test_file_matching(self):
        plan = _make_plan(
            approach=["Update dashboard.py with new panel"],
            files_to_modify=["src/dashboard.py", "src/config.py"],
        )
        checklist = extract_requirements_checklist(plan)

        assert len(checklist) == 1
        assert "src/dashboard.py" in checklist[0]["files"]

    def test_empty_approach_returns_empty(self):
        plan = _make_plan(approach=[])
        checklist = extract_requirements_checklist(plan)
        assert checklist == []

    def test_numbered_steps_stripped(self):
        plan = _make_plan(
            approach=[
                "1. Add first feature",
                "2. Create second feature",
            ],
        )
        checklist = extract_requirements_checklist(plan)

        assert len(checklist) == 2
        assert checklist[0]["description"].startswith("Add")
        assert checklist[1]["description"].startswith("Create")

    def test_short_stem_not_matched(self):
        """File stems <= 3 chars (like 'a.py') shouldn't match substring in description."""
        plan = _make_plan(
            approach=["Add a new panel to the dashboard"],
            files_to_modify=["a.py", "src/dashboard.py"],
        )
        checklist = extract_requirements_checklist(plan)

        assert len(checklist) == 1
        # "a.py" should NOT match (stem "a" is too short)
        assert "a.py" not in checklist[0]["files"]
        # "dashboard.py" should match (name appears in text)
        assert "src/dashboard.py" in checklist[0]["files"]

    def test_six_item_plan_produces_six_items(self):
        """Validates the specific scenario from PR #43: 6-panel dashboard."""
        plan = _make_plan(
            approach=[
                "Add memory hit rate panel with category breakdown",
                "Create self-evaluation retry rate panel",
                "Build context budget utilization panel",
                "Implement debate metrics panel",
                "Add tool pattern efficiency panel",
                "Create workflow success rate panel",
            ],
            files_to_modify=[
                "src/dashboard.py",
                "src/panels/memory.py",
                "src/panels/self_eval.py",
                "src/panels/budget.py",
            ],
        )
        checklist = extract_requirements_checklist(plan)
        assert len(checklist) == 6


class TestDecomposeThresholdChange:
    """Tests for the lowered decomposition threshold and requirements-count trigger."""

    def test_old_threshold_350_now_decomposes(self):
        """350 lines was below old threshold (500) but is at new threshold (350)."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=[f"f{i}.py" for i in range(5)],
            approach=[f"step {i}" for i in range(4)],
        )
        estimated = estimate_plan_lines(plan)
        assert estimated == 350
        assert decomposer.should_decompose(plan, estimated) is True

    def test_requirements_count_trigger(self):
        """6+ requirements + 200+ lines triggers decomposition even below line threshold."""
        decomposer = TaskDecomposer()
        plan = _make_plan(
            files_to_modify=["a.py", "b.py", "c.py"],
            approach=["s1"],
        )
        estimated = estimate_plan_lines(plan)
        assert estimated == 175  # below 350

        # Without requirements count: no decomposition
        assert decomposer.should_decompose(plan, estimated) is False

        # With 6 requirements and est >= 200: triggers
        assert decomposer.should_decompose(plan, 250, requirements_count=6) is True

    def test_requirements_count_below_min_lines(self):
        """Requirements count trigger doesn't fire below REQUIREMENTS_MIN_LINES."""
        decomposer = TaskDecomposer()
        plan = _make_plan(files_to_modify=["a.py", "b.py"])

        assert decomposer.should_decompose(plan, 150, requirements_count=8) is False

    def test_requirements_count_below_trigger(self):
        """5 requirements (below 6) doesn't trigger decomposition."""
        decomposer = TaskDecomposer()
        plan = _make_plan(files_to_modify=["a.py", "b.py"])

        assert decomposer.should_decompose(plan, 250, requirements_count=5) is False
