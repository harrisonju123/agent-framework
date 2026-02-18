"""Tests for PlanDocument model and serialization."""

import json
from datetime import datetime, timezone

import pytest

from agent_framework.core.task import PlanDocument, Task, TaskStatus, TaskType


class TestPlanDocument:
    """Tests for PlanDocument model."""

    def test_plan_document_creation(self):
        """Test basic PlanDocument creation with required fields."""
        plan = PlanDocument(
            objectives=["Implement user authentication", "Add session management"],
            approach=["Step 1: Add auth middleware", "Step 2: Create login endpoint"],
            success_criteria=["Users can log in", "Sessions persist across requests"],
        )

        assert len(plan.objectives) == 2
        assert len(plan.approach) == 2
        assert len(plan.success_criteria) == 2
        assert plan.risks == []
        assert plan.files_to_modify == []
        assert plan.dependencies == []

    def test_plan_document_with_all_fields(self):
        """Test PlanDocument with all optional fields."""
        plan = PlanDocument(
            objectives=["Add caching layer"],
            approach=["Integrate Redis", "Add cache decorators"],
            risks=["Redis unavailability - mitigate with fallback to memory cache"],
            success_criteria=["Response time under 100ms"],
            files_to_modify=["src/cache.py", "src/api/handlers.py"],
            dependencies=["redis>=4.0"],
        )

        assert plan.risks == ["Redis unavailability - mitigate with fallback to memory cache"]
        assert plan.files_to_modify == ["src/cache.py", "src/api/handlers.py"]
        assert plan.dependencies == ["redis>=4.0"]

    def test_plan_document_serialization(self):
        """Test PlanDocument JSON serialization."""
        plan = PlanDocument(
            objectives=["Test objective"],
            approach=["Step 1"],
            success_criteria=["Criterion 1"],
            files_to_modify=["file.py"],
        )

        json_str = plan.model_dump_json()
        data = json.loads(json_str)

        assert data["objectives"] == ["Test objective"]
        assert data["approach"] == ["Step 1"]
        assert data["success_criteria"] == ["Criterion 1"]
        assert data["files_to_modify"] == ["file.py"]

    def test_plan_document_deserialization(self):
        """Test PlanDocument JSON deserialization."""
        data = {
            "objectives": ["Objective A"],
            "approach": ["Step A", "Step B"],
            "risks": ["Risk X"],
            "success_criteria": ["Criterion A"],
            "files_to_modify": ["a.py", "b.py"],
            "dependencies": ["dep>=1.0"],
        }

        plan = PlanDocument(**data)

        assert plan.objectives == ["Objective A"]
        assert plan.approach == ["Step A", "Step B"]
        assert plan.risks == ["Risk X"]
        assert plan.success_criteria == ["Criterion A"]
        assert plan.files_to_modify == ["a.py", "b.py"]
        assert plan.dependencies == ["dep>=1.0"]


class TestPlanDocumentDictCoercion:
    """Tests for PlanDocument dict-to-list coercion validator."""

    def test_approach_as_dict_coerced_to_list_of_values(self):
        """Dict approach should yield values, not keys."""
        plan = PlanDocument(
            objectives=["Build feature"],
            approach={"step_1": "Clone repo", "step_2": "Implement logic"},
            success_criteria=["Tests pass"],
        )
        assert plan.approach == ["Clone repo", "Implement logic"]

    def test_approach_as_list_of_dicts_flattened(self):
        """List-of-dicts entries should be flattened to strings."""
        plan = PlanDocument(
            objectives=["Build feature"],
            approach=[
                {"step": "Clone repo", "detail": "Use git clone"},
                {"step": "Implement", "detail": "Write code"},
            ],
            success_criteria=["Tests pass"],
        )
        assert plan.approach == [
            "Clone repo - Use git clone",
            "Implement - Write code",
        ]

    def test_multiple_fields_as_dicts(self):
        """All list[str] fields should handle dict input."""
        plan = PlanDocument(
            objectives={"obj_1": "First objective", "obj_2": "Second objective"},
            approach=["Normal step"],
            risks={"risk_a": "Timeout risk", "risk_b": "Data loss"},
            success_criteria={"crit_1": "All tests pass"},
        )
        assert plan.objectives == ["First objective", "Second objective"]
        assert plan.risks == ["Timeout risk", "Data loss"]
        assert plan.success_criteria == ["All tests pass"]

    def test_normal_list_input_unchanged(self):
        """Normal list[str] input passes through untouched."""
        plan = PlanDocument(
            objectives=["Objective A", "Objective B"],
            approach=["Step 1", "Step 2"],
            success_criteria=["Criterion 1"],
            files_to_modify=["src/main.py"],
            dependencies=["requests>=2.0"],
        )
        assert plan.objectives == ["Objective A", "Objective B"]
        assert plan.approach == ["Step 1", "Step 2"]
        assert plan.files_to_modify == ["src/main.py"]
        assert plan.dependencies == ["requests>=2.0"]

    def test_model_validate_with_dict_approach(self):
        """model_validate() path also triggers coercion."""
        data = {
            "objectives": ["Build"],
            "approach": {"s1": "Clone", "s2": "Implement"},
            "success_criteria": ["Pass"],
        }
        plan = PlanDocument.model_validate(data)
        assert plan.approach == ["Clone", "Implement"]

    def test_mixed_list_with_dicts_and_strings(self):
        """List containing both strings and dicts should coerce only dicts."""
        plan = PlanDocument(
            objectives=["Build feature"],
            approach=[
                "Plain string step",
                {"step": "Dict step", "note": "with details"},
            ],
            success_criteria=["Done"],
        )
        assert plan.approach == [
            "Plain string step",
            "Dict step - with details",
        ]


class TestTaskWithPlan:
    """Tests for Task model with PlanDocument."""

    def _create_task(self, **kwargs):
        """Helper to create a task with defaults."""
        defaults = {
            "id": "test-123",
            "type": TaskType.ARCHITECTURE,
            "status": TaskStatus.PENDING,
            "priority": 1,
            "created_by": "architect",
            "assigned_to": "engineer",
            "created_at": datetime.now(timezone.utc),
            "title": "Test task",
            "description": "Test description",
        }
        defaults.update(kwargs)
        return Task(**defaults)

    def test_task_without_plan(self):
        """Test task creation without plan field."""
        task = self._create_task()
        assert task.plan is None

    def test_task_with_plan(self):
        """Test task creation with plan field."""
        plan = PlanDocument(
            objectives=["Implement feature X"],
            approach=["Design API", "Write code", "Add tests"],
            success_criteria=["Feature works as specified"],
        )

        task = self._create_task(plan=plan)

        assert task.plan is not None
        assert task.plan.objectives == ["Implement feature X"]
        assert len(task.plan.approach) == 3

    def test_task_plan_serialization(self):
        """Test task with plan serializes correctly."""
        plan = PlanDocument(
            objectives=["Objective"],
            approach=["Step 1"],
            success_criteria=["Done"],
            files_to_modify=["src/main.py"],
        )

        task = self._create_task(plan=plan)
        json_str = task.model_dump_json()
        data = json.loads(json_str)

        assert "plan" in data
        assert data["plan"]["objectives"] == ["Objective"]
        assert data["plan"]["files_to_modify"] == ["src/main.py"]

    def test_task_plan_deserialization(self):
        """Test task with plan deserializes from JSON."""
        data = {
            "id": "task-456",
            "type": "architecture",
            "status": "pending",
            "priority": 2,
            "created_by": "architect",
            "assigned_to": "engineer",
            "created_at": "2024-01-15T10:00:00",
            "title": "Architecture task",
            "description": "Plan the feature",
            "plan": {
                "objectives": ["Design system"],
                "approach": ["Analyze", "Design", "Document"],
                "risks": ["Scope creep"],
                "success_criteria": ["Design approved"],
                "files_to_modify": [],
                "dependencies": [],
            },
        }

        task = Task(**data)

        assert task.plan is not None
        assert task.plan.objectives == ["Design system"]
        assert task.plan.risks == ["Scope creep"]

    def test_task_plan_roundtrip(self):
        """Test task with plan survives JSON roundtrip."""
        plan = PlanDocument(
            objectives=["Build API"],
            approach=["Define endpoints", "Implement handlers", "Test"],
            risks=["API versioning complexity"],
            success_criteria=["All endpoints return correct data"],
            files_to_modify=["src/api/routes.py", "src/api/handlers.py"],
            dependencies=["fastapi>=0.100"],
        )

        original = self._create_task(plan=plan, context={"jira_key": "TEST-123"})

        json_str = original.model_dump_json()
        restored = Task(**json.loads(json_str))

        assert restored.plan is not None
        assert restored.plan.objectives == original.plan.objectives
        assert restored.plan.approach == original.plan.approach
        assert restored.plan.risks == original.plan.risks
        assert restored.plan.success_criteria == original.plan.success_criteria
        assert restored.plan.files_to_modify == original.plan.files_to_modify
        assert restored.plan.dependencies == original.plan.dependencies
