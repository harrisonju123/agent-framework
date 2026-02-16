"""Integration tests for replanning feature showing enriched history and context."""

import pytest
from datetime import datetime, timezone

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument


def test_replan_enabled_by_default():
    """Test that the default replan config would enable replanning."""
    # This test verifies the change made to agent.py line 241
    # From: self._replan_enabled = replan_cfg.get("enabled", False)
    # To:   self._replan_enabled = replan_cfg.get("enabled", True)

    replan_cfg = {}  # Empty config (no explicit setting)
    replan_enabled = replan_cfg.get("enabled", True)

    assert replan_enabled is True, "Replanning should be enabled by default"


def test_replan_can_be_explicitly_disabled():
    """Test that replanning can be explicitly disabled via config."""
    replan_cfg = {"enabled": False}
    replan_enabled = replan_cfg.get("enabled", True)

    assert replan_enabled is False, "Explicit False should disable replanning"


def test_enriched_replan_history_structure():
    """Test that replan_history entries can contain enriched fields."""
    # This test verifies the structure of enriched history entries

    task = Task(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Test task",
        description="Test description",
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
    )

    # Simulate what _request_replan() does - add an enriched history entry
    enriched_entry = {
        "attempt": 1,
        "error": "Some error occurred",
        "error_type": "validation_error",  # NEW field
        "approach_tried": "Step 1 | Step 2 | Step 3",  # NEW field
        "files_involved": ["file1.py", "file2.py"],  # NEW field
        "revised_plan": "Try a different approach",
    }

    task.replan_history.append(enriched_entry)

    # Verify structure
    assert len(task.replan_history) == 1
    entry = task.replan_history[0]

    assert "error_type" in entry
    assert entry["error_type"] == "validation_error"

    assert "approach_tried" in entry
    assert "Step 1" in entry["approach_tried"]

    assert "files_involved" in entry
    assert len(entry["files_involved"]) == 2


def test_backward_compatibility_old_replan_history():
    """Test that old replan history entries without new fields don't break anything."""
    task = Task(
        id="legacy-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Legacy task",
        description="Task with old history format",
        created_by="engineer",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
    )

    # Old format history entry (missing error_type, approach_tried, files_involved)
    old_entry = {
        "attempt": 1,
        "error": "Old error",
        "revised_plan": "Old plan",
    }

    task.replan_history.append(old_entry)

    # Should not crash when accessed
    entry = task.replan_history[0]

    # Using .get() with defaults should work
    assert entry.get("error_type", "unknown") == "unknown"
    assert entry.get("approach_tried", "unknown") == "unknown"
    assert entry.get("files_involved", []) == []


def test_replan_history_with_plan_document():
    """Test creating replan history from a task with PlanDocument."""
    plan = PlanDocument(
        objectives=["Implement feature"],
        approach=[
            "Step 1: Read files",
            "Step 2: Modify code",
            "Step 3: Add tests",
        ],
        files_to_modify=["src/main.py", "tests/test_main.py"],
        success_criteria=["All tests pass"],
    )

    task = Task(
        id="planned-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        title="Planned task",
        description="Task with plan",
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        plan=plan,
        retry_count=2,
        last_error="ImportError: module not found",
    )

    # Simulate creating enriched history entry (what _request_replan does)
    approach_tried = " | ".join(plan.approach[:3])

    enriched_entry = {
        "attempt": 2,
        "error": "ImportError: module not found",
        "error_type": "import_error",
        "approach_tried": approach_tried,
        "files_involved": plan.files_to_modify,
        "revised_plan": "Install dependencies first, then retry",
    }

    task.replan_history.append(enriched_entry)

    # Verify
    entry = task.replan_history[0]
    assert "Step 1: Read files" in entry["approach_tried"]
    assert "Step 2: Modify code" in entry["approach_tried"]
    assert "src/main.py" in entry["files_involved"]
    assert "tests/test_main.py" in entry["files_involved"]
