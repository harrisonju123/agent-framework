"""Tests for SummaryGenerator."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent_framework.core.task import PlanDocument, Task, TaskStatus, TaskType
from agent_framework.docs.summary_generator import SummaryGenerator
from agent_framework.queue.file_queue import FileQueue


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with queue structure."""
    return tmp_path


@pytest.fixture
def file_queue(temp_workspace):
    """Create a FileQueue instance."""
    return FileQueue(workspace=temp_workspace)


@pytest.fixture
def summary_generator(file_queue, temp_workspace):
    """Create a SummaryGenerator instance."""
    context_dir = temp_workspace / ".agent-context"
    return SummaryGenerator(queue=file_queue, output_dir=context_dir)


def create_task(
    task_id: str,
    task_type: TaskType,
    title: str,
    jira_key: str = None,
    plan: PlanDocument = None,
    result_summary: str = None,
    completed_at: datetime = None,
) -> Task:
    """Helper to create tasks for testing."""
    context = {}
    if jira_key:
        context["jira_key"] = jira_key

    return Task(
        id=task_id,
        type=task_type,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.utcnow(),
        title=title,
        description=f"Description for {title}",
        context=context,
        plan=plan,
        result_summary=result_summary,
        completed_at=completed_at or datetime.utcnow(),
        completed_by="engineer",
    )


class TestSummaryGeneratorSetup:
    """Tests for SummaryGenerator initialization."""

    def test_creates_output_directories(self, file_queue, temp_workspace):
        """Test that SummaryGenerator creates required directories."""
        context_dir = temp_workspace / ".agent-context"
        generator = SummaryGenerator(queue=file_queue, output_dir=context_dir)

        assert (context_dir / "plans").exists()
        assert (context_dir / "summaries").exists()
        assert (context_dir / "archives").exists()


class TestProjectSummary:
    """Tests for project summary generation."""

    def test_generate_empty_summary(self, summary_generator):
        """Test generating summary when no tasks exist."""
        path = summary_generator.generate_project_summary("TEST-999")

        assert path.exists()
        content = path.read_text()
        assert "TEST-999" in content
        assert "No completed tasks found" in content

    def test_generate_summary_with_tasks(self, summary_generator, file_queue):
        """Test generating summary with completed tasks."""
        # Create completed tasks
        tasks = [
            create_task(
                "arch-1",
                TaskType.ARCHITECTURE,
                "Design auth system",
                jira_key="TEST-100",
                result_summary="Designed JWT-based authentication",
            ),
            create_task(
                "impl-1",
                TaskType.IMPLEMENTATION,
                "Implement auth",
                jira_key="TEST-100",
                result_summary="Added login endpoint",
            ),
            create_task(
                "test-1",
                TaskType.TESTING,
                "Test auth",
                jira_key="TEST-100",
                result_summary="All tests pass",
            ),
        ]

        # Write tasks to completed directory
        for task in tasks:
            file_queue.mark_completed(task)

        path = summary_generator.generate_project_summary("TEST-100")

        assert path.exists()
        content = path.read_text()

        # Check sections exist
        assert "# TEST-100 - Project Summary" in content
        assert "## Planning" in content
        assert "## Implementation" in content
        assert "## Testing & Verification" in content

        # Check task content
        assert "Design auth system" in content
        assert "Implement auth" in content
        assert "Test auth" in content

    def test_summary_includes_plan(self, summary_generator, file_queue):
        """Test that plan content is included in summary."""
        plan = PlanDocument(
            objectives=["Add caching"],
            approach=["Step 1: Add Redis", "Step 2: Cache queries"],
            success_criteria=["Response time < 50ms"],
            files_to_modify=["src/cache.py"],
        )

        task = create_task(
            "arch-2",
            TaskType.ARCHITECTURE,
            "Design caching",
            jira_key="TEST-200",
            plan=plan,
        )
        file_queue.mark_completed(task)

        path = summary_generator.generate_project_summary("TEST-200")
        content = path.read_text()

        assert "Objectives" in content
        assert "Add caching" in content
        assert "Approach" in content
        assert "Add Redis" in content
        assert "Success Criteria" in content
        assert "Response time < 50ms" in content


class TestDailySummary:
    """Tests for daily summary generation."""

    def test_generate_empty_daily_summary(self, summary_generator):
        """Test generating daily summary with no tasks."""
        path = summary_generator.generate_daily_summary()

        assert path.exists()
        content = path.read_text()
        assert "Daily Summary" in content
        assert "No completed tasks" in content

    def test_generate_daily_summary_with_tasks(self, summary_generator, file_queue):
        """Test generating daily summary with tasks."""
        today = datetime.utcnow()

        tasks = [
            create_task(
                "task-1",
                TaskType.IMPLEMENTATION,
                "Feature A",
                jira_key="PROJ-1",
                completed_at=today,
            ),
            create_task(
                "task-2",
                TaskType.FIX,
                "Bug fix B",
                jira_key="PROJ-2",
                completed_at=today,
            ),
        ]

        for task in tasks:
            file_queue.mark_completed(task)

        path = summary_generator.generate_daily_summary(today)
        content = path.read_text()

        assert "Total tasks completed: 2" in content
        assert "PROJ-1" in content
        assert "PROJ-2" in content
        assert "Feature A" in content
        assert "Bug fix B" in content

    def test_daily_summary_filters_by_date(self, summary_generator, file_queue):
        """Test that daily summary only includes tasks from specified date."""
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)

        task_today = create_task(
            "today-task",
            TaskType.IMPLEMENTATION,
            "Today's task",
            completed_at=today,
        )
        task_yesterday = create_task(
            "yesterday-task",
            TaskType.IMPLEMENTATION,
            "Yesterday's task",
            completed_at=yesterday,
        )

        file_queue.mark_completed(task_today)
        file_queue.mark_completed(task_yesterday)

        path = summary_generator.generate_daily_summary(today)
        content = path.read_text()

        assert "Today's task" in content
        assert "Yesterday's task" not in content


class TestSavePlan:
    """Tests for saving plans to files."""

    def test_save_plan_returns_none_without_plan(self, summary_generator):
        """Test that save_plan returns None for tasks without plans."""
        task = create_task("no-plan", TaskType.IMPLEMENTATION, "No plan task")
        result = summary_generator.save_plan(task)
        assert result is None

    def test_save_plan_creates_file(self, summary_generator):
        """Test that save_plan creates a plan file."""
        plan = PlanDocument(
            objectives=["Goal 1"],
            approach=["Step 1"],
            success_criteria=["Done"],
        )
        task = create_task(
            "with-plan",
            TaskType.ARCHITECTURE,
            "Task with plan",
            jira_key="SAVE-123",
            plan=plan,
        )

        path = summary_generator.save_plan(task)

        assert path is not None
        assert path.exists()
        assert "SAVE-123" in path.name

        content = path.read_text()
        assert "Goal 1" in content
        assert "Step 1" in content


class TestTaskGrouping:
    """Tests for task grouping logic."""

    def test_groups_by_type(self, summary_generator, file_queue):
        """Test tasks are grouped correctly by type."""
        tasks = [
            create_task("p1", TaskType.PLANNING, "Planning 1", jira_key="GRP-1"),
            create_task("p2", TaskType.ARCHITECTURE, "Architecture 1", jira_key="GRP-1"),
            create_task("i1", TaskType.IMPLEMENTATION, "Impl 1", jira_key="GRP-1"),
            create_task("i2", TaskType.FIX, "Fix 1", jira_key="GRP-1"),
            create_task("t1", TaskType.TESTING, "Test 1", jira_key="GRP-1"),
            create_task("t2", TaskType.VERIFICATION, "Verify 1", jira_key="GRP-1"),
            create_task("o1", TaskType.COORDINATION, "Coord 1", jira_key="GRP-1"),
        ]

        for task in tasks:
            file_queue.mark_completed(task)

        path = summary_generator.generate_project_summary("GRP-1")
        content = path.read_text()

        # Verify grouping counts in overview
        assert "Planning tasks: 2" in content
        assert "Implementation tasks: 2" in content
        assert "Testing tasks: 2" in content
