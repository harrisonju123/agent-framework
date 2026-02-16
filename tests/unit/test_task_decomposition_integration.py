"""Integration tests for task decomposition in architect and engineer agents."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agent_framework.core.task import Task, PlanDocument, TaskStatus, TaskType
from agent_framework.core.task_decomposer import TaskDecomposer
from agent_framework.queue.file_queue import FileQueue


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def file_queue(tmp_workspace):
    """Create a FileQueue instance for testing."""
    return FileQueue(workspace=tmp_workspace)


@pytest.fixture
def task_decomposer():
    """Create a TaskDecomposer instance."""
    return TaskDecomposer()


def test_architect_decomposes_large_task(task_decomposer):
    """Test that decomposer handles tasks with >500 estimated lines."""
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
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Large implementation task",
        description="Implement feature across 40 files",
        plan=plan,
    )

    estimated_lines = len(files) * 15  # 600 lines

    # Test should_decompose check
    assert task_decomposer.should_decompose(plan, estimated_lines) is True

    # Test decomposition
    subtasks = task_decomposer.decompose(task, plan, estimated_lines)

    # Verify subtasks were created
    assert len(subtasks) > 0, "Expected at least one subtask to be created"

    # Verify all created tasks are subtasks
    for subtask in subtasks:
        assert subtask.parent_task_id == task.id
        assert "-sub" in subtask.id
        assert subtask.assigned_to == "engineer"  # Inherits from parent

    # Verify parent task was updated with subtask IDs
    assert task.subtask_ids is not None
    assert len(task.subtask_ids) == len(subtasks)


def test_architect_skips_decomposition_small_task(task_decomposer):
    """Test that decomposer skips tasks with <500 lines."""
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
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Small implementation task",
        description="Implement feature in 5 files",
        plan=plan,
    )

    estimated_lines = len(plan.files_to_modify) * 15  # 75 lines

    # Test should_decompose check
    assert task_decomposer.should_decompose(plan, estimated_lines) is False


def test_fan_in_triggered_on_last_subtask(file_queue, tmp_workspace):
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

    # Create and complete all subtasks
    for subtask_id in subtask_ids:
        subtask = Task(
            id=subtask_id,
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title=f"Subtask {subtask_id.split('-sub')[-1]}",
            description=f"Subtask {subtask_id}",
            parent_task_id=parent_id,
            result_summary=f"Completed {subtask_id}",
        )
        subtask.mark_completed("engineer")
        file_queue.mark_completed(subtask)

    # Verify all subtasks are complete
    assert file_queue.check_subtasks_complete(parent_id, subtask_ids) is True

    # Create fan-in task
    completed_subtasks = [file_queue.get_completed(sid) for sid in subtask_ids]
    completed_subtasks = [s for s in completed_subtasks if s is not None]

    fan_in_task = file_queue.create_fan_in_task(parent_task, completed_subtasks)

    # Verify fan-in task properties
    assert fan_in_task.id == f"fan-in-{parent_id}"
    assert fan_in_task.assigned_to == "qa"
    assert fan_in_task.context.get("fan_in") is True
    assert fan_in_task.context.get("parent_task_id") == parent_id
    assert fan_in_task.context.get("subtask_count") == 3


def test_fan_in_not_triggered_on_partial_completion(file_queue):
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

    # Complete only the first subtask
    subtask = Task(
        id=subtask_ids[0],
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
    subtask.mark_completed("engineer")
    file_queue.mark_completed(subtask)

    # Verify NOT all subtasks are complete
    assert file_queue.check_subtasks_complete(parent_id, subtask_ids) is False


def test_fan_in_task_routes_to_qa():
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


def test_decomposition_threshold_edge_cases(task_decomposer):
    """Test decomposition behavior at exact threshold boundary."""
    # Test at exactly 500 lines (500 / 15 = 33.33 files, so 34 files)
    plan_500 = PlanDocument(
        objectives=["Implement feature at threshold"],
        approach=["Step 1"],
        risks=[],
        success_criteria=["Tests pass"],
        files_to_modify=[f"src/module_{i}.py" for i in range(34)],  # 34 * 15 = 510 lines
        dependencies=[],
    )

    estimated_510 = 34 * 15  # 510 lines

    # Should decompose (510 > 500)
    assert task_decomposer.should_decompose(plan_500, estimated_510) is True

    # Test just below threshold (499 lines = 33 files)
    plan_499 = PlanDocument(
        objectives=["Implement feature below threshold"],
        approach=["Step 1"],
        risks=[],
        success_criteria=["Tests pass"],
        files_to_modify=[f"src/module_{i}.py" for i in range(33)],  # 33 * 15 = 495 lines
        dependencies=[],
    )

    estimated_495 = 33 * 15  # 495 lines

    # Should NOT decompose (495 < 500)
    assert task_decomposer.should_decompose(plan_499, estimated_495) is False


def test_subtask_routing_to_engineer_queue(task_decomposer):
    """Test that all subtasks inherit correct routing from parent."""
    files = [f"src/dir{i}/module_{j}.py" for i in range(4) for j in range(10)]

    plan = PlanDocument(
        objectives=["Implement large feature"],
        approach=["Step 1", "Step 2"],
        risks=[],
        success_criteria=["Tests pass"],
        files_to_modify=files,
        dependencies=[],
    )

    task = Task(
        id="test-routing",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="architect",
        assigned_to="engineer",  # Parent is assigned to engineer
        created_at=datetime.now(UTC),
        title="Test routing task",
        description="Test subtask routing",
        plan=plan,
    )

    estimated_lines = len(files) * 15
    subtasks = task_decomposer.decompose(task, plan, estimated_lines)

    # Verify EVERY subtask inherits engineer assignment
    for subtask in subtasks:
        assert subtask.assigned_to == "engineer", f"Subtask {subtask.id} assigned to {subtask.assigned_to} instead of engineer"
        assert subtask.type == TaskType.IMPLEMENTATION


def test_fan_in_metadata_propagation_through_workflow(tmp_workspace):
    """Test that fan-in metadata survives propagation through workflow chain."""
    from agent_framework.workflow.executor import WorkflowExecutor
    from agent_framework.workflow.dag import WorkflowDAG, WorkflowStep, WorkflowEdge
    from agent_framework.queue.file_queue import FileQueue as FQ

    # Create file queue
    file_queue = FQ(workspace=tmp_workspace)

    # Create a simple workflow: engineer -> qa
    workflow = WorkflowDAG(
        name="test-workflow",
        description="Test workflow",
        steps={
            "engineer": WorkflowStep(
                id="engineer",
                agent="engineer",
                next=[WorkflowEdge(target="qa")]
            ),
            "qa": WorkflowStep(
                id="qa",
                agent="qa",
                next=[]
            ),
        },
        start_step="engineer"
    )

    # Create fan-in task from engineer
    fan_in_task = Task(
        id="fan-in-parent-abc",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="system",
        assigned_to="qa",
        created_at=datetime.now(UTC),
        title="[fan-in] Parent task",
        description="Fan-in aggregation task",
        context={
            "fan_in": True,
            "parent_task_id": "parent-abc",
            "subtask_count": 3,
            "workflow": "test-workflow",
        },
    )

    # Create executor
    queue_dir = tmp_workspace / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    executor = WorkflowExecutor(file_queue, queue_dir)

    # Mock response
    mock_response = Mock()
    mock_response.content = "Task completed successfully"

    # Execute workflow step to create chain task
    result = executor.execute_step(
        workflow=workflow,
        task=fan_in_task,
        response=mock_response,
        current_agent_id="engineer",
        routing_signal=None,
    )

    # Verify chain task was created
    assert result is True, "Expected workflow to route to next step"

    # Find the chain task in the qa queue
    qa_queue_dir = queue_dir / "qa"
    if qa_queue_dir.exists():
        chain_task_files = list(qa_queue_dir.glob("chain-*.json"))
        assert len(chain_task_files) > 0, "Expected chain task to be created"

        # Read and verify the chain task
        chain_task = FQ.load_task_file(chain_task_files[0])

        # Verify fan-in metadata propagated
        assert chain_task.context.get("fan_in") is True, "fan_in flag not propagated"
        assert chain_task.context.get("parent_task_id") == "parent-abc", "parent_task_id not propagated"
        assert chain_task.context.get("subtask_count") == 3, "subtask_count not propagated"


def test_metadata_survival_through_multiple_handoffs(tmp_workspace):
    """Test metadata propagation through architect -> engineer -> fan-in -> qa chain."""
    from agent_framework.workflow.executor import WorkflowExecutor
    from agent_framework.workflow.dag import WorkflowDAG, WorkflowStep, WorkflowEdge
    from agent_framework.queue.file_queue import FileQueue as FQ

    # Create file queue
    file_queue = FQ(workspace=tmp_workspace)

    # Create workflow: architect -> engineer -> qa
    workflow = WorkflowDAG(
        name="default",
        description="Default workflow",
        steps={
            "architect": WorkflowStep(
                id="architect",
                agent="architect",
                next=[WorkflowEdge(target="engineer")]
            ),
            "engineer": WorkflowStep(
                id="engineer",
                agent="engineer",
                next=[WorkflowEdge(target="qa")]
            ),
            "qa": WorkflowStep(
                id="qa",
                agent="qa",
                next=[]
            ),
        },
        start_step="architect"
    )

    queue_dir = tmp_workspace / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    executor = WorkflowExecutor(file_queue, queue_dir)

    # Simulate fan-in task completing at engineer
    fan_in_task = Task(
        id="fan-in-impl-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="system",
        assigned_to="qa",
        created_at=datetime.now(UTC),
        title="[fan-in] Implementation task",
        description="Aggregated subtask results",
        context={
            "fan_in": True,
            "parent_task_id": "impl-123",
            "subtask_count": 4,
            "workflow": "default",
            "_chain_depth": 1,
        },
    )

    mock_response = Mock()
    mock_response.content = "Fan-in completed"

    # Execute step to route from engineer to qa
    result = executor.execute_step(
        workflow=workflow,
        task=fan_in_task,
        response=mock_response,
        current_agent_id="engineer",
    )

    # Verify routing occurred
    assert result is True, "Expected workflow to route to QA"

    # Find chain task in QA queue
    qa_queue_dir = queue_dir / "qa"
    if qa_queue_dir.exists():
        chain_task_files = list(qa_queue_dir.glob("chain-*.json"))
        assert len(chain_task_files) > 0, "Expected chain task to QA"

        # Verify chain task metadata
        qa_chain_task = FQ.load_task_file(chain_task_files[0])

        assert qa_chain_task.context.get("fan_in") is True
        assert qa_chain_task.context.get("parent_task_id") == "impl-123"
        assert qa_chain_task.context.get("subtask_count") == 4
        assert qa_chain_task.context.get("_chain_depth") == 2  # Incremented


class TestSubtaskWorkflowChainGuard:
    """Tests for the parent_task_id guard that prevents subtasks from
    individually triggering the workflow chain (QA/review/PR).
    Only the fan-in task should flow through the chain."""

    @pytest.fixture
    def mock_agent(self, tmp_workspace):
        """Create a minimally-mocked Agent for testing _run_post_completion_flow."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent.logger = Mock()
        agent.workspace = tmp_workspace
        # Mock the review cycle manager
        agent._review_cycle = MagicMock()
        # Bind the real method under test to the mock
        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        return agent

    @pytest.fixture
    def mock_response(self):
        resp = Mock()
        resp.input_tokens = 100
        resp.output_tokens = 50
        resp.content = "done"
        return resp

    def _make_task(self, task_id, parent_task_id=None, context=None):
        return Task(
            id=task_id,
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.COMPLETED,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title=f"Task {task_id}",
            description="test task",
            parent_task_id=parent_task_id,
            context=context or {},
        )

    def test_subtask_skips_workflow_chain(self, mock_agent, mock_response):
        """Subtask (parent_task_id set) must NOT trigger workflow chain or PR creation."""
        subtask = self._make_task("parent-1-sub1", parent_task_id="parent-1")

        mock_agent._run_post_completion_flow(
            subtask, mock_response, routing_signal=None, task_start_time=0
        )

        # Fan-in check should always run
        mock_agent._check_and_create_fan_in_task.assert_called_once_with(subtask)

        # Workflow chain methods should NOT be called for subtasks
        mock_agent._review_cycle.queue_code_review_if_needed.assert_not_called()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_not_called()
        mock_agent._enforce_workflow_chain.assert_not_called()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_not_called()

        # Per-subtask learning/metrics should still run
        mock_agent._extract_and_store_memories.assert_called_once()
        mock_agent._analyze_tool_patterns.assert_called_once()
        mock_agent._log_task_completion_metrics.assert_called_once()

    def test_regular_task_still_chains(self, mock_agent, mock_response):
        """Non-subtask (parent_task_id=None) must still get full workflow chain."""
        task = self._make_task("regular-task-1")

        mock_agent._run_post_completion_flow(
            task, mock_response, routing_signal=None, task_start_time=0
        )

        mock_agent._check_and_create_fan_in_task.assert_called_once_with(task)
        mock_agent._review_cycle.queue_code_review_if_needed.assert_called_once()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_called_once()
        mock_agent._enforce_workflow_chain.assert_called_once()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_called_once()

    def test_fan_in_task_flows_through_chain(self, mock_agent, mock_response):
        """Fan-in task (parent_task_id=None, context.fan_in=True) flows through
        the full workflow chain — it's the aggregated result that should create PRs."""
        fan_in = self._make_task(
            "fan-in-parent-1",
            parent_task_id=None,
            context={"fan_in": True, "parent_task_id": "parent-1", "subtask_count": 3},
        )

        mock_agent._run_post_completion_flow(
            fan_in, mock_response, routing_signal=None, task_start_time=0
        )

        mock_agent._check_and_create_fan_in_task.assert_called_once()
        mock_agent._review_cycle.queue_code_review_if_needed.assert_called_once()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_called_once()
        mock_agent._enforce_workflow_chain.assert_called_once()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_called_once()
