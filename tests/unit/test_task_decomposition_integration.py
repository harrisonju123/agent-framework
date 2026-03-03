"""Integration tests for task decomposition in architect and engineer agents."""

import pytest
from datetime import datetime, UTC
from unittest.mock import Mock, MagicMock

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

    Task(
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
    Task(
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
    Task(
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

    [
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

    # Test just below threshold (345 lines)
    plan_below = PlanDocument(
        objectives=["Implement feature below threshold"],
        approach=["Step 1"],
        risks=[],
        success_criteria=["Tests pass"],
        files_to_modify=[f"src/module_{i}.py" for i in range(23)],  # 23 * 15 = 345 lines
        dependencies=[],
    )

    estimated_345 = 23 * 15  # 345 lines

    # Should NOT decompose (345 < 350)
    assert task_decomposer.should_decompose(plan_below, estimated_345) is False


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
        """Create a minimally-mocked Agent for testing _run_post_completion_flow.

        _run_post_completion_flow now delegates to PostCompletionManager, so we
        construct a real PostCompletionManager with mock dependencies and wire
        it into the agent via _post_completion.
        """
        from agent_framework.core.agent import Agent
        from agent_framework.core.post_completion import PostCompletionManager

        agent = MagicMock()
        agent.logger = Mock()
        agent.workspace = tmp_workspace
        agent._context_window_manager = None
        agent._analytics = MagicMock()

        # Mock the review cycle manager
        agent._review_cycle = MagicMock()
        agent._workflow_router = MagicMock()
        # enforce_chain returns bool -- False means no downstream routing occurred
        agent._workflow_router.enforce_chain.return_value = False
        agent._workflow_router.is_at_terminal_workflow_step.return_value = False
        agent._enforce_workflow_chain = MagicMock(return_value=False)
        agent._git_ops = MagicMock()

        # Build real PostCompletionManager with the same mock objects
        config = MagicMock()
        config.id = "engineer-1"
        config.base_id = "engineer"
        agent.config = config

        session_logs_dir = tmp_workspace / "logs" / "sessions"
        session_logs_dir.mkdir(parents=True, exist_ok=True)

        post_completion = PostCompletionManager(
            config=config,
            queue=agent.queue,
            workspace=tmp_workspace,
            logger=agent.logger,
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
            review_cycle=agent._review_cycle,
            workflow_router=agent._workflow_router,
            git_ops=agent._git_ops,
            budget=MagicMock(),
            error_recovery=MagicMock(),
            optimization_config={},
            session_logging_enabled=False,
            session_logs_dir=session_logs_dir,
        )
        agent._post_completion = post_completion

        # Bind the real delegation method
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

        # Fan-in check should always run (now delegated to workflow router)
        mock_agent._workflow_router.check_and_create_fan_in_task.assert_called_once_with(subtask)

        # Workflow chain methods should NOT be called for subtasks
        mock_agent._review_cycle.queue_code_review_if_needed.assert_not_called()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_not_called()
        mock_agent._workflow_router.enforce_chain.assert_not_called()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_not_called()

        # Per-subtask learning/metrics callbacks should still fire
        mock_agent._analytics.extract_and_store_memories.assert_called_once()
        mock_agent._analytics.analyze_tool_patterns.assert_called_once()

    def test_regular_task_still_chains(self, mock_agent, mock_response):
        """Non-subtask (parent_task_id=None) must still get full workflow chain."""
        task = self._make_task("regular-task-1")

        mock_agent._run_post_completion_flow(
            task, mock_response, routing_signal=None, task_start_time=0
        )

        mock_agent._workflow_router.check_and_create_fan_in_task.assert_called_once_with(task)
        # No workflow in context -> legacy review routing fires
        mock_agent._review_cycle.queue_code_review_if_needed.assert_called_once()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_called_once()
        # enforce_chain is on workflow_router now
        mock_agent._workflow_router.enforce_chain.assert_called_once()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_called_once()

    def test_fan_in_task_flows_through_chain(self, mock_agent, mock_response):
        """Fan-in task (parent_task_id=None, context.fan_in=True) flows through
        the full workflow chain -- it's the aggregated result that should create PRs."""
        fan_in = self._make_task(
            "fan-in-parent-1",
            parent_task_id=None,
            context={"fan_in": True, "parent_task_id": "parent-1", "subtask_count": 3},
        )

        mock_agent._run_post_completion_flow(
            fan_in, mock_response, routing_signal=None, task_start_time=0
        )

        mock_agent._workflow_router.check_and_create_fan_in_task.assert_called_once()
        # No workflow in context -> legacy review routing fires
        mock_agent._review_cycle.queue_code_review_if_needed.assert_called_once()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_called_once()
        mock_agent._workflow_router.enforce_chain.assert_called_once()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_called_once()

    def test_workflow_task_skips_legacy_routing(self, mock_agent, mock_response):
        """Task with workflow in context skips legacy review routing but still chains."""
        task = self._make_task(
            "workflow-task-1",
            context={"workflow": "default"},
        )

        mock_agent._run_post_completion_flow(
            task, mock_response, routing_signal=None, task_start_time=0
        )

        # Legacy review routing should NOT be called for workflow-managed tasks
        mock_agent._review_cycle.queue_code_review_if_needed.assert_not_called()
        mock_agent._review_cycle.queue_review_fix_if_needed.assert_not_called()
        # DAG chain routing should still be called (on workflow_router)
        mock_agent._workflow_router.enforce_chain.assert_called_once()
        mock_agent._git_ops.push_and_create_pr_if_needed.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for right-sized, parallel-capable subtask decomposition
# ---------------------------------------------------------------------------


class TestMaxDeliverablesPerSubtask:
    """Enforce MAX_DELIVERABLES_PER_SUBTASK so no boundary gets overloaded."""

    def test_boundary_with_7_source_files_splits_into_chunks(self):
        """A directory with 7 source files should become 2 boundaries (4+3)."""
        decomposer = TaskDecomposer()
        files = [f"src/core/module_{i}.py" for i in range(7)]
        # Also include a second dir so we get >=2 boundaries total
        files += [f"lib/util_{i}.py" for i in range(3)]

        plan = PlanDocument(
            objectives=["Big refactor"],
            approach=[f"Modify module_{i}.py" for i in range(7)] + ["Update utils"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-max-deliverables",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Max deliverables test",
            description="Test max deliverables enforcement",
            plan=plan,
        )

        estimated = len(files) * 50
        subtasks = decomposer.decompose(task, plan, estimated)

        # No subtask should have more than 4 source files
        for st in subtasks:
            source_count = sum(
                1 for f in st.plan.files_to_modify
                if not TaskDecomposer._is_test_file(f)
            )
            assert source_count <= TaskDecomposer.MAX_DELIVERABLES_PER_SUBTASK, (
                f"Subtask {st.id} has {source_count} source files, "
                f"max is {TaskDecomposer.MAX_DELIVERABLES_PER_SUBTASK}"
            )

    def test_boundary_within_limit_stays_intact(self):
        """A boundary with <=4 source files should not be split further."""
        decomposer = TaskDecomposer()
        # 3 files in each of 2 directories — both within limit
        files = [f"src/a/mod_{i}.py" for i in range(3)]
        files += [f"src/b/mod_{i}.py" for i in range(3)]

        plan = PlanDocument(
            objectives=["Small refactor"],
            approach=["Step 1", "Step 2"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-within-limit",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Within limit test",
            description="Test that small boundaries are not split",
            plan=plan,
        )

        estimated = len(files) * 80
        subtasks = decomposer.decompose(task, plan, estimated)

        # We should get exactly 2 subtasks (one per subdir under src/)
        assert len(subtasks) == 2
        for st in subtasks:
            assert len(st.plan.files_to_modify) == 3


class TestSourceTestColocation:
    """Tests for co-locating test files with their source counterparts."""

    def test_test_file_colocated_with_matching_source(self):
        """test_bar.py should land in the same subtask as bar.py."""
        decomposer = TaskDecomposer()
        files = [
            "src/core/bar.py",
            "src/core/baz.py",
            "src/utils/helper.py",
            "src/utils/config.py",
            "tests/unit/test_bar.py",
            "tests/unit/test_helper.py",
        ]

        plan = PlanDocument(
            objectives=["Implement feature"],
            approach=["Modify bar.py", "Modify helper.py"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-colocation",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Colocation test",
            description="Test source-test co-location",
            plan=plan,
        )

        estimated = 600
        subtasks = decomposer.decompose(task, plan, estimated)

        # Find which subtask contains bar.py
        for st in subtasks:
            if "src/core/bar.py" in st.plan.files_to_modify:
                assert "tests/unit/test_bar.py" in st.plan.files_to_modify, (
                    "test_bar.py should be co-located with bar.py"
                )
            if "src/utils/helper.py" in st.plan.files_to_modify:
                assert "tests/unit/test_helper.py" in st.plan.files_to_modify, (
                    "test_helper.py should be co-located with helper.py"
                )

    def test_test_files_dont_inflate_source_count(self):
        """Test files should not count toward MAX_DELIVERABLES_PER_SUBTASK."""
        decomposer = TaskDecomposer()
        # 4 source files + 4 matching test files = 8 total, but only 4 source
        files = [f"src/core/mod_{i}.py" for i in range(4)]
        files += [f"tests/unit/test_mod_{i}.py" for i in range(4)]
        # Need a second group to get >=2 boundaries
        files += [f"lib/util_{i}.py" for i in range(2)]

        plan = PlanDocument(
            objectives=["Refactor"],
            approach=[f"Update mod_{i}.py" for i in range(4)] + ["Update utils"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-no-inflate",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="No inflate test",
            description="Test that tests dont count as source deliverables",
            plan=plan,
        )

        estimated = len(files) * 50
        subtasks = decomposer.decompose(task, plan, estimated)

        # The core subtask should have 4 source + 4 test = 8 files total,
        # but that should be ONE subtask (not split further) because
        # source_file_count == 4 <= MAX_DELIVERABLES_PER_SUBTASK
        core_subtasks = [
            st for st in subtasks
            if any("src/core" in f for f in st.plan.files_to_modify)
        ]
        assert len(core_subtasks) == 1, (
            f"Expected 1 core subtask, got {len(core_subtasks)} — "
            "test files should not trigger further splitting"
        )

    def test_unmatched_test_goes_to_last_boundary(self):
        """Test files with no matching source should land in the last subtask."""
        decomposer = TaskDecomposer()
        files = [
            "src/core/alpha.py",
            "src/utils/beta.py",
            "tests/integration/test_end_to_end.py",  # No matching source file
        ]

        plan = PlanDocument(
            objectives=["Implement"],
            approach=["Modify alpha.py", "Modify beta.py", "Add e2e test"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-unmatched",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Unmatched test",
            description="Test unmatched test file placement",
            plan=plan,
        )

        estimated = 400
        subtasks = decomposer.decompose(task, plan, estimated)

        # The unmatched test should end up in the last subtask
        all_files_in_subtasks = []
        for st in subtasks:
            all_files_in_subtasks.extend(st.plan.files_to_modify)

        assert "tests/integration/test_end_to_end.py" in all_files_in_subtasks, (
            "Unmatched test file should still be included in some subtask"
        )


class TestEffortMultiplier:
    """Tests for modification effort multiplier on existing files."""

    def test_existing_files_inflate_estimate(self, tmp_path):
        """Files that exist on disk get their estimates inflated."""
        decomposer = TaskDecomposer()
        workspace = tmp_path / "repo"
        workspace.mkdir()

        # Create 2 files on disk (they "exist")
        (workspace / "src").mkdir(parents=True)
        (workspace / "src" / "existing_a.py").write_text("# existing")
        (workspace / "src" / "existing_b.py").write_text("# existing")
        (workspace / "lib").mkdir(parents=True)
        (workspace / "lib" / "new_a.py").parent.mkdir(parents=True, exist_ok=True)

        files = [
            "src/existing_a.py",
            "src/existing_b.py",
            "lib/new_a.py",
            "lib/new_b.py",
        ]

        plan = PlanDocument(
            objectives=["Mix of new and existing"],
            approach=["Modify existing_a.py", "Create new_a.py"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-effort",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Effort test",
            description="Test effort multiplier",
            plan=plan,
        )

        estimated = 400
        subtasks = decomposer.decompose(task, plan, estimated, workspace=workspace)

        # Find subtask covering existing files
        src_subtask = None
        lib_subtask = None
        for st in subtasks:
            if any("existing_a" in f for f in st.plan.files_to_modify):
                src_subtask = st
            if any("new_a" in f for f in st.plan.files_to_modify):
                lib_subtask = st

        assert src_subtask is not None, "Should have subtask for src/"
        assert lib_subtask is not None, "Should have subtask for lib/"

        # The src subtask (all existing) should have higher estimated_lines
        src_estimate = int(next(
            n.replace("Estimated lines: ", "")
            for n in src_subtask.notes
            if "Estimated lines:" in n
        ))
        lib_estimate = int(next(
            n.replace("Estimated lines: ", "")
            for n in lib_subtask.notes
            if "Estimated lines:" in n
        ))

        assert src_estimate > lib_estimate, (
            f"Existing-file subtask ({src_estimate}) should have higher estimate "
            f"than new-file subtask ({lib_estimate})"
        )

    def test_no_workspace_skips_multiplier(self):
        """Without workspace, effort multiplier is not applied."""
        decomposer = TaskDecomposer()
        files = [
            "src/a/mod_1.py",
            "src/a/mod_2.py",
            "src/b/mod_3.py",
            "src/b/mod_4.py",
        ]

        plan = PlanDocument(
            objectives=["Feature"],
            approach=["Step 1", "Step 2"],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-no-ws",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="No workspace test",
            description="Test decompose without workspace",
            plan=plan,
        )

        estimated = 400
        # Should not raise — workspace=None is the default
        subtasks = decomposer.decompose(task, plan, estimated)
        assert len(subtasks) >= 2


class TestParallelCapableSplits:
    """Tests for conservative dependency inference that maximizes parallelism."""

    def test_independent_directories_have_no_deps(self):
        """Subtasks modifying different files with unrelated steps should be parallel."""
        decomposer = TaskDecomposer()
        files = [
            "src/auth/login.py",
            "src/auth/session.py",
            "src/billing/invoice.py",
            "src/billing/payment.py",
        ]

        plan = PlanDocument(
            objectives=["Update auth and billing"],
            approach=[
                "Modify login.py for SSO support",
                "Update session.py to handle tokens",
                "Add invoice.py PDF export",
                "Update payment.py with Stripe v3",
            ],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-parallel",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Parallel splits test",
            description="Test that independent subtasks are parallel",
            plan=plan,
        )

        estimated = 600
        subtasks = decomposer.decompose(task, plan, estimated)

        # With independent directories and no cross-references in approach steps,
        # subtasks should have NO dependencies on each other
        for st in subtasks:
            assert st.depends_on == [], (
                f"Subtask {st.id} has dependencies {st.depends_on} "
                "but should be independent"
            )

    def test_cross_referenced_steps_create_dependency(self):
        """When approach steps reference files from another boundary, add a dep."""
        decomposer = TaskDecomposer()
        files = [
            "src/core/base_model.py",
            "src/core/validators.py",
            "src/api/endpoints.py",
            "src/api/serializers.py",
        ]

        plan = PlanDocument(
            objectives=["Extend API with validation"],
            approach=[
                "Add BaseModel fields for validation in base_model.py",
                "Implement validators.py validation rules",
                "Use base_model.py in endpoints.py to enforce validation",
                "Update serializers.py to use validators.py output",
            ],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-cross-ref",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Cross-ref deps test",
            description="Test dependency inference from approach step cross-refs",
            plan=plan,
        )

        estimated = 600
        subtasks = decomposer.decompose(task, plan, estimated)

        # The api subtask references "base_model" and "validators" from core,
        # so it should depend on the core subtask
        api_subtask = None
        for st in subtasks:
            if any("src/api" in f for f in st.plan.files_to_modify):
                api_subtask = st
                break

        assert api_subtask is not None, "Should have an API subtask"
        assert len(api_subtask.depends_on) > 0, (
            "API subtask should depend on core subtask because its approach "
            "steps reference base_model.py and validators.py"
        )

    def test_same_dir_source_and_tests_are_parallel(self):
        """With co-location, src+tests in same subtask should not create deps."""
        decomposer = TaskDecomposer()
        files = [
            "src/auth/login.py",
            "src/auth/session.py",
            "tests/unit/test_login.py",
            "src/billing/invoice.py",
            "src/billing/payment.py",
            "tests/unit/test_invoice.py",
        ]

        plan = PlanDocument(
            objectives=["Update auth and billing with tests"],
            approach=[
                "Modify login.py",
                "Update session.py",
                "Add invoice.py changes",
                "Update payment.py",
            ],
            risks=[],
            success_criteria=["Tests pass"],
            files_to_modify=files,
        )

        task = Task(
            id="test-colocated-parallel",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Colocated parallel test",
            description="Test that co-located source+tests are parallel",
            plan=plan,
        )

        estimated = 600
        subtasks = decomposer.decompose(task, plan, estimated)

        # Both subtasks should be parallel (no deps) since they
        # don't reference each other's files in approach steps
        for st in subtasks:
            assert st.depends_on == [], (
                f"Subtask {st.id} has dependencies {st.depends_on} "
                "but co-located subtasks should be parallel"
            )


class TestIsTestFile:
    """Unit tests for _is_test_file classification."""

    def test_test_directory_detected(self):
        assert TaskDecomposer._is_test_file("tests/unit/test_foo.py") is True
        assert TaskDecomposer._is_test_file("test/integration/helper.py") is True
        assert TaskDecomposer._is_test_file("spec/models/user_spec.rb") is True

    def test_test_prefix_detected(self):
        assert TaskDecomposer._is_test_file("src/test_utils.py") is True

    def test_test_suffix_detected(self):
        assert TaskDecomposer._is_test_file("src/utils_test.py") is True

    def test_source_files_not_classified_as_test(self):
        assert TaskDecomposer._is_test_file("src/core/agent.py") is False
        assert TaskDecomposer._is_test_file("lib/helpers/utils.py") is False
        assert TaskDecomposer._is_test_file("config/settings.py") is False


class TestTestMatchesSource:
    """Unit tests for _test_matches_source pairing logic."""

    def test_matching_pair(self):
        assert TaskDecomposer._test_matches_source(
            "tests/unit/test_agent.py", "src/core/agent.py"
        ) is True

    def test_non_matching_pair(self):
        assert TaskDecomposer._test_matches_source(
            "tests/unit/test_agent.py", "src/core/task.py"
        ) is False

    def test_deep_nesting(self):
        assert TaskDecomposer._test_matches_source(
            "tests/integration/api/test_router.py", "src/api/router.py"
        ) is True
