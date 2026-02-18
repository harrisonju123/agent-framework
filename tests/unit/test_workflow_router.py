"""Tests for WorkflowRouter extracted from Agent class."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from agent_framework.core.workflow_router import WorkflowRouter
from agent_framework.core.config import WorkflowDefinition
from agent_framework.core.routing import RoutingSignal, WORKFLOW_COMPLETE
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Fixtures --

def _make_task(workflow="default", task_id="task-abc123def456", **ctx_overrides):
    context = {"workflow": workflow, **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


def _make_response(content="Done.", pr_url=None):
    """Minimal LLM response stub."""
    return SimpleNamespace(
        content=content if not pr_url else f"Created PR: {pr_url}",
        error=None,
        input_tokens=100,
        output_tokens=50,
        model_used="sonnet",
        latency_ms=1000,
        finish_reason="end_turn",
    )


DEFAULT_WORKFLOW = WorkflowDefinition(
    description="Default workflow",
    agents=["architect", "engineer", "qa"],
)

PR_WORKFLOW = WorkflowDefinition(
    description="Workflow with PR creator",
    agents=["architect", "engineer", "qa"],
    pr_creator="architect",
)

ANALYSIS_WORKFLOW = WorkflowDefinition(
    description="Analysis only",
    agents=["architect"],
)


@pytest.fixture
def queue(tmp_path):
    """FileQueue mock with real queue_dir and completed_dir for file existence checks."""
    q = MagicMock()
    q.queue_dir = tmp_path / "queues"
    q.queue_dir.mkdir()
    q.completed_dir = tmp_path / "completed"
    q.completed_dir.mkdir()
    return q


@pytest.fixture
def config():
    """Mock agent config."""
    cfg = MagicMock()
    cfg.id = "engineer"
    cfg.base_id = "engineer"
    return cfg


@pytest.fixture
def router(config, queue, tmp_path):
    """Create WorkflowRouter instance for testing."""
    logger = MagicMock()
    session_logger = MagicMock()
    workflows_config = {"default": DEFAULT_WORKFLOW, "analysis": ANALYSIS_WORKFLOW}
    agents_config = [
        SimpleNamespace(id="architect"),
        SimpleNamespace(id="engineer"),
        SimpleNamespace(id="qa"),
    ]

    # Initialize workflow executor
    from agent_framework.workflow.executor import WorkflowExecutor
    workflow_executor = WorkflowExecutor(queue, queue.queue_dir)

    return WorkflowRouter(
        config=config,
        queue=queue,
        workspace=tmp_path,
        logger=logger,
        session_logger=session_logger,
        workflows_config=workflows_config,
        workflow_executor=workflow_executor,
        agents_config=agents_config,
        multi_repo_manager=None,
    )


# -- Fan-in task creation --

class TestFanInTaskCreation:
    def test_creates_fan_in_when_all_subtasks_complete(self, router, queue):
        """Creates fan-in task when all subtasks are complete."""
        parent = _make_task(task_id="parent-123")
        parent.subtask_ids = ["sub-1", "sub-2", "sub-3"]

        subtask = _make_task(task_id="sub-3")
        subtask.parent_task_id = "parent-123"

        queue.find_task.return_value = parent
        queue.check_subtasks_complete.return_value = True
        queue._fan_in_already_created.return_value = False
        queue.get_completed.side_effect = [
            _make_task(task_id="sub-1"),
            _make_task(task_id="sub-2"),
            _make_task(task_id="sub-3"),
        ]

        fan_in = _make_task(task_id="fan-in-parent-123")
        queue.create_fan_in_task.return_value = fan_in

        router.check_and_create_fan_in_task(subtask)

        queue.create_fan_in_task.assert_called_once()
        queue.push.assert_called_once_with(fan_in, fan_in.assigned_to)

    def test_skips_when_not_all_complete(self, router, queue):
        """Does not create fan-in when some subtasks are pending."""
        parent = _make_task(task_id="parent-123")
        parent.subtask_ids = ["sub-1", "sub-2", "sub-3"]

        subtask = _make_task(task_id="sub-1")
        subtask.parent_task_id = "parent-123"

        queue.find_task.return_value = parent
        queue.check_subtasks_complete.return_value = False

        router.check_and_create_fan_in_task(subtask)

        queue.create_fan_in_task.assert_not_called()
        queue.push.assert_not_called()

    def test_skips_when_no_parent(self, router, queue):
        """Does not create fan-in for tasks without a parent."""
        task = _make_task()

        router.check_and_create_fan_in_task(task)

        queue.find_task.assert_not_called()


# -- Task decomposition --

class TestTaskDecomposition:
    def test_should_decompose_large_task(self, router):
        """Returns True for tasks with many files to modify."""
        task = _make_task()
        task.plan = MagicMock()
        task.plan.files_to_modify = ["file1.py", "file2.py", "file3.py", "file4.py"]
        task.parent_task_id = None

        with patch("agent_framework.core.task_decomposer.TaskDecomposer") as mock_decomposer:
            decomposer_instance = MagicMock()
            decomposer_instance.should_decompose.return_value = True
            mock_decomposer.return_value = decomposer_instance

            result = router.should_decompose_task(task)

            assert result is True

    def test_should_not_decompose_subtask(self, router):
        """Does not decompose tasks that are already subtasks."""
        task = _make_task()
        task.plan = MagicMock()
        task.plan.files_to_modify = ["file1.py", "file2.py"]
        task.parent_task_id = "parent-123"

        result = router.should_decompose_task(task)

        assert result is False

    def test_should_not_decompose_without_plan(self, router):
        """Does not decompose tasks without a plan."""
        task = _make_task()
        task.plan = None

        result = router.should_decompose_task(task)

        assert result is False

    def test_decompose_and_queue_subtasks(self, router, queue):
        """Decomposes task into subtasks and queues them."""
        task = _make_task(task_id="parent-123")
        task.plan = MagicMock()
        task.plan.files_to_modify = ["file1.py", "file2.py"]

        subtask1 = _make_task(task_id="parent-123-sub-0")
        subtask2 = _make_task(task_id="parent-123-sub-1")

        with patch("agent_framework.core.task_decomposer.TaskDecomposer") as mock_decomposer:
            decomposer_instance = MagicMock()
            decomposer_instance.decompose.return_value = [subtask1, subtask2]
            mock_decomposer.return_value = decomposer_instance

            router.decompose_and_queue_subtasks(task)

            assert queue.push.call_count == 2
            assert task.subtask_ids == [subtask1.id, subtask2.id]

    def test_decompose_persists_to_completed_dir(self, router, queue):
        """subtask_ids are written to the completed copy (not queue.update)."""
        task = _make_task(task_id="parent-456")
        task.plan = MagicMock()
        task.plan.files_to_modify = ["file1.py"]

        subtask1 = _make_task(task_id="parent-456-sub-0")

        # Simulate mark_completed() having already moved task to completed_dir
        completed_file = queue.completed_dir / f"{task.id}.json"
        completed_file.write_text(task.model_dump_json(indent=2))

        with patch("agent_framework.core.task_decomposer.TaskDecomposer") as mock_decomposer:
            decomposer_instance = MagicMock()
            decomposer_instance.decompose.return_value = [subtask1]
            mock_decomposer.return_value = decomposer_instance

            router.decompose_and_queue_subtasks(task)

        # queue.update should NOT have been called (completed file existed)
        queue.update.assert_not_called()

        # Verify subtask_ids were written to completed_dir
        import json
        persisted = json.loads(completed_file.read_text())
        assert persisted["subtask_ids"] == ["parent-456-sub-0"]

    def test_decompose_falls_back_to_queue_update(self, router, queue):
        """Falls back to queue.update() when completed file doesn't exist."""
        task = _make_task(task_id="parent-789")
        task.plan = MagicMock()
        task.plan.files_to_modify = ["file1.py"]

        subtask1 = _make_task(task_id="parent-789-sub-0")

        # No completed file — simulate task still in queue dir
        with patch("agent_framework.core.task_decomposer.TaskDecomposer") as mock_decomposer:
            decomposer_instance = MagicMock()
            decomposer_instance.decompose.return_value = [subtask1]
            mock_decomposer.return_value = decomposer_instance

            router.decompose_and_queue_subtasks(task)

        queue.update.assert_called_once_with(task)


# -- Workflow chain enforcement --

class TestEnforceChain:
    def test_skips_when_subtask_ids_set(self, router, queue):
        """Tasks already decomposed into subtasks skip chain routing entirely."""
        task = _make_task(workflow="default")
        task.subtask_ids = ["sub-1", "sub-2"]
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_not_called()
        router.logger.info.assert_any_call(
            f"Task {task.id} already decomposed into 2 subtasks, "
            f"skipping chain routing"
        )

    def test_queues_next_agent_no_pr(self, router, queue):
        """When no PR is created, chain task is queued to the next agent."""
        task = _make_task(workflow="default")
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"
        assert chain_task.assigned_to == "qa"

    def test_skips_last_agent_in_chain(self, router, queue):
        """Last agent in the chain has nobody to forward to."""
        router.config.base_id = "qa"
        task = _make_task(workflow="default")
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_not_called()

    def test_preview_routes_back_to_architect(self, router, queue):
        """Engineer completing a PREVIEW task routes back to architect, not QA."""
        task = _make_task(workflow="default")
        task.type = TaskType.PREVIEW
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "architect"
        assert chain_task.assigned_to == "architect"


# -- Terminal step detection --

class TestTerminalStepDetection:
    def test_intermediate_agent_is_not_terminal(self, router):
        """Engineer (middle of architect→engineer→qa) is not terminal."""
        task = _make_task(workflow="default")
        assert router.is_at_terminal_workflow_step(task) is False

    def test_last_agent_is_terminal(self, router):
        """QA (last in architect→engineer→qa) is terminal."""
        router.config.base_id = "qa"
        task = _make_task(workflow="default")
        assert router.is_at_terminal_workflow_step(task) is True

    def test_no_workflow_is_terminal(self, router):
        """Standalone tasks (no workflow key) default to terminal."""
        task = _make_task(workflow="default")
        del task.context["workflow"]
        assert router.is_at_terminal_workflow_step(task) is True


# -- Workflow context building --

class TestBuildWorkflowContext:
    def test_uses_task_context_changed_files(self, router):
        """Prefers changed_files from task context over git diff."""
        task = _make_task()
        task.context["changed_files"] = ["file1.py", "file2.py"]

        context = router.build_workflow_context(task)

        assert context["changed_files"] == ["file1.py", "file2.py"]

    def test_falls_back_to_git_diff(self, router):
        """Uses git diff when changed_files not in task context."""
        task = _make_task()

        with patch.object(router, "_get_changed_files", return_value=["file3.py"]):
            context = router.build_workflow_context(task)

            assert context["changed_files"] == ["file3.py"]

    def test_includes_test_results(self, router):
        """Includes test results if available in task context."""
        task = _make_task()
        task.context["test_result"] = "passed"

        context = router.build_workflow_context(task)

        assert context["test_result"] == "passed"

    def test_includes_verdict_when_present(self, router):
        """Passes verdict through so DAG condition evaluators can see it."""
        task = _make_task()
        task.context["verdict"] = "no_changes"

        context = router.build_workflow_context(task)

        assert context["verdict"] == "no_changes"

    def test_omits_verdict_when_absent(self, router):
        """No verdict in task context → no verdict in evaluation context."""
        task = _make_task()

        context = router.build_workflow_context(task)

        assert "verdict" not in context


# -- PR creation --

class TestPRCreation:
    def test_queues_pr_creation_at_terminal_step(self, router, queue):
        """Last agent queues PR creation task when pr_creator is configured."""
        router.config.base_id = "qa"
        router._workflows_config["pr_workflow"] = PR_WORKFLOW
        task = _make_task(workflow="pr_workflow", implementation_branch="feature/test")

        router.queue_pr_creation_if_needed(task, PR_WORKFLOW)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        assert pr_task.type == TaskType.PR_REQUEST
        assert pr_task.context["pr_creation_step"] is True

    def test_skips_pr_creation_when_no_pr_creator(self, router, queue):
        """Does not queue PR creation when workflow has no pr_creator."""
        task = _make_task(workflow="default")

        router.queue_pr_creation_if_needed(task, DEFAULT_WORKFLOW)

        queue.push.assert_not_called()

    def test_skips_pr_creation_when_pr_exists(self, router, queue):
        """Does not queue PR creation when pr_url already exists."""
        task = _make_task(workflow="pr_workflow")
        task.context["pr_url"] = "https://github.com/org/repo/pull/42"

        router.queue_pr_creation_if_needed(task, PR_WORKFLOW)

        queue.push.assert_not_called()

    def test_pr_task_id_uses_root_not_chain_prefix(self, router, queue):
        """PR task ID anchors on _root_task_id so chain hops don't nest 'chain-' prefixes."""
        router.config.base_id = "qa"
        router._workflows_config["pr_workflow"] = PR_WORKFLOW
        task = _make_task(
            workflow="pr_workflow",
            implementation_branch="feature/test",
            task_id="chain-chain-original-qa-1",
        )
        task.context["_root_task_id"] = "original"

        router.queue_pr_creation_if_needed(task, PR_WORKFLOW)

        queue.push.assert_called_once()
        pr_task = queue.push.call_args[0][0]
        # Should be "chain-original-architect-pr", not "chain-chain-chain-...original-qa-1-..."
        assert pr_task.id.count("chain-") == 1


# -- Agent routing --

class TestRouteToAgent:
    def test_routes_to_target_agent(self, router, queue):
        """Routes task to specified target agent."""
        task = _make_task()

        router.route_to_agent(task, "qa", "test failure")

        queue.push.assert_called_once()
        chain_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]
        assert target_queue == "qa"
        assert chain_task.assigned_to == "qa"

    def test_skips_duplicate_routing(self, router, queue):
        """Does not route if chain task already queued."""
        task = _make_task()

        # Mock the executor's duplicate check to return True
        router._workflow_executor._is_chain_task_already_queued = MagicMock(return_value=True)

        router.route_to_agent(task, "qa", "test failure")

        queue.push.assert_not_called()


# -- REVIEW/FIX guard for chain tasks --

class TestReviewFixGuard:
    def test_chain_review_task_passes_through_guard(self, router, queue):
        """Chain task with type REVIEW and chain_step=True is NOT blocked."""
        task = _make_task(chain_step=True, workflow_step="code_review")
        task.type = TaskType.REVIEW
        response = _make_response()

        router.enforce_chain(task, response)

        # Guard logs "handled by dedicated review routing" only for blocked tasks.
        blocked_calls = [
            call for call in router.logger.debug.call_args_list
            if "handled by dedicated review routing" in str(call)
        ]
        assert len(blocked_calls) == 0, "Chain REVIEW task was incorrectly blocked by guard"

    def test_non_chain_review_task_blocked(self, router, queue):
        """Legacy REVIEW task (no chain_step) IS blocked by the guard."""
        task = _make_task()
        task.type = TaskType.REVIEW
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_not_called()

    def test_non_chain_fix_task_blocked(self, router, queue):
        """Legacy FIX task (no chain_step) IS blocked by the guard."""
        task = _make_task()
        task.type = TaskType.FIX
        response = _make_response()

        router.enforce_chain(task, response)

        queue.push.assert_not_called()
