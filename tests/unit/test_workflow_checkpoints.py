"""Tests for workflow checkpoint functionality."""

import json
import os
import pytest
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.workflow.dag import (
    CheckpointConfig,
    WorkflowDAG,
    WorkflowStep,
    WorkflowEdge,
    EdgeCondition,
    EdgeConditionType,
)
from agent_framework.workflow.executor import WorkflowExecutor, resume_after_checkpoint


def create_test_task(task_id="test-task", status=TaskStatus.IN_PROGRESS):
    """Create a test task."""
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Test task",
        description="Test task for checkpoint testing",
        context={"workflow": "test_workflow"},
    )


def _build_checkpoint_workflow():
    """Build a workflow with a checkpoint at the engineer step."""
    return WorkflowDAG(
        name="test_workflow",
        description="Test workflow with checkpoint",
        steps={
            "engineer": WorkflowStep(
                id="engineer",
                agent="engineer",
                checkpoint=CheckpointConfig(
                    message="Review implementation before QA",
                    reason="High-risk deployment",
                ),
                next=[
                    WorkflowEdge(
                        target="qa",
                        condition=EdgeCondition(EdgeConditionType.ALWAYS),
                    )
                ],
            ),
            "qa": WorkflowStep(
                id="qa",
                agent="qa",
                next=[],
            ),
        },
        start_step="engineer",
    )


def _build_multi_checkpoint_workflow():
    """Build a workflow with checkpoints at two different steps."""
    return WorkflowDAG(
        name="multi_cp",
        description="Workflow with two checkpoints",
        steps={
            "engineer": WorkflowStep(
                id="engineer",
                agent="engineer",
                checkpoint=CheckpointConfig(
                    message="Review implementation",
                ),
                next=[
                    WorkflowEdge(
                        target="qa",
                        condition=EdgeCondition(EdgeConditionType.ALWAYS),
                    )
                ],
            ),
            "qa": WorkflowStep(
                id="qa",
                agent="qa",
                checkpoint=CheckpointConfig(
                    message="Review QA results before deploy",
                ),
                next=[],
            ),
        },
        start_step="engineer",
    )


# -- Task model tests --


def test_task_mark_awaiting_approval():
    """Mark task as awaiting approval at checkpoint."""
    task = create_test_task()

    task.mark_awaiting_approval("workflow-step1", "Review code changes before deployment")

    assert task.status == TaskStatus.AWAITING_APPROVAL
    assert task.checkpoint_reached == "workflow-step1"
    assert task.checkpoint_message == "Review code changes before deployment"
    assert task.approved_at is None
    assert task.approved_by is None


def test_task_mark_awaiting_approval_resets_prior_approval():
    """A new checkpoint clears any prior approval so it can't be bypassed."""
    task = create_test_task()
    task.approved_at = datetime.now(UTC)
    task.approved_by = "admin"
    task.checkpoint_reached = "old-checkpoint"

    task.mark_awaiting_approval("new-checkpoint", "Second gate")

    assert task.approved_at is None
    assert task.approved_by is None
    assert task.checkpoint_reached == "new-checkpoint"


def test_task_approve_checkpoint():
    """Approve a checkpoint — status becomes COMPLETED (LLM work is done)."""
    task = create_test_task(status=TaskStatus.AWAITING_APPROVAL)
    task.checkpoint_reached = "test-checkpoint"
    task.checkpoint_message = "Review required"

    task.approve_checkpoint("admin")

    assert task.status == TaskStatus.COMPLETED
    assert task.approved_at is not None
    assert task.approved_by == "admin"
    # Checkpoint info preserved for audit trail
    assert task.checkpoint_reached == "test-checkpoint"


# -- Executor tests --


def test_executor_pauses_at_checkpoint(tmp_path):
    """Executor pauses task at checkpoint before routing to next step."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)
    task = create_test_task()
    response = Mock()

    result = executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="engineer",
        routing_signal=None,
        context=None,
    )

    assert result is False
    assert task.status == TaskStatus.AWAITING_APPROVAL
    assert task.checkpoint_reached == "test_workflow-engineer"
    assert "Review implementation before QA" in task.checkpoint_message

    # Checkpoint file saved atomically
    checkpoint_dir = queue_dir / "checkpoints"
    assert checkpoint_dir.exists()
    checkpoint_file = checkpoint_dir / f"{task.id}.json"
    assert checkpoint_file.exists()

    # Verify saved JSON is valid
    saved = json.loads(checkpoint_file.read_text())
    assert saved["status"] == "awaiting_approval"


def test_executor_continues_after_approval(tmp_path):
    """Executor routes to next step after checkpoint approval."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)
    mock_queue.completed_dir = tmp_path / ".agent-communication" / "completed"
    mock_queue.completed_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)

    # Pre-approved task — checkpoint_reached must match current checkpoint
    task = create_test_task()
    task.checkpoint_reached = "test_workflow-engineer"
    task.approved_at = datetime.now(UTC)
    task.approved_by = "user"
    response = Mock()

    result = executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="engineer",
        routing_signal=None,
        context=None,
    )

    assert result is True
    mock_queue.push.assert_called_once()
    queued_task = mock_queue.push.call_args[0][0]
    assert queued_task.assigned_to == "qa"


def test_executor_blocks_stale_approval_from_different_checkpoint(tmp_path):
    """Approval for a different checkpoint doesn't bypass the current one."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)

    task = create_test_task()
    # Approval was for a different checkpoint
    task.checkpoint_reached = "other_workflow-other_step"
    task.approved_at = datetime.now(UTC)
    task.approved_by = "user"
    response = Mock()

    result = executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="engineer",
        routing_signal=None,
        context=None,
    )

    # Should pause, not route — stale approval doesn't count
    assert result is False
    assert task.status == TaskStatus.AWAITING_APPROVAL
    assert task.checkpoint_reached == "test_workflow-engineer"


def test_pr_at_terminal_step_terminates_workflow(tmp_path):
    """PR at a terminal step (no outgoing edges) terminates the workflow."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)

    task = create_test_task()
    task.context["pr_url"] = "https://github.com/org/repo/pull/42"
    response = Mock()

    # QA is the terminal step (no outgoing edges) — PR should terminate here
    result = executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="qa",
        routing_signal=None,
        context=None,
    )

    assert result is False
    assert task.context["pr_number"] == 42


def test_pr_at_non_terminal_step_continues_chain(tmp_path):
    """PR at a non-terminal step (engineer) continues to next agent (QA)."""
    # Use a simple workflow without checkpoints to isolate the PR behavior
    workflow = WorkflowDAG(
        name="simple",
        description="engineer → qa (no checkpoints)",
        steps={
            "engineer": WorkflowStep(
                id="engineer",
                agent="engineer",
                next=[WorkflowEdge(target="qa", condition=EdgeCondition(EdgeConditionType.ALWAYS))],
            ),
            "qa": WorkflowStep(id="qa", agent="qa", next=[]),
        },
        start_step="engineer",
    )

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)
    mock_queue.completed_dir = tmp_path / ".agent-communication" / "completed"
    mock_queue.completed_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)

    task = create_test_task()
    task.context["pr_url"] = "https://github.com/org/repo/pull/42"
    response = Mock()

    result = executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="engineer",
        routing_signal=None,
        context=None,
    )

    # Engineer is non-terminal — chain should continue to QA
    assert result is True
    mock_queue.push.assert_called_once()


# -- DAG model tests --


def test_checkpoint_config_fields():
    """CheckpointConfig holds message (required) and reason (optional)."""
    cp = CheckpointConfig(message="Review before deploy", reason="Production risk")
    assert cp.message == "Review before deploy"
    assert cp.reason == "Production risk"

    cp_no_reason = CheckpointConfig(message="Quick check")
    assert cp_no_reason.reason is None


def test_workflow_step_checkpoint_config():
    """Workflow step accepts CheckpointConfig."""
    step = WorkflowStep(
        id="deploy",
        agent="engineer",
        checkpoint=CheckpointConfig(
            message="Verify production deployment checklist",
            reason="Production deployment requires manual approval",
        ),
        next=[],
    )

    assert step.checkpoint is not None
    assert step.checkpoint.message == "Verify production deployment checklist"
    assert step.checkpoint.reason == "Production deployment requires manual approval"


def test_workflow_step_no_checkpoint():
    """Workflow step without checkpoint defaults to None."""
    step = WorkflowStep(
        id="test",
        agent="qa",
        next=[],
    )

    assert step.checkpoint is None


# -- CLI approve command tests --


def _write_checkpoint_file(checkpoint_dir, task):
    """Write a task as a checkpoint JSON file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / f"{task.id}.json"
    filepath.write_text(task.model_dump_json(indent=2))
    return filepath


def test_cli_approve_lists_checkpoints(tmp_path):
    """'agent approve' with no args lists pending checkpoints."""
    from agent_framework.cli.main import cli

    task = create_test_task(status=TaskStatus.AWAITING_APPROVAL)
    task.checkpoint_reached = "wf-engineer"
    task.checkpoint_message = "Review implementation"
    checkpoint_dir = tmp_path / ".agent-communication" / "queues" / "checkpoints"
    _write_checkpoint_file(checkpoint_dir, task)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "approve"])

    assert result.exit_code == 0
    assert "test-task" in result.output
    assert "Review implementation" in result.output


def test_cli_approve_no_checkpoints(tmp_path):
    """'agent approve' shows green message when nothing is pending."""
    from agent_framework.cli.main import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "approve"])

    assert result.exit_code == 0
    assert "No tasks awaiting approval" in result.output


def test_cli_approve_nonexistent_task(tmp_path):
    """'agent approve <bad-id>' shows error."""
    from agent_framework.cli.main import cli

    checkpoint_dir = tmp_path / ".agent-communication" / "queues" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "approve", "no-such-task"])

    assert result.exit_code == 0
    assert "No task found" in result.output


def test_cli_approve_specific_task(tmp_path):
    """'agent approve <id>' with confirmation re-queues and removes checkpoint file."""
    from agent_framework.cli.main import cli

    task = create_test_task(task_id="cp-task-1", status=TaskStatus.AWAITING_APPROVAL)
    task.checkpoint_reached = "wf-engineer"
    task.checkpoint_message = "Review required"
    checkpoint_dir = tmp_path / ".agent-communication" / "queues" / "checkpoints"
    cp_file = _write_checkpoint_file(checkpoint_dir, task)

    # Also need the engineer queue dir for FileQueue.push
    engineer_dir = tmp_path / ".agent-communication" / "queues" / "engineer"
    engineer_dir.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--workspace", str(tmp_path), "approve", "cp-task-1"],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Checkpoint approved" in result.output
    # Checkpoint file removed after successful re-queue
    assert not cp_file.exists()


def test_cli_approve_cancelled(tmp_path):
    """Declining confirmation preserves checkpoint file."""
    from agent_framework.cli.main import cli

    task = create_test_task(task_id="cp-task-2", status=TaskStatus.AWAITING_APPROVAL)
    task.checkpoint_reached = "wf-engineer"
    task.checkpoint_message = "Review required"
    checkpoint_dir = tmp_path / ".agent-communication" / "queues" / "checkpoints"
    cp_file = _write_checkpoint_file(checkpoint_dir, task)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--workspace", str(tmp_path), "approve", "cp-task-2"],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "cancelled" in result.output
    assert cp_file.exists()


# -- resume_after_checkpoint tests --


def test_resume_after_checkpoint_routes_to_next_step(tmp_path):
    """resume_after_checkpoint creates a chain task for the next agent."""
    # Build a minimal config file with a workflow that has plan → implement
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_content = {
        "workflows": {
            "test_wf": {
                "description": "test workflow",
                "steps": {
                    "plan": {
                        "agent": "architect",
                        "checkpoint": {"message": "Review plan"},
                        "next": [
                            {"target": "implement", "condition": "always"}
                        ],
                    },
                    "implement": {
                        "agent": "engineer",
                        "next": [],
                    },
                },
                "start_step": "plan",
            }
        },
        "repositories": [],
    }
    import yaml

    (config_dir / "agent-framework.yaml").write_text(yaml.dump(config_content))

    # Set up queue dirs
    queue_dir = tmp_path / ".agent-communication" / "queues"
    (queue_dir / "engineer").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "completed").mkdir(parents=True)

    from agent_framework.queue.file_queue import FileQueue

    queue = FileQueue(tmp_path)

    # Task at the plan checkpoint, already approved
    task = Task(
        id="task-1",
        type=TaskType.PLANNING,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="cli",
        assigned_to="architect",
        created_at=datetime.now(UTC),
        title="Plan something",
        description="Planning task",
        context={
            "workflow": "test_wf",
            "workflow_step": "plan",
        },
    )
    task.approved_at = datetime.now(UTC)
    task.approved_by = "user"
    task.checkpoint_reached = "test_wf-plan"

    result = resume_after_checkpoint(task, queue, tmp_path)

    assert result is True
    # Verify a chain task was queued for engineer
    engineer_files = list((queue_dir / "engineer").glob("*.json"))
    assert len(engineer_files) == 1
    chain_data = json.loads(engineer_files[0].read_text())
    assert chain_data["assigned_to"] == "engineer"
    assert "chain" in chain_data["id"]


def test_resume_after_checkpoint_no_workflow_returns_false(tmp_path):
    """Task without workflow context returns False."""
    mock_queue = Mock()

    task = create_test_task()
    task.context = {}  # No workflow key

    result = resume_after_checkpoint(task, mock_queue, tmp_path)

    assert result is False
    mock_queue.push.assert_not_called()


def test_resume_after_checkpoint_missing_config_returns_false(tmp_path):
    """Missing config file returns False gracefully."""
    mock_queue = Mock()

    task = create_test_task()
    task.context = {"workflow": "nonexistent"}

    result = resume_after_checkpoint(task, mock_queue, tmp_path)

    assert result is False
    mock_queue.push.assert_not_called()


def test_checkpoint_saves_workflow_step(tmp_path):
    """Checkpoint save stamps workflow_step in task context."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)

    executor = WorkflowExecutor(mock_queue, queue_dir)
    task = create_test_task()
    # Task starts without workflow_step (like initial planning tasks)
    assert "workflow_step" not in task.context

    executor.execute_step(
        workflow=workflow,
        task=task,
        response=Mock(),
        current_agent_id="engineer",
    )

    assert task.context["workflow_step"] == "engineer"
    # Also check the saved checkpoint file has it
    checkpoint_file = queue_dir / "checkpoints" / f"{task.id}.json"
    saved = json.loads(checkpoint_file.read_text())
    assert saved["context"]["workflow_step"] == "engineer"
