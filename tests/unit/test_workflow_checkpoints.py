"""Tests for workflow checkpoint functionality."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from src.agent_framework.core.task import Task, TaskStatus, TaskType
from src.agent_framework.workflow.dag import (
    WorkflowDAG,
    WorkflowStep,
    WorkflowEdge,
    EdgeCondition,
    EdgeConditionType,
)
from src.agent_framework.workflow.executor import WorkflowExecutor


def create_test_task(task_id="test-task", status=TaskStatus.IN_PROGRESS):
    """Create a test task."""
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=status,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.utcnow(),
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
                checkpoint={
                    "message": "Review implementation before QA",
                    "reason": "High-risk deployment",
                },
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


def test_task_approve_checkpoint():
    """Approve a checkpoint and resume workflow."""
    task = create_test_task(status=TaskStatus.AWAITING_APPROVAL)
    task.checkpoint_reached = "test-checkpoint"
    task.checkpoint_message = "Review required"

    task.approve_checkpoint("admin")

    assert task.status == TaskStatus.IN_PROGRESS
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

    executor = WorkflowExecutor(mock_queue, queue_dir)

    # Pre-approved task
    task = create_test_task()
    task.approved_at = datetime.utcnow()
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


def test_executor_pr_takes_precedence_over_checkpoint(tmp_path):
    """PR creation terminates workflow even if checkpoint is configured."""
    workflow = _build_checkpoint_workflow()

    mock_queue = Mock()
    queue_dir = tmp_path / ".agent-communication" / "queues"
    queue_dir.mkdir(parents=True)

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

    # PR creation should terminate workflow, not pause at checkpoint
    assert result is False
    assert task.status != TaskStatus.AWAITING_APPROVAL
    assert task.context["pr_number"] == 42


# -- DAG model tests --


def test_workflow_step_checkpoint_config():
    """Workflow step accepts checkpoint configuration."""
    step = WorkflowStep(
        id="deploy",
        agent="engineer",
        checkpoint={
            "message": "Verify production deployment checklist",
            "reason": "Production deployment requires manual approval",
        },
        next=[],
    )

    assert step.checkpoint is not None
    assert step.checkpoint["message"] == "Verify production deployment checklist"
    assert step.checkpoint["reason"] == "Production deployment requires manual approval"


def test_workflow_step_no_checkpoint():
    """Workflow step without checkpoint defaults to None."""
    step = WorkflowStep(
        id="test",
        agent="qa",
        next=[],
    )

    assert step.checkpoint is None
