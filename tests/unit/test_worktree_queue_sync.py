"""Tests for syncing LLM-created queue tasks from worktrees to the main queue."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.queue.file_queue import FileQueue


def _make_task_json(task_id, assigned_to="engineer"):
    """Create a minimal task JSON dict."""
    return {
        "id": task_id,
        "type": "implementation",
        "status": "pending",
        "priority": 1,
        "created_by": "architect",
        "assigned_to": assigned_to,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": f"Task {task_id}",
        "description": "A test task",
    }


@pytest.fixture
def main_workspace(tmp_path):
    """Main repo workspace with a real FileQueue."""
    workspace = tmp_path / "main"
    workspace.mkdir()
    return workspace


@pytest.fixture
def queue(main_workspace):
    return FileQueue(workspace=main_workspace)


@pytest.fixture
def agent(queue, main_workspace):
    config = AgentConfig(
        id="architect",
        name="Architect",
        queue="architect",
        prompt="You are an architect.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a.workspace = main_workspace
    a._active_worktree = None
    a.logger = MagicMock()
    # Initialize GitOperationsManager
    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=a.config,
        workspace=a.workspace,
        queue=a.queue,
        logger=a.logger,
        session_logger=a._session_logger if hasattr(a, '_session_logger') else None,
    )
    return a


class TestSyncWorktreeQueuedTasks:
    def test_no_worktree_is_noop(self, agent):
        """No active worktree — nothing to sync."""
        agent._active_worktree = None
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()
        agent.logger.info.assert_not_called()

    def test_worktree_without_queue_dir_is_noop(self, agent, tmp_path):
        """Worktree exists but has no .agent-communication/queues/."""
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()
        agent.logger.info.assert_not_called()

    def test_syncs_single_task(self, agent, queue, tmp_path):
        """Task written to worktree queue gets synced to main queue."""
        worktree = tmp_path / "worktree"
        wt_queue = worktree / ".agent-communication" / "queues" / "engineer"
        wt_queue.mkdir(parents=True)

        task_data = _make_task_json("impl-abc123")
        (wt_queue / "impl-abc123.json").write_text(json.dumps(task_data))

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        # Task should now be in main queue
        main_task_file = queue.queue_dir / "engineer" / "impl-abc123.json"
        assert main_task_file.exists()

        # Worktree copy should be removed
        assert not (wt_queue / "impl-abc123.json").exists()

    def test_syncs_multiple_tasks_across_agents(self, agent, queue, tmp_path):
        """Tasks for different agents all get synced."""
        worktree = tmp_path / "worktree"

        for agent_id, task_id in [("engineer", "impl-001"), ("engineer", "impl-002"), ("qa", "review-001")]:
            wt_queue = worktree / ".agent-communication" / "queues" / agent_id
            wt_queue.mkdir(parents=True, exist_ok=True)
            task_data = _make_task_json(task_id, assigned_to=agent_id)
            (wt_queue / f"{task_id}.json").write_text(json.dumps(task_data))

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        assert (queue.queue_dir / "engineer" / "impl-001.json").exists()
        assert (queue.queue_dir / "engineer" / "impl-002.json").exists()
        assert (queue.queue_dir / "qa" / "review-001.json").exists()

    def test_skips_non_agent_directories(self, agent, queue, tmp_path):
        """Directories like checkpoints and completed are ignored."""
        worktree = tmp_path / "worktree"

        # Create a file in checkpoints (should be skipped)
        checkpoints = worktree / ".agent-communication" / "queues" / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "plan-123.json").write_text(json.dumps(_make_task_json("plan-123")))

        # Create a real task too
        eng_queue = worktree / ".agent-communication" / "queues" / "engineer"
        eng_queue.mkdir(parents=True)
        (eng_queue / "impl-001.json").write_text(json.dumps(_make_task_json("impl-001")))

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        # Only the engineer task should be synced
        assert (queue.queue_dir / "engineer" / "impl-001.json").exists()
        assert not (queue.queue_dir / "checkpoints").exists()
        # Checkpoint file should still be in worktree (untouched)
        assert (checkpoints / "plan-123.json").exists()

    def test_skips_when_worktree_is_main_workspace(self, agent, queue, main_workspace):
        """Don't sync from main workspace back into itself."""
        agent._active_worktree = main_workspace
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()
        agent.logger.info.assert_not_called()

    def test_malformed_json_logged_and_skipped(self, agent, queue, tmp_path):
        """Bad JSON in worktree doesn't crash, gets logged as warning."""
        worktree = tmp_path / "worktree"
        wt_queue = worktree / ".agent-communication" / "queues" / "engineer"
        wt_queue.mkdir(parents=True)

        (wt_queue / "bad-task.json").write_text("not valid json{{{")

        # Also add a valid task
        task_data = _make_task_json("good-task")
        (wt_queue / "good-task.json").write_text(json.dumps(task_data))

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        # Good task synced
        assert (queue.queue_dir / "engineer" / "good-task.json").exists()
        # Bad task logged as warning
        agent.logger.warning.assert_called_once()
        assert "bad-task.json" in agent.logger.warning.call_args[0][0]

    def test_synced_task_is_valid(self, agent, queue, tmp_path):
        """Synced task can be loaded back as a proper Task object."""
        worktree = tmp_path / "worktree"
        wt_queue = worktree / ".agent-communication" / "queues" / "engineer"
        wt_queue.mkdir(parents=True)

        task_data = _make_task_json("impl-xyz789")
        task_data["depends_on"] = ["impl-abc123"]
        (wt_queue / "impl-xyz789.json").write_text(json.dumps(task_data))

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        # Load from main queue file and verify fields survived round-trip
        task_file = queue.queue_dir / "engineer" / "impl-xyz789.json"
        assert task_file.exists()
        synced = FileQueue.load_task_file(task_file)
        assert synced.id == "impl-xyz789"
        assert synced.assigned_to == "engineer"
        assert synced.depends_on == ["impl-abc123"]

    def test_skips_already_completed_tasks(self, agent, queue, tmp_path):
        """Stale worktree copies of completed tasks are cleaned up, not re-pushed."""
        worktree = tmp_path / "worktree"
        wt_queue = worktree / ".agent-communication" / "queues" / "engineer"
        wt_queue.mkdir(parents=True)

        task_data = _make_task_json("impl-done")
        (wt_queue / "impl-done.json").write_text(json.dumps(task_data))

        # Mark the task as completed in the main queue
        completed_task = Task(**task_data)
        completed_task.status = TaskStatus.COMPLETED
        queue.mark_completed(completed_task)

        agent._active_worktree = worktree
        agent._git_ops._active_worktree = agent._active_worktree
        agent._git_ops.sync_worktree_queued_tasks()

        # Task should NOT appear in the main queue
        main_task_file = queue.queue_dir / "engineer" / "impl-done.json"
        assert not main_task_file.exists()

        # Worktree copy should still be cleaned up
        assert not (wt_queue / "impl-done.json").exists()
