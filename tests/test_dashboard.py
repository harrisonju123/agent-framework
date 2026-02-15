"""Test dashboard functionality."""

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from agent_framework.core.activity import (
    ActivityManager,
    AgentActivity,
    AgentStatus,
    CurrentTask,
    ActivityEvent,
    TaskPhase,
)
from agent_framework.cli.dashboard import AgentDashboard
from agent_framework.core.config import RepositoryConfig


def test_activity_manager_basic_operations():
    """Test basic activity manager operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = ActivityManager(workspace)

        # Test update activity
        activity = AgentActivity(
            agent_id="test-agent",
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        )
        manager.update_activity(activity)

        # Test get activity
        retrieved = manager.get_activity("test-agent")
        assert retrieved is not None
        assert retrieved.agent_id == "test-agent"
        assert retrieved.status == AgentStatus.IDLE

        # Test get all activities
        activities = manager.get_all_activities()
        assert len(activities) == 1
        assert activities[0].agent_id == "test-agent"


def test_activity_manager_working_state():
    """Test activity manager with working agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = ActivityManager(workspace)

        # Create working activity
        activity = AgentActivity(
            agent_id="engineer",
            status=AgentStatus.WORKING,
            current_task=CurrentTask(
                id="task-123",
                title="Test task",
                type="implementation",
                started_at=datetime.now(timezone.utc)
            ),
            current_phase=TaskPhase.EXECUTING_LLM,
            last_updated=datetime.now(timezone.utc)
        )
        manager.update_activity(activity)

        # Retrieve and verify
        retrieved = manager.get_activity("engineer")
        assert retrieved.status == AgentStatus.WORKING
        assert retrieved.current_task.id == "task-123"
        assert retrieved.current_phase == TaskPhase.EXECUTING_LLM

        # Test elapsed time calculation
        elapsed = retrieved.get_elapsed_seconds()
        assert elapsed is not None
        assert elapsed >= 0


def test_activity_event_stream():
    """Test activity event stream."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = ActivityManager(workspace)

        # Append events
        events = [
            ActivityEvent(
                type="start",
                agent="engineer",
                task_id="task-1",
                title="Task 1",
                timestamp=datetime.now(timezone.utc)
            ),
            ActivityEvent(
                type="complete",
                agent="engineer",
                task_id="task-1",
                title="Task 1",
                timestamp=datetime.now(timezone.utc),
                duration_ms=5000
            ),
            ActivityEvent(
                type="fail",
                agent="engineer",
                task_id="task-2",
                title="Task 2",
                timestamp=datetime.now(timezone.utc),
                retry_count=2
            ),
        ]

        for event in events:
            manager.append_event(event)

        # Get recent events
        recent = manager.get_recent_events(limit=10)
        assert len(recent) == 3

        # Events should be in reverse order (most recent first)
        assert recent[0].type == "fail"
        assert recent[1].type == "complete"
        assert recent[2].type == "start"


def test_activity_stream_rotation():
    """Test that activity stream rotates after max events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = ActivityManager(workspace)
        manager.max_stream_events = 5  # Set low limit for testing

        # Append more events than max
        for i in range(10):
            event = ActivityEvent(
                type="start",
                agent="test",
                task_id=f"task-{i}",
                title=f"Task {i}",
                timestamp=datetime.now(timezone.utc)
            )
            manager.append_event(event)

        # With append-only optimization, file is trimmed when appends reach
        # max_stream_events, so file may have up to 2x max between trims.
        # After 10 events with max=5: trim at event 5 (→5), then 4 appends = 9.
        recent = manager.get_recent_events(limit=100)
        assert len(recent) <= manager.max_stream_events * 2
        # Most recent event should always be the last one appended
        assert recent[0].task_id == "task-9"


def test_dashboard_render():
    """Test dashboard can render without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create minimal config
        config_dir = workspace / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        agents_config = config_dir / "agents.yaml"
        agents_config.write_text("""
agents:
  - id: architect
    name: Technical Architect
    queue: architect
    enabled: true
    prompt: "Test prompt"

  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "Test prompt"
""")

        # Create activity manager and add some data
        activity_manager = ActivityManager(workspace)
        activity_manager.update_activity(AgentActivity(
            agent_id="architect",
            status=AgentStatus.IDLE,
            last_updated=datetime.now(timezone.utc)
        ))
        activity_manager.update_activity(AgentActivity(
            agent_id="engineer",
            status=AgentStatus.WORKING,
            current_task=CurrentTask(
                id="task-123",
                title="Test implementation",
                type="implementation",
                started_at=datetime.now(timezone.utc)
            ),
            current_phase=TaskPhase.EXECUTING_LLM,
            last_updated=datetime.now(timezone.utc)
        ))

        # Create dashboard and render
        dashboard = AgentDashboard(workspace)
        layout = dashboard.render()

        # Basic checks - should not raise exceptions
        assert layout is not None


def _make_dashboard_workspace(tmpdir):
    """Create a minimal workspace with agents config for dashboard tests."""
    workspace = Path(tmpdir)
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "agents.yaml").write_text("""
agents:
  - id: architect
    name: Technical Architect
    queue: architect
    enabled: true
    prompt: "Test prompt"
""")
    return workspace


def test_handle_key_n_returns_new_work():
    """Pressing 'n' returns NEW_WORK action."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)
        dashboard = AgentDashboard(workspace)
        assert dashboard._handle_key('n') == "NEW_WORK"


def test_prompt_new_work_with_default_repo(monkeypatch):
    """_prompt_new_work() queues task using default_repo, only prompts for goal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)
        repo = RepositoryConfig(github_repo="owner/myrepo", jira_project="PROJ")
        dashboard = AgentDashboard(workspace, default_repo=repo)

        monkeypatch.setattr("builtins.input", lambda prompt: "Implement feature X")

        result = dashboard._prompt_new_work()
        assert result == "Implement feature X"

        # Verify task was queued to architect
        stats = dashboard.queue.get_queue_stats("architect")
        assert stats["count"] == 1

        task = dashboard.queue.pop("architect")
        assert task.title == "Plan and delegate: Implement feature X"
        assert task.context["github_repo"] == "owner/myrepo"
        assert task.created_by == "dashboard"


def test_prompt_new_work_no_default_repo_prompts_selection(monkeypatch):
    """Without default_repo, prompts for both repo and goal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)

        # Write framework config with repos
        (workspace / "config" / "agent-framework.yaml").write_text("""
repositories:
  - github_repo: owner/repo-a
    jira_project: REPA
  - github_repo: owner/repo-b
""")
        dashboard = AgentDashboard(workspace)

        inputs = iter(["2", "Fix bug Y"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        result = dashboard._prompt_new_work()
        assert result == "Fix bug Y"

        task = dashboard.queue.pop("architect")
        assert task is not None
        assert task.context["github_repo"] == "owner/repo-b"


def test_prompt_new_work_empty_goal_returns_none(monkeypatch):
    """Empty goal input returns None without queuing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)
        repo = RepositoryConfig(github_repo="owner/myrepo")
        dashboard = AgentDashboard(workspace, default_repo=repo)

        monkeypatch.setattr("builtins.input", lambda prompt: "")

        result = dashboard._prompt_new_work()
        assert result is None
        assert dashboard.queue.get_queue_stats("architect")["count"] == 0


def test_prompt_new_work_keyboard_interrupt_returns_none(monkeypatch):
    """KeyboardInterrupt during input returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)
        repo = RepositoryConfig(github_repo="owner/myrepo")
        dashboard = AgentDashboard(workspace, default_repo=repo)

        def raise_interrupt(prompt):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_interrupt)

        result = dashboard._prompt_new_work()
        assert result is None


def test_help_overlay_includes_n_shortcut():
    """Help overlay lists the 'n' shortcut."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_dashboard_workspace(tmpdir)
        dashboard = AgentDashboard(workspace)
        panel = dashboard.render_help_overlay()
        # Panel.renderable is the Text object
        help_text = str(panel.renderable)
        assert "n" in help_text
        assert "Queue new work" in help_text


if __name__ == "__main__":
    # Run basic tests
    test_activity_manager_basic_operations()
    print("✓ Basic operations test passed")

    test_activity_manager_working_state()
    print("✓ Working state test passed")

    test_activity_event_stream()
    print("✓ Event stream test passed")

    test_activity_stream_rotation()
    print("✓ Stream rotation test passed")

    test_dashboard_render()
    print("✓ Dashboard render test passed")

    print("\nAll tests passed!")
