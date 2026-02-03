"""Test dashboard functionality."""

import json
from datetime import datetime
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


def test_activity_manager_basic_operations():
    """Test basic activity manager operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        manager = ActivityManager(workspace)

        # Test update activity
        activity = AgentActivity(
            agent_id="test-agent",
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
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
                started_at=datetime.utcnow()
            ),
            current_phase=TaskPhase.EXECUTING_LLM,
            last_updated=datetime.utcnow()
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
                timestamp=datetime.utcnow()
            ),
            ActivityEvent(
                type="complete",
                agent="engineer",
                task_id="task-1",
                title="Task 1",
                timestamp=datetime.utcnow(),
                duration_ms=5000
            ),
            ActivityEvent(
                type="fail",
                agent="engineer",
                task_id="task-2",
                title="Task 2",
                timestamp=datetime.utcnow(),
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
                timestamp=datetime.utcnow()
            )
            manager.append_event(event)

        # Should only keep last 5
        recent = manager.get_recent_events(limit=100)
        assert len(recent) <= 5


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
  - id: product-owner
    name: Product Owner
    queue: product-owner
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
            agent_id="product-owner",
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
        ))
        activity_manager.update_activity(AgentActivity(
            agent_id="engineer",
            status=AgentStatus.WORKING,
            current_task=CurrentTask(
                id="task-123",
                title="Test implementation",
                type="implementation",
                started_at=datetime.utcnow()
            ),
            current_phase=TaskPhase.EXECUTING_LLM,
            last_updated=datetime.utcnow()
        ))

        # Create dashboard and render
        dashboard = AgentDashboard(workspace)
        layout = dashboard.render()

        # Basic checks - should not raise exceptions
        assert layout is not None


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
