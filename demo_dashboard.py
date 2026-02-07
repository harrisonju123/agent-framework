#!/usr/bin/env python
"""Demo script to show dashboard functionality."""

import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import time

from agent_framework.core.activity import (
    ActivityManager,
    AgentActivity,
    AgentStatus,
    CurrentTask,
    ActivityEvent,
    TaskPhase,
)
from agent_framework.cli.dashboard import AgentDashboard


async def simulate_agent_activity(workspace: Path):
    """Simulate agent activity for demo purposes."""
    manager = ActivityManager(workspace)

    # Simulate agents starting
    agents = ["architect", "architect", "engineer", "qa"]
    for agent in agents:
        manager.update_activity(AgentActivity(
            agent_id=agent,
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
        ))

    # Simulate Product Owner starting work
    await asyncio.sleep(2)
    manager.update_activity(AgentActivity(
        agent_id="architect",
        status=AgentStatus.WORKING,
        current_task=CurrentTask(
            id="planning-PTO-1738573234",
            title="Create epic and breakdown for: Add MFA to login flow",
            type="planning",
            started_at=datetime.utcnow()
        ),
        current_phase=TaskPhase.EXPLORING_CODEBASE,
        last_updated=datetime.utcnow()
    ))

    manager.append_event(ActivityEvent(
        type="start",
        agent="architect",
        task_id="planning-PTO-1738573234",
        title="Create epic and breakdown for: Add MFA to login flow",
        timestamp=datetime.utcnow()
    ))

    # Progress through phases
    await asyncio.sleep(3)
    activity = manager.get_activity("architect")
    activity.current_phase = TaskPhase.CREATING_EPIC
    manager.update_activity(activity)

    await asyncio.sleep(3)
    activity = manager.get_activity("architect")
    activity.current_phase = TaskPhase.CREATING_SUBTASKS
    manager.update_activity(activity)

    # Engineer starts working
    await asyncio.sleep(2)
    manager.update_activity(AgentActivity(
        agent_id="engineer",
        status=AgentStatus.WORKING,
        current_task=CurrentTask(
            id="impl-PTO-1234-auth",
            title="Implement MFA backend",
            type="implementation",
            started_at=datetime.utcnow()
        ),
        current_phase=TaskPhase.EXECUTING_LLM,
        last_updated=datetime.utcnow()
    ))

    manager.append_event(ActivityEvent(
        type="start",
        agent="engineer",
        task_id="impl-PTO-1234-auth",
        title="Implement MFA backend",
        timestamp=datetime.utcnow()
    ))

    # Complete Product Owner task
    await asyncio.sleep(3)
    manager.update_activity(AgentActivity(
        agent_id="architect",
        status=AgentStatus.IDLE,
        last_updated=datetime.utcnow()
    ))

    manager.append_event(ActivityEvent(
        type="complete",
        agent="architect",
        task_id="planning-PTO-1738573234",
        title="Create epic and breakdown for: Add MFA to login flow",
        timestamp=datetime.utcnow(),
        duration_ms=8000
    ))

    # Simulate Engineer phases
    await asyncio.sleep(2)
    activity = manager.get_activity("engineer")
    activity.current_phase = TaskPhase.IMPLEMENTING
    manager.update_activity(activity)

    await asyncio.sleep(3)
    activity = manager.get_activity("engineer")
    activity.current_phase = TaskPhase.TESTING
    manager.update_activity(activity)

    # Add a failure event
    await asyncio.sleep(2)
    manager.append_event(ActivityEvent(
        type="fail",
        agent="engineer",
        task_id="impl-PTO-1235-bug",
        title="Fix auth bug",
        timestamp=datetime.utcnow(),
        retry_count=2
    ))

    # Engineer completes
    await asyncio.sleep(3)
    manager.update_activity(AgentActivity(
        agent_id="engineer",
        status=AgentStatus.IDLE,
        last_updated=datetime.utcnow()
    ))

    manager.append_event(ActivityEvent(
        type="complete",
        agent="engineer",
        task_id="impl-PTO-1234-auth",
        title="Implement MFA backend",
        timestamp=datetime.utcnow(),
        duration_ms=12000
    ))


async def main():
    """Main demo function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create config
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

  - id: architect
    name: Architect
    queue: architect
    enabled: true
    prompt: "Test prompt"

  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "Test prompt"

  - id: qa
    name: QA
    queue: qa
    enabled: true
    prompt: "Test prompt"
""")

        print("🎬 Starting dashboard demo...")
        print("   Press Ctrl+C to exit\n")

        # Start background task to simulate activity
        activity_task = asyncio.create_task(simulate_agent_activity(workspace))

        # Run dashboard
        dashboard = AgentDashboard(workspace)
        try:
            await dashboard.run(refresh_interval=1.0)
        except KeyboardInterrupt:
            print("\n\n✓ Demo complete!")

        # Cancel activity simulation
        activity_task.cancel()
        try:
            await activity_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
