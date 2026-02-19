"""Tests for activity stream event emission during task execution."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.activity import ActivityEvent, ActivityManager
from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(**overrides):
    defaults = dict(
        id="test-task-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Add feature X to the system",
        context={},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_agent():
    """Minimal mock agent with _initialize_task_execution bound for isolation."""
    agent = MagicMock()
    agent._initialize_task_execution = Agent._initialize_task_execution.__get__(agent)

    emitted = []
    agent.activity_manager.append_event.side_effect = emitted.append
    agent.activity_manager.update_activity = MagicMock()
    agent.config.id = "engineer-1"
    agent._agent_definition = None  # Skip JIRA transition

    agent._emitted = emitted
    return agent


class TestInitializeTaskExecutionRetryEvent:
    """_initialize_task_execution emits "retry" on self-eval retries, not "start"."""

    def test_fresh_task_emits_start_event(self):
        """A task with no self-eval history emits a "start" event."""
        task = _make_task(context={})
        agent = _make_agent()

        agent._initialize_task_execution(task, datetime.now(timezone.utc))

        assert len(agent._emitted) == 1
        event = agent._emitted[0]
        assert event.type == "start"
        assert event.retry_count is None

    def test_self_eval_retry_emits_retry_event(self):
        """A task that failed self-eval once emits a "retry" event, not "start"."""
        task = _make_task(context={"_self_eval_count": 1})
        agent = _make_agent()

        agent._initialize_task_execution(task, datetime.now(timezone.utc))

        assert len(agent._emitted) == 1
        event = agent._emitted[0]
        assert event.type == "retry"
        assert event.retry_count == 1

    def test_retry_count_reflects_eval_count(self):
        """retry_count in the event matches the actual _self_eval_count."""
        task = _make_task(context={"_self_eval_count": 3})
        agent = _make_agent()

        agent._initialize_task_execution(task, datetime.now(timezone.utc))

        event = agent._emitted[0]
        assert event.type == "retry"
        assert event.retry_count == 3


class TestActivityEventTitleTruncation:
    """ActivityEvent.title is capped at 80 chars to keep the stream file compact."""

    def _make_event(self, title: str) -> ActivityEvent:
        return ActivityEvent(
            type="start",
            agent="engineer-1",
            task_id="task-001",
            title=title,
            timestamp=datetime.now(timezone.utc),
        )

    def test_long_title_truncated(self):
        long_title = "A" * 120
        event = self._make_event(long_title)
        assert len(event.title) == 83  # 80 + "..."
        assert event.title.endswith("...")

    def test_short_title_unchanged(self):
        short_title = "Fix login bug"
        event = self._make_event(short_title)
        assert event.title == short_title

    def test_exact_boundary_unchanged(self):
        exact = "B" * 80
        event = self._make_event(exact)
        assert event.title == exact

    def test_round_trip_through_stream(self, tmp_path):
        """Write event with long title to stream, read back, title is truncated."""
        manager = ActivityManager(tmp_path)
        long_title = "Plan and delegate: " + "x" * 200
        event = self._make_event(long_title)
        manager.append_event(event)

        events = manager.get_recent_events(limit=1)
        assert len(events) == 1
        assert len(events[0].title) == 83
        assert events[0].title.endswith("...")
