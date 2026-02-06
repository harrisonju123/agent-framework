"""Tests for TeamBridge - autonomous/interactive pipeline bridge."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path

from agent_framework.core.team_bridge import TeamBridge
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def workspace(tmp_path):
    """Create a minimal workspace directory."""
    (tmp_path / "config" / "docs").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "queues" / "engineer").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def failed_task():
    return Task(
        id="task-impl-1234",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.FAILED,
        priority=2,
        created_by="cli",
        assigned_to="engineer",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Implement user auth",
        description="Add JWT-based authentication to the API.",
        retry_count=3,
        last_error="ConnectionError: JIRA server unreachable at https://jira.example.com",
        context={"jira_key": "PROJ-42", "github_repo": "myorg/myapp"},
        acceptance_criteria=["JWT tokens issued on login", "Tokens expire after 1h"],
    )


def test_build_escalation_context(workspace, failed_task):
    """Escalation context includes task details and error."""
    bridge = TeamBridge(workspace)
    context = bridge.build_escalation_context(failed_task)

    assert "Implement user auth" in context
    assert "task-impl-1234" in context
    assert "PROJ-42" in context
    assert "myorg/myapp" in context
    assert "ConnectionError" in context
    assert "JWT tokens issued on login" in context
    assert "Retry count:** 3" in context


def test_build_escalation_context_no_error(workspace):
    """Handles task with no error message."""
    bridge = TeamBridge(workspace)
    task = Task(
        id="task-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.FAILED,
        priority=2,
        created_by="cli",
        assigned_to="engineer",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Test task",
        description="Test description",
    )
    context = bridge.build_escalation_context(task)
    assert "No error message available" in context


def test_record_team_session(workspace):
    """Session metadata is written to teams directory."""
    bridge = TeamBridge(workspace)
    session_file = bridge.record_team_session(
        team_name="debug-12345",
        template="debug",
        source_task_id="task-impl-1234",
    )

    assert session_file.exists()
    data = json.loads(session_file.read_text())
    assert data["team_name"] == "debug-12345"
    assert data["template"] == "debug"
    assert data["source_task_id"] == "task-impl-1234"
    assert data["status"] == "active"
    assert "started_at" in data


def test_record_team_session_with_metadata(workspace):
    """Extra metadata is merged into session file."""
    bridge = TeamBridge(workspace)
    bridge.record_team_session(
        team_name="impl-99999",
        template="full",
        metadata={"repo": "myorg/myapp"},
    )

    session_file = workspace / ".agent-communication" / "teams" / "impl-99999.json"
    data = json.loads(session_file.read_text())
    assert data["repo"] == "myorg/myapp"


def test_get_active_teams(workspace):
    """Active teams are read from the teams directory."""
    bridge = TeamBridge(workspace)

    # Create a session
    bridge.record_team_session(team_name="test-team", template="full")

    teams = bridge.get_active_teams()
    assert len(teams) >= 1
    assert any(t["team_name"] == "test-team" for t in teams)


def test_get_active_teams_empty(workspace):
    """Returns empty list when no sessions exist."""
    bridge = TeamBridge(workspace)
    teams = bridge.get_active_teams()
    assert teams == []


def test_handoff_to_autonomous(workspace):
    """Tasks are pushed to the file queue."""
    bridge = TeamBridge(workspace)

    task_data = [
        {"title": "Fix auth bug", "description": "The JWT validation is broken"},
        {"title": "Add tests", "description": "Cover the auth flow"},
    ]

    queued_ids = bridge.handoff_to_autonomous(task_data, workflow="simple")

    assert len(queued_ids) == 2
    assert all(qid.startswith("team-handoff-") for qid in queued_ids)

    # Verify tasks were written to the queue
    queue_dir = workspace / ".agent-communication" / "queues" / "engineer"
    task_files = list(queue_dir.glob("*.json"))
    assert len(task_files) == 2


def test_build_team_claude_md(workspace):
    """CLAUDE.md content includes pipeline docs when available."""
    bridge = TeamBridge(workspace)

    # Write team context doc
    (workspace / "config" / "docs" / "team_context.md").write_text(
        "Use queue_task_for_agent to hand off."
    )

    md = bridge.build_team_claude_md(repo_path="/tmp/myrepo")
    assert "queue_task_for_agent" in md
    assert "/tmp/myrepo" in md
    assert "Agent Team Session" in md


def test_build_team_claude_md_no_docs(workspace):
    """CLAUDE.md still works when pipeline docs are missing."""
    bridge = TeamBridge(workspace)
    md = bridge.build_team_claude_md()
    assert "Agent Team Session" in md
    assert "queue_task_for_agent" in md  # fallback text mentions it


def test_mark_session_ended(workspace):
    """Session status is updated to 'ended' with a timestamp."""
    bridge = TeamBridge(workspace)
    bridge.record_team_session(team_name="test-end", template="debug")

    result = bridge.mark_session_ended("test-end")
    assert result is True

    session_file = workspace / ".agent-communication" / "teams" / "test-end.json"
    data = json.loads(session_file.read_text())
    assert data["status"] == "ended"
    assert "ended_at" in data


def test_mark_session_ended_missing(workspace):
    """Returns False when session file doesn't exist."""
    bridge = TeamBridge(workspace)
    result = bridge.mark_session_ended("nonexistent-team")
    assert result is False
