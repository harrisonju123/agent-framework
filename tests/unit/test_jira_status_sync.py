"""Tests for framework-level JIRA status sync in Agent."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.config import AgentDefinition
from agent_framework.core.task import Task, TaskStatus, TaskType


@dataclass
class FakeAgentConfig:
    id: str = "engineer"
    name: str = "Engineer"
    queue: str = "engineer"
    prompt: str = ""
    poll_interval: int = 30
    max_retries: int = 3
    timeout: int = 1800
    enable_sandbox: bool = False
    sandbox_image: str = ""
    sandbox_test_cmd: str = ""
    max_test_retries: int = 2
    validate_tasks: bool = False
    validation_mode: str = "warn"

    @property
    def base_id(self) -> str:
        parts = self.id.rsplit("-", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else self.id


def _make_task(jira_key="PROJ-42", **extra_context):
    ctx = {}
    if jira_key:
        ctx["jira_key"] = jira_key
    ctx.update(extra_context)
    return Task(
        id="test-task-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.utcnow(),
        title="Test task",
        description="desc",
        context=ctx,
    )


def _make_agent_definition(**overrides):
    defaults = dict(
        id="engineer",
        name="Software Engineer",
        queue="engineer",
        prompt="",
        jira_can_update_status=True,
        jira_allowed_transitions=["In Progress", "Code Review"],
        jira_on_start="In Progress",
        jira_on_complete="Code Review",
    )
    defaults.update(overrides)
    return AgentDefinition(**defaults)


def _make_agent(agent_definition=None, jira_client=None):
    """Build a minimal Agent with mocked internals for _sync_jira_status testing."""
    from agent_framework.core.agent import Agent

    config = FakeAgentConfig()
    queue = MagicMock()
    llm = MagicMock()
    activity_manager = MagicMock()

    agent = Agent.__new__(Agent)
    agent.config = config
    agent.queue = queue
    agent.llm = llm
    agent.jira_client = jira_client
    agent.jira_config = None
    agent.github_client = None
    agent.workspace = MagicMock()
    agent._agent_definition = agent_definition
    agent._mcp_enabled = True
    agent.activity_manager = activity_manager
    agent.logger = logging.getLogger("test.agent")
    return agent


# --- Guard condition tests ---


class TestSyncJiraStatusGuards:

    def test_skips_when_no_jira_key(self):
        task = _make_task(jira_key=None)
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        agent._sync_jira_status(task, "In Progress")
        jira.transition_ticket.assert_not_called()

    def test_skips_when_no_jira_client(self):
        task = _make_task()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=None)

        agent._sync_jira_status(task, "In Progress")

    def test_skips_when_no_agent_definition(self):
        task = _make_task()
        jira = MagicMock()
        agent = _make_agent(agent_definition=None, jira_client=jira)

        agent._sync_jira_status(task, "In Progress")
        jira.transition_ticket.assert_not_called()

    def test_skips_when_update_status_not_allowed(self):
        task = _make_task()
        jira = MagicMock()
        defn = _make_agent_definition(jira_can_update_status=False)
        agent = _make_agent(agent_definition=defn, jira_client=jira)

        agent._sync_jira_status(task, "In Progress")
        jira.transition_ticket.assert_not_called()

    def test_skips_when_transition_not_in_allowed_list(self, caplog):
        task = _make_task()
        jira = MagicMock()
        defn = _make_agent_definition(jira_allowed_transitions=["In Progress"])
        agent = _make_agent(agent_definition=defn, jira_client=jira)

        with caplog.at_level(logging.WARNING):
            agent._sync_jira_status(task, "Done")

        jira.transition_ticket.assert_not_called()
        assert "not in allowed transitions" in caplog.text


# --- Happy path tests ---


class TestSyncJiraStatusHappyPath:

    def test_transitions_ticket(self):
        task = _make_task()
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        agent._sync_jira_status(task, "In Progress")

        jira.transition_ticket.assert_called_once_with("PROJ-42", "In Progress")

    def test_adds_comment_when_provided(self):
        task = _make_task()
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        agent._sync_jira_status(task, "Code Review", comment="PR ready")

        jira.transition_ticket.assert_called_once_with("PROJ-42", "Code Review")
        jira.add_comment.assert_called_once_with("PROJ-42", "PR ready")

    def test_no_comment_when_not_provided(self):
        task = _make_task()
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        agent._sync_jira_status(task, "In Progress")

        jira.add_comment.assert_not_called()


# --- Failure resilience tests ---


class TestSyncJiraStatusResilience:

    def test_swallows_transition_error(self, caplog):
        task = _make_task()
        jira = MagicMock()
        jira.transition_ticket.side_effect = Exception("JIRA API down")
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        with caplog.at_level(logging.WARNING):
            agent._sync_jira_status(task, "In Progress")

        assert "Failed to transition JIRA" in caplog.text

    def test_never_raises(self):
        task = _make_task()
        jira = MagicMock()
        jira.transition_ticket.side_effect = RuntimeError("boom")
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)

        # Should not raise
        agent._sync_jira_status(task, "In Progress")


# --- Lifecycle integration tests ---


class TestLifecycleIntegration:

    def test_on_start_called_in_initialize(self):
        """_initialize_task_execution should call _sync_jira_status with jira_on_start."""
        task = _make_task()
        defn = _make_agent_definition()
        agent = _make_agent(agent_definition=defn, jira_client=MagicMock())

        with patch.object(agent, '_sync_jira_status') as mock_sync:
            agent._initialize_task_execution(task, "2024-01-01T00:00:00")

        mock_sync.assert_called_once_with(task, "In Progress")

    def test_on_start_skipped_when_no_config(self):
        """No jira_on_start → no _sync_jira_status call from init."""
        task = _make_task()
        defn = _make_agent_definition(jira_on_start=None)
        agent = _make_agent(agent_definition=defn, jira_client=MagicMock())

        with patch.object(agent, '_sync_jira_status') as mock_sync:
            agent._initialize_task_execution(task, "2024-01-01T00:00:00")

        mock_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_adds_jira_comment(self):
        """_handle_failure should add a JIRA comment on permanent failure."""
        task = _make_task()
        task.retry_count = 5
        task.last_error = "Tests failed"
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)
        agent.config.max_retries = 3
        agent.retry_handler = MagicMock()
        agent.retry_handler.max_retries = 3
        agent.retry_handler.can_create_escalation.return_value = False
        agent.escalation_handler = MagicMock()
        agent.escalation_handler.categorize_error.return_value = "unknown"
        agent._log_failed_escalation = MagicMock()

        await agent._handle_failure(task)

        jira.add_comment.assert_called_once()
        call_args = jira.add_comment.call_args
        assert call_args[0][0] == "PROJ-42"
        assert "failed after 5 retries" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_failure_skips_jira_when_no_key(self):
        """No jira_key → no JIRA comment on failure."""
        task = _make_task(jira_key=None)
        task.retry_count = 5
        task.last_error = "Tests failed"
        jira = MagicMock()
        agent = _make_agent(agent_definition=_make_agent_definition(), jira_client=jira)
        agent.config.max_retries = 3
        agent.retry_handler = MagicMock()
        agent.retry_handler.max_retries = 3
        agent.retry_handler.can_create_escalation.return_value = False
        agent.escalation_handler = MagicMock()
        agent.escalation_handler.categorize_error.return_value = "unknown"
        agent._log_failed_escalation = MagicMock()

        await agent._handle_failure(task)

        jira.add_comment.assert_not_called()
