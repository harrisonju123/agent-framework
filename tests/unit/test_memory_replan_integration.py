"""Tests for Memory ↔ Replanning integration.

Verifies that _request_replan() injects relevant memories from the repo
to inform revised approaches when tasks fail on retry 2+.
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse
from agent_framework.memory.memory_retriever import MemoryRetriever
from agent_framework.memory.memory_store import MemoryStore


def _make_task(**overrides) -> Task:
    defaults = dict(
        id="task-mem-replan-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Fix auth bug",
        description="Fix the authentication module",
        context={"github_repo": "myorg/myrepo"},
        notes=[],
        retry_count=2,
        last_error="Tests failed: 3 assertions",
        replan_history=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_llm_response(content: str, success: bool = True) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used="haiku",
        input_tokens=100,
        output_tokens=50,
        finish_reason="end_turn",
        latency_ms=200,
        success=success,
    )


def _build_agent(tmp_path, llm_mock=None, memory_enabled=True) -> Agent:
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        queue="engineer",
        prompt="You are a test agent.",
    )
    llm = llm_mock or AsyncMock()
    queue = MagicMock()

    with patch("agent_framework.core.agent.setup_rich_logging") as mock_log, \
         patch("agent_framework.workflow.executor.WorkflowExecutor"):
        mock_log.return_value = MagicMock()
        agent = Agent(
            config=config,
            llm=llm,
            queue=queue,
            workspace=tmp_path,
            replan_config={"enabled": True, "min_retry_for_replan": 2, "model": "haiku"},
            memory_config={"enabled": memory_enabled},
        )
    return agent


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory", enabled=True)


@pytest.fixture
def retriever(store):
    return MemoryRetriever(store)


class TestFormatForReplan:
    def test_returns_empty_when_no_memories(self, retriever):
        result = retriever.format_for_replan(
            repo_slug="myorg/myrepo",
            agent_type="engineer",
        )
        assert result == ""

    def test_prioritizes_conventions_test_commands_repo_structure(self, store, retriever):
        repo = "myorg/myrepo"
        agent = "engineer"

        # Store memories in different categories
        store.remember(repo, agent, "info", "Some general info")
        store.remember(repo, agent, "conventions", "Use PEP8 style")
        store.remember(repo, agent, "test_commands", "Run pytest with -v flag")
        store.remember(repo, agent, "repo_structure", "Tests in tests/unit/")
        store.remember(repo, agent, "debugging", "Use pdb for debugging")

        result = retriever.format_for_replan(repo, agent)

        # Priority categories should appear first
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        categories = [l.split("]")[0][3:] for l in lines]

        # First 3 should be priority categories (in some order)
        priority_cats = set(categories[:3])
        assert priority_cats == {"conventions", "test_commands", "repo_structure"}

    def test_respects_max_chars_budget(self, store, retriever):
        repo = "myorg/myrepo"
        agent = "engineer"

        # Fill with many memories
        for i in range(30):
            store.remember(repo, agent, "conventions", "x" * 100 + f" rule {i}")

        result = retriever.format_for_replan(repo, agent, max_chars=500)

        # Count chars in memory lines only (exclude header)
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        total_chars = sum(len(l) for l in lines)
        assert total_chars <= 500

    def test_default_max_chars_is_1500(self, store, retriever):
        repo = "myorg/myrepo"
        agent = "engineer"

        # Add many long memories
        for i in range(50):
            store.remember(repo, agent, "info", "y" * 200 + f" {i}")

        result = retriever.format_for_replan(repo, agent)

        lines = [l for l in result.split("\n") if l.startswith("- [")]
        total_chars = sum(len(l) for l in lines)
        assert total_chars <= 1500

    def test_returns_empty_when_only_header_would_be_included(self, retriever):
        # When max_chars is too small to fit any memories
        result = retriever.format_for_replan(
            repo_slug="myorg/myrepo",
            agent_type="engineer",
            max_chars=10,
        )
        assert result == ""

    def test_includes_header_and_entries(self, store, retriever):
        repo = "myorg/myrepo"
        agent = "engineer"
        store.remember(repo, agent, "conventions", "Use tabs not spaces")

        result = retriever.format_for_replan(repo, agent)
        assert "## Relevant Memories from This Repo" in result
        assert "- [conventions] Use tabs not spaces" in result


class TestRequestReplanWithMemory:
    async def test_includes_memory_context_when_enabled_and_memories_exist(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Revised plan"))
        agent = _build_agent(tmp_path, llm, memory_enabled=True)

        # Store some memories
        repo = "myorg/myrepo"
        agent._memory_store.remember(repo, agent.config.base_id, "conventions", "Always lint")
        agent._memory_store.remember(repo, agent.config.base_id, "test_commands", "Run go test ./...")

        task = _make_task(context={"github_repo": repo})

        await agent._request_replan(task)

        # Check that prompt included memory context
        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories from This Repo" in prompt
        assert "Always lint" in prompt or "Run go test" in prompt

    async def test_works_without_memory_when_disabled(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan B"))
        agent = _build_agent(tmp_path, llm, memory_enabled=False)
        task = _make_task()

        # Should not crash
        await agent._request_replan(task)

        # Verify no memory section in prompt
        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories" not in prompt

    async def test_works_when_repo_slug_is_none(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan C"))
        agent = _build_agent(tmp_path, llm, memory_enabled=True)

        # Task without github_repo in context
        task = _make_task(context={})

        await agent._request_replan(task)

        # Should work without crashing, no memory section
        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories" not in prompt

    async def test_works_when_no_memories_stored_yet(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan D"))
        agent = _build_agent(tmp_path, llm, memory_enabled=True)
        task = _make_task()

        # No memories stored for this repo
        await agent._request_replan(task)

        # Should work, no memory section added
        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "## Relevant Memories" not in prompt

    async def test_logs_memory_injection_when_memories_included(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan E"))
        agent = _build_agent(tmp_path, llm, memory_enabled=True)

        # Mock session logger to track calls
        agent._session_logger = MagicMock()

        repo = "myorg/myrepo"
        agent._memory_store.remember(repo, agent.config.base_id, "conventions", "Code style X")

        task = _make_task(context={"github_repo": repo})

        await agent._request_replan(task)

        # Verify session logger was called with memory_chars_injected
        log_calls = agent._session_logger.log.call_args_list
        replan_log = [c for c in log_calls if c[0][0] == "replan"]
        assert len(replan_log) > 0
        log_kwargs = replan_log[0][1]
        assert "memory_chars_injected" in log_kwargs
        assert log_kwargs["memory_chars_injected"] > 0

    async def test_instructions_include_use_repo_knowledge_hint(self, tmp_path):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response("Plan F"))
        agent = _build_agent(tmp_path, llm, memory_enabled=True)

        repo = "myorg/myrepo"
        agent._memory_store.remember(repo, agent.config.base_id, "repo_structure", "Structure X")

        task = _make_task(context={"github_repo": repo})

        await agent._request_replan(task)

        call_args = llm.complete.call_args
        prompt = call_args[0][0].prompt
        assert "Use the repo knowledge above to inform your revised approach" in prompt
