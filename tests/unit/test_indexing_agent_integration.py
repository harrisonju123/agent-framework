"""Tests for codebase indexing integration with Agent and PromptBuilder."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.task import Task, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_config():
    return AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
        poll_interval=30,
        max_retries=3,
        timeout=1800,
    )


@pytest.fixture
def impl_task():
    return Task(
        id="task-impl-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Fix UserService N+1 query",
        description="Optimize the user listing endpoint to batch-load associations.",
        context={
            "github_repo": "company/api",
            "user_goal": "Reduce database round trips in UserService",
            "upstream_summary": "QA found slow response on GET /users with 50+ rows in handler.go",
            "structured_findings": {
                "handler.go": [{"severity": "HIGH", "description": "N+1 in ListUsers"}],
                "service.go": [{"severity": "MEDIUM", "description": "Missing index"}],
            },
        },
    )


@pytest.fixture
def planning_task():
    return Task(
        id="task-plan-1",
        type=TaskType.PLANNING,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="architect",
        created_at=datetime.now(timezone.utc),
        title="Design caching layer",
        description="Plan the caching architecture.",
        context={"github_repo": "company/api"},
    )


@pytest.fixture
def prompt_ctx(agent_config, tmp_path):
    return PromptContext(
        config=agent_config,
        workspace=tmp_path,
        mcp_enabled=False,
        optimization_config={},
    )


# ---------------------------------------------------------------------------
# _init_code_indexing tests
# ---------------------------------------------------------------------------

class TestInitCodeIndexing:
    """Tests for Agent._init_code_indexing."""

    def _make_agent(self):
        agent = MagicMock()
        agent.workspace = Path("/tmp/test-workspace")
        agent.logger = MagicMock()
        agent._init_code_indexing = Agent._init_code_indexing.__get__(agent)
        return agent

    @patch("agent_framework.indexing.IndexQuery")
    @patch("agent_framework.indexing.CodebaseIndexer")
    @patch("agent_framework.indexing.IndexStore")
    def test_creates_objects_when_enabled(self, MockStore, MockIndexer, MockQuery):
        agent = self._make_agent()
        agent._init_code_indexing({"enabled": True, "max_symbols": 300})

        assert agent._code_indexing_enabled is True
        MockStore.assert_called_once_with(agent.workspace)
        MockIndexer.assert_called_once()
        MockQuery.assert_called_once()
        assert agent._code_indexer is not None
        assert agent._code_index_query is not None

    def test_stays_none_when_disabled(self):
        agent = self._make_agent()
        agent._init_code_indexing({"enabled": False})

        assert agent._code_indexing_enabled is False
        assert agent._code_indexer is None
        assert agent._code_index_query is None

    @patch("agent_framework.indexing.IndexQuery")
    @patch("agent_framework.indexing.CodebaseIndexer")
    @patch("agent_framework.indexing.IndexStore")
    def test_handles_none_config(self, MockStore, MockIndexer, MockQuery):
        agent = self._make_agent()
        agent._init_code_indexing(None)

        # Default is enabled=True
        assert agent._code_indexing_enabled is True
        assert agent._code_indexer is not None


# ---------------------------------------------------------------------------
# _try_index_codebase tests
# ---------------------------------------------------------------------------

class TestTryIndexCodebase:
    """Tests for Agent._try_index_codebase."""

    def _make_agent(self):
        agent = MagicMock()
        agent.config = AgentConfig(
            id="engineer", name="Engineer", queue="engineer",
            prompt="test", poll_interval=30, max_retries=3, timeout=1800,
        )
        agent.logger = MagicMock()
        agent._code_indexer = MagicMock()
        agent._code_indexing_inject_for = ["architect", "engineer", "qa"]
        agent._try_index_codebase = Agent._try_index_codebase.__get__(agent)
        return agent

    def test_calls_ensure_indexed(self, impl_task):
        agent = self._make_agent()
        repo_path = Path("/tmp/repo")

        agent._try_index_codebase(impl_task, repo_path)

        agent._code_indexer.ensure_indexed.assert_called_once_with("company/api", str(repo_path))

    def test_skips_excluded_agent_type(self, impl_task):
        agent = self._make_agent()
        agent._code_indexing_inject_for = ["architect"]

        agent._try_index_codebase(impl_task, Path("/tmp/repo"))

        agent._code_indexer.ensure_indexed.assert_not_called()

    def test_skips_when_no_github_repo(self):
        agent = self._make_agent()
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="No repo", description="No repo task", context={},
        )

        agent._try_index_codebase(task, Path("/tmp/repo"))

        agent._code_indexer.ensure_indexed.assert_not_called()

    def test_swallows_exceptions(self, impl_task):
        agent = self._make_agent()
        agent._code_indexer.ensure_indexed.side_effect = RuntimeError("disk full")

        # Should not raise
        agent._try_index_codebase(impl_task, Path("/tmp/repo"))

        agent.logger.debug.assert_called()

    def test_skips_when_no_indexer(self, impl_task):
        agent = self._make_agent()
        agent._code_indexer = None

        # Should return early without error
        agent._try_index_codebase(impl_task, Path("/tmp/repo"))


# ---------------------------------------------------------------------------
# _inject_codebase_index tests
# ---------------------------------------------------------------------------

class TestInjectCodebaseIndex:
    """Tests for PromptBuilder._inject_codebase_index."""

    def _make_builder(self, prompt_ctx, **overrides):
        for k, v in overrides.items():
            setattr(prompt_ctx, k, v)
        return PromptBuilder(prompt_ctx)

    def test_appends_section(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        mock_query.query_for_prompt.return_value = "## Codebase Index\n- UserService class"
        builder = self._make_builder(prompt_ctx, code_index_query=mock_query)

        result = builder._inject_codebase_index("base prompt", impl_task)

        assert "## Codebase Index" in result
        assert result.startswith("base prompt\n\n")

    def test_skips_when_query_is_none(self, prompt_ctx, impl_task):
        builder = self._make_builder(prompt_ctx, code_index_query=None)

        result = builder._inject_codebase_index("base prompt", impl_task)

        assert result == "base prompt"

    def test_skips_excluded_agent(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        builder = self._make_builder(
            prompt_ctx,
            code_index_query=mock_query,
            code_indexing_config={"inject_for_agents": ["architect"]},
        )

        result = builder._inject_codebase_index("base prompt", impl_task)

        assert result == "base prompt"
        mock_query.query_for_prompt.assert_not_called()

    def test_uses_overview_for_planning_tasks(self, prompt_ctx, planning_task):
        # Need architect config for planning task to pass the agent filter
        prompt_ctx.config = AgentConfig(
            id="architect", name="Architect", queue="architect",
            prompt="test", poll_interval=30, max_retries=3, timeout=1800,
        )
        mock_query = MagicMock()
        mock_query.format_overview_only.return_value = "## Overview\nModules: 3"
        builder = self._make_builder(prompt_ctx, code_index_query=mock_query)

        result = builder._inject_codebase_index("base prompt", planning_task)

        mock_query.format_overview_only.assert_called_once_with("company/api")
        mock_query.query_for_prompt.assert_not_called()
        assert "## Overview" in result

    def test_respects_compute_memory_budget(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        mock_query.query_for_prompt.return_value = "symbols"
        mock_cwm = MagicMock()
        mock_cwm.compute_memory_budget.return_value = 2000
        builder = self._make_builder(
            prompt_ctx,
            code_index_query=mock_query,
            context_window_manager=mock_cwm,
            code_indexing_config={"max_prompt_chars": 4000},
        )

        builder._inject_codebase_index("base prompt", impl_task)

        # Should pass min(4000, 2000) = 2000
        _, kwargs = mock_query.query_for_prompt.call_args
        assert kwargs["max_chars"] == 2000

    def test_skips_when_budget_is_zero(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        mock_cwm = MagicMock()
        mock_cwm.compute_memory_budget.return_value = 0
        builder = self._make_builder(
            prompt_ctx,
            code_index_query=mock_query,
            context_window_manager=mock_cwm,
        )

        result = builder._inject_codebase_index("base prompt", impl_task)

        assert result == "base prompt"
        mock_query.query_for_prompt.assert_not_called()

    def test_skips_when_no_github_repo(self, prompt_ctx):
        mock_query = MagicMock()
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="No repo", description="desc", context={},
        )
        builder = self._make_builder(prompt_ctx, code_index_query=mock_query)

        result = builder._inject_codebase_index("base prompt", task)

        assert result == "base prompt"

    def test_skips_when_section_empty(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        mock_query.query_for_prompt.return_value = ""
        builder = self._make_builder(prompt_ctx, code_index_query=mock_query)

        result = builder._inject_codebase_index("base prompt", impl_task)

        assert result == "base prompt"


# ---------------------------------------------------------------------------
# _build_index_query_goal tests
# ---------------------------------------------------------------------------

class TestBuildIndexQueryGoal:
    """Tests for PromptBuilder._build_index_query_goal."""

    def _make_builder(self, prompt_ctx):
        return PromptBuilder(prompt_ctx)

    def test_includes_title_and_description(self, prompt_ctx, impl_task):
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(impl_task)

        assert "Fix UserService N+1 query" in goal
        assert "Optimize the user listing" in goal

    def test_includes_upstream_summary(self, prompt_ctx, impl_task):
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(impl_task)

        assert "handler.go" in goal
        assert "slow response" in goal

    def test_includes_structured_findings_file_paths(self, prompt_ctx, impl_task):
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(impl_task)

        assert "handler.go" in goal
        assert "service.go" in goal

    def test_caps_upstream_at_1000_chars(self, prompt_ctx):
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="T", description="D",
            context={"upstream_summary": "x" * 2000},
        )
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(task)

        # Title "T" + desc "D" + 1000 x's + spaces = 1005
        upstream_portion = goal.split()[-1]  # last token is the capped upstream
        assert len(upstream_portion) == 1000

    def test_includes_user_goal(self, prompt_ctx, impl_task):
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(impl_task)

        assert "Reduce database round trips" in goal

    def test_skips_user_goal_if_same_as_title(self, prompt_ctx):
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Same text", description="desc",
            context={"user_goal": "Same text"},
        )
        builder = self._make_builder(prompt_ctx)
        goal = builder._build_index_query_goal(task)

        # "Same text" should appear only once (from title)
        assert goal.count("Same text") == 1


# ---------------------------------------------------------------------------
# build() end-to-end test
# ---------------------------------------------------------------------------

class TestBuildIncludesIndex:
    """Verify build() includes the codebase index section in the final prompt."""

    def test_build_includes_index_section(self, prompt_ctx, impl_task):
        mock_query = MagicMock()
        mock_query.query_for_prompt.return_value = "## Codebase Index\n- UserService"
        prompt_ctx.code_index_query = mock_query
        builder = PromptBuilder(prompt_ctx)

        prompt = builder.build(impl_task)

        assert "## Codebase Index" in prompt
        assert "UserService" in prompt
