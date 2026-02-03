"""Tests for optimization strategies."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMResponse
from agent_framework.queue.file_queue import FileQueue


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test-task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.utcnow(),
        title="Implement user authentication",
        description="Add JWT-based authentication to the API endpoints with proper token validation and refresh logic.",
        acceptance_criteria=["JWT tokens generated", "Token refresh works", "Protected endpoints secured"],
        deliverables=["AuthService", "JWTMiddleware", "Tests"],
        notes=["Use bcrypt for password hashing", "Token expiry: 1 hour"],
        context={
            "jira_key": "PROJ-123",
            "jira_project": "PROJ",
            "github_repo": "company/api",
            "mode": "implementation",
            "repository_name": "API Service",
        },
    )


@pytest.fixture
def agent_config():
    """Create test agent config."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        queue="test",
        prompt="You are a test agent.",
        poll_interval=30,
        max_retries=3,
        timeout=1800,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM backend."""
    llm = Mock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Task completed successfully",
        model_used="sonnet",
        input_tokens=1000,
        output_tokens=500,
        finish_reason="stop",
        latency_ms=1000,
        success=True,
    ))
    return llm


@pytest.fixture
def agent(agent_config, mock_llm, tmp_path):
    """Create a test agent with default (empty) optimization config."""
    queue = FileQueue(tmp_path)
    return Agent(
        agent_config,
        mock_llm,
        queue,
        tmp_path,
        optimization_config={},
    )


class TestMinimalTaskPrompts:
    """Test Strategy 1: Minimal Task Prompts."""

    def test_minimal_task_reduces_prompt_size(self, agent_config, mock_llm, sample_task, tmp_path):
        """Verify minimal task dict produces smaller prompts."""
        # Create agent without optimizations
        queue = FileQueue(tmp_path)
        agent_legacy = Agent(agent_config, mock_llm, queue, tmp_path, optimization_config={})
        legacy_prompt = agent_legacy._build_prompt_legacy(sample_task)

        # Create agent with optimizations
        agent_optimized = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={"enable_minimal_prompts": True, "enable_compact_json": True}
        )
        optimized_prompt = agent_optimized._build_prompt_optimized(sample_task)

        # Should be significantly smaller (at least 30%)
        assert len(optimized_prompt) < len(legacy_prompt) * 0.7, \
            f"Optimized prompt not smaller: {len(optimized_prompt)} vs {len(legacy_prompt)}"

    def test_minimal_task_includes_essential_fields(self, agent, sample_task):
        """Verify minimal task includes required fields."""
        minimal = agent._get_minimal_task_dict(sample_task)

        # Essential fields must be present
        assert "title" in minimal
        assert "description" in minimal
        assert "type" in minimal

        # Important fields should be present if set
        assert "acceptance_criteria" in minimal
        assert "deliverables" in minimal
        assert "notes" in minimal

        # Metadata should be excluded
        assert "created_at" not in minimal
        assert "created_by" not in minimal
        assert "assigned_to" not in minimal
        assert "retry_count" not in minimal

    def test_minimal_task_fallback_on_missing_fields(self, agent, sample_task):
        """Verify fallback to full dict when essential fields missing."""
        sample_task.title = ""  # Empty title

        minimal = agent._get_minimal_task_dict(sample_task)

        # Should fall back to full dict
        assert "created_at" in minimal  # Metadata present = full dict


class TestCompactJSON:
    """Test Strategy 4: Compact JSON Serialization."""

    def test_compact_json_removes_whitespace(self, agent, sample_task):
        """Verify compact JSON has no unnecessary whitespace."""
        import json

        # Get task dict and serialize both ways
        task_dict = agent._get_minimal_task_dict(sample_task)
        compact = json.dumps(task_dict, separators=(',', ':'))
        pretty = json.dumps(task_dict, indent=2)

        # Compact should be much smaller
        assert len(compact) < len(pretty) * 0.85  # At least 15% smaller


class TestResultSummarization:
    """Test Strategy 5: Result Summarization."""

    @pytest.mark.asyncio
    async def test_extract_summary_regex_patterns(self, agent, sample_task):
        """Verify regex extraction finds JIRA keys, PRs, files."""
        response = """
        Created JIRA tickets: PROJ-456, PROJ-457
        Pull request: https://github.com/company/api/pull/123
        Modified files: src/auth/service.py, src/auth/middleware.py
        """

        summary = await agent._extract_summary(response, sample_task)

        # Should extract all key information
        assert "PROJ-456" in summary
        assert "123" in summary  # PR number
        assert "src/auth" in summary

    @pytest.mark.asyncio
    async def test_extract_summary_avoids_false_positives(self, agent, sample_task):
        """Verify regex doesn't match non-JIRA patterns."""
        response = """
        Received HTTP-404 error
        Using UTF-8 encoding
        """

        summary = await agent._extract_summary(response, sample_task)

        # Should not extract HTTP-404 or UTF-8 as JIRA keys
        assert "HTTP-404" not in summary
        assert "UTF-8" not in summary

    @pytest.mark.asyncio
    async def test_extract_summary_recursion_guard(self, agent, sample_task):
        """Verify recursion guard prevents infinite loops."""
        # This should not raise recursion error
        summary = await agent._extract_summary("No patterns here", sample_task, _recursion_depth=1)

        # Should return fallback
        assert "implementation" in summary.lower()


class TestErrorTruncation:
    """Test Strategy 8: Error Truncation."""

    def test_error_truncation_preserves_structure(self, agent):
        """Verify error truncation keeps error type and key lines."""
        long_error = "ValueError: Invalid input\n" + "\n".join([f"Line {i}" for i in range(100)])

        truncated = agent.escalation_handler._truncate_error(long_error, max_lines=35)

        # Should preserve error type
        assert "ValueError" in truncated

        # Should indicate truncation
        assert "omitted" in truncated

        # Should be much shorter
        assert len(truncated.split('\n')) < len(long_error.split('\n')) * 0.5

    def test_error_truncation_handles_short_errors(self, agent):
        """Verify short errors are not truncated."""
        short_error = "ValueError: Invalid input"

        truncated = agent.escalation_handler._truncate_error(short_error)

        # Should be unchanged
        assert truncated == short_error

    def test_error_truncation_handles_already_truncated(self, agent):
        """Verify already truncated errors are not re-truncated."""
        already_truncated = "Error\n... (50 lines omitted) ..."

        truncated = agent.escalation_handler._truncate_error(already_truncated)

        # Should not be re-truncated
        assert truncated == already_truncated


class TestTokenBudgets:
    """Test Strategy 6: Token Tracking and Budgets."""

    def test_token_budget_from_config(self, agent_config, mock_llm, tmp_path, sample_task):
        """Verify token budgets can be configured."""
        queue = FileQueue(tmp_path)
        agent = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={
                "token_budgets": {
                    "implementation": 60000,
                }
            }
        )

        budget = agent._get_token_budget(TaskType.IMPLEMENTATION)
        assert budget == 60000

    def test_token_budget_uses_defaults(self, agent, sample_task):
        """Verify default budgets are used when not configured."""
        budget = agent._get_token_budget(TaskType.PLANNING)

        # Should use default
        assert budget == 30000


class TestCanaryRollout:
    """Test canary rollout selection."""

    def test_canary_selection_deterministic(self, agent_config, mock_llm, tmp_path, sample_task):
        """Verify canary selection is deterministic."""
        queue = FileQueue(tmp_path)
        agent = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={"canary_percentage": 50}
        )

        # Same task should always get same result
        result1 = agent._should_use_optimization(sample_task)
        result2 = agent._should_use_optimization(sample_task)

        assert result1 == result2

    def test_canary_percentage_zero_disabled(self, agent, sample_task):
        """Verify 0% canary disables optimizations."""
        # Agent fixture has canary_percentage=0 by default
        assert agent._should_use_optimization(sample_task) is False

    def test_canary_percentage_hundred_enabled(self, agent_config, mock_llm, tmp_path, sample_task):
        """Verify 100% canary enables all optimizations."""
        queue = FileQueue(tmp_path)
        agent = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={"canary_percentage": 100}
        )

        assert agent._should_use_optimization(sample_task) is True

    def test_optimization_override(self, agent, sample_task):
        """Verify task-level override takes precedence."""
        # Agent fixture has canary_percentage=0 by default
        sample_task.optimization_override = True
        sample_task.optimization_override_reason = "Testing override"

        # Should use optimization despite 0% canary
        assert agent._should_use_optimization(sample_task) is True


class TestCostEstimation:
    """Test cost estimation."""

    def test_cost_estimation_by_model(self, agent):
        """Verify cost estimation varies by model."""
        haiku_response = LLMResponse(
            content="Done",
            model_used="haiku",
            input_tokens=1000,
            output_tokens=500,
            finish_reason="stop",
            latency_ms=1000,
            success=True,
        )

        sonnet_response = LLMResponse(
            content="Done",
            model_used="sonnet",
            input_tokens=1000,
            output_tokens=500,
            finish_reason="stop",
            latency_ms=1000,
            success=True,
        )

        haiku_cost = agent._estimate_cost(haiku_response)
        sonnet_cost = agent._estimate_cost(sonnet_response)

        # Sonnet should cost more than Haiku
        assert sonnet_cost > haiku_cost


class TestShadowMode:
    """Test shadow mode validation."""

    def test_shadow_mode_uses_legacy_prompt(self, agent_config, mock_llm, tmp_path, sample_task):
        """Verify shadow mode uses legacy prompt despite optimizations."""
        queue = FileQueue(tmp_path)
        agent = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={
                "shadow_mode": True,
                "enable_minimal_prompts": True,
                "enable_compact_json": True,
            }
        )

        legacy_prompt = agent._build_prompt_legacy(sample_task)
        shadow_prompt = agent._build_prompt(sample_task)

        # Shadow mode should use legacy
        assert shadow_prompt == legacy_prompt

    def test_shadow_mode_records_metrics(self, agent_config, mock_llm, tmp_path, sample_task):
        """Verify shadow mode records comparison metrics."""
        queue = FileQueue(tmp_path)
        agent = Agent(
            agent_config,
            mock_llm,
            queue,
            tmp_path,
            optimization_config={"shadow_mode": True}
        )

        agent._build_prompt(sample_task)

        # Check metrics file was created
        metrics_file = tmp_path / ".agent-communication" / "metrics" / "optimization.jsonl"
        assert metrics_file.exists()

        # Verify metrics content
        import json
        with open(metrics_file) as f:
            metrics = json.loads(f.read())

        assert "savings_percent" in metrics
        assert metrics["task_id"] == sample_task.id
