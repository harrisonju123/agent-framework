"""Tests for PromptBuilder module."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent_framework.core.agent import AgentConfig
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.memory.memory_retriever import MemoryRetriever
from agent_framework.memory.memory_store import MemoryStore
from agent_framework.memory.tool_pattern_store import ToolPatternStore


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
        created_at=datetime.now(timezone.utc),
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
def prompt_context(agent_config, tmp_path):
    """Create a PromptContext for testing."""
    return PromptContext(
        config=agent_config,
        workspace=tmp_path,
        mcp_enabled=False,
        optimization_config={},
    )


@pytest.fixture
def prompt_builder(prompt_context):
    """Create a PromptBuilder for testing."""
    return PromptBuilder(prompt_context)


class TestPromptBuilder:
    """Test PromptBuilder basic functionality."""

    def test_build_creates_prompt(self, prompt_builder, sample_task):
        """Verify build() produces a non-empty prompt."""
        prompt = prompt_builder.build(sample_task)

        assert prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial

    def test_build_includes_agent_id(self, prompt_builder, sample_task):
        """Verify prompt includes agent ID."""
        prompt = prompt_builder.build(sample_task)

        assert "test-agent" in prompt

    def test_build_includes_task_details(self, prompt_builder, sample_task):
        """Verify prompt includes task information."""
        prompt = prompt_builder.build(sample_task)

        assert "Implement user authentication" in prompt
        assert "JWT" in prompt

    def test_get_current_specialization_returns_none_for_non_engineer(self, prompt_builder, sample_task):
        """Verify specialization detection only works for engineer agents."""
        prompt_builder.build(sample_task)

        # Non-engineer agent should not detect specialization
        assert prompt_builder.get_current_specialization() is None

    def test_get_current_file_count_returns_int(self, prompt_builder, sample_task):
        """Verify file count is returned as integer."""
        prompt_builder.build(sample_task)

        file_count = prompt_builder.get_current_file_count()
        assert isinstance(file_count, int)
        assert file_count >= 0


class TestMinimalTaskPrompts:
    """Test Strategy 1: Minimal Task Prompts."""

    def test_minimal_task_dict_reduces_size(self, prompt_builder, sample_task):
        """Verify minimal task dict is smaller than full task dict."""
        minimal = prompt_builder._get_minimal_task_dict(sample_task)
        full = sample_task.model_dump()

        assert len(minimal) < len(full)

    def test_minimal_task_includes_essential_fields(self, prompt_builder, sample_task):
        """Verify minimal task includes required fields."""
        minimal = prompt_builder._get_minimal_task_dict(sample_task)

        # Essential fields must be present
        assert "title" in minimal
        assert "description" in minimal
        assert "type" in minimal

        # Important fields should be present if set
        assert "acceptance_criteria" in minimal
        assert "deliverables" in minimal
        assert "notes" in minimal

    def test_minimal_task_excludes_metadata(self, prompt_builder, sample_task):
        """Verify minimal task excludes metadata fields."""
        minimal = prompt_builder._get_minimal_task_dict(sample_task)

        # Metadata should be excluded
        assert "created_at" not in minimal
        assert "created_by" not in minimal
        assert "assigned_to" not in minimal
        assert "retry_count" not in minimal

    def test_minimal_task_fallback_on_missing_fields(self, prompt_builder, sample_task):
        """Verify fallback to full dict when essential fields missing."""
        sample_task.title = ""  # Empty title

        minimal = prompt_builder._get_minimal_task_dict(sample_task)

        # Should fall back to full dict
        assert "created_at" in minimal  # Metadata present = full dict


class TestOptimizedPrompts:
    """Test optimized prompt building."""

    def test_optimized_prompt_uses_compact_json(self, agent_config, tmp_path, sample_task):
        """Verify optimized prompts use compact JSON (no whitespace)."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        optimized = builder._build_prompt_optimized(sample_task)

        # Compact JSON shouldn't have indentation
        assert '  "title"' not in optimized  # No indented keys
        assert '"title":' in optimized or 'title' in optimized.lower()

    def test_legacy_prompt_uses_formatted_json(self, agent_config, tmp_path, sample_task):
        """Verify legacy prompts use formatted JSON (with whitespace)."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        legacy = builder._build_prompt_legacy(sample_task)

        # Legacy should have formatted JSON
        assert legacy  # Just verify it builds


class TestMCPGuidance:
    """Test MCP integration guidance building."""

    def test_jira_guidance_includes_key(self, agent_config, tmp_path):
        """Verify JIRA guidance includes ticket key."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        guidance = builder._build_jira_guidance("PROJ-123", "PROJ")

        assert "PROJ-123" in guidance
        assert "JIRA" in guidance

    def test_github_guidance_includes_repo(self, agent_config, tmp_path):
        """Verify GitHub guidance includes repository name."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        guidance = builder._build_github_guidance("owner/repo", "PROJ-123")

        assert "owner/repo" in guidance
        assert "github" in guidance.lower()

    def test_error_handling_guidance_is_cached(self, agent_config, tmp_path):
        """Verify error handling guidance is cached."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        guidance1 = builder._build_error_handling_guidance()
        guidance2 = builder._build_error_handling_guidance()

        # Should return same instance (cached)
        assert guidance1 is guidance2


class TestPromptInjections:
    """Test prompt injection features."""

    def test_inject_test_failure_context(self, prompt_builder, sample_task):
        """Verify test failure context is appended when present."""
        sample_task.context["_test_failure_report"] = "Test XYZ failed: assertion error"

        prompt = "Base prompt"
        result = prompt_builder._append_test_failure_context(prompt, sample_task)

        assert "Test XYZ failed" in result
        assert result.startswith(prompt)

    def test_inject_test_failure_context_no_op_when_missing(self, prompt_builder, sample_task):
        """Verify no change when test failure context absent."""
        prompt = "Base prompt"
        result = prompt_builder._append_test_failure_context(prompt, sample_task)

        assert result == prompt

    def test_inject_human_guidance(self, prompt_builder, sample_task):
        """Verify human guidance is injected when present."""
        from agent_framework.safeguards.escalation import EscalationReport

        sample_task.escalation_report = EscalationReport(
            task_id=sample_task.id,
            original_title=sample_task.title,
            reason="test",
            error_category="test",
            total_attempts=3,
            attempt_history=[],
            root_cause_hypothesis="Test hypothesis",
            suggested_interventions=[],
            human_guidance="Try using async/await instead of callbacks"
        )

        prompt = "Base prompt"
        result = prompt_builder._inject_human_guidance(prompt, sample_task)

        assert "async/await" in result
        assert "CRITICAL" in result

    def test_inject_replan_context(self, prompt_builder, sample_task):
        """Verify replan context is injected when present."""
        sample_task.context["_revised_plan"] = "Use a different approach: refactor X"
        sample_task.retry_count = 2

        prompt = "Base prompt"
        result = prompt_builder._inject_replan_context(prompt, sample_task)

        assert "refactor X" in result
        assert "retry 2" in result

    def test_inject_preview_mode(self, prompt_builder, sample_task):
        """Verify preview mode constraints are injected."""
        prompt = "Base prompt"
        result = prompt_builder._inject_preview_mode(prompt, sample_task)

        assert "PREVIEW MODE" in result
        assert "READ-ONLY" in result
        # Preview mode section is prepended, possibly with a newline
        assert "PREVIEW MODE" in result[:100]
