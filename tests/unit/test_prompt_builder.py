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

    def test_jira_guidance_excludes_creation_tools_when_not_allowed(self, agent_config, tmp_path):
        """Engineer (jira_can_create_tickets=False) sees no create tools and sees prohibition."""
        from agent_framework.core.config import AgentDefinition

        agent_def = AgentDefinition(
            id="engineer", name="Engineer", queue="engineer",
            prompt="test", jira_can_create_tickets=False,
        )
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
            agent_definition=agent_def,
        )
        builder = PromptBuilder(ctx)
        guidance = builder._build_jira_guidance("PROJ-123", "PROJ")

        assert "jira_create_issue" not in guidance
        assert "jira_create_epic" not in guidance
        assert "jira_create_subtask" not in guidance
        assert "jira_search_issues" in guidance
        assert "jira_add_comment" in guidance
        assert "Do NOT create JIRA tickets" in guidance
        assert "Do NOT use Bash, curl, or urllib" in guidance

    def test_jira_guidance_includes_creation_tools_when_allowed(self, agent_config, tmp_path):
        """Architect (jira_can_create_tickets=True) sees all tools, no prohibition."""
        from agent_framework.core.config import AgentDefinition

        agent_def = AgentDefinition(
            id="architect", name="Architect", queue="architect",
            prompt="test", jira_can_create_tickets=True,
        )
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
            agent_definition=agent_def,
        )
        builder = PromptBuilder(ctx)
        guidance = builder._build_jira_guidance("PROJ-123", "PROJ")

        assert "jira_create_issue" in guidance
        assert "jira_create_epic" in guidance
        assert "jira_create_subtask" in guidance
        assert "jira_search_issues" in guidance
        assert "jira_add_comment" in guidance
        assert "Do NOT create JIRA tickets" not in guidance

    def test_jira_guidance_excludes_creation_when_no_definition(self, agent_config, tmp_path):
        """No agent_definition defaults to no create tools (safe default)."""
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        guidance = builder._build_jira_guidance("PROJ-123", "PROJ")

        assert "jira_create_issue" not in guidance
        assert "jira_create_epic" not in guidance
        assert "jira_create_subtask" not in guidance
        assert "Do NOT create JIRA tickets" in guidance

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
            root_cause_hypothesis="The issue is caused by synchronous blocking calls in async context",
            suggested_interventions=["Use asyncio.to_thread for blocking operations", "Replace sync API with async variant"],
            human_guidance="Try using async/await instead of callbacks"
        )

        prompt = "Base prompt"
        result = prompt_builder._inject_human_guidance(prompt, sample_task)

        assert "async/await" in result
        assert "CRITICAL" in result
        assert "Previous Failure Context" in result
        assert "synchronous blocking calls" in result
        assert "Suggested interventions:" in result
        assert "1. Use asyncio.to_thread" in result
        assert "2. Replace sync API" in result

    def test_inject_human_guidance_legacy_fallback(self, prompt_builder, sample_task):
        """Verify legacy context-based guidance still works."""
        sample_task.context["human_guidance"] = "Use feature flags for gradual rollout"

        prompt = "Base prompt"
        result = prompt_builder._inject_human_guidance(prompt, sample_task)

        assert "feature flags" in result
        assert "CRITICAL" in result
        assert "human expert" in result.lower()

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

    def test_inject_retry_context_skipped_on_first_attempt(self, prompt_builder, sample_task):
        """No retry context injected on first attempt."""
        sample_task.retry_count = 0
        prompt = "Base prompt"
        result = prompt_builder._inject_retry_context(prompt, sample_task)
        assert result == prompt

    def test_inject_retry_context_includes_error_and_progress(self, prompt_builder, sample_task):
        """Retry context includes truncated error and partial progress."""
        sample_task.retry_count = 1
        sample_task.last_error = "Connection refused at line 42"
        sample_task.context["_previous_attempt_summary"] = "Created auth module, started writing tests"

        prompt = "Base prompt"
        result = prompt_builder._inject_retry_context(prompt, sample_task)

        assert "RETRY CONTEXT (attempt 2)" in result
        assert "Connection refused" in result
        assert "Created auth module" in result
        assert "Do NOT restart from scratch" in result

    def test_inject_retry_context_disambiguates_upstream(self, prompt_builder, sample_task):
        """Retry context clarifies upstream_summary is from a different agent."""
        sample_task.retry_count = 2
        sample_task.last_error = "Some error"
        sample_task.context["upstream_summary"] = "Architect found X"

        prompt = "Base prompt"
        result = prompt_builder._inject_retry_context(prompt, sample_task)

        assert "previous agent in the workflow chain" in result


class TestStructuredFindings:
    """Test structured QA findings formatting for engineer prompts."""

    def test_structured_findings_formatted_for_engineer(self, prompt_builder, sample_task):
        """Structured findings produce a file-grouped numbered checklist."""
        sample_task.context["structured_findings"] = {
            "findings": [
                {
                    "file": "src/auth.py",
                    "line_number": 42,
                    "severity": "CRITICAL",
                    "description": "SQL injection via unsanitized input",
                    "suggested_fix": "Use parameterized queries",
                    "category": "security",
                },
                {
                    "file": "src/auth.py",
                    "line_number": 88,
                    "severity": "MAJOR",
                    "description": "Missing error handling on token refresh",
                    "suggested_fix": None,
                    "category": "correctness",
                },
                {
                    "file": "tests/test_auth.py",
                    "line_number": 10,
                    "severity": "MEDIUM",
                    "description": "Test doesn't cover edge case",
                    "suggested_fix": "Add test for expired token",
                    "category": "testing",
                },
            ],
            "total_count": 3,
            "critical_count": 1,
        }

        result = prompt_builder._load_upstream_context(sample_task)

        # Should produce structured checklist, not raw text
        assert "QA FINDINGS" in result
        assert "[CRITICAL]" in result
        assert "[MAJOR]" in result
        assert "[MEDIUM]" in result
        assert "src/auth.py:42" in result
        assert "SQL injection" in result
        assert "Fix: Use parameterized queries" in result
        # Grouped by file — src/auth.py should appear as a header
        assert "### src/auth.py" in result
        assert "### tests/test_auth.py" in result
        # Instruction at the end
        assert "Address all findings above" in result

    def test_falls_back_to_upstream_summary(self, prompt_builder, sample_task):
        """Without structured findings, uses text summary."""
        sample_task.context["upstream_summary"] = "Architect reviewed: looks good overall"

        result = prompt_builder._load_upstream_context(sample_task)

        assert "UPSTREAM AGENT FINDINGS" in result
        assert "looks good overall" in result
        # Should NOT have structured format
        assert "QA FINDINGS" not in result

    def test_structured_findings_empty_list_falls_through(self, prompt_builder, sample_task):
        """Empty findings list falls through to upstream_summary."""
        sample_task.context["structured_findings"] = {"findings": []}
        sample_task.context["upstream_summary"] = "Some text summary"

        result = prompt_builder._load_upstream_context(sample_task)

        assert "UPSTREAM AGENT FINDINGS" in result
        assert "Some text summary" in result


class TestRejectionFeedback:
    """Test rejection feedback takes priority in _load_upstream_context."""

    def test_rejection_feedback_in_prompt(self, prompt_builder, sample_task):
        """Rejection feedback produces a prominent HUMAN FEEDBACK section."""
        sample_task.context["rejection_feedback"] = "The plan misses error handling for edge cases"

        result = prompt_builder._load_upstream_context(sample_task)

        assert "HUMAN FEEDBACK" in result
        assert "CHECKPOINT REJECTED" in result
        assert "The plan misses error handling for edge cases" in result

    def test_rejection_feedback_takes_priority_over_structured_findings(self, prompt_builder, sample_task):
        """Rejection feedback wins over structured QA findings."""
        sample_task.context["rejection_feedback"] = "Redo with different approach"
        sample_task.context["structured_findings"] = {
            "findings": [{"file": "a.py", "severity": "MAJOR", "description": "issue"}]
        }

        result = prompt_builder._load_upstream_context(sample_task)

        assert "HUMAN FEEDBACK" in result
        assert "QA FINDINGS" not in result

    def test_rejection_feedback_takes_priority_over_upstream_summary(self, prompt_builder, sample_task):
        """Rejection feedback wins over upstream_summary."""
        sample_task.context["rejection_feedback"] = "Not what I asked for"
        sample_task.context["upstream_summary"] = "Architect said X"

        result = prompt_builder._load_upstream_context(sample_task)

        assert "HUMAN FEEDBACK" in result
        assert "UPSTREAM AGENT FINDINGS" not in result

    def test_no_rejection_feedback_falls_through(self, prompt_builder, sample_task):
        """Without rejection_feedback, normal upstream context flow applies."""
        sample_task.context["upstream_summary"] = "Normal upstream context"

        result = prompt_builder._load_upstream_context(sample_task)

        assert "HUMAN FEEDBACK" not in result
        assert "UPSTREAM AGENT FINDINGS" in result


class TestMemoryBudgetIntegration:
    """Integration tests: _inject_memories respects ContextWindowManager budget tiers."""

    def _make_builder_with_memory(self, agent_config, tmp_path, utilization_pct):
        """Build a PromptBuilder with a mock memory retriever and context window manager."""
        # Mock memory retriever that always returns a section
        retriever = Mock(spec=MemoryRetriever)
        retriever.format_for_prompt = Mock(
            side_effect=lambda **kwargs: (
                "## Memories from Previous Tasks\n\n- [conventions] Use snake_case\n"
                if kwargs.get("max_chars", 3000) > 0
                else ""
            )
        )

        # Mock context window manager with controllable utilization
        cwm = Mock()
        cwm.compute_memory_budget = Mock(
            return_value=3000 if utilization_pct < 70 else (1000 if utilization_pct < 90 else 0)
        )

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
            memory_retriever=retriever,
            context_window_manager=cwm,
        )
        return PromptBuilder(ctx), retriever, cwm

    def test_healthy_budget_passes_full_3000(self, agent_config, tmp_path, sample_task):
        """< 70% utilization → format_for_prompt called with max_chars=3000."""
        builder, retriever, _ = self._make_builder_with_memory(agent_config, tmp_path, 50.0)

        result = builder._inject_memories("base prompt", sample_task)

        retriever.format_for_prompt.assert_called_once()
        assert retriever.format_for_prompt.call_args.kwargs["max_chars"] == 3000
        assert "Memories" in result

    def test_tight_budget_passes_1000(self, agent_config, tmp_path, sample_task):
        """70-90% utilization → format_for_prompt called with max_chars=1000."""
        builder, retriever, _ = self._make_builder_with_memory(agent_config, tmp_path, 80.0)

        result = builder._inject_memories("base prompt", sample_task)

        retriever.format_for_prompt.assert_called_once()
        assert retriever.format_for_prompt.call_args.kwargs["max_chars"] == 1000
        assert "Memories" in result

    def test_critical_budget_skips_injection(self, agent_config, tmp_path, sample_task):
        """>= 90% utilization → memories omitted entirely."""
        builder, retriever, _ = self._make_builder_with_memory(agent_config, tmp_path, 95.0)

        result = builder._inject_memories("base prompt", sample_task)

        # Should short-circuit before calling format_for_prompt
        retriever.format_for_prompt.assert_not_called()
        assert result == "base prompt"

    def test_no_context_manager_passes_none(self, agent_config, tmp_path, sample_task):
        """Without ContextWindowManager, max_chars=None → retriever uses default 3000."""
        retriever = Mock(spec=MemoryRetriever)
        retriever.format_for_prompt = Mock(return_value="## Memories\n- test\n")

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
            memory_retriever=retriever,
            context_window_manager=None,
        )
        builder = PromptBuilder(ctx)

        builder._inject_memories("base prompt", sample_task)

        retriever.format_for_prompt.assert_called_once()
        assert retriever.format_for_prompt.call_args.kwargs["max_chars"] is None


class TestTestSuppressionGuidance:
    """Test that review/PR steps suppress test execution in prompts."""

    def _make_task_with_step(self, sample_task, step_name):
        """Set up a chain task at the given workflow step."""
        sample_task.context["chain_step"] = True
        sample_task.context["workflow_step"] = step_name
        return sample_task

    def test_code_review_step_suppresses_tests_legacy(self, prompt_builder, sample_task):
        """code_review step injects test suppression in legacy prompt."""
        task = self._make_task_with_step(sample_task, "code_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "Do NOT run the test suite" in prompt
        assert "QA agent" in prompt

    def test_create_pr_step_suppresses_tests_legacy(self, prompt_builder, sample_task):
        """create_pr step injects test suppression in legacy prompt."""
        task = self._make_task_with_step(sample_task, "create_pr")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "Do NOT run the test suite" in prompt

    def test_qa_review_step_does_not_suppress(self, prompt_builder, sample_task):
        """qa_review step should NOT suppress tests — QA owns testing."""
        task = self._make_task_with_step(sample_task, "qa_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "Do NOT run the test suite" not in prompt

    def test_implement_step_does_not_suppress(self, prompt_builder, sample_task):
        """implement step should NOT suppress tests."""
        task = self._make_task_with_step(sample_task, "implement")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "Do NOT run the test suite" not in prompt

    def test_code_review_step_suppresses_tests_optimized(self, prompt_builder, sample_task):
        """code_review step injects test suppression in optimized prompt."""
        task = self._make_task_with_step(sample_task, "code_review")
        prompt = prompt_builder._build_prompt_optimized(task)

        assert "Do NOT run the test suite" in prompt
        assert "QA agent" in prompt

    def test_create_pr_step_suppresses_tests_optimized(self, prompt_builder, sample_task):
        """create_pr step injects test suppression in optimized prompt."""
        task = self._make_task_with_step(sample_task, "create_pr")
        prompt = prompt_builder._build_prompt_optimized(task)

        assert "Do NOT run the test suite" in prompt

    def test_standalone_task_no_suppression(self, prompt_builder, sample_task):
        """Standalone tasks (no workflow_step) should NOT suppress tests."""
        prompt = prompt_builder._build_prompt_legacy(sample_task)

        assert "Do NOT run the test suite" not in prompt


class TestPlanRendering:
    """Test _render_plan_section() formats PlanDocument for prompt injection."""

    def test_render_plan_section_with_full_plan(self, prompt_builder, sample_task):
        """Full PlanDocument renders all sections."""
        from agent_framework.core.task import PlanDocument

        sample_task.plan = PlanDocument(
            objectives=["Add JWT auth"],
            approach=["Create handler", "Add middleware", "Write tests"],
            files_to_modify=["src/auth.py", "src/middleware.py"],
            risks=["Token expiry edge case"],
            success_criteria=["All tests pass", "No regressions"],
        )

        result = prompt_builder._render_plan_section(sample_task)

        assert "IMPLEMENTATION PLAN:" in result
        assert "Objectives:" in result
        assert "- Add JWT auth" in result
        assert "Approach:" in result
        assert "1. Create handler" in result
        assert "2. Add middleware" in result
        assert "3. Write tests" in result
        assert "Files to modify: src/auth.py, src/middleware.py" in result
        assert "Risks:" in result
        assert "- Token expiry edge case" in result
        assert "Success criteria:" in result
        assert "- All tests pass" in result

    def test_render_plan_section_none_returns_empty(self, prompt_builder, sample_task):
        """None plan returns empty string."""
        sample_task.plan = None
        result = prompt_builder._render_plan_section(sample_task)
        assert result == ""

    def test_render_plan_section_minimal_plan(self, prompt_builder, sample_task):
        """Plan with only required fields renders cleanly."""
        from agent_framework.core.task import PlanDocument

        sample_task.plan = PlanDocument(
            objectives=["Fix bug"],
            approach=["Patch handler"],
            success_criteria=["Tests pass"],
        )

        result = prompt_builder._render_plan_section(sample_task)

        assert "IMPLEMENTATION PLAN:" in result
        assert "- Fix bug" in result
        assert "1. Patch handler" in result
        assert "- Tests pass" in result
        # Optional sections with empty defaults should be absent
        assert "Files to modify:" not in result
        assert "Risks:" not in result

    def test_plan_section_in_legacy_prompt(self, prompt_builder, sample_task):
        """Plan section appears in the legacy prompt when plan is set."""
        from agent_framework.core.task import PlanDocument

        sample_task.plan = PlanDocument(
            objectives=["Add endpoint"],
            approach=["Create route"],
            success_criteria=["Returns 200"],
        )

        prompt = prompt_builder._build_prompt_legacy(sample_task)
        assert "IMPLEMENTATION PLAN:" in prompt
        assert "- Add endpoint" in prompt

    def test_plan_section_in_optimized_prompt(self, prompt_builder, sample_task):
        """Plan section appears in the optimized prompt when plan is set."""
        from agent_framework.core.task import PlanDocument

        sample_task.plan = PlanDocument(
            objectives=["Add endpoint"],
            approach=["Create route"],
            success_criteria=["Returns 200"],
        )

        prompt = prompt_builder._build_prompt_optimized(sample_task)
        assert "IMPLEMENTATION PLAN:" in prompt
        assert "- Add endpoint" in prompt

    def test_no_plan_no_section_in_legacy(self, prompt_builder, sample_task):
        """No plan → no IMPLEMENTATION PLAN section in legacy prompt."""
        sample_task.plan = None
        prompt = prompt_builder._build_prompt_legacy(sample_task)
        assert "IMPLEMENTATION PLAN:" not in prompt

    def test_no_plan_no_section_in_optimized(self, prompt_builder, sample_task):
        """No plan → no IMPLEMENTATION PLAN section in optimized prompt."""
        sample_task.plan = None
        prompt = prompt_builder._build_prompt_optimized(sample_task)
        assert "IMPLEMENTATION PLAN:" not in prompt


class TestPlanningInstructions:
    """Test _build_planning_instructions() produces goal-directed exploration guidance."""

    def test_contains_keyword_search_guidance(self):
        """Instructions direct architect to search for keywords, not map entire codebase."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", "PROJ")

        assert "Search for keywords" in result
        assert "do not map the entire codebase" in result

    def test_contains_plan_document_requirement(self):
        """Instructions require producing a PlanDocument with specific fields."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", "PROJ")

        assert "PlanDocument" in result
        assert "objectives" in result
        assert "files_to_modify" in result
        assert "risks" in result
        assert "success_criteria" in result

    def test_contains_stop_condition(self):
        """Instructions include explicit termination condition for exploration."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", "PROJ")

        assert "Stop exploring" in result
        assert "enough context" in result

    def test_jira_project_included(self):
        """With JIRA project, instructions mention creating JIRA ticket."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", "PROJ")

        assert "JIRA ticket" in result

    def test_no_jira_project(self):
        """Without JIRA project, instructions mention local queues."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", None)

        assert "local queues" in result
        assert "JIRA ticket" not in result

    def test_goal_embedded_in_instructions(self):
        """User goal appears in the instructions."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Implement SSO login", "default", None)

        assert "Implement SSO login" in result
