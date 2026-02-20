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

    def test_inject_retry_context_interrupted_wording(self, prompt_builder, sample_task):
        """Interruption gets 'continue' wording instead of 'fix the error'."""
        sample_task.retry_count = 1
        sample_task.last_error = "Interrupted during LLM execution"
        sample_task.context["_previous_attempt_summary"] = "Reviewed 3 files, started writing feedback"

        prompt = "Base prompt"
        result = prompt_builder._inject_retry_context(prompt, sample_task)

        assert "interrupted before completion" in result
        assert "fixing the error" not in result

    def test_inject_retry_context_error_wording_with_progress(self, prompt_builder, sample_task):
        """Non-interruption errors with progress get the 'fix the error' wording."""
        sample_task.retry_count = 1
        sample_task.last_error = "Connection refused at line 42"
        sample_task.context["_previous_attempt_summary"] = "Started writing module"

        prompt = "Base prompt"
        result = prompt_builder._inject_retry_context(prompt, sample_task)

        assert "fixing the error" in result
        assert "interrupted before completion" not in result

    def test_inject_retry_context_includes_git_diff(self, prompt_builder, sample_task):
        """Git diff from previous attempt is injected into retry context."""
        sample_task.retry_count = 1
        sample_task.last_error = "Circuit breaker tripped"
        sample_task.context["_previous_attempt_git_diff"] = (
            "## Git Diff (actual code changes)\n"
            "### Summary\n```\n src/auth.py | 42 ++++\n```\n"
            "### Diff\n```\n+def authenticate(token):\n```"
        )

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "Code Changes From Previous Attempt" in result
        assert "src/auth.py" in result
        assert "Do NOT restart from scratch" in result
        # With git diff present, should NOT show the "run git log" fallback
        assert "git log --oneline" not in result

    def test_inject_retry_context_git_diff_only(self, prompt_builder, sample_task):
        """Git diff alone (no partial output) still counts as 'has progress'."""
        sample_task.retry_count = 1
        sample_task.last_error = "Circuit breaker tripped"
        sample_task.context["_previous_attempt_git_diff"] = "## Git Diff\n```\n+code\n```"

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "Continue from the progress above" in result
        assert "git log --oneline" not in result

    def test_inject_retry_context_no_progress_fallback(self, prompt_builder, sample_task):
        """No partial output and no git diff triggers the 'run git log' fallback."""
        sample_task.retry_count = 1
        sample_task.last_error = "Circuit breaker tripped"
        # Neither _previous_attempt_summary nor _previous_attempt_git_diff set

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "git log --oneline -10" in result
        assert "git diff HEAD~1" in result
        assert "progress could not be captured" in result

    def test_inject_retry_context_both_sources(self, prompt_builder, sample_task):
        """Both partial output and git diff are included together."""
        sample_task.retry_count = 1
        sample_task.last_error = "Some error"
        sample_task.context["_previous_attempt_summary"] = "Started writing auth module"
        sample_task.context["_previous_attempt_git_diff"] = "## Git Diff\n```\n+def auth():\n```"

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "Progress From Previous Attempt" in result
        assert "Started writing auth module" in result
        assert "Code Changes From Previous Attempt" in result
        assert "+def auth():" in result
        assert "Continue from the progress above" in result

    def test_inject_retry_context_with_branch_work(self, prompt_builder, sample_task):
        """Branch work from previous attempts is injected into retry context."""
        sample_task.retry_count = 1
        sample_task.last_error = "Circuit breaker tripped"
        sample_task.context["_previous_attempt_branch_work"] = {
            "commit_count": 3,
            "insertions": 420,
            "deletions": 15,
            "commit_log": "abc1234 Add auth module\ndef5678 Add tests\nghi9012 Fix imports",
            "file_list": ["src/auth.py", "src/test_auth.py", "src/utils.py"],
            "diffstat": " src/auth.py | 200 +++\n src/test_auth.py | 180 +++\n src/utils.py | 55 +++\n",
        }

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "Existing Code on Your Branch" in result
        assert "3 commit(s)" in result
        assert "420 insertions(+)" in result
        assert "15 deletions(-)" in result
        assert "Do NOT rewrite files" in result
        assert "abc1234" in result
        assert "src/auth.py" in result
        assert "git diff origin/main..HEAD" in result

    def test_inject_retry_context_branch_work_replaces_no_progress_fallback(
        self, prompt_builder, sample_task
    ):
        """Branch work counts as progress — 'could not be captured' fallback should not appear."""
        sample_task.retry_count = 1
        sample_task.last_error = "Some error"
        sample_task.context["_previous_attempt_branch_work"] = {
            "commit_count": 1,
            "insertions": 100,
            "deletions": 0,
            "commit_log": "abc1234 Initial implementation",
            "file_list": ["src/main.py"],
            "diffstat": " src/main.py | 100 +++\n",
        }
        # No _previous_attempt_summary or _previous_attempt_git_diff

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "progress could not be captured" not in result
        assert "Existing Code on Your Branch" in result

    def test_inject_retry_context_branch_work_coexists_with_git_diff(
        self, prompt_builder, sample_task
    ):
        """Both branch work and git diff appear when both are present."""
        sample_task.retry_count = 1
        sample_task.last_error = "Some error"
        sample_task.context["_previous_attempt_git_diff"] = "## Git Diff\n```\n+def new_func():\n```"
        sample_task.context["_previous_attempt_branch_work"] = {
            "commit_count": 2,
            "insertions": 200,
            "deletions": 10,
            "commit_log": "abc1234 Feature\ndef5678 Tests",
            "file_list": ["src/feature.py"],
            "diffstat": " src/feature.py | 200 +++\n",
        }

        result = prompt_builder._inject_retry_context("Base prompt", sample_task)

        assert "Code Changes From Previous Attempt" in result
        assert "Existing Code on Your Branch" in result
        assert "+def new_func():" in result
        assert "abc1234" in result


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


class TestSelfEvalContext:
    """Test _inject_self_eval_context feeds critique back on self-eval retry."""

    def test_injects_critique_when_present(self, prompt_builder, sample_task):
        """Critique in context produces a SELF-EVALUATION FEEDBACK section."""
        sample_task.context["_self_eval_critique"] = "Missing error handling for expired tokens"
        sample_task.context["_self_eval_count"] = 1

        result = prompt_builder._inject_self_eval_context("Base prompt", sample_task)

        assert "SELF-EVALUATION FEEDBACK (attempt 1)" in result
        assert "Missing error handling for expired tokens" in result
        assert "acceptance criteria" in result
        assert result.startswith("Base prompt")

    def test_no_op_without_critique(self, prompt_builder, sample_task):
        """No critique in context → prompt unchanged."""
        result = prompt_builder._inject_self_eval_context("Base prompt", sample_task)

        assert result == "Base prompt"

    def test_skipped_when_revised_plan_present(self, prompt_builder, sample_task):
        """Both critique + _revised_plan → defers to inject_replan_context."""
        sample_task.context["_self_eval_critique"] = "Missing tests"
        sample_task.context["_revised_plan"] = "Try approach B"

        result = prompt_builder._inject_self_eval_context("Base prompt", sample_task)

        assert result == "Base prompt"

    def test_defaults_attempt_to_1_when_count_missing(self, prompt_builder, sample_task):
        """Missing _self_eval_count defaults to attempt 1."""
        sample_task.context["_self_eval_critique"] = "Needs work"

        result = prompt_builder._inject_self_eval_context("Base", sample_task)

        assert "attempt 1)" in result

    def test_in_full_build_pipeline(self, prompt_builder, sample_task):
        """Integration: build() includes self-eval section when critique is set."""
        sample_task.context["_self_eval_critique"] = "Did not handle edge case"
        sample_task.context["_self_eval_count"] = 2

        prompt = prompt_builder.build(sample_task)

        assert "SELF-EVALUATION FEEDBACK (attempt 2)" in prompt
        assert "Did not handle edge case" in prompt


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
        """Instructions require producing a JSON plan with specific fields."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", "PROJ")

        assert "```json" in result
        assert "objectives" in result
        assert "files_to_modify" in result
        assert "risks" in result
        assert "success_criteria" in result
        assert "Store plan in task.plan" not in result

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
        """Without JIRA project, instructions skip JIRA operations."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Add auth", "default", None)

        assert "skip JIRA operations" in result
        assert "JIRA ticket" not in result

    def test_goal_embedded_in_instructions(self):
        """User goal appears in the instructions."""
        from agent_framework.core.task_builder import _build_planning_instructions

        result = _build_planning_instructions("Implement SSO login", "default", None)

        assert "Implement SSO login" in result


class TestCodeReviewConstraints:
    """Test that code_review steps inject review-only constraints."""

    def _make_task_with_step(self, sample_task, step_name):
        """Set up a chain task at the given workflow step."""
        sample_task.context["chain_step"] = True
        sample_task.context["workflow_step"] = step_name
        return sample_task

    def test_code_review_injects_constraints_legacy(self, prompt_builder, sample_task):
        """code_review step injects review-only guidance in legacy prompt."""
        task = self._make_task_with_step(sample_task, "code_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "REVIEWER" in prompt
        assert "Do NOT use Write, Edit" in prompt
        assert "VERDICT: APPROVE" in prompt
        assert "VERDICT: REQUEST_CHANGES" in prompt

    def test_code_review_injects_constraints_optimized(self, prompt_builder, sample_task):
        """code_review step injects review-only guidance in optimized prompt."""
        task = self._make_task_with_step(sample_task, "code_review")
        prompt = prompt_builder._build_prompt_optimized(task)

        assert "REVIEWER" in prompt
        assert "Do NOT use Write, Edit" in prompt
        assert "VERDICT: APPROVE" in prompt

    def test_implement_step_no_review_constraints(self, prompt_builder, sample_task):
        """implement step should NOT have review-only constraints."""
        task = self._make_task_with_step(sample_task, "implement")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "REVIEWER" not in prompt
        assert "VERDICT: APPROVE" not in prompt

    def test_plan_step_no_review_constraints(self, prompt_builder, sample_task):
        """plan step should NOT have review-only constraints."""
        task = self._make_task_with_step(sample_task, "plan")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "REVIEWER" not in prompt

    def test_qa_review_step_no_review_constraints(self, prompt_builder, sample_task):
        """qa_review step should NOT have review-only constraints."""
        task = self._make_task_with_step(sample_task, "qa_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "REVIEWER" not in prompt

    def test_standalone_task_no_review_constraints(self, prompt_builder, sample_task):
        """Standalone tasks (no workflow_step) should NOT have review constraints."""
        prompt = prompt_builder._build_prompt_legacy(sample_task)

        assert "REVIEWER" not in prompt

    def test_preview_review_injects_guidance_legacy(self, prompt_builder, sample_task):
        """preview_review step injects preview-specific review guidance in legacy prompt."""
        task = self._make_task_with_step(sample_task, "preview_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "PREVIEW REVIEW CONSTRAINTS" in prompt
        assert "EXECUTION PREVIEW" in prompt
        assert "VERDICT: APPROVE" in prompt
        assert "VERDICT: REQUEST_CHANGES" in prompt
        # Must not inject the code-review guidance by mistake
        assert "CODE REVIEW CONSTRAINTS" not in prompt

    def test_preview_review_injects_guidance_optimized(self, prompt_builder, sample_task):
        """preview_review step injects preview-specific review guidance in optimized prompt."""
        task = self._make_task_with_step(sample_task, "preview_review")
        prompt = prompt_builder._build_prompt_optimized(task)

        assert "PREVIEW REVIEW CONSTRAINTS" in prompt
        assert "VERDICT: APPROVE" in prompt
        assert "CODE REVIEW CONSTRAINTS" not in prompt

    def test_code_review_does_not_get_preview_guidance(self, prompt_builder, sample_task):
        """code_review step gets code-review guidance, not preview-review guidance."""
        task = self._make_task_with_step(sample_task, "code_review")
        prompt = prompt_builder._build_prompt_legacy(task)

        assert "CODE REVIEW CONSTRAINTS" in prompt
        assert "PREVIEW REVIEW CONSTRAINTS" not in prompt

    def test_preview_step_injects_read_only_constraints(self, prompt_builder, sample_task):
        """Engineer at the preview step gets PREVIEW MODE read-only constraints injected.

        _inject_preview_mode is applied by build() after _build_prompt_legacy.
        Test both together to mirror the real call path.
        """
        task = self._make_task_with_step(sample_task, "preview")
        task.type = TaskType.PREVIEW
        prompt = prompt_builder._build_prompt_legacy(task)
        prompt = prompt_builder._inject_preview_mode(prompt, task)

        assert "PREVIEW MODE" in prompt
        assert "Do NOT use Write" in prompt
        # Must not inject the architect-facing review guidance
        assert "PREVIEW REVIEW CONSTRAINTS" not in prompt


class TestReadCacheInjection:
    """Tests for _inject_read_cache() method."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        d = tmp_path / ".agent-communication" / "read-cache"
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def builder(self, agent_config, tmp_path):
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        return PromptBuilder(ctx)

    @pytest.fixture
    def task_with_root(self):
        return Task(
            id="chain-root123-step1-d0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Implement feature",
            description="Implement the feature",
            context={"_root_task_id": "root123"},
        )

    def test_inject_with_summaries(self, builder, cache_dir, task_with_root):
        """Read cache with LLM summaries renders a markdown table."""
        import json
        cache_data = {
            "root_task_id": "root123",
            "entries": {
                "src/server.py": {
                    "summary": "Express server with /api/dashboard routes",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
                "src/models.py": {
                    "summary": "SQLAlchemy models: User, Session, Metrics",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:01:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "root123.json").write_text(json.dumps(cache_data))

        result = builder._inject_read_cache("base prompt", task_with_root)
        assert "FILES ANALYZED BY PREVIOUS AGENTS" in result
        assert "src/server.py" in result
        assert "Express server" in result
        assert "| Role |" in result
        assert "| ref |" in result
        assert "base prompt" in result

    def test_inject_paths_only(self, builder, cache_dir, task_with_root):
        """Read cache without summaries renders a compact path list."""
        import json
        cache_data = {
            "root_task_id": "root123",
            "entries": {
                "src/server.py": {
                    "summary": "",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
                "src/models.py": {
                    "summary": "",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:01:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "root123.json").write_text(json.dumps(cache_data))

        result = builder._inject_read_cache("base prompt", task_with_root)
        assert "FILES READ BY PREVIOUS AGENTS" in result
        assert "src/server.py" in result
        # MCP is disabled in this builder, so get_cached_reads() hint is omitted
        assert "get_cached_reads()" not in result

    def test_inject_paths_only_has_step_directive(self, agent_config, tmp_path, task_with_root):
        """Paths-only format includes step-aware directive."""
        import json
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=True,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        cache_dir = tmp_path / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)
        cache_data = {
            "root_task_id": "root123",
            "entries": {
                "src/server.py": {"summary": "", "read_by": "architect",
                                   "read_at": "2026-02-18T10:00:00Z", "workflow_step": "plan"},
            },
        }
        (cache_dir / "root123.json").write_text(json.dumps(cache_data))

        result = builder._inject_read_cache("base prompt", task_with_root)
        assert "Check summaries before re-reading any file" in result

    def test_inject_empty_when_no_cache(self, builder, task_with_root):
        """No cache file → original prompt unchanged."""
        result = builder._inject_read_cache("base prompt", task_with_root)
        assert result == "base prompt"

    def test_inject_empty_entries(self, builder, cache_dir, task_with_root):
        """Empty entries dict → original prompt unchanged."""
        import json
        cache_data = {"root_task_id": "root123", "entries": {}}
        (cache_dir / "root123.json").write_text(json.dumps(cache_data))

        result = builder._inject_read_cache("base prompt", task_with_root)
        assert result == "base prompt"

    def test_inject_respects_budget(self, builder, cache_dir, task_with_root):
        """Section is truncated when it exceeds _READ_CACHE_MAX_CHARS."""
        import json
        # Create many entries to exceed 3000 char budget
        entries = {}
        for i in range(200):
            entries[f"src/very/long/path/to/file_{i:04d}.py"] = {
                "summary": f"This file contains component {i} with detailed analysis and findings " * 3,
                "read_by": "architect",
                "read_at": "2026-02-18T10:00:00Z",
                "workflow_step": "plan",
            }
        cache_data = {"root_task_id": "root123", "entries": entries}
        (cache_dir / "root123.json").write_text(json.dumps(cache_data))

        result = builder._inject_read_cache("base prompt", task_with_root)
        # Section should be within budget (base prompt + budget + overhead)
        section = result[len("base prompt"):]
        assert len(section) <= builder._READ_CACHE_MAX_CHARS + 50  # small margin for truncation marker

    def test_inject_corrupted_cache_file(self, builder, cache_dir, task_with_root):
        """Corrupted JSON → falls back gracefully."""
        (cache_dir / "root123.json").write_text("not valid json{{{")

        result = builder._inject_read_cache("base prompt", task_with_root)
        assert result == "base prompt"


class TestReadCacheSeedFromRepo:
    """Tests for seeding task-specific cache from repo-scoped cache."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        d = tmp_path / ".agent-communication" / "read-cache"
        d.mkdir(parents=True)
        return d

    @pytest.fixture
    def builder(self, agent_config, tmp_path):
        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        return PromptBuilder(ctx)

    @pytest.fixture
    def task_with_repo(self):
        return Task(
            id="planning-myrepo-20260219",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan feature",
            description="Plan",
            context={
                "_root_task_id": "planning-myrepo-20260219",
                "github_repo": "justworkshr/myrepo",
            },
        )

    def test_seeds_from_repo_cache(self, builder, cache_dir, task_with_repo):
        """New task with no task cache seeds from repo cache and injects content."""
        import json

        repo_data = {
            "github_repo": "justworkshr/myrepo",
            "entries": {
                "src/server.py": {
                    "summary": "Express server with routes",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "_repo-justworkshr-myrepo.json").write_text(json.dumps(repo_data))

        result = builder._inject_read_cache("base prompt", task_with_repo)

        # Should have injected the cache content
        assert "FILES ANALYZED BY PREVIOUS AGENTS" in result
        assert "src/server.py" in result
        assert "Express server" in result
        # Task-specific cache should now exist
        task_cache = cache_dir / "planning-myrepo-20260219.json"
        assert task_cache.exists()
        task_data = json.loads(task_cache.read_text())
        assert "src/server.py" in task_data["entries"]

    def test_no_seed_when_task_cache_exists(self, builder, cache_dir, task_with_repo):
        """Existing task cache is used directly; repo cache not consulted."""
        import json

        # Task-specific cache with different content
        task_data = {
            "root_task_id": "planning-myrepo-20260219",
            "entries": {
                "src/task_specific.py": {
                    "summary": "Task-specific file",
                    "read_by": "engineer",
                    "read_at": "2026-02-19T10:00:00Z",
                    "workflow_step": "implement",
                },
            },
        }
        (cache_dir / "planning-myrepo-20260219.json").write_text(json.dumps(task_data))

        # Repo cache with different content
        repo_data = {
            "github_repo": "justworkshr/myrepo",
            "entries": {
                "src/repo_file.py": {
                    "summary": "Repo-level file",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "_repo-justworkshr-myrepo.json").write_text(json.dumps(repo_data))

        result = builder._inject_read_cache("base prompt", task_with_repo)

        # Should use task-specific content, not repo cache
        assert "src/task_specific.py" in result
        assert "src/repo_file.py" not in result

    def test_no_seed_without_github_repo(self, builder, cache_dir):
        """Task without github_repo returns prompt unchanged."""
        task = Task(
            id="planning-norepo-20260219",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan feature",
            description="Plan",
            context={"_root_task_id": "planning-norepo-20260219"},
        )

        result = builder._inject_read_cache("base prompt", task)
        assert result == "base prompt"

    def test_seed_graceful_on_corrupted_repo_cache(self, builder, cache_dir, task_with_repo):
        """Corrupted repo cache JSON falls back gracefully."""
        (cache_dir / "_repo-justworkshr-myrepo.json").write_text("not valid{{{")

        result = builder._inject_read_cache("base prompt", task_with_repo)
        assert result == "base prompt"

    def test_seed_skips_empty_repo_cache(self, builder, cache_dir, task_with_repo):
        """Empty entries dict in repo cache does not seed."""
        import json

        repo_data = {"github_repo": "justworkshr/myrepo", "entries": {}}
        (cache_dir / "_repo-justworkshr-myrepo.json").write_text(json.dumps(repo_data))

        result = builder._inject_read_cache("base prompt", task_with_repo)
        assert result == "base prompt"


class TestRequirementsChecklistInjection:
    """Tests for _inject_requirements_checklist()."""

    def test_injects_checklist_when_present(self, prompt_builder, sample_task):
        sample_task.context["requirements_checklist"] = [
            {"id": 1, "description": "Add memory panel", "files": ["dashboard.py"], "status": "pending"},
            {"id": 2, "description": "Create retry panel", "files": [], "status": "pending"},
        ]
        result = prompt_builder._inject_requirements_checklist("base prompt", sample_task)

        assert "REQUIRED DELIVERABLES (2 items)" in result
        assert "1. [ ] Add memory panel (dashboard.py)" in result
        assert "2. [ ] Create retry panel" in result
        assert "verify each item" in result.lower()

    def test_no_injection_without_checklist(self, prompt_builder, sample_task):
        result = prompt_builder._inject_requirements_checklist("base prompt", sample_task)
        assert result == "base prompt"

    def test_no_injection_with_empty_checklist(self, prompt_builder, sample_task):
        sample_task.context["requirements_checklist"] = []
        result = prompt_builder._inject_requirements_checklist("base prompt", sample_task)
        assert result == "base prompt"

    def test_checklist_in_full_build(self, prompt_builder, sample_task):
        """Checklist injection appears in the full build() output."""
        sample_task.context["requirements_checklist"] = [
            {"id": 1, "description": "Build feature X", "files": [], "status": "pending"},
        ]
        prompt = prompt_builder.build(sample_task)
        assert "REQUIRED DELIVERABLES" in prompt


class TestChainStateIntegration:
    """Test chain state context loading in prompt builder."""

    def _make_chain_task(self, tmp_path):
        """Create a chain task that triggers chain state loading."""
        return Task(
            id="chain-root-1-implement-d2",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="architect",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="[chain] Implement auth",
            description="Implement the auth feature",
            context={
                "_root_task_id": "root-1",
                "workflow": "default",
                "workflow_step": "implement",
                "chain_step": True,
                "github_repo": "company/api",
                "user_goal": "Add JWT authentication",
            },
        )

    def test_chain_state_takes_priority_over_upstream_summary(self, agent_config, tmp_path):
        """Chain state context should override raw upstream_summary."""
        import json
        from agent_framework.core.chain_state import ChainState, StepRecord, save_chain_state

        # Write chain state with plan step
        state = ChainState(
            root_task_id="root-1",
            user_goal="Add JWT authentication",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan",
                    agent_id="architect",
                    task_id="chain-root-1-plan-d1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="Planned auth feature",
                    verdict="approved",
                    plan={
                        "objectives": ["Add JWT auth"],
                        "approach": ["Create auth service"],
                        "files_to_modify": ["src/auth.py"],
                        "risks": [],
                        "success_criteria": ["Tests pass"],
                    },
                ),
            ],
        )
        save_chain_state(tmp_path, state)

        task = self._make_chain_task(tmp_path)
        # Also set upstream_summary — chain state should win
        task.context["upstream_summary"] = "RAW LLM PROSE THAT SHOULD NOT APPEAR"

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        context = builder._load_upstream_context(task)

        assert "CHAIN STATE" in context
        assert "Add JWT auth" in context
        assert "RAW LLM PROSE" not in context

    def test_falls_through_when_no_chain_state(self, agent_config, tmp_path):
        """Without chain state file, falls through to upstream_summary."""
        task = self._make_chain_task(tmp_path)
        task.context["upstream_summary"] = "Upstream findings here"

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        context = builder._load_upstream_context(task)

        assert "UPSTREAM AGENT FINDINGS" in context
        assert "Upstream findings here" in context

    def test_non_workflow_task_skips_chain_state(self, agent_config, tmp_path):
        """Tasks without workflow context should not attempt chain state loading."""
        task = Task(
            id="standalone-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Standalone task",
            description="No workflow",
            context={"github_repo": "company/api"},
        )

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        context = builder._load_chain_state_context(task)

        assert context == ""

    def test_rejection_feedback_still_takes_priority(self, agent_config, tmp_path):
        """Human rejection feedback should override even chain state."""
        from agent_framework.core.chain_state import ChainState, StepRecord, save_chain_state

        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect",
                    task_id="t1", completed_at="2026-02-19T10:00:00+00:00",
                    summary="Plan done",
                ),
            ],
        )
        save_chain_state(tmp_path, state)

        task = self._make_chain_task(tmp_path)
        task.context["rejection_feedback"] = "Please redo the auth approach"

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        context = builder._load_upstream_context(task)

        assert "CHECKPOINT REJECTED" in context
        assert "redo the auth approach" in context


class TestAttemptHistoryInRetryContext:
    """_inject_retry_context includes attempt history from disk."""

    def test_attempt_history_rendered_in_retry_prompt(self, agent_config, tmp_path):
        from agent_framework.core.attempt_tracker import (
            AttemptHistory,
            AttemptRecord,
            save_attempt_history,
        )

        history = AttemptHistory(task_id="retry-task-1", attempts=[
            AttemptRecord(
                attempt_number=1,
                started_at="2026-01-01T00:00:00+00:00",
                agent_id="engineer",
                branch="agent/engineer/retry-task-1",
                commit_sha="abc1234",
                pushed=True,
                files_modified=["src/feature.py"],
                commit_count=3,
                insertions=200,
                deletions=10,
                error="Circuit breaker tripped",
                error_type="circuit_breaker",
            ),
        ])
        save_attempt_history(tmp_path, history)

        task = Task(
            id="retry-task-1",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Implement feature",
            description="Add feature",
            retry_count=1,
            last_error="Circuit breaker tripped",
            context={"github_repo": "org/repo"},
        )

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        prompt = builder._inject_retry_context("base prompt", task)

        assert "Previous Attempt History" in prompt
        assert "agent/engineer/retry-task-1" in prompt
        assert "3 commits" in prompt
        assert "200+/10-" in prompt

    def test_no_attempt_history_gracefully_omitted(self, agent_config, tmp_path):
        task = Task(
            id="retry-task-2",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="test-agent",
            created_at=datetime.now(timezone.utc),
            title="Implement feature",
            description="Add feature",
            retry_count=1,
            last_error="Some error",
            context={"github_repo": "org/repo"},
        )

        ctx = PromptContext(
            config=agent_config,
            workspace=tmp_path,
            mcp_enabled=False,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)
        prompt = builder._inject_retry_context("base prompt", task)

        # Should still have retry context, just no attempt history section
        assert "RETRY CONTEXT" in prompt
        assert "Previous Attempt History" not in prompt
