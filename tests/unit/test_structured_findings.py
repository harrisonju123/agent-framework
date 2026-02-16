"""Tests for structured findings parsing and formatting."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.review_cycle import QAFinding, ReviewOutcome
from agent_framework.core.task import Task, TaskStatus, TaskType


@pytest.fixture
def agent():
    """Create a minimal ReviewCycleManager instance for testing.

    Note: The test calls agent.parse_structured_findings() which now lives
    in ReviewCycleManager. For backward compatibility with tests, we create
    a mock object that has the ReviewCycleManager methods.
    """
    from agent_framework.core.review_cycle import ReviewCycleManager

    config = AgentConfig(
        id="qa",
        name="QA",
        queue="qa",
        prompt="Test",
    )

    # Create a ReviewCycleManager with minimal mocked dependencies
    manager = ReviewCycleManager(
        config=config,
        queue=MagicMock(),
        logger=MagicMock(),
        agent_definition=None,
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )

    # Return the manager (tests call agent.parse_structured_findings, etc.)
    return manager


class TestParseStructuredFindings:
    """Tests for parse_structured_findings() method."""

    def test_parses_code_fence_json_with_findings_wrapper(self, agent):
        """Should extract findings from JSON code fence with 'findings' wrapper."""
        content = '''Review complete.

```json
{
  "findings": [
    {
      "id": "finding-1",
      "severity": "CRITICAL",
      "category": "security",
      "file": "auth.py",
      "line": 42,
      "description": "SQL injection vulnerability",
      "suggested_fix": "Use parameterized queries"
    }
  ],
  "summary": "Found 1 critical issue",
  "total_count": 1,
  "critical_count": 1,
  "high_count": 0,
  "major_count": 0
}
```

Please fix.'''
        result = agent.parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"
        assert result[0].file == "auth.py"
        assert result[0].line_number == 42
        assert result[0].description == "SQL injection vulnerability"
        assert result[0].suggested_fix == "Use parameterized queries"

    def test_parses_code_fence_json_array_format(self, agent):
        """Should extract findings from JSON array in code fence."""
        content = '''Found issues:

```json
[
  {
    "file": "utils.py",
    "line_number": 10,
    "severity": "HIGH",
    "category": "performance",
    "description": "Inefficient loop",
    "suggested_fix": "Use list comprehension"
  }
]
```
'''
        result = agent.parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].severity == "HIGH"
        assert result[0].file == "utils.py"

    def test_parses_inline_json_with_findings_key(self, agent):
        """Should extract findings from inline JSON object."""
        content = '''Review result: {"findings": [{"file": "test.py", "line": 5, "severity": "MEDIUM", "category": "style", "description": "Missing docstring", "suggested_fix": "Add docstring"}], "total_count": 1}'''
        result = agent.parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].severity == "MEDIUM"
        assert result[0].description == "Missing docstring"

    def test_returns_none_for_text_only(self, agent):
        """Should return None for non-JSON content."""
        content = "CRITICAL: SQL injection in auth.py:42"
        result = agent.parse_structured_findings(content)
        assert result is None

    def test_returns_none_for_invalid_json(self, agent):
        """Should return None for malformed JSON."""
        content = '''```json
{
  "findings": [
    {invalid json here
  ]
}
```'''
        result = agent.parse_structured_findings(content)
        assert result is None

    def test_handles_multiple_findings(self, agent):
        """Should parse multiple findings from JSON."""
        content = '''```json
{
  "findings": [
    {
      "file": "a.py",
      "line_number": 1,
      "severity": "CRITICAL",
      "category": "security",
      "description": "Issue 1",
      "suggested_fix": "Fix 1"
    },
    {
      "file": "b.py",
      "line_number": 2,
      "severity": "HIGH",
      "category": "correctness",
      "description": "Issue 2",
      "suggested_fix": "Fix 2"
    }
  ]
}
```'''
        result = agent.parse_structured_findings(content)
        assert result is not None
        assert len(result) == 2
        assert result[0].file == "a.py"
        assert result[1].file == "b.py"

    def test_handles_missing_optional_fields(self, agent):
        """Should handle findings with missing optional fields."""
        content = '''```json
{
  "findings": [
    {
      "severity": "LOW",
      "category": "style",
      "description": "Minor style issue"
    }
  ]
}
```'''
        result = agent.parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].file == ""
        assert result[0].line_number is None
        assert result[0].suggested_fix is None


class TestFormatFindingsChecklist:
    """Tests for format_findings_checklist() method."""

    def test_formats_with_file_and_line(self, agent):
        """Should format finding with file and line number."""
        findings = [
            QAFinding(
                file="auth.py",
                line_number=42,
                severity="CRITICAL",
                category="security",
                description="SQL injection",
                suggested_fix="Use parameterized queries",
            )
        ]
        result = agent.format_findings_checklist(findings)
        assert "1. 🔴 CRITICAL: Security (auth.py:42)" in result
        assert "**Issue**: SQL injection" in result
        assert "**Suggested Fix**: Use parameterized queries" in result

    def test_formats_with_file_only(self, agent):
        """Should format finding with file but no line number."""
        findings = [
            QAFinding(
                file="utils.py",
                line_number=None,
                severity="HIGH",
                category="performance",
                description="Slow function",
                suggested_fix=None,
            )
        ]
        result = agent.format_findings_checklist(findings)
        assert "1. 🟠 HIGH: Performance (utils.py)" in result
        assert "**Issue**: Slow function" in result
        assert "**Suggested Fix**" not in result

    def test_formats_without_location(self, agent):
        """Should format finding without file or line."""
        findings = [
            QAFinding(
                file="",
                line_number=None,
                severity="MEDIUM",
                category="testing",
                description="Missing tests",
                suggested_fix="Add unit tests",
            )
        ]
        result = agent.format_findings_checklist(findings)
        assert "1. 🔵 MEDIUM: Testing" in result
        assert "auth.py" not in result

    def test_formats_multiple_findings(self, agent):
        """Should format multiple findings with proper numbering."""
        findings = [
            QAFinding(
                file="a.py",
                line_number=1,
                severity="CRITICAL",
                category="security",
                description="Issue 1",
                suggested_fix="Fix 1",
            ),
            QAFinding(
                file="b.py",
                line_number=2,
                severity="MAJOR",
                category="correctness",
                description="Issue 2",
                suggested_fix="Fix 2",
            ),
            QAFinding(
                file="c.py",
                line_number=3,
                severity="SUGGESTION",
                category="style",
                description="Issue 3",
                suggested_fix=None,
            ),
        ]
        result = agent.format_findings_checklist(findings)
        assert "1. 🔴 CRITICAL" in result
        assert "2. 🟡 MAJOR" in result
        assert "3. 💡 SUGGESTION" in result

    def test_formats_all_severity_levels(self, agent):
        """Should use correct emoji for each severity level."""
        severities = [
            ("CRITICAL", "🔴"),
            ("HIGH", "🟠"),
            ("MAJOR", "🟡"),
            ("MEDIUM", "🔵"),
            ("MINOR", "⚪"),
            ("LOW", "⚪"),
            ("SUGGESTION", "💡"),
        ]
        for severity, expected_emoji in severities:
            findings = [
                QAFinding(
                    file="test.py",
                    line_number=1,
                    severity=severity,
                    category="general",
                    description="Test issue",
                    suggested_fix=None,
                )
            ]
            result = agent.format_findings_checklist(findings)
            assert expected_emoji in result, f"Missing emoji for {severity}"


class TestExtractReviewFindingsIntegration:
    """Integration tests for extract_review_findings() using new parser."""

    def test_uses_new_parser_for_code_fence(self, agent):
        """Should use new parser for code fence JSON."""
        content = '''```json
{
  "findings": [
    {
      "file": "test.py",
      "line": 1,
      "severity": "HIGH",
      "category": "bug",
      "description": "Logic error",
      "suggested_fix": "Fix logic"
    }
  ]
}
```'''
        summary, findings = agent.extract_review_findings(content)
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"
        assert "test.py:1" in summary

    def test_falls_back_to_legacy_for_text(self, agent):
        """Should fall back to legacy regex for text-only content."""
        content = """CRITICAL: SQL injection in auth.py:42
HIGH: Memory leak in cache.py:100"""
        summary, findings = agent.extract_review_findings(content)
        assert "CRITICAL: SQL injection" in summary
        assert "HIGH: Memory leak" in summary

    def test_builds_summary_from_structured_findings(self, agent):
        """Should build summary from structured findings."""
        content = '''```json
{
  "findings": [
    {
      "file": "auth.py",
      "line_number": 42,
      "severity": "CRITICAL",
      "category": "security",
      "description": "SQL injection",
      "suggested_fix": "Use ORM"
    }
  ]
}
```'''
        summary, findings = agent.extract_review_findings(content)
        assert len(findings) == 1
        assert "CRITICAL: SQL injection (auth.py:42)" in summary


def _make_task(
    task_type=TaskType.REVIEW,
    assigned_to="qa",
    task_id="review-task-abc123",
    **ctx_overrides,
):
    """Helper to create test tasks."""
    context = {
        "jira_key": "PROJ-42",
        "pr_url": "https://github.com/org/repo/pull/99",
        "pr_number": 99,
        "github_repo": "org/repo",
        "workflow": "standard",
        **ctx_overrides,
    }
    return Task(
        id=task_id,
        type=task_type,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="engineer",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Review PR #99",
        description="Review the PR.",
        context=context,
    )


class TestBuildReviewFixTask:
    """Tests for build_review_fix_task() with structured findings."""

    def test_structured_findings_in_context(self, agent):
        """Structured findings are stored in fix task context."""
        task = _make_task()

        # Simulate structured findings JSON in outcome
        findings_json = json.dumps({
            "findings": [
                {
                    "id": "f1",
                    "severity": "CRITICAL",
                    "category": "security",
                    "file": "auth.py",
                    "line": 42,
                    "description": "SQL injection",
                    "suggested_fix": "Use parameterized queries",
                    "resolved": False,
                }
            ],
            "summary": "Found 1 critical issue",
            "total_count": 1,
            "critical_count": 1,
            "high_count": 0,
            "major_count": 0,
        })

        # Create structured findings for outcome
        structured_findings = [
            QAFinding(
                file="auth.py",
                line_number=42,
                severity="CRITICAL",
                category="security",
                description="SQL injection",
                suggested_fix="Use parameterized queries",
            )
        ]

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            findings_summary=findings_json,
            structured_findings=structured_findings,
        )

        fix_task = agent.build_review_fix_task(task, outcome, cycle_count=1)

        # Verify structured_findings in context
        assert "structured_findings" in fix_task.context
        assert fix_task.context["structured_findings"]["total_count"] == 1
        assert len(fix_task.context["structured_findings"]["findings"]) == 1

        # Verify description has checklist
        assert "### 1. 🔴 CRITICAL" in fix_task.description
        assert "auth.py:42" in fix_task.description
        assert "SQL injection" in fix_task.description

        # Verify acceptance criteria
        assert "All 1 issues addressed" in fix_task.acceptance_criteria[0]

    def test_legacy_format_still_works(self, agent):
        """Legacy text format works when no JSON present."""
        task = _make_task()

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            findings_summary="CRITICAL: SQL injection in auth.py:42",
        )

        fix_task = agent.build_review_fix_task(task, outcome, cycle_count=1)

        # No structured_findings in context
        assert "structured_findings" not in fix_task.context

        # Legacy description format
        assert "## Review Findings" in fix_task.description
        assert "CRITICAL: SQL injection" in fix_task.description

        # Generic acceptance criteria
        assert "All identified issues addressed" in fix_task.acceptance_criteria[0]

    def test_multiple_findings_with_counts(self, agent):
        """Multiple findings with correct counts and checklist."""
        task = _make_task()

        findings_json = json.dumps({
            "findings": [
                {
                    "id": "f1",
                    "severity": "CRITICAL",
                    "category": "security",
                    "file": "auth.py",
                    "line": 42,
                    "description": "SQL injection",
                    "suggested_fix": "Use parameterized queries",
                    "resolved": False,
                },
                {
                    "id": "f2",
                    "severity": "HIGH",
                    "category": "performance",
                    "file": "api.py",
                    "line": 10,
                    "description": "N+1 query",
                    "suggested_fix": "Use select_related",
                    "resolved": False,
                },
                {
                    "id": "f3",
                    "severity": "MAJOR",
                    "category": "correctness",
                    "file": "utils.py",
                    "line": 5,
                    "description": "Off-by-one error",
                    "suggested_fix": "Use <= instead of <",
                    "resolved": False,
                },
            ],
            "summary": "Found 3 issues",
            "total_count": 3,
            "critical_count": 1,
            "high_count": 1,
            "major_count": 1,
        })

        structured_findings = [
            QAFinding(
                file="auth.py",
                line_number=42,
                severity="CRITICAL",
                category="security",
                description="SQL injection",
                suggested_fix="Use parameterized queries",
            ),
            QAFinding(
                file="api.py",
                line_number=10,
                severity="HIGH",
                category="performance",
                description="N+1 query",
                suggested_fix="Use select_related",
            ),
            QAFinding(
                file="utils.py",
                line_number=5,
                severity="MAJOR",
                category="correctness",
                description="Off-by-one error",
                suggested_fix="Use <= instead of <",
            ),
        ]

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=True,
            has_test_failures=False,
            has_change_requests=False,
            has_major_issues=True,
            findings_summary=findings_json,
            structured_findings=structured_findings,
        )

        fix_task = agent.build_review_fix_task(task, outcome, cycle_count=1)

        # Verify count in description and acceptance criteria
        assert "QA review found 3 issue(s)" in fix_task.description
        assert "All 3 issues addressed" in fix_task.acceptance_criteria[0]

        # Verify all findings in checklist
        assert "### 1. 🔴 CRITICAL" in fix_task.description
        assert "### 2. 🟠 HIGH" in fix_task.description
        assert "### 3. 🟡 MAJOR" in fix_task.description

    def test_context_preserves_essential_fields(self, agent):
        """Fix task context preserves essential fields and strips review_* keys."""
        task = _make_task(
            review_started_at="2024-01-01",
            custom_field="should_keep",
        )

        outcome = ReviewOutcome(
            approved=False,
            has_critical_issues=False,
            has_test_failures=True,
            has_change_requests=False,
            findings_summary="Tests failed",
        )

        fix_task = agent.build_review_fix_task(task, outcome, cycle_count=2)

        # Essential fields preserved
        assert fix_task.context["pr_url"] == "https://github.com/org/repo/pull/99"
        assert fix_task.context["pr_number"] == 99
        assert fix_task.context["jira_key"] == "PROJ-42"
        assert fix_task.context["github_repo"] == "org/repo"
        assert fix_task.context["workflow"] == "standard"
        assert fix_task.context["_review_cycle_count"] == 2

        # Custom field preserved
        assert fix_task.context["custom_field"] == "should_keep"

        # review_* fields stripped
        assert "review_started_at" not in fix_task.context
