"""Tests for structured findings parsing and formatting."""

from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent, AgentConfig, QAFinding


@pytest.fixture
def agent():
    """Create a minimal Agent instance for testing."""
    config = AgentConfig(
        id="qa",
        name="QA",
        queue="qa",
        prompt="Test",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.logger = MagicMock()
    return a


class TestParseStructuredFindings:
    """Tests for _parse_structured_findings() method."""

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
        result = agent._parse_structured_findings(content)
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
        result = agent._parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].severity == "HIGH"
        assert result[0].file == "utils.py"

    def test_parses_inline_json_with_findings_key(self, agent):
        """Should extract findings from inline JSON object."""
        content = '''Review result: {"findings": [{"file": "test.py", "line": 5, "severity": "MEDIUM", "category": "style", "description": "Missing docstring", "suggested_fix": "Add docstring"}], "total_count": 1}'''
        result = agent._parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].severity == "MEDIUM"
        assert result[0].description == "Missing docstring"

    def test_returns_none_for_text_only(self, agent):
        """Should return None for non-JSON content."""
        content = "CRITICAL: SQL injection in auth.py:42"
        result = agent._parse_structured_findings(content)
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
        result = agent._parse_structured_findings(content)
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
        result = agent._parse_structured_findings(content)
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
        result = agent._parse_structured_findings(content)
        assert result is not None
        assert len(result) == 1
        assert result[0].file == ""
        assert result[0].line_number is None
        assert result[0].suggested_fix is None


class TestFormatFindingsChecklist:
    """Tests for _format_findings_checklist() method."""

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
        result = agent._format_findings_checklist(findings)
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
        result = agent._format_findings_checklist(findings)
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
        result = agent._format_findings_checklist(findings)
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
        result = agent._format_findings_checklist(findings)
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
            result = agent._format_findings_checklist(findings)
            assert expected_emoji in result, f"Missing emoji for {severity}"


class TestExtractReviewFindingsIntegration:
    """Integration tests for _extract_review_findings() using new parser."""

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
        summary, findings = agent._extract_review_findings(content)
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"
        assert "test.py:1" in summary

    def test_falls_back_to_legacy_for_text(self, agent):
        """Should fall back to legacy regex for text-only content."""
        content = """CRITICAL: SQL injection in auth.py:42
HIGH: Memory leak in cache.py:100"""
        summary, findings = agent._extract_review_findings(content)
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
        summary, findings = agent._extract_review_findings(content)
        assert len(findings) == 1
        assert "CRITICAL: SQL injection (auth.py:42)" in summary
