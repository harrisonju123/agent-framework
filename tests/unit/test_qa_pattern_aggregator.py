"""Tests for QAPatternAggregator — recurring pattern detection and warning formatting."""

import pytest

from agent_framework.core.qa_pattern_aggregator import (
    MAX_WARNING_CHARS,
    QAPatternAggregator,
    RECURRENCE_THRESHOLD,
)
from agent_framework.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def aggregator(store):
    return QAPatternAggregator(memory_store=store, repo_slug="org/repo")


REPO = "org/repo"


def _add_qa_finding(store, content, task_id, severity="HIGH", category="security", file_path=None):
    """Helper to add a QA finding to memory."""
    tags = [f"severity:{severity}", f"category:{category}"]
    if file_path:
        tags.append(f"file:{file_path}")
    store.remember(
        repo_slug=REPO,
        agent_type="shared",
        category="qa_findings",
        content=content,
        source_task_id=task_id,
        tags=tags,
    )


class TestRecurringPatternDetection:
    def test_no_patterns_when_below_threshold(self, store, aggregator):
        """Findings from fewer than RECURRENCE_THRESHOLD tasks aren't flagged."""
        _add_qa_finding(store, "SQL injection risk in query builder", "task-1")
        _add_qa_finding(store, "SQL injection risk in query builder", "task-2")

        patterns = aggregator.get_recurring_patterns()
        assert len(patterns) == 0

    def test_detects_pattern_at_threshold(self, store, aggregator):
        """Findings from exactly RECURRENCE_THRESHOLD tasks are flagged."""
        for i in range(RECURRENCE_THRESHOLD):
            _add_qa_finding(
                store,
                f"SQL injection risk in query builder (task {i})",
                f"task-{i}",
            )

        patterns = aggregator.get_recurring_patterns()
        assert len(patterns) >= 1
        assert patterns[0].occurrence_count >= RECURRENCE_THRESHOLD

    def test_groups_normalized_content(self, store, aggregator):
        """Findings with different file paths but same description are grouped."""
        _add_qa_finding(store, "src/foo.py:10 missing input validation", "task-1", file_path="src/foo.py")
        _add_qa_finding(store, "src/bar.py:20 missing input validation", "task-2", file_path="src/bar.py")
        _add_qa_finding(store, "src/baz.py:30 missing input validation", "task-3", file_path="src/baz.py")

        patterns = aggregator.get_recurring_patterns()
        assert len(patterns) >= 1

    def test_returns_empty_for_disabled_store(self, tmp_path):
        disabled_store = MemoryStore(workspace=tmp_path, enabled=False)
        agg = QAPatternAggregator(memory_store=disabled_store, repo_slug=REPO)
        assert agg.get_recurring_patterns() == []

    def test_returns_empty_for_no_findings(self, aggregator):
        assert aggregator.get_recurring_patterns() == []


class TestWarningsForFiles:
    def test_filters_by_file_path(self, store, aggregator):
        for i in range(3):
            _add_qa_finding(
                store,
                f"Missing validation in handler (task {i})",
                f"task-{i}",
                file_path="src/handlers/auth.py",
            )

        relevant = aggregator.get_warnings_for_files(["src/handlers/auth.py"])
        assert len(relevant) >= 1

    def test_includes_patterns_without_file(self, store, aggregator):
        for i in range(3):
            _add_qa_finding(
                store,
                f"Generic security issue (task {i})",
                f"task-{i}",
            )

        relevant = aggregator.get_warnings_for_files(["any/file.py"])
        # Patterns without file_pattern should be included
        assert len(relevant) >= 1

    def test_returns_all_when_no_target_files(self, store, aggregator):
        for i in range(3):
            _add_qa_finding(store, f"Issue (task {i})", f"task-{i}")

        relevant = aggregator.get_warnings_for_files([])
        # Empty target_files returns all patterns
        assert len(relevant) >= 1


class TestFormatWarnings:
    def test_empty_patterns_returns_empty_string(self, aggregator):
        assert aggregator.format_warnings_section([]) == ""

    def test_format_includes_header(self, store, aggregator):
        for i in range(3):
            _add_qa_finding(store, f"Test issue (task {i})", f"task-{i}")

        patterns = aggregator.get_recurring_patterns()
        result = aggregator.format_warnings_section(patterns)
        assert "## RECURRING QA WARNINGS" in result

    def test_format_truncates_at_max_chars(self, store, aggregator):
        for i in range(3):
            _add_qa_finding(store, f"Very long finding description that should be truncated (task {i})", f"task-{i}")

        patterns = aggregator.get_recurring_patterns()
        result = aggregator.format_warnings_section(patterns, max_chars=100)
        assert len(result) <= 100


class TestNormalization:
    def test_strips_line_numbers(self):
        assert ":123" not in QAPatternAggregator._normalize_finding("error at file.py:123")

    def test_strips_file_paths(self):
        result = QAPatternAggregator._normalize_finding("src/handlers/auth.py has an issue")
        assert "src/handlers" not in result

    def test_collapses_whitespace(self):
        result = QAPatternAggregator._normalize_finding("too   many    spaces")
        assert "  " not in result


class TestSeverityRanking:
    def test_highest_severity_returns_critical(self):
        assert QAPatternAggregator._highest_severity(["LOW", "CRITICAL", "HIGH"]) == "CRITICAL"

    def test_highest_severity_empty_returns_medium(self):
        assert QAPatternAggregator._highest_severity([]) == "MEDIUM"
