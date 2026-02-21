"""Tests for QAPatternAggregator — recurring pattern detection and formatting."""

import time
from unittest.mock import MagicMock

import pytest

from agent_framework.core.qa_pattern_aggregator import (
    MAX_WARNINGS,
    MAX_WARNINGS_CHARS,
    RECURRENCE_THRESHOLD,
    QAPatternAggregator,
    RecurringPattern,
)
from agent_framework.memory.memory_store import MemoryEntry


def _make_memory(content, tags=None):
    return MemoryEntry(
        category="qa_findings",
        content=content,
        tags=tags or [],
        created_at=time.time(),
        last_accessed=time.time(),
    )


@pytest.fixture
def memory_store():
    store = MagicMock()
    store.enabled = True
    return store


@pytest.fixture
def aggregator(memory_store):
    return QAPatternAggregator(memory_store=memory_store)


class TestGetRecurringPatterns:
    def test_no_patterns_when_memory_disabled(self):
        store = MagicMock()
        store.enabled = False
        agg = QAPatternAggregator(memory_store=store)
        assert agg.get_recurring_patterns("org/repo") == []

    def test_no_patterns_when_no_memories(self, aggregator, memory_store):
        memory_store.recall.return_value = []
        result = aggregator.get_recurring_patterns("org/repo")
        assert result == []

    def test_detects_pattern_above_threshold(self, aggregator, memory_store):
        """Same content appearing 3+ times should be detected."""
        content = "QA finding: HIGH correctness in handler.go: missing error check"
        memories = [
            _make_memory(content, tags=["high", "correctness", "file:handler.go"])
            for _ in range(RECURRENCE_THRESHOLD)
        ]
        memory_store.recall.return_value = memories
        result = aggregator.get_recurring_patterns("org/repo")
        assert len(result) == 1
        assert result[0].occurrence_count >= RECURRENCE_THRESHOLD

    def test_ignores_pattern_below_threshold(self, aggregator, memory_store):
        """Patterns appearing less than 3 times should be filtered out."""
        memory_store.recall.return_value = [
            _make_memory("QA finding: LOW style in main.go: trailing whitespace",
                         tags=["low", "style"]),
            _make_memory("QA finding: LOW style in main.go: trailing whitespace",
                         tags=["low", "style"]),
        ]
        result = aggregator.get_recurring_patterns("org/repo")
        assert len(result) == 0

    def test_filters_by_relevant_files(self, aggregator, memory_store):
        """When relevant_files is provided, only matching patterns should return."""
        content = "QA finding: HIGH correctness in handler.go: missing error check"
        memories = [
            _make_memory(content, tags=["high", "correctness", "file:handler.go"])
            for _ in range(3)
        ]
        memory_store.recall.return_value = memories

        # Match
        result = aggregator.get_recurring_patterns(
            "org/repo", relevant_files={"handler.go"}
        )
        assert len(result) == 1

        # No match
        result = aggregator.get_recurring_patterns(
            "org/repo", relevant_files={"other.go"}
        )
        assert len(result) == 0

    def test_caps_at_max_warnings(self, aggregator, memory_store):
        """Should not return more than MAX_WARNINGS patterns."""
        memories = []
        for i in range(MAX_WARNINGS + 3):
            content = f"QA finding: HIGH issue in file{i}.go: problem {i}"
            for _ in range(RECURRENCE_THRESHOLD):
                memories.append(_make_memory(content, tags=["high", f"file:file{i}.go"]))

        memory_store.recall.return_value = memories
        result = aggregator.get_recurring_patterns("org/repo")
        assert len(result) <= MAX_WARNINGS

    def test_sorted_by_occurrence_count(self, aggregator, memory_store):
        content_high = "QA finding: HIGH correctness in handler.go: missing check"
        content_low = "QA finding: LOW style in main.go: trailing ws"
        memories = (
            [_make_memory(content_high, tags=["high"]) for _ in range(5)]
            + [_make_memory(content_low, tags=["low"]) for _ in range(3)]
        )
        memory_store.recall.return_value = memories
        result = aggregator.get_recurring_patterns("org/repo")
        assert len(result) == 2
        assert result[0].occurrence_count >= result[1].occurrence_count


class TestFormatWarnings:
    def test_empty_patterns_returns_empty(self):
        agg = QAPatternAggregator()
        assert agg.format_warnings([]) == ""

    def test_formats_pattern_with_header(self):
        agg = QAPatternAggregator()
        patterns = [
            RecurringPattern(
                description="Missing error handling",
                occurrence_count=5,
                severity="HIGH",
                category="correctness",
                file_paths=["handler.go"],
            )
        ]
        result = agg.format_warnings(patterns)
        assert "RECURRING QA WARNINGS" in result
        assert "HIGH" in result
        assert "Missing error handling" in result
        assert "5x" in result

    def test_respects_max_chars(self):
        agg = QAPatternAggregator()
        patterns = [
            RecurringPattern(
                description="A" * 200,
                occurrence_count=10,
                severity="CRITICAL",
                category="security",
                file_paths=[f"file{i}.go" for i in range(20)],
            )
            for _ in range(10)
        ]
        result = agg.format_warnings(patterns)
        assert len(result) <= MAX_WARNINGS_CHARS + 10  # Small tolerance for truncation marker


class TestNormalizeKey:
    def test_empty_content(self):
        assert QAPatternAggregator._normalize_key("") == ""

    def test_normalizes_to_lowercase(self):
        key = QAPatternAggregator._normalize_key("QA FINDING: HIGH")
        assert key == key.lower()

    def test_truncates_to_80_chars(self):
        key = QAPatternAggregator._normalize_key("A" * 200)
        assert len(key) == 80


class TestFileMatches:
    def test_exact_match(self):
        assert QAPatternAggregator._file_matches("handler.go", {"handler.go"})

    def test_suffix_match(self):
        assert QAPatternAggregator._file_matches("src/handler.go", {"handler.go"})

    def test_no_match(self):
        assert not QAPatternAggregator._file_matches("other.go", {"handler.go"})
