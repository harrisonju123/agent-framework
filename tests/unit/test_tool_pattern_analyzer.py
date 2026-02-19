"""Tests for tool pattern analyzer — session log anti-pattern detection."""

import json
from pathlib import Path

import pytest

from agent_framework.memory.tool_pattern_analyzer import (
    ToolPatternAnalyzer,
    ToolPatternRecommendation,
)


def _write_session(tmp_path: Path, tool_calls: list) -> Path:
    """Write a minimal JSONL session file from a list of tool call dicts."""
    path = tmp_path / "session.jsonl"
    lines = []
    for i, tc in enumerate(tool_calls):
        entry = {
            "ts": "2026-02-15T00:00:00Z",
            "event": "tool_call",
            "task_id": "test-task",
            "tool": tc["tool"],
            "sequence": i + 1,
        }
        if "input" in tc:
            entry["input"] = tc["input"]
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines))
    return path


class TestSequentialReads:
    def test_detects_three_reads_no_grep(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Read", "input": {"file_path": "/b.py"}},
            {"tool": "Read", "input": {"file_path": "/c.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "sequential-reads" in ids

    def test_no_detection_with_grep_in_window(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"pattern": "foo"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Read", "input": {"file_path": "/b.py"}},
            {"tool": "Read", "input": {"file_path": "/c.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "sequential-reads" not in ids

    def test_no_detection_with_two_reads(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Read", "input": {"file_path": "/b.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "sequential-reads" not in ids


class TestGrepThenReadSame:
    def test_detects_grep_then_read_same_file(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"path": "/src/foo.py", "pattern": "bar"}},
            {"tool": "Read", "input": {"file_path": "/src/foo.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "grep-then-read-same" in ids

    def test_no_detection_for_different_files(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"path": "/src/foo.py", "pattern": "bar"}},
            {"tool": "Read", "input": {"file_path": "/src/other.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "grep-then-read-same" not in ids

    def test_no_detection_for_directory_grep(self, tmp_path):
        """Grep on a directory path shouldn't match Read on a specific file."""
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"path": "/src/", "pattern": "bar"}},
            {"tool": "Read", "input": {"file_path": "/src/foo.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "grep-then-read-same" not in ids


class TestRepeatedGlob:
    def test_detects_duplicate_glob_pattern(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Glob", "input": {"pattern": "**/*.py"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Glob", "input": {"pattern": "**/*.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "repeated-glob" in ids

    def test_no_detection_for_different_patterns(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Glob", "input": {"pattern": "**/*.py"}},
            {"tool": "Glob", "input": {"pattern": "**/*.js"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "repeated-glob" not in ids


class TestBashForSearch:
    def test_detects_bash_grep(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Bash", "input": {"command": "grep -r 'foo' src/"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "bash-for-search" in ids

    def test_detects_bash_find(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Bash", "input": {"command": "find . -name '*.py'"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "bash-for-search" in ids

    def test_detects_standalone_cat(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Bash", "input": {"command": "cat /etc/hosts"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "bash-for-search" in ids

    def test_no_detection_for_pipe_tail_cat(self, tmp_path):
        """cat at end of pipe (pager disable) is not an anti-pattern."""
        session = _write_session(tmp_path, [
            {"tool": "Bash", "input": {"command": "git log | cat"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "bash-for-search" not in ids

    def test_no_detection_for_normal_bash(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Bash", "input": {"command": "npm install"}},
            {"tool": "Read", "input": {"file_path": "/a.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "bash-for-search" not in ids


class TestReadWithoutLimit:
    def test_detects_full_read_after_grep(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"path": "/src/foo.py", "pattern": "bar"}},
            {"tool": "Bash", "input": {"command": "npm test"}},
            {"tool": "Read", "input": {"file_path": "/src/foo.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "read-without-limit" in ids

    def test_no_detection_with_limit(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Grep", "input": {"path": "/src/foo.py", "pattern": "bar"}},
            {"tool": "Read", "input": {"file_path": "/src/foo.py", "offset": 10, "limit": 20}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "read-without-limit" not in ids


class TestChunkedReread:
    def test_detects_three_reads_same_file(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 0, "limit": 100}},
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 100, "limit": 100}},
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 200, "limit": 100}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "chunked-reread" in ids

    def test_no_detection_with_two_reads(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 0, "limit": 100}},
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 100, "limit": 100}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "chunked-reread" not in ids

    def test_no_detection_different_files(self, tmp_path):
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Read", "input": {"file_path": "/b.py"}},
            {"tool": "Read", "input": {"file_path": "/c.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "chunked-reread" not in ids

    def test_non_adjacent_reads_still_detected(self, tmp_path):
        """Reads of the same file spread across the session still fire."""
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 0, "limit": 50}},
            {"tool": "Grep", "input": {"pattern": "foo"}},
            {"tool": "Bash", "input": {"command": "npm test"}},
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 50, "limit": 50}},
            {"tool": "Glob", "input": {"pattern": "**/*.py"}},
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": 100, "limit": 50}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        ids = {r.pattern_id for r in results}
        assert "chunked-reread" in ids

    def test_many_reads_fires_once(self, tmp_path):
        """10 reads of the same file should produce exactly one recommendation."""
        calls = [
            {"tool": "Read", "input": {"file_path": "/src/big.py", "offset": i * 50, "limit": 50}}
            for i in range(10)
        ]
        session = _write_session(tmp_path, calls)
        results = ToolPatternAnalyzer().analyze_session(session)
        chunked = [r for r in results if r.pattern_id == "chunked-reread"]
        assert len(chunked) == 1


class TestEdgeCases:
    def test_empty_session(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert ToolPatternAnalyzer().analyze_session(path) == []

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n{bad json too\n")
        assert ToolPatternAnalyzer().analyze_session(path) == []

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert ToolPatternAnalyzer().analyze_session(path) == []

    def test_non_tool_events_ignored(self, tmp_path):
        path = tmp_path / "session.jsonl"
        lines = [
            json.dumps({"event": "task_start", "agent": "engineer"}),
            json.dumps({"event": "prompt_built", "prompt_length": 1000}),
        ]
        path.write_text("\n".join(lines))
        assert ToolPatternAnalyzer().analyze_session(path) == []

    def test_missing_input_fields_handled(self, tmp_path):
        """Tool calls with missing input should not crash."""
        session = _write_session(tmp_path, [
            {"tool": "Read"},
            {"tool": "Read"},
            {"tool": "Read"},
        ])
        # Should not raise
        results = ToolPatternAnalyzer().analyze_session(session)
        # 3 reads but all with empty file_path, so unique files < 3
        assert isinstance(results, list)

    def test_each_pattern_detected_once(self, tmp_path):
        """Even with many windows matching, each pattern_id appears at most once."""
        session = _write_session(tmp_path, [
            {"tool": "Read", "input": {"file_path": "/a.py"}},
            {"tool": "Read", "input": {"file_path": "/b.py"}},
            {"tool": "Read", "input": {"file_path": "/c.py"}},
            {"tool": "Read", "input": {"file_path": "/d.py"}},
            {"tool": "Read", "input": {"file_path": "/e.py"}},
        ])
        results = ToolPatternAnalyzer().analyze_session(session)
        pattern_ids = [r.pattern_id for r in results]
        assert len(pattern_ids) == len(set(pattern_ids))
