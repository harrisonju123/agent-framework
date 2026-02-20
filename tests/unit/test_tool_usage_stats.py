"""Tests for compute_tool_usage_stats() — quantitative tool call metrics."""

from agent_framework.memory.tool_pattern_analyzer import (
    ToolUsageStats,
    compute_tool_usage_stats,
)


def _make_call(tool: str, **input_fields) -> dict:
    """Build a minimal tool_call event dict."""
    entry = {"event": "tool_call", "tool": tool}
    if input_fields:
        entry["input"] = input_fields
    return entry


class TestEmptyInput:
    def test_empty_list_returns_zeros(self):
        stats = compute_tool_usage_stats([])
        assert stats.total_calls == 0
        assert stats.tool_distribution == {}
        assert stats.duplicate_reads == {}
        assert stats.read_before_write_ratio == 0.0
        assert stats.edit_write_count == 0
        assert stats.exploration_count == 0
        assert stats.edit_density == 0.0
        assert stats.files_read == []
        assert stats.files_written == []


class TestDistribution:
    def test_basic_counting(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Read", file_path="/b.py"),
            _make_call("Grep", pattern="foo"),
            _make_call("Edit", file_path="/a.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.total_calls == 4
        assert stats.tool_distribution == {"Read": 2, "Grep": 1, "Edit": 1}

    def test_single_tool_type(self):
        calls = [_make_call("Read", file_path=f"/{i}.py") for i in range(5)]
        stats = compute_tool_usage_stats(calls)
        assert stats.total_calls == 5
        assert stats.tool_distribution == {"Read": 5}


class TestDuplicateReads:
    def test_detects_files_read_twice_or_more(self):
        calls = [
            _make_call("Read", file_path="/config.py"),
            _make_call("Read", file_path="/models.py"),
            _make_call("Read", file_path="/config.py"),
            _make_call("Read", file_path="/config.py"),
            _make_call("Read", file_path="/models.py"),
            _make_call("Read", file_path="/views.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.duplicate_reads == {"/config.py": 3, "/models.py": 2}

    def test_single_reads_excluded(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Read", file_path="/b.py"),
            _make_call("Read", file_path="/c.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.duplicate_reads == {}

    def test_empty_file_path_ignored(self):
        calls = [
            _make_call("Read", file_path=""),
            _make_call("Read", file_path=""),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.duplicate_reads == {}


class TestReadBeforeWriteRatio:
    def test_all_written_files_were_read(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Read", file_path="/b.py"),
            _make_call("Edit", file_path="/a.py"),
            _make_call("Write", file_path="/b.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.read_before_write_ratio == 1.0

    def test_no_written_files_were_read(self):
        calls = [
            _make_call("Read", file_path="/x.py"),
            _make_call("Edit", file_path="/a.py"),
            _make_call("Write", file_path="/b.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.read_before_write_ratio == 0.0

    def test_partial_coverage(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Edit", file_path="/a.py"),
            _make_call("Write", file_path="/b.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.read_before_write_ratio == 0.5

    def test_no_writes_yields_zero(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Grep", pattern="foo"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.read_before_write_ratio == 0.0


class TestEditDensity:
    def test_all_edits(self):
        calls = [
            _make_call("Edit", file_path="/a.py"),
            _make_call("Write", file_path="/b.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.edit_density == 1.0
        assert stats.edit_write_count == 2

    def test_no_edits(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Grep", pattern="foo"),
            _make_call("Bash", command="npm test"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.edit_density == 0.0
        assert stats.edit_write_count == 0

    def test_mixed_calls(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Grep", pattern="foo"),
            _make_call("Read", file_path="/b.py"),
            _make_call("Edit", file_path="/a.py"),
            _make_call("Bash", command="npm test"),
        ]
        stats = compute_tool_usage_stats(calls)
        # 1 edit out of 5 total
        assert stats.edit_density == 0.2
        assert stats.edit_write_count == 1

    def test_notebook_edit_counted(self):
        calls = [
            _make_call("NotebookEdit", notebook_path="/nb.ipynb"),
            _make_call("Read", file_path="/a.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.edit_write_count == 1
        assert stats.edit_density == 0.5


class TestExplorationCount:
    def test_counts_exploration_tools(self):
        calls = [
            _make_call("Read", file_path="/a.py"),
            _make_call("Grep", pattern="foo"),
            _make_call("Glob", pattern="**/*.py"),
            _make_call("Bash", command="npm test"),
            _make_call("Edit", file_path="/a.py"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.exploration_count == 4


class TestFileOrdering:
    def test_files_read_ordered_by_first_seen(self):
        calls = [
            _make_call("Read", file_path="/c.py"),
            _make_call("Read", file_path="/a.py"),
            _make_call("Read", file_path="/b.py"),
            _make_call("Read", file_path="/a.py"),  # duplicate
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.files_read == ["/c.py", "/a.py", "/b.py"]

    def test_files_written_ordered_by_first_seen(self):
        calls = [
            _make_call("Write", file_path="/z.py"),
            _make_call("Edit", file_path="/a.py"),
            _make_call("Edit", file_path="/z.py"),  # duplicate
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.files_written == ["/z.py", "/a.py"]

    def test_notebook_path_tracked(self):
        calls = [
            _make_call("NotebookEdit", notebook_path="/nb.ipynb"),
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.files_written == ["/nb.ipynb"]


class TestMissingInputFields:
    def test_calls_without_input_do_not_crash(self):
        calls = [
            {"event": "tool_call", "tool": "Read"},
            {"event": "tool_call", "tool": "Edit"},
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.total_calls == 2
        assert stats.files_read == []
        assert stats.files_written == []

    def test_calls_with_none_input(self):
        calls = [
            {"event": "tool_call", "tool": "Read", "input": None},
        ]
        stats = compute_tool_usage_stats(calls)
        assert stats.total_calls == 1
