"""Tests for the analyze-session CLI command and re-read observability metrics."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Avoid circular import from core/__init__.py by pre-importing task before analytics
import agent_framework.core.task  # noqa: F401
from agent_framework.analytics.agentic_metrics import AgenticMetrics
from click.testing import CliRunner

from agent_framework.cli.main import cli
from agent_framework.core.session_logger import SessionLogger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


# --- SessionLogger.log_read_dedup_stats ---


class TestLogReadDedupStats:
    def test_logs_dedup_stats_event(self, tmp_path):
        logs_dir = tmp_path / "logs"
        sl = SessionLogger(logs_dir, "task-1", enabled=True)
        try:
            read_stats = {"/a.py": 3, "/b.py": 1, "/c.py": 5}
            sl.log_read_dedup_stats(read_stats, total_reads=9)

            path = logs_dir / "sessions" / "task-1.jsonl"
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1

            evt = json.loads(lines[0])
            assert evt["event"] == "read_dedup_stats"
            assert evt["total_reads"] == 9
            # /a.py read 3x (2 dupes), /c.py read 5x (4 dupes) = 6 dupes total
            assert evt["duplicate_reads"] == 6
            assert evt["unique_files"] == 3  # 9 - 6
            assert evt["worst_file"] == "/c.py"
            assert evt["worst_count"] == 5
            # Only files with count > 1 in files_reread
            assert evt["files_reread"] == {"/a.py": 3, "/c.py": 5}
        finally:
            sl.close()

    def test_empty_stats_skipped(self, tmp_path):
        logs_dir = tmp_path / "logs"
        sl = SessionLogger(logs_dir, "task-2", enabled=True)
        try:
            sl.log_read_dedup_stats({}, total_reads=0)
            path = logs_dir / "sessions" / "task-2.jsonl"
            assert not path.exists()
        finally:
            sl.close()

    def test_disabled_logger_skipped(self, tmp_path):
        logs_dir = tmp_path / "logs"
        sl = SessionLogger(logs_dir, "task-3", enabled=False)
        try:
            sl.log_read_dedup_stats({"/a.py": 3}, total_reads=3)
            path = logs_dir / "sessions" / "task-3.jsonl"
            assert not path.exists()
        finally:
            sl.close()


# --- ToolUsageMetrics new fields ---


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "activity").mkdir(parents=True)
    (tmp_path / ".agent-communication" / "debates").mkdir(parents=True)
    return tmp_path


class TestRereadMetricsFields:
    def test_reread_interrupts_counted(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "reread_threshold_exceeded", "task_id": "t1",
             "file_path": "/a.py", "read_count": 4},
            {"ts": _now_iso(), "event": "reread_threshold_exceeded", "task_id": "t1",
             "file_path": "/b.py", "read_count": 3},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 50, "tool_distribution": {"Read": 50},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.reread_interrupts == 2

    def test_exploration_escalations_by_level(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "exploration_escalation", "task_id": "t1",
             "level": 2, "tool_calls": 60},
            {"ts": _now_iso(), "event": "exploration_force_halt", "task_id": "t1",
             "level": 3, "tool_calls": 80},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 80, "tool_distribution": {"Read": 80},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "exploration_escalation", "task_id": "t2",
             "level": 2, "tool_calls": 55},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t2",
             "total_calls": 55, "tool_distribution": {"Read": 55},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        esc = report.tool_usage.exploration_escalations
        assert esc[2] == 2  # two level-2 escalations
        assert esc[3] == 1  # one level-3 force halt

    def test_avg_duplicate_read_rate(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "read_dedup_stats", "task_id": "t1",
             "total_reads": 20, "duplicate_reads": 5, "unique_files": 15,
             "worst_file": "/a.py", "worst_count": 3,
             "files_reread": {"/a.py": 3}},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 30, "tool_distribution": {"Read": 20},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        _write_session(workspace, "t2", [
            {"ts": _now_iso(), "event": "read_dedup_stats", "task_id": "t2",
             "total_reads": 10, "duplicate_reads": 0, "unique_files": 10,
             "worst_file": "", "worst_count": 0, "files_reread": {}},
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t2",
             "total_calls": 15, "tool_distribution": {"Read": 10},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        # 5 dupes / 30 total reads = 0.167
        assert report.tool_usage.avg_duplicate_read_rate == pytest.approx(0.167, abs=0.001)

    def test_no_reread_events_defaults(self, workspace):
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "tool_usage_stats", "task_id": "t1",
             "total_calls": 10, "tool_distribution": {"Read": 10},
             "duplicate_reads": {}, "read_before_write_ratio": 1.0,
             "edit_density": 0.5},
        ])
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.reread_interrupts == 0
        assert report.tool_usage.exploration_escalations == {}
        assert report.tool_usage.avg_duplicate_read_rate == 0.0

    def test_reread_fields_in_empty_report(self, workspace):
        """No sessions at all — new fields still present with defaults."""
        report = AgenticMetrics(workspace).generate_report(hours=24)
        assert report.tool_usage.reread_interrupts == 0
        assert report.tool_usage.exploration_escalations == {}
        assert report.tool_usage.avg_duplicate_read_rate == 0.0


# --- analyze-session CLI command ---


class TestAnalyzeSessionCLI:
    def test_basic_output(self, workspace):
        ts = _now_iso()
        _write_session(workspace, "test-task", [
            {"ts": ts, "event": "task_start", "task_id": "test-task"},
            {"ts": ts, "event": "tool_call", "task_id": "test-task",
             "tool": "Read", "sequence": 1,
             "input": {"file_path": "/src/main.py"}},
            {"ts": ts, "event": "tool_call", "task_id": "test-task",
             "tool": "Read", "sequence": 2,
             "input": {"file_path": "/src/main.py"}},
            {"ts": ts, "event": "tool_call", "task_id": "test-task",
             "tool": "Edit", "sequence": 3,
             "input": {"file_path": "/src/main.py"}},
            {"ts": ts, "event": "llm_complete", "task_id": "test-task",
             "model": "claude-3.5-sonnet", "cost": 0.05,
             "tokens_in": 5000, "tokens_out": 1200, "duration_ms": 3200},
            {"ts": ts, "event": "task_complete", "task_id": "test-task"},
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(workspace), "analyze-session", "test-task"])
        assert result.exit_code == 0
        assert "test-task" in result.output
        assert "Read" in result.output
        assert "Edit" in result.output
        assert "$0.050" in result.output
        # Duplicate detection
        assert "1 duplicate" in result.output
        assert "/src/main.py" in result.output

    def test_missing_session(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(tmp_path), "analyze-session", "nonexistent"])
        assert result.exit_code == 0
        assert "No session log found" in result.output

    def test_empty_session(self, workspace):
        path = workspace / "logs" / "sessions" / "empty.jsonl"
        path.write_text("")
        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(workspace), "analyze-session", "empty"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_circuit_breaker_events(self, workspace):
        ts = _now_iso()
        _write_session(workspace, "cb-task", [
            {"ts": ts, "event": "task_start", "task_id": "cb-task"},
            {"ts": ts, "event": "reread_threshold_exceeded", "task_id": "cb-task",
             "file_path": "/hot.py", "read_count": 5},
            {"ts": ts, "event": "exploration_escalation", "task_id": "cb-task",
             "level": 2, "tool_calls": 60},
            {"ts": ts, "event": "tool_call", "task_id": "cb-task",
             "tool": "Read", "sequence": 1,
             "input": {"file_path": "/hot.py"}},
            {"ts": ts, "event": "llm_complete", "task_id": "cb-task",
             "model": "claude-3.5-sonnet", "cost": 0.01,
             "tokens_in": 1000, "tokens_out": 200, "duration_ms": 500},
            {"ts": ts, "event": "task_complete", "task_id": "cb-task"},
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(workspace), "analyze-session", "cb-task"])
        assert result.exit_code == 0
        assert "Circuit Breaker" in result.output
        assert "/hot.py" in result.output
        assert "5" in result.output

    def test_chunked_reads_detected(self, workspace):
        ts = _now_iso()
        _write_session(workspace, "chunk-task", [
            {"ts": ts, "event": "task_start", "task_id": "chunk-task"},
            {"ts": ts, "event": "tool_call", "task_id": "chunk-task",
             "tool": "Read", "sequence": 1,
             "input": {"file_path": "/big.py"}},
            {"ts": ts, "event": "tool_call", "task_id": "chunk-task",
             "tool": "Read", "sequence": 3,
             "input": {"file_path": "/big.py"}},
            {"ts": ts, "event": "llm_complete", "task_id": "chunk-task",
             "model": "claude-3.5-sonnet", "cost": 0.01,
             "tokens_in": 1000, "tokens_out": 200, "duration_ms": 500},
            {"ts": ts, "event": "task_complete", "task_id": "chunk-task"},
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(workspace), "analyze-session", "chunk-task"])
        assert result.exit_code == 0
        assert "Chunked Reads" in result.output
        assert "/big.py" in result.output

    def test_no_llm_events(self, workspace):
        """Session with only tool calls, no llm_complete — still works."""
        ts = _now_iso()
        _write_session(workspace, "no-llm", [
            {"ts": ts, "event": "task_start", "task_id": "no-llm"},
            {"ts": ts, "event": "tool_call", "task_id": "no-llm",
             "tool": "Read", "sequence": 1,
             "input": {"file_path": "/x.py"}},
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["-w", str(workspace), "analyze-session", "no-llm"])
        assert result.exit_code == 0
        assert "$0.000" in result.output
        assert "Read" in result.output
