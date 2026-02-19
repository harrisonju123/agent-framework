"""Tests for agentic feature metrics aggregator."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent_framework.analytics.agentic_metrics import AgenticMetricsAggregator


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_jsonl(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_memory_stats_empty_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        agg = AgenticMetricsAggregator(Path(tmpdir))
        stats = agg._compute_memory_stats()
        assert stats.total_entries == 0
        assert stats.accessed_entries == 0
        assert stats.hit_rate == 0.0


def test_memory_stats_counts_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        mem_dir = workspace / ".agent-communication" / "memory" / "owner__repo"
        _write_json(mem_dir / "engineer.json", [
            {"category": "conventions", "content": "use snake_case", "access_count": 3},
            {"category": "repo_structure", "content": "src/ layout", "access_count": 0},
            {"category": "test_commands", "content": "pytest", "access_count": 1},
        ])

        agg = AgenticMetricsAggregator(workspace)
        stats = agg._compute_memory_stats()
        assert stats.total_entries == 3
        assert stats.accessed_entries == 2  # access_count 3 and 1
        assert stats.hit_rate == round(2 / 3, 3)


def test_self_eval_stats_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        agg = AgenticMetricsAggregator(Path(tmpdir))
        stats = agg._compute_self_eval_stats()
        assert stats.total_evaluations == 0
        assert stats.failed_evaluations == 0
        assert stats.retry_rate == 0.0


def test_self_eval_stats_counts_failures():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        sessions_dir = workspace / "logs" / "sessions"

        _write_jsonl(sessions_dir / "task-1.jsonl", [
            {"event": "self_eval", "verdict": "PASS"},
            {"event": "self_eval", "verdict": "FAIL"},
        ])
        _write_jsonl(sessions_dir / "task-2.jsonl", [
            {"event": "self_eval", "verdict": "AUTO_PASS"},
        ])

        agg = AgenticMetricsAggregator(workspace)
        stats = agg._compute_self_eval_stats()
        assert stats.total_evaluations == 3
        assert stats.failed_evaluations == 1
        assert stats.retry_rate == round(1 / 3, 3)


def test_replan_stats_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        agg = AgenticMetricsAggregator(Path(tmpdir))
        stats = agg._compute_replan_stats()
        assert stats.sessions_with_replan == 0
        assert stats.total_sessions == 0
        assert stats.trigger_rate == 0.0


def test_replan_stats_counts_sessions():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        sessions_dir = workspace / "logs" / "sessions"

        # Session with replan
        _write_jsonl(sessions_dir / "task-a.jsonl", [
            {"event": "tool_call", "tool": "Read"},
            {"event": "replan", "retry": 1},
            {"event": "task_complete", "status": "completed"},
        ])
        # Session without replan
        _write_jsonl(sessions_dir / "task-b.jsonl", [
            {"event": "tool_call", "tool": "Bash"},
            {"event": "task_complete", "status": "completed"},
        ])

        agg = AgenticMetricsAggregator(workspace)
        stats = agg._compute_replan_stats()
        assert stats.total_sessions == 2
        assert stats.sessions_with_replan == 1
        assert stats.trigger_rate == 0.5


def test_specialization_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        activity_dir = workspace / ".agent-communication" / "activity"

        _write_json(activity_dir / "engineer-1.json", {"specialization": "backend"})
        _write_json(activity_dir / "engineer-2.json", {"specialization": "frontend"})
        _write_json(activity_dir / "engineer-3.json", {"specialization": "backend"})
        _write_json(activity_dir / "architect.json", {})  # no specialization

        agg = AgenticMetricsAggregator(workspace)
        dist = agg._compute_specialization_distribution()
        assert dist["backend"] == 2
        assert dist["frontend"] == 1
        assert dist["none"] == 1


def test_context_budget_stats_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        agg = AgenticMetricsAggregator(Path(tmpdir))
        stats = agg._compute_context_budget_stats()
        assert stats.total_tasks_with_tokens == 0
        assert stats.budget_exceeded_events == 0
        assert stats.avg_utilization_pct is None


def test_context_budget_stats_reads_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        comm_dir = workspace / ".agent-communication"
        comm_dir.mkdir(parents=True, exist_ok=True)

        stream_file = comm_dir / "activity-stream.jsonl"
        _write_jsonl(stream_file, [
            {
                "type": "complete",
                "agent": "engineer",
                "task_id": "t1",
                "title": "Task 1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_tokens": 20000,
                "output_tokens": 5000,  # total 25000 = 62.5% of 40000 default
            },
            {
                "type": "token_budget_exceeded",
                "agent": "engineer",
                "task_id": "t2",
                "title": "Budget exceeded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "complete",
                "agent": "qa",
                "task_id": "t3",
                "title": "Task 3",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # No token fields — should be skipped
            },
        ])

        agg = AgenticMetricsAggregator(workspace)
        stats = agg._compute_context_budget_stats()
        assert stats.total_tasks_with_tokens == 1
        assert stats.budget_exceeded_events == 1
        assert stats.avg_utilization_pct is not None
        # 25000 / 40000 * 100 = 62.5
        assert stats.avg_utilization_pct == 62.5
