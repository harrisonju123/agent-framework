"""Tests for agentic observability metrics."""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import (
    AgenticMetrics,
    AgenticMetricsReport,
    ContextBudgetMetrics,
    MemoryMetrics,
    ReplanMetrics,
    SelfEvalMetrics,
)


def _ts(offset_hours: float = 0) -> str:
    """Generate an ISO timestamp offset from now."""
    dt = datetime.now(timezone.utc) - timedelta(hours=offset_hours)
    return dt.isoformat()


def _write_session(sessions_dir: Path, task_id: str, events: list[dict]) -> Path:
    """Write a session JSONL file and return the path."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{task_id}.jsonl"
    lines = [json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def sessions_dir(workspace):
    d = workspace / "logs" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def metrics(workspace):
    return AgenticMetrics(workspace)


# ===== Empty / no data =====


class TestEmptyState:
    def test_no_sessions_dir(self, workspace):
        m = AgenticMetrics(workspace)
        report = m.generate_report(hours=24)
        assert report.memory.total_sessions == 0
        assert report.self_eval.total_evals == 0
        assert report.replan.total_replans == 0
        assert report.context_budget.total_completions == 0

    def test_empty_sessions_dir(self, metrics, sessions_dir):
        report = metrics.generate_report(hours=24)
        assert report.memory.total_sessions == 0

    def test_output_file_created(self, workspace, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
        ])
        m = AgenticMetrics(workspace)
        m.generate_report(hours=24)
        output = workspace / ".agent-communication" / "metrics" / "agentics.json"
        assert output.exists()
        data = json.loads(output.read_text())
        assert "memory" in data
        assert "self_eval" in data


# ===== Memory metrics =====


class TestMemoryMetrics:
    def test_session_with_recall(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.9), "event": "memory_recall", "task_id": "task-1", "repo": "myrepo", "chars_injected": 500, "categories": ["patterns"]},
        ])
        report = metrics.generate_report(hours=24)
        assert report.memory.sessions_with_recall == 1
        assert report.memory.sessions_without_recall == 0
        assert report.memory.hit_rate_pct == 100.0
        assert report.memory.avg_chars_injected == 500.0
        assert report.memory.total_recalls == 1

    def test_session_without_recall(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.5), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.memory.sessions_with_recall == 0
        assert report.memory.sessions_without_recall == 1
        assert report.memory.hit_rate_pct == 0.0

    def test_multiple_recalls_in_session(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1.5), "event": "memory_recall", "task_id": "task-1", "repo": "r", "chars_injected": 200},
            {"ts": _ts(1.0), "event": "memory_recall", "task_id": "task-1", "repo": "r", "chars_injected": 800},
        ])
        report = metrics.generate_report(hours=24)
        assert report.memory.total_recalls == 2
        assert report.memory.avg_chars_injected == 500.0  # (200 + 800) / 2

    def test_hit_rate_across_sessions(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.5), "event": "memory_recall", "task_id": "task-1", "repo": "r", "chars_injected": 100},
        ])
        _write_session(sessions_dir, "task-2", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-2", "task_type": "planning"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.memory.sessions_with_recall == 1
        assert report.memory.sessions_without_recall == 1
        assert report.memory.hit_rate_pct == 50.0


# ===== Self-eval metrics =====


class TestSelfEvalMetrics:
    def test_all_verdicts_counted(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1.5), "event": "self_eval", "task_id": "task-1", "verdict": "AUTO_PASS"},
            {"ts": _ts(1.0), "event": "self_eval", "task_id": "task-1", "verdict": "PASS"},
            {"ts": _ts(0.5), "event": "self_eval", "task_id": "task-1", "verdict": "FAIL"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.self_eval.total_evals == 3
        assert report.self_eval.auto_pass_count == 1
        assert report.self_eval.pass_count == 1
        assert report.self_eval.fail_count == 1

    def test_catch_rate(self, metrics, sessions_dir):
        # 2 PASS, 1 FAIL → catch rate = 1/3 = 33.3%
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.9), "event": "self_eval", "task_id": "task-1", "verdict": "PASS"},
            {"ts": _ts(0.8), "event": "self_eval", "task_id": "task-1", "verdict": "PASS"},
            {"ts": _ts(0.7), "event": "self_eval", "task_id": "task-1", "verdict": "FAIL"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.self_eval.catch_rate_pct == 33.3

    def test_auto_pass_excluded_from_catch_rate(self, metrics, sessions_dir):
        # AUTO_PASS doesn't count in catch rate denominator
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.9), "event": "self_eval", "task_id": "task-1", "verdict": "AUTO_PASS"},
            {"ts": _ts(0.8), "event": "self_eval", "task_id": "task-1", "verdict": "FAIL"},
        ])
        report = metrics.generate_report(hours=24)
        # non_auto = 0 PASS + 1 FAIL = 1, catch = 1/1 = 100%
        assert report.self_eval.catch_rate_pct == 100.0
        assert report.self_eval.auto_pass_rate_pct == 50.0

    def test_case_insensitive_verdicts(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.9), "event": "self_eval", "task_id": "task-1", "verdict": "auto_pass"},
            {"ts": _ts(0.8), "event": "self_eval", "task_id": "task-1", "verdict": "pass"},
            {"ts": _ts(0.7), "event": "self_eval", "task_id": "task-1", "verdict": "fail"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.self_eval.total_evals == 3

    def test_no_evals(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "planning"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.self_eval.total_evals == 0
        assert report.self_eval.catch_rate_pct == 0.0


# ===== Replan metrics =====


class TestReplanMetrics:
    def test_replan_then_complete(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1.5), "event": "replan", "task_id": "task-1", "retry": 1, "previous_error": "test fail"},
            {"ts": _ts(0.5), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.replan.total_replans == 1
        assert report.replan.tasks_with_replans == 1
        assert report.replan.tasks_completed_after_replan == 1
        assert report.replan.replan_success_rate_pct == 100.0

    def test_replan_then_fail(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1.5), "event": "replan", "task_id": "task-1", "retry": 1},
            {"ts": _ts(0.5), "event": "task_failed", "task_id": "task-1", "error": "still broken"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.replan.tasks_with_replans == 1
        assert report.replan.tasks_completed_after_replan == 0
        assert report.replan.replan_success_rate_pct == 0.0

    def test_multiple_replans_same_task(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(3), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(2), "event": "replan", "task_id": "task-1", "retry": 1},
            {"ts": _ts(1), "event": "replan", "task_id": "task-1", "retry": 2},
            {"ts": _ts(0.5), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.replan.total_replans == 2
        assert report.replan.tasks_with_replans == 1
        assert report.replan.tasks_completed_after_replan == 1

    def test_no_replans(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.5), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        report = metrics.generate_report(hours=24)
        assert report.replan.total_replans == 0
        assert report.replan.replan_success_rate_pct == 0.0


# ===== Context budget metrics =====


class TestContextBudgetMetrics:
    def test_basic_bucketing(self, metrics, sessions_dir):
        # implementation budget = 50000. 10000 tokens = 20% → 0-25 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 7000, "tokens_out": 3000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.total_completions == 1
        assert report.context_budget.band_0_25_pct == 1

    def test_over_budget_band(self, metrics, sessions_dir):
        # planning budget = 30000. 35000 tokens = 116% → over_100 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "planning"},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 25000, "tokens_out": 10000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.band_over_100_pct == 1

    def test_multiple_completions_in_session(self, metrics, sessions_dir):
        # implementation budget = 50000
        # 10k tokens = 20% → 0-25 band
        # 30k tokens = 60% → 50-75 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(3), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(2), "event": "llm_complete", "task_id": "task-1", "tokens_in": 5000, "tokens_out": 5000},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 20000, "tokens_out": 10000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.total_completions == 2
        assert report.context_budget.band_0_25_pct == 1
        assert report.context_budget.band_50_75_pct == 1

    def test_fallback_budget_for_unknown_task_type(self, metrics, sessions_dir):
        # Unknown type → 40000 fallback. 20000 tokens = 50% → 25-50 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "unknown_type"},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 15000, "tokens_out": 5000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.band_25_50_pct == 1

    def test_no_task_start_uses_fallback(self, metrics, sessions_dir):
        # No task_start → 40000 fallback. 10000 = 25% → 0-25 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 5000, "tokens_out": 5000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.band_0_25_pct == 1

    def test_zero_tokens_skipped(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 0, "tokens_out": 0},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.total_completions == 0

    def test_avg_utilization(self, metrics, sessions_dir):
        # implementation = 50000 budget
        # 25000 tokens = 50% util
        # 37500 tokens = 75% util
        # avg = 62.5%
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(3), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(2), "event": "llm_complete", "task_id": "task-1", "tokens_in": 15000, "tokens_out": 10000},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 25000, "tokens_out": 12500},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.avg_utilization_pct == 62.5


# ===== Time window filtering =====


class TestTimeFiltering:
    def test_old_sessions_excluded_by_mtime(self, metrics, sessions_dir):
        path = _write_session(sessions_dir, "old-task", [
            {"ts": _ts(48), "event": "task_start", "task_id": "old-task", "task_type": "implementation"},
        ])
        # Backdate mtime to 3 days ago
        old_mtime = time.time() - (3 * 86400)
        os.utime(path, (old_mtime, old_mtime))

        report = metrics.generate_report(hours=24)
        assert report.memory.total_sessions == 0

    def test_events_outside_window_excluded(self, metrics, sessions_dir):
        # File is recent (mtime ok), but events are 48h old
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(48), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(47), "event": "memory_recall", "task_id": "task-1", "repo": "r", "chars_injected": 100},
            {"ts": _ts(1), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        report = metrics.generate_report(hours=24)
        # Only task_complete is within 24h, no memory_recall
        assert report.memory.sessions_with_recall == 0
        assert report.memory.total_sessions == 1


# ===== Edge cases =====


class TestEdgeCases:
    def test_malformed_jsonl_lines_skipped(self, metrics, sessions_dir):
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / "task-1.jsonl"
        lines = [
            json.dumps({"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"}),
            "NOT VALID JSON",
            json.dumps({"ts": _ts(0.5), "event": "task_complete", "task_id": "task-1"}),
        ]
        path.write_text("\n".join(lines) + "\n")

        report = metrics.generate_report(hours=24)
        assert report.memory.total_sessions == 1

    def test_missing_fields_handled(self, metrics, sessions_dir):
        # self_eval without verdict, memory_recall without chars_injected
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(0.9), "event": "self_eval", "task_id": "task-1"},
            {"ts": _ts(0.8), "event": "memory_recall", "task_id": "task-1"},
        ])
        report = metrics.generate_report(hours=24)
        # self_eval with no verdict → empty string upper() → not matched
        assert report.self_eval.total_evals == 0
        # memory_recall still counted even with 0 chars
        assert report.memory.total_recalls == 1
        assert report.memory.avg_chars_injected == 0.0

    def test_empty_session_file(self, metrics, sessions_dir):
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "empty-task.jsonl").write_text("")
        report = metrics.generate_report(hours=24)
        assert report.memory.total_sessions == 0

    def test_hyphenated_task_type_normalized(self, metrics, sessions_dir):
        # bug-fix → bug_fix, budget = 30000. 15000 = 50% → 25-50 band
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(2), "event": "task_start", "task_id": "task-1", "task_type": "bug-fix"},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-1", "tokens_in": 10000, "tokens_out": 5000},
        ])
        report = metrics.generate_report(hours=24)
        assert report.context_budget.band_25_50_pct == 1


# ===== Multi-session integration =====


class TestMultiSession:
    def test_full_scenario(self, metrics, sessions_dir):
        # Task 1: has memory, self-eval PASS, no replan, low token usage
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(3), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
            {"ts": _ts(2.5), "event": "memory_recall", "task_id": "task-1", "repo": "r", "chars_injected": 300},
            {"ts": _ts(2), "event": "llm_complete", "task_id": "task-1", "tokens_in": 5000, "tokens_out": 3000},
            {"ts": _ts(1.5), "event": "self_eval", "task_id": "task-1", "verdict": "PASS"},
            {"ts": _ts(1), "event": "task_complete", "task_id": "task-1", "status": "completed"},
        ])
        # Task 2: no memory, self-eval FAIL → replan → complete
        _write_session(sessions_dir, "task-2", [
            {"ts": _ts(3), "event": "task_start", "task_id": "task-2", "task_type": "planning"},
            {"ts": _ts(2.5), "event": "llm_complete", "task_id": "task-2", "tokens_in": 20000, "tokens_out": 8000},
            {"ts": _ts(2), "event": "self_eval", "task_id": "task-2", "verdict": "FAIL"},
            {"ts": _ts(1.5), "event": "replan", "task_id": "task-2", "retry": 1},
            {"ts": _ts(1), "event": "llm_complete", "task_id": "task-2", "tokens_in": 15000, "tokens_out": 5000},
            {"ts": _ts(0.5), "event": "self_eval", "task_id": "task-2", "verdict": "PASS"},
            {"ts": _ts(0.2), "event": "task_complete", "task_id": "task-2", "status": "completed"},
        ])

        report = metrics.generate_report(hours=24)

        # Memory: 1/2 sessions have recall
        assert report.memory.hit_rate_pct == 50.0

        # Self-eval: 2 PASS + 1 FAIL = 3 total, catch = 1/3 = 33.3%
        assert report.self_eval.total_evals == 3
        assert report.self_eval.catch_rate_pct == 33.3

        # Replan: 1 task replanned, completed after
        assert report.replan.total_replans == 1
        assert report.replan.replan_success_rate_pct == 100.0

        # Budget: 3 llm_complete events total
        assert report.context_budget.total_completions == 3


# ===== Report serialization =====


class TestReportSerialization:
    def test_report_is_valid_json(self, metrics, sessions_dir):
        _write_session(sessions_dir, "task-1", [
            {"ts": _ts(1), "event": "task_start", "task_id": "task-1", "task_type": "implementation"},
        ])
        report = metrics.generate_report(hours=24)
        serialized = report.model_dump_json()
        parsed = json.loads(serialized)
        assert "generated_at" in parsed
        assert "time_range_hours" in parsed
        assert parsed["time_range_hours"] == 24
