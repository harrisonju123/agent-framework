"""Tests for AgenticsMetrics — session-log-based agentic observability."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import (
    AgenticsMetrics,
    AgenticsReport,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sessions_dir(workspace: Path) -> Path:
    d = workspace / "logs" / "sessions"
    d.mkdir(parents=True)
    return d


def _write_session(sessions_dir: Path, task_id: str, events: list[dict]) -> None:
    """Write a JSONL session file with the given events, stamped to now."""
    now = datetime.now(timezone.utc)
    path = sessions_dir / f"{task_id}.jsonl"
    with path.open("w") as f:
        for ev in events:
            if "ts" not in ev:
                ev = {"ts": now.isoformat(), **ev}
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# Empty / missing state
# ---------------------------------------------------------------------------


class TestEmptySessions:
    def test_no_sessions_dir_returns_zeros(self, workspace: Path):
        # sessions dir does not exist at all
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)

        assert isinstance(report, AgenticsReport)
        assert report.sessions_scanned == 0
        assert report.memory_hit_rate.total_recalls == 0
        assert report.self_eval.total_evals == 0
        assert report.replan.tasks_with_replan == 0
        assert report.context_budget.total_llm_calls == 0

    def test_empty_sessions_dir(self, workspace: Path, sessions_dir: Path):
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.sessions_scanned == 0

    def test_output_file_is_written(self, workspace: Path, sessions_dir: Path):
        _write_session(sessions_dir, "t1", [{"event": "task_start"}])
        reporter = AgenticsMetrics(workspace)
        reporter.generate_report(hours=24)
        out = workspace / ".agent-communication" / "metrics" / "agentics.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "memory_hit_rate" in data


# ---------------------------------------------------------------------------
# Time-window filtering
# ---------------------------------------------------------------------------


class TestTimeWindowFiltering:
    def test_old_events_excluded(self, workspace: Path, sessions_dir: Path):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _write_session(
            sessions_dir,
            "old-task",
            [{"ts": old_ts, "event": "memory_recall", "chars_injected": 500}],
        )
        reporter = AgenticsMetrics(workspace)
        # Only look back 24 hours — old event should be excluded
        report = reporter.generate_report(hours=24)
        assert report.sessions_scanned == 0
        assert report.memory_hit_rate.total_recalls == 0

    def test_recent_events_included(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "new-task",
            [{"event": "memory_recall", "chars_injected": 300}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.sessions_scanned == 1
        assert report.memory_hit_rate.total_recalls == 1


# ---------------------------------------------------------------------------
# Memory hit rate
# ---------------------------------------------------------------------------


class TestMemoryHitRate:
    def test_sessions_counted_correctly(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "with-recall",
            [{"event": "memory_recall", "chars_injected": 400}],
        )
        _write_session(
            sessions_dir,
            "no-recall",
            [{"event": "task_start"}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)

        m = report.memory_hit_rate
        assert m.sessions_with_recall == 1
        assert m.sessions_without_recall == 1
        assert m.total_recalls == 1
        assert m.avg_chars_injected == 400.0

    def test_multiple_recalls_per_session(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "t1",
            [
                {"event": "memory_recall", "chars_injected": 200},
                {"event": "memory_recall", "chars_injected": 600},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)

        m = report.memory_hit_rate
        assert m.sessions_with_recall == 1
        assert m.total_recalls == 2
        assert m.avg_chars_injected == 400.0

    def test_no_chars_injected_field_defaults_zero(
        self, workspace: Path, sessions_dir: Path
    ):
        _write_session(
            sessions_dir, "t1", [{"event": "memory_recall"}]
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.memory_hit_rate.avg_chars_injected == 0.0


# ---------------------------------------------------------------------------
# Self-eval metrics
# ---------------------------------------------------------------------------


class TestSelfEvalMetrics:
    def test_verdicts_counted(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "t1",
            [
                {"event": "self_eval", "verdict": "AUTO_PASS"},
                {"event": "self_eval", "verdict": "PASS"},
                {"event": "self_eval", "verdict": "FAIL"},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)

        s = report.self_eval
        assert s.total_evals == 3
        assert s.auto_pass_count == 1
        assert s.pass_count == 1
        assert s.fail_count == 1

    def test_catch_rate_calculation(self, workspace: Path, sessions_dir: Path):
        # 2 FAIL out of 4 total → 50% catch rate
        _write_session(
            sessions_dir,
            "t1",
            [
                {"event": "self_eval", "verdict": "PASS"},
                {"event": "self_eval", "verdict": "FAIL"},
                {"event": "self_eval", "verdict": "PASS"},
                {"event": "self_eval", "verdict": "FAIL"},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.self_eval.catch_rate_percent == 50.0

    def test_auto_pass_rate(self, workspace: Path, sessions_dir: Path):
        # 1 AUTO_PASS out of 2 → 50%
        _write_session(
            sessions_dir,
            "t1",
            [
                {"event": "self_eval", "verdict": "AUTO_PASS"},
                {"event": "self_eval", "verdict": "PASS"},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.self_eval.auto_pass_rate_percent == 50.0

    def test_no_evals_zero_rates(self, workspace: Path, sessions_dir: Path):
        _write_session(sessions_dir, "t1", [{"event": "task_start"}])
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.self_eval.total_evals == 0
        assert report.self_eval.catch_rate_percent == 0.0


# ---------------------------------------------------------------------------
# Replan metrics
# ---------------------------------------------------------------------------


class TestReplanMetrics:
    def test_no_replans(self, workspace: Path, sessions_dir: Path):
        _write_session(sessions_dir, "t1", [{"event": "task_start"}])
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.replan.tasks_with_replan == 0
        assert report.replan.success_after_replan_percent == 0.0

    def test_replan_with_success(self, workspace: Path, sessions_dir: Path):
        now = datetime.now(timezone.utc)
        _write_session(
            sessions_dir,
            "t1",
            [
                {"ts": (now - timedelta(minutes=5)).isoformat(), "event": "replan", "retry": 1},
                {"ts": now.isoformat(), "event": "task_complete", "status": "completed"},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        r = report.replan
        assert r.tasks_with_replan == 1
        assert r.total_replan_events == 1
        assert r.tasks_completed_after_replan == 1
        assert r.success_after_replan_percent == 100.0

    def test_replan_without_success(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "t1",
            [{"event": "replan", "retry": 1}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        r = report.replan
        assert r.tasks_with_replan == 1
        assert r.tasks_completed_after_replan == 0
        assert r.success_after_replan_percent == 0.0

    def test_multiple_replans_same_task(self, workspace: Path, sessions_dir: Path):
        now = datetime.now(timezone.utc)
        _write_session(
            sessions_dir,
            "t1",
            [
                {"ts": (now - timedelta(minutes=10)).isoformat(), "event": "replan", "retry": 1},
                {"ts": (now - timedelta(minutes=5)).isoformat(), "event": "replan", "retry": 2},
                {"ts": now.isoformat(), "event": "task_complete", "status": "completed"},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        r = report.replan
        assert r.tasks_with_replan == 1
        assert r.total_replan_events == 2
        assert r.tasks_completed_after_replan == 1

    def test_partial_success_across_tasks(self, workspace: Path, sessions_dir: Path):
        now = datetime.now(timezone.utc)
        # Task 1: replanned and succeeded
        _write_session(
            sessions_dir,
            "t1",
            [
                {"ts": (now - timedelta(minutes=5)).isoformat(), "event": "replan", "retry": 1},
                {"ts": now.isoformat(), "event": "task_complete", "status": "completed"},
            ],
        )
        # Task 2: replanned but did not complete
        _write_session(
            sessions_dir,
            "t2",
            [{"event": "replan", "retry": 1}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        r = report.replan
        assert r.tasks_with_replan == 2
        assert r.tasks_completed_after_replan == 1
        assert r.success_after_replan_percent == 50.0


# ---------------------------------------------------------------------------
# Context budget metrics
# ---------------------------------------------------------------------------


class TestContextBudgetMetrics:
    def test_llm_calls_summed(self, workspace: Path, sessions_dir: Path):
        _write_session(
            sessions_dir,
            "t1",
            [
                {"event": "llm_complete", "tokens_in": 1000, "tokens_out": 500},
                {"event": "llm_complete", "tokens_in": 2000, "tokens_out": 800},
            ],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        cb = report.context_budget
        assert cb.total_llm_calls == 2
        assert cb.avg_tokens_in == 1500.0
        assert cb.avg_tokens_out == 650.0

    def test_utilization_bucket_assignment(self, workspace: Path, sessions_dir: Path):
        # 200k nominal budget; 50k total = 25% → "0-25%" bucket
        _write_session(
            sessions_dir,
            "t1",
            [{"event": "llm_complete", "tokens_in": 40_000, "tokens_out": 10_000}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.context_budget.utilization_buckets["0-25%"] == 1
        assert report.context_budget.utilization_buckets["25-50%"] == 0

    def test_high_utilization_bucket(self, workspace: Path, sessions_dir: Path):
        # 160k total = 80% → "75-100%" bucket
        _write_session(
            sessions_dir,
            "t1",
            [{"event": "llm_complete", "tokens_in": 120_000, "tokens_out": 40_000}],
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.context_budget.utilization_buckets["75-100%"] == 1

    def test_no_llm_calls_zeros(self, workspace: Path, sessions_dir: Path):
        _write_session(sessions_dir, "t1", [{"event": "task_start"}])
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        cb = report.context_budget
        assert cb.total_llm_calls == 0
        assert cb.avg_tokens_in == 0.0
        assert cb.avg_total_tokens == 0.0

    def test_missing_token_fields_default_zero(
        self, workspace: Path, sessions_dir: Path
    ):
        _write_session(
            sessions_dir, "t1", [{"event": "llm_complete"}]
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        assert report.context_budget.total_llm_calls == 1
        assert report.context_budget.avg_tokens_in == 0.0


# ---------------------------------------------------------------------------
# Malformed / partial JSONL resilience
# ---------------------------------------------------------------------------


class TestMalformedInput:
    def test_malformed_lines_skipped(self, workspace: Path, sessions_dir: Path):
        path = sessions_dir / "bad.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        path.write_text(
            f"not-json\n"
            f'{{"ts": "{now}", "event": "memory_recall", "chars_injected": 100}}\n'
        )
        reporter = AgenticsMetrics(workspace)
        report = reporter.generate_report(hours=24)
        # The valid line should be processed
        assert report.memory_hit_rate.total_recalls == 1

    def test_missing_ts_field_skipped(self, workspace: Path, sessions_dir: Path):
        path = sessions_dir / "no-ts.jsonl"
        path.write_text('{"event": "memory_recall", "chars_injected": 100}\n')
        reporter = AgenticsMetrics(workspace)
        # Should not raise; the event is skipped (no ts → ValueError on fromisoformat)
        report = reporter.generate_report(hours=24)
        assert report.sessions_scanned == 0
