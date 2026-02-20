"""Tests for the LlmMetrics session-log aggregator."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.llm_metrics import LlmMetrics, LlmMetricsReport


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Workspace with the standard directory layout."""
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    return tmp_path


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    """Write a session JSONL file for the given task."""
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _old_iso(hours: int = 48) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _llm_event(
    task_id: str = "t1",
    model: str = "claude-sonnet-4-20250514",
    tokens_in: int = 1000,
    tokens_out: int = 500,
    cost: float = 0.01,
    duration_ms: float = 2000,
    ts: str | None = None,
) -> dict:
    return {
        "ts": ts or _now_iso(),
        "event": "llm_complete",
        "task_id": task_id,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "duration_ms": duration_ms,
    }


class TestEmptyReport:
    def test_empty_report_no_sessions(self, workspace):
        """No session files → valid report with zeros."""
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, LlmMetricsReport)
        assert report.total_llm_calls == 0
        assert report.total_cost == 0.0
        assert report.total_tokens_in == 0
        assert report.total_tokens_out == 0
        assert report.overall_token_efficiency == 0.0
        assert report.model_tiers == []
        assert report.top_cost_tasks == []
        assert report.trends == []
        assert report.latency.sample_count == 0

    def test_empty_report_no_sessions_dir(self, tmp_path):
        """Missing sessions directory → valid report with zeros."""
        report = LlmMetrics(tmp_path).generate_report(hours=24)
        assert report.total_llm_calls == 0


class TestModelTierNormalization:
    def test_normalizes_opus(self, workspace):
        _write_session(workspace, "t1", [_llm_event(model="claude-opus-4-20250514")])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert len(report.model_tiers) == 1
        assert report.model_tiers[0].tier == "opus"

    def test_normalizes_sonnet(self, workspace):
        _write_session(workspace, "t1", [_llm_event(model="claude-sonnet-4-20250514")])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.model_tiers[0].tier == "sonnet"

    def test_normalizes_haiku(self, workspace):
        _write_session(workspace, "t1", [_llm_event(model="claude-haiku-3-5-20241022")])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.model_tiers[0].tier == "haiku"

    def test_unknown_model_passthrough(self, workspace):
        _write_session(workspace, "t1", [_llm_event(model="gpt-4o")])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.model_tiers[0].tier == "gpt-4o"


class TestModelTierAggregation:
    def test_per_tier_cost_and_share(self, workspace):
        """Multiple tiers → correct per-tier cost/call/share sums."""
        _write_session(workspace, "t1", [
            _llm_event(model="claude-sonnet-4-20250514", cost=0.03),
            _llm_event(model="claude-sonnet-4-20250514", cost=0.02),
            _llm_event(model="claude-haiku-3-5-20241022", cost=0.005),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)

        tiers_by_name = {t.tier: t for t in report.model_tiers}
        assert "sonnet" in tiers_by_name
        assert "haiku" in tiers_by_name

        sonnet = tiers_by_name["sonnet"]
        assert sonnet.call_count == 2
        assert round(sonnet.total_cost, 6) == 0.05

        haiku = tiers_by_name["haiku"]
        assert haiku.call_count == 1
        assert round(haiku.total_cost, 6) == 0.005

        # Cost share should sum to ~100%
        total_share = sum(t.cost_share_pct for t in report.model_tiers)
        assert abs(total_share - 100.0) < 0.2


class TestTaskCostSummary:
    def test_sorted_by_cost_desc(self, workspace):
        """Top 10 sorted by cost descending."""
        _write_session(workspace, "cheap", [_llm_event(task_id="cheap", cost=0.001)])
        _write_session(workspace, "expensive", [_llm_event(task_id="expensive", cost=0.10)])
        _write_session(workspace, "mid", [_llm_event(task_id="mid", cost=0.05)])

        report = LlmMetrics(workspace).generate_report(hours=24)
        costs = [t.total_cost for t in report.top_cost_tasks]
        assert costs == sorted(costs, reverse=True)
        assert report.top_cost_tasks[0].task_id == "expensive"

    def test_top_10_limit(self, workspace):
        """Only top 10 tasks returned even if more exist."""
        for i in range(15):
            _write_session(workspace, f"t{i}", [_llm_event(task_id=f"t{i}", cost=0.01 * i)])

        report = LlmMetrics(workspace).generate_report(hours=24)
        assert len(report.top_cost_tasks) == 10


class TestTokenEfficiency:
    def test_efficiency_calculation(self, workspace):
        """Token efficiency = tokens_out / tokens_in."""
        _write_session(workspace, "t1", [
            _llm_event(tokens_in=2000, tokens_out=1000),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.overall_token_efficiency == 0.5
        assert report.top_cost_tasks[0].token_efficiency == 0.5

    def test_zero_tokens_in(self, workspace):
        """0 input tokens → efficiency 0.0, not division error."""
        _write_session(workspace, "t1", [
            _llm_event(tokens_in=0, tokens_out=100),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.overall_token_efficiency == 0.0
        assert report.top_cost_tasks[0].token_efficiency == 0.0


class TestLatencyPercentiles:
    def test_known_durations(self, workspace):
        """Known durations → correct p50/p90/p99."""
        events = [
            _llm_event(duration_ms=ms)
            for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        ]
        _write_session(workspace, "t1", events)
        report = LlmMetrics(workspace).generate_report(hours=24)

        assert report.latency.sample_count == 10
        assert report.latency.avg_ms == 550.0
        assert report.latency.max_ms == 1000.0
        # p50 should be around the median
        assert 400 <= report.latency.p50_ms <= 600

    def test_single_sample(self, workspace):
        """n=1 edge case: all percentiles equal the single value."""
        _write_session(workspace, "t1", [_llm_event(duration_ms=42)])
        report = LlmMetrics(workspace).generate_report(hours=24)

        assert report.latency.sample_count == 1
        assert report.latency.p50_ms == 42.0
        assert report.latency.p90_ms == 42.0
        assert report.latency.p99_ms == 42.0
        assert report.latency.max_ms == 42.0


class TestFloatDurationMs:
    def test_float_duration_truncated_to_int(self, workspace):
        """Real data has float duration_ms — must not cause Pydantic validation error."""
        _write_session(workspace, "t1", [_llm_event(duration_ms=1234.567)])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.top_cost_tasks[0].total_duration_ms == 1234


class TestCostTrends:
    def test_hourly_buckets(self, workspace):
        """Events at different hours → separate buckets."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)

        _write_session(workspace, "t1", [
            _llm_event(ts=now.isoformat(), cost=0.05),
            _llm_event(ts=hour_ago.isoformat(), cost=0.03),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 2

    def test_same_hour_merged(self, workspace):
        """Events in the same hour → single bucket."""
        now = datetime.now(timezone.utc)
        _write_session(workspace, "t1", [
            _llm_event(ts=now.isoformat(), cost=0.01),
            _llm_event(ts=(now - timedelta(minutes=5)).isoformat(), cost=0.02),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert len(report.trends) == 1
        assert report.trends[0].call_count == 2
        assert round(report.trends[0].total_cost, 6) == 0.03


class TestTimeFiltering:
    def test_old_events_excluded(self, workspace):
        """Events older than the window are excluded."""
        _write_session(workspace, "t1", [
            _llm_event(ts=_old_iso(hours=48), cost=0.10),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.total_llm_calls == 0
        assert report.total_cost == 0.0


class TestNullCostHandling:
    def test_null_cost_treated_as_zero(self, workspace):
        """cost: None treated as 0.0."""
        event = _llm_event(cost=0.05)
        event["cost"] = None
        _write_session(workspace, "t1", [event])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.total_cost == 0.0
        assert report.total_llm_calls == 1


class TestNonLlmEventsIgnored:
    def test_mixed_event_types(self, workspace):
        """Only llm_complete events are counted."""
        _write_session(workspace, "t1", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "t1"},
            _llm_event(),
            {"ts": _now_iso(), "event": "memory_recall", "task_id": "t1", "chars_injected": 500},
            {"ts": _now_iso(), "event": "tool_call", "task_id": "t1", "tool": "Read"},
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.total_llm_calls == 1
        assert len(report.top_cost_tasks) == 1


class TestTasksWithLlmCalls:
    def test_counts_only_tasks_with_llm_events(self, workspace):
        """tasks_with_llm_calls excludes tasks that have no llm_complete events."""
        _write_session(workspace, "has_llm", [_llm_event(task_id="has_llm")])
        _write_session(workspace, "no_llm", [
            {"ts": _now_iso(), "event": "task_start", "task_id": "no_llm"},
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.tasks_with_llm_calls == 1


class TestPrimaryModel:
    def test_most_used_model_wins(self, workspace):
        """primary_model reflects the most-used tier for a task."""
        _write_session(workspace, "t1", [
            _llm_event(model="claude-sonnet-4-20250514"),
            _llm_event(model="claude-sonnet-4-20250514"),
            _llm_event(model="claude-haiku-3-5-20241022"),
        ])
        report = LlmMetrics(workspace).generate_report(hours=24)
        assert report.top_cost_tasks[0].primary_model == "sonnet"
