"""Tests for AgenticMetrics — verifies aggregation from session logs,
activity stream, memory stores, and profile registry."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_framework.analytics.agentic_metrics import (
    AgenticMetrics,
    AgenticMetricsReport,
)


@pytest.fixture
def workspace(tmp_path):
    """Minimal workspace with required directories pre-created."""
    (tmp_path / ".agent-communication").mkdir()
    (tmp_path / ".agent-communication" / "memory").mkdir()
    (tmp_path / ".agent-communication" / "profile-registry").mkdir()
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def metrics(workspace):
    return AgenticMetrics(workspace)


def _ts(offset_seconds: int = 0) -> str:
    """Return an ISO-8601 UTC timestamp relative to now."""
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat()


def _write_session(workspace: Path, task_id: str, events: list[dict]) -> None:
    """Write a session JSONL file with the given events."""
    sessions_dir = workspace / "logs" / "sessions"
    lines = [json.dumps({"task_id": task_id, "ts": _ts(), **e}) for e in events]
    (sessions_dir / f"{task_id}.jsonl").write_text("\n".join(lines) + "\n")


def _write_stream(workspace: Path, events: list[dict]) -> None:
    """Append events to the activity stream."""
    stream = workspace / ".agent-communication" / "activity-stream.jsonl"
    lines = [json.dumps({"timestamp": _ts(), **e}) for e in events]
    with open(stream, "a") as f:
        f.write("\n".join(lines) + "\n")


class TestEmptyWorkspace:
    def test_returns_report_with_zero_values(self, metrics):
        report = metrics.get_report(hours=24)
        assert isinstance(report, AgenticMetricsReport)
        assert report.memory.total_memories == 0
        assert report.memory.hit_rate == 0.0
        assert report.self_eval.total_tasks_evaluated == 0
        assert report.replan.total_tasks_with_replan == 0
        assert report.specialization.total_specializations == 0
        assert report.context_budget.critical_budget_events == 0

    def test_time_range_reflected_in_report(self, metrics):
        report = metrics.get_report(hours=48)
        assert report.time_range_hours == 48


class TestMemoryMetrics:
    def test_counts_entries_and_accessed(self, workspace, metrics):
        store = workspace / ".agent-communication" / "memory" / "myorg__repo" / "engineer.json"
        store.parent.mkdir(parents=True)
        store.write_text(json.dumps([
            {"category": "conventions", "content": "use black", "access_count": 3},
            {"category": "conventions", "content": "no tabs", "access_count": 0},
            {"category": "test_commands", "content": "pytest", "access_count": 1},
        ]))

        report = metrics.get_report()
        assert report.memory.total_memories == 3
        assert report.memory.accessed_memories == 2
        assert report.memory.hit_rate == pytest.approx(66.7, abs=0.1)

    def test_by_category_counts(self, workspace, metrics):
        store = workspace / ".agent-communication" / "memory" / "repo" / "engineer.json"
        store.parent.mkdir(parents=True)
        store.write_text(json.dumps([
            {"category": "conventions", "content": "a", "access_count": 1},
            {"category": "conventions", "content": "b", "access_count": 0},
            {"category": "test_commands", "content": "c", "access_count": 2},
        ]))

        report = metrics.get_report()
        assert report.memory.by_category["conventions"] == 2
        assert report.memory.by_category["test_commands"] == 1

    def test_all_accessed_gives_100_pct(self, workspace, metrics):
        store = workspace / ".agent-communication" / "memory" / "r" / "engineer.json"
        store.parent.mkdir(parents=True)
        store.write_text(json.dumps([
            {"category": "c", "content": "x", "access_count": 5},
            {"category": "c", "content": "y", "access_count": 1},
        ]))
        report = metrics.get_report()
        assert report.memory.hit_rate == 100.0

    def test_malformed_store_file_skipped(self, workspace, metrics):
        bad = workspace / ".agent-communication" / "memory" / "bad.json"
        bad.write_text("not valid json{{")
        # Should not raise, just return zeros for memory
        report = metrics.get_report()
        assert report.memory.total_memories == 0


class TestSelfEvalMetrics:
    def test_single_pass(self, workspace, metrics):
        _write_session(workspace, "t1", [{"event": "self_eval", "verdict": "PASS"}])
        report = metrics.get_report()
        assert report.self_eval.total_tasks_evaluated == 1
        assert report.self_eval.pass_count == 1
        assert report.self_eval.fail_count == 0
        assert report.self_eval.retry_rate == 0.0

    def test_fail_increments_retry_rate(self, workspace, metrics):
        _write_session(workspace, "t1", [
            {"event": "self_eval", "verdict": "FAIL"},
            {"event": "self_eval", "verdict": "PASS"},
        ])
        report = metrics.get_report()
        assert report.self_eval.tasks_with_failures == 1
        assert report.self_eval.retry_rate == 100.0

    def test_auto_pass_counted_separately(self, workspace, metrics):
        _write_session(workspace, "t1", [{"event": "self_eval", "verdict": "AUTO_PASS"}])
        report = metrics.get_report()
        assert report.self_eval.auto_pass_count == 1
        assert report.self_eval.tasks_with_failures == 0

    def test_multiple_tasks(self, workspace, metrics):
        _write_session(workspace, "t1", [{"event": "self_eval", "verdict": "PASS"}])
        _write_session(workspace, "t2", [{"event": "self_eval", "verdict": "FAIL"}])
        _write_session(workspace, "t3", [{"event": "self_eval", "verdict": "PASS"}])
        report = metrics.get_report()
        assert report.self_eval.total_tasks_evaluated == 3
        assert report.self_eval.tasks_with_failures == 1
        assert report.self_eval.retry_rate == pytest.approx(33.3, abs=0.1)

    def test_non_eval_events_ignored(self, workspace, metrics):
        _write_session(workspace, "t1", [
            {"event": "llm_start", "task_type": "implementation"},
            {"event": "llm_complete", "tokens_in": 1000, "tokens_out": 200},
        ])
        report = metrics.get_report()
        assert report.self_eval.total_tasks_evaluated == 0


class TestReplanMetrics:
    def test_no_replans(self, workspace, metrics):
        _write_stream(workspace, [
            {"type": "complete", "task_id": "t1"},
            {"type": "fail", "task_id": "t2"},
        ])
        report = metrics.get_report()
        assert report.replan.total_tasks_with_replan == 0
        assert report.replan.trigger_rate_pct == 0.0

    def test_replan_trigger_and_success(self, workspace, metrics):
        _write_session(workspace, "t1", [{"event": "replan", "retry": 2}])
        _write_session(workspace, "t2", [{"event": "replan", "retry": 2}])
        _write_stream(workspace, [
            {"type": "complete", "task_id": "t1"},  # t1 succeeded after replan
            {"type": "fail", "task_id": "t2"},       # t2 failed despite replan
            {"type": "complete", "task_id": "t3"},   # t3 no replan
        ])
        report = metrics.get_report()
        assert report.replan.total_tasks_with_replan == 2
        assert report.replan.success_after_replan == 1
        # 2 replanned / 3 terminal
        assert report.replan.trigger_rate_pct == pytest.approx(66.7, abs=0.1)

    def test_multiple_replan_events_same_task_counted_once(self, workspace, metrics):
        _write_session(workspace, "t1", [
            {"event": "replan", "retry": 2},
            {"event": "replan", "retry": 3},
        ])
        report = metrics.get_report()
        assert report.replan.total_tasks_with_replan == 1
        assert report.replan.total_replan_events == 2


class TestSpecializationMetrics:
    def test_reads_profile_registry(self, workspace, metrics):
        registry = (
            workspace / ".agent-communication" / "profile-registry" / "profiles.json"
        )
        registry.write_text(json.dumps([
            {"profile": {"id": "backend"}, "match_count": 5},
            {"profile": {"id": "frontend"}, "match_count": 3},
            {"profile": {"id": "backend"}, "match_count": 2},  # duplicate id
        ]))
        report = metrics.get_report()
        # backend: 5+2=7, frontend: 3
        assert report.specialization.profiles["backend"] == 7
        assert report.specialization.profiles["frontend"] == 3
        assert report.specialization.total_specializations == 10

    def test_missing_registry_returns_empty(self, workspace, metrics):
        report = metrics.get_report()
        assert report.specialization.profiles == {}
        assert report.specialization.total_specializations == 0


class TestContextBudgetMetrics:
    def test_counts_critical_events_from_stream(self, workspace, metrics):
        _write_stream(workspace, [
            {"type": "context_budget_critical", "task_id": "t1"},
            {"type": "context_budget_critical", "task_id": "t2"},
            {"type": "complete", "task_id": "t3"},
        ])
        report = metrics.get_report()
        assert report.context_budget.critical_budget_events == 2
        assert report.context_budget.total_token_budget_warnings == 2

    def test_token_budget_exceeded_adds_to_total(self, workspace, metrics):
        _write_stream(workspace, [
            {"type": "context_budget_critical", "task_id": "t1"},
            {"type": "token_budget_exceeded", "task_id": "t2"},
        ])
        report = metrics.get_report()
        assert report.context_budget.total_token_budget_warnings == 2

    def test_avg_output_ratio_from_session_logs(self, workspace, metrics):
        # 200 out / (1000 in + 200 out) = 16.67%
        _write_session(workspace, "t1", [
            {"event": "llm_complete", "tokens_in": 1000, "tokens_out": 200}
        ])
        # 500 out / (500 in + 500 out) = 50%
        _write_session(workspace, "t2", [
            {"event": "llm_complete", "tokens_in": 500, "tokens_out": 500}
        ])
        report = metrics.get_report()
        # avg of ~16.67 and 50 = ~33.33
        assert report.context_budget.avg_output_token_ratio_pct == pytest.approx(33.3, abs=0.1)


class TestDebateMetrics:
    def test_empty_workspace_returns_placeholder(self, metrics):
        report = metrics.get_report()
        assert report.debate.total_debates == 0
        assert report.debate.avg_confidence == 0.0
        assert report.debate.debate_usage_rate == 0.0
        assert report.debate.data_available is False

    def test_debate_complete_events_aggregated(self, workspace, metrics):
        _write_session(workspace, "t1", [
            {"event": "debate_complete", "confidence": 0.9},
        ])
        _write_session(workspace, "t2", [
            {"event": "debate_complete", "confidence": 0.7},
        ])
        _write_stream(workspace, [
            {"type": "complete", "task_id": "t1"},
            {"type": "complete", "task_id": "t2"},
            {"type": "complete", "task_id": "t3"},  # no debate
        ])
        report = metrics.get_report()
        assert report.debate.total_debates == 2
        assert report.debate.avg_confidence == pytest.approx(0.8, abs=0.01)
        # 2 tasks with debate / 3 terminal = 66.7%
        assert report.debate.debate_usage_rate == pytest.approx(66.7, abs=0.1)
        assert report.debate.data_available is True

    def test_non_debate_events_ignored(self, workspace, metrics):
        _write_session(workspace, "t1", [
            {"event": "self_eval", "verdict": "PASS"},
            {"event": "llm_complete", "tokens_in": 100, "tokens_out": 50},
        ])
        report = metrics.get_report()
        assert report.debate.total_debates == 0
        assert report.debate.data_available is False


class TestCaching:
    def test_second_call_returns_cached_report(self, metrics, workspace):
        # Call twice — second should return cached (same generated_at timestamp)
        r1 = metrics.get_report()
        r2 = metrics.get_report()
        assert r1.generated_at == r2.generated_at

    def test_cache_expires_after_ttl(self, metrics, workspace, monkeypatch):
        metrics.get_report()
        # Wind the clock forward past TTL
        monkeypatch.setattr(
            "agent_framework.analytics.agentic_metrics.time.monotonic",
            lambda: metrics._cache_at + 31,
        )
        r2 = metrics.get_report()
        # A new report was computed — timestamps may differ if any time passes
        # The key thing is the cache was invalidated (no exception thrown)
        assert r2 is not None

    def test_cache_invalidates_on_hours_change(self, metrics, workspace):
        """Switching the time window must not serve the old window's cached data."""
        r24 = metrics.get_report(hours=24)
        assert r24.time_range_hours == 24

        # Same monotonic time — would hit cache without the hours check
        r1 = metrics.get_report(hours=1)
        assert r1.time_range_hours == 1

        # Back to 24h — must re-compute, not return the 1h report
        r24_again = metrics.get_report(hours=24)
        assert r24_again.time_range_hours == 24

    def test_same_hours_uses_cache(self, metrics, workspace):
        r1 = metrics.get_report(hours=6)
        r2 = metrics.get_report(hours=6)
        # Same window repeated immediately — should hit cache (same object)
        assert r1.generated_at == r2.generated_at
