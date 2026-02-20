"""Tests for workflow chain metrics aggregation."""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent_framework.analytics.chain_metrics import (
    ChainMetrics,
    ChainMetricsReport,
)
from agent_framework.core.chain_state import ChainState, StepRecord, save_chain_state


def _make_step(
    step_id: str,
    agent_id: str = "engineer",
    task_id: str = "task-1",
    completed_at: str | None = None,
    verdict: str | None = None,
    error: str | None = None,
    files_modified: list[str] | None = None,
) -> StepRecord:
    return StepRecord(
        step_id=step_id,
        agent_id=agent_id,
        task_id=task_id,
        completed_at=completed_at or datetime.now(timezone.utc).isoformat(),
        summary=f"Step {step_id} completed",
        verdict=verdict,
        error=error,
        files_modified=files_modified or [],
    )


def _make_chain_state(
    root_task_id: str,
    steps: list[StepRecord],
    workflow: str = "default",
    attempt: int = 1,
) -> ChainState:
    return ChainState(
        root_task_id=root_task_id,
        user_goal="Test task",
        workflow=workflow,
        steps=steps,
        attempt=attempt,
    )


class TestChainMetricsEmpty:
    def test_empty_directory(self, tmp_path):
        """No chain state files → zero-valued report."""
        metrics = ChainMetrics(tmp_path)
        report = metrics.generate_report(hours=24)

        assert report.total_chains == 0
        assert report.completed_chains == 0
        assert report.chain_completion_rate == 0.0
        assert report.avg_chain_depth == 0.0
        assert report.avg_files_modified == 0.0
        assert report.avg_attempts == 0.0
        assert report.step_type_metrics == []
        assert report.top_failing_steps == []
        assert report.recent_chains == []

    def test_nonexistent_directory(self, tmp_path):
        """Workspace with no .agent-communication dir at all."""
        metrics = ChainMetrics(tmp_path / "nonexistent")
        report = metrics.generate_report(hours=24)
        assert report.total_chains == 0


class TestSingleChain:
    def test_full_five_step_chain(self, tmp_path):
        """Full plan→implement→code_review→qa_review→create_pr chain."""
        base_time = datetime.now(timezone.utc)
        steps = [
            _make_step("plan", agent_id="architect",
                       completed_at=(base_time + timedelta(minutes=1)).isoformat(),
                       verdict="approved"),
            _make_step("implement", agent_id="engineer",
                       completed_at=(base_time + timedelta(minutes=5)).isoformat(),
                       files_modified=["src/foo.py", "src/bar.py"]),
            _make_step("code_review", agent_id="architect",
                       completed_at=(base_time + timedelta(minutes=7)).isoformat(),
                       verdict="approved"),
            _make_step("qa_review", agent_id="qa",
                       completed_at=(base_time + timedelta(minutes=10)).isoformat(),
                       verdict="approved"),
            _make_step("create_pr", agent_id="engineer",
                       completed_at=(base_time + timedelta(minutes=11)).isoformat()),
        ]

        state = _make_chain_state("root-1", steps)
        save_chain_state(tmp_path, state)

        metrics = ChainMetrics(tmp_path)
        report = metrics.generate_report(hours=24)

        assert report.total_chains == 1
        assert report.completed_chains == 1
        assert report.chain_completion_rate == 100.0
        assert report.avg_chain_depth == 5.0
        assert report.avg_files_modified == 2.0
        assert len(report.step_type_metrics) == 5
        assert report.recent_chains[0].root_task_id == "root-1"

    def test_single_step_chain_approved(self, tmp_path):
        """Chain with single approved plan step — completed (has approval, no active step)."""
        steps = [_make_step("plan", agent_id="architect", verdict="approved")]
        state = _make_chain_state("root-single", steps)
        save_chain_state(tmp_path, state)

        metrics = ChainMetrics(tmp_path)
        report = metrics.generate_report(hours=24)

        assert report.total_chains == 1
        assert report.completed_chains == 1
        summary = report.recent_chains[0]
        assert summary.total_duration_seconds == 0.0

    def test_single_step_chain_no_verdict(self, tmp_path):
        """Chain with single step and no verdict — not completed (no approval signal)."""
        steps = [_make_step("implement", agent_id="engineer")]
        state = _make_chain_state("root-no-verdict", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)
        assert report.completed_chains == 0


class TestStepFailureTracking:
    def test_error_counted_as_failure(self, tmp_path):
        steps = [
            _make_step("plan", agent_id="architect", verdict="approved"),
            _make_step("implement", agent_id="engineer", error="Context exhausted"),
        ]
        state = _make_chain_state("root-err", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        impl_metrics = next(m for m in report.step_type_metrics if m.step_id == "implement")
        assert impl_metrics.failure_count == 1
        assert impl_metrics.success_count == 0
        assert len(report.top_failing_steps) >= 1

    def test_needs_fix_counted_as_failure(self, tmp_path):
        steps = [
            _make_step("plan", agent_id="architect", verdict="approved"),
            _make_step("implement", agent_id="engineer",
                       files_modified=["src/foo.py"]),
            _make_step("code_review", agent_id="architect", verdict="needs_fix"),
        ]
        state = _make_chain_state("root-fix", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        review_metrics = next(m for m in report.step_type_metrics if m.step_id == "code_review")
        assert review_metrics.failure_count == 1
        assert review_metrics.success_rate == 0.0

        # Chain not completed (has needs_fix verdict and no create_pr)
        assert report.completed_chains == 0

    def test_approved_not_counted_as_failure(self, tmp_path):
        steps = [
            _make_step("qa_review", agent_id="qa", verdict="approved"),
        ]
        state = _make_chain_state("root-ok", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        qa_metrics = next(m for m in report.step_type_metrics if m.step_id == "qa_review")
        assert qa_metrics.failure_count == 0
        assert qa_metrics.success_count == 1


class TestDurationComputation:
    def test_inter_step_timing(self, tmp_path):
        """Duration is delta between consecutive step completed_at timestamps."""
        base = datetime(2026, 2, 20, 10, 0, 0, tzinfo=timezone.utc)
        steps = [
            _make_step("plan", completed_at=base.isoformat()),
            _make_step("implement", completed_at=(base + timedelta(seconds=120)).isoformat()),
            _make_step("code_review", completed_at=(base + timedelta(seconds=180)).isoformat()),
        ]
        state = _make_chain_state("root-dur", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        # implement duration = 120s (from plan), code_review = 60s (from implement)
        impl_metrics = next(m for m in report.step_type_metrics if m.step_id == "implement")
        assert impl_metrics.avg_duration_seconds == pytest.approx(120.0)

        review_metrics = next(m for m in report.step_type_metrics if m.step_id == "code_review")
        assert review_metrics.avg_duration_seconds == pytest.approx(60.0)

        # plan has no duration (first step)
        plan_metrics = next(m for m in report.step_type_metrics if m.step_id == "plan")
        assert plan_metrics.avg_duration_seconds == 0.0

    def test_chain_total_duration(self, tmp_path):
        base = datetime(2026, 2, 20, 10, 0, 0, tzinfo=timezone.utc)
        steps = [
            _make_step("plan", completed_at=base.isoformat()),
            _make_step("create_pr", completed_at=(base + timedelta(seconds=600)).isoformat()),
        ]
        state = _make_chain_state("root-total", steps)
        save_chain_state(tmp_path, state)

        report = ChainMetrics(tmp_path).generate_report(hours=24)
        summary = report.recent_chains[0]
        assert summary.total_duration_seconds == pytest.approx(600.0)


class TestTimeRangeFiltering:
    def test_old_files_excluded(self, tmp_path):
        """Chain state files older than the cutoff window are excluded."""
        steps = [_make_step("plan", agent_id="architect")]
        state = _make_chain_state("root-old", steps)
        save_chain_state(tmp_path, state)

        # Set file mtime to 48 hours ago
        chain_file = tmp_path / ".agent-communication" / "chain-state" / "root-old.json"
        old_time = time.time() - (48 * 3600)
        os.utime(chain_file, (old_time, old_time))

        report = ChainMetrics(tmp_path).generate_report(hours=24)
        assert report.total_chains == 0

    def test_recent_files_included(self, tmp_path):
        """Chain state files within the cutoff window are included."""
        steps = [_make_step("plan", agent_id="architect")]
        state = _make_chain_state("root-new", steps)
        save_chain_state(tmp_path, state)

        # File just created — within 24h window
        report = ChainMetrics(tmp_path).generate_report(hours=24)
        assert report.total_chains == 1


class TestMultipleChains:
    def test_aggregation_across_chains(self, tmp_path):
        """Metrics correctly aggregate across 3 chains."""
        base = datetime.now(timezone.utc)

        # Chain 1: completed, 2 files, attempt 1
        save_chain_state(tmp_path, _make_chain_state("root-a", [
            _make_step("plan", completed_at=base.isoformat()),
            _make_step("implement", files_modified=["a.py", "b.py"],
                       completed_at=(base + timedelta(seconds=60)).isoformat()),
            _make_step("create_pr",
                       completed_at=(base + timedelta(seconds=120)).isoformat()),
        ]))

        # Chain 2: completed, 1 file, attempt 2
        save_chain_state(tmp_path, _make_chain_state("root-b", [
            _make_step("plan", completed_at=base.isoformat()),
            _make_step("implement", files_modified=["c.py"],
                       completed_at=(base + timedelta(seconds=90)).isoformat()),
            _make_step("create_pr",
                       completed_at=(base + timedelta(seconds=150)).isoformat()),
        ], attempt=2))

        # Chain 3: incomplete (failed at review), 3 files, attempt 1
        save_chain_state(tmp_path, _make_chain_state("root-c", [
            _make_step("plan", completed_at=base.isoformat()),
            _make_step("implement", files_modified=["d.py", "e.py", "f.py"],
                       completed_at=(base + timedelta(seconds=45)).isoformat()),
            _make_step("code_review", verdict="needs_fix",
                       completed_at=(base + timedelta(seconds=100)).isoformat()),
        ]))

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        assert report.total_chains == 3
        assert report.completed_chains == 2
        assert report.chain_completion_rate == pytest.approx(200 / 3)
        assert report.avg_chain_depth == pytest.approx(3.0)  # (3+3+3)/3
        assert report.avg_files_modified == pytest.approx(2.0)  # (2+1+3)/3
        assert report.avg_attempts == pytest.approx(4 / 3)  # (1+2+1)/3

    def test_step_metrics_aggregate_across_chains(self, tmp_path):
        """Same step type from different chains aggregates correctly."""
        base = datetime.now(timezone.utc)

        for i, root_id in enumerate(["root-x", "root-y"]):
            save_chain_state(tmp_path, _make_chain_state(root_id, [
                _make_step("plan", completed_at=base.isoformat()),
                _make_step("implement",
                           completed_at=(base + timedelta(seconds=60 * (i + 1))).isoformat()),
            ]))

        report = ChainMetrics(tmp_path).generate_report(hours=24)

        impl_metrics = next(m for m in report.step_type_metrics if m.step_id == "implement")
        assert impl_metrics.total_count == 2
        # Durations: 60s (root-x) and 120s (root-y) → avg=90s
        assert impl_metrics.avg_duration_seconds == pytest.approx(90.0)

    def test_recent_chains_limited_to_ten(self, tmp_path):
        """recent_chains returns at most 10 entries."""
        base = datetime.now(timezone.utc)

        for i in range(15):
            save_chain_state(tmp_path, _make_chain_state(f"root-{i:03d}", [
                _make_step("plan", completed_at=base.isoformat()),
            ]))

        report = ChainMetrics(tmp_path).generate_report(hours=24)
        assert len(report.recent_chains) == 10


class TestReportModel:
    def test_serialization_roundtrip(self, tmp_path):
        """Report can be serialized to JSON and fields are correct types."""
        steps = [_make_step("plan", agent_id="architect")]
        save_chain_state(tmp_path, _make_chain_state("root-ser", steps))

        report = ChainMetrics(tmp_path).generate_report(hours=24)
        json_str = report.model_dump_json()
        parsed = ChainMetricsReport.model_validate_json(json_str)

        assert parsed.total_chains == 1
        assert isinstance(parsed.generated_at, datetime)
