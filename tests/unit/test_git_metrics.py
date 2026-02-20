"""Tests for git behavioral metrics collection and reporting."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_framework.analytics.git_metrics import (
    GitMetrics,
    GitMetricsReport,
    GitMetricsSummary,
    TaskGitMetrics,
)
from agent_framework.core.chain_state import (
    ChainState,
    StepRecord,
    save_chain_state,
)
from agent_framework.core.git_operations import GitOperationsManager


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with standard directory structure."""
    (tmp_path / ".agent-communication" / "chain-state").mkdir(parents=True)
    (tmp_path / "logs" / "sessions").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def chain_state_with_commits(workspace):
    """A chain state with realistic commit and diff data."""
    state = ChainState(
        root_task_id="root-100",
        user_goal="Add user authentication",
        workflow="default",
        implementation_branch="agent/engineer/PROJ-100",
        steps=[
            StepRecord(
                step_id="plan",
                agent_id="architect",
                task_id="chain-root-100-plan-d1",
                completed_at="2026-02-19T10:00:00+00:00",
                summary="Planned auth feature",
                verdict="approved",
                plan={"objectives": ["Add auth"]},
            ),
            StepRecord(
                step_id="implement",
                agent_id="engineer",
                task_id="chain-root-100-implement-d2",
                completed_at="2026-02-19T10:15:00+00:00",
                summary="Implemented auth service",
                files_modified=["src/auth.py", "src/middleware.py"],
                commit_shas=["abc1234", "def5678"],
                lines_added=150,
                lines_removed=20,
            ),
            StepRecord(
                step_id="code_review",
                agent_id="architect",
                task_id="chain-root-100-code_review-d3",
                completed_at="2026-02-19T10:20:00+00:00",
                summary="Approved",
                verdict="approved",
            ),
            StepRecord(
                step_id="qa_review",
                agent_id="qa",
                task_id="chain-root-100-qa_review-d4",
                completed_at="2026-02-19T10:25:00+00:00",
                summary="All tests pass",
                verdict="approved",
            ),
        ],
    )
    save_chain_state(workspace, state)
    return state


def _write_session_events(workspace, task_id, events):
    """Write JSONL session log for a task."""
    path = workspace / "logs" / "sessions" / f"{task_id}.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# --- StepRecord round-trip with lines_added/lines_removed ---


class TestStepRecordDiffFields:
    def test_lines_added_removed_defaults(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="done",
        )
        assert record.lines_added == 0
        assert record.lines_removed == 0

    def test_lines_added_removed_set(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="done",
            lines_added=150, lines_removed=20,
        )
        assert record.lines_added == 150
        assert record.lines_removed == 20

    def test_roundtrip_preserves_diff_fields(self, workspace):
        state = ChainState(
            root_task_id="root-rt",
            user_goal="test roundtrip",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="done",
                    lines_added=300, lines_removed=45,
                ),
            ],
        )
        save_chain_state(workspace, state)

        from agent_framework.core.chain_state import load_chain_state
        loaded = load_chain_state(workspace, "root-rt")
        assert loaded.steps[0].lines_added == 300
        assert loaded.steps[0].lines_removed == 45

    def test_backward_compat_missing_diff_fields(self, workspace):
        """Old chain state JSON without lines_added/lines_removed defaults to 0."""
        state_dir = workspace / ".agent-communication" / "chain-state"
        data = {
            "root_task_id": "root-old",
            "user_goal": "test",
            "workflow": "default",
            "steps": [{
                "step_id": "implement",
                "agent_id": "engineer",
                "task_id": "t1",
                "completed_at": "2026-02-19T10:00:00+00:00",
                "summary": "done",
                "files_modified": ["a.py"],
                "commit_shas": ["abc"],
                # no lines_added or lines_removed
            }],
        }
        (state_dir / "root-old.json").write_text(json.dumps(data))

        from agent_framework.core.chain_state import load_chain_state
        loaded = load_chain_state(workspace, "root-old")
        assert loaded.steps[0].lines_added == 0
        assert loaded.steps[0].lines_removed == 0


# --- _log_push_event ---


class TestLogPushEvent:
    def test_logs_success(self):
        session_logger = MagicMock()
        mgr = GitOperationsManager(
            config=MagicMock(id="engineer"),
            workspace=Path("/tmp/test"),
            queue=MagicMock(),
            logger=MagicMock(),
            session_logger=session_logger,
        )
        mgr._log_push_event("feature/auth", success=True)

        session_logger.log.assert_called_once_with(
            "git_push", branch="feature/auth", success=True,
        )

    def test_logs_failure_with_error(self):
        session_logger = MagicMock()
        mgr = GitOperationsManager(
            config=MagicMock(id="engineer"),
            workspace=Path("/tmp/test"),
            queue=MagicMock(),
            logger=MagicMock(),
            session_logger=session_logger,
        )
        mgr._log_push_event("feature/auth", success=False, error="rejected")

        session_logger.log.assert_called_once_with(
            "git_push", branch="feature/auth", success=False, error="rejected",
        )

    def test_truncates_long_error(self):
        session_logger = MagicMock()
        mgr = GitOperationsManager(
            config=MagicMock(id="engineer"),
            workspace=Path("/tmp/test"),
            queue=MagicMock(),
            logger=MagicMock(),
            session_logger=session_logger,
        )
        long_error = "x" * 1000
        mgr._log_push_event("branch", success=False, error=long_error)

        call_kwargs = session_logger.log.call_args[1]
        assert len(call_kwargs["error"]) == 500

    def test_noop_without_session_logger(self):
        mgr = GitOperationsManager(
            config=MagicMock(id="engineer"),
            workspace=Path("/tmp/test"),
            queue=MagicMock(),
            logger=MagicMock(),
            session_logger=None,
        )
        # Should not raise
        mgr._log_push_event("branch", success=True)


# --- GitMetrics.generate_report() ---


class TestGitMetricsReport:
    def test_empty_workspace(self, workspace):
        metrics = GitMetrics(workspace)
        report = metrics.generate_report(hours=24)

        assert report.total_tasks == 0
        assert report.per_task == []
        assert report.summary.avg_commits_per_task == 0.0
        assert report.summary.push_success_rate == 0.0

    def test_single_task_with_commits(self, workspace, chain_state_with_commits):
        metrics = GitMetrics(workspace)
        report = metrics.generate_report(hours=24)

        assert report.total_tasks == 1
        task_m = report.per_task[0]
        assert task_m.root_task_id == "root-100"
        assert task_m.total_commits == 2
        assert task_m.total_insertions == 150
        assert task_m.total_deletions == 20

    def test_aggregates_commits_across_steps(self, workspace):
        """Multiple implement steps (fix cycle) sum commits."""
        state = ChainState(
            root_task_id="root-200",
            user_goal="Multi-step task",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="first pass",
                    commit_shas=["a1", "a2", "a3"], lines_added=200, lines_removed=10,
                ),
                StepRecord(
                    step_id="code_review", agent_id="architect", task_id="t2",
                    completed_at="2026-02-19T10:10:00+00:00", summary="needs fix",
                    verdict="needs_fix",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t3",
                    completed_at="2026-02-19T10:20:00+00:00", summary="second pass",
                    commit_shas=["b1"], lines_added=50, lines_removed=30,
                ),
            ],
        )
        save_chain_state(workspace, state)

        report = GitMetrics(workspace).generate_report(hours=24)
        task_m = report.per_task[0]
        assert task_m.total_commits == 4
        assert task_m.total_insertions == 250
        assert task_m.total_deletions == 40

    def test_push_events_aggregated(self, workspace, chain_state_with_commits):
        """Push events from session logs are counted."""
        now = datetime.now(timezone.utc)
        _write_session_events(workspace, "chain-root-100-implement-d2", [
            {"ts": now.isoformat(), "event": "git_push", "task_id": "chain-root-100-implement-d2", "branch": "feature", "success": True},
            {"ts": now.isoformat(), "event": "git_push", "task_id": "chain-root-100-implement-d2", "branch": "feature", "success": False, "error": "rejected"},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        task_m = report.per_task[0]
        assert task_m.push_attempts == 2
        assert task_m.push_successes == 1

    def test_push_success_rate_summary(self, workspace):
        """Summary push success rate across all sessions."""
        now = datetime.now(timezone.utc)
        _write_session_events(workspace, "task-a", [
            {"ts": now.isoformat(), "event": "git_push", "task_id": "task-a", "branch": "b1", "success": True},
            {"ts": now.isoformat(), "event": "git_push", "task_id": "task-a", "branch": "b1", "success": True},
        ])
        _write_session_events(workspace, "task-b", [
            {"ts": now.isoformat(), "event": "git_push", "task_id": "task-b", "branch": "b2", "success": False},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        # 2 successes out of 3 attempts
        assert report.summary.push_success_rate == pytest.approx(0.667, abs=0.001)

    def test_summary_lines_per_commit(self, workspace, chain_state_with_commits):
        """Average insertions/deletions per commit."""
        report = GitMetrics(workspace).generate_report(hours=24)
        # 150 insertions / 2 commits = 75.0
        assert report.summary.avg_insertions_per_commit == 75.0
        # 20 deletions / 2 commits = 10.0
        assert report.summary.avg_deletions_per_commit == 10.0

    def test_summary_zero_commits(self, workspace):
        """No commits → zero averages, no division by zero."""
        state = ChainState(
            root_task_id="root-empty",
            user_goal="Empty task",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="planned",
                ),
            ],
        )
        save_chain_state(workspace, state)

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.summary.avg_insertions_per_commit == 0.0
        assert report.summary.avg_deletions_per_commit == 0.0

    def test_report_model_structure(self, workspace, chain_state_with_commits):
        report = GitMetrics(workspace).generate_report(hours=24)
        assert isinstance(report, GitMetricsReport)
        assert isinstance(report.summary, GitMetricsSummary)
        assert all(isinstance(t, TaskGitMetrics) for t in report.per_task)
        assert report.time_range_hours == 24
        assert report.generated_at is not None


# --- Edit-to-commit timing ---


class TestEditToCommitTiming:
    def test_normal_case(self, workspace, chain_state_with_commits):
        """Edit tool followed by git commit → positive delta."""
        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        _write_session_events(workspace, "chain-root-100-implement-d2", [
            {"ts": base.isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Edit", "input": {"file_path": "src/auth.py"}},
            {"ts": (base + timedelta(seconds=5)).isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Write", "input": {"file_path": "src/middleware.py"}},
            {"ts": (base + timedelta(seconds=120)).isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Bash", "input": {"command": "git commit -m 'Add auth'"}},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        task_m = report.per_task[0]
        # 120 seconds from first edit to first commit
        assert task_m.first_edit_to_commit_secs == 120.0

    def test_no_edit_events(self, workspace, chain_state_with_commits):
        """No Write/Edit calls → None."""
        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        _write_session_events(workspace, "chain-root-100-implement-d2", [
            {"ts": base.isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Read", "input": {"file_path": "src/auth.py"}},
            {"ts": (base + timedelta(seconds=60)).isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Bash", "input": {"command": "git commit -m 'fix'"}},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.per_task[0].first_edit_to_commit_secs is None

    def test_no_commit_events(self, workspace, chain_state_with_commits):
        """Edit calls but no git commit → None."""
        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        _write_session_events(workspace, "chain-root-100-implement-d2", [
            {"ts": base.isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Edit", "input": {"file_path": "src/auth.py"}},
            {"ts": (base + timedelta(seconds=60)).isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Bash", "input": {"command": "python -m pytest"}},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.per_task[0].first_edit_to_commit_secs is None

    def test_no_session_events(self, workspace, chain_state_with_commits):
        """No session log at all → None."""
        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.per_task[0].first_edit_to_commit_secs is None

    def test_commit_before_edit_clamps_to_zero(self, workspace, chain_state_with_commits):
        """Clock skew where commit timestamp < edit timestamp → 0.0."""
        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        _write_session_events(workspace, "chain-root-100-implement-d2", [
            {"ts": (base + timedelta(seconds=60)).isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Edit", "input": {"file_path": "src/auth.py"}},
            {"ts": base.isoformat(), "event": "tool_call", "task_id": "chain-root-100-implement-d2", "tool": "Bash", "input": {"command": "git commit -m 'fix'"}},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.per_task[0].first_edit_to_commit_secs == 0.0

    def test_latency_percentiles(self, workspace):
        """P50 and P90 computed from multiple tasks."""
        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        for i, latency in enumerate([30, 60, 90, 120, 180, 240, 300, 360, 420, 480]):
            root_id = f"root-lat-{i}"
            state = ChainState(
                root_task_id=root_id,
                user_goal="test",
                workflow="default",
                steps=[
                    StepRecord(
                        step_id="implement", agent_id="engineer",
                        task_id=f"t-{i}",
                        completed_at="2026-02-19T10:00:00+00:00",
                        summary="done", commit_shas=["abc"],
                    ),
                ],
            )
            save_chain_state(workspace, state)

            _write_session_events(workspace, f"t-{i}", [
                {"ts": base.isoformat(), "event": "tool_call", "task_id": f"t-{i}", "tool": "Edit", "input": {"file_path": "a.py"}},
                {"ts": (base + timedelta(seconds=latency)).isoformat(), "event": "tool_call", "task_id": f"t-{i}", "tool": "Bash", "input": {"command": "git commit -m 'x'"}},
            ])

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.summary.p50_edit_to_commit_secs is not None
        assert report.summary.p90_edit_to_commit_secs is not None
        # P50 of [30,60,90,120,180,240,300,360,420,480] = 210
        assert report.summary.p50_edit_to_commit_secs == 210.0

    def test_single_task_latency_uses_value_for_p90(self, workspace):
        """With only one task, p90 falls back to the single value."""
        state = ChainState(
            root_task_id="root-single",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t-single",
                    completed_at="2026-02-19T10:00:00+00:00", summary="done",
                    commit_shas=["abc"],
                ),
            ],
        )
        save_chain_state(workspace, state)

        base = datetime.now(timezone.utc) - timedelta(minutes=30)
        _write_session_events(workspace, "t-single", [
            {"ts": base.isoformat(), "event": "tool_call", "task_id": "t-single", "tool": "Write", "input": {"file_path": "a.py"}},
            {"ts": (base + timedelta(seconds=100)).isoformat(), "event": "tool_call", "task_id": "t-single", "tool": "Bash", "input": {"command": "git commit -m 'x'"}},
        ])

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.summary.p50_edit_to_commit_secs == 100.0
        assert report.summary.p90_edit_to_commit_secs == 100.0


# --- Cutoff filtering ---


class TestCutoffFiltering:
    def test_old_chain_state_excluded(self, workspace):
        """Chain state file older than cutoff is not included."""
        import os

        state = ChainState(
            root_task_id="root-old",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t-old",
                    completed_at="2026-02-17T10:00:00+00:00", summary="done",
                    commit_shas=["old"],
                ),
            ],
        )
        save_chain_state(workspace, state)

        # Set mtime to 3 days ago
        old_path = workspace / ".agent-communication" / "chain-state" / "root-old.json"
        old_time = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
        os.utime(old_path, (old_time, old_time))

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.total_tasks == 0


# --- Multiple tasks ---


class TestMultipleTasks:
    def test_multiple_tasks_aggregated(self, workspace):
        for i in range(3):
            state = ChainState(
                root_task_id=f"root-multi-{i}",
                user_goal=f"Task {i}",
                workflow="default",
                steps=[
                    StepRecord(
                        step_id="implement", agent_id="engineer",
                        task_id=f"t-multi-{i}",
                        completed_at="2026-02-19T10:00:00+00:00",
                        summary="done",
                        commit_shas=[f"sha-{i}-a", f"sha-{i}-b"],
                        lines_added=(i + 1) * 100,
                        lines_removed=(i + 1) * 10,
                    ),
                ],
            )
            save_chain_state(workspace, state)

        report = GitMetrics(workspace).generate_report(hours=24)
        assert report.total_tasks == 3
        assert report.summary.avg_commits_per_task == 2.0
        # Total insertions: 100+200+300=600, total commits: 6 → avg 100
        assert report.summary.avg_insertions_per_commit == 100.0
