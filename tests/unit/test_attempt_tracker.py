"""Tests for attempt_tracker module — disk-persisted retry awareness."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.attempt_tracker import (
    AttemptHistory,
    AttemptRecord,
    load_attempt_history,
    save_attempt_history,
    record_attempt,
    render_for_retry,
    get_last_pushed_branch,
    _classify_error,
    _truncate_error,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(**overrides):
    defaults = dict(
        id="task-abc123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.now(timezone.utc),
        title="Implement feature",
        description="Add feature",
        context={"github_repo": "org/repo"},
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_record(**overrides):
    defaults = dict(
        attempt_number=1,
        started_at=datetime.now(timezone.utc).isoformat(),
        ended_at=datetime.now(timezone.utc).isoformat(),
        agent_id="engineer",
        branch="agent/engineer/task-abc123",
        commit_sha="abc1234",
        pushed=True,
        files_modified=["src/feature.py", "tests/test_feature.py"],
        commit_count=3,
        insertions=200,
        deletions=10,
        error="Circuit breaker tripped",
        error_type="circuit_breaker",
    )
    defaults.update(overrides)
    return AttemptRecord(**defaults)


class TestAttemptHistoryDiskIO:
    """Load/save round-trips through disk correctly."""

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_attempt_history(tmp_path, "nonexistent") is None

    def test_save_and_load_roundtrip(self, tmp_path):
        record = _make_record()
        history = AttemptHistory(task_id="task-1", attempts=[record])

        save_attempt_history(tmp_path, history)
        loaded = load_attempt_history(tmp_path, "task-1")

        assert loaded is not None
        assert loaded.task_id == "task-1"
        assert len(loaded.attempts) == 1
        assert loaded.attempts[0].branch == record.branch
        assert loaded.attempts[0].commit_sha == record.commit_sha
        assert loaded.attempts[0].pushed is True
        assert loaded.attempts[0].insertions == 200

    def test_multiple_attempts_append(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(attempt_number=1),
        ])
        save_attempt_history(tmp_path, history)

        # Append second attempt
        loaded = load_attempt_history(tmp_path, "task-1")
        loaded.attempts.append(_make_record(attempt_number=2, error="Context exhausted"))
        save_attempt_history(tmp_path, loaded)

        reloaded = load_attempt_history(tmp_path, "task-1")
        assert len(reloaded.attempts) == 2
        assert reloaded.attempts[0].attempt_number == 1
        assert reloaded.attempts[1].attempt_number == 2

    def test_schema_evolution_tolerates_unknown_fields(self, tmp_path):
        """Unknown fields in JSON don't crash deserialization."""
        data = {
            "task_id": "task-1",
            "attempts": [{
                "attempt_number": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "agent_id": "engineer",
                "branch": "feature/x",
                "future_field": "should be ignored",
                "another_new_field": 42,
            }],
        }
        path = tmp_path / ".agent-communication" / "attempt-history" / "task-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

        loaded = load_attempt_history(tmp_path, "task-1")
        assert loaded is not None
        assert loaded.attempts[0].branch == "feature/x"

    def test_corrupt_json_returns_none(self, tmp_path):
        path = tmp_path / ".agent-communication" / "attempt-history" / "task-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json")

        assert load_attempt_history(tmp_path, "task-1") is None

    def test_missing_task_id_returns_none(self, tmp_path):
        path = tmp_path / ".agent-communication" / "attempt-history" / "task-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"attempts": []}))

        assert load_attempt_history(tmp_path, "task-1") is None


class TestRecordAttempt:
    """record_attempt() commits, pushes, collects stats, and persists."""

    def _mock_git(self, **overrides):
        """Build a side_effect map for run_git_command calls."""
        defaults = {
            "rev-parse --abbrev-ref HEAD": MagicMock(returncode=0, stdout="agent/engineer/task-1\n"),
            "status --porcelain": MagicMock(returncode=0, stdout=" M src/file.py\n"),
            "add -A": MagicMock(returncode=0),
            "commit": MagicMock(returncode=0),
            "push": MagicMock(returncode=0),
            "rev-parse --short HEAD": MagicMock(returncode=0, stdout="abc1234\n"),
            "rev-parse --verify": MagicMock(returncode=0),
            "rev-list --count": MagicMock(returncode=0, stdout="3\n"),
            "diff --stat": MagicMock(returncode=0, stdout=" 2 files changed, 100 insertions(+), 5 deletions(-)\n"),
            "diff --name-only": MagicMock(returncode=0, stdout="src/file.py\ntests/test_file.py\n"),
        }
        defaults.update(overrides)
        return defaults

    def _git_side_effect(self, mock_map):
        def side_effect(args, **kwargs):
            key = " ".join(args[:3]) if len(args) >= 3 else " ".join(args)
            for pattern, result in mock_map.items():
                if key.startswith(pattern.split()[0]) and all(p in key for p in pattern.split()):
                    return result
            return MagicMock(returncode=1, stdout="", stderr="")
        return side_effect

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_commits_and_pushes(self, mock_git, tmp_path):
        mock_map = self._mock_git()
        mock_git.side_effect = self._git_side_effect(mock_map)

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
            error="Circuit breaker tripped",
        )

        assert result is not None
        assert result.branch == "agent/engineer/task-1"
        assert result.commit_sha == "abc1234"
        assert result.pushed is True
        assert result.error == "Circuit breaker tripped"
        assert result.error_type == "circuit_breaker"

        # Verify persisted to disk
        loaded = load_attempt_history(tmp_path, task.id)
        assert loaded is not None
        assert len(loaded.attempts) == 1

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_skips_commit_when_clean(self, mock_git, tmp_path):
        mock_map = self._mock_git(**{
            "status --porcelain": MagicMock(returncode=0, stdout=""),
        })
        mock_git.side_effect = self._git_side_effect(mock_map)

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is not None
        # No commit made but still records attempt
        assert result.pushed is True

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_returns_none_on_main_branch(self, mock_git, tmp_path):
        mock_git.return_value = MagicMock(returncode=0, stdout="main\n")

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is None

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_returns_none_on_detached_head(self, mock_git, tmp_path):
        mock_git.return_value = MagicMock(returncode=0, stdout="HEAD\n")

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is None

    def test_record_attempt_returns_none_missing_dir(self, tmp_path):
        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=tmp_path / "nonexistent",
        )

        assert result is None

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_push_failure_still_persists(self, mock_git, tmp_path):
        mock_map = self._mock_git(**{
            "push": MagicMock(returncode=1, stderr="rejected"),
        })
        mock_git.side_effect = self._git_side_effect(mock_map)

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is not None
        assert result.pushed is False
        # Still persisted
        loaded = load_attempt_history(tmp_path, task.id)
        assert loaded is not None

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_exception_returns_none(self, mock_git, tmp_path):
        mock_git.side_effect = RuntimeError("git broken")

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is None

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_idempotent_after_auto_commit(self, mock_git, tmp_path):
        """If safety_commit already ran, status is clean — skip commit, still push."""
        mock_map = self._mock_git(**{
            "status --porcelain": MagicMock(returncode=0, stdout=""),
        })
        mock_git.side_effect = self._git_side_effect(mock_map)

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
        )

        assert result is not None
        assert result.pushed is True


class TestRenderForRetry:
    """render_for_retry() formats attempt history for the retry prompt."""

    def test_renders_attempt_history_with_stats(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(
                attempt_number=1,
                branch="agent/engineer/task-1",
                commit_count=5,
                insertions=320,
                deletions=12,
                pushed=True,
                error="Circuit breaker tripped",
                files_modified=["src/router.py", "src/model.py", "tests/test_router.py"],
            ),
        ])
        save_attempt_history(tmp_path, history)

        rendered = render_for_retry(tmp_path, "task-1")

        assert "Previous Attempt History" in rendered
        assert "Attempt 1" in rendered
        assert "agent/engineer/task-1" in rendered
        assert "5 commits" in rendered
        assert "320+/12-" in rendered
        assert "pushed=True" in rendered
        assert "Circuit breaker tripped" in rendered
        assert "src/router.py" in rendered

    def test_empty_history_returns_empty_string(self, tmp_path):
        assert render_for_retry(tmp_path, "nonexistent") == ""

    def test_no_attempts_returns_empty_string(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[])
        save_attempt_history(tmp_path, history)
        assert render_for_retry(tmp_path, "task-1") == ""

    def test_truncates_long_file_lists(self, tmp_path):
        files = [f"src/file_{i}.py" for i in range(30)]
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(files_modified=files, pushed=True),
        ])
        save_attempt_history(tmp_path, history)

        rendered = render_for_retry(tmp_path, "task-1")

        assert "and 10 more" in rendered

    def test_last_pushed_branch_directive(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(branch="feature/x", pushed=True),
        ])
        save_attempt_history(tmp_path, history)

        rendered = render_for_retry(tmp_path, "task-1")

        assert "`feature/x`" in rendered
        assert "git log --oneline" in rendered


class TestGetLastPushedBranch:
    """get_last_pushed_branch() returns the correct branch."""

    def test_returns_most_recent_pushed_branch(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(attempt_number=1, branch="branch-v1", pushed=True),
            _make_record(attempt_number=2, branch="branch-v2", pushed=True),
        ])
        save_attempt_history(tmp_path, history)

        assert get_last_pushed_branch(tmp_path, "task-1") == "branch-v2"

    def test_returns_none_when_no_pushes(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(attempt_number=1, pushed=False),
        ])
        save_attempt_history(tmp_path, history)

        assert get_last_pushed_branch(tmp_path, "task-1") is None

    def test_returns_none_when_no_history(self, tmp_path):
        assert get_last_pushed_branch(tmp_path, "nonexistent") is None

    def test_skips_unpushed_returns_earlier_pushed(self, tmp_path):
        history = AttemptHistory(task_id="task-1", attempts=[
            _make_record(attempt_number=1, branch="branch-v1", pushed=True),
            _make_record(attempt_number=2, branch="branch-v2", pushed=False),
        ])
        save_attempt_history(tmp_path, history)

        assert get_last_pushed_branch(tmp_path, "task-1") == "branch-v1"


class TestErrorClassification:
    """_classify_error categorizes errors correctly."""

    def test_circuit_breaker(self):
        assert _classify_error("Circuit breaker tripped: 20 consecutive") == "circuit_breaker"

    def test_context_exhausted(self):
        assert _classify_error("Context window exhausted") == "context_exhausted"

    def test_interrupted(self):
        assert _classify_error("Interrupted during LLM execution") == "interrupted"

    def test_timeout(self):
        assert _classify_error("Request timeout after 300s") == "timeout"

    def test_other(self):
        assert _classify_error("Some weird error") == "other"

    def test_none(self):
        assert _classify_error(None) is None


class TestTruncateError:
    """_truncate_error caps error message length."""

    def test_short_error_unchanged(self):
        assert _truncate_error("short") == "short"

    def test_long_error_truncated(self):
        long = "x" * 600
        result = _truncate_error(long, max_len=500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_none_returns_none(self):
        assert _truncate_error(None) is None


class TestCostFields:
    """Cost fields round-trip through disk and render correctly."""

    def test_cost_fields_roundtrip(self, tmp_path):
        record = _make_record(input_tokens=5000, output_tokens=1200, cost_usd=0.42)
        history = AttemptHistory(task_id="task-cost", attempts=[record])

        save_attempt_history(tmp_path, history)
        loaded = load_attempt_history(tmp_path, "task-cost")

        assert loaded is not None
        a = loaded.attempts[0]
        assert a.input_tokens == 5000
        assert a.output_tokens == 1200
        assert a.cost_usd == 0.42

    def test_old_format_without_cost_fields(self, tmp_path):
        """JSON from before cost fields were added loads with defaults."""
        data = {
            "task_id": "task-old",
            "attempts": [{
                "attempt_number": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "agent_id": "engineer",
                "branch": "feature/x",
                "pushed": True,
                "error": "timeout",
            }],
        }
        path = tmp_path / ".agent-communication" / "attempt-history" / "task-old.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

        loaded = load_attempt_history(tmp_path, "task-old")
        assert loaded is not None
        a = loaded.attempts[0]
        assert a.input_tokens == 0
        assert a.output_tokens == 0
        assert a.cost_usd is None

    def test_render_includes_cost(self, tmp_path):
        history = AttemptHistory(task_id="task-cost", attempts=[
            _make_record(cost_usd=1.23, pushed=True),
        ])
        save_attempt_history(tmp_path, history)

        rendered = render_for_retry(tmp_path, "task-cost")

        assert "cost=$1.23" in rendered

    def test_render_omits_cost_when_none(self, tmp_path):
        history = AttemptHistory(task_id="task-nocost", attempts=[
            _make_record(cost_usd=None, pushed=True),
        ])
        save_attempt_history(tmp_path, history)

        rendered = render_for_retry(tmp_path, "task-nocost")

        assert "cost=" not in rendered

    @patch("agent_framework.core.attempt_tracker.run_git_command")
    def test_record_attempt_passes_cost_fields(self, mock_git, tmp_path):
        """record_attempt forwards cost kwargs into the persisted AttemptRecord."""
        from agent_framework.core.attempt_tracker import record_attempt

        def git_side_effect(args, **kwargs):
            key = " ".join(args[:2])
            responses = {
                "rev-parse --abbrev-ref": MagicMock(returncode=0, stdout="agent/test/task-1\n"),
                "rev-parse --short": MagicMock(returncode=0, stdout="abc123\n"),
                "rev-parse --verify": MagicMock(returncode=0),
                "status --porcelain": MagicMock(returncode=0, stdout=""),
                "push origin": MagicMock(returncode=0),
                "rev-list --count": MagicMock(returncode=0, stdout="1\n"),
                "diff --stat": MagicMock(returncode=0, stdout=""),
                "diff --name-only": MagicMock(returncode=0, stdout=""),
            }
            for pattern, result in responses.items():
                if key.startswith(pattern.split()[0]) and all(p in key for p in pattern.split()):
                    return result
            return MagicMock(returncode=1, stdout="", stderr="")

        mock_git.side_effect = git_side_effect

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        task = _make_task()
        result = record_attempt(
            workspace=tmp_path,
            task=task,
            agent_id="engineer",
            working_dir=worktree,
            error="timeout",
            input_tokens=8000,
            output_tokens=2000,
            cost_usd=1.75,
        )

        assert result is not None
        assert result.input_tokens == 8000
        assert result.output_tokens == 2000
        assert result.cost_usd == 1.75

        # Verify persisted
        loaded = load_attempt_history(tmp_path, task.id)
        assert loaded.attempts[0].cost_usd == 1.75
