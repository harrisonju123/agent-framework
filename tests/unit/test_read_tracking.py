"""Tests for file read tracking and escalating exploration limits in CheckpointManager."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from agent_framework.core.checkpoint_manager import CheckpointManager, FileReadInfo
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_task(task_id="test-read-1"):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="test-agent",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test",
        context={"github_repo": "org/repo"},
    )


def _make_checkpoint_manager(
    *,
    exploration_threshold=50,
    reread_threshold=3,
    escalation_multipliers=None,
    working_dir=None,
    cached_paths=frozenset(),
):
    """Build CheckpointManager with mock dependencies."""
    task = _make_task()
    circuit_breaker = asyncio.Event()
    mgr = CheckpointManager(
        task=task,
        working_dir=working_dir or Path("/tmp/fake-workdir"),
        is_implementation_step=True,
        max_consecutive_tool_calls=15,
        max_consecutive_diagnostic_calls=5,
        exploration_threshold=exploration_threshold,
        workflow_step="implement",
        git_ops=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
        context_window_manager=None,
        logger=MagicMock(),
        circuit_breaker_event=circuit_breaker,
        agent_id="test-agent",
        agent_base_id="engineer",
        reread_threshold=reread_threshold,
        escalation_multipliers=escalation_multipliers,
        cached_paths=cached_paths,
    )
    return mgr, circuit_breaker


# ---------------------------------------------------------------------------
# FileReadInfo dataclass
# ---------------------------------------------------------------------------


class TestFileReadInfo:
    def test_defaults(self):
        info = FileReadInfo()
        assert info.count == 0
        assert info.first_seq == 0
        assert info.last_seq == 0
        assert info.was_full_read is False
        assert info.chunked_sequences == []

    def test_mutable_default_independence(self):
        """Each instance gets its own chunked_sequences list."""
        a = FileReadInfo()
        b = FileReadInfo()
        a.chunked_sequences.append(1)
        assert b.chunked_sequences == []


# ---------------------------------------------------------------------------
# Basic read tracking
# ---------------------------------------------------------------------------


class TestReadTracking:
    def test_single_read_tracked(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", "core/checkpoint_manager.py")

        assert "core/checkpoint_manager.py" in mgr._file_reads
        info = mgr._file_reads["core/checkpoint_manager.py"]
        assert info.count == 1
        assert info.first_seq == 1
        assert info.last_seq == 1

    def test_multiple_reads_increment_count(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/agent.py")

        info = mgr._file_reads["core/agent.py"]
        assert info.count == 3
        assert info.first_seq == 1
        assert info.last_seq == 3

    def test_different_files_tracked_separately(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/config.py")
        mgr.on_tool_activity("Read", "core/agent.py")

        assert mgr._file_reads["core/agent.py"].count == 2
        assert mgr._file_reads["core/config.py"].count == 1

    def test_none_summary_ignored(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", None)
        assert len(mgr._file_reads) == 0

    def test_empty_summary_ignored(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", "  ")
        # Whitespace-only path still gets tracked (edge case, but it's a valid summary)
        # The path would be empty after strip... let's verify behavior
        # Actually "  ".strip() == "" which is falsy, so _track_file_read returns early
        assert len(mgr._file_reads) == 0

    def test_non_read_tools_not_tracked(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Edit", "core/agent.py")
        mgr.on_tool_activity("Write", "core/config.py")
        mgr.on_tool_activity("Bash", "git status")
        assert len(mgr._file_reads) == 0


# ---------------------------------------------------------------------------
# Chunked read detection
# ---------------------------------------------------------------------------


class TestChunkedReadDetection:
    def test_sequential_reads_detected_as_chunked(self):
        """3 reads of the same file within 5-call window marks as chunked."""
        mgr, _ = _make_checkpoint_manager(reread_threshold=10)
        mgr.on_tool_activity("Read", "models/task.py")  # seq 1
        mgr.on_tool_activity("Read", "models/task.py")  # seq 2, gap=1
        mgr.on_tool_activity("Read", "models/task.py")  # seq 3, gap=1

        info = mgr._file_reads["models/task.py"]
        assert info.count == 3
        # Sequences 2 and 3 should be in chunked_sequences (within window of prior read)
        assert 2 in info.chunked_sequences
        assert 3 in info.chunked_sequences

    def test_distant_reads_not_chunked(self):
        """Reads separated by more than _CHUNKED_READ_WINDOW are not chunked."""
        mgr, _ = _make_checkpoint_manager(reread_threshold=10)
        mgr.on_tool_activity("Read", "models/task.py")  # seq 1
        # Fill gap with other tool calls
        for i in range(6):
            mgr.on_tool_activity("Bash", f"cmd-{i}")
        mgr.on_tool_activity("Read", "models/task.py")  # seq 8, gap=7

        info = mgr._file_reads["models/task.py"]
        assert info.count == 2
        assert info.chunked_sequences == []

    def test_mixed_files_chunked_independently(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=10)
        mgr.on_tool_activity("Read", "a.py")  # seq 1
        mgr.on_tool_activity("Read", "b.py")  # seq 2
        mgr.on_tool_activity("Read", "a.py")  # seq 3, gap from a.py=2 (within window)
        mgr.on_tool_activity("Read", "b.py")  # seq 4, gap from b.py=2 (within window)

        assert 3 in mgr._file_reads["a.py"].chunked_sequences
        assert 4 in mgr._file_reads["b.py"].chunked_sequences


# ---------------------------------------------------------------------------
# Re-read threshold triggers interrupt
# ---------------------------------------------------------------------------


class TestRereadThreshold:
    def test_threshold_triggers_interrupt(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=3)
        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/agent.py")
        assert not mgr._reread_interrupted

        mgr.on_tool_activity("Read", "core/agent.py")
        assert mgr._reread_interrupted

    def test_threshold_logs_session_event(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=2)
        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/agent.py")

        mgr._session_logger.log.assert_any_call(
            "reread_threshold_exceeded",
            file="core/agent.py",
            count=2,
            threshold=2,
        )

    def test_interrupt_fires_only_once(self):
        """Once _reread_interrupted is set, further re-reads don't re-trigger."""
        mgr, _ = _make_checkpoint_manager(reread_threshold=2)
        mgr.on_tool_activity("Read", "a.py")
        mgr.on_tool_activity("Read", "a.py")  # triggers interrupt
        assert mgr._reread_interrupted

        # Reset mock to track new calls
        mgr._session_logger.log.reset_mock()
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "b.py")  # b.py also hits threshold

        # Should NOT fire reread_threshold_exceeded again
        for call in mgr._session_logger.log.call_args_list:
            assert call[0][0] != "reread_threshold_exceeded"

    def test_below_threshold_no_interrupt(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=5)
        for _ in range(4):
            mgr.on_tool_activity("Read", "core/agent.py")
        assert not mgr._reread_interrupted

    def test_threshold_trips_circuit_breaker_event(self):
        """Re-read threshold sets the circuit breaker event to interrupt the LLM."""
        mgr, cb = _make_checkpoint_manager(reread_threshold=3)
        assert not cb.is_set()

        mgr.on_tool_activity("Read", "core/agent.py")
        mgr.on_tool_activity("Read", "core/agent.py")
        assert not cb.is_set()

        mgr.on_tool_activity("Read", "core/agent.py")
        assert cb.is_set()

    def test_circuit_breaker_not_set_below_threshold(self):
        mgr, cb = _make_checkpoint_manager(reread_threshold=4)
        for _ in range(3):
            mgr.on_tool_activity("Read", "core/agent.py")
        assert not cb.is_set()

    def test_circuit_breaker_set_only_once(self):
        """Event.set() is idempotent, but verify second file doesn't cause issues."""
        mgr, cb = _make_checkpoint_manager(reread_threshold=2)
        mgr.on_tool_activity("Read", "a.py")
        mgr.on_tool_activity("Read", "a.py")  # trips breaker
        assert cb.is_set()

        # Second file also hitting threshold should not crash
        mgr._reread_interrupted = False  # reset to allow re-trigger check
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "b.py")
        # Event is still set (idempotent)
        assert cb.is_set()


# ---------------------------------------------------------------------------
# get_read_stats / get_worst_reread
# ---------------------------------------------------------------------------


class TestReadStats:
    def test_get_read_stats_only_returns_2_plus(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=10)
        mgr.on_tool_activity("Read", "a.py")
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "c.py")
        mgr.on_tool_activity("Read", "c.py")
        mgr.on_tool_activity("Read", "c.py")

        stats = mgr.get_read_stats()
        assert "a.py" not in stats  # only 1 read
        assert stats["b.py"] == 2
        assert stats["c.py"] == 3

    def test_get_read_stats_empty_when_no_rereads(self):
        mgr, _ = _make_checkpoint_manager()
        mgr.on_tool_activity("Read", "a.py")
        mgr.on_tool_activity("Read", "b.py")
        assert mgr.get_read_stats() == {}

    def test_get_worst_reread_returns_correct_file(self):
        mgr, _ = _make_checkpoint_manager(reread_threshold=10)
        mgr.on_tool_activity("Read", "a.py")
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "b.py")
        mgr.on_tool_activity("Read", "a.py")

        result = mgr.get_worst_reread()
        assert result == ("b.py", 3)

    def test_get_worst_reread_none_when_empty(self):
        mgr, _ = _make_checkpoint_manager()
        assert mgr.get_worst_reread() is None


# ---------------------------------------------------------------------------
# Escalating exploration levels
# ---------------------------------------------------------------------------


class TestEscalatingExploration:
    def test_level_1_fires_at_threshold(self):
        mgr, cb = _make_checkpoint_manager(exploration_threshold=5)
        for i in range(5):
            mgr.on_tool_activity("Grep", f"pattern-{i}")

        assert mgr._exploration_level == 1
        assert not cb.is_set()

    def test_level_2_fires_at_2x_threshold(self):
        mgr, cb = _make_checkpoint_manager(exploration_threshold=5)
        for i in range(10):
            mgr.on_tool_activity("Grep", f"pattern-{i}")

        assert mgr._exploration_level == 2
        assert not cb.is_set()

    def test_level_3_fires_at_3x_threshold_and_trips_breaker(self):
        mgr, cb = _make_checkpoint_manager(
            exploration_threshold=5,
            working_dir=Path("/nonexistent/path/for/test"),
        )
        for i in range(15):
            mgr.on_tool_activity("Grep", f"pattern-{i}")

        assert mgr._exploration_level == 3
        assert cb.is_set()

    def test_level_3_calls_safety_commit(self):
        mgr, _ = _make_checkpoint_manager(
            exploration_threshold=5,
            working_dir=Path("/nonexistent/test/path"),
        )
        # Mock working_dir to make it "exist" for safety_commit but not for checkpoint
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mgr.working_dir = mock_dir
        # Disable periodic checkpoints so only the force-halt commit fires
        mgr.checkpoint_interval = 99999

        for i in range(15):
            mgr.on_tool_activity("Grep", f"pattern-{i}")

        mgr._git_ops.safety_commit.assert_called_once()
        commit_msg = mgr._git_ops.safety_commit.call_args[0][1]
        assert "force halt" in commit_msg

    def test_level_1_emits_activity_event(self):
        mgr, _ = _make_checkpoint_manager(exploration_threshold=3)
        for i in range(3):
            mgr.on_tool_activity("Glob", f"**/*.py-{i}")

        mgr._activity_manager.append_event.assert_called()
        event = mgr._activity_manager.append_event.call_args[0][0]
        assert event.type == "exploration_alert"

    def test_level_2_emits_activity_event_with_wrap_up_title(self):
        mgr, _ = _make_checkpoint_manager(exploration_threshold=3)
        for i in range(6):
            mgr.on_tool_activity("Glob", f"**/*.py-{i}")

        calls = mgr._activity_manager.append_event.call_args_list
        titles = [c[0][0].title for c in calls]
        assert any("WRAP UP" in t for t in titles)

    def test_level_3_emits_circuit_breaker_event(self):
        mgr, _ = _make_checkpoint_manager(exploration_threshold=3)
        for i in range(9):
            mgr.on_tool_activity("Glob", f"**/*.py-{i}")

        calls = mgr._activity_manager.append_event.call_args_list
        # Level 3 emits circuit_breaker event type
        event_types = [c[0][0].type for c in calls]
        assert "circuit_breaker" in event_types

    def test_levels_fire_only_once(self):
        """Levels don't re-trigger once set."""
        mgr, _ = _make_checkpoint_manager(exploration_threshold=5)

        for i in range(7):
            mgr.on_tool_activity("Glob", f"p-{i}")

        assert mgr._exploration_level == 1

        # Count how many times level 1 event was logged
        alert_calls = [
            c for c in mgr._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_calls) == 1

    def test_custom_multipliers(self):
        """Custom escalation multipliers are respected."""
        mgr, cb = _make_checkpoint_manager(
            exploration_threshold=10,
            escalation_multipliers=[1.0, 1.5, 2.0],
            working_dir=Path("/nonexistent/test/path"),
        )

        for i in range(15):
            mgr.on_tool_activity("Glob", f"p-{i}")

        # 1.0 * 10 = 10 -> level 1
        # 1.5 * 10 = 15 -> level 2
        assert mgr._exploration_level == 2
        assert not cb.is_set()

        for i in range(5):
            mgr.on_tool_activity("Glob", f"q-{i}")

        # 2.0 * 10 = 20 -> level 3
        assert mgr._exploration_level == 3
        assert cb.is_set()

    def test_exploration_level_replaces_boolean_flag(self):
        """The old exploration_alerted boolean is gone; _exploration_level is the new state."""
        mgr, _ = _make_checkpoint_manager()
        assert not hasattr(mgr, "exploration_alerted")
        assert mgr._exploration_level == 0

    def test_level_3_without_working_dir(self):
        """Level 3 doesn't crash when working_dir is None."""
        mgr, cb = _make_checkpoint_manager(
            exploration_threshold=3,
            working_dir=Path("/nonexistent/test/path"),
        )
        mgr.working_dir = None

        for i in range(9):
            mgr.on_tool_activity("Glob", f"p-{i}")

        assert mgr._exploration_level == 3
        assert cb.is_set()
        mgr._git_ops.safety_commit.assert_not_called()

    def test_session_logger_records_level_in_events(self):
        mgr, _ = _make_checkpoint_manager(exploration_threshold=3)
        for i in range(6):
            mgr.on_tool_activity("Glob", f"p-{i}")

        # Check level 1 event has level=1
        alert_calls = [
            c for c in mgr._session_logger.log.call_args_list
            if c[0][0] == "exploration_alert"
        ]
        assert len(alert_calls) == 1
        assert alert_calls[0][1]["level"] == 1

        # Check level 2 event has level=2
        escalation_calls = [
            c for c in mgr._session_logger.log.call_args_list
            if c[0][0] == "exploration_escalation"
        ]
        assert len(escalation_calls) == 1
        assert escalation_calls[0][1]["level"] == 2


# ---------------------------------------------------------------------------
# Cached paths: free first read for files already in the prompt
# ---------------------------------------------------------------------------


class TestCachedPaths:
    def test_cached_file_starts_at_count_zero(self):
        """First read of a cached file should have count=0 (free read)."""
        mgr, _ = _make_checkpoint_manager(
            cached_paths=frozenset(["src/agent_framework/core/agent.py"]),
        )
        mgr.on_tool_activity("Read", "core/agent.py")

        info = mgr._file_reads["core/agent.py"]
        assert info.count == 0

    def test_non_cached_file_starts_at_count_one(self):
        """Non-cached files should still start at count=1."""
        mgr, _ = _make_checkpoint_manager(
            cached_paths=frozenset(["src/agent_framework/core/agent.py"]),
        )
        mgr.on_tool_activity("Read", "core/config.py")

        info = mgr._file_reads["core/config.py"]
        assert info.count == 1

    def test_cached_file_needs_threshold_plus_one_to_trip(self):
        """Cached file needs threshold+1 reads to trip the circuit breaker."""
        mgr, cb = _make_checkpoint_manager(
            reread_threshold=3,
            cached_paths=frozenset(["src/agent_framework/core/agent.py"]),
        )
        # 3 reads: counts go 0, 1, 2 — all below threshold of 3
        for _ in range(3):
            mgr.on_tool_activity("Read", "core/agent.py")
        assert not mgr._reread_interrupted
        assert not cb.is_set()

        # 4th read: count=3, hits threshold → trips
        mgr.on_tool_activity("Read", "core/agent.py")
        assert mgr._reread_interrupted
        assert cb.is_set()

    def test_suffix_matching(self):
        """Cache paths with deep prefixes match abbreviated 3-segment tool summaries."""
        mgr, _ = _make_checkpoint_manager(
            cached_paths=frozenset([
                "src/agent_framework/core/prompt_builder.py",
                "src/agent_framework/utils/cascade.py",
            ]),
        )
        # Tool summaries are abbreviated to last 3 segments
        mgr.on_tool_activity("Read", "core/prompt_builder.py")
        mgr.on_tool_activity("Read", "utils/cascade.py")

        assert mgr._file_reads["core/prompt_builder.py"].count == 0
        assert mgr._file_reads["utils/cascade.py"].count == 0

    def test_empty_cached_paths_preserves_default_behavior(self):
        """Empty cached_paths doesn't change behavior."""
        mgr, _ = _make_checkpoint_manager(cached_paths=frozenset())
        mgr.on_tool_activity("Read", "core/agent.py")

        info = mgr._file_reads["core/agent.py"]
        assert info.count == 1
