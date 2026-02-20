"""Tests for handoff latency observability.

Covers:
- ActivityEvent "queued" type + correlation fields
- WorkflowExecutor queued event emission
- HandoffRecord computation from event streams
- Handoff summary aggregation (avg, p50, p90)
- Pending/delayed handoff detection
- Pre-instrumentation fallback (complete+start, no queued)
- Persistent handoff log write/read
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.activity import ActivityEvent, ActivityManager
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.analytics.performance_metrics import (
    HandoffRecord,
    HandoffSummary,
    PerformanceMetrics,
)
from agent_framework.workflow.executor import WorkflowExecutor


# -- Helpers --

def _ts(offset_s: int = 0) -> datetime:
    """Fixed base time + offset for deterministic tests."""
    base = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_s)


def _make_event(type: str, agent: str, task_id: str, offset_s: int,
                root_task_id: str = "", source_task_id: str = "",
                **extra) -> dict:
    event = {
        "type": type,
        "agent": agent,
        "task_id": task_id,
        "title": f"[chain] task",
        "timestamp": _ts(offset_s).isoformat(),
    }
    if root_task_id:
        event["root_task_id"] = root_task_id
    if source_task_id:
        event["source_task_id"] = source_task_id
    event.update(extra)
    return event


def _make_task(task_id="task-abc", **ctx_overrides):
    context = {"workflow": "default", **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


# -- ActivityEvent model tests --

class TestActivityEventQueuedType:
    def test_queued_event_accepted(self):
        event = ActivityEvent(
            type="queued",
            agent="engineer",
            task_id="chain-abc-implement-d1",
            title="[chain] Implement feature",
            timestamp=_ts(),
            root_task_id="abc",
            source_task_id="task-abc",
        )
        assert event.type == "queued"
        assert event.root_task_id == "abc"
        assert event.source_task_id == "task-abc"

    def test_correlation_fields_optional(self):
        """Existing event types still work without new fields."""
        event = ActivityEvent(
            type="start",
            agent="engineer",
            task_id="task-1",
            title="Start task",
            timestamp=_ts(),
        )
        assert event.root_task_id is None
        assert event.source_task_id is None

    def test_root_task_id_on_start_event(self):
        event = ActivityEvent(
            type="start",
            agent="engineer",
            task_id="task-1",
            title="Start task",
            timestamp=_ts(),
            root_task_id="root-1",
        )
        assert event.root_task_id == "root-1"

    def test_root_task_id_on_complete_event(self):
        event = ActivityEvent(
            type="complete",
            agent="architect",
            task_id="task-2",
            title="Complete review",
            timestamp=_ts(),
            root_task_id="root-2",
        )
        assert event.root_task_id == "root-2"


# -- Executor queued event emission tests --

class TestExecutorQueuedEvent:
    @pytest.fixture
    def queue(self, tmp_path):
        q = MagicMock()
        q.queue_dir = tmp_path / "queues"
        q.queue_dir.mkdir()
        q.completed_dir = tmp_path / "completed"
        q.completed_dir.mkdir()
        return q

    @pytest.fixture
    def activity_manager(self, tmp_path):
        return ActivityManager(workspace=tmp_path)

    def test_queued_event_emitted_after_push(self, queue, activity_manager, tmp_path):
        executor = WorkflowExecutor(
            queue, queue.queue_dir,
            workspace=tmp_path,
            activity_manager=activity_manager,
        )
        task = _make_task(
            workflow_step="plan",
            _root_task_id="root-123",
        )

        from agent_framework.workflow.dag import WorkflowStep
        target_step = WorkflowStep(id="implement", agent="engineer")

        executor._route_to_step(task, target_step, MagicMock(steps={"implement": target_step}, is_terminal_step=lambda x: False), "architect", None)

        queue.push.assert_called_once()

        events = activity_manager.get_recent_events(limit=10)
        queued_events = [e for e in events if e.type == "queued"]
        assert len(queued_events) == 1

        qe = queued_events[0]
        assert qe.agent == "engineer"
        assert qe.root_task_id == "root-123"
        assert qe.source_task_id == task.id

    def test_no_queued_event_when_push_fails(self, queue, activity_manager, tmp_path):
        queue.push.side_effect = RuntimeError("Queue full")
        executor = WorkflowExecutor(
            queue, queue.queue_dir,
            workspace=tmp_path,
            activity_manager=activity_manager,
        )
        task = _make_task(workflow_step="plan", _root_task_id="root-456")

        from agent_framework.workflow.dag import WorkflowStep
        target_step = WorkflowStep(id="implement", agent="engineer")

        executor._route_to_step(task, target_step, MagicMock(steps={"implement": target_step}, is_terminal_step=lambda x: False), "architect", None)

        events = activity_manager.get_recent_events(limit=10)
        queued_events = [e for e in events if e.type == "queued"]
        assert len(queued_events) == 0

    def test_no_queued_event_when_dedup_prevents_push(self, queue, activity_manager, tmp_path):
        """If chain task is already queued, push is skipped entirely."""
        executor = WorkflowExecutor(
            queue, queue.queue_dir,
            workspace=tmp_path,
            activity_manager=activity_manager,
        )
        task = _make_task(workflow_step="plan", _root_task_id="root-789")

        from agent_framework.workflow.dag import WorkflowStep
        target_step = WorkflowStep(id="implement", agent="engineer")

        # Pre-create the chain task file so dedup triggers
        chain_id = f"chain-root-789-implement-d1"
        agent_dir = queue.queue_dir / "engineer"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / f"{chain_id}.json").write_text(json.dumps({"title": "[chain] Implement feature X"}))

        executor._route_to_step(task, target_step, MagicMock(steps={"implement": target_step}, is_terminal_step=lambda x: False), "architect", None)

        queue.push.assert_not_called()
        events = activity_manager.get_recent_events(limit=10)
        assert all(e.type != "queued" for e in events)

    def test_no_activity_manager_does_not_crash(self, queue, tmp_path):
        """Executor works without activity_manager (backward compat)."""
        executor = WorkflowExecutor(queue, queue.queue_dir, workspace=tmp_path)
        task = _make_task(workflow_step="plan", _root_task_id="root-000")

        from agent_framework.workflow.dag import WorkflowStep
        target_step = WorkflowStep(id="implement", agent="engineer")

        executor._route_to_step(task, target_step, MagicMock(steps={"implement": target_step}, is_terminal_step=lambda x: False), "architect", None)

        queue.push.assert_called_once()


# -- Handoff metrics computation tests --

class TestHandoffRecordComputation:
    @pytest.fixture
    def metrics(self, tmp_path):
        return PerformanceMetrics(workspace=tmp_path)

    def test_complete_queued_start_triple(self, metrics):
        """Standard case: complete→queued→start produces a full record."""
        events = [
            _make_event("complete", "architect", "task-A", 0, root_task_id="root-1"),
            _make_event("queued", "engineer", "task-B", 2, root_task_id="root-1", source_task_id="task-A"),
            _make_event("start", "engineer", "task-B", 30, root_task_id="root-1"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 1

        r = records[0]
        assert r.from_agent == "architect"
        assert r.to_agent == "engineer"
        assert r.from_task_id == "task-A"
        assert r.to_task_id == "task-B"
        assert r.total_handoff_ms == 30_000
        assert r.queue_wait_ms == 28_000
        assert r.post_completion_ms == 2_000
        assert r.status == "completed"

    def test_queued_but_not_started_is_pending(self, metrics):
        """Queued event with no matching start → pending status."""
        events = [
            _make_event("complete", "architect", "task-A", 0, root_task_id="root-1"),
            _make_event("queued", "engineer", "task-B", 2, root_task_id="root-1", source_task_id="task-A"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 1
        assert records[0].status == "pending"
        assert records[0].total_handoff_ms is None

    def test_delayed_handoff_detection(self, metrics):
        """Handoff exceeding threshold is flagged delayed."""
        events = [
            _make_event("complete", "architect", "task-A", 0, root_task_id="root-1"),
            _make_event("queued", "engineer", "task-B", 1, root_task_id="root-1", source_task_id="task-A"),
            _make_event("start", "engineer", "task-B", 90, root_task_id="root-1"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 1
        assert records[0].status == "delayed"
        assert records[0].total_handoff_ms == 90_000

    def test_pre_instrumentation_fallback(self, metrics):
        """Complete+start without queued event still produces a record."""
        events = [
            _make_event("complete", "architect", "task-A", 0, root_task_id="root-1"),
            _make_event("start", "engineer", "task-B", 26, root_task_id="root-1"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 1

        r = records[0]
        assert r.from_agent == "architect"
        assert r.to_agent == "engineer"
        assert r.queued_at is None
        assert r.total_handoff_ms == 26_000
        assert r.queue_wait_ms is None

    def test_multiple_handoffs_in_chain(self, metrics):
        """Multiple transitions in a single chain are all captured."""
        events = [
            _make_event("complete", "architect", "task-plan", 0, root_task_id="root-1"),
            _make_event("queued", "engineer", "task-impl", 1, root_task_id="root-1", source_task_id="task-plan"),
            _make_event("start", "engineer", "task-impl", 30, root_task_id="root-1"),
            _make_event("complete", "engineer", "task-impl", 120, root_task_id="root-1"),
            _make_event("queued", "architect", "task-review", 121, root_task_id="root-1", source_task_id="task-impl"),
            _make_event("start", "architect", "task-review", 145, root_task_id="root-1"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 2

        transitions = {(r.from_agent, r.to_agent) for r in records}
        assert ("architect", "engineer") in transitions
        assert ("engineer", "architect") in transitions

    def test_missing_complete_event_uses_queued_as_anchor(self, metrics):
        """If no complete event exists for source, record still created."""
        events = [
            _make_event("queued", "engineer", "task-B", 5, root_task_id="root-1", source_task_id="task-A"),
            _make_event("start", "engineer", "task-B", 35, root_task_id="root-1"),
        ]
        records = metrics._compute_handoff_records(events)
        assert len(records) == 1
        r = records[0]
        assert r.from_agent == "unknown"
        assert r.queue_wait_ms == 30_000
        # post_completion_ms is 0 since completed_at falls back to queued_at
        assert r.post_completion_ms == 0


# -- Summary aggregation tests --

class TestHandoffSummaryAggregation:
    @pytest.fixture
    def metrics(self, tmp_path):
        return PerformanceMetrics(workspace=tmp_path)

    def test_summary_by_transition(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="architect", to_agent="engineer",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), queued_at=_ts(1), started_at=_ts(20),
                total_handoff_ms=20_000, queue_wait_ms=19_000, post_completion_ms=1_000,
                status="completed",
            ),
            HandoffRecord(
                root_task_id="r2", from_agent="architect", to_agent="engineer",
                from_task_id="t3", to_task_id="t4",
                completed_at=_ts(0), queued_at=_ts(2), started_at=_ts(40),
                total_handoff_ms=40_000, queue_wait_ms=38_000, post_completion_ms=2_000,
                status="completed",
            ),
        ]
        summaries = metrics._aggregate_handoff_summaries(records)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.transition == "architect\u2192engineer"
        assert s.count == 2
        assert s.avg_total_ms == 30_000.0
        # With 2 sorted values [20k, 40k], index len//2=1 picks the higher
        assert s.p50_total_ms == 40_000
        assert s.p90_total_ms == 40_000
        assert s.avg_queue_wait_ms == 28_500.0
        assert s.failed_count == 0
        assert s.delayed_count == 0

    def test_summary_counts_pending_and_delayed(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="a", to_agent="b",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), queued_at=_ts(1),
                status="pending",
            ),
            HandoffRecord(
                root_task_id="r2", from_agent="a", to_agent="b",
                from_task_id="t3", to_task_id="t4",
                completed_at=_ts(0), queued_at=_ts(1), started_at=_ts(90),
                total_handoff_ms=90_000, queue_wait_ms=89_000, post_completion_ms=1_000,
                status="delayed",
            ),
        ]
        summaries = metrics._aggregate_handoff_summaries(records)
        assert len(summaries) == 1
        assert summaries[0].failed_count == 1
        assert summaries[0].delayed_count == 1

    def test_multiple_transitions_produce_separate_summaries(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="architect", to_agent="engineer",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), total_handoff_ms=10_000, status="completed",
            ),
            HandoffRecord(
                root_task_id="r1", from_agent="engineer", to_agent="qa",
                from_task_id="t2", to_task_id="t3",
                completed_at=_ts(0), total_handoff_ms=15_000, status="completed",
            ),
        ]
        summaries = metrics._aggregate_handoff_summaries(records)
        assert len(summaries) == 2
        transitions = {s.transition for s in summaries}
        assert "architect\u2192engineer" in transitions
        assert "engineer\u2192qa" in transitions


# -- Persistent log tests --

class TestHandoffPersistentLog:
    @pytest.fixture
    def metrics(self, tmp_path):
        return PerformanceMetrics(workspace=tmp_path)

    def test_persist_and_read_records(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="architect", to_agent="engineer",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), queued_at=_ts(1), started_at=_ts(30),
                total_handoff_ms=30_000, queue_wait_ms=29_000, post_completion_ms=1_000,
                status="completed",
            ),
        ]
        metrics._persist_handoff_records(records)

        loaded = metrics.read_handoff_log()
        assert len(loaded) == 1
        assert loaded[0].from_task_id == "t1"
        assert loaded[0].to_task_id == "t2"

    def test_dedup_prevents_duplicate_writes(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="a", to_agent="b",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), status="completed",
            ),
        ]
        metrics._persist_handoff_records(records)
        metrics._persist_handoff_records(records)  # same records again

        loaded = metrics.read_handoff_log()
        assert len(loaded) == 1

    def test_empty_records_no_file_write(self, metrics):
        metrics._persist_handoff_records([])
        assert not metrics.handoff_log_file.exists()

    def test_generate_handoff_report(self, metrics):
        records = [
            HandoffRecord(
                root_task_id="r1", from_agent="architect", to_agent="engineer",
                from_task_id="t1", to_task_id="t2",
                completed_at=_ts(0), queued_at=_ts(1), started_at=_ts(25),
                total_handoff_ms=25_000, queue_wait_ms=24_000, post_completion_ms=1_000,
                status="completed",
            ),
            HandoffRecord(
                root_task_id="r2", from_agent="engineer", to_agent="qa",
                from_task_id="t3", to_task_id="t4",
                completed_at=_ts(0), queued_at=_ts(2),
                status="pending",
            ),
        ]
        metrics._persist_handoff_records(records)

        report = metrics.generate_handoff_report()
        assert len(report.records) == 2
        assert len(report.summaries) == 2
        assert len(report.pending_handoffs) == 1
        assert report.pending_handoffs[0].to_task_id == "t4"


# -- Integration: generate_report includes handoff_summaries --

class TestPerformanceReportHandoffs:
    def test_report_includes_handoff_summaries(self, tmp_path):
        """Full generate_report() wires handoff metrics into the report."""
        metrics = PerformanceMetrics(workspace=tmp_path)

        # Use recent timestamps so they survive the hours=1 cutoff
        now = datetime.now(timezone.utc)
        def _recent(offset_s):
            return (now - timedelta(seconds=300) + timedelta(seconds=offset_s)).isoformat()

        events = [
            {"type": "start", "agent": "architect", "task_id": "task-A",
             "title": "Plan", "timestamp": _recent(0), "root_task_id": "root-1"},
            {"type": "complete", "agent": "architect", "task_id": "task-A",
             "title": "Plan", "timestamp": _recent(60), "root_task_id": "root-1",
             "duration_ms": 60000, "input_tokens": 100, "output_tokens": 50, "cost": 0.01},
            {"type": "queued", "agent": "engineer", "task_id": "task-B",
             "title": "[chain] task", "timestamp": _recent(62),
             "root_task_id": "root-1", "source_task_id": "task-A"},
            {"type": "start", "agent": "engineer", "task_id": "task-B",
             "title": "[chain] task", "timestamp": _recent(90), "root_task_id": "root-1"},
            {"type": "complete", "agent": "engineer", "task_id": "task-B",
             "title": "[chain] task", "timestamp": _recent(200), "root_task_id": "root-1",
             "duration_ms": 110000, "input_tokens": 200, "output_tokens": 100, "cost": 0.05},
        ]

        stream_file = tmp_path / ".agent-communication" / "activity-stream.jsonl"
        stream_file.parent.mkdir(parents=True, exist_ok=True)
        stream_file.write_text('\n'.join(json.dumps(e) for e in events) + '\n')

        report = metrics.generate_report(hours=1)
        assert len(report.handoff_summaries) >= 1

        summary = report.handoff_summaries[0]
        assert summary.transition == "architect\u2192engineer"
        assert summary.count == 1
