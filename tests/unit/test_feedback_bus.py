"""Tests for FeedbackBus — event dispatch, memory persistence, session logging."""

from unittest.mock import MagicMock

import pytest

from agent_framework.core.feedback_bus import FeedbackBus, FeedbackEvent


@pytest.fixture
def memory_store(tmp_path):
    from agent_framework.memory.memory_store import MemoryStore
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def bus(memory_store):
    return FeedbackBus(
        memory_store=memory_store,
        repo_slug="org/repo",
        agent_type="engineer",
    )


class TestFeedbackEvent:
    def test_event_has_timestamp(self):
        event = FeedbackEvent(source="test", category="cat", content="hello")
        assert event.timestamp > 0

    def test_event_default_tags_and_metadata(self):
        event = FeedbackEvent(source="test", category="cat", content="hello")
        assert event.tags == []
        assert event.metadata == {}


class TestEmit:
    def test_emit_stores_event_in_log(self, bus):
        event = FeedbackEvent(source="self_eval", category="self_eval_failures", content="missed tests")
        bus.emit(event)
        assert bus.event_count == 1
        assert bus.get_events()[0] is event

    def test_emit_dispatches_to_consumers(self, bus):
        received = []
        bus.register_consumer("self_eval_failures", lambda e: received.append(e))

        event = FeedbackEvent(source="self_eval", category="self_eval_failures", content="missed tests")
        bus.emit(event)

        assert len(received) == 1
        assert received[0].content == "missed tests"

    def test_emit_only_dispatches_matching_category(self, bus):
        received = []
        bus.register_consumer("qa_findings", lambda e: received.append(e))

        event = FeedbackEvent(source="self_eval", category="self_eval_failures", content="missed tests")
        bus.emit(event)

        assert len(received) == 0

    def test_emit_persists_to_memory(self, bus, memory_store):
        event = FeedbackEvent(
            source="self_eval",
            category="self_eval_failures",
            content="missed tests",
            tags=["test_tag"],
            metadata={"task_id": "task-123"},
        )
        bus.emit(event)

        entries = memory_store.recall("org/repo", "engineer", category="self_eval_failures")
        assert len(entries) == 1
        assert entries[0].content == "missed tests"
        assert "test_tag" in entries[0].tags

    def test_emit_persist_false_skips_memory(self, bus, memory_store):
        event = FeedbackEvent(source="test", category="test_cat", content="ephemeral")
        bus.emit(event, persist=False)

        entries = memory_store.recall("org/repo", "engineer", category="test_cat")
        assert len(entries) == 0

    def test_consumer_error_does_not_crash_bus(self, bus):
        def bad_consumer(e):
            raise RuntimeError("consumer failed")

        bus.register_consumer("cat", bad_consumer)
        event = FeedbackEvent(source="test", category="cat", content="hello")
        bus.emit(event)  # Should not raise
        assert bus.event_count == 1


class TestEventCounts:
    def test_get_event_counts_by_category(self, bus):
        bus.emit(FeedbackEvent(source="a", category="cat1", content="x"), persist=False)
        bus.emit(FeedbackEvent(source="b", category="cat1", content="y"), persist=False)
        bus.emit(FeedbackEvent(source="c", category="cat2", content="z"), persist=False)

        counts = bus.get_event_counts_by_category()
        assert counts == {"cat1": 2, "cat2": 1}

    def test_filter_events_by_category(self, bus):
        bus.emit(FeedbackEvent(source="a", category="cat1", content="x"), persist=False)
        bus.emit(FeedbackEvent(source="b", category="cat2", content="y"), persist=False)

        cat1_events = bus.get_events(category="cat1")
        assert len(cat1_events) == 1
        assert cat1_events[0].source == "a"


class TestSessionLogging:
    def test_emit_logs_to_session_logger(self):
        session_logger = MagicMock()
        bus = FeedbackBus(session_logger=session_logger)

        event = FeedbackEvent(source="self_eval", category="self_eval_failures", content="missed tests")
        bus.emit(event, persist=False)

        session_logger.log_feedback_emitted.assert_called_once_with(
            source="self_eval",
            category="self_eval_failures",
            content="missed tests",
        )
