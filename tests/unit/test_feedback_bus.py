"""Tests for FeedbackBus — event emission, consumer routing, memory persistence."""

from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.feedback_bus import FeedbackBus, FeedbackEvent


@pytest.fixture
def memory_store():
    store = MagicMock()
    store.enabled = True
    store.remember = MagicMock(return_value=True)
    return store


@pytest.fixture
def session_logger():
    logger = MagicMock()
    logger.log = MagicMock()
    return logger


@pytest.fixture
def bus(memory_store, session_logger):
    return FeedbackBus(
        memory_store=memory_store,
        session_logger=session_logger,
        repo_slug="org/repo",
        agent_type="engineer",
    )


class TestEmit:
    def test_emit_increments_count(self, bus):
        event = FeedbackEvent(source="test", category="test_cat", content="test content")
        bus.emit(event)
        assert bus.emit_count == 1

    def test_emit_persists_to_memory(self, bus, memory_store):
        event = FeedbackEvent(
            source="self_eval",
            category="self_eval_failures",
            content="Missed error handling",
            tags=["error_handling"],
        )
        bus.emit(event)
        memory_store.remember.assert_called_once_with(
            repo_slug="org/repo",
            agent_type="engineer",
            category="self_eval_failures",
            content="Missed error handling",
            tags=["error_handling"],
        )
        assert bus.persist_count == 1

    def test_emit_without_persist(self, bus, memory_store):
        event = FeedbackEvent(source="test", category="test_cat", content="test")
        bus.emit(event, persist=False)
        memory_store.remember.assert_not_called()
        assert bus.persist_count == 0

    def test_emit_logs_to_session(self, bus, session_logger):
        event = FeedbackEvent(source="qa", category="qa_findings", content="Missing tests")
        bus.emit(event)
        session_logger.log.assert_called_once()
        call_args = session_logger.log.call_args
        assert call_args[0][0] == "feedback_emitted"
        assert call_args[1]["source"] == "qa"
        assert call_args[1]["category"] == "qa_findings"

    def test_emit_without_memory_store(self, session_logger):
        bus = FeedbackBus(session_logger=session_logger)
        event = FeedbackEvent(source="test", category="test", content="test")
        bus.emit(event)
        assert bus.emit_count == 1
        assert bus.persist_count == 0


class TestConsumers:
    def test_consumer_receives_matching_event(self, bus):
        received = []
        bus.register_consumer("test_cat", lambda e: received.append(e))

        event = FeedbackEvent(source="test", category="test_cat", content="hello")
        bus.emit(event, persist=False)

        assert len(received) == 1
        assert received[0].content == "hello"

    def test_consumer_ignores_non_matching_category(self, bus):
        received = []
        bus.register_consumer("cat_a", lambda e: received.append(e))

        event = FeedbackEvent(source="test", category="cat_b", content="hello")
        bus.emit(event, persist=False)

        assert len(received) == 0

    def test_multiple_consumers_same_category(self, bus):
        received_a = []
        received_b = []
        bus.register_consumer("test_cat", lambda e: received_a.append(e))
        bus.register_consumer("test_cat", lambda e: received_b.append(e))

        event = FeedbackEvent(source="test", category="test_cat", content="hello")
        bus.emit(event, persist=False)

        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_consumer_error_does_not_stop_emission(self, bus, memory_store):
        """Consumer exception should not prevent persistence or other consumers."""
        def bad_consumer(e):
            raise ValueError("consumer error")

        received = []
        bus.register_consumer("test_cat", bad_consumer)
        bus.register_consumer("test_cat", lambda e: received.append(e))

        event = FeedbackEvent(source="test", category="test_cat", content="hello")
        bus.emit(event)

        assert len(received) == 1
        memory_store.remember.assert_called_once()


class TestUpdateContext:
    def test_update_context_changes_repo_and_agent(self, bus, memory_store):
        bus.update_context("new-org/new-repo", "qa")
        event = FeedbackEvent(source="test", category="test", content="test")
        bus.emit(event)

        memory_store.remember.assert_called_once()
        assert memory_store.remember.call_args[1]["repo_slug"] == "new-org/new-repo"
        assert memory_store.remember.call_args[1]["agent_type"] == "qa"


class TestFeedbackEvent:
    def test_event_has_timestamp(self):
        event = FeedbackEvent(source="test", category="test", content="test")
        assert event.timestamp is not None
        assert "T" in event.timestamp  # ISO format

    def test_event_default_tags_and_metadata(self):
        event = FeedbackEvent(source="test", category="test", content="test")
        assert event.tags == []
        assert event.metadata == {}
