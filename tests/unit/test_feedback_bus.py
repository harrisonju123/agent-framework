"""Tests for the cross-feature feedback bus."""

from unittest.mock import MagicMock, call

import pytest

from agent_framework.core.feedback_bus import (
    MAX_MISSED_CRITERIA_PER_REPO,
    QA_RECURRING_THRESHOLD,
    _finding_key,
    aggregate_qa_findings,
    format_known_pitfalls,
    store_debate_specialization_hint,
    store_self_eval_failure,
)
from agent_framework.memory.memory_store import MemoryEntry


# -- Helpers --


def _make_memory_store(*, enabled=True, existing_memories=None):
    """Build a mock MemoryStore with controllable recall results."""
    store = MagicMock()
    store.enabled = enabled
    store.remember = MagicMock(return_value=True)
    store.recall = MagicMock(return_value=existing_memories or [])
    return store


def _make_session_logger():
    return MagicMock()


# ============================================================
# store_self_eval_failure
# ============================================================


class TestStoreSelfEvalFailure:
    def test_stores_missed_criteria_from_critique(self):
        store = _make_memory_store()
        logger = _make_session_logger()

        criteria = [
            "Implement authentication middleware",
            "Add database migration script",
            "Write integration tests",
        ]
        # Critique mentions keywords from first two criteria
        critique = (
            "FAIL: The authentication middleware is not implemented. "
            "The database migration script is missing. "
            "Tests appear to be present."
        )

        count = store_self_eval_failure(
            store, logger,
            task_id="task-1",
            repo_slug="owner/repo",
            agent_type="engineer",
            acceptance_criteria=criteria,
            critique=critique,
        )

        assert count == 2
        assert store.remember.call_count == 2

        # Verify first stored memory
        first_call = store.remember.call_args_list[0]
        assert first_call.kwargs["category"] == "missed_criteria"
        assert first_call.kwargs["repo_slug"] == "owner/repo"
        assert "authentication" in first_call.kwargs["content"].lower()

        # Session logger called
        logger.log.assert_called_once_with(
            "feedback_bus_self_eval_stored",
            task_id="task-1",
            repo="owner/repo",
            criteria_stored=2,
            total_criteria=3,
        )

    def test_returns_zero_when_disabled(self):
        store = _make_memory_store(enabled=False)
        count = store_self_eval_failure(
            store, None,
            task_id="t", repo_slug="r", agent_type="e",
            acceptance_criteria=["foo"], critique="FAIL foo",
        )
        assert count == 0
        store.remember.assert_not_called()

    def test_returns_zero_when_no_criteria(self):
        store = _make_memory_store()
        count = store_self_eval_failure(
            store, None,
            task_id="t", repo_slug="r", agent_type="e",
            acceptance_criteria=[], critique="FAIL",
        )
        assert count == 0

    def test_caps_at_max_per_repo(self):
        store = _make_memory_store()
        # All criteria have matching words in critique
        criteria = [f"Implement feature number {i} correctly" for i in range(10)]
        critique = " ".join(
            f"feature number {i} implement correctly" for i in range(10)
        )

        count = store_self_eval_failure(
            store, None,
            task_id="t", repo_slug="r", agent_type="e",
            acceptance_criteria=criteria, critique=critique,
        )
        assert count <= MAX_MISSED_CRITERIA_PER_REPO

    def test_no_match_when_criterion_not_in_critique(self):
        store = _make_memory_store()
        criteria = ["Implement caching layer with Redis"]
        critique = "FAIL: The authentication middleware is broken"

        count = store_self_eval_failure(
            store, None,
            task_id="t", repo_slug="r", agent_type="e",
            acceptance_criteria=criteria, critique=critique,
        )
        assert count == 0
        store.remember.assert_not_called()

    def test_handles_none_memory_store(self):
        count = store_self_eval_failure(
            None, None,
            task_id="t", repo_slug="r", agent_type="e",
            acceptance_criteria=["foo"], critique="FAIL foo",
        )
        assert count == 0


# ============================================================
# aggregate_qa_findings
# ============================================================


class TestAggregateQaFindings:
    def test_stores_recurring_findings(self):
        """Findings that already exist in memory get stored with incremented count."""
        existing_mem = MemoryEntry(
            category="recurring_qa_findings",
            content="(1x) [error] Missing error handling in controller",
            tags=["qa_recurring", "error"],
        )
        store = _make_memory_store(existing_memories=[existing_mem])
        logger = _make_session_logger()

        findings = [
            {"file": "src/controller.py", "severity": "error",
             "description": "Missing error handling in controller"},
        ]

        count = aggregate_qa_findings(
            store, logger,
            task_id="task-1",
            repo_slug="owner/repo",
            structured_findings=findings,
        )

        assert count == 1
        assert store.remember.called
        # Content should have incremented count
        stored_content = store.remember.call_args.kwargs["content"]
        assert "(2x)" in stored_content

    def test_does_not_store_single_occurrence(self):
        """First-time findings (no existing memory) should not be stored."""
        store = _make_memory_store(existing_memories=[])
        findings = [
            {"file": "src/foo.py", "severity": "warning",
             "description": "Unused import statement detected"},
        ]

        count = aggregate_qa_findings(
            store, None,
            task_id="t", repo_slug="r",
            structured_findings=findings,
        )

        assert count == 0
        store.remember.assert_not_called()

    def test_duplicate_findings_in_same_task(self):
        """Multiple identical findings within one task count together."""
        store = _make_memory_store(existing_memories=[])
        findings = [
            {"file": "src/a.py", "severity": "error", "description": "Missing error handling"},
            {"file": "src/a.py", "severity": "error", "description": "Missing error handling"},
        ]

        count = aggregate_qa_findings(
            store, None,
            task_id="t", repo_slug="r",
            structured_findings=findings,
        )

        # 2 occurrences within same task meets threshold
        assert count == 1

    def test_returns_zero_when_disabled(self):
        store = _make_memory_store(enabled=False)
        count = aggregate_qa_findings(
            store, None,
            task_id="t", repo_slug="r",
            structured_findings=[{"description": "x"}],
        )
        assert count == 0

    def test_returns_zero_for_empty_findings(self):
        store = _make_memory_store()
        count = aggregate_qa_findings(
            store, None,
            task_id="t", repo_slug="r",
            structured_findings=[],
        )
        assert count == 0

    def test_session_logger_called(self):
        """Session logger records aggregation event when findings are stored."""
        existing_mem = MemoryEntry(
            category="recurring_qa_findings",
            content="(1x) [error] Missing error handling",
            tags=["qa_recurring"],
        )
        store = _make_memory_store(existing_memories=[existing_mem])
        logger = _make_session_logger()

        findings = [
            {"file": "src/x.py", "severity": "error",
             "description": "Missing error handling"},
        ]

        aggregate_qa_findings(
            store, logger,
            task_id="task-1", repo_slug="owner/repo",
            structured_findings=findings,
        )

        logger.log.assert_called_once_with(
            "feedback_bus_qa_aggregated",
            task_id="task-1",
            repo="owner/repo",
            findings_input=1,
            recurring_stored=1,
        )


# ============================================================
# _finding_key
# ============================================================


class TestFindingKey:
    def test_same_finding_produces_same_key(self):
        f1 = {"file": "src/foo.py", "severity": "error", "description": "Missing return type"}
        f2 = {"file": "src/foo.py", "severity": "error", "description": "Missing return type"}
        assert _finding_key(f1) == _finding_key(f2)

    def test_different_files_produce_different_keys(self):
        f1 = {"file": "src/foo.py", "severity": "error", "description": "Missing return type"}
        f2 = {"file": "src/bar.py", "severity": "error", "description": "Missing return type"}
        assert _finding_key(f1) != _finding_key(f2)

    def test_different_descriptions_produce_different_keys(self):
        f1 = {"file": "src/foo.py", "severity": "error", "description": "Missing return type"}
        f2 = {"file": "src/foo.py", "severity": "error", "description": "Unused variable x"}
        assert _finding_key(f1) != _finding_key(f2)


# ============================================================
# store_debate_specialization_hint
# ============================================================


class TestStoreDebateSpecializationHint:
    def test_stores_backend_hint_for_high_confidence(self):
        store = _make_memory_store()
        logger = _make_session_logger()

        domain = store_debate_specialization_hint(
            store, logger,
            repo_slug="owner/repo",
            debate_topic="Should we use Redis or in-memory cache?",
            synthesis="The API server and database cache layer strongly favor Redis for persistence.",
            confidence="high",
        )

        assert domain == "backend"
        store.remember.assert_called_once()
        call_kwargs = store.remember.call_args.kwargs
        assert call_kwargs["category"] == "specialization_hints"
        assert "backend" in call_kwargs["tags"]

        logger.log.assert_called_once()
        log_kwargs = logger.log.call_args.kwargs
        assert log_kwargs["repo"] == "owner/repo"
        assert log_kwargs["domain"] == "backend"
        assert log_kwargs["score"] >= 2
        assert log_kwargs["confidence"] == "high"

    def test_ignores_low_confidence(self):
        store = _make_memory_store()
        domain = store_debate_specialization_hint(
            store, None,
            repo_slug="r",
            debate_topic="Should we use Redis?",
            synthesis="Redis is good for caching API responses from the server.",
            confidence="low",
        )
        assert domain is None
        store.remember.assert_not_called()

    def test_ignores_when_no_domain_keywords(self):
        store = _make_memory_store()
        domain = store_debate_specialization_hint(
            store, None,
            repo_slug="r",
            debate_topic="Should we use tabs or spaces?",
            synthesis="Tabs are more accessible but spaces are more common.",
            confidence="high",
        )
        assert domain is None
        store.remember.assert_not_called()

    def test_returns_none_when_disabled(self):
        store = _make_memory_store(enabled=False)
        domain = store_debate_specialization_hint(
            store, None,
            repo_slug="r",
            debate_topic="x",
            synthesis="API server database endpoint",
            confidence="high",
        )
        assert domain is None

    def test_detects_frontend_domain(self):
        store = _make_memory_store()
        domain = store_debate_specialization_hint(
            store, None,
            repo_slug="r",
            debate_topic="Component architecture",
            synthesis="The React component tree needs better DOM management for the UI.",
            confidence="high",
        )
        assert domain == "frontend"

    def test_detects_infrastructure_domain(self):
        store = _make_memory_store()
        domain = store_debate_specialization_hint(
            store, None,
            repo_slug="r",
            debate_topic="Deployment strategy",
            synthesis="We should use Docker containers deployed to Kubernetes with Terraform.",
            confidence="very high",
        )
        assert domain == "infrastructure"


# ============================================================
# format_known_pitfalls
# ============================================================


class TestFormatKnownPitfalls:
    def test_formats_recurring_and_missed(self):
        recurring = [
            MemoryEntry(category="recurring_qa_findings",
                        content="(3x) [error] Missing null check"),
        ]
        missed = [
            MemoryEntry(category="missed_criteria",
                        content="Commonly missed: Add input validation"),
        ]

        store = MagicMock()
        store.enabled = True

        def mock_recall(repo_slug, agent_type, category, limit=20):
            if category == "recurring_qa_findings":
                return recurring
            if category == "missed_criteria":
                return missed
            return []

        store.recall = mock_recall

        result = format_known_pitfalls(
            store,
            repo_slug="owner/repo",
            agent_type="engineer",
        )

        assert "## Known Pitfalls" in result
        assert "Recurring QA Findings" in result
        assert "(3x)" in result
        assert "Commonly Missed Criteria" in result
        assert "input validation" in result

    def test_returns_empty_when_no_findings(self):
        store = _make_memory_store(existing_memories=[])
        result = format_known_pitfalls(
            store, repo_slug="r", agent_type="e",
        )
        assert result == ""

    def test_returns_empty_when_disabled(self):
        store = _make_memory_store(enabled=False)
        result = format_known_pitfalls(
            store, repo_slug="r", agent_type="e",
        )
        assert result == ""

    def test_respects_max_chars(self):
        recurring = [
            MemoryEntry(category="recurring_qa_findings",
                        content=f"(2x) [error] Issue number {i} " + "x" * 100)
            for i in range(20)
        ]

        store = MagicMock()
        store.enabled = True
        store.recall = MagicMock(return_value=recurring)

        result = format_known_pitfalls(
            store, repo_slug="r", agent_type="e", max_chars=300,
        )

        # Should be truncated well under the full output
        assert len(result) < 500

    def test_handles_none_memory_store(self):
        result = format_known_pitfalls(
            None, repo_slug="r", agent_type="e",
        )
        assert result == ""
