"""Tests for FeedbackBus — cross-feature learning loop."""

from unittest.mock import MagicMock

import pytest

from agent_framework.core.feedback_bus import (
    CATEGORY_QA_RECURRING,
    CATEGORY_SELF_EVAL_GAPS,
    CATEGORY_SPECIALIZATION_HINT,
    SHARED_AGENT_TYPE,
    FeedbackBus,
    _extract_missed_criteria,
)
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def session_logger():
    return MagicMock()


@pytest.fixture
def bus(store, session_logger):
    return FeedbackBus(memory_store=store, session_logger=session_logger)


@pytest.fixture
def repo():
    return "myorg/myrepo"


def _make_task(**overrides) -> Task:
    from datetime import datetime, timezone

    defaults = dict(
        id="task-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        context={"github_repo": "myorg/myrepo"},
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestSelfEvalStorage:
    def test_stores_raw_critique_when_no_criteria_match(self, bus, store, repo):
        task = _make_task(
            context={"github_repo": repo, "_self_eval_critique": "Missing edge case handling"},
        )
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(entries) == 1
        assert "Missing edge case handling" in entries[0].content

    def test_stores_matched_criteria(self, bus, store, repo):
        task = _make_task(
            acceptance_criteria=["All tests pass", "Error handling complete"],
            context={
                "github_repo": repo,
                "_self_eval_critique": "Tests are not passing, several test failures found",
            },
        )
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(entries) >= 1
        contents = [e.content for e in entries]
        assert any("All tests pass" in c for c in contents)

    def test_no_critique_no_storage(self, bus, store, repo):
        task = _make_task()
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(entries) == 0

    def test_emits_session_event(self, bus, session_logger, repo):
        task = _make_task(
            context={"github_repo": repo, "_self_eval_critique": "Failed checks"},
        )
        bus.process(task, repo)

        session_logger.log.assert_any_call(
            "feedback_bus_self_eval_stored",
            repo=repo,
            missed_count=1,
            task_id=task.id,
        )

    def test_skips_when_memory_disabled(self, tmp_path, repo):
        disabled_store = MemoryStore(workspace=tmp_path, enabled=False)
        bus = FeedbackBus(memory_store=disabled_store)
        task = _make_task(
            context={"github_repo": repo, "_self_eval_critique": "Failed"},
        )
        bus.process(task, repo)
        # No error, no entries
        assert disabled_store.recall(repo, "engineer") == []


class TestQARecurrence:
    def test_first_occurrence_stored_per_agent(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing error handling", "file": "api.py"}],
                },
            },
        )
        bus.process(task, repo)

        agent_entries = store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING)
        assert len(agent_entries) == 1

        # Not yet promoted to shared (only 1 occurrence)
        shared_entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_QA_RECURRING)
        assert len(shared_entries) == 0

    def test_recurring_finding_promoted_to_shared(self, bus, store, repo):
        # First task establishes the finding
        task1 = _make_task(
            id="task-001",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing error handling", "file": "api.py"}],
                },
            },
        )
        bus.process(task1, repo)

        # Second task with same finding from different task ID
        task2 = _make_task(
            id="task-002",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing error handling", "file": "api.py"}],
                },
            },
        )
        bus.process(task2, repo)

        shared_entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_QA_RECURRING)
        assert len(shared_entries) == 1
        assert "Missing error handling" in shared_entries[0].content

    def test_emits_event_on_promotion(self, bus, session_logger, store, repo):
        # Seed first occurrence
        task1 = _make_task(
            id="task-001",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing tests"}],
                },
            },
        )
        bus.process(task1, repo)

        # Second triggers promotion
        task2 = _make_task(
            id="task-002",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing tests"}],
                },
            },
        )
        bus.process(task2, repo)

        session_logger.log.assert_any_call(
            "feedback_bus_qa_recurring_detected",
            repo=repo,
            promoted_count=1,
            total_findings=1,
            task_id="task-002",
        )

    def test_no_findings_no_storage(self, bus, store, repo):
        task = _make_task(context={"github_repo": repo})
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING)
        assert len(entries) == 0

    def test_empty_description_skipped(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "", "file": "api.py"}],
                },
            },
        )
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING)
        assert len(entries) == 0


class TestSpecializationFromDebate:
    def test_high_confidence_domain_debate_stored(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Which database technology should we use for caching?",
                    "recommendation": "Use Redis for hot cache layer",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)

        entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT)
        assert len(entries) == 1
        assert "Redis" in entries[0].content

    def test_low_confidence_ignored(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Should we use a new framework?",
                    "recommendation": "Maybe consider it",
                    "confidence": "low",
                },
            },
        )
        bus.process(task, repo)

        entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT)
        assert len(entries) == 0

    def test_non_domain_topic_ignored(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Should we add more comments?",
                    "recommendation": "Yes, add comments",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)

        entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT)
        assert len(entries) == 0

    def test_emits_session_event(self, bus, session_logger, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Which database technology to use?",
                    "recommendation": "PostgreSQL",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)

        session_logger.log.assert_any_call(
            "feedback_bus_specialization_updated",
            repo=repo,
            topic="Which database technology to use?",
            confidence="high",
            task_id=task.id,
        )

    def test_no_debate_result_no_storage(self, bus, store, repo):
        task = _make_task()
        bus.process(task, repo)

        entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT)
        assert len(entries) == 0


class TestExtractMissedCriteria:
    def test_matches_criteria_by_keyword(self):
        critique = "The tests are failing with multiple errors in the test suite"
        criteria = ["All tests pass", "API docs updated"]
        result = _extract_missed_criteria(critique, criteria)
        assert "All tests pass" in result

    def test_no_match_returns_empty(self):
        critique = "Everything looks good"
        criteria = ["Database migration complete"]
        result = _extract_missed_criteria(critique, criteria)
        assert result == []

    def test_empty_criteria_returns_empty(self):
        result = _extract_missed_criteria("some critique", [])
        assert result == []

    def test_short_words_ignored(self):
        # Criteria with only short words should be skipped
        critique = "Something failed"
        criteria = ["Do it"]
        result = _extract_missed_criteria(critique, criteria)
        assert result == []

    def test_multiple_criteria_matched(self):
        critique = "Tests are failing and error handling is incomplete throughout the code"
        criteria = ["All tests pass", "Error handling complete", "API docs updated"]
        result = _extract_missed_criteria(critique, criteria)
        assert "All tests pass" in result
        assert "Error handling complete" in result
        assert "API docs updated" not in result


class TestExtractFileTags:
    def test_extracts_file_path(self):
        from agent_framework.core.feedback_bus import _extract_file_tags

        finding = {"description": "issue", "file": "src/api.py"}
        tags = _extract_file_tags(finding)
        assert "qa_finding" in tags
        assert "src/api.py" in tags

    def test_no_file_returns_base_tag(self):
        from agent_framework.core.feedback_bus import _extract_file_tags

        finding = {"description": "issue"}
        tags = _extract_file_tags(finding)
        assert tags == ["qa_finding"]

    def test_empty_file_returns_base_tag(self):
        from agent_framework.core.feedback_bus import _extract_file_tags

        finding = {"description": "issue", "file": ""}
        tags = _extract_file_tags(finding)
        assert tags == ["qa_finding"]


class TestReplanSuccessDelegation:
    """Verify store_replan_success delegates to ErrorRecoveryManager."""

    def test_delegates_to_error_recovery(self, store, session_logger, repo):
        error_recovery = MagicMock()
        bus = FeedbackBus(
            memory_store=store,
            session_logger=session_logger,
            error_recovery=error_recovery,
        )
        task = _make_task(
            context={"github_repo": repo},
            replan_history=[{"error_type": "test_failure", "files_involved": ["api.py"]}],
        )
        bus.process(task, repo)
        error_recovery.store_replan_outcome.assert_called_once_with(task, repo)

    def test_skips_when_no_replan_history(self, store, session_logger, repo):
        error_recovery = MagicMock()
        bus = FeedbackBus(
            memory_store=store,
            session_logger=session_logger,
            error_recovery=error_recovery,
        )
        task = _make_task(context={"github_repo": repo})
        bus.process(task, repo)
        error_recovery.store_replan_outcome.assert_not_called()

    def test_skips_when_no_error_recovery(self, bus, store, repo):
        """Bus without error_recovery just skips replan delegation."""
        task = _make_task(
            context={"github_repo": repo},
            replan_history=[{"error_type": "test_failure"}],
        )
        bus.process(task, repo)  # No exception

    def test_skips_when_task_not_completed(self, store, session_logger, repo):
        error_recovery = MagicMock()
        bus = FeedbackBus(
            memory_store=store,
            session_logger=session_logger,
            error_recovery=error_recovery,
        )
        task = _make_task(
            status=TaskStatus.FAILED,
            context={"github_repo": repo},
            replan_history=[{"error_type": "test_failure"}],
        )
        bus.process(task, repo)
        error_recovery.store_replan_outcome.assert_not_called()


class TestProcessAllCollectors:
    """Verify that process() fires multiple collectors when context has data for all."""

    def test_all_collectors_fire_together(self, bus, store, repo):
        task = _make_task(
            id="task-multi",
            acceptance_criteria=["Tests pass"],
            context={
                "github_repo": repo,
                "_self_eval_critique": "Tests are not passing",
                "structured_findings": {
                    "findings": [{"description": "Missing validation", "file": "api.py"}],
                },
                "debate_result": {
                    "topic": "Which database technology for caching?",
                    "recommendation": "Use Redis",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)

        # Self-eval gap stored
        eval_entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(eval_entries) >= 1

        # QA finding stored (per-agent, not promoted since first occurrence)
        qa_entries = store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING)
        assert len(qa_entries) >= 1

        # Specialization hint stored
        spec_entries = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT)
        assert len(spec_entries) == 1


class TestEdgeCases:
    """Edge cases for FeedbackBus collectors."""

    def test_structured_findings_non_dict_skipped(self, bus, store, repo):
        """structured_findings that isn't a dict is ignored."""
        task = _make_task(
            context={"github_repo": repo, "structured_findings": "not a dict"},
        )
        bus.process(task, repo)
        assert store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING) == []

    def test_debate_result_non_dict_skipped(self, bus, store, repo):
        """debate_result that isn't a dict is ignored."""
        task = _make_task(
            context={"github_repo": repo, "debate_result": "not a dict"},
        )
        bus.process(task, repo)
        assert store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT) == []

    def test_debate_empty_topic_skipped(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "",
                    "recommendation": "Use Redis",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)
        assert store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT) == []

    def test_debate_empty_recommendation_skipped(self, bus, store, repo):
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Which database technology?",
                    "recommendation": "",
                    "confidence": "high",
                },
            },
        )
        bus.process(task, repo)
        assert store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT) == []

    def test_same_task_finding_does_not_self_promote(self, bus, store, repo):
        """Processing the same task twice shouldn't promote to shared (same task ID)."""
        task = _make_task(
            id="task-dup",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing tests"}],
                },
            },
        )
        bus.process(task, repo)
        bus.process(task, repo)

        shared = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_QA_RECURRING)
        assert len(shared) == 0

    def test_critique_truncated_at_500_chars(self, bus, store, repo):
        """Long critique without matching criteria is truncated to 500 chars."""
        long_critique = "X" * 1000
        task = _make_task(
            context={"github_repo": repo, "_self_eval_critique": long_critique},
        )
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(entries) == 1
        # "Self-eval failed: " prefix + 500 chars
        assert len(entries[0].content) <= len("Self-eval failed: ") + 500

    def test_multiple_findings_mixed_recurrence(self, bus, store, repo):
        """Multiple findings in one task: only the recurring one gets promoted."""
        # Seed first occurrence of one finding
        task1 = _make_task(
            id="task-seed",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing error handling"}],
                },
            },
        )
        bus.process(task1, repo)

        # Second task with recurring + new finding
        task2 = _make_task(
            id="task-new",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [
                        {"description": "Missing error handling"},
                        {"description": "Unused import"},
                    ],
                },
            },
        )
        bus.process(task2, repo)

        shared = store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_QA_RECURRING)
        assert len(shared) == 1
        assert "Missing error handling" in shared[0].content

    def test_assigned_to_empty_defaults_to_engineer(self, bus, store, repo):
        """Task with empty assigned_to falls back to 'engineer' agent type."""
        task = _make_task(
            assigned_to="",
            context={"github_repo": repo, "_self_eval_critique": "Failed checks"},
        )
        bus.process(task, repo)

        entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(entries) == 1

    def test_different_agent_types_stored_separately(self, store, repo):
        """Findings from architect vs engineer are stored under their respective types."""
        bus = FeedbackBus(memory_store=store)
        task_eng = _make_task(
            id="eng-task",
            assigned_to="engineer",
            context={
                "github_repo": repo,
                "_self_eval_critique": "Engineer missed edge cases",
            },
        )
        task_arch = _make_task(
            id="arch-task",
            assigned_to="architect",
            context={
                "github_repo": repo,
                "_self_eval_critique": "Architect missed design review",
            },
        )
        bus.process(task_eng, repo)
        bus.process(task_arch, repo)

        eng_entries = store.recall(repo, "engineer", category=CATEGORY_SELF_EVAL_GAPS)
        arch_entries = store.recall(repo, "architect", category=CATEGORY_SELF_EVAL_GAPS)
        assert len(eng_entries) == 1
        assert len(arch_entries) == 1
        assert "Engineer" in eng_entries[0].content
        assert "Architect" in arch_entries[0].content

    def test_findings_empty_list_no_storage(self, bus, store, repo):
        """structured_findings with empty findings list stores nothing."""
        task = _make_task(
            context={
                "github_repo": repo,
                "structured_findings": {"findings": []},
            },
        )
        bus.process(task, repo)
        assert store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING) == []

    def test_debate_missing_recommendation_key(self, bus, store, repo):
        """debate_result missing the 'recommendation' key is safely skipped."""
        task = _make_task(
            context={
                "github_repo": repo,
                "debate_result": {
                    "topic": "Which database technology?",
                    "confidence": "high",
                    # 'recommendation' key missing
                },
            },
        )
        bus.process(task, repo)
        assert store.recall(repo, SHARED_AGENT_TYPE, category=CATEGORY_SPECIALIZATION_HINT) == []

    def test_qa_findings_stored_under_correct_agent(self, bus, store, repo):
        """QA findings are stored under the task's assigned_to agent, not 'shared'."""
        task = _make_task(
            assigned_to="qa",
            context={
                "github_repo": repo,
                "structured_findings": {
                    "findings": [{"description": "Missing validation"}],
                },
            },
        )
        bus.process(task, repo)

        qa_entries = store.recall(repo, "qa", category=CATEGORY_QA_RECURRING)
        assert len(qa_entries) == 1
        # Not stored under engineer
        engineer_entries = store.recall(repo, "engineer", category=CATEGORY_QA_RECURRING)
        assert len(engineer_entries) == 0

    def test_replan_in_progress_skipped(self, store, session_logger, repo):
        """Tasks still in-progress don't trigger replan storage."""
        error_recovery = MagicMock()
        bus = FeedbackBus(
            memory_store=store,
            session_logger=session_logger,
            error_recovery=error_recovery,
        )
        task = _make_task(
            status=TaskStatus.IN_PROGRESS,
            context={"github_repo": repo},
            replan_history=[{"error_type": "test_failure"}],
        )
        bus.process(task, repo)
        error_recovery.store_replan_outcome.assert_not_called()


class TestCriteriaExtractionEdgeCases:
    """Additional edge cases for _extract_missed_criteria matching logic."""

    def test_partial_word_overlap_requires_half_match(self):
        """Criterion with 4+ words requires >= half to match."""
        critique = "The database migration scripts are missing"
        criteria = ["Database migration complete and verified"]
        result = _extract_missed_criteria(critique, criteria)
        assert "Database migration complete and verified" in result

    def test_single_long_word_criterion_matches(self):
        """Criterion with only one significant word still matches."""
        critique = "Authentication is broken across the app"
        criteria = ["Authentication works"]
        result = _extract_missed_criteria(critique, criteria)
        assert "Authentication works" in result

    def test_case_insensitive_matching(self):
        """Matching is case-insensitive."""
        critique = "ALL TESTS ARE FAILING"
        criteria = ["all tests pass"]
        result = _extract_missed_criteria(critique, criteria)
        assert "all tests pass" in result
