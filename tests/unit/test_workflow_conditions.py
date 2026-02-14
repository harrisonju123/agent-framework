"""Tests for workflow condition evaluators."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.core.routing import RoutingSignal
from agent_framework.workflow.dag import EdgeCondition, EdgeConditionType
from agent_framework.workflow.conditions import (
    ConditionRegistry,
    AlwaysCondition,
    PRCreatedCondition,
    NoPRCondition,
    ApprovedCondition,
    NeedsFixCondition,
    TestPassedCondition,
    TestFailedCondition,
    FilesMatchCondition,
    PRSizeUnderCondition,
    SignalTargetCondition,
)


def _make_task(**context_overrides):
    """Create a test task."""
    context = {"workflow": "default", **context_overrides}
    return Task(
        id="test-task-123",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.utcnow(),
        title="Test task",
        description="Test description",
        context=context,
    )


def _make_response(content="Task completed successfully"):
    """Create a mock response."""
    return SimpleNamespace(content=content, error=None)


class TestAlwaysCondition:
    def test_always_true(self):
        """ALWAYS condition always returns True."""
        condition = EdgeCondition(EdgeConditionType.ALWAYS)
        task = _make_task()
        response = _make_response()

        evaluator = AlwaysCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_registry_always(self):
        """Test ALWAYS condition through registry."""
        condition = EdgeCondition(EdgeConditionType.ALWAYS)
        task = _make_task()
        response = _make_response()

        assert ConditionRegistry.evaluate(condition, task, response) is True


class TestPRCreatedCondition:
    def test_pr_in_context(self):
        """PR URL in task context."""
        condition = EdgeCondition(EdgeConditionType.PR_CREATED)
        task = _make_task(pr_url="https://github.com/org/repo/pull/42")
        response = _make_response()

        evaluator = PRCreatedCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_pr_in_response(self):
        """PR URL in response content."""
        condition = EdgeCondition(EdgeConditionType.PR_CREATED)
        task = _make_task()
        response = _make_response("Created PR: https://github.com/org/repo/pull/99")

        evaluator = PRCreatedCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_no_pr(self):
        """No PR created."""
        condition = EdgeCondition(EdgeConditionType.PR_CREATED)
        task = _make_task()
        response = _make_response("Implemented feature X")

        evaluator = PRCreatedCondition()
        assert evaluator.evaluate(condition, task, response) is False


class TestNoPRCondition:
    def test_no_pr_true(self):
        """NO_PR is true when no PR created."""
        condition = EdgeCondition(EdgeConditionType.NO_PR)
        task = _make_task()
        response = _make_response("Implemented feature")

        evaluator = NoPRCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_no_pr_false(self):
        """NO_PR is false when PR exists."""
        condition = EdgeCondition(EdgeConditionType.NO_PR)
        task = _make_task(pr_url="https://github.com/org/repo/pull/42")
        response = _make_response()

        evaluator = NoPRCondition()
        assert evaluator.evaluate(condition, task, response) is False


class TestApprovedCondition:
    def test_approved_keywords(self):
        """Detect approval keywords in response."""
        condition = EdgeCondition(EdgeConditionType.APPROVED)
        task = _make_task()

        approval_phrases = [
            "Code looks good, approved",
            "LGTM - ready to merge",
            "All checks pass, ready for merge",
            "Tests passed, approved",
        ]

        evaluator = ApprovedCondition()
        for phrase in approval_phrases:
            response = _make_response(phrase)
            assert evaluator.evaluate(condition, task, response) is True, \
                f"Failed to detect approval in: {phrase}"

    def test_rejection_keywords(self):
        """Detect rejection keywords in response."""
        condition = EdgeCondition(EdgeConditionType.APPROVED)
        task = _make_task()

        rejection_phrases = [
            "Tests failed, needs fix",
            "Found issues in the code",
            "Problems detected in implementation",
        ]

        evaluator = ApprovedCondition()
        for phrase in rejection_phrases:
            response = _make_response(phrase)
            assert evaluator.evaluate(condition, task, response) is False, \
                f"Incorrectly approved: {phrase}"

    def test_no_clear_verdict(self):
        """Default to False when no clear approval/rejection."""
        condition = EdgeCondition(EdgeConditionType.APPROVED)
        task = _make_task()
        response = _make_response("Reviewed the code")

        evaluator = ApprovedCondition()
        assert evaluator.evaluate(condition, task, response) is False


class TestNeedsFixCondition:
    def test_needs_fix_true(self):
        """NEEDS_FIX is true when not approved."""
        condition = EdgeCondition(EdgeConditionType.NEEDS_FIX)
        task = _make_task()
        response = _make_response("Issues found, needs fix")

        evaluator = NeedsFixCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_needs_fix_false(self):
        """NEEDS_FIX is false when approved."""
        condition = EdgeCondition(EdgeConditionType.NEEDS_FIX)
        task = _make_task()
        response = _make_response("All good, approved")

        evaluator = NeedsFixCondition()
        assert evaluator.evaluate(condition, task, response) is False


class TestTestPassedCondition:
    def test_test_passed_in_response(self):
        """Detect test pass in response."""
        condition = EdgeCondition(EdgeConditionType.TEST_PASSED)
        task = _make_task()
        response = _make_response("All tests passed successfully")

        evaluator = TestPassedCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_test_failed_in_response(self):
        """Detect test failure in response."""
        condition = EdgeCondition(EdgeConditionType.TEST_PASSED)
        task = _make_task()
        response = _make_response("Tests failed with 3 errors")

        evaluator = TestPassedCondition()
        assert evaluator.evaluate(condition, task, response) is False

    def test_test_passed_in_context(self):
        """Check test result in context."""
        condition = EdgeCondition(EdgeConditionType.TEST_PASSED)
        task = _make_task()
        response = _make_response()
        context = {"test_result": "passed"}

        evaluator = TestPassedCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is True

    def test_test_failed_in_context(self):
        """Check test failure in context."""
        condition = EdgeCondition(EdgeConditionType.TEST_PASSED)
        task = _make_task()
        response = _make_response()
        context = {"test_result": "failed"}

        evaluator = TestPassedCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is False

    def test_no_test_info_defaults_true(self):
        """Default to True when no test information."""
        condition = EdgeCondition(EdgeConditionType.TEST_PASSED)
        task = _make_task()
        response = _make_response("Completed implementation")

        evaluator = TestPassedCondition()
        assert evaluator.evaluate(condition, task, response) is True


class TestFilesMatchCondition:
    def test_pattern_match(self):
        """Match files against glob pattern."""
        condition = EdgeCondition(
            EdgeConditionType.FILES_MATCH,
            params={"pattern": "*.md"}
        )
        task = _make_task()
        response = _make_response()
        context = {"changed_files": ["README.md", "CONTRIBUTING.md"]}

        evaluator = FilesMatchCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is True

    def test_pattern_no_match(self):
        """No files match pattern."""
        condition = EdgeCondition(
            EdgeConditionType.FILES_MATCH,
            params={"pattern": "*.md"}
        )
        task = _make_task()
        response = _make_response()
        context = {"changed_files": ["src/main.py", "tests/test_main.py"]}

        evaluator = FilesMatchCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is False

    def test_changed_files_in_task_context(self):
        """Get changed files from task context."""
        condition = EdgeCondition(
            EdgeConditionType.FILES_MATCH,
            params={"pattern": "*.py"}
        )
        task = _make_task(changed_files=["src/main.py"])
        response = _make_response()

        evaluator = FilesMatchCondition()
        assert evaluator.evaluate(condition, task, response) is True

    def test_no_changed_files(self):
        """Return False when no files to check."""
        condition = EdgeCondition(
            EdgeConditionType.FILES_MATCH,
            params={"pattern": "*.md"}
        )
        task = _make_task()
        response = _make_response()

        evaluator = FilesMatchCondition()
        assert evaluator.evaluate(condition, task, response) is False


class TestPRSizeUnderCondition:
    def test_size_under_threshold(self):
        """PR size is under threshold."""
        condition = EdgeCondition(
            EdgeConditionType.PR_SIZE_UNDER,
            params={"max_files": 5}
        )
        task = _make_task()
        response = _make_response()
        context = {"changed_files": ["file1.py", "file2.py", "file3.py"]}

        evaluator = PRSizeUnderCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is True

    def test_size_over_threshold(self):
        """PR size is over threshold."""
        condition = EdgeCondition(
            EdgeConditionType.PR_SIZE_UNDER,
            params={"max_files": 3}
        )
        task = _make_task()
        response = _make_response()
        context = {"changed_files": ["f1.py", "f2.py", "f3.py", "f4.py", "f5.py"]}

        evaluator = PRSizeUnderCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is False

    def test_size_exactly_at_threshold(self):
        """PR size exactly at threshold (should be false, needs to be UNDER)."""
        condition = EdgeCondition(
            EdgeConditionType.PR_SIZE_UNDER,
            params={"max_files": 3}
        )
        task = _make_task()
        response = _make_response()
        context = {"changed_files": ["f1.py", "f2.py", "f3.py"]}

        evaluator = PRSizeUnderCondition()
        assert evaluator.evaluate(condition, task, response, context=context) is False


class TestSignalTargetCondition:
    def test_signal_matches_target(self):
        """Routing signal matches edge target."""
        condition = EdgeCondition(
            EdgeConditionType.SIGNAL_TARGET,
            params={"target": "qa"}
        )
        task = _make_task()
        response = _make_response()
        routing_signal = RoutingSignal(
            target_agent="qa",
            reason="Ready for review",
            timestamp="2026-02-13T10:00:00Z",
            source_agent="engineer"
        )

        evaluator = SignalTargetCondition()
        assert evaluator.evaluate(condition, task, response, routing_signal=routing_signal) is True

    def test_signal_no_match(self):
        """Routing signal doesn't match edge target."""
        condition = EdgeCondition(
            EdgeConditionType.SIGNAL_TARGET,
            params={"target": "qa"}
        )
        task = _make_task()
        response = _make_response()
        routing_signal = RoutingSignal(
            target_agent="architect",
            reason="Need design review",
            timestamp="2026-02-13T10:00:00Z",
            source_agent="engineer"
        )

        evaluator = SignalTargetCondition()
        assert evaluator.evaluate(condition, task, response, routing_signal=routing_signal) is False

    def test_no_routing_signal(self):
        """Return False when no routing signal provided."""
        condition = EdgeCondition(
            EdgeConditionType.SIGNAL_TARGET,
            params={"target": "qa"}
        )
        task = _make_task()
        response = _make_response()

        evaluator = SignalTargetCondition()
        assert evaluator.evaluate(condition, task, response, routing_signal=None) is False


class TestConditionRegistry:
    def test_evaluate_all_conditions(self):
        """Registry can evaluate all condition types."""
        task = _make_task()
        response = _make_response()

        condition_types = [
            EdgeConditionType.ALWAYS,
            EdgeConditionType.PR_CREATED,
            EdgeConditionType.NO_PR,
            EdgeConditionType.APPROVED,
            EdgeConditionType.NEEDS_FIX,
            EdgeConditionType.TEST_PASSED,
            EdgeConditionType.TEST_FAILED,
        ]

        for cond_type in condition_types:
            if cond_type in [EdgeConditionType.FILES_MATCH, EdgeConditionType.PR_SIZE_UNDER, EdgeConditionType.SIGNAL_TARGET]:
                continue  # Skip conditions that require parameters

            condition = EdgeCondition(cond_type)
            # Should not raise exception
            result = ConditionRegistry.evaluate(condition, task, response)
            assert isinstance(result, bool)

    def test_unknown_condition_type(self):
        """Unknown condition type returns False."""
        # Create a mock condition with invalid type (bypass validation)
        condition = EdgeCondition.__new__(EdgeCondition)
        condition.type = "INVALID_CONDITION"
        condition.params = {}

        task = _make_task()
        response = _make_response()

        result = ConditionRegistry.evaluate(condition, task, response)
        assert result is False
