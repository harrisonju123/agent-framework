"""Condition evaluators for workflow edge transitions."""

import fnmatch
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.task import Task
    from ..core.routing import RoutingSignal

from .dag import EdgeCondition, EdgeConditionType

logger = logging.getLogger(__name__)


class ConditionEvaluator(ABC):
    """Base class for condition evaluators."""

    @abstractmethod
    def evaluate(
        self,
        condition: EdgeCondition,
        task: "Task",
        response: Any,
        routing_signal: Optional["RoutingSignal"] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate whether the condition is met.

        Args:
            condition: The condition to evaluate
            task: The current task
            response: The LLM response
            routing_signal: Optional routing signal from agent
            context: Additional evaluation context

        Returns:
            True if condition is met, False otherwise
        """
        pass


class AlwaysCondition(ConditionEvaluator):
    """Unconditional edge (always true)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        return True


class PRCreatedCondition(ConditionEvaluator):
    """True if a PR was created in this task."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        # Check task context for pr_url
        if task.context and "pr_url" in task.context:
            return True

        # Check response content for PR URL patterns
        if hasattr(response, "content") and response.content:
            content = str(response.content)
            # Common PR URL patterns
            pr_patterns = [
                "github.com/",
                "/pull/",
                "Created PR:",
                "Pull Request:",
            ]
            if any(pattern in content for pattern in pr_patterns):
                return True

        return False


class NoPRCondition(ConditionEvaluator):
    """True if no PR was created (inverse of PRCreatedCondition)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        pr_condition = PRCreatedCondition()
        return not pr_condition.evaluate(condition, task, response, routing_signal, context)


class ApprovedCondition(ConditionEvaluator):
    """True if QA approved the changes."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        # Check for approval indicators in response
        if hasattr(response, "content") and response.content:
            content = str(response.content).lower()
            approval_keywords = [
                "approved",
                "looks good",
                "lgtm",
                "ready for merge",
                "passes all checks",
            ]
            rejection_keywords = [
                "needs fix",
                "failed",
                "issues found",
                "problems detected",
            ]

            # Reject if rejection keywords found
            if any(keyword in content for keyword in rejection_keywords):
                return False

            # Approve if approval keywords found
            if any(keyword in content for keyword in approval_keywords):
                return True

        # Default to False (require explicit approval)
        return False


class NeedsFixCondition(ConditionEvaluator):
    """True if QA found issues that need fixing (inverse of ApprovedCondition)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        approved_condition = ApprovedCondition()
        return not approved_condition.evaluate(condition, task, response, routing_signal, context)


class TestPassedCondition(ConditionEvaluator):
    """True if tests passed."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        # Check context for test results
        if context and "test_result" in context:
            return context["test_result"] == "passed"

        # Check response content for test pass indicators
        if hasattr(response, "content") and response.content:
            content = str(response.content).lower()
            if "tests passed" in content or "all tests pass" in content:
                return True
            if "test failed" in content or "tests failed" in content:
                return False

        # Default to True if no test information found
        return True


class TestFailedCondition(ConditionEvaluator):
    """True if tests failed (inverse of TestPassedCondition)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        test_passed = TestPassedCondition()
        return not test_passed.evaluate(condition, task, response, routing_signal, context)


class FilesMatchCondition(ConditionEvaluator):
    """True if changed files match a glob pattern."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        pattern = condition.params.get("pattern", "")
        if not pattern:
            logger.warning("files_match condition missing 'pattern' parameter")
            return False

        # Get changed files from context
        changed_files = []
        if context and "changed_files" in context:
            changed_files = context["changed_files"]
        elif task.context and "changed_files" in task.context:
            changed_files = task.context["changed_files"]

        if not changed_files:
            # No files to check, default to False
            return False

        # Check if any file matches the pattern
        for file_path in changed_files:
            if fnmatch.fnmatch(str(file_path), pattern):
                return True

        return False


class PRSizeUnderCondition(ConditionEvaluator):
    """True if PR size (number of changed files) is under threshold."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        max_files = condition.params.get("max_files", 0)
        if max_files <= 0:
            logger.warning("pr_size_under condition has invalid 'max_files' parameter")
            return False

        # Get changed files count
        changed_files = []
        if context and "changed_files" in context:
            changed_files = context["changed_files"]
        elif task.context and "changed_files" in task.context:
            changed_files = task.context["changed_files"]

        return len(changed_files) < max_files


class SignalTargetCondition(ConditionEvaluator):
    """True if routing signal targets this edge's target agent."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        if not routing_signal:
            return False

        target = condition.params.get("target", "")
        if not target:
            logger.warning("signal_target condition missing 'target' parameter")
            return False

        # Check if routing signal targets this specific agent
        return routing_signal.target_agent == target


class ConditionRegistry:
    """Registry mapping condition types to evaluators."""

    _evaluators: Dict[EdgeConditionType, ConditionEvaluator] = {
        EdgeConditionType.ALWAYS: AlwaysCondition(),
        EdgeConditionType.PR_CREATED: PRCreatedCondition(),
        EdgeConditionType.NO_PR: NoPRCondition(),
        EdgeConditionType.APPROVED: ApprovedCondition(),
        EdgeConditionType.NEEDS_FIX: NeedsFixCondition(),
        EdgeConditionType.TEST_PASSED: TestPassedCondition(),
        EdgeConditionType.TEST_FAILED: TestFailedCondition(),
        EdgeConditionType.FILES_MATCH: FilesMatchCondition(),
        EdgeConditionType.PR_SIZE_UNDER: PRSizeUnderCondition(),
        EdgeConditionType.SIGNAL_TARGET: SignalTargetCondition(),
    }

    @classmethod
    def evaluate(
        cls,
        condition: EdgeCondition,
        task: "Task",
        response: Any,
        routing_signal: Optional["RoutingSignal"] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate a condition using the appropriate evaluator.

        Args:
            condition: The condition to evaluate
            task: The current task
            response: The LLM response
            routing_signal: Optional routing signal from agent
            context: Additional evaluation context

        Returns:
            True if condition is met, False otherwise
        """
        evaluator = cls._evaluators.get(condition.type)
        if not evaluator:
            logger.error(f"No evaluator found for condition type: {condition.type}")
            return False

        try:
            return evaluator.evaluate(condition, task, response, routing_signal, context)
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.type}: {e}")
            return False

    @classmethod
    def register(cls, condition_type: EdgeConditionType, evaluator: ConditionEvaluator):
        """Register a custom condition evaluator."""
        cls._evaluators[condition_type] = evaluator
