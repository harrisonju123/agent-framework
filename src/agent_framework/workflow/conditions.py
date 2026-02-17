"""Condition evaluators for workflow edge transitions."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.task import Task
    from ..core.routing import RoutingSignal

from .dag import EdgeCondition, EdgeConditionType

logger = logging.getLogger(__name__)

# Rejection patterns — specific phrases that indicate actionable issues.
# Word-boundary anchored to avoid matching negated forms ("No issues found").
_REJECTION_PATTERNS = [
    re.compile(r'\bneeds?\s+fix', re.IGNORECASE),
    re.compile(r'\brequest[_\s]changes?\b', re.IGNORECASE),
    re.compile(r'\bchanges?\s+requested\b', re.IGNORECASE),
]

# Approval patterns — explicit positive verdicts.
_APPROVAL_PATTERNS = [
    re.compile(r'\bapproved?\b', re.IGNORECASE),
    re.compile(r'\blgtm\b', re.IGNORECASE),
    re.compile(r'\bready\s+for\s+merge\b', re.IGNORECASE),
    re.compile(r'\bpasses?\s+all\s+checks?\b', re.IGNORECASE),
    re.compile(r'\blooks\s+good\b', re.IGNORECASE),
]


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
        """Evaluate whether the condition is met."""
        pass


class AlwaysCondition(ConditionEvaluator):
    """Unconditional edge (always true)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        return True


class PRCreatedCondition(ConditionEvaluator):
    """True if a PR was created in this task."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        import re
        pr_pattern = re.compile(r'https://github\.com/[^/]+/[^/]+/pull/\d+')

        if task.context and "pr_url" in task.context:
            return bool(pr_pattern.search(task.context["pr_url"]))

        if hasattr(response, "content") and response.content:
            return bool(pr_pattern.search(str(response.content)))

        return False


class NoPRCondition(ConditionEvaluator):
    """True if no PR was created (inverse of PRCreatedCondition)."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        pr_condition = PRCreatedCondition()
        return not pr_condition.evaluate(condition, task, response, routing_signal, context)


class ApprovedCondition(ConditionEvaluator):
    """True if QA approved the changes.

    Prefers structured verdict from task context (set by _parse_review_outcome)
    to avoid fragile keyword matching on free-text responses.
    """

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        # Prefer structured verdict from task context or evaluation context
        verdict = None
        if context and "verdict" in context:
            verdict = context["verdict"]
        elif task.context and "verdict" in task.context:
            verdict = task.context["verdict"]

        if verdict is not None:
            return str(verdict).lower() in ("approved", "approve", "lgtm")

        # Fall back to keyword heuristic on response content
        if hasattr(response, "content") and response.content:
            content = str(response.content)

            # Check rejection first — word-boundary patterns avoid matching
            # negated forms like "No issues found"
            if any(p.search(content) for p in _REJECTION_PATTERNS):
                return False
            if any(p.search(content) for p in _APPROVAL_PATTERNS):
                return True

        return False


class NeedsFixCondition(ConditionEvaluator):
    """True if QA found issues that need fixing.

    Mirrors ApprovedCondition: prefers structured verdict, falls back to keywords.
    """

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        # Prefer structured verdict
        verdict = None
        if context and "verdict" in context:
            verdict = context["verdict"]
        elif task.context and "verdict" in task.context:
            verdict = task.context["verdict"]

        if verdict is not None:
            return str(verdict).lower() in ("needs_fix", "needs fix", "rejected", "changes_requested")

        approved_condition = ApprovedCondition()
        return not approved_condition.evaluate(condition, task, response, routing_signal, context)


class TestPassedCondition(ConditionEvaluator):
    """True if tests passed."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        if context and "test_result" in context:
            return context["test_result"] == "passed"

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
    """True if any changed file matches a glob pattern.

    Uses PurePath.match() to handle paths with directory separators
    (e.g., 'docs/README.md' matches '**/*.md').
    """

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        from pathlib import PurePath

        pattern = condition.params.get("pattern", "")
        if not pattern:
            logger.warning("files_match condition missing 'pattern' parameter")
            return False

        changed_files = []
        if context and "changed_files" in context:
            changed_files = context["changed_files"]
        elif task.context and "changed_files" in task.context:
            changed_files = task.context["changed_files"]

        if not changed_files:
            return False

        for file_path in changed_files:
            if PurePath(file_path).match(pattern):
                return True

        return False


class PRSizeUnderCondition(ConditionEvaluator):
    """True if PR size (number of changed files) is under threshold."""

    def evaluate(self, condition, task, response, routing_signal=None, context=None) -> bool:
        max_files = condition.params.get("max_files", 0)
        if max_files <= 0:
            logger.warning("pr_size_under condition has invalid 'max_files' parameter")
            return False

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

        return routing_signal.target_agent == target


def _default_evaluators() -> Dict[EdgeConditionType, ConditionEvaluator]:
    """Build a fresh evaluator map so ConditionRegistry.register() in tests
    doesn't pollute global state."""
    return {
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


class ConditionRegistry:
    """Registry mapping condition types to evaluators."""

    _evaluators: Dict[EdgeConditionType, ConditionEvaluator] = _default_evaluators()

    @classmethod
    def evaluate(
        cls,
        condition: EdgeCondition,
        task: "Task",
        response: Any,
        routing_signal: Optional["RoutingSignal"] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate a condition using the appropriate evaluator."""
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

    @classmethod
    def reset(cls):
        """Restore default evaluators (useful in tests)."""
        cls._evaluators = _default_evaluators()
