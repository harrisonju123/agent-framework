"""Tests for ModelSelector — model selection and timeout tiers."""

import pytest

from agent_framework.core.task import TaskType
from agent_framework.llm.model_selector import ModelSelector


@pytest.fixture
def selector():
    return ModelSelector(
        cheap_model="haiku",
        default_model="sonnet",
        premium_model="opus",
    )


class TestModelSelection:
    """Model selection by task type and retry count."""

    def test_planning_uses_premium(self, selector):
        assert selector.select(TaskType.PLANNING) == "opus"

    def test_architecture_uses_premium(self, selector):
        assert selector.select(TaskType.ARCHITECTURE) == "opus"

    def test_review_uses_premium(self, selector):
        assert selector.select(TaskType.REVIEW) == "opus"

    def test_qa_verification_uses_premium(self, selector):
        assert selector.select(TaskType.QA_VERIFICATION) == "opus"

    def test_escalation_uses_premium(self, selector):
        assert selector.select(TaskType.ESCALATION) == "opus"

    def test_implementation_uses_default(self, selector):
        assert selector.select(TaskType.IMPLEMENTATION) == "sonnet"

    def test_testing_uses_cheap(self, selector):
        assert selector.select(TaskType.TESTING) == "haiku"

    def test_documentation_uses_cheap(self, selector):
        assert selector.select(TaskType.DOCUMENTATION) == "haiku"

    def test_fix_uses_cheap(self, selector):
        assert selector.select(TaskType.FIX) == "haiku"

    def test_high_retry_count_escalates_to_premium(self, selector):
        """Any task type escalates to premium after 3 retries."""
        assert selector.select(TaskType.IMPLEMENTATION, retry_count=3) == "opus"
        assert selector.select(TaskType.TESTING, retry_count=5) == "opus"

    def test_low_retry_count_does_not_escalate(self, selector):
        assert selector.select(TaskType.IMPLEMENTATION, retry_count=2) == "sonnet"


class TestTimeoutSelection:
    """Timeout tier assignments."""

    def test_review_gets_large_timeout(self, selector):
        """REVIEW was moved to large tier to accommodate Opus."""
        assert selector.select_timeout(TaskType.REVIEW) == 3600

    def test_implementation_gets_large_timeout(self, selector):
        assert selector.select_timeout(TaskType.IMPLEMENTATION) == 3600

    def test_planning_gets_large_timeout(self, selector):
        assert selector.select_timeout(TaskType.PLANNING) == 3600

    def test_testing_gets_bounded_timeout(self, selector):
        assert selector.select_timeout(TaskType.TESTING) == 1800

    def test_pr_request_gets_bounded_timeout(self, selector):
        assert selector.select_timeout(TaskType.PR_REQUEST) == 1800

    def test_documentation_gets_simple_timeout(self, selector):
        assert selector.select_timeout(TaskType.DOCUMENTATION) == 900

    def test_coordination_gets_simple_timeout(self, selector):
        assert selector.select_timeout(TaskType.COORDINATION) == 900
