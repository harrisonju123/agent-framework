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


def test_preview_uses_default_model(selector):
    """PREVIEW tasks use sonnet (default model), not premium or cheap."""
    model = selector.select(TaskType.PREVIEW)
    assert model == selector.default_model


def test_preview_uses_bounded_timeout(selector):
    """PREVIEW tasks get bounded timeout (30min) since they're planning-only."""
    timeout = selector.select_timeout(TaskType.PREVIEW)
    assert timeout == selector.timeout_bounded


class TestSpecializationRouting:
    """Specialization-aware model routing for IMPLEMENTATION tasks."""

    def test_backend_high_file_count_uses_premium(self, selector):
        """Backend with >=8 files routes to premium model."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="backend",
            file_count=8,
        )
        assert model == "opus"

    def test_backend_low_file_count_uses_default(self, selector):
        """Backend with <8 files uses default model."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="backend",
            file_count=7,
        )
        assert model == "sonnet"

    def test_infrastructure_high_file_count_uses_premium(self, selector):
        """Infrastructure with >=8 files routes to premium model."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="infrastructure",
            file_count=10,
        )
        assert model == "opus"

    def test_frontend_low_file_count_uses_cheap(self, selector):
        """Frontend with <=5 files routes to cheap model."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="frontend",
            file_count=3,
        )
        assert model == "haiku"

    def test_frontend_high_file_count_uses_default(self, selector):
        """Frontend with >5 files uses default model."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="frontend",
            file_count=8,
        )
        assert model == "sonnet"

    def test_unknown_specialization_uses_premium_with_high_file_count(self, selector):
        """Auto-generated profiles (non-frontend) with >=8 files route to premium."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile="unknown-spec",
            file_count=10,
        )
        assert model == "opus"

    def test_no_specialization_uses_default(self, selector):
        """Implementation without specialization uses default."""
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=0,
            specialization_profile=None,
            file_count=10,
        )
        assert model == "sonnet"

    def test_retry_escalation_overrides_specialization(self, selector):
        """Retry count >= 3 takes priority over specialization routing."""
        # Frontend with low file count would normally be cheap, but retry wins
        model = selector.select(
            TaskType.IMPLEMENTATION,
            retry_count=3,
            specialization_profile="frontend",
            file_count=3,
        )
        assert model == "opus"

    def test_specialization_only_affects_implementation(self, selector):
        """Specialization routing only applies to IMPLEMENTATION tasks."""
        # Planning is always premium regardless of specialization
        model = selector.select(
            TaskType.PLANNING,
            retry_count=0,
            specialization_profile="frontend",
            file_count=3,
        )
        assert model == "opus"

        # Testing is always cheap regardless of specialization
        model = selector.select(
            TaskType.TESTING,
            retry_count=0,
            specialization_profile="backend",
            file_count=10,
        )
        assert model == "haiku"

    def test_enhancement_backend_high_file_count_uses_premium(self, selector):
        """ENHANCEMENT behaves like IMPLEMENTATION — backend + >=8 files → premium."""
        model = selector.select(
            TaskType.ENHANCEMENT,
            retry_count=0,
            specialization_profile="backend",
            file_count=8,
        )
        assert model == "opus"

    def test_enhancement_frontend_low_file_count_uses_cheap(self, selector):
        """ENHANCEMENT behaves like IMPLEMENTATION — frontend + <=5 files → cheap."""
        model = selector.select(
            TaskType.ENHANCEMENT,
            retry_count=0,
            specialization_profile="frontend",
            file_count=3,
        )
        assert model == "haiku"

    def test_fix_with_specialization_high_file_count_uses_default(self, selector):
        """FIX tasks with a specialization and >=8 files escalate from cheap to sonnet."""
        model = selector.select(
            TaskType.FIX,
            retry_count=0,
            specialization_profile="backend",
            file_count=8,
        )
        assert model == "sonnet"

    def test_fix_with_specialization_low_file_count_stays_cheap(self, selector):
        """FIX tasks with specialization but <8 files keep the cheap model."""
        model = selector.select(
            TaskType.FIX,
            retry_count=0,
            specialization_profile="backend",
            file_count=7,
        )
        assert model == "haiku"

    def test_fix_without_specialization_stays_cheap(self, selector):
        """FIX tasks without a specialization always use the cheap model."""
        model = selector.select(
            TaskType.FIX,
            retry_count=0,
            specialization_profile=None,
            file_count=20,
        )
        assert model == "haiku"
