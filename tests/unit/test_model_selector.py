"""Tests for ModelSelector — model selection and timeout tiers."""

import pytest
from unittest.mock import MagicMock, patch

from agent_framework.core.task import TaskType
from agent_framework.llm.model_selector import ModelSelector
from agent_framework.llm.intelligent_router import (
    IntelligentRouter,
    RoutingDecision,
    RoutingSignals,
)
from agent_framework.llm.model_success_store import ModelSuccessStore


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


class TestIntelligentRouterDelegation:
    """Intelligent router integration with ModelSelector."""

    def test_delegates_to_intelligent_router(self, selector, tmp_path):
        """When router and signals are provided, selector delegates to the router."""
        store = ModelSuccessStore(tmp_path, enabled=True)
        router = IntelligentRouter(store)
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=800,
            file_count=10,
        )
        model = selector.select(
            TaskType.IMPLEMENTATION,
            intelligent_router=router,
            routing_signals=signals,
        )
        # High complexity → opus tier → maps to "opus" model
        assert model == "opus"

    def test_stashes_routing_decision(self, selector, tmp_path):
        """Selector stashes the routing decision for caller to inspect."""
        store = ModelSuccessStore(tmp_path, enabled=True)
        router = IntelligentRouter(store)
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=50,
            file_count=1,
        )
        selector.select(
            TaskType.IMPLEMENTATION,
            intelligent_router=router,
            routing_signals=signals,
        )
        decision = selector._last_routing_decision
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_tier in ("haiku", "sonnet", "opus")

    def test_fallback_on_router_error(self, selector):
        """If the router raises, selector falls back to static logic."""
        mock_router = MagicMock()
        mock_router.select.side_effect = RuntimeError("boom")
        signals = RoutingSignals(task_type="implementation")

        model = selector.select(
            TaskType.IMPLEMENTATION,
            intelligent_router=mock_router,
            routing_signals=signals,
        )
        # Static logic: IMPLEMENTATION → sonnet
        assert model == "sonnet"

    def test_static_routing_when_no_router(self, selector):
        """Without a router, static logic applies."""
        model = selector.select(TaskType.IMPLEMENTATION)
        assert model == "sonnet"

    def test_static_routing_when_no_signals(self, selector, tmp_path):
        """Router provided but no signals → static fallback."""
        store = ModelSuccessStore(tmp_path, enabled=True)
        router = IntelligentRouter(store)
        model = selector.select(
            TaskType.IMPLEMENTATION,
            intelligent_router=router,
        )
        assert model == "sonnet"

    def test_tier_to_model_mapping(self, selector, tmp_path):
        """Router tier names map to selector's configured model identifiers."""
        store = ModelSuccessStore(tmp_path, enabled=True)
        router = IntelligentRouter(store)

        # Low complexity → haiku
        signals = RoutingSignals(task_type="implementation", estimated_lines=30, file_count=1)
        model = selector.select(TaskType.IMPLEMENTATION, intelligent_router=router, routing_signals=signals)
        assert model == "haiku"

        # Medium complexity → sonnet
        signals = RoutingSignals(task_type="implementation", estimated_lines=300, file_count=5)
        model = selector.select(TaskType.IMPLEMENTATION, intelligent_router=router, routing_signals=signals)
        assert model == "sonnet"

        # High complexity → opus
        signals = RoutingSignals(task_type="implementation", estimated_lines=800, file_count=10)
        model = selector.select(TaskType.IMPLEMENTATION, intelligent_router=router, routing_signals=signals)
        assert model == "opus"
