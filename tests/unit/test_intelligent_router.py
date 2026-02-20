"""Tests for IntelligentRouter and ModelSuccessStore."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agent_framework.llm.intelligent_router import (
    IntelligentRouter,
    RoutingDecision,
    RoutingSignals,
)
from agent_framework.llm.model_success_store import ModelSuccessStore, _normalize_model_tier


@pytest.fixture
def tmp_workspace(tmp_path):
    return tmp_path


@pytest.fixture
def success_store(tmp_workspace):
    return ModelSuccessStore(tmp_workspace, enabled=True)


@pytest.fixture
def router(success_store):
    return IntelligentRouter(success_store)


class TestComplexityScoring:
    """Complexity signal drives tier affinity."""

    def test_high_complexity_prefers_opus(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=800,
            file_count=10,
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "opus"

    def test_low_complexity_prefers_haiku(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=50,
            file_count=2,
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "haiku"

    def test_medium_complexity_prefers_sonnet(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "sonnet"


class TestHistoricalScoring:
    """Historical success rates influence tier selection."""

    def test_good_success_rate_boosts_tier(self, tmp_workspace):
        store = ModelSuccessStore(tmp_workspace, enabled=True)
        # Record enough samples for haiku with high success rate
        for _ in range(10):
            store.record_outcome("org/repo", "haiku", "implementation", True, 0.05)
        # Record poor success rate for sonnet
        for _ in range(10):
            store.record_outcome("org/repo", "sonnet", "implementation", False, 0.30)

        router = IntelligentRouter(store)
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
            repo_slug="org/repo",
        )
        decision = router.select(signals)
        # Haiku has better historical score despite medium complexity favoring sonnet
        assert decision.scores.get("haiku", 0) > decision.scores.get("sonnet", 0)

    def test_insufficient_samples_ignored(self, tmp_workspace):
        store = ModelSuccessStore(tmp_workspace, enabled=True)
        # Only 2 samples — below min_historical_samples (5)
        store.record_outcome("org/repo", "haiku", "implementation", True, 0.05)
        store.record_outcome("org/repo", "haiku", "implementation", True, 0.05)

        router = IntelligentRouter(store)
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
            repo_slug="org/repo",
        )
        decision = router.select(signals)
        # Should still pick sonnet (medium complexity) since historical data is insufficient
        assert decision.chosen_tier == "sonnet"


class TestBudgetConstraint:
    """Budget signals constrain tier selection."""

    def test_low_budget_forces_cheaper(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=800,
            file_count=10,
            budget_remaining_usd=0.10,
            budget_ceiling_usd=5.0,
        )
        decision = router.select(signals)
        # Opus costs ~$1.50, only haiku ($0.05) is affordable
        assert decision.chosen_tier == "haiku"

    def test_no_budget_info_neutral(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=800,
            file_count=10,
        )
        decision = router.select(signals)
        # Without budget info, complexity drives the decision
        assert decision.chosen_tier == "opus"

    def test_zero_budget_prefers_cheapest(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
            budget_remaining_usd=0.00,
            budget_ceiling_usd=5.0,
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "haiku"


class TestRetryPenalty:
    """Retry history penalizes failed tiers."""

    def test_failed_model_skipped(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
            retry_history=[("claude-sonnet-4-20250514", None)],
        )
        decision = router.select(signals)
        # Sonnet penalized, should pick something else
        assert decision.chosen_tier != "sonnet"

    def test_multiple_failures_narrow_choices(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
            retry_history=[
                ("claude-sonnet-4-20250514", None),
                ("claude-haiku-3-5-20241022", None),
            ],
        )
        decision = router.select(signals)
        # Both sonnet and haiku penalized — opus is the remaining choice
        assert decision.chosen_tier == "opus"

    def test_no_retry_history_neutral(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=300,
            file_count=5,
        )
        decision = router.select(signals)
        # Medium complexity, no retry penalty — sonnet preferred
        assert decision.chosen_tier == "sonnet"


class TestSpecializationScoring:
    """Specialization profiles influence tier affinity."""

    def test_frontend_low_files_prefers_haiku(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=100,
            file_count=3,
            specialization_profile="frontend",
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "haiku"

    def test_backend_high_files_prefers_opus(self, router):
        signals = RoutingSignals(
            task_type="implementation",
            estimated_lines=500,
            file_count=12,
            specialization_profile="backend",
        )
        decision = router.select(signals)
        assert decision.chosen_tier == "opus"


class TestFallbackOnError:
    """Router returns well-formed decisions in all cases."""

    def test_scores_for_all_tiers(self, router):
        signals = RoutingSignals(task_type="implementation")
        decision = router.select(signals)
        assert "haiku" in decision.scores
        assert "sonnet" in decision.scores
        assert "opus" in decision.scores

    def test_returns_routing_decision(self, router):
        signals = RoutingSignals(task_type="implementation")
        decision = router.select(signals)
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_tier in ("haiku", "sonnet", "opus")


class TestModelSuccessStore:
    """JSONL-backed model success tracking."""

    def test_record_and_retrieve(self, success_store):
        success_store.record_outcome("org/repo", "sonnet", "implementation", True, 0.30)
        success_store.record_outcome("org/repo", "sonnet", "implementation", False, 0.30)

        rate = success_store.get_success_rate("org/repo", "sonnet", "implementation")
        assert rate == 0.5
        assert success_store.get_sample_count("org/repo", "sonnet", "implementation") == 2

    def test_missing_key_returns_none(self, success_store):
        rate = success_store.get_success_rate("org/repo", "opus", "testing")
        assert rate is None
        assert success_store.get_sample_count("org/repo", "opus", "testing") == 0

    def test_disabled_store_is_noop(self, tmp_workspace):
        store = ModelSuccessStore(tmp_workspace, enabled=False)
        store.record_outcome("org/repo", "sonnet", "implementation", True, 0.30)
        assert store.get_success_rate("org/repo", "sonnet", "implementation") is None
        assert store.get_sample_count("org/repo", "sonnet", "implementation") == 0

    def test_persistence_across_instances(self, tmp_workspace):
        store1 = ModelSuccessStore(tmp_workspace, enabled=True)
        store1.record_outcome("org/repo", "haiku", "testing", True, 0.05)
        store1.record_outcome("org/repo", "haiku", "testing", True, 0.05)
        store1.record_outcome("org/repo", "haiku", "testing", False, 0.05)

        # New instance should load from JSONL
        store2 = ModelSuccessStore(tmp_workspace, enabled=True)
        assert store2.get_sample_count("org/repo", "haiku", "testing") == 3
        rate = store2.get_success_rate("org/repo", "haiku", "testing")
        assert abs(rate - 2 / 3) < 0.01

    def test_model_tier_normalization(self):
        assert _normalize_model_tier("claude-3-5-haiku-20241022") == "haiku"
        assert _normalize_model_tier("claude-sonnet-4-20250514") == "sonnet"
        assert _normalize_model_tier("claude-opus-4-20250514") == "opus"
        assert _normalize_model_tier("unknown-model") == "unknown-model"

    def test_jsonl_file_written(self, tmp_workspace):
        store = ModelSuccessStore(tmp_workspace, enabled=True)
        store.record_outcome("org/repo", "sonnet", "implementation", True, 0.30)

        jsonl_path = tmp_workspace / ".agent-communication" / "metrics" / "model_success.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["repo"] == "org/repo"
        assert record["model"] == "sonnet"
        assert record["task_type"] == "implementation"
        assert record["success"] is True
        assert record["cost"] == 0.3
