"""Intelligent model routing based on multi-signal scoring.

Scores each model tier (haiku, sonnet, opus) using weighted signals:
- Complexity: estimated lines + file count
- Historical success: per-repo success rate from ModelSuccessStore
- Specialization: profile-based routing preference
- Budget: remaining budget vs tier cost
- Retry penalty: penalize models that already failed this task
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .model_success_store import ModelSuccessStore

logger = logging.getLogger(__name__)

# Approximate per-task cost by tier (USD) — used for budget scoring
_TIER_COST_ESTIMATE: Dict[str, float] = {
    "haiku": 0.05,
    "sonnet": 0.30,
    "opus": 1.50,
}

# Complexity thresholds for tier affinity
_LOW_COMPLEXITY = 200   # lines — haiku territory
_HIGH_COMPLEXITY = 600  # lines — opus territory

# Model tiers in ascending capability order
_TIERS = ["haiku", "sonnet", "opus"]


@dataclass
class RoutingSignals:
    """All signals available for model routing decisions."""
    task_type: str
    retry_count: int = 0
    specialization_profile: Optional[str] = None
    file_count: int = 0
    estimated_lines: int = 0
    budget_remaining_usd: Optional[float] = None
    budget_ceiling_usd: Optional[float] = None
    repo_slug: str = ""
    retry_history: List[Tuple[str, Optional[str]]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Result of the routing scoring process."""
    chosen_tier: str
    scores: Dict[str, float]
    signals: Dict[str, object]
    fallback: bool = False


class IntelligentRouter:
    """Multi-signal scoring engine for model tier selection.

    Weights are configurable — defaults match the architect's spec:
      complexity=0.3, historical=0.25, specialization=0.2, budget=0.15, retry=0.1
    """

    def __init__(
        self,
        success_store: ModelSuccessStore,
        complexity_weight: float = 0.3,
        historical_weight: float = 0.25,
        specialization_weight: float = 0.2,
        budget_weight: float = 0.15,
        retry_weight: float = 0.1,
        min_historical_samples: int = 5,
    ):
        self._store = success_store
        self._weights = {
            "complexity": complexity_weight,
            "historical": historical_weight,
            "specialization": specialization_weight,
            "budget": budget_weight,
            "retry": retry_weight,
        }
        self._min_samples = min_historical_samples

    def select(self, signals: RoutingSignals) -> RoutingDecision:
        """Score all tiers and return the best one, constrained by budget ceiling."""
        scores: Dict[str, float] = {}

        for tier in _TIERS:
            scores[tier] = self._score_tier(tier, signals)

        # Budget ceiling constraint: exclude tiers whose estimated cost exceeds remaining budget.
        # Always keep the cheapest tier as a last resort when nothing is strictly affordable.
        if signals.budget_ceiling_usd is not None and signals.budget_remaining_usd is not None:
            affordable = {
                t: s for t, s in scores.items()
                if _TIER_COST_ESTIMATE.get(t, 0) <= signals.budget_remaining_usd
            }
            if affordable:
                scores = affordable
            else:
                # Nothing affordable — constrain to cheapest tier only
                cheapest = min(_TIERS, key=lambda t: _TIER_COST_ESTIMATE.get(t, 0))
                scores = {cheapest: scores.get(cheapest, 0.0)}

        chosen = max(scores, key=scores.get)

        return RoutingDecision(
            chosen_tier=chosen,
            scores={t: round(s, 4) for t, s in scores.items()},
            signals=_signals_to_dict(signals),
            fallback=False,
        )

    def _score_tier(self, tier: str, signals: RoutingSignals) -> float:
        """Compute weighted score for a single tier."""
        complexity = self._complexity_score(tier, signals)
        historical = self._historical_score(tier, signals)
        specialization = self._specialization_score(tier, signals)
        budget = self._budget_score(tier, signals)
        retry = self._retry_penalty(tier, signals)

        # When historical data is insufficient, redistribute its weight
        weights = dict(self._weights)
        if historical is None:
            redistribute = weights["historical"]
            weights["historical"] = 0.0
            # Spread evenly across other signals
            other_keys = [k for k in weights if k != "historical" and weights[k] > 0]
            if other_keys:
                bonus = redistribute / len(other_keys)
                for k in other_keys:
                    weights[k] += bonus
            historical = 0.0

        return (
            weights["complexity"] * complexity
            + weights["historical"] * historical
            + weights["specialization"] * specialization
            + weights["budget"] * budget
            + weights["retry"] * retry
        )

    def _complexity_score(self, tier: str, signals: RoutingSignals) -> float:
        """Higher-capability tiers score better for complex tasks.

        Returns 0-1 where 1 means this tier is the best fit for the complexity level.
        """
        total_complexity = signals.estimated_lines + (signals.file_count * 30)

        if total_complexity <= _LOW_COMPLEXITY:
            # Low complexity: haiku preferred
            return {"haiku": 1.0, "sonnet": 0.6, "opus": 0.3}.get(tier, 0.5)
        elif total_complexity >= _HIGH_COMPLEXITY:
            # High complexity: opus preferred
            return {"haiku": 0.2, "sonnet": 0.6, "opus": 1.0}.get(tier, 0.5)
        else:
            # Medium complexity: sonnet preferred
            return {"haiku": 0.4, "sonnet": 1.0, "opus": 0.7}.get(tier, 0.5)

    def _historical_score(self, tier: str, signals: RoutingSignals) -> Optional[float]:
        """Return success rate if sufficient samples exist, else None."""
        if not signals.repo_slug or not self._store.enabled:
            return None

        sample_count = self._store.get_sample_count(
            signals.repo_slug, tier, signals.task_type,
        )
        if sample_count < self._min_samples:
            return None

        rate = self._store.get_success_rate(
            signals.repo_slug, tier, signals.task_type,
        )
        return rate if rate is not None else None

    def _specialization_score(self, tier: str, signals: RoutingSignals) -> float:
        """Score based on specialization profile matching.

        Frontend + low file count favors cheap; backend/infra + high count favors premium.
        """
        profile = signals.specialization_profile
        if not profile:
            return 0.5  # No signal — neutral

        if profile == "frontend" and signals.file_count <= 5:
            return {"haiku": 1.0, "sonnet": 0.5, "opus": 0.3}.get(tier, 0.5)

        if profile != "frontend" and signals.file_count >= 8:
            return {"haiku": 0.2, "sonnet": 0.6, "opus": 1.0}.get(tier, 0.5)

        # Default: slight preference for sonnet
        return {"haiku": 0.5, "sonnet": 0.8, "opus": 0.6}.get(tier, 0.5)

    def _budget_score(self, tier: str, signals: RoutingSignals) -> float:
        """Cheaper tiers score better when budget is tight."""
        if signals.budget_remaining_usd is None:
            return 0.5  # No budget info — neutral

        tier_cost = _TIER_COST_ESTIMATE.get(tier, 0.3)
        remaining = signals.budget_remaining_usd

        if remaining <= 0:
            # Out of budget: strongly prefer cheapest
            return {"haiku": 1.0, "sonnet": 0.3, "opus": 0.1}.get(tier, 0.5)

        # Ratio of remaining budget to tier cost — higher = more affordable
        ratio = remaining / tier_cost if tier_cost > 0 else 10.0
        if ratio >= 10:
            return 0.8  # Plenty of budget — all tiers viable
        elif ratio >= 3:
            return 0.7
        elif ratio >= 1:
            return 0.5
        else:
            # Can't afford this tier
            return 0.1

    def _retry_penalty(self, tier: str, signals: RoutingSignals) -> float:
        """Penalize tiers that already failed on this task.

        Returns 0-1 where 1 means no penalty (tier hasn't been tried).
        """
        if not signals.retry_history:
            return 0.5  # No retry info — neutral

        from .model_success_store import _normalize_model_tier

        failed_tiers = set()
        for model, _error_type in signals.retry_history:
            failed_tiers.add(_normalize_model_tier(model))

        if tier in failed_tiers:
            return 0.0  # Already failed — heavy penalty
        return 1.0  # Hasn't been tried — bonus


def _signals_to_dict(signals: RoutingSignals) -> dict:
    """Convert RoutingSignals to a JSON-serializable dict for logging."""
    return {
        "task_type": signals.task_type,
        "retry_count": signals.retry_count,
        "specialization_profile": signals.specialization_profile,
        "file_count": signals.file_count,
        "estimated_lines": signals.estimated_lines,
        "budget_remaining_usd": signals.budget_remaining_usd,
        "budget_ceiling_usd": signals.budget_ceiling_usd,
        "repo_slug": signals.repo_slug,
        "retry_history": [
            {"model": m, "error_type": e} for m, e in signals.retry_history
        ],
    }
