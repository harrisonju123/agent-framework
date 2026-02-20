"""Model selection logic ported from Bash system."""

import logging
from typing import Optional

from ..core.task import TaskType

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Model selection based on task type and retry count.

    Ported from scripts/async-agent-runner.sh lines 301-337.
    Supports optional delegation to IntelligentRouter for multi-signal scoring.
    """

    def __init__(
        self,
        cheap_model: str = "claude-haiku-4-5-20251001",
        default_model: str = "claude-sonnet-4-5-20250929",
        premium_model: str = "opus",
        timeout_large: int = 3600,
        timeout_bounded: int = 1800,
        timeout_simple: int = 900,
    ):
        self.cheap_model = cheap_model
        self.default_model = default_model
        self.premium_model = premium_model
        self.timeout_large = timeout_large
        self.timeout_bounded = timeout_bounded
        self.timeout_simple = timeout_simple

    # Map tier names to model identifiers for intelligent router output
    @property
    def _tier_to_model(self):
        return {
            "haiku": self.cheap_model,
            "sonnet": self.default_model,
            "opus": self.premium_model,
        }

    def select(
        self,
        task_type: TaskType,
        retry_count: int = 0,
        specialization_profile: str = None,
        file_count: int = 0,
        intelligent_router: Optional[object] = None,
        routing_signals: Optional[object] = None,
    ) -> str:
        """
        Select model based on task type, retry count, and specialization.

        When intelligent_router and routing_signals are provided, delegates to
        the multi-signal scoring engine. Falls back to static logic on any error.

        Routing priority (static fallback):
        1. retry_count >= 3 → premium (task is difficult)
        2. task_type in premium_types → premium (fixed premium tasks)
        3. (IMPLEMENTATION|ENHANCEMENT) + frontend + file_count <= 5 → cheap
        4. (IMPLEMENTATION|ENHANCEMENT) + non-frontend + file_count >= 8 → premium
           (covers auto-generated profiles like "grpc" that aren't explicitly "backend")
        5. task_type in cheap_types → cheap, except FIX-family + specialization +
           file_count >= 8 → sonnet (complex multi-file bug fixes need more than haiku)
        6. Default → sonnet

        Args:
            task_type: Type of task being performed
            retry_count: Number of times task has been retried
            specialization_profile: Specialization ID (backend, frontend, infrastructure, or
                auto-generated IDs like "grpc")
            file_count: Number of files involved in the task
            intelligent_router: Optional IntelligentRouter instance
            routing_signals: Optional RoutingSignals for the intelligent router
        """
        # Intelligent routing delegation (when enabled)
        if intelligent_router is not None and routing_signals is not None:
            try:
                decision = intelligent_router.select(routing_signals)
                model = self._tier_to_model.get(decision.chosen_tier, self.default_model)
                logger.debug(
                    f"Intelligent routing: tier={decision.chosen_tier}, model={model}, "
                    f"scores={decision.scores}"
                )
                # Stash decision for caller to log — avoids coupling router to session logger
                self._last_routing_decision = decision
                return model
            except Exception as e:
                logger.warning(f"Intelligent router failed, falling back to static routing: {e}")

        # Priority 1: Escalate to stronger model if task keeps failing
        if retry_count >= 3:
            return self.premium_model

        # Priority 2: High-stakes tasks where quality matters most (fixed premium)
        premium_types = {
            TaskType.ESCALATION,
            TaskType.PLANNING,
            TaskType.ARCHITECTURE,
            TaskType.REVIEW,
            TaskType.QA_VERIFICATION,
        }
        if task_type in premium_types:
            return self.premium_model

        # Priority 3-4: Specialization-aware routing for implementation-like tasks.
        # ENHANCEMENT is grouped with IMPLEMENTATION — both produce code that benefits
        # from the same specialization signals.
        impl_types = {TaskType.IMPLEMENTATION, TaskType.ENHANCEMENT}
        if task_type in impl_types and specialization_profile:
            # Frontend with low file count → cheap (UI changes tend to be contained)
            if specialization_profile == "frontend" and file_count <= 5:
                return self.cheap_model
            # Any non-frontend profile (including auto-generated ones like "grpc") with
            # high file count → premium.  We use != "frontend" rather than an allowlist
            # so that auto-generated profiles aren't silently dropped.
            if specialization_profile != "frontend" and file_count >= 8:
                return self.premium_model
            # 6–7 files with any profile: no override — falls through to default (sonnet).
            # This is intentional dead space between the cheap and premium thresholds.

        # Priority 5: Cheap tasks. FIX-family is cheap by default, but a specialized
        # fix touching >=8 files likely spans subsystems and benefits from sonnet.
        fix_types = {TaskType.FIX, TaskType.BUGFIX, TaskType.BUG_FIX}
        cheap_types = {
            TaskType.TESTING,
            TaskType.VERIFICATION,
            TaskType.COORDINATION,
            TaskType.STATUS_REPORT,
            TaskType.DOCUMENTATION,
        } | fix_types
        if task_type in cheap_types:
            if task_type in fix_types and specialization_profile and file_count >= 8:
                return self.default_model
            return self.cheap_model

        # Priority 7: Default for implementation, architecture, planning, etc.
        return self.default_model

    def select_timeout(self, task_type: TaskType) -> int:
        """
        Select timeout based on task type scope.

        Timeout tiers:
        - Large (1hr): IMPLEMENTATION, ARCHITECTURE, ANALYSIS, PLANNING, ESCALATION, ENHANCEMENT, REVIEW
        - Bounded (30min): TESTING, VERIFICATION, FIX, BUGFIX, PR_REQUEST, PREVIEW
        - Simple (15min): DOCUMENTATION, COORDINATION, STATUS_REPORT
        """
        simple_types = {
            TaskType.DOCUMENTATION,
            TaskType.COORDINATION,
            TaskType.STATUS_REPORT,
        }
        bounded_types = {
            TaskType.TESTING,
            TaskType.VERIFICATION,
            TaskType.QA_VERIFICATION,
            TaskType.FIX,
            TaskType.BUGFIX,
            TaskType.BUG_FIX,
            TaskType.PR_REQUEST,
            TaskType.PREVIEW,
        }

        if task_type in simple_types:
            return self.timeout_simple
        if task_type in bounded_types:
            return self.timeout_bounded
        return self.timeout_large
