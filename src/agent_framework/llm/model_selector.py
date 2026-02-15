"""Model selection logic ported from Bash system."""

from ..core.task import TaskType


class ModelSelector:
    """
    Model selection based on task type and retry count.

    Ported from scripts/async-agent-runner.sh lines 301-337.
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

    def select(self, task_type: TaskType, retry_count: int = 0) -> str:
        """
        Select model based on task type and retry count.

        Logic:
        1. If retry_count >= 3, use premium model (task is difficult)
        2. If task_type is cheap (testing, fix, docs), use cheap model
        3. If task_type is escalation, use premium model
        4. Otherwise use default model
        """
        # Escalate to stronger model if task keeps failing
        if retry_count >= 3:
            return self.premium_model

        # High-stakes tasks where quality matters most
        premium_types = {
            TaskType.ESCALATION,
            TaskType.PLANNING,
            TaskType.ARCHITECTURE,
            TaskType.REVIEW,
            TaskType.QA_VERIFICATION,
        }
        if task_type in premium_types:
            return self.premium_model

        # Cheap tasks
        cheap_types = {
            TaskType.TESTING,
            TaskType.VERIFICATION,
            TaskType.FIX,
            TaskType.BUGFIX,
            TaskType.BUG_FIX,
            TaskType.COORDINATION,
            TaskType.STATUS_REPORT,
            TaskType.DOCUMENTATION,
        }
        if task_type in cheap_types:
            return self.cheap_model

        # Default for implementation, architecture, planning, etc.
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
