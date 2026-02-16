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

    def select(
        self,
        task_type: TaskType,
        retry_count: int = 0,
        specialization_profile: str = None,
        file_count: int = 0,
    ) -> str:
        """
        Select model based on task type, retry count, and specialization.

        Routing priority:
        1. retry_count >= 3 → premium (task is difficult)
        2. task_type in premium_types → premium (fixed premium tasks)
        3. IMPLEMENTATION + backend/infra + high file count → premium
        4. IMPLEMENTATION + frontend + low file count → cheap
        5. task_type in cheap_types → cheap (fixed cheap tasks)
        6. Default → sonnet

        Args:
            task_type: Type of task being performed
            retry_count: Number of times task has been retried
            specialization_profile: Specialization ID (backend, frontend, infrastructure)
            file_count: Number of files involved in the task
        """
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

        # Priority 3-4: Specialization-aware routing (IMPLEMENTATION tasks only)
        if task_type == TaskType.IMPLEMENTATION and specialization_profile:
            # Backend/infra with high file count → premium
            if specialization_profile in {"backend", "infrastructure"} and file_count >= 8:
                return self.premium_model
            # Frontend with low file count → cheap
            if specialization_profile == "frontend" and file_count <= 5:
                return self.cheap_model

        # Priority 5: Cheap tasks (fixed cheap)
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

        # Priority 6: Default for implementation, architecture, planning, etc.
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
