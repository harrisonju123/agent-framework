"""Model selection logic ported from Bash system."""

from ..core.task import TaskType


class ModelSelector:
    """
    Model selection based on task type and retry count.

    Ported from scripts/async-agent-runner.sh lines 301-337.
    """

    def __init__(
        self,
        cheap_model: str = "claude-3-5-haiku-20241022",
        default_model: str = "claude-sonnet-4-20250514",
        premium_model: str = "claude-opus-4-20250514",
    ):
        self.cheap_model = cheap_model
        self.default_model = default_model
        self.premium_model = premium_model

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

        # Check task type
        if task_type == TaskType.ESCALATION:
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
