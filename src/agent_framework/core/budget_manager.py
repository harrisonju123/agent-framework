"""Budget and cost tracking manager."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from ..llm.base import LLMBackend, LLMResponse
    from ..utils.rich_logging import ContextLogger
    from .session_logger import SessionLogger
    from .activity import ActivityManager
    from .task import PlanDocument

from .task import Task, TaskType
from .activity import ActivityEvent


# Model pricing (per 1M tokens, as of 2025-01)
MODEL_PRICING = {
    "haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
}

# Budget warning threshold (30% over budget)
BUDGET_WARNING_THRESHOLD = 1.3


class BudgetManager:
    """Manages token budgets, cost estimation, and completion metrics."""

    def __init__(
        self,
        agent_id: str,
        optimization_config: dict,
        logger: "ContextLogger",
        session_logger: "SessionLogger",
        llm: "LLMBackend",
        workspace: Path,
        activity_manager: "ActivityManager",
        model_success_store: object = None,
    ):
        self.agent_id = agent_id
        self.optimization_config = optimization_config
        self.logger = logger
        self.session_logger = session_logger
        self.llm = llm
        self.workspace = workspace
        self.activity_manager = activity_manager
        self._model_success_store = model_success_store

    def get_token_budget(self, task_type: TaskType) -> int:
        """
        Get expected token budget for task type.

        Implements Strategy 6 (Token Tracking) from the optimization plan.

        Budgets can be configured via optimization.token_budgets in config,
        otherwise uses sensible defaults.
        """
        # Default budgets
        default_budgets = {
            "planning": 30000,
            "implementation": 50000,
            "testing": 20000,
            "escalation": 80000,
            "review": 25000,
            "architecture": 40000,
            "coordination": 15000,
            "documentation": 15000,
            "fix": 30000,
            "bugfix": 30000,
            "bug-fix": 30000,
            "verification": 20000,
            "status_report": 10000,
            "enhancement": 40000,
        }

        # Allow runtime override via optimization config
        configured_budgets = self.optimization_config.get("token_budgets", {})

        # Handle both TaskType enum and string
        if isinstance(task_type, str):
            task_type_str = task_type.lower()
        else:
            task_type_str = task_type.value.lower()

        # Normalize hyphens to underscores for backward compatibility
        # (e.g., "bug-fix" → "bug_fix" to match config keys)
        task_type_str = task_type_str.replace("-", "_")

        return configured_budgets.get(task_type_str, default_budgets.get(task_type_str, 40000))

    def estimate_cost(self, response: "LLMResponse") -> float:
        """
        Estimate cost based on model and token usage.

        Prefers CLI-reported cost when available (accounts for prompt caching
        discounts). Falls back to static MODEL_PRICING calculation for other
        backends or when reported cost is unavailable.
        """
        if response.reported_cost_usd is not None:
            return response.reported_cost_usd

        model_name_lower = response.model_used.lower()

        # Detect model family
        if "haiku" in model_name_lower:
            model_type = "haiku"
        elif "opus" in model_name_lower:
            model_type = "opus"
        elif "sonnet" in model_name_lower:
            model_type = "sonnet"
        else:
            # Unknown model, assume sonnet pricing as conservative estimate
            self.logger.warning(
                f"Unknown model '{response.model_used}', assuming Sonnet pricing for cost estimate"
            )
            model_type = "sonnet"

        prices = MODEL_PRICING.get(model_type, MODEL_PRICING["sonnet"])
        cost = (
            response.input_tokens / 1_000_000 * prices["input"] +
            response.output_tokens / 1_000_000 * prices["output"]
        )
        return cost

    def derive_effort_from_plan(self, plan: Optional["PlanDocument"]) -> str:
        """Derive t-shirt size from plan when architect didn't set estimated_effort."""
        if not plan or (not plan.files_to_modify and not plan.approach):
            return "M"
        from .task_decomposer import estimate_plan_lines
        estimated_lines = estimate_plan_lines(plan)
        if estimated_lines < 150:
            return "XS"
        elif estimated_lines < 350:
            return "S"
        elif estimated_lines < 600:
            return "M"
        elif estimated_lines < 1000:
            return "L"
        return "XL"

    def get_effort_ceiling(self, effort_key: str) -> Optional[float]:
        """Look up USD ceiling for effort size. Returns None if not configured."""
        ceilings = self.optimization_config.get("effort_budget_ceilings", {})
        return ceilings.get(effort_key)

    def log_task_completion_metrics(
        self, task: Task, response: "LLMResponse", task_start_time: datetime,
        *, tool_call_count: Optional[int] = None,
    ) -> None:
        """Log token usage, cost, and completion events."""
        total_tokens = response.input_tokens + response.output_tokens
        budget = self.get_token_budget(task.type)
        cost = self.estimate_cost(response)

        duration = (datetime.now(timezone.utc) - task_start_time).total_seconds()
        self.logger.token_usage(response.input_tokens, response.output_tokens, cost)
        self.logger.task_completed(duration, tokens_used=total_tokens)

        # Budget warning
        if self.optimization_config.get("enable_token_budget_warnings", False):
            threshold = self.optimization_config.get("budget_warning_threshold", BUDGET_WARNING_THRESHOLD)
            if total_tokens > budget * threshold:
                self.logger.warning(
                    f"Task {task.id} EXCEEDED TOKEN BUDGET: "
                    f"{total_tokens} tokens (budget: {budget}, "
                    f"{int(threshold * 100)}% threshold: {budget * threshold:.0f})"
                )
                self.activity_manager.append_event(ActivityEvent(
                    type="token_budget_exceeded",
                    agent=self.agent_id,
                    task_id=task.id,
                    title=f"Token budget exceeded: {total_tokens} > {budget}",
                    timestamp=datetime.now(timezone.utc)
                ))

        # Append complete event
        duration_ms = int((datetime.now(timezone.utc) - task_start_time).total_seconds() * 1000)
        pr_url = task.context.get("pr_url")
        self.activity_manager.append_event(ActivityEvent(
            type="complete",
            agent=self.agent_id,
            task_id=task.id,
            title=task.title,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            pr_url=pr_url,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost,
            tool_call_count=tool_call_count,
        ))

        self.session_logger.log(
            "task_complete",
            status="completed",
            duration_ms=duration_ms,
        )

        # Record success outcome for intelligent routing
        if self._model_success_store is not None and response.model_used:
            repo_slug = task.context.get("github_repo", "")
            task_type_str = task.type if isinstance(task.type, str) else task.type.value
            self._model_success_store.record_outcome(
                repo_slug=repo_slug,
                model_tier=response.model_used,
                task_type=task_type_str,
                success=True,
                cost=cost,
            )
