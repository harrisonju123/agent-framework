"""Base LLM backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..core.task import TaskType


@dataclass
class LLMRequest:
    """Request to LLM backend."""
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # None = use automatic selection
    max_tokens: int = 4096
    temperature: float = 0.7
    task_type: Optional[TaskType] = None  # For automatic model selection
    retry_count: int = 0  # For escalation to stronger model
    context: dict = None  # Task context (github_repo, jira_project, etc.)
    working_dir: Optional[str] = None  # Working directory for subprocess execution
    agents: Optional[dict] = None  # Teammate definitions for Claude Agent Teams (--agents flag)
    specialization_profile: Optional[str] = None  # Specialization ID (backend, frontend, infrastructure)
    file_count: int = 0  # Number of files involved (for complexity-based routing)
    estimated_lines: int = 0  # Estimated implementation lines (for intelligent routing complexity signal)
    budget_remaining_usd: Optional[float] = None  # Remaining USD budget for the task tree (for intelligent routing)
    allowed_tools: Optional[List[str]] = None  # Restrict to these tools via --allowedTools (None = all tools allowed)
    append_system_prompt: Optional[str] = None  # Appended to Claude CLI's system prompt via --append-system-prompt
    env_vars: Optional[dict] = None  # Extra env vars merged into subprocess environment (e.g. venv PATH)


@dataclass
class LLMResponse:
    """Response from LLM backend."""
    content: str
    model_used: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None
    reported_cost_usd: Optional[float] = None  # CLI-reported cost (accounts for prompt caching discounts)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(
        self,
        request: LLMRequest,
        task_id: Optional[str] = None,
        on_tool_activity: Optional[Callable] = None,
        on_session_tool_call: Optional[Callable] = None,
        on_session_tool_result: Optional[Callable] = None,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            request: The LLM request with prompt and configuration.
            task_id: Optional task identifier for logging/streaming output.
            on_tool_activity: Optional callback invoked with (tool_name, summary)
                when a tool_use event is parsed from the stream.
            on_session_tool_call: Optional callback invoked with (tool_name, tool_input, tool_use_id)
                for structured session logging of tool calls.
            on_session_tool_result: Optional callback invoked with
                (tool_name, success, result_size, tool_use_id) for tool result tracking.
        """
        pass

    def cancel(self) -> None:
        """Cancel any in-flight LLM request. Default no-op for backends that
        don't support cancellation."""
        pass

    def get_partial_output(self) -> str:
        """Return accumulated output from the in-flight LLM call."""
        return ""

    @abstractmethod
    def select_model(
        self,
        task_type: TaskType,
        retry_count: int,
        specialization_profile: str = None,
        file_count: int = 0,
        estimated_lines: int = 0,
        budget_remaining_usd: Optional[float] = None,
    ) -> str:
        """
        Select appropriate model based on task type, retry count, and specialization.

        Ported logic from scripts/async-agent-runner.sh with specialization routing:
        - retry_count >= 3 -> opus (premium)
        - task_type in [testing, fix, etc.] -> haiku (cheap)
        - IMPLEMENTATION + backend/infra + high file count -> opus (premium)
        - IMPLEMENTATION + frontend + low file count -> haiku (cheap)
        - default -> sonnet

        When intelligent routing is enabled, delegates to IntelligentRouter
        for multi-signal scoring.
        """
        pass
