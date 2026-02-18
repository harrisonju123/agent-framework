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
    allowed_tools: Optional[List[str]] = None  # Restrict to these tools via --allowedTools (None = all tools allowed)


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
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            request: The LLM request with prompt and configuration.
            task_id: Optional task identifier for logging/streaming output.
            on_tool_activity: Optional callback invoked with (tool_name, summary)
                when a tool_use event is parsed from the stream.
            on_session_tool_call: Optional callback invoked with (tool_name, tool_input)
                for structured session logging of tool calls.
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
    ) -> str:
        """
        Select appropriate model based on task type, retry count, and specialization.

        Ported logic from scripts/async-agent-runner.sh with specialization routing:
        - retry_count >= 3 -> opus (premium)
        - task_type in [testing, fix, etc.] -> haiku (cheap)
        - IMPLEMENTATION + backend/infra + high file count -> opus (premium)
        - IMPLEMENTATION + frontend + low file count -> haiku (cheap)
        - default -> sonnet
        """
        pass
