"""Base LLM backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

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


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(
        self,
        request: LLMRequest,
        task_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            request: The LLM request with prompt and configuration.
            task_id: Optional task identifier for logging/streaming output.
        """
        pass

    @abstractmethod
    def select_model(self, task_type: TaskType, retry_count: int) -> str:
        """
        Select appropriate model based on task type and retry count.

        Ported logic from scripts/async-agent-runner.sh:
        - retry_count >= 3 -> opus (premium)
        - task_type in [testing, fix, etc.] -> haiku (cheap)
        - default -> sonnet
        """
        pass
