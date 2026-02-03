"""LiteLLM backend implementation."""

import time
from typing import Optional

import litellm

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType


class LiteLLMBackend(LLMBackend):
    """LLM backend using LiteLLM for API-based model access."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cheap_model: str = "claude-3-5-haiku-20241022",
        default_model: str = "claude-sonnet-4-20250514",
        premium_model: str = "claude-opus-4-20250514",
    ):
        self.api_key = api_key
        self.model_selector = ModelSelector(cheap_model, default_model, premium_model)

        # Configure litellm
        if api_key:
            litellm.api_key = api_key

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request via LiteLLM."""
        start_time = time.time()

        # Select model
        if request.model:
            model = request.model
        elif request.task_type:
            model = self.select_model(request.task_type, request.retry_count)
        else:
            model = self.model_selector.default_model

        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            # Call litellm
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content,
                model_used=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResponse(
                content="",
                model_used=model,
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def select_model(self, task_type: TaskType, retry_count: int) -> str:
        """Select appropriate model based on task type and retry count."""
        return self.model_selector.select(task_type, retry_count)
