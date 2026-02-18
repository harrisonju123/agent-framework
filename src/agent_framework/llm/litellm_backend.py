"""LiteLLM direct API backend implementation.

Text-only completion using the litellm Python library.
No MCP tools, no agent teams, no file editing — useful for
lightweight tasks or non-Claude models.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType

logger = logging.getLogger(__name__)

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LiteLLMBackend(LLMBackend):
    """LLM backend using litellm Python library for direct API calls.

    Supports any LiteLLM-compatible model provider. Does not support
    MCP tools, agent teams, or streaming tool use — only text completions.
    """

    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        cheap_model: str = "claude-haiku-4-5-20251001",
        default_model: str = "claude-sonnet-4-5-20250929",
        premium_model: str = "claude-sonnet-4-5-20250929",
        logs_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Install it with: "
                "pip install 'agent-framework[litellm]'"
            )

        self.api_key = api_key
        self.api_base = api_base
        self.model_selector = ModelSelector(
            cheap_model, default_model, premium_model,
        )
        self.logs_dir = logs_dir or Path("logs")
        self.timeout = timeout

    async def complete(
        self,
        request: LLMRequest,
        task_id: Optional[str] = None,
        on_tool_activity: Optional[Callable] = None,
        on_session_tool_call: Optional[Callable] = None,
    ) -> LLMResponse:
        """Send a completion request via litellm.acompletion().

        Builds messages from system_prompt + prompt, calls the API,
        and maps the response to LLMResponse.
        """
        start_time = time.time()

        if request.model:
            model = request.model
        elif request.task_type:
            model = self.select_model(
                request.task_type,
                request.retry_count,
                request.specialization_profile,
                request.file_count,
            )
        else:
            model = self.model_selector.default_model

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        log_file = None
        log_file_path = None

        try:
            if task_id:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                log_file_path = self.logs_dir / f"litellm-{task_id}.log"
                log_file = open(log_file_path, "a")
                log_file.write(f"=== LiteLLM Task: {task_id} (attempt {request.retry_count + 1}) ===\n")
                log_file.write(f"Model: {model}\n")
                log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 50 + "\n\n")
                log_file.flush()
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await asyncio.wait_for(
                litellm.acompletion(**kwargs),
                timeout=self.timeout,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            # litellm tracks cost via response._hidden_params
            cost = None
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass

            if log_file:
                log_file.write(content)
                log_file.write(f"\n\n{'=' * 50}\n")
                log_file.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Duration: {latency_ms/1000:.1f}s\n")
                log_file.write(f"Tokens: {input_tokens} in / {output_tokens} out\n")
                if cost is not None:
                    log_file.write(f"Cost: ${cost:.4f}\n")

            return LLMResponse(
                content=content,
                model_used=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                success=True,
                reported_cost_usd=cost,
            )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            if log_file:
                log_file.write(f"\n\nTIMEOUT after {self.timeout} seconds\n")
            return LLMResponse(
                content="",
                model_used=model,
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                latency_ms=latency_ms,
                success=False,
                error=f"LiteLLM call timed out after {self.timeout} seconds",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if log_file:
                log_file.write(f"\n\nERROR: {e}\n")
            logger.error(f"LiteLLM call failed: {e}")
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

        finally:
            if log_file:
                log_file.close()

    def select_model(
        self,
        task_type: TaskType,
        retry_count: int,
        specialization_profile: str = None,
        file_count: int = 0,
    ) -> str:
        """Select appropriate model based on task type, retry count, and specialization."""
        return self.model_selector.select(
            task_type,
            retry_count,
            specialization_profile,
            file_count,
        )
