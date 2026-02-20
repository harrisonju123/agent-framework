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
        intelligent_routing_config: Optional[dict] = None,
        model_success_store: Optional[object] = None,
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

        # Intelligent routing (optional)
        self._intelligent_router = None
        self._model_success_store = model_success_store
        ir_cfg = intelligent_routing_config or {}
        if ir_cfg.get("enabled") and model_success_store is not None:
            from .intelligent_router import IntelligentRouter
            self._intelligent_router = IntelligentRouter(
                success_store=model_success_store,
                complexity_weight=ir_cfg.get("complexity_weight", 0.3),
                historical_weight=ir_cfg.get("historical_weight", 0.25),
                specialization_weight=ir_cfg.get("specialization_weight", 0.2),
                budget_weight=ir_cfg.get("budget_weight", 0.15),
                retry_weight=ir_cfg.get("retry_weight", 0.1),
                min_historical_samples=ir_cfg.get("min_historical_samples", 5),
            )

    async def complete(
        self,
        request: LLMRequest,
        task_id: Optional[str] = None,
        on_tool_activity: Optional[Callable] = None,
        on_session_tool_call: Optional[Callable] = None,
        on_session_tool_result: Optional[Callable] = None,
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
                request.estimated_lines,
                request.budget_remaining_usd,
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
        estimated_lines: int = 0,
        budget_remaining_usd: Optional[float] = None,
    ) -> str:
        """Select appropriate model based on task type, retry count, and specialization."""
        router = self._intelligent_router
        routing_signals = None

        if router is not None:
            from .intelligent_router import RoutingSignals
            task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
            routing_signals = RoutingSignals(
                task_type=task_type_str,
                retry_count=retry_count,
                specialization_profile=specialization_profile,
                file_count=file_count,
                estimated_lines=estimated_lines,
                budget_remaining_usd=budget_remaining_usd,
            )

        return self.model_selector.select(
            task_type,
            retry_count,
            specialization_profile,
            file_count,
            intelligent_router=router,
            routing_signals=routing_signals,
        )
