"""Claude CLI subprocess backend implementation."""

import asyncio
import time
from typing import Optional

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType


class ClaudeCLIBackend(LLMBackend):
    """
    LLM backend using Claude CLI subprocess.

    Ported from scripts/async-agent-runner.sh lines 355-358.
    """

    def __init__(
        self,
        executable: str = "claude",
        max_turns: int = 999,
        cheap_model: str = "haiku",
        default_model: str = "sonnet",
        premium_model: str = "opus",
    ):
        self.executable = executable
        self.max_turns = max_turns
        self.model_selector = ModelSelector(cheap_model, default_model, premium_model)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send a completion request via Claude CLI subprocess.

        Spawns: echo "$PROMPT" | claude --model "$MODEL" --dangerously-skip-permissions --max-turns 999
        """
        start_time = time.time()

        # Select model
        if request.model:
            model = request.model
        elif request.task_type:
            model = self.select_model(request.task_type, request.retry_count)
        else:
            model = self.model_selector.default_model

        # Build command
        cmd = [
            self.executable,
            "--model", model,
            "--dangerously-skip-permissions",
            "--max-turns", str(self.max_turns),
        ]

        # Build prompt (combine system + user)
        full_prompt = ""
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            full_prompt = request.prompt

        try:
            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(input=full_prompt.encode())
            latency_ms = (time.time() - start_time) * 1000

            if process.returncode == 0:
                return LLMResponse(
                    content=stdout.decode(),
                    model_used=model,
                    input_tokens=0,  # CLI doesn't report token usage
                    output_tokens=0,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                    success=True,
                )
            else:
                error_msg = stderr.decode() if stderr else f"Exit code {process.returncode}"
                return LLMResponse(
                    content="",
                    model_used=model,
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=error_msg,
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
