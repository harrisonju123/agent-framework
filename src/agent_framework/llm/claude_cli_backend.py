"""Claude CLI subprocess backend implementation."""

import asyncio
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType


class ClaudeCLIBackend(LLMBackend):
    """
    LLM backend using Claude CLI subprocess.

    Ported from scripts/async-agent-runner.sh lines 355-358.
    """

    # Default timeout for LLM calls (5 minutes)
    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        executable: str = "claude",
        max_turns: int = 999,
        cheap_model: str = "haiku",
        default_model: str = "sonnet",
        premium_model: str = "opus",
        mcp_config_path: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.executable = executable
        self.max_turns = max_turns
        self.timeout = timeout
        self.model_selector = ModelSelector(cheap_model, default_model, premium_model)

        # Expand environment variables in MCP config if provided
        if mcp_config_path:
            self.mcp_config_path = self._expand_mcp_config(Path(mcp_config_path))
        else:
            self.mcp_config_path = None

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

        # Add MCP config if specified
        if self.mcp_config_path:
            cmd.extend(["--mcp-config", str(self.mcp_config_path)])

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

            # Apply timeout to prevent indefinite hangs
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=full_prompt.encode()),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                latency_ms = (time.time() - start_time) * 1000
                return LLMResponse(
                    content="",
                    model_used=model,
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=f"Claude CLI timed out after {self.timeout} seconds",
                )

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

    def _expand_mcp_config(self, config_path: Path) -> Path:
        """Expand environment variables in MCP config and write to temp file."""
        with open(config_path) as f:
            config = json.load(f)

        # Recursively expand ${VAR} in all string values
        expanded = self._expand_env_vars_recursive(config)

        # Write to temporary file
        temp_dir = Path.home() / ".cache" / "agent-framework"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "mcp-config-expanded.json"

        with open(temp_path, 'w') as f:
            json.dump(expanded, f, indent=2)

        return temp_path

    def _expand_env_vars_recursive(self, obj):
        """Recursively expand ${VAR} in dict/list/str."""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars_recursive(item) for item in obj]
        elif isinstance(obj, str):
            # Expand ${VAR} patterns
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            return re.sub(r'\$\{(\w+)\}', replace_var, obj)
        else:
            return obj
