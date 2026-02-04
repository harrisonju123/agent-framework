"""Claude CLI subprocess backend implementation."""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType

logger = logging.getLogger(__name__)


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
        timeout_large: int = 3600,
        timeout_bounded: int = 1800,
        timeout_simple: int = 900,
        logs_dir: Optional[Path] = None,
    ):
        self.executable = executable
        self.max_turns = max_turns
        self.timeout = timeout  # Fallback timeout when task_type not specified
        self.model_selector = ModelSelector(
            cheap_model, default_model, premium_model,
            timeout_large=timeout_large,
            timeout_bounded=timeout_bounded,
            timeout_simple=timeout_simple,
        )
        self.logs_dir = logs_dir or Path("logs")

        # Expand environment variables in MCP config if provided
        if mcp_config_path:
            self.mcp_config_path = self._expand_mcp_config(Path(mcp_config_path))
        else:
            self.mcp_config_path = None

    async def complete(
        self,
        request: LLMRequest,
        task_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a completion request via Claude CLI subprocess.

        Spawns: echo "$PROMPT" | claude --model "$MODEL" --dangerously-skip-permissions --max-turns 999

        Output is streamed to a log file in real-time for visibility into long-running tasks.
        """
        start_time = time.time()

        # Select model
        if request.model:
            model = request.model
        elif request.task_type:
            model = self.select_model(request.task_type, request.retry_count)
        else:
            model = self.model_selector.default_model

        # Select timeout based on task type (dynamic timeout per task scope)
        if request.task_type:
            timeout = self.model_selector.select_timeout(request.task_type)
        else:
            timeout = self.timeout

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

        # Set up streaming log file
        log_file_path = None
        log_file = None
        if task_id:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = self.logs_dir / f"claude-cli-{task_id}.log"
            log_file = open(log_file_path, "w")
            log_file.write(f"=== Claude CLI Task: {task_id} ===\n")
            log_file.write(f"Model: {model}\n")
            log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Timeout: {timeout}s\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.flush()
            logger.info(f"Streaming Claude CLI output to {log_file_path}")

        try:
            # Prepare clean environment (exclude problematic beta flag)
            env = os.environ.copy()
            env.pop('CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS', None)

            # Run subprocess with streaming output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Send prompt to stdin
            process.stdin.write(full_prompt.encode())
            await process.stdin.drain()
            process.stdin.close()

            # Stream output with timeout
            stdout_chunks = []
            stderr_chunks = []
            timed_out = False

            async def read_stream(stream, chunks, name):
                """Read from stream and write to log file in real-time."""
                try:
                    while True:
                        chunk = await asyncio.wait_for(
                            stream.read(4096),
                            timeout=60  # Read timeout per chunk
                        )
                        if not chunk:
                            break
                        decoded = chunk.decode(errors='replace')
                        chunks.append(decoded)
                        if log_file:
                            log_file.write(decoded)
                            log_file.flush()
                except asyncio.TimeoutError:
                    pass  # Individual read timeout, continue
                except Exception as e:
                    logger.debug(f"Stream read error ({name}): {e}")

            try:
                # Read stdout and stderr concurrently with overall timeout
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(process.stdout, stdout_chunks, "stdout"),
                        read_stream(process.stderr, stderr_chunks, "stderr"),
                    ),
                    timeout=timeout
                )
                await process.wait()
            except asyncio.TimeoutError:
                timed_out = True
                if log_file:
                    log_file.write(f"\n\n{'=' * 50}\n")
                    log_file.write(f"⚠️  TIMEOUT after {timeout} seconds\n")
                    log_file.write(f"Process killed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.flush()
                logger.warning(f"Claude CLI timed out after {timeout}s, killing process")
                process.kill()
                await process.wait()

            latency_ms = (time.time() - start_time) * 1000
            stdout_text = "".join(stdout_chunks)
            stderr_text = "".join(stderr_chunks)

            if log_file:
                log_file.write(f"\n\n{'=' * 50}\n")
                log_file.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Duration: {latency_ms/1000:.1f}s\n")
                log_file.write(f"Exit code: {process.returncode}\n")
                log_file.write(f"Timed out: {timed_out}\n")
                log_file.close()

            if timed_out:
                return LLMResponse(
                    content=stdout_text,  # Include partial output
                    model_used=model,
                    input_tokens=0,
                    output_tokens=0,
                    finish_reason="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=f"Claude CLI timed out after {timeout} seconds. Output logged to {log_file_path}",
                )

            if process.returncode == 0:
                return LLMResponse(
                    content=stdout_text,
                    model_used=model,
                    input_tokens=0,  # CLI doesn't report token usage
                    output_tokens=0,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                    success=True,
                )
            else:
                error_msg = stderr_text or stdout_text or f"Exit code {process.returncode}"
                logger.error(
                    f"Claude CLI failed: returncode={process.returncode}, "
                    f"stderr={stderr_text[:500]}, stdout={stdout_text[:500]}"
                )
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
            if log_file:
                log_file.write(f"\n\n{'=' * 50}\n")
                log_file.write(f"ERROR: {e}\n")
                log_file.close()
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
