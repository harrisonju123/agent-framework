"""Claude CLI subprocess backend implementation."""

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Set

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

        # Clean up stale MCP cache files from terminated processes
        self._cleanup_stale_cache_files()

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
            "--print",  # Non-interactive mode - write to stdout and exit
            "--model", model,
            "--dangerously-skip-permissions",
            "--max-turns", str(self.max_turns),
        ]

        # Add MCP config if specified
        if self.mcp_config_path:
            cmd.extend([
                "--mcp-config", str(self.mcp_config_path),
                "--strict-mcp-config",  # Only use our MCP config, ignore global ~/.claude/mcp_settings.json
            ])

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
            log_file.write(f"Working Directory: {request.working_dir or 'current directory'}\n")
            log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Timeout: {timeout}s\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.flush()
            logger.info(f"Streaming Claude CLI output to {log_file_path}")

        try:
            # Prepare clean environment (exclude problematic beta flag)
            env = os.environ.copy()
            env.pop('CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS', None)

            # Determine working directory for subprocess
            # Use working_dir from request if provided, otherwise inherit current directory
            cwd = request.working_dir if request.working_dir else None

            # Run subprocess with streaming output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            # Send prompt to stdin
            process.stdin.write(full_prompt.encode())
            await process.stdin.drain()
            process.stdin.close()

            # Stream output with timeout
            stdout_chunks = []
            stderr_chunks = []
            timed_out = False

            # Track whether we've written the stderr header
            stderr_header_written = [False]

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
                            # Write stderr header only once when first stderr content arrives
                            if name == "stderr" and not stderr_header_written[0] and decoded.strip():
                                log_file.write(f"\n{'='*50}\n")
                                log_file.write(f"STDERR:\n")
                                log_file.write(f"{'='*50}\n")
                                stderr_header_written[0] = True
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
                log_file.write(f"SUMMARY\n")
                log_file.write(f"{'=' * 50}\n")
                log_file.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Duration: {latency_ms/1000:.1f}s\n")
                log_file.write(f"Exit code: {process.returncode}\n")
                log_file.write(f"Timed out: {timed_out}\n")
                if stderr_text:
                    log_file.write(f"\nSTDERR Summary:\n{stderr_text[:1000]}\n")
                if process.returncode != 0:
                    log_file.write(f"\n⚠️  FAILED - See stderr above for details\n")
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
                # Build detailed error message with both stdout and stderr
                error_parts = [f"Exit code {process.returncode}"]
                if stderr_text.strip():
                    error_parts.append(f"STDERR: {stderr_text.strip()}")
                if stdout_text.strip():
                    error_parts.append(f"STDOUT: {stdout_text.strip()}")
                error_msg = " | ".join(error_parts)

                logger.error(
                    f"Claude CLI failed: returncode={process.returncode}\n"
                    f"STDERR: {stderr_text[:1000]}\n"
                    f"STDOUT: {stdout_text[:1000]}\n"
                    f"Log: {log_file_path}"
                )
                return LLMResponse(
                    content=stdout_text,  # Include output even on failure for debugging
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
        """Expand environment variables in MCP config and write to process-specific temp file."""
        with open(config_path) as f:
            config = json.load(f)

        # Collect all env vars referenced in the config
        env_vars_used = self._collect_env_vars(config)

        # Recursively expand ${VAR} in all string values
        expanded = self._expand_env_vars_recursive(config)

        # Create hash of environment values for cache key
        env_hash = hashlib.md5(
            json.dumps({k: os.environ.get(k, '') for k in sorted(env_vars_used)}).encode()
        ).hexdigest()[:8]

        # Write to process-specific file
        temp_dir = Path.home() / ".cache" / "agent-framework"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"mcp-config-{os.getpid()}-{env_hash}.json"

        with open(temp_path, 'w') as f:
            json.dump(expanded, f, indent=2)

        logger.debug(f"Expanded MCP config to process-specific file: {temp_path}")
        return temp_path

    def _collect_env_vars(self, obj, vars_set: Optional[Set[str]] = None) -> Set[str]:
        """Recursively collect all ${VAR} references in the config."""
        if vars_set is None:
            vars_set = set()

        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_env_vars(v, vars_set)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_env_vars(item, vars_set)
        elif isinstance(obj, str):
            # Find all ${VAR} patterns
            for match in re.finditer(r'\$\{(\w+)\}', obj):
                vars_set.add(match.group(1))

        return vars_set

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
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(
                        f"MCP config references undefined environment variable: {var_name}"
                    )
                    return match.group(0)  # Keep placeholder if var not set
                return value
            return re.sub(r'\$\{(\w+)\}', replace_var, obj)
        else:
            return obj

    def _cleanup_stale_cache_files(self):
        """Remove stale MCP config cache files from terminated processes."""
        temp_dir = Path.home() / ".cache" / "agent-framework"
        if not temp_dir.exists():
            return

        for cache_file in temp_dir.glob("mcp-config-*.json"):
            # Extract PID from filename (format: mcp-config-{PID}-{HASH}.json)
            try:
                parts = cache_file.stem.split("-")
                if len(parts) >= 3:
                    pid = int(parts[2])
                    # Check if process still exists
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        # Process is dead, remove stale cache
                        cache_file.unlink()
                        logger.debug(f"Removed stale MCP cache: {cache_file}")
            except (ValueError, IndexError):
                pass
