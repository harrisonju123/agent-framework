"""Claude CLI subprocess backend implementation."""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional, Set

from .base import LLMBackend, LLMRequest, LLMResponse
from .model_selector import ModelSelector
from ..core.task import TaskType

logger = logging.getLogger(__name__)

# Env vars stripped from the Claude CLI subprocess to prevent the LLM's Bash tool
# from accessing credentials. MCP servers get tokens from the expanded config file's
# env block, not from the parent process environment.
_SENSITIVE_ENV_VARS = frozenset({'JIRA_API_TOKEN', 'JIRA_EMAIL'})


def _summarize_tool_input(tool_name: str, tool_input: dict) -> Optional[str]:
    """Extract a short human-readable summary from tool input."""
    if not tool_input:
        return None

    if tool_name in ("Read", "Edit", "Write"):
        path = tool_input.get("file_path") or tool_input.get("path", "")
        if path:
            # Last 3 path segments for brevity
            parts = path.replace("\\", "/").split("/")
            return "/".join(parts[-3:]) if len(parts) > 3 else path
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:60] if cmd else None
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        if pattern and path:
            parts = path.replace("\\", "/").split("/")
            short_path = "/".join(parts[-2:]) if len(parts) > 2 else path
            return f'"{pattern}" in {short_path}'
        return f'"{pattern}"' if pattern else None
    elif tool_name == "Glob":
        return tool_input.get("pattern") or None
    else:
        # MCP tools or others — look for common identifier keys
        for key in ("issueKey", "owner", "repo", "query", "jql", "summary"):
            if key in tool_input:
                val = str(tool_input[key])
                return val[:60] if val else None

    return None


def _process_stream_line(
    line: str,
    text_chunks: list,
    usage_result: dict,
    log_file=None,
    on_tool_activity: Optional[Callable] = None,
    on_session_tool_call: Optional[Callable] = None,
):
    """Parse a single JSON line from --output-format stream-json.

    Extracts text content for log streaming and captures token usage
    from the final result event.

    Args:
        line: Raw line from stdout (may or may not be JSON)
        text_chunks: Accumulator for assistant text content
        usage_result: Dict to populate with usage data from result event
        log_file: Optional file handle for real-time log output
        on_tool_activity: Optional callback invoked with (tool_name, tool_input_summary)
        on_session_tool_call: Optional callback invoked with (tool_name, tool_input_dict)
            for structured session logging
    """
    line = line.strip()
    if not line:
        return

    try:
        event = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        # Not JSON — CLI version mismatch or non-JSON output, treat as raw text
        text_chunks.append(line + "\n")
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()
        return

    event_type = event.get("type")

    if event_type == "assistant":
        message = event.get("message", {})

        # Accumulate per-turn usage as fallback when result event never arrives
        # (known CLI bug: github.com/anthropics/claude-code/issues/1920)
        msg_usage = message.get("usage", {})
        if msg_usage:
            usage_result["input_tokens"] = (
                usage_result.get("input_tokens", 0) + msg_usage.get("input_tokens", 0)
            )
            usage_result["output_tokens"] = (
                usage_result.get("output_tokens", 0) + msg_usage.get("output_tokens", 0)
            )

        # Extract text from message content blocks
        for block in message.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                text_chunks.append(text)
                if log_file:
                    log_file.write(text)
                    log_file.flush()
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "unknown")
                marker = f"\n[Tool Call: {tool_name}]\n"
                text_chunks.append(marker)
                if log_file:
                    log_file.write(marker)
                    log_file.flush()
                tool_input = block.get("input", {})
                if on_tool_activity:
                    summary = _summarize_tool_input(tool_name, tool_input)
                    on_tool_activity(tool_name, summary)
                if on_session_tool_call:
                    on_session_tool_call(tool_name, tool_input)

    elif event_type == "result":
        # Result event carries authoritative session cost, but its usage field
        # may only reflect the final turn — not the full multi-turn session.
        # Use the larger of accumulated vs result to avoid under-reporting.
        usage = event.get("usage", {})
        usage_result["input_tokens"] = max(
            usage_result.get("input_tokens", 0), usage.get("input_tokens", 0)
        )
        usage_result["output_tokens"] = max(
            usage_result.get("output_tokens", 0), usage.get("output_tokens", 0)
        )
        usage_result["total_cost_usd"] = event.get("total_cost_usd")

        result_text = event.get("result", "")
        if result_text:
            usage_result["result_text"] = result_text

    elif event_type == "system":
        if log_file:
            subtype = event.get("subtype", "")
            session_id = event.get("session_id", "")
            if subtype == "init" and session_id:
                log_file.write(f"[Session: {session_id}]\n")
                log_file.flush()

    else:
        logger.debug(f"Unknown stream-json event type: {event_type}")


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
        proxy_env: Optional[dict] = None,
        use_max_account: bool = False,
        max_idle_timeouts: int = 5,
    ):
        self.executable = executable
        self.max_turns = max_turns
        self.timeout = timeout  # Fallback timeout when task_type not specified
        self.proxy_env = proxy_env or {}
        self.use_max_account = use_max_account
        self.max_idle_timeouts = max_idle_timeouts
        self.model_selector = ModelSelector(
            cheap_model, default_model, premium_model,
            timeout_large=timeout_large,
            timeout_bounded=timeout_bounded,
            timeout_simple=timeout_simple,
        )
        self.logs_dir = logs_dir or Path("logs")
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._partial_output: list[str] = []

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
        on_tool_activity: Optional[Callable] = None,
        on_session_tool_call: Optional[Callable] = None,
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
            model = self.select_model(
                request.task_type,
                request.retry_count,
                request.specialization_profile,
                request.file_count,
            )
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
            "--output-format", "stream-json",  # Line-delimited JSON with token usage
            "--verbose",  # Include system events for session tracking
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

        # Add agent teammates for Claude Agent Teams
        if request.agents:
            cmd.extend(["--agents", json.dumps(request.agents)])
            logger.debug(f"Team mode active: teammates={list(request.agents.keys())}")

        # Restrict tool access for PREVIEW tasks — enforces read-only mode at the
        # CLI level. Bash is still allowed for read-only exploration (git log, ls, etc.)
        # so this is defense-in-depth alongside the prompt injection, not airtight.
        if request.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(request.allowed_tools)])
            logger.debug(f"Tool restriction active: allowed_tools={request.allowed_tools}")

        # Behavioral directive appended to the LLM's system prompt (e.g. read-efficiency hints)
        if request.append_system_prompt:
            cmd.extend(["--append-system-prompt", request.append_system_prompt])

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

        try:
            # Open file inside try block so finally always closes it
            if log_file_path:
                log_file = open(log_file_path, "a")
                log_file.write(f"=== Claude CLI Task: {task_id} (attempt {request.retry_count + 1}) ===\n")
                log_file.write(f"Model: {model}\n")
                log_file.write(f"Working Directory: {request.working_dir or 'current directory'}\n")
                log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Timeout: {timeout}s\n")
                log_file.write("=" * 50 + "\n\n")
                log_file.flush()
                logger.info(f"Streaming Claude CLI output to {log_file_path}")

            # Prepare clean environment (disable experimental betas for AWS Bedrock compatibility)
            # Note: CLAUDECODE is stripped at process startup in run_agent.py
            env = os.environ.copy()
            if self.use_max_account:
                # Strip all proxy/API vars so Claude CLI uses the OAuth/Max account directly
                for key in ('ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN'):
                    env.pop(key, None)
                logger.debug("Stripped Anthropic env vars for Max account mode")
            else:
                env.update(self.proxy_env)
            env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] = '1'
            if request.env_vars:
                env.update(request.env_vars)
            for key in _SENSITIVE_ENV_VARS:
                env.pop(key, None)

            if task_id:
                env['AGENT_TASK_ID'] = task_id
                # Root task ID for cross-step read cache keying
                root_id = request.context.get('_root_task_id', task_id) if request.context else task_id
                env['AGENT_ROOT_TASK_ID'] = root_id
                # Workflow step for read cache attribution
                workflow_step = request.context.get('workflow_step', '') if request.context else ''
                if workflow_step:
                    env['WORKFLOW_STEP'] = workflow_step

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
            self._current_process = process

            # Send prompt to stdin
            process.stdin.write(full_prompt.encode())
            await process.stdin.drain()
            process.stdin.close()

            # Stream output with timeout
            text_chunks = []     # Human-readable text extracted from JSON events
            self._partial_output = text_chunks  # Alias so partial output survives cancellation
            usage_result = {}    # Token usage and cost from final result event
            stderr_chunks = []
            timed_out = False

            # Track whether we've written the stderr header
            stderr_header_written = [False]

            async def read_stdout_stream_json(stream):
                """Read stdout as line-delimited JSON, parse each event."""
                buffer = b""
                consecutive_timeouts = 0
                max_idle_timeouts = self.max_idle_timeouts
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.read(4096),
                                timeout=60
                            )
                        except asyncio.TimeoutError:
                            consecutive_timeouts += 1
                            # If subprocess already exited, no more data is coming
                            if process.returncode is not None:
                                logger.debug(
                                    f"stdout: process exited (rc={process.returncode}), "
                                    f"stopping after {consecutive_timeouts} idle timeout(s)"
                                )
                                break
                            if consecutive_timeouts >= max_idle_timeouts:
                                logger.warning(
                                    f"stdout: {consecutive_timeouts} consecutive read timeouts "
                                    f"({consecutive_timeouts * 60}s idle), giving up"
                                )
                                break
                            logger.debug(f"stdout: read timeout #{consecutive_timeouts}, retrying")
                            continue
                        if not chunk:
                            # Process any remaining buffered data
                            if buffer:
                                _process_stream_line(
                                    buffer.decode(errors='replace'),
                                    text_chunks, usage_result, log_file,
                                    on_tool_activity,
                                    on_session_tool_call,
                                )
                            break
                        consecutive_timeouts = 0
                        buffer += chunk
                        # Split on newlines to get complete JSON lines
                        while b"\n" in buffer:
                            line_bytes, buffer = buffer.split(b"\n", 1)
                            _process_stream_line(
                                line_bytes.decode(errors='replace'),
                                text_chunks, usage_result, log_file,
                                on_tool_activity,
                                on_session_tool_call,
                            )
                except Exception as e:
                    logger.debug(f"Stream read error (stdout): {e}")

            async def read_stderr_stream(stream):
                """Read stderr chunks (unchanged — not JSON formatted)."""
                consecutive_timeouts = 0
                max_idle_timeouts = self.max_idle_timeouts
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.read(4096),
                                timeout=60
                            )
                        except asyncio.TimeoutError:
                            consecutive_timeouts += 1
                            if process.returncode is not None:
                                logger.debug(
                                    f"stderr: process exited (rc={process.returncode}), "
                                    f"stopping after {consecutive_timeouts} idle timeout(s)"
                                )
                                break
                            if consecutive_timeouts >= max_idle_timeouts:
                                logger.warning(
                                    f"stderr: {consecutive_timeouts} consecutive read timeouts "
                                    f"({consecutive_timeouts * 60}s idle), giving up"
                                )
                                break
                            continue
                        if not chunk:
                            break
                        consecutive_timeouts = 0
                        decoded = chunk.decode(errors='replace')
                        stderr_chunks.append(decoded)
                        if log_file:
                            if not stderr_header_written[0] and decoded.strip():
                                log_file.write(f"\n{'='*50}\n")
                                log_file.write(f"STDERR:\n")
                                log_file.write(f"{'='*50}\n")
                                stderr_header_written[0] = True
                            log_file.write(decoded)
                            log_file.flush()
                except Exception as e:
                    logger.debug(f"Stream read error (stderr): {e}")

            async def wait_for_process():
                """Wait for subprocess exit, then close streams to unblock readers."""
                await process.wait()
                # Subprocess done — close transport to force EOF on readers
                try:
                    process.stdout._transport.close()
                except Exception:
                    pass
                try:
                    process.stderr._transport.close()
                except Exception:
                    pass

            try:
                # Race stream readers against process exit and overall timeout.
                # wait_for_process closes the transports when the subprocess dies,
                # which unblocks any hung stream.read() calls immediately.
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stdout_stream_json(process.stdout),
                        read_stderr_stream(process.stderr),
                        wait_for_process(),
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                timed_out = True
                if log_file:
                    log_file.write(f"\n\n{'=' * 50}\n")
                    log_file.write(f"TIMEOUT after {timeout} seconds\n")
                    log_file.write(f"Process killed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.flush()
                logger.warning(f"Claude CLI timed out after {timeout}s, killing process")
                process.kill()
                await process.wait()

            latency_ms = (time.time() - start_time) * 1000
            # Prefer authoritative result text from result event, fall back to accumulated chunks
            content = usage_result.get("result_text") or "".join(text_chunks)
            stderr_text = "".join(stderr_chunks)

            # Extract usage data (available if result event was received)
            input_tokens = usage_result.get("input_tokens", 0)
            output_tokens = usage_result.get("output_tokens", 0)
            reported_cost = usage_result.get("total_cost_usd")

            if log_file:
                log_file.write(f"\n\n{'=' * 50}\n")
                log_file.write(f"SUMMARY\n")
                log_file.write(f"{'=' * 50}\n")
                log_file.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Duration: {latency_ms/1000:.1f}s\n")
                log_file.write(f"Exit code: {process.returncode}\n")
                log_file.write(f"Timed out: {timed_out}\n")
                log_file.write(f"Tokens: {input_tokens} in / {output_tokens} out\n")
                if reported_cost is not None:
                    log_file.write(f"Cost: ${reported_cost:.4f}\n")
                if stderr_text:
                    log_file.write(f"\nSTDERR Summary:\n{stderr_text[:1000]}\n")
                if process.returncode != 0:
                    log_file.write(f"\nFAILED - See stderr above for details\n")

            if timed_out:
                return LLMResponse(
                    content=content,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=f"Claude CLI timed out after {timeout} seconds. Output logged to {log_file_path}",
                    reported_cost_usd=reported_cost,
                )

            if process.returncode == 0:
                return LLMResponse(
                    content=content,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                    success=True,
                    reported_cost_usd=reported_cost,
                )
            else:
                # Build detailed error message with both stdout and stderr
                error_parts = [f"Exit code {process.returncode}"]
                if stderr_text.strip():
                    error_parts.append(f"STDERR: {stderr_text.strip()}")
                if content.strip():
                    error_parts.append(f"STDOUT: {content.strip()}")
                error_msg = " | ".join(error_parts)

                # Truncate bloated errors (e.g. full stack traces from Claude CLI)
                # so task.last_error stays readable for retries and escalations
                from ..safeguards.escalation import EscalationHandler
                error_msg = EscalationHandler().truncate_error(error_msg)

                logger.error(
                    f"Claude CLI failed: returncode={process.returncode}\n"
                    f"STDERR: {stderr_text[:1000]}\n"
                    f"STDOUT: {content[:1000]}\n"
                    f"Log: {log_file_path}"
                )
                return LLMResponse(
                    content=content,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason="error",
                    latency_ms=latency_ms,
                    success=False,
                    error=error_msg,
                    reported_cost_usd=reported_cost,
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if log_file:
                log_file.write(f"\n\n{'=' * 50}\n")
                log_file.write(f"ERROR: {e}\n")
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
            self._current_process = None
            if log_file:
                log_file.close()

    def cancel(self) -> None:
        """Kill the in-flight claude CLI subprocess if one is running."""
        proc = self._current_process
        if proc and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def get_partial_output(self) -> str:
        """Return accumulated output from the in-flight LLM call."""
        return "".join(self._partial_output) if self._partial_output else ""

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
        temp_dir.chmod(0o700)
        temp_path = temp_dir / f"mcp-config-{os.getpid()}-{env_hash}.json"

        with open(temp_path, 'w') as f:
            json.dump(expanded, f, indent=2)
        temp_path.chmod(0o600)

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
