"""Rich logging with structured context and better formatting."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AgentLogFormatter(logging.Formatter):
    """Custom formatter with task context."""

    def __init__(self, agent_id: str, use_colors: bool = True):
        super().__init__()
        self.agent_id = agent_id
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with context."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Add task context if available
        task_context = ""
        if hasattr(record, "jira_key"):
            task_context = f"[{record.jira_key}] "
        elif hasattr(record, "task_id"):
            task_context = f"[{record.task_id[:8]}...] "

        # Add phase if available
        phase_context = ""
        if hasattr(record, "phase"):
            phase_context = f"[{record.phase}] "

        # Color codes (only if enabled)
        if self.use_colors:
            level_colors = {
                "DEBUG": "\033[36m",      # Cyan
                "INFO": "\033[32m",       # Green
                "WARNING": "\033[33m",    # Yellow
                "ERROR": "\033[31m",      # Red
                "CRITICAL": "\033[35m",   # Magenta
            }
            reset = "\033[0m"
            level_color = level_colors.get(record.levelname, "")
        else:
            level_color = ""
            reset = ""

        return (
            f"{timestamp} {level_color}{record.levelname:8s}{reset} "
            f"[{self.agent_id}] {phase_context}{task_context}{record.getMessage()}"
        )


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds task context to all log messages."""

    def __init__(self, logger: logging.Logger, agent_id: str):
        super().__init__(logger, {})
        self.agent_id = agent_id
        self.current_task_id: Optional[str] = None
        self.current_jira_key: Optional[str] = None
        self.current_phase: Optional[str] = None

    def set_task_context(
        self,
        task_id: Optional[str] = None,
        jira_key: Optional[str] = None,
        phase: Optional[str] = None,
    ):
        """Set current task context for logging."""
        if task_id:
            self.current_task_id = task_id
        if jira_key:
            self.current_jira_key = jira_key
        if phase is not None:  # Allow clearing phase with None
            self.current_phase = phase

    def clear_context(self):
        """Clear task context."""
        self.current_task_id = None
        self.current_jira_key = None
        self.current_phase = None

    def process(self, msg, kwargs):
        """Add context to log record."""
        extra = kwargs.get("extra", {})

        if self.current_task_id:
            extra["task_id"] = self.current_task_id
        if self.current_jira_key:
            extra["jira_key"] = self.current_jira_key
        if self.current_phase:
            extra["phase"] = self.current_phase

        kwargs["extra"] = extra
        return msg, kwargs

    def task_started(self, task_id: str, title: str, jira_key: Optional[str] = None):
        """Log task start with context."""
        self.set_task_context(task_id=task_id, jira_key=jira_key)
        self.info(f"📋 Starting task: {title}")

    def phase_change(self, phase: str):
        """Log phase change."""
        self.set_task_context(phase=phase)
        phase_emoji = {
            "analyzing": "🔍",
            "planning": "📝",
            "executing_llm": "🤖",
            "implementing": "⚙️",
            "testing": "🧪",
            "committing": "💾",
            "creating_pr": "🔀",
            "updating_jira": "📊",
        }
        emoji = phase_emoji.get(phase.lower(), "▶️")
        self.info(f"{emoji} Phase: {phase}")

    def task_completed(self, duration_seconds: float, tokens_used: Optional[int] = None):
        """Log task completion with metrics."""
        msg = f"✅ Task completed in {duration_seconds:.1f}s"
        if tokens_used:
            msg += f" ({tokens_used:,} tokens)"
        self.info(msg)
        self.clear_context()

    def task_failed(self, error: str, retry_count: int):
        """Log task failure."""
        self.error(f"❌ Task failed (attempt {retry_count + 1}): {error}")
        self.clear_context()

    def token_usage(self, input_tokens: int, output_tokens: int, cost: float):
        """Log token usage and cost."""
        total = input_tokens + output_tokens
        self.info(
            f"💰 Tokens: {input_tokens:,} in + {output_tokens:,} out = {total:,} total "
            f"(~${cost:.4f})"
        )

    def progress(self, message: str):
        """Log progress update."""
        self.info(f"⏳ {message}")


def setup_rich_logging(
    agent_id: str,
    workspace: Path,
    log_level: str = "INFO",
    use_file: bool = True,
    use_json: bool = False,
) -> ContextLogger:
    """
    Setup rich logging with better formatting.

    Args:
        agent_id: Agent identifier
        workspace: Workspace path
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_file: Write to log file
        use_json: Use JSON structured logging

    Returns:
        ContextLogger instance
    """
    # Use PID to ensure unique logger per process (prevents collision with other replicas)
    unique_logger_name = f"{agent_id}-{os.getpid()}"
    logger = logging.getLogger(unique_logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Close existing handlers before clearing (prevents file descriptor leak)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    if use_json:
        # JSON structured logging for parsing (fixed format string)
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","agent":"%(agent)s","level":"%(levelname)s",'
            '"message":"%(message)s","module":"%(module)s","function":"%(funcName)s"}',
            defaults={'agent': agent_id}
        )
    else:
        # Human-readable formatting with colors for console
        formatter = AgentLogFormatter(agent_id, use_colors=True)

    # Check if stdout is redirected (running as subprocess)
    # If redirected, skip console handler to avoid duplicate logs
    stdout_is_redirected = not sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False

    # Console handler (only if not redirected)
    if not stdout_is_redirected:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (always add when use_file=True)
    if use_file:
        log_dir = workspace / "logs"
        log_dir.mkdir(exist_ok=True)

        # Use plain formatter for files (no ANSI codes)
        plain_formatter = AgentLogFormatter(agent_id, use_colors=False)

        file_handler = logging.FileHandler(log_dir / f"{agent_id}.log")
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    # Wrap in context logger
    context_logger = ContextLogger(logger, agent_id)

    return context_logger
