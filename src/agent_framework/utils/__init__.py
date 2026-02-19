"""Shared utility functions for the agent framework."""

from .atomic_io import atomic_write_json, atomic_write_model
from .error_handling import (
    log_and_reraise,
    log_and_ignore,
    safe_call,
    handle_filesystem_errors,
    handle_network_errors,
    ErrorContext,
)
from .stream_parser import parse_jsonl_to_dicts, parse_jsonl_to_models
from .subprocess_utils import (
    SubprocessError,
    run_command,
    run_git_command,
    run_with_retry,
    check_command_exists,
    get_command_output,
)
from .process_utils import kill_process_tree
from .file_summarizer import summarize_file
from .type_helpers import get_type_str
from .validators import validate_branch_name, validate_identifier, validate_owner_repo

__all__ = [
    # Atomic I/O
    "atomic_write_json",
    "atomic_write_model",
    # Error handling
    "log_and_reraise",
    "log_and_ignore",
    "safe_call",
    "handle_filesystem_errors",
    "handle_network_errors",
    "ErrorContext",
    # Stream parsing
    "parse_jsonl_to_dicts",
    "parse_jsonl_to_models",
    # Subprocess utilities
    "SubprocessError",
    "run_command",
    "run_git_command",
    "run_with_retry",
    "check_command_exists",
    "get_command_output",
    # Process management
    "kill_process_tree",
    # File summarizer
    "summarize_file",
    # Type helpers
    "get_type_str",
    # Validators
    "validate_branch_name",
    "validate_identifier",
    "validate_owner_repo",
]
