"""Standardized subprocess utilities for command execution."""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


class SubprocessError(Exception):
    """Exception raised when a subprocess command fails."""

    def __init__(self, cmd: str, returncode: int, stderr: str, stdout: str = ""):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(
            f"Command failed with exit code {returncode}: {cmd}\nstderr: {stderr}"
        )


def run_command(
    cmd: Union[str, List[str]],
    *,
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
    shell: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command with standardized error handling.

    Args:
        cmd: Command to run (string or list)
        cwd: Working directory
        capture_output: Capture stdout/stderr
        check: Raise exception on non-zero exit
        timeout: Timeout in seconds
        env: Environment variables
        shell: Use shell execution

    Returns:
        CompletedProcess with stdout, stderr, returncode

    Raises:
        SubprocessError: If check=True and command fails
        subprocess.TimeoutExpired: If timeout exceeded
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=env,
            shell=shell,
            check=False,  # We handle check ourselves for better error messages
        )

        if check and result.returncode != 0:
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            raise SubprocessError(
                cmd=cmd_str,
                returncode=result.returncode,
                stderr=result.stderr,
                stdout=result.stdout,
            )

        return result

    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {cmd}")
        raise


def run_git_command(
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """
    Run a git command with standardized error handling.

    Args:
        args: Git command arguments (without 'git' prefix)
        cwd: Working directory (git repo)
        check: Raise exception on non-zero exit
        timeout: Timeout in seconds (default: 30)

    Returns:
        CompletedProcess with stdout, stderr, returncode

    Raises:
        SubprocessError: If check=True and command fails
        subprocess.TimeoutExpired: If timeout exceeded
    """
    cmd = ["git"] + args

    try:
        return run_command(
            cmd,
            cwd=cwd,
            capture_output=True,
            check=check,
            timeout=timeout,
        )
    except SubprocessError as e:
        # Add git-specific context to error
        logger.error(f"Git command failed in {cwd}: {' '.join(args)}")
        raise


def run_with_retry(
    cmd: Union[str, List[str]],
    *,
    max_retries: int = 3,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command with automatic retry on failure.

    Useful for network operations or commands that may fail transiently.

    Args:
        cmd: Command to run
        max_retries: Maximum number of retry attempts
        cwd: Working directory
        timeout: Timeout per attempt

    Returns:
        CompletedProcess on success

    Raises:
        SubprocessError: If all retries exhausted
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return run_command(cmd, cwd=cwd, check=True, timeout=timeout)
        except SubprocessError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Command failed (attempt {attempt + 1}/{max_retries}), retrying: {cmd}"
                )
            continue

    # All retries failed
    logger.error(f"Command failed after {max_retries} attempts: {cmd}")
    raise last_error


def check_command_exists(command: str) -> bool:
    """
    Check if a command exists in PATH.

    Args:
        command: Command name to check

    Returns:
        True if command exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["which", command],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_command_output(
    cmd: Union[str, List[str]],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 30,
) -> str:
    """
    Run a command and return its output (stdout).

    Args:
        cmd: Command to run
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Command stdout as string (stripped)

    Raises:
        SubprocessError: If command fails
    """
    result = run_command(cmd, cwd=cwd, check=True, timeout=timeout)
    return result.stdout.strip()
