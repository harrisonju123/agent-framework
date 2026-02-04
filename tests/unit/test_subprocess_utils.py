"""Tests for subprocess_utils."""

import subprocess
from pathlib import Path
import pytest

from agent_framework.utils.subprocess_utils import (
    SubprocessError,
    run_command,
    run_git_command,
    check_command_exists,
    get_command_output,
)


def test_subprocess_error_includes_context():
    """Test SubprocessError includes all context."""
    error = SubprocessError(
        cmd="git status",
        returncode=1,
        stderr="error message",
        stdout="output",
        cwd=Path("/tmp"),
        timed_out=True,
    )

    assert error.cmd == "git status"
    assert error.returncode == 1
    assert error.stderr == "error message"
    assert error.stdout == "output"
    assert error.cwd == Path("/tmp")
    assert error.timed_out is True
    assert "/tmp" in str(error)
    assert "timed out" in str(error)


def test_run_command_success():
    """Test run_command succeeds for valid command."""
    result = run_command(["echo", "hello"], check=True)
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_command_failure_raises():
    """Test run_command raises SubprocessError on failure."""
    with pytest.raises(SubprocessError) as exc_info:
        run_command(["false"], check=True)

    assert exc_info.value.returncode != 0


def test_run_command_failure_no_check():
    """Test run_command does not raise when check=False."""
    result = run_command(["false"], check=False)
    assert result.returncode != 0


def test_run_command_timeout():
    """Test run_command handles timeout."""
    with pytest.raises(SubprocessError) as exc_info:
        run_command(["sleep", "10"], check=True, timeout=1)

    assert exc_info.value.timed_out is True


def test_run_git_command_success(tmp_path):
    """Test run_git_command succeeds in git repo."""
    # Initialize a git repo
    run_command(["git", "init"], cwd=tmp_path, check=True)
    run_command(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    run_command(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # Test git command
    result = run_git_command(["status"], cwd=tmp_path, check=True)
    assert result.returncode == 0
    assert "branch" in result.stdout.lower() or "initial commit" in result.stdout.lower()


def test_run_git_command_flexible_timeout(tmp_path):
    """Test run_git_command accepts None timeout."""
    # Initialize a git repo
    run_command(["git", "init"], cwd=tmp_path, check=True)

    # Test with no timeout
    result = run_git_command(["status"], cwd=tmp_path, timeout=None)
    assert result.returncode == 0


def test_check_command_exists_true():
    """Test check_command_exists returns True for existing command."""
    assert check_command_exists("echo") is True
    assert check_command_exists("git") is True


def test_check_command_exists_false():
    """Test check_command_exists returns False for non-existing command."""
    assert check_command_exists("nonexistent_command_12345") is False


def test_get_command_output():
    """Test get_command_output returns stdout."""
    output = get_command_output(["echo", "hello world"])
    assert output == "hello world"


def test_get_command_output_strips_whitespace():
    """Test get_command_output strips whitespace."""
    output = get_command_output(["echo", "  hello  "])
    assert output == "hello"


def test_run_command_with_cwd(tmp_path):
    """Test run_command respects cwd parameter."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    result = run_command(["ls"], cwd=tmp_path, check=True)
    assert "test.txt" in result.stdout
