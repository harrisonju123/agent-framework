"""Tests for error_handling utilities."""

import logging
import pytest

from agent_framework.utils.error_handling import (
    log_and_reraise,
    log_and_ignore,
    safe_call,
    ErrorContext,
)


def test_log_and_reraise_raises_exception():
    """Test that log_and_reraise re-raises the exception."""
    with pytest.raises(ValueError):
        try:
            raise ValueError("test error")
        except ValueError as e:
            log_and_reraise(e, "Test context")


def test_log_and_ignore_does_not_raise(caplog):
    """Test that log_and_ignore does not re-raise."""
    with caplog.at_level(logging.WARNING):
        try:
            raise ValueError("test error")
        except ValueError as e:
            log_and_ignore(e, "Test context")

    assert "Test context: test error" in caplog.text


def test_safe_call_returns_result():
    """Test that safe_call returns result on success."""
    def success_func(x):
        return x * 2

    result = safe_call(success_func, 5)
    assert result == 10


def test_safe_call_returns_default_on_error():
    """Test that safe_call returns default on error."""
    def error_func():
        raise ValueError("error")

    result = safe_call(error_func, default=42)
    assert result == 42


def test_safe_call_logs_errors(caplog):
    """Test that safe_call logs errors."""
    def error_func():
        raise ValueError("test error")

    with caplog.at_level(logging.ERROR):
        safe_call(error_func, error_message="Custom error")

    assert "Custom error: test error" in caplog.text


def test_error_context_raises_on_error():
    """Test ErrorContext raises when raise_on_error=True."""
    with pytest.raises(ValueError):
        with ErrorContext("test operation", raise_on_error=True):
            raise ValueError("test error")


def test_error_context_suppresses_error():
    """Test ErrorContext suppresses error when raise_on_error=False."""
    with ErrorContext("test operation", raise_on_error=False) as ctx:
        raise ValueError("test error")

    assert ctx.error is not None
    assert isinstance(ctx.error, ValueError)


def test_error_context_bool_true_on_success():
    """Test ErrorContext.__bool__ returns True when no error."""
    with ErrorContext("test operation", raise_on_error=False) as ctx:
        pass

    assert bool(ctx) is True


def test_error_context_bool_false_on_error():
    """Test ErrorContext.__bool__ returns False when error occurred."""
    with ErrorContext("test operation", raise_on_error=False) as ctx:
        raise ValueError("test error")

    assert bool(ctx) is False


def test_error_context_get_result_returns_value_on_success():
    """Test ErrorContext.get_result returns value when no error."""
    with ErrorContext("test operation", raise_on_error=False) as ctx:
        result = 42

    assert ctx.get_result(result) == 42


def test_error_context_get_result_returns_default_on_error():
    """Test ErrorContext.get_result returns default when error occurred."""
    with ErrorContext("test operation", raise_on_error=False, default_value="default") as ctx:
        raise ValueError("test error")

    assert ctx.get_result("unused") == "default"


def test_error_context_usage_pattern():
    """Test realistic usage pattern with ErrorContext."""
    def risky_operation():
        return "success"

    with ErrorContext("risky operation", raise_on_error=False, default_value="failed") as ctx:
        result = risky_operation()

    final_result = ctx.get_result(result)
    assert final_result == "success"
    assert bool(ctx) is True
