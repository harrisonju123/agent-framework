"""Standardized error handling utilities."""

import functools
import logging
from typing import TypeVar, Callable, Optional, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


def log_and_reraise(
    error: Exception,
    message: str,
    *,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log an error with context and re-raise it.

    Args:
        error: Exception to log and re-raise
        message: Context message to log
        logger_instance: Logger to use (defaults to module logger)
        level: Log level (default: ERROR)

    Raises:
        The original exception
    """
    log = logger_instance or logger
    log.log(level, f"{message}: {error}")
    raise


def log_and_ignore(
    error: Exception,
    message: str,
    *,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.WARNING,
) -> None:
    """
    Log an error and ignore it (don't re-raise).

    Use for non-critical errors that should not interrupt the flow.

    Args:
        error: Exception to log
        message: Context message to log
        logger_instance: Logger to use (defaults to module logger)
        level: Log level (default: WARNING)
    """
    log = logger_instance or logger
    log.log(level, f"{message}: {error}")


def safe_call(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_errors: bool = True,
    error_message: str = "Error in function call",
    **kwargs,
) -> Optional[T]:
    """
    Safely call a function, catching and logging exceptions.

    Args:
        func: Function to call
        *args: Positional arguments for func
        default: Default value to return on error
        log_errors: Whether to log errors
        error_message: Message to log on error
        **kwargs: Keyword arguments for func

    Returns:
        Function result on success, default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"{error_message}: {e}")
        return default


def handle_filesystem_errors(
    operation: str,
    *,
    logger_instance: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator to handle common filesystem errors with consistent messaging.

    Args:
        operation: Description of the operation (e.g., "read file", "write file")
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        Decorator function
    """
    log = logger_instance or logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except PermissionError as e:
                log.error(f"Permission denied during {operation}: {e}")
                raise
            except FileNotFoundError as e:
                log.error(f"File not found during {operation}: {e}")
                raise
            except OSError as e:
                # Disk full, etc.
                log.error(f"OS error during {operation}: {e}")
                raise
            except Exception as e:
                log.error(f"Unexpected error during {operation}: {e}")
                raise

        return wrapper

    return decorator


def handle_network_errors(
    operation: str,
    *,
    logger_instance: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator to handle common network errors with consistent messaging.

    Args:
        operation: Description of the operation (e.g., "fetch data", "API call")
        logger_instance: Logger to use (defaults to module logger)

    Returns:
        Decorator function
    """
    log = logger_instance or logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except TimeoutError as e:
                log.error(f"Timeout during {operation}: {e}")
                raise
            except ConnectionError as e:
                log.error(f"Connection error during {operation}: {e}")
                raise
            except Exception as e:
                log.error(f"Unexpected error during {operation}: {e}")
                raise

        return wrapper

    return decorator


class ErrorContext:
    """
    Context manager for handling errors with consistent logging.

    Usage:
        with ErrorContext("reading configuration", raise_on_error=False):
            config = load_config()
    """

    def __init__(
        self,
        operation: str,
        *,
        raise_on_error: bool = True,
        default_value: Any = None,
        logger_instance: Optional[logging.Logger] = None,
        log_level: int = logging.ERROR,
    ):
        """
        Initialize error context.

        Args:
            operation: Description of the operation
            raise_on_error: Whether to re-raise exceptions
            default_value: Value to return if error and not raising
            logger_instance: Logger to use
            log_level: Log level for errors
        """
        self.operation = operation
        self.raise_on_error = raise_on_error
        self.default_value = default_value
        self.logger = logger_instance or logger
        self.log_level = log_level
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            self.logger.log(
                self.log_level,
                f"Error during {self.operation}: {exc_val}",
            )

            if self.raise_on_error:
                return False  # Re-raise exception
            else:
                return True  # Suppress exception

    def get_result(self, result: Any = None) -> Any:
        """
        Get result or default value if error occurred.

        Args:
            result: The actual result (if operation succeeded)

        Returns:
            Result if no error, default_value if error occurred
        """
        if self.error is not None:
            return self.default_value
        return result
