"""Atomic file I/O operations."""

import os
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def atomic_write_json(file_path: Path, content: str, max_retries: int = 3) -> None:
    """
    Atomically write content to a file using temp file + rename.

    This prevents partial writes and ensures the file is either fully written
    or not written at all (in case of crashes/interruptions).

    Args:
        file_path: Target file path
        content: Content to write (typically JSON string)
        max_retries: Maximum number of retry attempts on failure

    Raises:
        OSError: If write fails after all retries
    """
    # Use PID to avoid temp file collisions between processes
    tmp_file = file_path.with_suffix(f'{file_path.suffix}.tmp.{os.getpid()}')

    last_error = None
    for attempt in range(max_retries):
        try:
            tmp_file.write_text(content)
            tmp_file.rename(file_path)
            return
        except OSError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to write {file_path} (attempt {attempt + 1}/{max_retries}): {e}"
                )
            continue
        finally:
            # Clean up temp file if it still exists
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    # All retries failed
    logger.error(f"Failed to write {file_path} after {max_retries} attempts: {last_error}")
    raise last_error


def atomic_write_text(file_path: Path, content: str) -> None:
    """Semantic alias for atomic_write_json — for non-JSON text content.

    Delegates to atomic_write_json which handles temp file + rename.
    """
    atomic_write_json(file_path, content)


def atomic_write_model(file_path: Path, model: BaseModel, indent: int = 2) -> None:
    """
    Atomically write a Pydantic model to JSON file.

    Args:
        file_path: Target file path
        model: Pydantic model to serialize
        indent: JSON indentation (default: 2)
    """
    atomic_write_json(file_path, model.model_dump_json(indent=indent))
