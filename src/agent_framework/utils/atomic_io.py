"""Atomic file I/O operations."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel


def atomic_write_json(file_path: Path, content: str) -> None:
    """
    Atomically write content to a file using temp file + rename.

    This prevents partial writes and ensures the file is either fully written
    or not written at all (in case of crashes/interruptions).

    Args:
        file_path: Target file path
        content: Content to write (typically JSON string)
    """
    tmp_file = file_path.with_suffix(f'{file_path.suffix}.tmp')
    tmp_file.write_text(content)
    tmp_file.rename(file_path)


def atomic_write_model(file_path: Path, model: BaseModel, indent: int = 2) -> None:
    """
    Atomically write a Pydantic model to JSON file.

    Args:
        file_path: Target file path
        model: Pydantic model to serialize
        indent: JSON indentation (default: 2)
    """
    atomic_write_json(file_path, model.model_dump_json(indent=indent))
