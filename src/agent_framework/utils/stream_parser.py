"""Utilities for parsing JSONL (JSON Lines) streams."""

import json
from typing import Any, TypeVar
import logging

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def parse_jsonl_to_models(
    content: str,
    model_class: type[T],
    *,
    strict: bool = False
) -> list[T]:
    """
    Parse JSONL content into list of Pydantic models.

    Args:
        content: JSONL content (one JSON object per line)
        model_class: Pydantic model class to parse into
        strict: If True, raise on parse errors; if False, skip invalid lines

    Returns:
        List of successfully parsed model instances

    Raises:
        ValidationError: If strict=True and a line fails to parse
    """
    models = []
    for line in content.strip().split('\n'):
        if not line.strip():
            continue

        try:
            data = json.loads(line)
            models.append(model_class(**data))
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            if strict:
                raise
            logger.debug(f"Failed to parse JSONL line: {e}")

    return models


def parse_jsonl_to_dicts(
    content: str,
    *,
    strict: bool = False
) -> list[dict[str, Any]]:
    """
    Parse JSONL content into list of dictionaries.

    Args:
        content: JSONL content (one JSON object per line)
        strict: If True, raise on parse errors; if False, skip invalid lines

    Returns:
        List of successfully parsed dictionaries

    Raises:
        json.JSONDecodeError: If strict=True and a line fails to parse
    """
    dicts = []
    for line in content.strip().split('\n'):
        if not line.strip():
            continue

        try:
            data = json.loads(line)
            dicts.append(data)
        except json.JSONDecodeError as e:
            if strict:
                raise
            logger.debug(f"Failed to parse JSONL line: {e}")

    return dicts
