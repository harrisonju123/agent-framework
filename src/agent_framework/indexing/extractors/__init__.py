"""Language-specific symbol extractors."""

from typing import Optional

from .base import BaseExtractor
from .go_extractor import GoExtractor
from .js_extractor import JSExtractor
from .python_extractor import PythonExtractor
from .ruby_extractor import RubyExtractor

_EXTRACTOR_MAP: dict[str, type[BaseExtractor]] = {
    "python": PythonExtractor,
    "go": GoExtractor,
    "ruby": RubyExtractor,
    "javascript": JSExtractor,
    "typescript": JSExtractor,
}


def get_extractor_for_language(language: str) -> Optional[BaseExtractor]:
    cls = _EXTRACTOR_MAP.get(language.lower())
    return cls() if cls else None


__all__ = [
    "BaseExtractor",
    "GoExtractor",
    "JSExtractor",
    "PythonExtractor",
    "RubyExtractor",
    "get_extractor_for_language",
]
