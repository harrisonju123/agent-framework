"""Persistent per-repo structural codebase index."""

from .models import CodebaseIndex, ModuleEntry, SymbolEntry, SymbolKind
from .extractors.base import BaseExtractor

__all__ = [
    "CodebaseIndex",
    "ModuleEntry",
    "SymbolEntry",
    "SymbolKind",
    "BaseExtractor",
]
