"""Data models for the per-repo structural codebase index."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    STRUCT = "struct"
    INTERFACE = "interface"
    MODULE = "module"


class SymbolEntry(BaseModel):
    name: str
    kind: SymbolKind
    file_path: str
    line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None

    class Config:
        use_enum_values = True


class ModuleEntry(BaseModel):
    path: str
    description: str = ""
    language: str
    file_count: int = 0
    key_files: list[str] = Field(default_factory=list)


class CodebaseIndex(BaseModel):
    repo_slug: str
    commit_sha: str
    language: str
    total_files: int = 0
    total_lines: int = 0
    modules: list[ModuleEntry] = Field(default_factory=list)
    symbols: list[SymbolEntry] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    test_directories: list[str] = Field(default_factory=list)
    version: int = 1

    class Config:
        use_enum_values = True
