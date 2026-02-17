"""Tests for indexing data models."""

import json

import pytest
from pydantic import ValidationError

from agent_framework.indexing.models import (
    CodebaseIndex,
    ModuleEntry,
    SymbolEntry,
    SymbolKind,
)


def _make_symbol(**overrides) -> SymbolEntry:
    defaults = dict(
        name="MyClass",
        kind=SymbolKind.CLASS,
        file_path="src/models.py",
        line=10,
    )
    defaults.update(overrides)
    return SymbolEntry(**defaults)


def _make_module(**overrides) -> ModuleEntry:
    defaults = dict(
        path="src/core",
        language="python",
    )
    defaults.update(overrides)
    return ModuleEntry(**defaults)


def _make_index(**overrides) -> CodebaseIndex:
    defaults = dict(
        repo_slug="org/repo",
        commit_sha="abc123",
        language="python",
    )
    defaults.update(overrides)
    return CodebaseIndex(**defaults)


class TestSymbolEntry:

    def test_creation_all_fields(self):
        sym = _make_symbol(
            signature="class MyClass(Base):",
            docstring="A model class.",
            parent="module",
        )
        assert sym.name == "MyClass"
        assert sym.kind == "class"
        assert sym.file_path == "src/models.py"
        assert sym.line == 10
        assert sym.signature == "class MyClass(Base):"
        assert sym.docstring == "A model class."
        assert sym.parent == "module"

    def test_optional_fields_default_none(self):
        sym = _make_symbol()
        assert sym.signature is None
        assert sym.docstring is None
        assert sym.parent is None

    def test_kind_serializes_as_string(self):
        sym = _make_symbol(kind=SymbolKind.FUNCTION)
        assert sym.kind == "function"


class TestSymbolKind:

    def test_enum_values(self):
        assert SymbolKind.CLASS == "class"
        assert SymbolKind.FUNCTION == "function"
        assert SymbolKind.METHOD == "method"
        assert SymbolKind.STRUCT == "struct"
        assert SymbolKind.INTERFACE == "interface"
        assert SymbolKind.MODULE == "module"

    def test_all_values_count(self):
        assert len(SymbolKind) == 6


class TestModuleEntry:

    def test_creation_with_defaults(self):
        mod = _make_module()
        assert mod.path == "src/core"
        assert mod.description == ""
        assert mod.language == "python"
        assert mod.file_count == 0
        assert mod.key_files == []

    def test_creation_with_values(self):
        mod = _make_module(
            description="Core module",
            file_count=5,
            key_files=["main.py", "config.py"],
        )
        assert mod.description == "Core module"
        assert mod.file_count == 5
        assert mod.key_files == ["main.py", "config.py"]


class TestCodebaseIndex:

    def test_creation_with_defaults(self):
        idx = _make_index()
        assert idx.repo_slug == "org/repo"
        assert idx.commit_sha == "abc123"
        assert idx.language == "python"
        assert idx.total_files == 0
        assert idx.total_lines == 0
        assert idx.modules == []
        assert idx.symbols == []
        assert idx.entry_points == []
        assert idx.test_directories == []
        assert idx.version == 1

    def test_creation_with_populated_data(self):
        sym = _make_symbol()
        mod = _make_module()
        idx = _make_index(
            total_files=42,
            total_lines=5000,
            modules=[mod],
            symbols=[sym],
            entry_points=["main.py"],
            test_directories=["tests/"],
        )
        assert idx.total_files == 42
        assert idx.total_lines == 5000
        assert len(idx.modules) == 1
        assert len(idx.symbols) == 1
        assert idx.entry_points == ["main.py"]
        assert idx.test_directories == ["tests/"]

    def test_json_roundtrip(self):
        sym = _make_symbol(signature="def foo():", docstring="Does foo.")
        mod = _make_module(description="Core", file_count=3, key_files=["a.py"])
        original = _make_index(
            total_files=10,
            modules=[mod],
            symbols=[sym],
            entry_points=["main.py"],
        )

        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = CodebaseIndex(**data)

        assert restored.repo_slug == original.repo_slug
        assert restored.commit_sha == original.commit_sha
        assert len(restored.symbols) == 1
        assert restored.symbols[0].name == "MyClass"
        assert restored.symbols[0].kind == "class"
        assert len(restored.modules) == 1
        assert restored.modules[0].description == "Core"

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            SymbolEntry(
                name="Bad",
                kind="not_a_kind",
                file_path="x.py",
                line=1,
            )
