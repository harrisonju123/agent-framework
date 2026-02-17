"""Tests for BaseExtractor concrete methods."""

import pytest

from agent_framework.indexing.extractors.base import (
    DOCSTRING_MAX_LEN,
    BaseExtractor,
)
from agent_framework.indexing.models import SymbolEntry


class _StubExtractor(BaseExtractor):
    """Concrete subclass for testing base methods."""

    def extract_symbols(self, file_path: str, source: str) -> list[SymbolEntry]:
        return []


class TestRankKeyFiles:

    def test_priority_ordering(self):
        ext = _StubExtractor()
        files = ["utils.py", "main.py", "handler.py", "model.py", "README.md"]
        ranked = ext._rank_key_files(files)
        # model > handler > main based on KEY_FILE_PATTERNS order
        assert ranked[0] == "model.py"
        assert ranked[1] == "handler.py"
        assert ranked[2] == "main.py"

    def test_empty_list(self):
        ext = _StubExtractor()
        assert ext._rank_key_files([]) == []

    def test_case_insensitive(self):
        ext = _StubExtractor()
        files = ["MyService.py", "MAIN.go", "Controller.rb"]
        ranked = ext._rank_key_files(files)
        assert "MyService.py" in ranked
        assert "MAIN.go" in ranked
        assert "Controller.rb" in ranked

    def test_no_matching_files_returns_empty(self):
        ext = _StubExtractor()
        files = ["foo.py", "bar.py", "baz.py"]
        assert ext._rank_key_files(files) == []

    def test_max_five_results(self):
        ext = _StubExtractor()
        files = [
            "model.py", "handler.py", "service.py", "main.py",
            "router.py", "controller.py", "config.py", "schema.py",
        ]
        ranked = ext._rank_key_files(files)
        assert len(ranked) <= 5


class TestTruncateDocstring:

    def test_short_unchanged(self):
        ext = _StubExtractor()
        assert ext._truncate_docstring("hello") == "hello"

    def test_exact_limit_unchanged(self):
        ext = _StubExtractor()
        s = "x" * DOCSTRING_MAX_LEN
        assert ext._truncate_docstring(s) == s

    def test_long_truncated(self):
        ext = _StubExtractor()
        s = "x" * 200
        result = ext._truncate_docstring(s)
        assert len(result) == DOCSTRING_MAX_LEN
        assert result.endswith("...")


class TestFindModuleDescription:

    def test_reads_init_py_docstring(self, tmp_path):
        ext = _StubExtractor()
        init = tmp_path / "__init__.py"
        init.write_text('"""Core utilities for processing."""\n\nimport os\n')
        assert ext._find_module_description(str(tmp_path)) == "Core utilities for processing."

    def test_reads_doc_go(self, tmp_path):
        ext = _StubExtractor()
        doc = tmp_path / "doc.go"
        doc.write_text("// Package auth handles authentication.\n// It supports OAuth2.\npackage auth\n")
        assert ext._find_module_description(str(tmp_path)) == "Package auth handles authentication. It supports OAuth2."

    def test_falls_back_to_readme(self, tmp_path):
        ext = _StubExtractor()
        readme = tmp_path / "README.md"
        readme.write_text("# My Module\n\nThis module does great things.\n\nMore details here.\n")
        assert ext._find_module_description(str(tmp_path)) == "This module does great things."

    def test_returns_empty_on_missing_files(self, tmp_path):
        ext = _StubExtractor()
        assert ext._find_module_description(str(tmp_path)) == ""

    def test_init_py_takes_priority_over_readme(self, tmp_path):
        ext = _StubExtractor()
        init = tmp_path / "__init__.py"
        init.write_text('"""From init."""\n')
        readme = tmp_path / "README.md"
        readme.write_text("# Readme\n\nFrom readme.\n")
        assert ext._find_module_description(str(tmp_path)) == "From init."


class TestExtractModule:

    def test_returns_module_entry(self, tmp_path):
        ext = _StubExtractor()
        files = ["model.py", "utils.py", "handler.py", "test_foo.py"]
        mod = ext.extract_module(str(tmp_path), files, "python")
        assert mod.path == str(tmp_path)
        assert mod.language == "python"
        assert mod.file_count == 4
        assert "model.py" in mod.key_files
        assert "handler.py" in mod.key_files


class TestAbstractEnforcement:

    def test_extract_symbols_not_implemented(self):
        """BaseExtractor cannot be instantiated without implementing extract_symbols."""
        with pytest.raises(TypeError):
            BaseExtractor()
