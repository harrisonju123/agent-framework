"""Tests for file_summarizer.summarize_file()."""

import os

import pytest

from agent_framework.utils.file_summarizer import summarize_file


class TestSummarizePython:
    def test_classes_and_functions(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text(
            "class Agent:\n    pass\n\nclass Config:\n    pass\n\n"
            "def run():\n    pass\n\nasync def setup():\n    pass\n"
        )
        result = summarize_file(str(f))
        assert "classes: Agent, Config" in result
        assert "funcs: run, setup" in result
        assert result.startswith("11L")

    def test_syntax_error_falls_back(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        result = summarize_file(str(f))
        assert result == "1L"

    def test_empty_python(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = summarize_file(str(f))
        assert result == "0L; empty"


class TestSummarizeJsTs:
    def test_exported_symbols(self, tmp_path):
        f = tmp_path / "module.ts"
        f.write_text(
            "export function cacheFileRead() {}\n"
            "export class ReadCacheStore {}\n"
            "export interface Config {}\n"
            "const internal = 1;\n"
        )
        result = summarize_file(str(f))
        assert "exports: cacheFileRead, ReadCacheStore, Config" in result
        assert result.startswith("4L")

    def test_no_exports(self, tmp_path):
        f = tmp_path / "util.js"
        f.write_text("const x = 1;\nconst y = 2;\n")
        result = summarize_file(str(f))
        assert result == "2L"

    def test_tsx_extension(self, tmp_path):
        f = tmp_path / "component.tsx"
        f.write_text("export default function App() { return null; }\n")
        result = summarize_file(str(f))
        assert "exports: App" in result

    def test_jsx_extension(self, tmp_path):
        f = tmp_path / "component.jsx"
        f.write_text("export const Widget = () => {};\n")
        result = summarize_file(str(f))
        assert "exports: Widget" in result


class TestSummarizeConfig:
    def test_yaml(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nother: 42\n")
        assert summarize_file(str(f)) == "2L; config"

    def test_yml(self, tmp_path):
        f = tmp_path / "config.yml"
        f.write_text("a: 1\n")
        assert summarize_file(str(f)) == "1L; config"

    def test_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}\n')
        assert summarize_file(str(f)) == "1L; config"

    def test_toml(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[tool.pytest]\ntimeout = 30\n")
        assert summarize_file(str(f)) == "2L; config"


class TestSummarizeDocs:
    def test_markdown(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome text.\n")
        assert summarize_file(str(f)) == "3L; docs"

    def test_rst(self, tmp_path):
        f = tmp_path / "index.rst"
        f.write_text("Title\n=====\n")
        assert summarize_file(str(f)) == "2L; docs"

    def test_txt(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("line1\nline2\n")
        assert summarize_file(str(f)) == "2L; docs"


class TestEdgeCases:
    def test_binary_file(self, tmp_path):
        f = tmp_path / "image.bin"
        f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
        assert summarize_file(str(f)) == "Binary file"

    def test_missing_file(self):
        assert summarize_file("/nonexistent/path/file.py") == ""

    def test_unknown_extension(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("row1\nrow2\nrow3\n")
        assert summarize_file(str(f)) == "3L"

    def test_no_extension(self, tmp_path):
        f = tmp_path / "Makefile"
        f.write_text("all:\n\techo hello\n")
        assert summarize_file(str(f)) == "2L"

    def test_truncation_at_max_length(self, tmp_path):
        f = tmp_path / "big.py"
        # Generate many top-level functions to exceed max_length
        funcs = [f"def func_{i}(): pass" for i in range(50)]
        f.write_text("\n".join(funcs) + "\n")
        result = summarize_file(str(f), max_length=60)
        assert len(result) <= 60
        assert result.endswith("...")

    def test_unreadable_file(self, tmp_path):
        f = tmp_path / "secret.py"
        f.write_text("x = 1")
        os.chmod(str(f), 0o000)
        try:
            assert summarize_file(str(f)) == ""
        finally:
            os.chmod(str(f), 0o644)
